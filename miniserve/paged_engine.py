"""
M3 第 4 步：把缓存池接进 M2 的 Engine —— PagedEngine

和 M2 Engine 的区别只在 KV Cache 怎么管：
    M2：每个请求自己一份 LayerKVCache，每轮 torch.cat 变长；attention 逐请求 for 循环（packed_attention）
    M3：所有请求共用一个预先开好的 KVCachePool，每个请求只记一张块表
        - 请求加入 running：块表是空的，prefill 时按 prompt 长度分块（paged_prefill_forward）
        - 每轮 decode：块不够就补一块，新 k、v 写进池子，所有请求一次算完（paged_decode_forward）
        - 请求结束：把它的块全部还给分配器，马上可以分给别的请求

调度（static / continuous）、step()、run_all() 的用法和 M2 完全一样，直接继承。

没做的（M3 规划里的第 ⑤ 步，可选）：块不够时排队 / 抢占。
    现在只在【加入】时检查空闲块够不够装 prompt，不够就先不放进 running；
    decode 途中块用完会直接报 RuntimeError —— 所以池子要开够（测速脚本按最坏情况算好了块数）。

检查："D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" tests/check_paged_engine.py
"""
import torch

from miniserve.engine import Engine
from miniserve.paged import KVCachePool, blocks_needed, paged_decode_forward, paged_prefill_forward


class PagedEngine(Engine):
    def __init__(self, model, max_batch_size: int, num_blocks: int, block_size: int = 16,
                 policy: str = "continuous") -> None:
        super().__init__(model, max_batch_size, policy)
        p = next(model.parameters())
        self.pool = KVCachePool(model.config, num_blocks, block_size, device=p.device, dtype=p.dtype)
        self.block_tables: dict[int, list[int]] = {}     # rid → 这个请求的块表

        # 显存利用率统计（每轮累加，测速脚本用）
        self.peak_blocks_used = 0      # 同时被占用的块数最多是多少
        self.sum_tokens = 0            # Σ 每轮池子里真正存着的 token 数
        self.sum_slots = 0             # Σ 每轮分出去的块一共有多少个槽位
        # 对比用：如果不分页、每个请求一加入就预留一整段连续显存，要预留多少个 token 的位置
        self.sum_reserve_max = 0       # Σ 每轮：每个请求预留模型最大长度（GPT-2 是 1024）
        self.sum_reserve_exact = 0     # Σ 每轮：每个请求预留 prompt + max_new_tokens（要事先知道会生成多长）

    def _schedule(self) -> None:
        if self.policy == "static" and self.running:
            return
        bs = self.pool.block_size
        # 这一轮还能用的块：空闲块里先给老请求每个留 1 块（decode 时最多补 1 块），
        # 新放进来的请求要到 _run_batch 里才真正分块，所以在这里一边放一边扣
        budget = self.pool.allocator.num_free - len(self.running)
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting[0]
            need = blocks_needed(len(req.prompt) + 1, bs)   # prompt + 下一轮第一个 decode 的 token
            if need > budget:
                if not self.running:                     # 没有请求在跑、也就不会有块还回来：再等也没用
                    raise RuntimeError(f"缓存池太小：请求 {req.rid} 需要 {need} 块，池子一共只有 {budget} 块空闲")
                break                                    # 空闲块不够：先等别的请求结束还块
            budget -= need
            self.waiting.popleft()
            self.block_tables[req.rid] = []
            self.running.append(req)

    def _run_batch(self) -> None:
        device = next(self.model.parameters()).device
        prefill = [r for r in self.running if r.num_computed == 0]
        decode = [r for r in self.running if r.num_computed > 0]

        if prefill:                                      # 新请求：整个 prompt 一起 prefill
            logits = paged_prefill_forward(self.model, [r.prompt for r in prefill],
                                           [self.block_tables[r.rid] for r in prefill], self.pool)
            for r, nid in zip(prefill, logits.argmax(dim=-1).tolist()):
                r.output.append(nid)
                r.num_computed = len(r.prompt)

        if decode:                                       # 老请求：各送上一轮生成的 1 个 token，一次算完
            tokens = torch.tensor([r.output[-1] for r in decode], device=device)
            positions = [r.num_computed for r in decode]  # 缓存里已有 num_computed 个 token，新 token 就在这个位置
            logits = paged_decode_forward(self.model, tokens, positions,
                                          [self.block_tables[r.rid] for r in decode], self.pool)
            for r, nid in zip(decode, logits.argmax(dim=-1).tolist()):
                r.output.append(nid)
                r.num_computed += 1

        used = len(self.pool.allocator.used)
        self.peak_blocks_used = max(self.peak_blocks_used, used)
        self.sum_tokens += sum(r.num_computed for r in self.running)
        self.sum_slots += used * self.pool.block_size
        self.sum_reserve_max += self.model.config.block_size * len(self.running)
        self.sum_reserve_exact += sum(len(r.prompt) + r.max_new_tokens for r in self.running)

    def step(self) -> list:
        done = super().step()                            # 调度、跑一轮、移走结束的请求（M2 的逻辑）
        for r in done:                                   # 结束的请求：块全部还回去
            for b in self.block_tables.pop(r.rid):
                self.pool.allocator.free(b)
        return done

    @property
    def kv_utilization(self) -> float:
        """平均每轮：池子里真正存着 token 的槽位 / 分出去的槽位（只有每个请求最后一块没装满会浪费）"""
        return self.sum_tokens / self.sum_slots if self.sum_slots else 0.0

    @property
    def reserve_max_utilization(self) -> float:
        """对比：每个请求预留最大长度时的平均利用率"""
        return self.sum_tokens / self.sum_reserve_max if self.sum_reserve_max else 0.0

    @property
    def reserve_exact_utilization(self) -> float:
        """对比：每个请求预留 prompt + max_new_tokens 时的平均利用率"""
        return self.sum_tokens / self.sum_reserve_exact if self.sum_reserve_exact else 0.0
