"""
M2 第 2 步：推理引擎 —— 管理一堆请求的排队、加入、生成、结束。

两个队列：
    waiting：已经到达、还没开始算的请求
    running：正在生成的请求（每个都有自己的 KV Cache）

每调用一次 engine.step()，就做一轮：
    ① 调度：从 waiting 里挑请求放进 running（放多少、什么时候放，由调度策略决定）
    ② 把 running 里所有请求这一轮要送的 token 压平，做一次 packed_forward
         刚加入的请求：送整个 prompt（prefill）
         已经在跑的请求：只送上一步生成的那 1 个 token（decode）
    ③ 每个请求选出下一个 token，接到它的 output 后面
    ④ 生成够了的请求从 running 里移走（它的 KV Cache 也就释放了）

两种调度策略（第 3 步会对比它们的吞吐）：
    "static"     ：running 里的请求【全部】结束后，才放下一批进来
    "continuous" ：每一轮都检查，只要 running 有空位就立刻补新请求进来

检查：python tests/check_engine.py
"""
from collections import deque
from dataclasses import dataclass, field

import torch

from miniserve.model import new_kv_cache
from miniserve.packed import packed_forward


@dataclass
class Request:
    rid: int                       # 请求编号
    prompt: list                   # prompt 的 token id 列表
    max_new_tokens: int            # 要生成几个 token
    output: list = field(default_factory=list)  # 已经生成的 token
    kv_cache: list = None          # 加入 running 时才创建
    num_computed: int = 0          # 已经送进过模型的 token 数（也就是缓存里有几个 token）

    @property
    def finished(self):
        return len(self.output) >= self.max_new_tokens


class Engine:
    def __init__(self, model, max_batch_size, policy="continuous"):
        assert policy in ("static", "continuous")
        self.model = model
        self.max_batch_size = max_batch_size   # running 里最多同时有几个请求
        self.policy = policy
        self.waiting = deque()
        self.running = []
        self.num_steps = 0                     # 一共跑了几轮 step（第 3 步统计用）

    def add_request(self, req):
        self.waiting.append(req)

    def has_unfinished(self):
        return len(self.waiting) > 0 or len(self.running) > 0

    # -----------------------------------------------------------------------
    # 第 2a 步：让 running 里的所有请求一起往前走一步
    # -----------------------------------------------------------------------
    def _run_batch(self):
        device = next(self.model.parameters()).device
        tokens, positions, seq_lens = [], [], []

        for req in self.running:
            # TODO ① 决定这个请求这一轮送哪些 token：
            #        还没送过（req.num_computed == 0）：送整个 prompt              → prefill
            #        已经送过：只送 req.output 的最后一个                          → decode
            #      ② 它们的位置编号：从 req.num_computed 开始往后数
            #      ③ 分别加进 tokens、positions（用 extend），再把个数加进 seq_lens
            pass

        tokens = torch.tensor(tokens, device=device)
        positions = torch.tensor(positions, device=device)
        logits = packed_forward(self.model, tokens, positions, seq_lens,
                                [req.kv_cache for req in self.running])
        next_ids = logits.argmax(dim=-1).tolist()     # 每个请求一个新 token（变成 Python 列表）

        # TODO ④ 对每个请求（和 next_ids、seq_lens 一一对应）：
        #        把新 token 接到 req.output 后面；req.num_computed 加上这一轮送进去的个数
        #      提示：for req, nid, n in zip(self.running, next_ids, seq_lens)

    # -----------------------------------------------------------------------
    # 第 2b 步：调度 + 收尾
    # -----------------------------------------------------------------------
    def _schedule(self):
        # TODO 按策略从 waiting 里挑请求放进 running：
        #   "static"    ：只有 running 为空时，才一次放进最多 max_batch_size 个
        #   "continuous"：只要 running 没满，就一直从 waiting 里补
        #   放进 running 的请求，要给它建好缓存：req.kv_cache = new_kv_cache(self.model.config)
        #   提示：self.waiting.popleft() 取出最早到达的请求
        pass

    def step(self):
        """跑一轮；返回这一轮结束的请求列表"""
        self._schedule()
        if not self.running:
            return []
        self._run_batch()
        self.num_steps += 1

        # TODO 把已经生成够的请求（req.finished）从 running 里移走，并作为返回值
        #   提示：先挑出 finished 的，再让 self.running 只保留没结束的
        return []

    def run_all(self):
        """一直跑到所有请求都结束；按 rid 返回每个请求生成的 token"""
        results = {}
        while self.has_unfinished():
            for req in self.step():
                results[req.rid] = req.output
        return results
