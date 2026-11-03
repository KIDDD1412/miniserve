"""
M2：把多个请求"压平"成一排，一起做前向（packed / varlen 前向）。

多个请求长度不同，怎么放进一个 batch？
    不补齐（padding），而是把所有请求的 token 首尾相接排成一排：
        请求 A 的 3 个 token | 请求 B 的 1 个 token | 请求 C 的 5 个 token   → x: (9, C)
    再用 seq_lens = [3, 1, 5] 记住每个请求占几个。

这样：
    - 线性层、LayerNorm、MLP 都是每个 token 各算各的 → 直接对 (9, C) 算，一次矩阵乘服务所有请求
      （这就是 batching 的收益：读一次权重，给所有请求用）
    - 只有 attention 需要看"同一个请求里的其他 token" → 按 seq_lens 切开，每个请求单独算

每个请求有自己的 KV Cache（M1 的 new_kv_cache 建的那种列表），所以 prefill 和 decode 可以混在同一批里：
    新请求送整个 prompt（prefill），老请求只送 1 个新 token（decode）。

检查：python tests/check_packed.py
"""
import math

import torch


# ---------------------------------------------------------------------------
# M2 第 1a 步：压平之后的 attention
# ---------------------------------------------------------------------------
def packed_attention(attn, x, seq_lens, layer_caches):
    """
    attn:         你写的 CausalSelfAttention（只借用它的 c_attn、c_proj 和 n_head）
    x:            (N, C)，所有请求的 token 排成一排，N = sum(seq_lens)
    seq_lens:     每个请求这次送进来几个 token，比如 [3, 1, 5]
    layer_caches: 每个请求在【这一层】的 LayerKVCache，和 seq_lens 一一对应
    返回：         (N, C)
    """
    N, C = x.shape
    nh = attn.n_head
    hs = C // nh

    # 所有请求一起算 q、k、v：一次矩阵乘
    q, k, v = attn.c_attn(x).split(C, dim=-1)        # 各 (N, C)

    outs = []
    start = 0
    for n, cache in zip(seq_lens, layer_caches):
        # TODO 对这一个请求（它占 x 里的第 start ~ start+n 行）：
        #   ① 切出它自己的 q、k、v                       各 (n, C)        提示：切片 [start:start + n]
        #   ② 变成你熟悉的形状 (1, nh, n, hs)              提示：先 view(1, n, nh, hs) 再 transpose(1, 2)
        #   ③ 和 M1 一样：拼上缓存里的旧 k、v，再存回缓存
        #   ④ 和 M1 一样：算 scores、(n, L) 的 mask（diagonal=L-n）、softmax、乘 v   → (1, nh, n, hs)
        #   ⑤ 合并头变回 (n, C)，加进 outs                提示：transpose(1, 2).contiguous().view(n, C)
        nq = q[start:start + n]
        nk = k[start:start + n]
        nv = v[start:start + n]
        nq = torch.transpose(nq.view(1, n, nh, hs), 1, 2)
        nk = torch.transpose(nk.view(1, n, nh, hs), 1, 2)
        nv = torch.transpose(nv.view(1, n, nh, hs), 1, 2)
        if cache is not None:
            if cache.k is not None:
                nk = torch.cat([cache.k, nk], dim = 2)
            if cache.v is not None:
                nv = torch.cat([cache.v, nv], dim = 2)
            cache.k = nk
            cache.v = nv
        L = nk.size(2)
        scores = nq @ nk.transpose(-2, -1) / math.sqrt(hs)
        mask = torch.tril(torch.ones(n, L,device = x.device), diagonal = L - n)
        scores.masked_fill_(mask == 0, float('-inf'))
        att = scores.softmax(dim=-1)
        y = att @ nv
        y = y.transpose(1, 2).contiguous().view(n, C)
        outs.append(y)
        start += n

    y = torch.cat(outs, dim=0)                        # 再拼回 (N, C)
    return attn.c_proj(y)                             # 输出投影也是所有请求一起算


# ---------------------------------------------------------------------------
# M2 第 1b 步：压平之后的整个模型
# ---------------------------------------------------------------------------
@torch.no_grad()
def packed_forward(model, tokens, positions, seq_lens, kv_caches):
    """
    model:     你写的 GPT2
    tokens:    (N,) 所有请求这次送进来的 token id，排成一排
    positions: (N,) 每个 token 在它自己那个请求里的位置编号
    seq_lens:  每个请求这次送进来几个 token
    kv_caches: 每个请求一份 new_kv_cache() 建的缓存（一个列表，每层一个 LayerKVCache）
    返回：      (len(seq_lens), vocab_size)，每个请求【最后一个 token】的 logits（用来选下一个词）
    """
    # TODO 照着 GPT2.forward 写，区别只有三处：
    #   ① x = 词嵌入 + 位置嵌入，这次形状是 (N, C)；位置编号直接用传进来的 positions
    #   ② 每个 Block 里 attention 那一行换成 packed_attention：
    #        x = x + packed_attention(block.attn, block.ln_1(x), seq_lens, 第 i 层的缓存们)
    #        x = x + block.mlp(block.ln_2(x))            ← MLP 不用改，(N, C) 直接算
    #      "第 i 层的缓存们" = 每个请求的 kv_caches 里取第 i 层：[c[i] for c in kv_caches]
    #   ③ 只取每个请求最后一个 token 的 logits
    #        每个请求最后一个 token 在第几行？提示：seq_lens 累加起来再减 1
    raise NotImplementedError
