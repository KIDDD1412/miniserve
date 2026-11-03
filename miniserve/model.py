"""
MiniServe 的 GPT-2 模型（从零实现）。

按地图从小往大写，每写完一个类就运行检查：
    python tests/check_model.py

当前进度：M0 ✅  M1 第 1~3 步 ✅  M1 第 4 步 generate_cached
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024   # 最大序列长度（wpe 的行数）
    vocab_size: int = 50257  # 词表大小（wte 的行数）
    n_layer: int = 12        # Block 的层数（h.0 ~ h.11）
    n_head: int = 12         # 注意力头数
    n_embd: int = 768        # 隐藏维度 C


# ---------------------------------------------------------------------------
# M1：KV Cache 的"容器"
#   每一层各有一个 LayerKVCache，存着这一层到目前为止所有 token 的 k 和 v
#   一开始是空的（None）；每次前向时，把新 token 的 k、v 拼到后面
# ---------------------------------------------------------------------------
class LayerKVCache:
    def __init__(self):
        self.k = None   # (B, nh, 已缓存的长度, hs)
        self.v = None


def new_kv_cache(config):
    """给整个模型建一份空缓存：一个列表，每层一个 LayerKVCache"""
    return [LayerKVCache() for _ in range(config.n_layer)]


# ---------------------------------------------------------------------------
# 第 1 步：因果多头自注意力（对应地图上的 h.N.attn）
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head

        # c_attn：一次算出 q、k、v，就是练习 1 的 w_qkv（多了一个 bias）
        # nn.Linear(输入维度, 输出维度)，调用 self.c_attn(x) 等价于 x @ W + b
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    # （M1 第 2 步）给 forward 加上缓存：
    #   ① 签名改成 def forward(self, x, cache=None)：不传 cache 时行为和原来完全一样
    #   ② 拆完头、得到 k, v 之后：
    #        如果 cache 不是 None：
    #            如果 cache.k 不是 None：把 cache.k 和新的 k 拼起来（v 同理）   提示：torch.cat，沿 T 那一维
    #            把拼好的 k、v 存回 cache.k、cache.v
    #   ③ 现在 k 的长度 L 可能比 q 的长度 T 长（L = 已缓存的 + 新来的），mask 要改成 (T, L)：
    #        mask = torch.tril(torch.ones(T, L, device=x.device), diagonal=L - T)
    #      diagonal=L-T 的意思：第 i 个新 token 能看到"所有旧 token + 新 token 里第 0~i 个"
    def forward(self, x, cache = None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, -1)
        q = torch.transpose(q.view(B, T, self.n_head, C//self.n_head), 1, 2)
        k = torch.transpose(k.view(B, T, self.n_head, C // self.n_head), 1, 2)
        v = torch.transpose(v.view(B, T, self.n_head, C // self.n_head), 1, 2)
        if cache is not None:
            if cache.k is not None:
                k = torch.cat([cache.k, k], 2)
            if cache.v is not None:
                v = torch.cat([cache.v, v], 2)
            cache.k = k
            cache.v = v
        L = k.size(2)
        scores = q @ k.transpose(-1 ,-2) / math.sqrt(C // self.n_head)
        mask = torch.tril(torch.ones(T, L, device = x.device),diagonal = L - T)
        scores.masked_fill_(mask == 0, float('-inf'))
        attn = scores.softmax(dim=-1)
        y = attn @ v
        y= y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)



# ---------------------------------------------------------------------------
# 第 2 步：MLP（对应地图上的 h.N.mlp）
#   c_fc：768 -> 3072（放大 4 倍）
#   GELU：没有参数；GPT-2 用的是近似版，要写 approximate="tanh"
#   c_proj：3072 -> 768（缩回来）
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        # 三步：c_fc -> GELU -> c_proj
        #   GELU 用 F.gelu(..., approximate="tanh")
        return self.c_proj(F.gelu(self.c_fc(x),approximate = "tanh"))



# ---------------------------------------------------------------------------
# 第 3 步：Block（对应地图上的 h.N，GPT-2 一共 12 个）
#   x = x + attn( ln_1(x) )
#   x = x + mlp ( ln_2(x) )
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # LayerNorm 也是一个带参数的对象（weight 和 bias 各 768 维），用法和 nn.Linear 一样
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # 再定义 3 个零件：self.attn、self.ln_2、self.mlp
        #   提示：attn 和 mlp 就是你写好的 CausalSelfAttention 和 MLP，也是传 config 来创建
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    # TODO（M1 第 3 步）签名改成 forward(self, x, cache=None)，把 cache 原样传给 self.attn
    def forward(self, x, cache=None):
        # 两行，照着上面注释里的公式写
        x = x + self.attn(self.ln_1(x), cache = cache)
        x = x + self.mlp(self.ln_2(x))
        return x

# ---------------------------------------------------------------------------
# 第 4 步：完整的 GPT-2
#   idx (B, T)：token id（整数）
#     -> wte(idx) + wpe(位置)        (B, T, C)
#     -> 12 个 Block                 (B, T, C)
#     -> ln_f                        (B, T, C)
#     -> lm_head                     (B, T, vocab_size)   每个位置对词表里每个词的打分
# ---------------------------------------------------------------------------
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # nn.Embedding(行数, 每行维度)：一张查找表，wte(idx) 就是按 idx 取出对应的行
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # wpe：位置嵌入表，行数是最大序列长度 block_size
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # nn.ModuleList：装多个模块的列表（普通 list 的话 PyTorch 找不到里面的参数）
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # ln_f：最后一个 LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)
        # lm_head：768 -> vocab_size，GPT-2 的 lm_head 没有 bias
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying：让 lm_head 和 wte 用同一份权重（一行赋值）
        self.lm_head.weight = self.wte.weight

    # TODO（M1 第 3 步）给整个模型加上缓存：
    #   ① 签名改成 forward(self, idx, kv_cache=None)，kv_cache 是 new_kv_cache() 建的列表（每层一个）
    #   ② 位置编号要接着之前的往后数：
    #        past = 已经缓存了几个 token（没传缓存或缓存还是空的时候是 0）
    #               提示：看第 0 层缓存的 k 的第 2 维
    #        pos = torch.arange(past, past + T, device=idx.device)
    #   ③ 第 i 个 Block 用第 i 层的缓存：block(x, cache=...)
    #        提示：for i, block in enumerate(self.h)；没传 kv_cache 时每层都传 None
    def forward(self, idx, kv_cache = None):
        B, T = idx.size()
        if kv_cache is not None and kv_cache[0].k is not None:
            past = kv_cache[0].k.size(2)
        else:
            past = 0
        pos = torch.arange(past, past + T, device = idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for i, block in enumerate(self.h):
            layer_cache = None if kv_cache is None else kv_cache[i]
            x = block(x, cache = layer_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------------
# 第 5a 步：加载 HF 的预训练权重
#   state_dict()：模型的"参数字典"，名字 -> 张量
#   名字就是属性路径，比如你的 self.h[0].attn.c_attn.weight 叫 "h.0.attn.c_attn.weight"
#   HF 的名字多一个前缀，比如 "transformer.h.0.attn.c_attn.weight"
# ---------------------------------------------------------------------------
def from_pretrained(hf_name="openai-community/gpt2"):
    from transformers import GPT2LMHeadModel

    model = GPT2(GPTConfig())
    sd = model.state_dict()                                          # 你的模型：名字 -> 张量
    hf_sd = GPT2LMHeadModel.from_pretrained(hf_name).state_dict()    # HF 的模型：名字 -> 张量

    # 这 4 种权重需要转置：HF 的 Conv1D 存的是 (in, out)，nn.Linear 存的是 (out, in)
    transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")

    with torch.no_grad():
        for hf_key, hf_tensor in hf_sd.items():
            if hf_key == "lm_head.weight":
                continue
            key = hf_key.removeprefix("transformer.")
            if key.endswith(transposed):
                sd[key].copy_(hf_tensor.t())
            else:
                sd[key].copy_(hf_tensor)


    return model
# ---------------------------------------------------------------------------
# 第 5b 步：生成文字（贪心解码）
#   每一步：把目前为止的整个序列送进模型 -> 取最后一个位置的打分 -> 选分数最高的词 -> 接到序列末尾
#   @torch.no_grad()：告诉 PyTorch 这里只推理不训练，不用记录梯度（更快、更省显存）
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, idx, max_new_tokens):
    """idx: (B, T) 的 token id  ->  返回 (B, T + max_new_tokens)"""
    for _ in range(max_new_tokens):
        logits = model(idx)
        next_id = logits[:, -1, :].argmax(dim = -1, keepdim = True)
        idx = torch.cat([idx,next_id], dim = -1)
    return idx


# ---------------------------------------------------------------------------
# M1 第 4 步：带 KV Cache 的生成
#   prefill：把整个 prompt 一次送进去，顺便把每层的 k、v 存进缓存
#   decode ：之后每一步只送 1 个新 token，旧 token 的 k、v 直接从缓存里拿
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_cached(model, idx, max_new_tokens):
    """和 generate 的输入输出完全一样，只是内部用了 KV Cache"""
    kv_cache = new_kv_cache(model.config)

    #  prefill：logits = model(整个 idx, kv_cache=kv_cache)
    logits = model(idx, kv_cache = kv_cache)
    for _ in range(max_new_tokens):
        #   ② 从 logits 的最后一个位置选出 next_id                  (B, 1)
        #   ③ 拼到 idx 后面（idx 仍然要保留完整序列，最后要返回）
        #   ④ decode：只把 next_id 送进模型（不是整个 idx！），得到下一轮的 logits
        next_id = logits[:, -1, :].argmax(dim = -1, keepdim = True)
        idx = torch.cat([idx, next_id], dim = -1)
        logits = model(next_id, kv_cache = kv_cache)
    return idx
