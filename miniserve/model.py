"""
MiniServe 的 GPT-2 模型（从零实现）。

按地图从小往大写，每写完一个类就运行检查：
    python tests/check_model.py

当前进度：第 1 步 ✅  第 2 步 ✅  第 3 步 ✅  第 4 步 GPT2
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
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, -1)
        q =torch.transpose(q.view(B, T, self.n_head, C//self.n_head), 1, 2)
        k = torch.transpose(k.view(B, T, self.n_head, C // self.n_head), 1, 2)
        v = torch.transpose(v.view(B, T, self.n_head, C // self.n_head), 1, 2)
        scores = q @ torch.transpose(k, -1 ,-2) / math.sqrt(C // self.n_head)
        mask = torch.tril(torch.ones(T, T, device = x.device))
        scores.masked_fill_(mask == 0, float('-inf'))
        attn = scores.softmax(dim=-1)
        y = attn @ v
        y= y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
        #   ① qkv = self.c_attn(x)                   (B, T, 3C)
        #   ② 拆成 q, k, v                            各 (B, T, C)
        #   ③ 拆头                                    各 (B, nh, T, hs)
        #   ④ 注意力分数                              (B, nh, T, T)
        #   ⑤ 因果 mask + softmax                     (B, nh, T, T)
        #   ⑥ 乘 v                                    (B, nh, T, hs)
        #   ⑦ 合并头                                  (B, T, C)
        #   ⑧ 输出投影 self.c_proj                    (B, T, C)
        # 注意：头数用 self.n_head，尺寸一律从 x 的形状里取，不要写死数字



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
        # TODO 三步：c_fc -> GELU -> c_proj
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
        # TODO 再定义 3 个零件：self.attn、self.ln_2、self.mlp
        #   提示：attn 和 mlp 就是你写好的 CausalSelfAttention 和 MLP，也是传 config 来创建
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # TODO 两行，照着上面注释里的公式写
        x = x + self.attn(self.ln_1(x))
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
        # TODO wpe：位置嵌入表，行数是最大序列长度 block_size
        self.wpe = None
        # nn.ModuleList：装多个模块的列表（普通 list 的话 PyTorch 找不到里面的参数）
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # TODO ln_f：最后一个 LayerNorm
        self.ln_f = None
        # lm_head：768 -> vocab_size，GPT-2 的 lm_head 没有 bias
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # TODO weight tying：让 lm_head 和 wte 用同一份权重（一行赋值）

    def forward(self, idx):
        B, T = idx.size()
        # TODO
        #   ① 位置编号 pos = 0, 1, ..., T-1     提示：torch.arange，别忘了 device=idx.device
        #   ② x = 词嵌入 + 位置嵌入             (B, T, C) + (T, C) 会自动广播
        #   ③ 依次经过 self.h 里的每个 Block    提示：for 循环
        #   ④ 经过 self.ln_f
        #   ⑤ logits = self.lm_head(x)，返回 logits
        raise NotImplementedError
