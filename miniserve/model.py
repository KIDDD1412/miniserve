"""
MiniServe 的 GPT-2 模型（从零实现）。

按地图从小往大写，每写完一个类就运行检查：
    python tests/check_model.py

当前进度：第 1~5a 步 ✅  第 5b 步 generate
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
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # nn.ModuleList：装多个模块的列表（普通 list 的话 PyTorch 找不到里面的参数）
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # TODO ln_f：最后一个 LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)
        # lm_head：768 -> vocab_size，GPT-2 的 lm_head 没有 bias
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # TODO weight tying：让 lm_head 和 wte 用同一份权重（一行赋值）
        self.lm_head.weight = self.wte.weight
    def forward(self, idx):
        B, T = idx.size()
        # TODO
        #   ① 位置编号 pos = 0, 1, ..., T-1     提示：torch.arange，别忘了 device=idx.device
        #   ② x = 词嵌入 + 位置嵌入             (B, T, C) + (T, C) 会自动广播
        #   ③ 依次经过 self.h 里的每个 Block    提示：for 循环
        #   ④ 经过 self.ln_f
        #   ⑤ logits = self.lm_head(x)，返回 logits
        pos = torch.arange(T, device = idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
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
            # TODO 对 HF 的每一个参数：
            #   ① 如果是 "lm_head.weight"，跳过（它和 wte 共享，拷 wte 时已经一起拷好了）  提示：continue
            #   ② 把 HF 的名字变成你的名字：去掉开头的 "transformer."                        提示：str.removeprefix
            #   ③ 名字以 transposed 中的任意一个结尾 -> 拷贝 hf_tensor.t()，否则直接拷贝 hf_tensor
            #      提示：str.endswith 可以直接传一个元组；拷贝用 sd[key].copy_(...)
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
        # TODO 四步：
        #   ① logits = model(idx)                          (B, T, vocab_size)
        #   ② 只取最后一个位置的打分                        (B, vocab_size)    提示：切片 [:, -1, :]
        #   ③ 选出分数最高的词 next_id                      (B, 1)             提示：argmax(dim=-1, keepdim=True)
        #   ④ 把 next_id 拼到 idx 后面，得到新的 idx        (B, T+1)           提示：torch.cat，想想沿哪一维
        logits = model(idx)
        next_id = logits[:, -1, :].argmax(dim = -1, keepdim = True)
        idx = torch.cat([idx,next_id], dim = -1)
    return idx
