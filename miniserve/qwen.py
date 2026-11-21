r"""
M4：Qwen2.5-0.5B 模型（从零实现）。

和 GPT-2 比，结构上有 4 处不同，每处是一步：
    第 ② 步 RMSNorm 代替 LayerNorm（不减均值、没有 bias）
    第 ③ 步 RoPE 代替 wpe（位置信息不加在输入上，而是在每一层"转"进 q 和 k）
    第 ④ 步 GQA attention：14 个 q 头共用 2 个 k/v 头；q/k/v 的 Linear 有 bias，o_proj 没有
    第 ⑤ 步 SwiGLU MLP 代替 GELU MLP（gate、up 两个投影，SiLU 当门控）
第 ⑥ 步把它们拼成整个模型。其余部分（残差、最后一个 norm、lm_head 和词嵌入共享权重）和 GPT-2 一样。

模块名和 HF 权重文件里的名字一一对应，只差一个 "model." 前缀，
所以加载权重时只需要去掉前缀；HF 的 Qwen2 用的是 nn.Linear，也不用像 GPT-2 那样转置。

检查（在项目根目录 D:\infra\miniserve 下）：
    "D:\Program Files\Anaconda\envs\miniserve\python.exe" tests/check_qwen.py
"""
import json
import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

QWEN_NAME = "Qwen/Qwen2.5-0.5B"


# ---------------------------------------------------------------------------
# 第 ① 步：模型配置
#   字段名和 HF 的 config.json 完全一样；数值一律从 config.json 读，不在代码里手写
#   config.json 里还有 use_sliding_window: false、max_position_embeddings 等字段，推理用不上，没读
# ---------------------------------------------------------------------------
@dataclass
class QwenConfig:
    vocab_size: int            # 词表大小（embed_tokens 的行数）
    hidden_size: int           # 隐藏维度 C
    intermediate_size: int     # MLP 中间层的维度
    num_hidden_layers: int     # decoder 层数
    num_attention_heads: int   # q 的头数
    num_key_value_heads: int   # k、v 的头数（GQA：比 q 的头数少）
    rms_norm_eps: float        # RMSNorm 里加的那个 eps
    rope_theta: float          # RoPE 的底数
    tie_word_embeddings: bool  # lm_head 是否和 embed_tokens 共用同一份权重

    @property
    def head_dim(self) -> int:
        """每个头的维度：C // q 的头数（k、v 的头和 q 的头一样宽，只是个数少）"""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, name: str = QWEN_NAME) -> "QwenConfig":
        """从 HF 缓存里读 config.json（第一次会下载）"""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(name, "config.json")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(**{field.name: d[field.name] for field in fields(cls)})


# ---------------------------------------------------------------------------
# 第 ② 步：RMSNorm（对应 HF 的 input_layernorm / post_attention_layernorm / norm）
#   LayerNorm：先减均值，再除以标准差，再乘 weight、加 bias
#   RMSNorm ：不减均值，直接除以"均方根" sqrt(mean(x²) + eps)，再乘 weight；没有 bias
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # (dim,)，只有缩放，没有 bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., dim)，最后一维是隐藏维度；前面几维随意（(B, T, C) 和 packed 的 (N, C) 都要能用）
        返回：和 x 同形状
            y = x / sqrt(mean(x²) + eps) * weight      mean 是沿最后一维、对每个 token 单独求
        """
        #   ① 每个 token 求 x² 沿最后一维的平均        要保留这一维，下一步才能和 x 广播
        #   ② 加 eps，再开根号取倒数                  零件：torch.rsqrt
        #   ③ 乘回 x，再乘 self.weight
        m = x.pow(2).mean(-1, keepdim=True)
        m = torch.rsqrt(m + self.eps)
        return m * x * self.weight


# ---------------------------------------------------------------------------
# 第 ③ 步：RoPE（旋转位置编码）
#   GPT-2：位置信息是一张表 wpe，查出来加到输入上，只在最开始加一次
#   RoPE ：没有表、没有参数。每一层算出 q、k 之后，把它们按 token 的位置"转"一个角度
#
#   怎么转：把一个头的 64 维分成 32 对，第 i 对是 (x[i], x[i + 32])（前一半第 i 个配后一半第 i 个）
#           每一对看成平面上的一个点，转 角度 = 位置 × inv_freq[i]
#           inv_freq[i] = theta ** (-2i / head_dim)：i 越大转得越慢（第 0 对每个位置转 1 弧度，最后一对几乎不转）
#   为什么有用：q 在位置 m 转了 m·f，k 在位置 n 转了 n·f，两者点积只和角度差 (m - n)·f 有关
#              → attention 分数里自然带上了"相对距离"
#
#   公式（二维旋转，对每一对）：  新x1 = x1·cos − x2·sin
#                              新x2 = x2·cos + x1·sin
#   写成整个向量：  x · cos + rotate_half(x) · sin      其中 rotate_half(x) = (−x2, x1) 拼起来
#   （rotate_half 只是给"拼出 (−x2, x1)"这一步起的名字，HF 里它是一个单独的小函数；
#     这里不要求单独定义，直接在 apply_rope 里写成一行 torch.cat 就行）
#   （cos、sin 也要排成 64 维：前 32 个和后 32 个一样，这样第 i 维和第 i + 32 维用的是同一个角度）
# ---------------------------------------------------------------------------
def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    positions: 整数张量，每个 token 在自己请求里的位置；形状随意，比如 (T,) 或 (B,)
    head_dim:  每个头的维度（Qwen2.5-0.5B 是 64）
    theta:     RoPE 的底数（config.rope_theta，Qwen2.5 是 1e6）
    返回：cos, sin，形状都是 (*positions.shape, head_dim)，float32（和 HF 一样用 float32 算）
    """
    #   ① 算 (head_dim // 2) 个频率 inv_freq（公式见上面）       零件：torch.arange（带步长；放在 positions 的设备上）
    #   ② 每个位置 × 每个频率 = 角度                          零件：广播（positions 末尾加一维）；positions 先转 float32
    #   ③ 角度复制一份拼在后面，凑满 head_dim                  零件：torch.cat
    #   ④ 返回角度的 cos 和 sin
    two_i = torch.arange(0, head_dim, 2, device=positions.device)
    inv_freq = theta ** (-two_i / head_dim)
    p = positions.float()[..., None]
    a = p * inv_freq
    a = torch.cat([a, a], dim=-1)
    return a.cos(), a.sin()






def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:        维数不固定，只保证【最后一维是 head_dim】。代码里只能用 x.shape[-1]，不能按 4 维去拆 x.shape
              会传进来的形状有：(B, n_head, T, head_dim)（第 ④ 步的 attention）
                               (N, n_head, head_dim)（第 ⑦ 步的 packed 路径，只有 3 维）
    cos, sin: rope_cos_sin 的结果，形状要能广播成 x 的形状（调用的人负责调好形状）
    返回：和 x 同形状。不要改动 x 本身    
    """
    #   ① 把 x 沿最后一维切成前一半 x1、后一半 x2              零件：切片 x[..., :half]
    #   ② 拼出 rotate_half(x) = (−x2, x1)                     零件：torch.cat
    #   ③ 按上面的公式组合
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return x * cos + sin * torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------------------
# 第 ④ 步：GQA attention（对应 HF 的 self_attn）
#   q_proj：C -> 14 × 64 = 896；k_proj、v_proj：C -> 2 × 64 = 128（都有 bias）
#   o_proj：896 -> C（没有 bias）
#
#   GQA（Grouped-Query Attention）：q 有 14 个头，k、v 只有 2 个头
#     14 个 q 头分成 2 组，每组 7 个（n_rep = 14 // 2）：q 头 0~6 共用 k/v 头 0，q 头 7~13 共用 k/v 头 1
#     好处：KV Cache 只存 2 个头，是 14 个头的 1/7
#     算 attention 时，把 k、v 的每个头复制 7 份、排成 14 个头，之后就和 M0 的多头 attention 一模一样
#
#   和 M0 的 CausalSelfAttention 比，只多了两件事：
#     ① 拆头之后，q 和 k 要用 RoPE 转一下（v 不转）
#     ② 算分数之前，k、v 的头要复制成和 q 一样多
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.head_dim = config.head_dim
        C = config.hidden_size
        self.q_proj = nn.Linear(C, self.n_head * self.head_dim, bias=True)
        self.k_proj = nn.Linear(C, self.n_kv_head * self.head_dim, bias=True)
        self.v_proj = nn.Linear(C, self.n_kv_head * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, C, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache=None) -> torch.Tensor:
        """
        x:        固定 3 维 (B, T, C)：B 个序列，每个这次送进来 T 个 token（这里可以按 3 维拆 x.shape）
        cos, sin: 固定 2 维 (T, head_dim)：这 T 个 token 的位置算出来的（B 个序列的位置一样，所以没有 B 这一维）
                  可以直接和 (B, 头数, T, head_dim) 广播
        cache:    None，或者 M1 的 LayerKVCache（cache.k、cache.v 是 None 或 (B, n_kv_head, 已缓存长度, head_dim)）
        返回：     (B, T, C)
        """
        # （第 ④ 步）（形状以 Qwen2.5-0.5B 为例：C = 896，n_head = 14，n_kv_head = 2，head_dim = 64）：
        #   ① 把 x 分别送进 __init__ 里建好的三个线性层，得到 q、k、v
        #        q: (B, T, 896)    k、v: (B, T, 128)                   零件：self.q_proj(x) 这样调用
        #   ② 拆头：把最后一维拆成"头数 × head_dim"，再把头这一维换到 T 前面
        #        q: (B, 14, T, 64)  k、v: (B, 2, T, 64)                零件：view + transpose（M0 写过，注意头数不一样）
        #   ③ 用 RoPE 转 q 和 k（v 不转），形状不变                     零件：apply_rope
        #   ④ 有 cache 时：把旧的 k、v 拼在新的前面，再存回 cache
        #        拼完 k、v: (B, 2, L, 64)，L = 已缓存的 + T              照 M1；这时还没复制，存的是 2 个头
        #   ⑤ k、v 的每个头原地复制 n_head // n_kv_head 份（= 7）
        #        k、v: (B, 14, L, 64)                                  零件：repeat_interleave
        #   ⑥ 算分数、加因果 mask、softmax、乘 v
        #        分数: (B, 14, T, L)  结果: (B, 14, T, 64)              照 M1（除以的是 sqrt(head_dim)）
        #   ⑦ 合并头，再送进 o_proj
        #        (B, 14, T, 64) → (B, T, 896) → o_proj → (B, T, C)     零件：transpose + contiguous + view，self.o_proj
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        B, T, C = q.shape
        head_dim = self.head_dim
        n_kv_head = self.n_kv_head
        n_head = self.n_head
        q = torch.transpose(q.view(B, T, n_head, head_dim), 1, 2)
        k = torch.transpose(k.view(B, T, n_kv_head, head_dim), 1, 2)
        v = torch.transpose(v.view(B, T, n_kv_head, head_dim), 1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            if cache.k is not None:
                k = torch.cat([cache.k, k], dim = 2)
            if cache.v is not None:
                v = torch.cat([cache.v, v], dim = 2)
            cache.k = k
            cache.v = v
        L = k.size(2)
        k = k.repeat_interleave(n_head // n_kv_head, dim = 1)
        v = v.repeat_interleave(n_head // n_kv_head, dim = 1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(T, L,  device=scores.device), diagonal = L - T)
        scores.masked_fill_(mask == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        y = attn @ v
        y = y.transpose(2, 1).contiguous().view(B, T, C)
        return self.o_proj(y)



# ---------------------------------------------------------------------------
# 第 ⑤ 步：SwiGLU MLP（对应 HF 的 mlp）
#   gate_proj、up_proj：C -> intermediate_size；down_proj：intermediate_size -> C（都没有 bias）
#
#   GPT-2 的 MLP：  down( GELU( up(x) ) )                      一条路：放大 → 激活 → 缩回
#   SwiGLU：        down( SiLU( gate(x) ) * up(x) )            两条路：gate 那条过激活当"阀门"，
#                                                                       和 up 那条逐个元素相乘，再缩回
#   SiLU(z) = z * sigmoid(z)（也叫 Swish），HF 的 config 里写的是 hidden_act: "silu"
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        C, I = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(C, I, bias=False)
        self.up_proj = nn.Linear(C, I, bias=False)
        self.down_proj = nn.Linear(I, C, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:    维数不固定，只保证【最后一维是 C】。线性层和逐元素运算本来就只看最后一维，
              所以这里不需要拆 x.shape，直接对 x 算就行
              会传进来的形状有：(B, T, C)（第 ⑥ 步的整个模型）
                               (N, C)（第 ⑦ 步的 packed 路径）、(B, C)（第 ⑦ 步的 paged decode）
        返回：和 x 同形状
        """
        # 第 ⑤ 步）（形状以 x 为 (B, T, C)、C = 896、intermediate_size = 4864 为例）：
        #   ① 把 x 送进 gate_proj，再过 SiLU 激活，得到"阀门"      (B, T, 4864)      零件：self.gate_proj、F.silu
        #   ② 把 x 送进 up_proj                                     (B, T, 4864)      零件：self.up_proj
        #   ③ 两者逐个元素相乘                                      (B, T, 4864)      零件：*
        #   ④ 送进 down_proj 缩回 C 维，返回                         (B, T, 896)       零件：self.down_proj
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# 第 ⑥a 步：一个 decoder 层（对应 HF 的 model.layers.N，一共 24 个）
#   和 GPT-2 的 Block 结构完全一样（pre-norm + 两个残差），只是零件换了名字：
#       GPT-2：  x = x + attn( ln_1(x) )                 x = x + mlp( ln_2(x) )
#       Qwen2：  x = x + self_attn( input_layernorm(x) )  x = x + mlp( post_attention_layernorm(x) )
#   不同的是 self_attn 多要两个参数 cos、sin（RoPE 在 attention 里面转 q、k）
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        C, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = RMSNorm(C, eps)            # attention 之前的 norm（相当于 GPT-2 的 ln_1）
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(C, eps)   # MLP 之前的 norm（相当于 GPT-2 的 ln_2）
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache=None) -> torch.Tensor:
        """
        x:        固定 3 维 (B, T, C)
        cos, sin: (T, head_dim)，原样传给 self_attn
        cache:    None，或者这一层的 LayerKVCache，原样传给 self_attn
        返回：     (B, T, C)
        """
        # TODO（第 ⑥a 步）：
        #   ① x 先过 input_layernorm，再送进 self_attn（带上 cos、sin、cache），结果加回 x      (B, T, C)
        #   ② x 先过 post_attention_layernorm，再送进 mlp，结果加回 x                          (B, T, C)
        #   ③ 返回 x
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache = cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x



# ---------------------------------------------------------------------------
# 第 ⑥b 步：整个模型（对应 HF 的 Qwen2ForCausalLM）
#   idx (B, T)：token id
#     -> embed_tokens(idx)                  (B, T, C)       没有 wpe：位置信息在每层的 attention 里由 RoPE 加
#     -> 24 个 DecoderLayer                 (B, T, C)       每层都用同一份 cos、sin（只在进层之前算一次）
#     -> norm                               (B, T, C)
#     -> lm_head                            (B, T, vocab_size)
# ---------------------------------------------------------------------------
class Qwen2(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        C = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, C)    # 相当于 GPT-2 的 wte；没有 wpe
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(C, config.rms_norm_eps)               # 相当于 GPT-2 的 ln_f
        self.lm_head = nn.Linear(C, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight        # 和 GPT-2 一样共享权重

    def forward(self, idx: torch.Tensor, kv_cache: list | None = None) -> torch.Tensor:
        """
        idx:      固定 2 维 (B, T) 的 token id（整数）
        kv_cache: None，或者 new_kv_cache(config) 建的列表（每层一个 LayerKVCache，第 i 层用第 i 个）
        返回：     logits (B, T, vocab_size)
        """
        # （第 ⑥b 步）（形状以 C = 896、head_dim = 64 为例）：
        #   ① 算出已经缓存了几个 token（past）：没传缓存、或者缓存还是空的时候是 0          照 M1 的 GPT2.forward
        #   ② 生成这次 T 个 token 的位置 past, past + 1, …, past + T − 1                   (T,)       零件：torch.arange（放在 idx 的设备上）
        #   ③ 用位置算出 cos、sin                                                         (T, 64)    零件：rope_cos_sin，参数从 self.config 里拿
        #   ④ idx 送进 embed_tokens 得到 x                                                (B, T, 896)
        #   ⑤ 把 cos、sin 转成和 x 一样的类型（现在都是 float32，看不出区别；⑧ 测速用 bf16 时必须有这一步）
        #                                                                                 零件：cos.to(x.dtype)
        #   ⑥ 依次过 24 个 DecoderLayer，第 i 层传第 i 层的缓存（没传 kv_cache 时每层传 None） 照 M1 的 GPT2.forward
        #   ⑦ 过 norm，再过 lm_head，返回                                                  (B, T, vocab_size)
        B, T = idx.size()
        if kv_cache is not None and kv_cache[0].k is not None:
            past = kv_cache[0].k.size(2)
        else:
            past = 0
        pos = torch.arange(past, past + T, device=idx.device)
        cos, sin = rope_cos_sin(pos, self.config.head_dim, self.config.rope_theta)
        x = self.embed_tokens(idx)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        for i ,layer in enumerate(self.layers):
            layercache = None if kv_cache is None else kv_cache[i]
            x = layer(x, cos, sin, layercache)
        return self.lm_head(self.norm(x))



def new_kv_cache(config: QwenConfig) -> list:
    """给整个模型建一份空缓存：每层一个 M1 的 LayerKVCache（GPT-2 的 new_kv_cache 读的是 config.n_layer，这里字段名不一样）"""
    from miniserve.model import LayerKVCache

    return [LayerKVCache() for _ in range(config.num_hidden_layers)]


# ---------------------------------------------------------------------------
# 第 ⑥c 步：带 KV Cache 的贪心解码（和 M1 的 generate_cached 一样）
#   prefill：整个 prompt 一次送进去，顺便填满每层的缓存
#   decode ：之后每轮只送 1 个新 token
#   这里不管 eos（生成结束符），固定生成 max_new_tokens 个，方便和 HF 逐 token 对比
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model: Qwen2, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """
    idx: 固定 2 维 (B, T) 的 token id
    返回：(B, T + max_new_tokens)，前 T 个就是 idx
    """
    # TODO（第 ⑥c 步）：
    #   ① 建一份空缓存                                                       零件：new_kv_cache(model.config)
    #   ② prefill：整个 idx 带着缓存送进 model，得到 logits                   (B, T, vocab_size)
    #   ③ 循环 max_new_tokens 次：
    #        从 logits 的最后一个位置选出分数最高的 next_id                    (B, 1)       零件：argmax(dim=-1, keepdim=True)
    #        拼到 idx 后面                                                    (B, 当前长度 + 1)
    #        只把 next_id 带着缓存送进 model，得到下一轮的 logits              (B, 1, vocab_size)
    #   ④ 返回 idx
    kv_cache = new_kv_cache(model.config)
    logits = model(idx, kv_cache)
    for _ in range(max_new_tokens):
        next_id = logits[:, -1, :].argmax(dim = -1, keepdim = True)
        idx = torch.cat([idx, next_id], dim = -1)
        logits = model(next_id, kv_cache)
    return idx


# ---------------------------------------------------------------------------
# 第 ① 步：加载预训练权重
#   HF 的权重文件 model.safetensors：一个"名字 -> 张量"的字典，按 bf16 存
#   safetensors 是只存张量的格式（不像 .bin 用 pickle，加载时不会执行任意代码）
#   文件里没有 lm_head.weight：它和 embed_tokens 共用一份（tie_word_embeddings），拷 embed_tokens 就够了
# ---------------------------------------------------------------------------
def from_pretrained(name: str = QWEN_NAME, dtype: torch.dtype = torch.float32) -> Qwen2:
    """返回 eval 模式、参数类型为 dtype 的 Qwen2（在 CPU 上；要用 GPU 自己再 .to("cuda")）"""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config = QwenConfig.from_pretrained(name)
    model = Qwen2(config)
    sd = model.state_dict()                                      # 我们的模型：名字 -> 张量（float32）
    hf_sd = load_file(hf_hub_download(name, "model.safetensors"))  # 权重文件：名字 -> 张量（bf16）

    copied = set()
    with torch.no_grad():
        for hf_key, hf_tensor in hf_sd.items():
            key = hf_key.removeprefix("model.")
            if key not in sd:
                raise KeyError(f"权重文件里的 {hf_key} 在模型里找不到对应的参数（模块名写错了？）")
            if sd[key].shape != hf_tensor.shape:
                raise ValueError(f"{key} 的形状不一致：模型里 {tuple(sd[key].shape)}，文件里 {tuple(hf_tensor.shape)}")
            sd[key].copy_(hf_tensor)          # copy_ 会把 bf16 自动转成目标的类型（float32）
            copied.add(key)

    tied = {"lm_head.weight"} if config.tie_word_embeddings else set()
    missing = set(sd) - copied - tied
    if missing:
        raise KeyError(f"这些参数在权重文件里没找到：{sorted(missing)[:5]} ...")
    return model.to(dtype).eval()
