r"""
逐步检查 miniserve/qwen.py：和 HF transformers 的 Qwen2 实现做数值对比（沿用 M0 check_model.py 的做法）。

用法（在项目根目录 D:\infra\miniserve 下）：
    "D:\Program Files\Anaconda\envs\miniserve\python.exe" tests/check_qwen.py

按顺序检查，遇到第一个没写完 / 没通过的步骤就停。
权重已经在本机的 HF 缓存里，所以这里用离线模式（不联网检查更新，启动更快）。
"""
import functools
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 必须在 import transformers / huggingface_hub 之前设

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve import qwen as Q  # noqa: E402

RTOL = 1e-5  # 相对误差：float32 下数值越大，绝对误差也越大


def compare(name: str, got, want: torch.Tensor, rtol: float = RTOL) -> float:
    assert isinstance(got, torch.Tensor), f"{name} 应该返回 Tensor，实际是 {type(got).__name__}（是不是忘了 return？）"
    assert got.shape == want.shape, f"{name} 形状应为 {tuple(want.shape)}，实际是 {tuple(got.shape)}"
    err = ((got - want).abs().max() / want.abs().max().clamp(min=1.0)).item()
    assert err < rtol, f"{name} 形状对了，但数值和 HF 不一致（最大相对误差 {err:.2e}）"
    return err


@functools.cache
def our_model() -> "Q.Qwen2":
    """第 ① 步加载一次，后面的步骤借用里面的真实权重（真实权重的数值分布比随机数更像实际情况）"""
    return Q.from_pretrained()


# ---------------------------------------------------------------------------
# 第 ①a 步：config
# ---------------------------------------------------------------------------
def check_config() -> None:
    from transformers import AutoConfig

    ours = Q.QwenConfig.from_pretrained()
    hf = AutoConfig.from_pretrained(Q.QWEN_NAME)
    # transformers 5.x 把 rope_theta 挪进了 rope_parameters 字典里
    hf_theta = getattr(hf, "rope_theta", None) or hf.rope_parameters["rope_theta"]
    pairs = [
        ("vocab_size", ours.vocab_size, hf.vocab_size),
        ("hidden_size", ours.hidden_size, hf.hidden_size),
        ("intermediate_size", ours.intermediate_size, hf.intermediate_size),
        ("num_hidden_layers", ours.num_hidden_layers, hf.num_hidden_layers),
        ("num_attention_heads", ours.num_attention_heads, hf.num_attention_heads),
        ("num_key_value_heads", ours.num_key_value_heads, hf.num_key_value_heads),
        ("rms_norm_eps", ours.rms_norm_eps, hf.rms_norm_eps),
        ("rope_theta", ours.rope_theta, hf_theta),
        ("tie_word_embeddings", ours.tie_word_embeddings, hf.tie_word_embeddings),
        ("head_dim", ours.head_dim, getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads),
    ]
    for name, a, b in pairs:
        assert a == b, f"config 的 {name} 不一致：我们 {a}，HF {b}"
    assert ours.num_attention_heads % ours.num_key_value_heads == 0, "q 的头数应该是 k/v 头数的整数倍"
    print(f"         {ours.num_hidden_layers} 层，C = {ours.hidden_size}，"
          f"q 头 {ours.num_attention_heads} 个 / k、v 头 {ours.num_key_value_heads} 个，head_dim = {ours.head_dim}，"
          f"MLP 中间层 {ours.intermediate_size}，词表 {ours.vocab_size}")


# ---------------------------------------------------------------------------
# 第 ①b 步：from_pretrained（和权重文件逐个参数对比）
# ---------------------------------------------------------------------------
def check_from_pretrained() -> None:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    model = our_model()
    assert isinstance(model, Q.Qwen2), f"from_pretrained 应该返回 Qwen2，实际是 {type(model).__name__}"
    assert model.lm_head.weight is model.embed_tokens.weight, "lm_head 和 embed_tokens 没有共享权重"
    n_params = sum(p.numel() for p in model.parameters())   # 共享的权重只算一次
    assert n_params == 494_032_768, f"参数量应为 494,032,768，实际是 {n_params:,}"

    sd = model.state_dict()
    with safe_open(hf_hub_download(Q.QWEN_NAME, "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        for hf_key in keys:
            key = hf_key.removeprefix("model.")
            want = f.get_tensor(hf_key).to(torch.float32)
            assert torch.equal(sd[key], want), f"参数 {key} 和权重文件不一致"
    print(f"         权重文件里 {len(keys)} 个张量全部对上；参数量 {n_params:,}（lm_head 和 embed_tokens 共用一份）")


# ---------------------------------------------------------------------------
# 第 ② 步：RMSNorm
# ---------------------------------------------------------------------------
def _rmsnorm_wrong_versions(x: torch.Tensor, w: torch.Tensor, eps: float) -> list[tuple[str, torch.Tensor]]:
    """常见写错的版本：输出和哪个对上，就提示哪个"""
    ms = x.pow(2).mean(-1, keepdim=True)
    xc = x - x.mean(-1, keepdim=True)
    wrong_dims = [
        ("求平均的维度不对：应该沿最后一维（每个 token 自己的 C 个数）",
         x * torch.rsqrt(x.pow(2).mean(d, keepdim=True) + eps) * w)
        for d in range(x.dim() - 1)
    ]
    return wrong_dims + [
        ("减了均值：这是 LayerNorm 的做法，RMSNorm 不减均值",
         xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps) * w),
        ("用了方差（var 会先减均值）：RMSNorm 要的是 x² 的平均",
         x * torch.rsqrt(x.var(-1, keepdim=True, unbiased=False) + eps) * w),
        ("用了方差（var 会先减均值）：RMSNorm 要的是 x² 的平均",
         x * torch.rsqrt(x.var(-1, keepdim=True) + eps) * w),
        ("没乘 self.weight", x * torch.rsqrt(ms + eps)),
        ("eps 加在了根号外面：应该是 sqrt(mean(x²) + eps)", x / (ms.sqrt() + eps) * w),
        ("没加 eps", x * torch.rsqrt(ms) * w),
        ("用了求和而不是平均", x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps) * w),
        ("乘了 sqrt：rsqrt 已经是 1 / sqrt，要乘 rsqrt，或者除以 sqrt", x * (ms + eps).sqrt() * w),
        ("平均的是 x 不是 x²", x * torch.rsqrt(x.mean(-1, keepdim=True) + eps) * w),
        ("把 eps 写死成了 1e-6：要用 self.eps（不同的 RMSNorm 可以有不同的 eps）",
         x * torch.rsqrt(ms + 1e-6) * w),
    ]


def _check_one_rmsnorm(name: str, norm, ref, x: torch.Tensor) -> float:
    x_before = x.clone()
    with torch.no_grad():
        try:
            got = norm(x)
        except NotImplementedError:   # 它是 RuntimeError 的子类，要先放过去，交给 main 显示 TODO
            raise
        except RuntimeError as e:
            raise AssertionError(f"{name}：运行出错（{e}）。是不是求平均时没保留最后一维（keepdim），"
                                 "导致 (..., 1) 变成了 (...)，没法和 x 广播？") from None
        want = ref(x)
    assert torch.equal(x, x_before), f"{name}：forward 把输入 x 改掉了（用了原地操作？比如 x.mul_ 或 x *= ...）"
    try:
        return compare(name, got, want)
    except AssertionError:
        if isinstance(got, torch.Tensor):
            for hint, wrong in _rmsnorm_wrong_versions(x, norm.weight.detach(), ref.variance_epsilon):
                if wrong.shape == got.shape and torch.allclose(got, wrong, rtol=1e-5, atol=1e-6, equal_nan=True):
                    raise AssertionError(f"{name} 和 HF 不一致：{hint}") from None
        raise


def check_rmsnorm() -> float:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    cfg = our_model().config
    C, eps = cfg.hidden_size, cfg.rms_norm_eps
    real_w = our_model().layers[0].input_layernorm.weight.detach().clone()   # 真实权重，不全是 1

    def make_pair(eps_: float):
        norm = Q.RMSNorm(C, eps_)
        ref = Qwen2RMSNorm(C, eps=eps_)
        with torch.no_grad():
            norm.weight.copy_(real_w)
            ref.weight.copy_(real_w)
        return norm, ref

    norm, ref = make_pair(eps)
    errs = [
        # 均值不为 0：写成 LayerNorm（减了均值）或用了方差，都能被抓出来
        _check_one_rmsnorm("RMSNorm 输出 (B, T, C)", norm, ref, torch.randn(2, 5, C) * 3 + 1),
        # packed 路径里输入是 (N, C)
        _check_one_rmsnorm("RMSNorm 输出 (N, C)", norm, ref, torch.randn(7, C) * 3 + 1),
        # x 很小时 mean(x²) ≈ 1e-8，比 eps = 1e-6 还小：没加 eps / eps 位置错，结果会差很多
        _check_one_rmsnorm("RMSNorm 输出（x 很小，eps 起作用）", norm, ref, torch.randn(2, 5, C) * 1e-4),
    ]
    # 换一个 eps：如果 forward 里把 1e-6 写死了，而不是用 self.eps，这里会不一致
    norm2, ref2 = make_pair(1e-2)
    errs.append(_check_one_rmsnorm("RMSNorm 输出（换一个 eps）", norm2, ref2, torch.randn(2, 5, C) * 0.1))
    return max(errs)


# ---------------------------------------------------------------------------
# 第 ③ 步：RoPE
# ---------------------------------------------------------------------------
ROPE_ATOL = 1e-3   # 角度最大约 1000 弧度，float32 在 1000 附近的精度约 6e-5，不同的等价算法会差几个这个量级


def hf_cos_sin(positions: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """用 HF 的 Qwen2RotaryEmbedding 算参考答案（改 config 里的 head_dim 和 theta）"""
    import copy

    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    cfg = copy.deepcopy(AutoConfig.from_pretrained(Q.QWEN_NAME))
    cfg.head_dim = head_dim
    cfg.rope_parameters = {"rope_type": "default", "rope_theta": theta}
    rot = Qwen2RotaryEmbedding(cfg)
    return rot(torch.zeros(1, dtype=torch.float32), positions)


def _cos_sin_wrong_versions(positions: torch.Tensor, head_dim: int, theta: float):
    p = positions.float()[..., None]
    i2 = torch.arange(0, head_dim, 2).float()
    good = 1.0 / theta ** (i2 / head_dim)
    cands = {
        "复制方式不对：要用 cat 拼成 (f0 … f31, f0 … f31)，不是 f0 f0 f1 f1 …（Qwen 是前后两半配对，不是相邻两维配对）":
            (p * good).repeat_interleave(2, dim=-1),
        "频率的指数少了负号（或者忘了取倒数）：inv_freq = theta ** (−2i / head_dim)":
            torch.cat([p * theta ** (i2 / head_dim)] * 2, dim=-1),
        "指数应该是 2i / head_dim（i = 0, 1, …），也就是 arange 带步长 2 再除以 head_dim":
            torch.cat([p * (1.0 / theta ** (torch.arange(head_dim // 2).float() / head_dim))] * 2, dim=-1),
        "theta 写死了：要用传进来的 theta 参数":
            torch.cat([p * (1.0 / 1e6 ** (i2 / head_dim))] * 2, dim=-1),
        "theta 写死了：要用传进来的 theta 参数（Qwen2.5 的 theta 是 1e6，不是原版 RoPE 的 1e4）":
            torch.cat([p * (1.0 / 1e4 ** (i2 / head_dim))] * 2, dim=-1),
    }
    return [(hint, ang.cos(), ang.sin()) for hint, ang in cands.items()]


def _check_cos_sin_case(name: str, positions: torch.Tensor, head_dim: int, theta: float) -> float:
    try:
        out = Q.rope_cos_sin(positions, head_dim, theta)
    except NotImplementedError:
        raise
    except RuntimeError as e:
        raise AssertionError(f"{name}：运行出错（{e}）。head_dim 是不是写死了？角度拼完应该正好 head_dim 维") from None
    assert isinstance(out, tuple) and len(out) == 2, f"rope_cos_sin 应该返回 (cos, sin) 两个张量，实际是 {type(out).__name__}"
    cos, sin = out
    want_cos, want_sin = hf_cos_sin(positions, head_dim, theta)
    shape = (*positions.shape, head_dim)
    if cos.shape != shape or sin.shape != shape:
        hint = ("忘了把角度复制一份拼在后面？" if cos.shape[-1] == head_dim // 2
                else "head_dim 是不是写死了？要用传进来的 head_dim 参数")
        raise AssertionError(f"{name}：cos、sin 的形状应为 {shape}，实际是 {tuple(cos.shape)}、{tuple(sin.shape)}（{hint}）")
    assert cos.dtype == torch.float32, f"{name}：cos、sin 应该是 float32，实际是 {cos.dtype}"
    if torch.allclose(cos, want_sin, atol=ROPE_ATOL) and torch.allclose(sin, want_cos, atol=ROPE_ATOL):
        raise AssertionError(f"{name}：cos 和 sin 返回反了")
    err = max((cos - want_cos).abs().max().item(), (sin - want_sin).abs().max().item())
    if err > ROPE_ATOL:
        for hint, wc, ws in _cos_sin_wrong_versions(positions, head_dim, theta):
            if torch.allclose(cos, wc, atol=ROPE_ATOL) and torch.allclose(sin, ws, atol=ROPE_ATOL):
                raise AssertionError(f"{name} 和 HF 不一致：{hint}")
        raise AssertionError(f"{name} 和 HF 不一致（最大误差 {err:.2e}）")
    return err


def check_rope_cos_sin() -> float:
    cfg = our_model().config
    errs = [
        _check_cos_sin_case("cos/sin（Qwen 的 head_dim、theta，positions 一维）",
                            torch.tensor([0, 1, 2, 3, 7, 100, 513, 1000]), cfg.head_dim, cfg.rope_theta),
        # 换 head_dim 和 theta：抓"写死了"；positions 二维：抓只支持一维的写法
        _check_cos_sin_case("cos/sin（head_dim = 16、theta = 1e4，positions 二维）",
                            torch.tensor([[0, 5, 9], [2, 3, 40]]), 16, 10000.0),
    ]
    # 设备：在 "meta" 设备上跑一遍（不真正算，只看张量放在哪），没把 arange 放到 positions 的设备上会出错
    try:
        cos, _ = Q.rope_cos_sin(torch.arange(4, device="meta"), cfg.head_dim, cfg.rope_theta)
    except RuntimeError as e:
        raise AssertionError(f"positions 在别的设备上时出错（{e}）。arange 要加 device=positions.device") from None
    assert cos.device.type == "meta", "positions 在别的设备上时，cos 跑到了 CPU 上：arange 要加 device=positions.device"
    return max(errs)


def _rotate_wrong_versions(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    inter = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return [
        ("rotate_half 的符号反了：应该是 (−x2, x1)，不是 (x2, −x1)", x * cos + torch.cat([x2, -x1], -1) * sin),
        ("rotate_half 的顺序不对：应该是 (−x2, x1)", x * cos + torch.cat([-x1, x2], -1) * sin),
        ("cos 和 sin 用反了：x 乘 cos，rotate_half(x) 乘 sin", x * sin + torch.cat([-x2, x1], -1) * cos),
        ("少了 rotate_half(x) · sin 那一项", x * cos),
        ("拼的是 (x1, x2)，这就是 x 本身，没有交换：rotate_half 要拼成 (−x2, x1)", x * cos + x * sin),
        ("配对方式不对：Qwen 是前后两半配对 (x[i], x[i + 半])，不是相邻两维 (x[2i], x[2i+1])", x * cos + inter * sin),
        ("一半的位置写死成了 32：要用 x.shape[-1] // 2",
         x * cos + torch.cat([-x[..., 32:], x[..., :32]], -1) * sin),
    ]


def _check_apply_case(name: str, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, want: torch.Tensor) -> float:
    x_before = x.clone()
    try:
        got = Q.apply_rope(x, cos, sin)
    except NotImplementedError:
        raise
    except RuntimeError as e:
        raise AssertionError(f"{name}：运行出错（{e}）。切一半时是不是把 head_dim 的一半写死了？") from None
    assert torch.equal(x, x_before), f"{name}：apply_rope 把输入 x 改掉了（用了原地操作？）"
    try:
        return compare(name, got, want)
    except AssertionError:
        if isinstance(got, torch.Tensor):
            for hint, wrong in _rotate_wrong_versions(x, cos, sin):
                if wrong.shape == got.shape and torch.allclose(got, wrong, rtol=1e-5, atol=1e-6):
                    raise AssertionError(f"{name} 和 HF 不一致：{hint}") from None
        raise


def check_apply_rope() -> float:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    cfg = our_model().config
    hs = cfg.head_dim
    errs = []

    # (B, n_head, T, hs)：第 ④ 步 attention 里的形状；cos (T, hs) 直接广播
    pos = torch.arange(10, 15)
    cos, sin = hf_cos_sin(pos, hs, cfg.rope_theta)
    x = torch.randn(2, 3, 5, hs)
    want = apply_rotary_pos_emb(x, x, cos[None], sin[None])[0]
    errs.append(_check_apply_case("apply_rope (B, n_head, T, hs)", x, cos, sin, want))

    # packed 的形状 (N, n_head, hs)：cos 要调成 (N, 1, hs) 才能广播
    pos = torch.tensor([0, 1, 2, 0, 1, 7])
    cos, sin = hf_cos_sin(pos, hs, cfg.rope_theta)
    x = torch.randn(6, 3, hs)
    want = apply_rotary_pos_emb(x.transpose(0, 1)[None], x.transpose(0, 1)[None], cos[None], sin[None])[0][0].transpose(0, 1)
    errs.append(_check_apply_case("apply_rope (N, n_head, hs)", x, cos[:, None, :], sin[:, None, :], want))

    # head_dim = 16：抓把一半写死成 32 的
    pos = torch.arange(4)
    cos, sin = hf_cos_sin(pos, 16, cfg.rope_theta)
    x = torch.randn(1, 2, 4, 16)
    want = apply_rotary_pos_emb(x, x, cos[None], sin[None])[0]
    errs.append(_check_apply_case("apply_rope（head_dim = 16）", x, cos, sin, want))
    return max(errs)


def check_rope_relative() -> None:
    """RoPE 的核心性质：q·k 只和两个位置的差有关。用你写的两个函数验证"""
    cfg = our_model().config
    hs, theta = cfg.head_dim, cfg.rope_theta
    q, k = torch.randn(hs), torch.randn(hs)

    def score(m: int, n: int) -> float:
        cos, sin = Q.rope_cos_sin(torch.tensor([m, n]), hs, theta)
        return (Q.apply_rope(q, cos[0], sin[0]) @ Q.apply_rope(k, cos[1], sin[1])).item()

    same_gap = [(3, 1), (103, 101), (503, 501)]
    scores = [score(m, n) for m, n in same_gap]
    print("         q 在位置 m、k 在位置 n，转完之后的点积：")
    for (m, n), s in zip(same_gap, scores):
        print(f"           m = {m:3d}, n = {n:3d}（差 2）：{s:+.4f}")
    print(f"           m =   9, n =   1（差 8）：{score(9, 1):+.4f}    ← 距离变了，分数就变了")
    assert max(scores) - min(scores) < 1e-3 * max(1.0, abs(scores[0])), "距离相同、分数却不同：旋转没写对"


# ---------------------------------------------------------------------------
# 第 ④ 步：GQA attention
# ---------------------------------------------------------------------------
def _hf_attention(state_dict: dict):
    import copy

    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    cfg = copy.deepcopy(AutoConfig.from_pretrained(Q.QWEN_NAME))
    cfg._attn_implementation = "eager"      # 最朴素的实现：显式 mask + softmax
    ref = Qwen2Attention(cfg, layer_idx=0).eval()
    ref.load_state_dict(state_dict)         # 参数名和我们的一样：q_proj.weight、q_proj.bias ...
    return ref


def _ref_attention(attn, x, cos, sin, rep="interleave", rope_q=True, rope_k=True, rope_v=False,
                   scale_dim=None, mask=True, o_proj=True) -> torch.Tensor:
    """参考实现（用 HF 的 apply_rotary_pos_emb）。改某个开关就得到一种常见的写错版本，用来给提示"""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    B, T, C = x.shape
    nh, nkv, hs = attn.n_head, attn.n_kv_head, attn.head_dim
    q = attn.q_proj(x).view(B, T, nh, hs).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, nkv, hs).transpose(1, 2)
    v = attn.v_proj(x).view(B, T, nkv, hs).transpose(1, 2)
    rq, _ = apply_rotary_pos_emb(q, q, cos[None], sin[None])
    rk, rv = apply_rotary_pos_emb(k, v, cos[None], sin[None])
    q, k, v = (rq if rope_q else q), (rk if rope_k else k), (rv if rope_v else v)
    if rep == "interleave":
        k, v = k.repeat_interleave(nh // nkv, dim=1), v.repeat_interleave(nh // nkv, dim=1)
    else:   # "repeat"：整体重复，头的顺序变成 0,1,0,1,...
        k, v = k.repeat(1, nh // nkv, 1, 1), v.repeat(1, nh // nkv, 1, 1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(scale_dim or hs)
    if mask:
        scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
    y = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, nh * hs)
    return attn.o_proj(y) if o_proj else y


def _attention_wrong_versions(attn, x, cos, sin):
    C = x.shape[-1]
    return [
        ("k、v 的头复制方式不对：要用 repeat_interleave（0,0,…,0,1,1,…,1），"
         "不是 repeat（0,1,0,1,…）——q 头 0~6 要共用 k/v 头 0", dict(rep="repeat")),
        ("k 没有转 RoPE", dict(rope_k=False)),
        ("q 没有转 RoPE", dict(rope_q=False)),
        ("v 也转了 RoPE：只有 q、k 要转", dict(rope_v=True)),
        ("除以的应该是 sqrt(head_dim)，不是 sqrt(C)", dict(scale_dim=C)),
        ("少了因果 mask", dict(mask=False)),
        ("最后少了 o_proj", dict(o_proj=False)),
    ]


def check_attention() -> float:
    from miniserve.model import LayerKVCache

    cfg = our_model().config
    hs, nkv = cfg.head_dim, cfg.num_key_value_heads
    attn = Q.Attention(cfg)
    attn.load_state_dict(our_model().layers[0].self_attn.state_dict())   # 第 0 层的真实权重
    attn.eval()
    ref = _hf_attention(attn.state_dict())

    B, T = 2, 7
    x = torch.randn(B, T, cfg.hidden_size)
    cos, sin = hf_cos_sin(torch.arange(T), hs, cfg.rope_theta)
    hf_mask = torch.full((T, T), float("-inf")).triu(1)[None, None]       # HF 要的是加到分数上的 mask
    with torch.no_grad():
        want = ref(x, (cos[None], sin[None]), hf_mask)[0]

    # 1）不带缓存，一次送 7 个 token
    with torch.no_grad():
        try:
            got = attn(x, cos, sin)
        except NotImplementedError:
            raise
        except RuntimeError as e:
            raise AssertionError(f"运行出错（{e}）。k、v 拆头时用的是 n_kv_head 吗？"
                                 "复制完 k、v 的头数和 q 一样吗？") from None
        try:
            err = compare("attention 输出（不带缓存）", got, want, rtol=1e-4)   # 运算顺序不同，误差在 1e-5 量级
        except AssertionError:
            if isinstance(got, torch.Tensor):
                for hint, flags in _attention_wrong_versions(attn, x, cos, sin):
                    wrong = _ref_attention(attn, x, cos, sin, **flags)
                    if wrong.shape == got.shape and torch.allclose(got, wrong, rtol=1e-4, atol=1e-5):
                        raise AssertionError(f"attention 输出和 HF 不一致：{hint}") from None
            raise

        # 因果性：改最后一个 token，前面的输出不应该变
        x2 = x.clone()
        x2[:, -1] += 1.0
        got2 = attn(x2, cos, sin)
        assert torch.allclose(got[:, :-1], got2[:, :-1], atol=1e-5), "前面的 token 看到了后面的：检查因果 mask"

    # 2）带缓存：先送 4 个（prefill），再送 1 个，再送 2 个；拼起来应该和一次送 7 个一样
    cache = LayerKVCache()
    outs = []
    with torch.no_grad():
        for a, b in [(0, 4), (4, 5), (5, 7)]:
            outs.append(attn(x[:, a:b], cos[a:b], sin[a:b], cache=cache))
            assert cache.k is not None and cache.v is not None, "传了 cache，但 cache.k / cache.v 还是 None：忘了存回去？"
            want_shape = (B, nkv, b, hs)
            if tuple(cache.k.shape) == (B, cfg.num_attention_heads, b, hs):
                raise AssertionError(f"cache.k 有 {cfg.num_attention_heads} 个头：存进缓存的应该是复制之前的 k、v"
                                     f"（{nkv} 个头），先存缓存、再复制，GQA 才能省显存")
            assert tuple(cache.k.shape) == want_shape and tuple(cache.v.shape) == want_shape, (
                f"送完前 {b} 个 token 后，cache.k、cache.v 的形状应为 {want_shape}，"
                f"实际是 {tuple(cache.k.shape)}、{tuple(cache.v.shape)}")
        # 缓存里的 k 应该是转过 RoPE 的
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        k_raw = attn.k_proj(x).view(B, T, nkv, hs).transpose(1, 2)
        k_rot, _ = apply_rotary_pos_emb(k_raw, k_raw, cos[None], sin[None])
        if torch.allclose(cache.k, k_raw, atol=1e-5):
            raise AssertionError("缓存里存的 k 没转 RoPE：要先转 RoPE 再存（位置在写入时就定了，以后不用再转）")
        assert torch.allclose(cache.k, k_rot, atol=1e-4), "缓存里的 k 和 HF 算出来的（转过 RoPE 的）k 不一致：拼接是不是旧的在前、新的在后？"
        got_cached = torch.cat(outs, dim=1)
    try:
        err = max(err, compare("attention 输出（带缓存，分 3 次送）", got_cached, want, rtol=1e-4))
    except AssertionError as e:
        raise AssertionError(f"{e}。不带缓存是对的，问题在带缓存的部分：mask 是不是 (T, L) 且 diagonal = L − T？"
                             "拼接是不是旧的在前、新的在后？") from None
    return err


# ---------------------------------------------------------------------------
# 第 ⑤ 步：SwiGLU MLP
# ---------------------------------------------------------------------------
def _mlp_wrong_versions(mlp, x):
    g, u, d = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    F_ = torch.nn.functional
    return [
        ("SiLU 加在了 up 那条路上：应该是 SiLU(gate(x)) * up(x)", d(g(x) * F_.silu(u(x)))),
        ("用了 GELU：Qwen 的激活是 SiLU（F.silu）", d(F_.gelu(g(x)) * u(x))),
        ("用了 GELU（tanh 近似）：Qwen 的激活是 SiLU（F.silu）", d(F_.gelu(g(x), approximate="tanh") * u(x))),
        ("用了 ReLU：Qwen 的激活是 SiLU（F.silu）", d(F_.relu(g(x)) * u(x))),
        ("用了 sigmoid：SiLU(z) = z * sigmoid(z)，直接用 F.silu", d(torch.sigmoid(g(x)) * u(x))),
        ("gate 没过激活", d(g(x) * u(x))),
        ("两条路是相加，应该是逐个元素相乘", d(F_.silu(g(x)) + u(x))),
        ("没乘 up(x)：只有 gate 那一条路", d(F_.silu(g(x)))),
    ]


def check_mlp() -> float:
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    cfg = our_model().config
    mlp = Q.MLP(cfg)
    mlp.load_state_dict(our_model().layers[0].mlp.state_dict())     # 第 0 层的真实权重
    ref = Qwen2MLP(AutoConfig.from_pretrained(Q.QWEN_NAME))
    ref.load_state_dict(mlp.state_dict())

    errs = []
    for name, x in [("MLP 输出 (B, T, C)", torch.randn(2, 5, cfg.hidden_size)),
                    ("MLP 输出 (N, C)", torch.randn(7, cfg.hidden_size))]:
        x_before = x.clone()
        with torch.no_grad():
            try:
                got = mlp(x)
            except NotImplementedError:
                raise
            except (RuntimeError, ValueError) as e:
                raise AssertionError(f"{name}：运行出错（{e}）。是不是按固定维数拆了 x.shape？"
                                     "这里不用拆，线性层和 * 只看最后一维") from None
            want = ref(x)
            if isinstance(got, torch.Tensor) and got.shape[-1] == cfg.intermediate_size:
                raise AssertionError(f"{name}：最后一维是 {cfg.intermediate_size}，少了 down_proj 缩回 C 维")
            assert torch.equal(x, x_before), f"{name}：forward 把输入 x 改掉了（用了原地操作？）"
            try:
                errs.append(compare(name, got, want))
            except AssertionError:
                if isinstance(got, torch.Tensor):
                    for hint, wrong in _mlp_wrong_versions(mlp, x):
                        if wrong.shape == got.shape and torch.allclose(got, wrong, rtol=1e-4, atol=1e-5):
                            raise AssertionError(f"{name} 和 HF 不一致：{hint}") from None
                raise
    return max(errs)


STEPS = [
    ("第 ①a 步 config", check_config),
    ("第 ①b 步 from_pretrained", check_from_pretrained),
    ("第 ② 步 RMSNorm", check_rmsnorm),
    ("第 ③a 步 rope_cos_sin", check_rope_cos_sin),
    ("第 ③b 步 apply_rope", check_apply_rope),
    ("第 ③c 步 相对位置性质（演示）", check_rope_relative),
    ("第 ④ 步 GQA attention", check_attention),
    ("第 ⑤ 步 SwiGLU MLP", check_mlp),
]


def main() -> None:
    torch.manual_seed(0)
    for label, check in STEPS:
        try:
            err = check()
        except NotImplementedError:
            print(f"[ TODO ] {label}：还没写")
            break
        except AssertionError as e:
            print(f"[ FAIL ] {label}：{e}")
            break
        except Exception as e:  # 形状不匹配等运行时错误
            print(f"[ERROR ] {label}：{type(e).__name__}: {e}")
            break
        else:
            detail = "" if err is None else f"（与 HF 最大相对误差 {err:.1e}）"
            print(f"[ PASS ] {label}{detail}")


if __name__ == "__main__":
    main()
