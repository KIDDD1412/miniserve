"""
逐步检查 M1 的 KV Cache。

核心思路：同一段序列，"一次性整段送进去"和"带着缓存分几次送进去"，结果必须完全一样。

用法（在项目根目录 D:\\infra\\miniserve 下）：
    python tests/check_kv_cache.py
"""
import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve import model as M  # noqa: E402

RTOL = 1e-5


def compare(name, got, want):
    assert isinstance(got, torch.Tensor), f"{name} 应该是 Tensor，实际是 {type(got).__name__}"
    assert got.shape == want.shape, f"{name} 形状应为 {tuple(want.shape)}，实际是 {tuple(got.shape)}"
    err = ((got - want).abs().max() / want.abs().max().clamp(min=1.0)).item()
    assert err < RTOL, f"{name} 形状对了，但数值不一致（最大相对误差 {err:.2e}）"
    return err


def run_in_chunks(attn, x, chunks):
    """按 chunks 指定的长度，把 x 分几段送进 attention，每段都带着同一份缓存"""
    cache = M.LayerKVCache()
    outs, start = [], 0
    for n in chunks:
        outs.append(attn(x[:, start:start + n], cache=cache))
        start += n
    return torch.cat(outs, dim=1), cache


def check_attention_cache():
    if "cache" not in inspect.signature(M.CausalSelfAttention.forward).parameters:
        raise NotImplementedError
    cfg = M.GPTConfig()
    attn = M.CausalSelfAttention(cfg).eval()
    B, T = 2, 7
    x = torch.randn(B, T, cfg.n_embd)
    hs = cfg.n_embd // cfg.n_head

    with torch.no_grad():
        full = attn(x)
        assert isinstance(full, torch.Tensor), "不传 cache 时应该和原来一样返回 Tensor"

        # 情况 1：先整段处理前 T-1 个（prefill），再处理最后 1 个（decode）
        got, cache = run_in_chunks(attn, x, [T - 1, 1])
        assert cache.k is not None, "cache.k 还是 None：拼好的 k 没有存回缓存"
        assert tuple(cache.k.shape) == (B, cfg.n_head, T, hs), (
            f"处理完 {T} 个 token 后 cache.k 应为 {(B, cfg.n_head, T, hs)}，实际是 {tuple(cache.k.shape)}"
        )
        compare("prefill + 1 步 decode", got, full)

        # 情况 2：一个一个 token 地送（纯 decode）
        got, _ = run_in_chunks(attn, x, [1] * T)
        compare("逐 token decode", got, full)

        # 情况 3：不规则地分段，检查 mask 的 diagonal 偏移
        try:
            got, _ = run_in_chunks(attn, x, [3, 1, 2, 1])
            compare("分段 [3, 1, 2, 1]", got, full)
        except AssertionError as e:
            raise AssertionError(f"{e}。前两种情况都对，只有分段出错：检查 mask 是不是 (T, L)，diagonal 是不是 L - T") from None
    return 0.0


def check_model_cache():
    if "kv_cache" not in inspect.signature(M.GPT2.forward).parameters:
        raise NotImplementedError
    # 用一个小模型测，跑得快；也顺便检查有没有把 768、12 之类的数字写死
    cfg = M.GPTConfig(block_size=64, vocab_size=100, n_layer=3, n_head=4, n_embd=32)
    model = M.GPT2(cfg).eval()
    B, T = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        full = model(idx)
        kv_cache = M.new_kv_cache(cfg)
        outs, start = [], 0
        for n in [5, 1, 1, 1]:  # 先 prefill 5 个，再 decode 3 步
            outs.append(model(idx[:, start:start + n], kv_cache=kv_cache))
            start += n
        got = torch.cat(outs, dim=1)

    hs = cfg.n_embd // cfg.n_head
    for i, layer in enumerate(kv_cache):
        assert layer.k is not None, f"第 {i} 层的缓存是空的：每个 Block 都要传入自己那一层的缓存"
        extra = "（长度是层数的倍数：是不是所有层都往同一个缓存里存？）" if layer.k.shape[2] == T * cfg.n_layer else ""
        assert tuple(layer.k.shape) == (B, cfg.n_head, T, hs), (
            f"第 {i} 层 cache.k 应为 {(B, cfg.n_head, T, hs)}，实际是 {tuple(layer.k.shape)}{extra}"
        )
    assert kv_cache[0].k is not kv_cache[1].k, "不同层存的是同一份 k：每层要用自己的缓存"
    try:
        compare("分段带缓存的 logits", got, full)
    except AssertionError as e:
        raise AssertionError(f"{e}。检查：位置编号有没有从 past 接着往后数？") from None
    return 0.0


STEPS = [
    ("M1 第 2 步：带缓存的 attention", check_attention_cache),
    ("M1 第 3 步：带缓存的整个模型", check_model_cache),
]


def main():
    torch.manual_seed(0)
    for label, check in STEPS:
        try:
            check()
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
            print(f"[ PASS ] {label}")


if __name__ == "__main__":
    main()
