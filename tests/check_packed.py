"""
检查 M2 的压平前向：多个请求压成一排一起算，结果必须和"每个请求单独用 M1 的代码算"完全一样。

用法（在项目根目录 D:\\infra\\miniserve 下）：
    python tests/check_packed.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve import model as M  # noqa: E402
from miniserve import packed as P  # noqa: E402

RTOL = 1e-5
CFG = M.GPTConfig(block_size=64, vocab_size=100, n_layer=3, n_head=4, n_embd=32)


def compare(name, got, want):
    assert isinstance(got, torch.Tensor), f"{name} 应该是 Tensor，实际是 {type(got).__name__}"
    assert got.shape == want.shape, f"{name} 形状应为 {tuple(want.shape)}，实际是 {tuple(got.shape)}"
    err = ((got - want).abs().max() / want.abs().max().clamp(min=1.0)).item()
    assert err < RTOL, f"{name} 形状对了，但数值不一致（最大相对误差 {err:.2e}）"


def _todo_guard(fn, *args):
    try:
        return fn(*args)
    except (RuntimeError, ValueError) as e:
        if "non-empty" in str(e):  # outs 还是空的：for 循环里还没写
            raise NotImplementedError from None
        raise


def check_packed_attention():
    attn = M.CausalSelfAttention(CFG).eval()
    rounds = [[3, 1, 5], [1, 2, 1]]  # 第 1 轮：3 个请求各自 prefill；第 2 轮：混合（decode / 再送 2 个 / decode）
    packed_caches = [M.LayerKVCache() for _ in rounds[0]]
    single_caches = [M.LayerKVCache() for _ in rounds[0]]

    with torch.no_grad():
        for r, lens in enumerate(rounds):
            xs = [torch.randn(n, CFG.n_embd) for n in lens]
            got = _todo_guard(P.packed_attention, attn, torch.cat(xs), lens, packed_caches)
            want = torch.cat([attn(x[None], cache=c)[0] for x, c in zip(xs, single_caches)])
            try:
                compare(f"第 {r + 1} 轮输出", got, want)
            except AssertionError as e:
                hint = "（第 1 轮对了、第 2 轮错：检查缓存有没有拼接和存回、mask 的 diagonal）" if r == 1 else ""
                raise AssertionError(f"{e}{hint}") from None
            for i, (pc, sc) in enumerate(zip(packed_caches, single_caches)):
                assert pc.k is not None and pc.k.shape == sc.k.shape, (
                    f"第 {r + 1} 轮后请求 {i} 的缓存形状应为 {tuple(sc.k.shape)}，"
                    f"实际是 {None if pc.k is None else tuple(pc.k.shape)}"
                )


def check_packed_forward():
    model = M.GPT2(CFG).eval()
    n_req = 3
    packed_caches = [M.new_kv_cache(CFG) for _ in range(n_req)]
    single_caches = [M.new_kv_cache(CFG) for _ in range(n_req)]
    past = [0] * n_req
    rounds = [[4, 2, 6], [1, 1, 1], [1, 3, 1]]

    with torch.no_grad():
        for r, lens in enumerate(rounds):
            seqs = [torch.randint(0, CFG.vocab_size, (n,)) for n in lens]
            tokens = torch.cat(seqs)
            positions = torch.cat([torch.arange(p, p + n) for p, n in zip(past, lens)])
            got = P.packed_forward(model, tokens, positions, lens, packed_caches)
            want = torch.stack([model(s[None], kv_cache=c)[0, -1] for s, c in zip(seqs, single_caches)])
            try:
                compare(f"第 {r + 1} 轮 logits", got, want)
            except AssertionError as e:
                raise AssertionError(f"{e}。检查：位置是不是用了传进来的 positions？"
                                     "每层传给 packed_attention 的是不是第 i 层的缓存？"
                                     "取的是不是每个请求最后一个 token？") from None
            past = [p + n for p, n in zip(past, lens)]


STEPS = [
    ("M2 第 1a 步：packed_attention", check_packed_attention),
    ("M2 第 1b 步：packed_forward", check_packed_forward),
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
