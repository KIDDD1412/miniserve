"""
M1 第 4 步：对比有无 KV Cache 时，生成 N 个新 token 的总耗时。

用法（在项目根目录 D:\\infra\\miniserve 下）：
    python benchmarks/bench_kv_cache.py

计时方法和 bench_no_cache.py 一样：先预热，计时前后都 torch.cuda.synchronize()。
完整跑一遍大约需要 1 分钟（不带缓存生成 1000 个 token 比较慢）。
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve.model import from_pretrained, generate, generate_cached  # noqa: E402

PROMPT_LEN = 16
NEW_TOKENS = [50, 200, 500, 1000]


def timed(fn, *args):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def main():
    assert torch.cuda.is_available(), "这个测速需要 GPU"
    print(f"设备：{torch.cuda.get_device_name(0)}")
    # 无缓存生成时，每一步的张量都比上一步大一点，PyTorch 的缓存池会不断申请新显存、留着旧的。
    # 在 Windows 上超过显存后会悄悄挪到内存里，速度慢几十倍。
    # 限制只用 80% 显存：到上限时 PyTorch 会先释放缓存池里没在用的块，而不是溢出到内存。
    torch.cuda.set_per_process_memory_fraction(0.8)
    model = from_pretrained().to("cuda").eval()
    idx = torch.randint(0, model.config.vocab_size, (1, PROMPT_LEN), device="cuda")

    # 预热两种生成方式
    generate(model, idx, 5)
    generate_cached(model, idx, 5)

    print(f"{'新 token 数':>10}  {'无缓存 (s)':>10}  {'有缓存 (s)':>10}  {'提速':>6}  {'有缓存每 token (ms)':>18}")
    for n in NEW_TOKENS:
        out_slow, t_slow = timed(generate, model, idx, n)
        out_fast, t_fast = timed(generate_cached, model, idx, n)
        same = "" if torch.equal(out_slow, out_fast) else "  [!] 两种方式生成的结果不一致"
        print(f"{n:>10}  {t_slow:>10.2f}  {t_fast:>10.2f}  {t_slow / t_fast:>5.1f}x  {t_fast / n * 1000:>18.1f}{same}")


if __name__ == "__main__":
    main()
