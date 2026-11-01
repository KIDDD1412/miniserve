"""
M1 第 1 步：测量没有 KV Cache 时，"一次前向"的耗时随序列长度怎么变化。

用法（在项目根目录 D:\\infra\\miniserve 下）：
    python benchmarks/bench_no_cache.py

运行之前，先在纸上写下你的预测：长度 1024 的一次前向，会比长度 16 慢多少倍？
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve.model import from_pretrained  # noqa: E402

LENGTHS = [16, 64, 128, 256, 512, 1024]
WARMUP = 3
REPEAT = 10


@torch.no_grad()
def time_forward(model, T, device):
    """返回：输入长度为 T 时，一次 model(idx) 的平均耗时（毫秒）"""
    idx = torch.randint(0, model.config.vocab_size, (1, T), device=device)

    # TODO 四步：
    #   ① 预热：先跑 WARMUP 次 model(idx)，结果不要（GPU 第一次调用有额外的初始化开销）
    #   ② 同步并开始计时：torch.cuda.synchronize()，然后 start = time.perf_counter()
    #   ③ 跑 REPEAT 次 model(idx)
    #   ④ 再同步一次，记下 end，返回 (end - start) / REPEAT * 1000
    #   为什么要 synchronize：GPU 是异步执行的，不同步的话，你测到的只是"把任务交给 GPU"的时间
    for i in range(WARMUP):
        model(idx)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(REPEAT):
        model(idx)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / REPEAT * 1000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    model = from_pretrained().to(device).eval()

    print(f"{'序列长度':>8}  {'一次前向 (ms)':>14}  {'相对长度 16':>10}")
    base = None
    for T in LENGTHS:
        ms = time_forward(model, T, device)
        base = base or ms
        print(f"{T:>8}  {ms:>14.2f}  {ms / base:>10.1f}x")


if __name__ == "__main__":
    main()
