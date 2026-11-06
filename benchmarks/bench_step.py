"""
M2 第 4 步：验证"CPU 发任务跟不上"的解释 —— decode 一轮 step 的时间都花在哪了？

bench_batching.py 的数据说明：每轮耗时 ≈ 固定部分 + 每个请求 × 3.55 ms。
猜测的原因：packed_attention 对每个请求写了 for 循环，每个请求每层约 11 个小 kernel，
每个 kernel 的耗时主要是 CPU 那边"发任务"的固定开销，GPU 真正在算的时间很少。

这个脚本对 B = 1, 2, 4, ... 个请求（全部处在 decode 阶段，上下文长度相同），测两个数：
    墙钟时间：一轮 step() 从开始到结束一共多久（你写，和 M1 的计时一样）
    GPU 时间：这一轮里 GPU 真正在执行 kernel 的时间加起来（用 PyTorch profiler 测，已写好）
如果猜测是对的：墙钟时间随 B 线性增长，而 GPU 时间只占其中一小部分。

运行之前，先写下你的预测：B=16 时，GPU 真正在算的时间占一轮的百分之多少？

先跑检查（几秒钟）："D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" tests/check_bench_step.py
再测速（约 1 分钟）："D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" benchmarks/bench_step.py
测之前和上次一样：插电、最佳性能模式、测的时候别用电脑。
运行时会打印一些 "USDT: ... profiler_start / profiler_stop" 的行，是 profiler 在 Windows 上的调试信息，不用管。
"""
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve.engine import Engine, Request  # noqa: E402
from miniserve.model import from_pretrained  # noqa: E402

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
CONTEXT_LEN = 128     # 每个请求的 prompt 长度（decode 时的上下文长度从这里开始）
REPS = 30             # 每个 B 连续测几轮取平均


def make_engine(model, num_requests, context_len=CONTEXT_LEN, seed=0):
    """建一个 Engine，放进 num_requests 个请求，先跑完 prefill 和几轮 decode 预热：之后每一轮都是纯 decode"""
    g = torch.Generator().manual_seed(seed)
    engine = Engine(model, max_batch_size=num_requests, policy="continuous")
    for i in range(num_requests):
        prompt = torch.randint(0, model.config.vocab_size, (context_len,), generator=g).tolist()
        engine.add_request(Request(rid=i, prompt=prompt, max_new_tokens=10_000))   # 设得很大：测的时候不会有请求结束
    engine.step()           # 第 1 轮：所有请求一起 prefill
    for _ in range(3):      # 预热几轮 decode
        engine.step()
    return engine


# ---------------------------------------------------------------------------
# 第 4a 步：一轮 step 的墙钟时间
# ---------------------------------------------------------------------------
def time_steps(engine, reps):
    """连续跑 reps 轮 engine.step()，返回平均每轮的墙钟时间，单位【毫秒】"""
    # TODO 和 M1 的 bench_no_cache.py 一样：
    #   ① 同步，记下开始时间
    #   ② 调用 reps 次 engine.step()
    #   ③ 再同步，记下结束时间
    #   ④ 返回平均每轮多少毫秒
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        engine.step()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) *1000 / reps


def gpu_time_per_step(engine, reps):
    """用 profiler 记下这 reps 轮里 GPU 执行的每一个 kernel，把它们的耗时加起来，返回平均每轮多少毫秒"""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:   # with 块里 GPU 上发生的事都会被记下来
        for _ in range(reps):
            engine.step()
        torch.cuda.synchronize()
    # key_averages()：按 kernel 名字汇总；device_time_total：这种 kernel 在 GPU 上一共执行了多少微秒
    total_us = sum(e.device_time_total for e in prof.key_averages() if e.device_type.name == "CUDA")
    return total_us / reps / 1000


def main():
    assert torch.cuda.is_available(), "这个测速需要 GPU"
    torch.cuda.set_per_process_memory_fraction(0.8)
    model = from_pretrained().to("cuda").eval()

    print(f"\n设备：{torch.cuda.get_device_name(0)}，float32；所有请求都在 decode 阶段，上下文约 {CONTEXT_LEN}，"
          f"每个 B 连续 {REPS} 轮取平均")
    print("\n| 请求数 B | 每轮墙钟 (ms) | 每 token (ms) | 每轮 GPU 计算 (ms) | GPU 忙碌占比 |")
    print("|---|---|---|---|---|")
    for b in BATCH_SIZES:
        engine = make_engine(model, b)
        wall = time_steps(engine, REPS)
        gpu = gpu_time_per_step(engine, 5)
        print(f"| {b} | {wall:.1f} | {wall / b:.2f} | {gpu:.1f} | {gpu / wall * 100:.0f}% |")


if __name__ == "__main__":
    main()
