"""
M3 第 6 步：Paged KV Cache 的效果 —— 和 M2 对比

三张表（都打印成 Markdown，可以直接贴进 README）：
    表 1 吞吐：和 bench_batching.py 同一批 32 个请求（seed=0）
             逐个 generate_cached / M2 continuous / M3 paged continuous，batch = 4, 8, 16, 32
    表 2 decode 一轮的时间：和 bench_step.py 一样（所有请求都在 decode，上下文约 128）
             M2 和 M3 各自的墙钟时间、GPU 计算时间、GPU 忙碌占比，B = 1 ~ 32
    表 3 KV 缓存的显存利用率（表 1 里 M3 batch=16 那次）：
             按块分配 vs 每个请求预留最大长度 vs 每个请求预留 prompt + max_new_tokens

运行之前，先写下你的预测：
    ① batch=16 时，M3 的吞吐是 M2 的几倍？
    ② B=16 时，M3 一轮 decode 的 GPU 忙碌占比是多少？（M2 是 13%）

先跑检查（几秒钟）：
    "D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" tests/check_paged.py
    "D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" tests/check_paged_engine.py
再测速（约 8~10 分钟）：
    "D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" benchmarks/bench_paged.py
测之前和以前一样：插电、最佳性能模式、测的时候别用电脑。
运行时会打印一些 "USDT: ... profiler_start / profiler_stop" 的行，是 profiler 的调试信息，不用管。

关于表 1 的"峰值显存"：M3 的缓存池是开工前按【最坏情况】一次开好的（每个请求都按 prompt + max_new_tokens
的最大值算），所以这一列 M3 可能反而比 M2 高。分页在显存上的好处看表 3：同样的请求，按块分配几乎不浪费，
而"每个请求预留一整段"会浪费一大半 —— 真实系统（vLLM）是把剩余显存全部开成池子，再靠表 3 这样的利用率塞进更多请求。
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_batching import NUM_REQUESTS, count_same, make_workload, run_engine, run_sequential, throughput  # noqa: E402
from bench_step import CONTEXT_LEN, REPS, gpu_time_per_step, make_engine, time_steps  # noqa: E402

from miniserve.engine import Request  # noqa: E402
from miniserve.model import from_pretrained  # noqa: E402
from miniserve.paged import blocks_needed  # noqa: E402
from miniserve.paged_engine import PagedEngine  # noqa: E402

BLOCK_SIZE = 16                     # 每块装 16 个 token（vLLM 的默认值）
THROUGHPUT_BATCH_SIZES = [4, 8, 16, 32]
STEP_BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_paged(model, workload, max_batch_size: int, block_size: int = BLOCK_SIZE):
    """和 bench_batching.run_engine 一样，只是换成 PagedEngine；池子按最坏情况开：每个请求都可能用到最长的那么多块"""
    worst = max(blocks_needed(len(p) + n, block_size) for p, n in workload)
    engine = PagedEngine(model, max_batch_size=max_batch_size, num_blocks=max_batch_size * worst,
                         block_size=block_size, policy="continuous")
    for i, (prompt, n) in enumerate(workload):
        engine.add_request(Request(rid=i, prompt=prompt, max_new_tokens=n))
    sync()
    start = time.perf_counter()
    results = engine.run_all()
    sync()
    seconds = time.perf_counter() - start
    return seconds, engine.num_steps, [results[i] for i in range(len(workload))], engine


def make_paged_engine(model, num_requests: int, context_len: int = CONTEXT_LEN, seed: int = 0) -> PagedEngine:
    """和 bench_step.make_engine 一样（同样的 prompt），只是换成 PagedEngine：prefill + 3 轮 decode 预热后，每轮都是纯 decode"""
    g = torch.Generator().manual_seed(seed)
    per_request = blocks_needed(context_len + 64, BLOCK_SIZE)       # 测量期间每个请求最多再长 40 来个 token，留够余量
    engine = PagedEngine(model, max_batch_size=num_requests, num_blocks=num_requests * per_request,
                         block_size=BLOCK_SIZE, policy="continuous")
    for i in range(num_requests):
        prompt = torch.randint(0, model.config.vocab_size, (context_len,), generator=g).tolist()
        engine.add_request(Request(rid=i, prompt=prompt, max_new_tokens=10_000))
    engine.step()
    for _ in range(3):
        engine.step()
    return engine


def main():
    assert torch.cuda.is_available(), "这个测速需要 GPU"
    torch.cuda.set_per_process_memory_fraction(0.8)
    model = from_pretrained().to("cuda").eval()
    vocab = model.config.vocab_size
    kv_bytes_per_token = 2 * model.config.n_layer * model.config.n_embd * 4     # k、v × 层数 × C × float32 4 字节

    workload = make_workload(vocab, NUM_REQUESTS)
    new_tokens = [n for _, n in workload]
    print(f"\n设备：{torch.cuda.get_device_name(0)}，float32，贪心解码，KV 块大小 {BLOCK_SIZE}")
    print(f"请求：{NUM_REQUESTS} 个同时到达（和 bench_batching 同一批），一共要生成 {sum(new_tokens)} 个新 token")

    # 预热
    warmup = make_workload(vocab, 4, new_tokens_range=(8, 8), seed=1)
    run_sequential(model, warmup)
    run_engine(model, warmup, "continuous", 4)
    run_paged(model, warmup, 4)

    # ---------------- 表 1：吞吐 ----------------
    print("\n### 表 1：吞吐\n")
    print("| 跑法 | 耗时 (s) | 轮数 | 吞吐 (tok/s) | 相对逐个 | 相对 M2（同 batch） | 结果和基线一致 | 峰值显存 (MB) |")
    print("|---|---|---|---|---|---|---|---|")
    torch.cuda.reset_peak_memory_stats()
    t_seq, ref = run_sequential(model, workload)
    tput_seq = throughput(workload, t_seq)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"| 逐个 generate_cached | {t_seq:.1f} | - | {tput_seq:.0f} | 1.00x | - | - | {peak:.0f} |")

    kv_stats = None
    for bs in THROUGHPUT_BATCH_SIZES:
        torch.cuda.reset_peak_memory_stats()
        t2, steps2, out2 = run_engine(model, workload, "continuous", bs)
        tput2 = throughput(workload, t2)
        peak2 = torch.cuda.max_memory_allocated() / 2**20
        print(f"| M2 continuous（batch={bs}） | {t2:.1f} | {steps2} | {tput2:.0f} | {tput2 / tput_seq:.2f}x | 1.00x "
              f"| {count_same(out2, ref)}/{NUM_REQUESTS} | {peak2:.0f} |")

        torch.cuda.reset_peak_memory_stats()
        t3, steps3, out3, engine = run_paged(model, workload, bs)
        tput3 = throughput(workload, t3)
        peak3 = torch.cuda.max_memory_allocated() / 2**20
        print(f"| **M3 paged（batch={bs}）** | {t3:.1f} | {steps3} | {tput3:.0f} | {tput3 / tput_seq:.2f}x "
              f"| {tput3 / tput2:.2f}x | {count_same(out3, ref)}/{NUM_REQUESTS} | {peak3:.0f} |")
        if bs == 16:
            # 表 3 只要几个数字：现在就记下来，不要把整个 engine 留着。
            # engine 里有缓存池（batch=16 时约 774 MB），留着的话后面 batch=32 那两行的峰值显存会把它也算进去
            kv_stats = dict(kv=engine.kv_utilization, exact=engine.reserve_exact_utilization,
                            max=engine.reserve_max_utilization, peak_blocks=engine.peak_blocks_used,
                            num_blocks=engine.pool.allocator.num_blocks)
        del engine

    # ---------------- 表 2：decode 一轮的时间 ----------------
    print(f"\n### 表 2：decode 一轮的时间（所有请求都在 decode，上下文约 {CONTEXT_LEN}，每个 B 连续 {REPS} 轮取平均）\n")
    print("| 请求数 B | M2 墙钟 (ms) | M2 GPU (ms) | M2 GPU 忙碌 | M3 墙钟 (ms) | M3 GPU (ms) | M3 GPU 忙碌 | 墙钟加速 |")
    print("|---|---|---|---|---|---|---|---|")
    for b in STEP_BATCH_SIZES:
        e2 = make_engine(model, b)
        w2, g2 = time_steps(e2, REPS), gpu_time_per_step(e2, 5)
        del e2
        e3 = make_paged_engine(model, b)
        w3, g3 = time_steps(e3, REPS), gpu_time_per_step(e3, 5)
        del e3
        print(f"| {b} | {w2:.1f} | {g2:.1f} | {g2 / w2 * 100:.0f}% | {w3:.1f} | {g3:.1f} | {g3 / w3 * 100:.0f}% "
              f"| {w2 / w3:.2f}x |")

    # ---------------- 表 3：KV 缓存显存利用率 ----------------
    s = kv_stats
    print("\n### 表 3：KV 缓存的显存利用率（表 1 里 M3 batch=16 那次，每轮平均）\n")
    print("| 分配方式 | 利用率（真正存着 token 的位置 / 分出去的位置） |")
    print("|---|---|")
    print(f"| 按块分配（块大小 {BLOCK_SIZE}，M3） | {s['kv'] * 100:.1f}% |")
    print(f"| 每个请求预留 prompt + max_new_tokens（要事先知道生成多长） | {s['exact'] * 100:.1f}% |")
    print(f"| 每个请求预留模型最大长度 {model.config.block_size} | {s['max'] * 100:.1f}% |")
    print(f"\nM3 batch=16 时同时占用的块最多 {s['peak_blocks']} 个，"
          f"约 {s['peak_blocks'] * BLOCK_SIZE * kv_bytes_per_token / 2**20:.0f} MB 的 KV 缓存"
          f"（池子一共开了 {s['num_blocks']} 块，"
          f"约 {s['num_blocks'] * BLOCK_SIZE * kv_bytes_per_token / 2**20:.0f} MB）")


if __name__ == "__main__":
    main()
