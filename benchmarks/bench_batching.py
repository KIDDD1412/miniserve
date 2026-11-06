"""
M2 第 3 步：在 GPU 上用真实 GPT-2 测吞吐，对比三种跑法：
    ① 逐个请求调用 generate_cached（M1 的做法：一次只服务一个请求）
    ② Engine static     （一批请求全部结束，才放下一批进来）
    ③ Engine continuous （running 一有空位，就立刻补新请求）

吞吐（throughput）= 一共生成的新 token 数 / 总耗时，单位 tokens/s（prompt 的 token 不算）。

模拟的请求：所有请求同时到达（离线吞吐测试），prompt 长度和 max_new_tokens 都随机、差别很大。
这正是 static 吃亏的地方：同一批里只要有一个请求要生成很长，其他请求早早结束让出来的位置，就只能空着等它。

运行之前，先在纸上写下你的预测：
    ① batch=16 时，continuous 比逐个跑快几倍？
    ② continuous 比 static 快多少？

先跑检查（几秒钟）：python tests/check_bench_batching.py
全部通过后再测速：  python benchmarks/bench_batching.py      （完整跑一遍约 6 分钟）

计时方法和 M1 一样：先预热，计时前后都 torch.cuda.synchronize()。
测速前：笔记本插上电源，Windows 电源模式选"最佳性能"，测的时候不要用电脑做别的事 ——
这个测速对 CPU 的状态也很敏感（为什么？跑完结果再分析）。
"""
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve.engine import Engine, Request  # noqa: E402
from miniserve.model import from_pretrained, generate_cached  # noqa: E402

NUM_REQUESTS = 32
PROMPT_LEN_RANGE = (16, 256)    # prompt 长度在这个范围里随机（两端都包含）
NEW_TOKENS_RANGE = (16, 512)    # max_new_tokens 在这个范围里随机
BATCH_SIZES = [4, 8, 16]        # Engine 的 max_batch_size 依次试这几个


def make_workload(vocab_size, num_requests, prompt_len_range=PROMPT_LEN_RANGE,
                  new_tokens_range=NEW_TOKENS_RANGE, seed=0):
    """生成一批模拟请求：返回 [(prompt, max_new_tokens), ...]，prompt 是 token id 的列表"""
    rng = random.Random(seed)   # 一个单独的随机数生成器；种子固定 → 每次运行得到同一批请求，结果可复现
    workload = []
    for _ in range(num_requests):
        prompt_len = rng.randint(*prompt_len_range)                      # randint(a, b)：a~b 的整数，两端都包含
        prompt = [rng.randrange(vocab_size) for _ in range(prompt_len)]  # randrange(n)：0~n-1 的整数
        workload.append((prompt, rng.randint(*new_tokens_range)))
    return workload


# ---------------------------------------------------------------------------
# 第 3a 步：逐个请求跑（基线）
# ---------------------------------------------------------------------------
def run_sequential(model, workload):
    """
    一个一个地用 generate_cached 生成：上一个请求生成完，才开始下一个。
    返回 (总耗时秒数, outputs)
        outputs[i]：第 i 个请求【新生成】的 token（Python 列表，不含 prompt），长度 = 它的 max_new_tokens
    """
    device = next(model.parameters()).device
    # TODO ① 同步，然后用 time.perf_counter() 记下开始时间
    #      ② 对 workload 里的每个 (prompt, n)：
    #           idx = torch.tensor([prompt], device=device)        → (1, T)：外面多套一层 []，就是 batch=1
    #           out = generate_cached(model, idx, n)               → (1, T + n)
    #           只留新生成的部分，转成 Python 列表：out[0, len(prompt):].tolist()，加进 outputs
    #      ③ 全部跑完后再同步一次，算出总耗时
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = []
    for (prompt, n) in workload:
        idx = torch.tensor([prompt], device = device)
        out = generate_cached(model, idx, n)
        outputs.append(out[0, len(prompt):].tolist())
    torch.cuda.synchronize()
    end = time.perf_counter()
    return end-start, outputs



# ---------------------------------------------------------------------------
# 第 3b 步：用 Engine 跑
# ---------------------------------------------------------------------------
def run_engine(model, workload, policy, max_batch_size):
    """
    把所有请求一次性加进 Engine（模拟它们同时到达），再 run_all() 一直跑到全部结束。
    返回 (总耗时秒数, engine.num_steps, outputs)
        outputs[i]：第 i 个请求生成的 token 列表（顺序和 run_sequential 的一致）
    """
    # TODO ① 新建 Engine：Engine(model, max_batch_size=max_batch_size, policy=policy)
    #      ② 对 workload 里的第 i 个 (prompt, n)：建 Request(rid=i, prompt=prompt, max_new_tokens=n)，add_request 进去
    #           提示：for i, (prompt, n) in enumerate(workload)
    #      ③ 和 3a 一样计时，计时里只放 engine.run_all()（建 Engine、建请求不算）
    #      ④ run_all() 返回的是 {rid: output} 字典，按 rid 0, 1, 2, ... 的顺序把 output 取出来，排成列表
    #           想一想：字典里的请求，是按什么顺序放进去的？
    engine = Engine(model, max_batch_size = max_batch_size, policy = policy)
    for i, (prompt, n) in enumerate(workload):
        req = Request(rid = i, prompt = prompt, max_new_tokens = n)
        engine.add_request(req)
    torch.cuda.synchronize()
    start = time.perf_counter()
    results = engine.run_all()
    torch.cuda.synchronize()
    end = time.perf_counter()
    outputs = [results[i] for i in range(len(workload))]
    return end-start, engine.num_steps, outputs




# ---------------------------------------------------------------------------
# 第 3c 步：吞吐
# ---------------------------------------------------------------------------
def throughput(workload, seconds):
    """吞吐（tokens/s）= 一共生成的新 token 数 / 总耗时（prompt 的 token 不算）"""
    # TODO 一共生成的新 token 数 = 每个请求的 max_new_tokens 加起来
    return sum(n for _, n in workload)/seconds

def count_same(outputs, ref):
    """有几个请求生成的 token 和基线（逐个 generate_cached）完全一致"""
    return sum(a == b for a, b in zip(outputs, ref))


def main():
    assert torch.cuda.is_available(), "这个测速需要 GPU"
    torch.cuda.set_per_process_memory_fraction(0.8)   # 防止显存溢出到内存（见 bench_kv_cache.py 里的说明）
    model = from_pretrained().to("cuda").eval()
    vocab = model.config.vocab_size

    workload = make_workload(vocab, NUM_REQUESTS)
    prompt_lens = [len(p) for p, _ in workload]
    new_tokens = [n for _, n in workload]
    print(f"\n设备：{torch.cuda.get_device_name(0)}，float32，贪心解码")
    print(f"请求：{NUM_REQUESTS} 个同时到达，prompt 长度 {min(prompt_lens)}~{max(prompt_lens)}，"
          f"max_new_tokens {min(new_tokens)}~{max(new_tokens)}，一共要生成 {sum(new_tokens)} 个新 token")

    # 预热：第一次调用各种 CUDA kernel 时有初始化开销，不能算进正式计时
    warmup = make_workload(vocab, 4, new_tokens_range=(8, 8), seed=1)
    run_sequential(model, warmup)
    run_engine(model, warmup, "continuous", 4)

    # 结果打印成 Markdown 表格，可以直接贴进 README
    print("\n| 跑法 | 耗时 (s) | 轮数 | 平均每轮 token 数 | 吞吐 (tok/s) | 加速比 | 结果和基线一致 | 峰值显存 (MB) |")
    print("|---|---|---|---|---|---|---|---|")

    torch.cuda.reset_peak_memory_stats()               # 清空"峰值显存"的记录，下面量的就是这次运行的峰值
    t_seq, ref = run_sequential(model, workload)
    tput_seq = throughput(workload, t_seq)
    peak = torch.cuda.max_memory_allocated() / 2**20  # 这次运行中，张量最多同时占了多少显存（MB）
    print(f"| 逐个 generate_cached | {t_seq:.1f} | - | 1 | {tput_seq:.0f} | 1.00x | - | {peak:.0f} |")

    for bs in BATCH_SIZES:
        for policy in ("static", "continuous"):
            torch.cuda.reset_peak_memory_stats()
            t, steps, outputs = run_engine(model, workload, policy, bs)
            tput = throughput(workload, t)
            peak = torch.cuda.max_memory_allocated() / 2**20
            same = count_same(outputs, ref)
            print(f"| {policy}（batch={bs}） | {t:.1f} | {steps} | {sum(new_tokens) / steps:.1f} | {tput:.0f} "
                  f"| {tput / tput_seq:.2f}x | {same}/{NUM_REQUESTS} | {peak:.0f} |")


if __name__ == "__main__":
    main()
