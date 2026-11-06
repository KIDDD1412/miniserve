"""
检查 M2 第 3 步测速脚本（benchmarks/bench_batching.py）里你写的三个函数：
    3a run_sequential、3b run_engine、3c throughput

用一个很小的随机模型，几秒钟就能跑完；全部通过之后，再去跑真正的测速：
    python benchmarks/bench_batching.py

用法（在项目根目录 D:\\infra\\miniserve 下）：
    python tests/check_bench_batching.py
"""
import math
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import bench_batching as B  # noqa: E402
from miniserve import model as M  # noqa: E402
from miniserve.engine import Engine, Request  # noqa: E402

CFG = M.GPTConfig(block_size=64, vocab_size=100, n_layer=3, n_head=4, n_embd=32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 每个请求的 max_new_tokens 各不相同：请求【结束】的先后顺序和 rid 的顺序不一样
WORKLOAD = [
    ([5, 7, 9], 7),
    ([11, 3], 2),
    ([2, 4, 6, 8, 10], 5),
    ([1], 1),
    ([9, 9, 8, 7], 9),
    ([3, 14, 15, 92, 65], 3),
]


class UserCodeError(Exception):
    """你写的函数运行时报错了"""


def make_model():
    torch.manual_seed(0)
    # float64：避免两种算法的微小浮点误差恰好让 argmax 选了不同的词（和 check_engine.py 一样）
    return M.GPT2(CFG).double().to(DEVICE).eval()


def call(fn, *args):
    """调用你写的函数；运行时报错的话，把完整的报错打印出来"""
    try:
        return fn(*args)
    except NotImplementedError:
        raise
    except Exception:
        traceback.print_exc()
        raise UserCodeError(f"{fn.__name__} 运行时报错了（完整报错见上面）") from None


def expected_outputs(model):
    """每个请求单独用 generate_cached 生成的新 token"""
    outs = []
    for prompt, n in WORKLOAD:
        out = M.generate_cached(model, torch.tensor([prompt], device=DEVICE), n)
        outs.append(out[0, len(prompt):].tolist())
    return outs


def expected_steps(model, policy, max_batch_size):
    """直接用 Engine 跑一遍，看应该跑几轮"""
    engine = Engine(model, max_batch_size=max_batch_size, policy=policy)
    for i, (prompt, n) in enumerate(WORKLOAD):
        engine.add_request(Request(rid=i, prompt=prompt, max_new_tokens=n))
    engine.run_all()
    return engine.num_steps


def check_time(t, name):
    assert isinstance(t, (int, float)), f"{name} 返回的耗时应该是一个数（秒），实际是 {type(t).__name__}"
    assert t > 0, f"{name} 返回的耗时是 {t}：应该是【结束时间 - 开始时间】，是不是减反了？"
    assert t > 1e-3, (
        f"{name} 返回的耗时只有 {t * 1000:.3f} ms，太短了：生成的代码有没有放在两次 perf_counter() 之间？"
    )


def check_outputs(outputs, want, name):
    assert isinstance(outputs, list), f"{name} 返回的 outputs 应该是 Python 列表，实际是 {type(outputs).__name__}"
    assert len(outputs) == len(WORKLOAD), (
        f"{name} 返回的 outputs 里应该有 {len(WORKLOAD)} 个请求的结果，实际有 {len(outputs)} 个"
    )
    for i, out in enumerate(outputs):
        hint = "（tensor 要用 .tolist() 转）" if isinstance(out, torch.Tensor) else ""
        assert isinstance(out, list), f"{name}：outputs[{i}] 是 {type(out).__name__}，应该是 Python 列表{hint}"
        assert all(type(x) is int for x in out), (
            f"{name}：outputs[{i}] 里装的不是 Python 整数（是 {type(out[0]).__name__}）：用 .tolist() 转，不要用 list()"
        )
    if outputs != want and sorted(outputs) == sorted(want):
        raise AssertionError(
            f"{name}：每个请求的结果都对，但顺序不对。run_all() 的字典是按请求【结束】的先后放进去的，"
            "要按 rid 0, 1, 2, ... 依次从字典里取"
        )
    for i, (out, (prompt, n)) in enumerate(zip(outputs, WORKLOAD)):
        if len(out) == len(prompt) + n:
            raise AssertionError(f"{name}：outputs[{i}] 把 prompt 也算进去了，只留新生成的 {n} 个 token")
        assert len(out) == n, f"{name}：outputs[{i}] 应该有 {n} 个 token（它的 max_new_tokens），实际有 {len(out)} 个"
    for i, (got, w) in enumerate(zip(outputs, want)):
        assert got == w, f"{name}：第 {i} 个请求生成的是 {got}，应该是 {w}"


def check_sequential():
    model = make_model()
    result = call(B.run_sequential, model, WORKLOAD)
    assert isinstance(result, tuple) and len(result) == 2, "run_sequential 应该返回两个值：return 耗时, outputs"
    t, outputs = result
    check_time(t, "run_sequential")
    check_outputs(outputs, expected_outputs(model), "run_sequential")


def check_engine_run():
    model = make_model()
    want = expected_outputs(model)
    for policy, bs in [("static", 2), ("continuous", 2), ("continuous", 3)]:
        name = f"run_engine（{policy}，max_batch_size={bs}）"
        result = call(B.run_engine, model, WORKLOAD, policy, bs)
        assert isinstance(result, tuple) and len(result) == 3, (
            "run_engine 应该返回三个值：return 耗时, engine.num_steps, outputs"
        )
        t, steps, outputs = result
        check_time(t, name)
        want_steps = expected_steps(model, policy, bs)
        assert steps == want_steps, (
            f"{name} 返回的轮数是 {steps}，应该是 {want_steps}："
            "返回的是不是 engine.num_steps？policy 和 max_batch_size 有没有传给 Engine？"
        )
        check_outputs(outputs, want, name)


def check_throughput():
    got = call(B.throughput, WORKLOAD, 2.0)
    want = sum(n for _, n in WORKLOAD) / 2.0
    assert got is not None, "throughput 没有返回值：是不是忘了 return？"
    assert isinstance(got, (int, float)), f"throughput 应该返回一个数，实际是 {type(got).__name__}"
    if math.isclose(got, sum(len(p) + n for p, n in WORKLOAD) / 2.0):
        raise AssertionError("throughput 把 prompt 的 token 也算进去了：吞吐只算新生成的 token")
    assert math.isclose(got, want), f"6 个请求一共生成 {want * 2:.0f} 个 token、用时 2 秒，吞吐应为 {want}，实际是 {got}"


STEPS = [
    ("M2 第 3a 步：run_sequential", check_sequential),
    ("M2 第 3b 步：run_engine", check_engine_run),
    ("M2 第 3c 步：throughput", check_throughput),
]


def main():
    for label, check in STEPS:
        try:
            check()
        except NotImplementedError:
            print(f"[ TODO ] {label}：还没写")
            break
        except AssertionError as e:
            print(f"[ FAIL ] {label}：{e}")
            break
        except UserCodeError as e:
            print(f"[ERROR ] {label}：{e}")
            break
        else:
            print(f"[ PASS ] {label}")
    else:
        print("\n全部通过，可以运行测速了：python benchmarks/bench_batching.py")


if __name__ == "__main__":
    main()
