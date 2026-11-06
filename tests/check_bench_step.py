"""
检查 M2 第 4 步测速脚本（benchmarks/bench_step.py）里你写的 time_steps。
用一个很小的随机模型，几秒钟就能跑完；通过之后再去跑真正的测速。

用法（在项目根目录 D:\\infra\\miniserve 下）：
    "D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" tests/check_bench_step.py
"""
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import bench_step as B  # noqa: E402
from miniserve import model as M  # noqa: E402

CFG = M.GPTConfig(block_size=128, vocab_size=100, n_layer=3, n_head=4, n_embd=32)
REPS = 20


class UserCodeError(Exception):
    """你写的函数运行时报错了"""


def make_engine():
    torch.manual_seed(0)
    model = M.GPT2(CFG).to("cuda").eval()
    return B.make_engine(model, 4, context_len=8)


def reference_ms(engine, reps):
    """自己量一下：平均每轮多少毫秒"""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        engine.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1000


def check_time_steps():
    engine = make_engine()
    events = []                      # 按发生顺序记下："step" 或 "clock"（调用了 time.perf_counter()）
    original = engine.step

    def counting_step():
        events.append("step")
        return original()

    class LoggedTime:                # 代替 bench_step 里的 time 模块：perf_counter 被调用时记一笔
        def __getattr__(self, name):
            return getattr(time, name)

        def perf_counter(self):
            events.append("clock")
            return time.perf_counter()

    engine.step = counting_step
    B.time = LoggedTime()
    try:
        got = B.time_steps(engine, REPS)
    except NotImplementedError:
        raise
    except Exception:
        traceback.print_exc()
        raise UserCodeError("time_steps 运行时报错了（完整报错见上面）") from None
    finally:
        B.time = time

    steps = events.count("step")
    assert steps == REPS, f"time_steps(engine, {REPS}) 应该调用 {REPS} 次 engine.step()，实际调用了 {steps} 次"
    if "clock" in events:
        first, last = events.index("step"), len(events) - 1 - events[::-1].index("step")
        assert "clock" in events[:first] and "clock" in events[last:], (
            "step() 没有全部放在两次 perf_counter() 之间：开始计时要在第一轮之前，结束计时要在最后一轮之后"
        )
    assert got is not None, "time_steps 没有返回值：是不是忘了 return？"
    assert isinstance(got, (int, float)), f"time_steps 应该返回一个数（毫秒），实际是 {type(got).__name__}"
    assert got > 0, f"time_steps 返回的是 {got}：应该是【结束时间 - 开始时间】，是不是减反了？"

    engine.step = original
    want = min(reference_ms(engine, REPS) for _ in range(3))   # 量 3 次取最小，减少波动
    ratio = got / want
    if ratio < 0.01:
        raise AssertionError(
            f"返回 {got:.5f}，而一轮大约 {want:.2f} ms：单位是不是还是秒？要乘 1000 变成毫秒"
            if 0.0003 < ratio < 0.003 else
            f"返回 {got:.5f} ms，太短了，而一轮大约 {want:.2f} ms：step() 有没有放在两次 perf_counter() 之间？"
        )
    if ratio > REPS / 3:
        raise AssertionError(
            f"返回 {got:.2f} ms，而一轮大约 {want:.2f} ms：是不是返回了 {REPS} 轮的总时间？要除以 reps 得到平均每轮"
        )
    assert 1 / 3 < ratio < 3, f"返回 {got:.2f} ms，而一轮大约 {want:.2f} ms，差得太多，检查一下计时和单位"


STEPS = [("M2 第 4a 步：time_steps", check_time_steps)]


def main():
    if not torch.cuda.is_available():
        print("这个检查需要 GPU：确认运行的是 miniserve 环境的 python.exe")
        return
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
        print('\n通过了，可以运行测速了："D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" benchmarks/bench_step.py')


if __name__ == "__main__":
    main()
