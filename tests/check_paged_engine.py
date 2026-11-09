"""
检查 M3 第 4 步：PagedEngine（miniserve/paged_engine.py）

每个请求的结果都必须和"单独用 M1 的 generate_cached 生成"逐 token 一致；
另外检查：块用完会还、池子很小时请求会等块（结果照样对）、轮数和 M2 Engine 一样。

用法（在项目根目录 D:\\infra\\miniserve 下）：
    "D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" tests/check_paged_engine.py
"""
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve import model as M  # noqa: E402
from miniserve.engine import Engine, Request  # noqa: E402
from miniserve.paged import blocks_needed  # noqa: E402
from miniserve.paged_engine import PagedEngine  # noqa: E402

CFG = M.GPTConfig(block_size=64, vocab_size=100, n_layer=3, n_head=4, n_embd=32)
SPECS = [([3, 1, 4], 3), ([1, 5], 8), ([9, 2, 6, 5], 2), ([3, 5, 8, 9, 7], 6), ([2], 4),
         ([7] * 9, 5), ([4, 4, 2, 1, 9, 9, 3, 3, 8, 1, 2, 6], 7), ([6, 6], 1)]


def make_model():
    torch.manual_seed(0)
    return M.GPT2(CFG).double().eval()      # float64：避免浮点误差让 argmax 选了不同的词


def expected(model, prompt, n_new):
    out = M.generate_cached(model, torch.tensor([prompt]), n_new)
    return out[0, len(prompt):].tolist()


def make_requests():
    return [Request(rid=i, prompt=list(p), max_new_tokens=n) for i, (p, n) in enumerate(SPECS)]


def run(engine):
    for r in make_requests():
        engine.add_request(r)
    return engine.run_all()


def check_outputs(model, got, label):
    for i, (p, n) in enumerate(SPECS):
        want = expected(model, p, n)
        assert got.get(i) == want, f"{label}：请求 {i} 生成的是 {got.get(i)}，应该是 {want}"


def check_all_freed(engine, num_blocks, label):
    a = engine.pool.allocator
    assert a.num_free == num_blocks and not a.used, f"{label}：全部结束后还有 {len(a.used)} 个块没还"
    assert not engine.block_tables, f"{label}：全部结束后 block_tables 里还有 {list(engine.block_tables)}"


def check_big_pool():
    model = make_model()
    for policy in ("static", "continuous"):
        engine = PagedEngine(model, max_batch_size=3, num_blocks=64, block_size=4, policy=policy)
        got = run(engine)
        check_outputs(model, got, f"{policy}、池子够大")
        check_all_freed(engine, 64, policy)

        m2 = Engine(model, max_batch_size=3, policy=policy)
        run(m2)
        assert engine.num_steps == m2.num_steps, (
            f"{policy}：PagedEngine 跑了 {engine.num_steps} 轮，M2 Engine 跑了 {m2.num_steps} 轮，调度应该一样"
        )


def check_tight_pool():
    """池子只够同时放一两个请求：后面的请求要等前面的还块，块会被反复使用"""
    model = make_model()
    bs = 4
    worst = max(blocks_needed(len(p) + n, bs) for p, n in SPECS)
    num_blocks = worst + 2
    engine = PagedEngine(model, max_batch_size=4, num_blocks=num_blocks, block_size=bs)

    waited = []
    original = engine._schedule

    def recording_schedule():
        original()
        if engine.waiting and len(engine.running) < engine.max_batch_size:
            waited.append(engine.num_steps)          # 有空位、有请求在等，却没放进来：说明在等块

    engine._schedule = recording_schedule
    got = run(engine)
    check_outputs(model, got, "池子很小")
    check_all_freed(engine, num_blocks, "池子很小")
    assert waited, "池子很小时应该出现请求等块的情况（测试本身没测到点子上）"
    assert engine.peak_blocks_used <= num_blocks
    assert 0 < engine.kv_utilization <= 1


def check_too_small():
    model = make_model()
    engine = PagedEngine(model, max_batch_size=2, num_blocks=2, block_size=4)
    engine.add_request(Request(rid=0, prompt=[1] * 12, max_new_tokens=2))     # 要 4 块，池子只有 2 块
    try:
        engine.run_all()
    except RuntimeError as e:
        assert "缓存池太小" in str(e), f"报错信息不对：{e}"
    else:
        raise AssertionError("池子装不下 prompt 时应该报错，而不是一直空转")


STEPS = [
    ("M3 第 4a 步：池子够大时结果和 generate_cached 一致、块全部还回、轮数和 M2 一样", check_big_pool),
    ("M3 第 4b 步：池子很小时请求等块、块反复使用，结果照样一致", check_tight_pool),
    ("M3 第 4c 步：池子装不下 prompt 时报错", check_too_small),
]


def main():
    for label, check in STEPS:
        try:
            check()
        except AssertionError as e:
            print(f"[ FAIL ] {label}：{e}")
            break
        except Exception as e:
            traceback.print_exc()
            print(f"[ERROR ] {label}：{type(e).__name__}: {e}")
            break
        else:
            print(f"[ PASS ] {label}")


if __name__ == "__main__":
    main()
