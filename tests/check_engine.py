"""
检查 M2 第 2 步的推理引擎：多个请求经过调度、一起生成，每个请求的结果都必须和
"单独用 M1 的 generate_cached 生成"逐 token 一致。

用法（在项目根目录 D:\\infra\\miniserve 下）：
    python tests/check_engine.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve import model as M  # noqa: E402
from miniserve.engine import Engine, Request  # noqa: E402

CFG = M.GPTConfig(block_size=64, vocab_size=100, n_layer=3, n_head=4, n_embd=32)


def make_model():
    torch.manual_seed(0)
    # float64：避免两种算法的微小浮点误差恰好让 argmax 选了不同的词
    return M.GPT2(CFG).double().eval()


def expected(model, prompt, n_new):
    out = M.generate_cached(model, torch.tensor([prompt]), n_new)
    return out[0, len(prompt):].tolist()


def compare_outputs(model, reqs, got):
    for r in reqs:
        want = expected(model, r.prompt, r.max_new_tokens)
        assert r.rid in got, f"请求 {r.rid} 没有出现在结果里：结束的请求有没有从 step() 返回？"
        assert got[r.rid] == want, f"请求 {r.rid} 生成的是 {got[r.rid]}，应该是 {want}"


def check_run_batch():
    model = make_model()
    engine = Engine(model, max_batch_size=3)
    prompts = [[5, 7, 9], [11], [2, 4, 6, 8, 10]]
    reqs = [Request(rid=i, prompt=p, max_new_tokens=4) for i, p in enumerate(prompts)]
    for r in reqs:                      # 跳过调度，直接放进 running
        r.kv_cache = M.new_kv_cache(CFG)
        engine.running.append(r)

    try:
        engine._run_batch()
    except Exception:
        if all(r.num_computed == 0 and not r.output for r in reqs):
            raise NotImplementedError from None
        raise
    assert all(len(r.output) == 1 for r in reqs), "第 1 轮（prefill）之后每个请求应该有 1 个新 token：检查 ④"
    assert [r.num_computed for r in reqs] == [len(p) for p in prompts], (
        f"第 1 轮之后 num_computed 应为 {[len(p) for p in prompts]}，实际是 {[r.num_computed for r in reqs]}"
    )
    for _ in range(3):
        engine._run_batch()
    assert [r.num_computed for r in reqs] == [len(p) + 3 for p in prompts], (
        "decode 每一轮应该只送 1 个 token：num_computed 每轮加 1"
    )
    compare_outputs(model, reqs, {r.rid: r.output for r in reqs})


def run_engine(policy, record):
    model = make_model()
    engine = Engine(model, max_batch_size=2, policy=policy)
    specs = [([3, 1, 4], 3), ([1, 5], 8), ([9, 2, 6, 5], 2), ([3, 5, 8, 9, 7], 6), ([2], 4)]
    reqs = [Request(rid=i, prompt=p, max_new_tokens=n) for i, (p, n) in enumerate(specs)]
    for r in reqs:
        engine.add_request(r)

    original = engine._run_batch

    def recording_run_batch():
        record.append([r.rid for r in engine.running])
        original()

    engine._run_batch = recording_run_batch
    # 和 engine.run_all() 一样，但加了上限，防止没写完时死循环
    got = {}
    for _ in range(200):
        if not engine.has_unfinished():
            break
        for req in engine.step():
            got[req.rid] = req.output
    else:
        if not record:
            raise NotImplementedError
        raise AssertionError(f"[{policy}] 跑了 200 轮还没结束：生成够的请求有没有从 running 里移走？")
    return model, reqs, got, engine


def check_engine():
    for policy in ("static", "continuous"):
        record = []
        try:
            model, reqs, got, engine = run_engine(policy, record)
        except Exception:
            if not record:
                raise NotImplementedError from None
            raise
        assert record, f"[{policy}] 一次 _run_batch 都没跑：_schedule 有没有把请求放进 running？"
        assert max(len(b) for b in record) <= 2, (
            f"[{policy}] running 里同时有 {max(len(b) for b in record)} 个请求，超过了 max_batch_size=2"
        )
        compare_outputs(model, reqs, got)
        if policy == "static":
            for prev, cur in zip(record, record[1:]):
                new = set(cur) - set(prev)
                assert not new or not (set(prev) & set(cur)), (
                    f"[static] 上一批 {prev} 还没全部结束，就放进了新请求 {sorted(new)}"
                )
            static_steps = engine.num_steps
        else:
            assert engine.num_steps < static_steps, (
                f"[continuous] 用了 {engine.num_steps} 轮，没有比 static 的 {static_steps} 轮少："
                "有空位时是不是立刻补了新请求？"
            )
            print(f"         static 用了 {static_steps} 轮，continuous 用了 {engine.num_steps} 轮")


STEPS = [
    ("M2 第 2a 步：_run_batch", check_run_batch),
    ("M2 第 2b 步：调度 + 收尾（static / continuous）", check_engine),
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
        except Exception as e:  # 形状不匹配等运行时错误
            print(f"[ERROR ] {label}：{type(e).__name__}: {e}")
            break
        else:
            print(f"[ PASS ] {label}")


if __name__ == "__main__":
    main()
