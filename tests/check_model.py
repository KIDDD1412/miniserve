"""
逐步检查 miniserve/model.py：把 HF GPT-2 的权重拷进你的模块，用同一个输入对比输出。

用法（在项目根目录 D:\\infra\\miniserve 下）：
    python tests/check_model.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import GPT2LMHeadModel  # noqa: E402

from miniserve import model as M  # noqa: E402

HF_NAME = "openai-community/gpt2"
RTOL = 1e-5  # 相对误差：float32 下数值越大，绝对误差也越大


def copy_linear(dst, src):
    """HF 的 Conv1D 权重是 (in, out)，nn.Linear 是 (out, in)，所以要转置。"""
    with torch.no_grad():
        dst.weight.copy_(src.weight.t())
        dst.bias.copy_(src.bias)


def compare(name, got, want):
    assert isinstance(got, torch.Tensor), f"{name} 应该返回 Tensor，实际是 {type(got)}"
    assert got.shape == want.shape, f"{name} 形状应为 {tuple(want.shape)}，实际是 {tuple(got.shape)}"
    err = ((got - want).abs().max() / want.abs().max().clamp(min=1.0)).item()
    assert err < RTOL, f"{name} 形状对了，但数值和 HF 不一致（最大相对误差 {err:.2e}）"
    return err


def check_attention(hf):
    cfg = M.GPTConfig()
    attn = M.CausalSelfAttention(cfg)
    assert isinstance(attn.c_proj, torch.nn.Linear), "self.c_proj 还没定义成 nn.Linear"
    src = hf.transformer.h[0].attn
    copy_linear(attn.c_attn, src.c_attn)
    copy_linear(attn.c_proj, src.c_proj)

    x = torch.randn(2, 7, cfg.n_embd)
    with torch.no_grad():
        got = attn(x)
        want = src(x)[0]
    err = compare("attention 输出", got, want)

    # 因果性：改动最后一个 token，前面位置的输出不应该变
    x2 = x.clone()
    x2[:, -1] += 1.0
    with torch.no_grad():
        got2 = attn(x2)
    assert torch.allclose(got[:, :-1], got2[:, :-1], atol=1e-6), "前面的 token 看到了未来：检查因果 mask"
    return err


def check_mlp(hf):
    cfg = M.GPTConfig()
    mlp = M.MLP(cfg)
    assert isinstance(mlp.c_fc, torch.nn.Linear), "self.c_fc 还没定义成 nn.Linear"
    assert isinstance(mlp.c_proj, torch.nn.Linear), "self.c_proj 还没定义成 nn.Linear"
    src = hf.transformer.h[0].mlp
    assert tuple(mlp.c_fc.weight.shape) == tuple(src.c_fc.weight.t().shape), (
        f"c_fc 的维度不对：应该是 nn.Linear({cfg.n_embd}, {4 * cfg.n_embd})"
    )
    copy_linear(mlp.c_fc, src.c_fc)
    copy_linear(mlp.c_proj, src.c_proj)

    x = torch.randn(2, 7, cfg.n_embd)
    with torch.no_grad():
        got = mlp(x)
        want = src(x)
    try:
        return compare("MLP 输出", got, want)
    except AssertionError:
        with torch.no_grad():
            exact = mlp.c_proj(torch.nn.functional.gelu(mlp.c_fc(x)))
        if (exact - got).abs().max() < 1e-6:
            raise AssertionError("GELU 用了默认的精确版，GPT-2 需要 approximate=\"tanh\"") from None
        raise


STEPS = [
    ("第 1 步 CausalSelfAttention", check_attention),
    ("第 2 步 MLP", check_mlp),
]


def main():
    torch.manual_seed(0)
    print("加载 HF GPT-2 ...")
    hf = GPT2LMHeadModel.from_pretrained(HF_NAME, attn_implementation="sdpa").eval()
    for label, check in STEPS:
        try:
            err = check(hf)
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
            print(f"[ PASS ] {label}（与 HF 最大相对误差 {err:.1e}）")


if __name__ == "__main__":
    main()
