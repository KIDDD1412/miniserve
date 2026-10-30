"""
PyTorch tensor shape 热身练习（为 M0 手写 GPT-2 做准备）

用法：
    pip install torch
    python tensor_exercises.py        # 跑全部练习
    python tensor_exercises.py 3      # 只跑第 3 题

规则：只修改标着 TODO 的函数体，不要改下面的测试代码。
这 5 题会一步步拼出 GPT-2 里的「因果多头自注意力」：
5 题全过，就等于你已经写完了 attention 的核心。第 6 题是 M1 的预习。

符号约定（和 nanoGPT 一致）：
    B  = batch size，一次处理几条句子
    T  = 序列长度，每条句子有几个 token
    C  = 隐藏维度 n_embd
    nh = 注意力头数 n_head
    hs = 每个头的维度 head_size = C // nh

做题建议：卡住时先在函数里 print(x.shape)，把每一步的形状写在纸上。
"""
import math
import sys
from math import inf

import torch
import torch.nn.functional as F
from mpmath.math2 import sqrt


# ---------------------------------------------------------------------------
# 练习 1：一次矩阵乘法算出 Q、K、V
# ---------------------------------------------------------------------------
def ex1_qkv(x, w_qkv):
    """
    x:     (B, T, C)   输入
    w_qkv: (C, 3C)     把 Q/K/V 三个投影矩阵横向拼在一起的权重
                       （前 C 列是 W_q，中间 C 列是 W_k，最后 C 列是 W_v）
    返回 q, k, v，形状都是 (B, T, C)

    提示：(B, T, C) @ (C, 3C) 得到什么形状？前面的 B、T 维度会怎样？
          然后查一下 tensor.split 的用法。
    """
    # TODO
    qkv = x@w_qkv
    q,k,v = qkv.split(C,dim=-1)
    return q,k,v


# ---------------------------------------------------------------------------
# 练习 2：拆成多个头
# ---------------------------------------------------------------------------
def ex2_split_heads(t, n_head):
    """
    t: (B, T, C)  ->  返回 (B, nh, T, hs)

    含义：最后一维 C 被切成 nh 段，每段长 hs，第 h 段属于第 h 个头。
    提示：需要两步（一个改形状，一个交换维度）。
    思考：为什么直接 t.view(B, nh, T, hs) 是错的？（测试会抓住这个坑）
    """
    # TODO
    hs = C//n_head
    step1 = t.view(B,T,n_head,hs)
    return torch.transpose(step1,1,2)


# ---------------------------------------------------------------------------
# 练习 3：注意力分数
# ---------------------------------------------------------------------------
def ex3_scores(q, k):
    """
    q, k: (B, nh, T, hs)
    返回缩放后的注意力分数 (B, nh, T, T)，即  q @ k^T / sqrt(hs)

    提示：k 的哪两个维度需要交换？scores[b, h, i, j] 表示什么？
    """
    # TODO
    return q @ torch.transpose(k,-1,-2)/sqrt(HS)


# ---------------------------------------------------------------------------
# 练习 4：因果 mask + softmax
# ---------------------------------------------------------------------------
def ex4_causal_softmax(scores):
    """
    scores: (B, nh, T, T)
    把"看到未来"的位置（列号 j > 行号 i）填成 -inf，再沿最后一维做 softmax。
    返回 (B, nh, T, T)：每行之和为 1，右上三角全为 0。

    提示：torch.tril、torch.ones、masked_fill。
    torch.tril 是 PyTorch 中用于提取张量下三角部分（Lower Triangular）的函数，会将主对角线以上的元素全部置为 0。
    思考：为什么填 -inf 而不是填 0？
    """
    # TODO
    mask = torch.tril(torch.ones(T,T))
    scores.masked_fill_(mask == 0,float('-inf'))
    return torch.softmax(scores,dim=-1)


# ---------------------------------------------------------------------------
# 练习 5：完整的因果多头自注意力
# ---------------------------------------------------------------------------
def ex5_attention(x, w_qkv, n_head):
    """
    把前 4 题串起来（不含最后的输出投影 c_proj）：
        x (B, T, C) -> q, k, v -> 拆头 -> 分数 -> mask + softmax -> 乘 v -> 合并头
    返回 (B, T, C)

    提示：合并头是练习 2 的逆操作。
    思考：transpose 之后直接 .view 可能会报错，为什么？怎么解决？
    """
    # TODO
    B,T,C = x.shape
    q,k,v = ex1_qkv(x,w_qkv)
    q = ex2_split_heads(q, n_head)
    k = ex2_split_heads(k, n_head)
    v = ex2_split_heads(v, n_head)
    scores = ex3_scores(q, k)
    ms_score = ex4_causal_softmax(scores)
    y = ms_score @ v
    y.transpose(1,2)
    y_c = y.contiguous()
    return y.transpose(1,2).contiguous().view(B,T,C)


# ---------------------------------------------------------------------------
# 练习 6（附加题，预习 M1）：带 KV Cache 的一步解码
# ---------------------------------------------------------------------------
def ex6_decode_step(q_new, k_new, v_new, k_cache, v_cache):
    """
    解码阶段每一步只来 1 个新 token。
    q_new, k_new, v_new: (B, nh, 1, hs)        新 token 的 q / k / v
    k_cache, v_cache:    (B, nh, T_past, hs)   之前所有 token 的 k / v

    返回 (out, k_cache, v_cache)：
        out:              (B, nh, 1, hs)          新 token 的注意力输出
        k_cache, v_cache: (B, nh, T_past + 1, hs) 拼上新 token 之后的缓存

    提示：torch.cat，注意沿哪一维拼。
    思考：这里为什么不需要因果 mask？
    """
    # TODO
    k_cache = torch.cat([k_cache,k_new], dim = 2)
    v_cache = torch.cat([v_cache,v_new], dim = 2)
    scores = ex3_scores(q_new, k_cache)
    att = torch.softmax(scores, dim=-1)
    out = att @ v_cache
    return out, k_cache, v_cache


# ===========================================================================
# 以下是测试代码，不需要修改
# ===========================================================================
B, T, C, NH = 2, 5, 12, 3
HS = C // NH
DT = torch.float64


def _rand(*shape):
    return torch.randn(*shape, dtype=DT)


def _expect_shape(name, t, shape):
    assert isinstance(t, torch.Tensor), f"{name} 应该是 Tensor，实际是 {type(t)}"
    assert tuple(t.shape) == tuple(shape), (
        f"{name} 形状应为 {tuple(shape)}，实际是 {tuple(t.shape)}"
    )


def _expect_close(name, got, want):
    _expect_shape(name, got, want.shape)
    err = (got - want).abs().max().item()
    assert err < 1e-6, f"{name} 形状对了，但数值不对（最大误差 {err:.2e}）"


def _per_head_reference(x, w_qkv, n_head):
    """逐个头单独计算的参考实现，用来检验你的批量写法。"""
    c = x.shape[-1]
    hs = c // n_head
    outs = []
    for h in range(n_head):
        cols = slice(h * hs, (h + 1) * hs)
        qh = x @ w_qkv[:, :c][:, cols]
        kh = x @ w_qkv[:, c:2 * c][:, cols]
        vh = x @ w_qkv[:, 2 * c:][:, cols]
        outs.append(F.scaled_dot_product_attention(qh, kh, vh, is_causal=True))
    return torch.cat(outs, dim=-1)


def check_ex1():
    x, w = _rand(B, T, C), _rand(C, 3 * C)
    out = ex1_qkv(x, w)
    assert isinstance(out, (tuple, list)) and len(out) == 3, "应该返回 (q, k, v) 三个张量"
    for name, got, lo in zip("qkv", out, (0, C, 2 * C)):
        _expect_shape(name, got, (B, T, C))
        _expect_close(name, got, torch.einsum("btc,cd->btd", x, w[:, lo:lo + C]))


def check_ex2():
    t = _rand(B, T, C)
    out = ex2_split_heads(t, NH)
    _expect_shape("输出", out, (B, NH, T, HS))
    for h in range(NH):
        want = t[:, :, h * HS:(h + 1) * HS]
        got = out[:, h]
        err = (got - want).abs().max().item()
        assert err < 1e-12, (
            f"第 {h} 个头的内容不对：out[:, {h}] 应该等于 t[:, :, {h * HS}:{(h + 1) * HS}]。"
            "提示：直接 view 成 (B, nh, T, hs) 会把数据切错位置"
        )


def check_ex3():
    q, k = _rand(B, NH, T, HS), _rand(B, NH, T, HS)
    out = ex3_scores(q, k)
    _expect_shape("输出", out, (B, NH, T, T))
    want = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(HS)
    if torch.allclose(out, want * math.sqrt(HS)):
        raise AssertionError("忘了除以 sqrt(hs)")
    _expect_close("输出", out, want)


def check_ex4():
    scores = _rand(B, NH, T, T)
    out = ex4_causal_softmax(scores)
    _expect_shape("输出", out, (B, NH, T, T))
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert out[..., future].abs().max().item() < 1e-12, "右上三角（未来位置）应该全为 0"
    row_sum = out.sum(-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum)), "每一行之和应该为 1，检查 softmax 的 dim"
    for i in range(T):
        _expect_close(f"第 {i} 行", out[..., i, :i + 1], torch.softmax(scores[..., i, :i + 1], dim=-1))


def check_ex5():
    x, w = _rand(B, T, C), _rand(C, 3 * C)
    out = ex5_attention(x, w, NH)
    _expect_close("输出", out, _per_head_reference(x, w, NH))


def check_ex6():
    t_past = 4
    q_all, k_all, v_all = (_rand(B, NH, t_past + 1, HS) for _ in range(3))
    out, k_cache, v_cache = ex6_decode_step(
        q_all[:, :, -1:], k_all[:, :, -1:], v_all[:, :, -1:],
        k_all[:, :, :-1], v_all[:, :, :-1],
    )
    _expect_close("k_cache", k_cache, k_all)
    _expect_close("v_cache", v_cache, v_all)
    want = F.scaled_dot_product_attention(q_all, k_all, v_all, is_causal=True)[:, :, -1:]
    _expect_close("out", out, want)


def check_other_size():
    """换一组尺寸重跑：函数里如果用了文件顶部的全局 B/T/C/HS，这里会失败。"""
    b, t, c, nh = 1, 7, 16, 4
    x, w = _rand(b, t, c), _rand(c, 3 * c)
    _expect_close("换尺寸后的练习 5 输出", ex5_attention(x, w, nh), _per_head_reference(x, w, nh))
    q_all, k_all, v_all = (_rand(b, nh, t, c // nh) for _ in range(3))
    out, _, _ = ex6_decode_step(
        q_all[:, :, -1:], k_all[:, :, -1:], v_all[:, :, -1:],
        k_all[:, :, :-1], v_all[:, :, :-1],
    )
    want = F.scaled_dot_product_attention(q_all, k_all, v_all, is_causal=True)[:, :, -1:]
    _expect_close("换尺寸后的练习 6 输出", out, want)
    scores = _rand(b, nh, t, t)
    before = scores.clone()
    ex4_causal_softmax(scores)
    assert torch.equal(scores, before), "练习 4 修改了传进来的 scores（原地操作会影响调用者的数据）"


CHECKS = [check_ex1, check_ex2, check_ex3, check_ex4, check_ex5, check_ex6, check_other_size]


def main():
    torch.manual_seed(0)
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    passed = 0
    for idx, check in enumerate(CHECKS, start=1):
        if only is not None and idx != only:
            continue
        label = {6: "练习 6（附加）", 7: "换尺寸检查"}.get(idx, f"练习 {idx}")
        try:
            check()
        except NotImplementedError:
            print(f"[ TODO ] {label}：还没写")
        except AssertionError as e:
            print(f"[ FAIL ] {label}：{e}")
        except Exception as e:  # 形状不匹配等运行时错误
            print(f"[ERROR ] {label}：{type(e).__name__}: {e}")
        else:
            print(f"[ PASS ] {label}")
            passed += 1
    total = 1 if only is not None else len(CHECKS)
    print(f"\n通过 {passed} / {total}")


if __name__ == "__main__":
    main()
