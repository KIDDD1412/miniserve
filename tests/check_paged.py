"""
检查 M3（Paged KV Cache）：miniserve/paged.py

用法（在项目根目录 D:\\infra\\miniserve 下）：
    "D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" tests/check_paged.py
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniserve import paged as P  # noqa: E402


class UserCodeError(Exception):
    """你写的代码运行时报错了"""


def call(fn, *args):
    """调用你写的函数；运行时出了意料之外的错，就把完整报错打印出来"""
    try:
        return fn(*args)
    except NotImplementedError:
        raise
    except Exception:
        traceback.print_exc()
        raise UserCodeError(f"{fn.__name__} 运行时报错了（完整报错见上面）") from None


def check_blocks_needed():
    cases = [(0, 4), (1, 4), (3, 4), (4, 4), (5, 4), (8, 4), (9, 4), (16, 16), (17, 16), (100, 16)]
    for n, bs in cases:
        want = (n + bs - 1) // bs
        got = call(P.blocks_needed, n, bs)
        assert got is not None, "blocks_needed 没有返回值：是不是忘了 return？"
        assert isinstance(got, int), (
            f"blocks_needed({n}, {bs}) 返回的是 {got!r}（{type(got).__name__}），应该是整数（用 / 做除法会得到浮点数，要用 //）"
        )
        if got != want:
            hint = ""
            if n % bs == 0 and got == want + 1:
                hint = "：正好装满时多算了一块（n // block_size + 1 在整除时会多 1）"
            elif got == n // bs:
                hint = "：n // block_size 是向下取整，没装满的最后一块也要算"
            raise AssertionError(f"{n} 个 token、block_size={bs}，应该要 {want} 块，你算出 {got} 块{hint}")


def new_allocator(n):
    return P.BlockAllocator(n)


def check_allocator():
    a = new_allocator(4)

    # can_allocate
    for n, want in [(0, True), (3, True), (4, True), (5, False)]:
        got = call(a.can_allocate, n)
        assert got == want, f"4 个块全空闲时，can_allocate({n}) 应该是 {want}，实际是 {got}"

    # 分配光所有块
    ids = []
    for i in range(4):
        b = call(a.allocate)
        assert b is not None, "allocate 没有返回值：要 return 分到的块号"
        ids.append(b)
        assert a.num_free == 3 - i, f"分配了 {i + 1} 个块后应该剩 {3 - i} 个空闲块，实际剩 {a.num_free} 个"
    assert sorted(ids) == [0, 1, 2, 3], f"分配 4 次应该分到 0~3 号块各一次，实际分到 {ids}"
    assert ids == [0, 1, 2, 3], f"新建的分配器应该依次分出 0、1、2、3 号块，实际是 {ids}：要从最前面取（popleft），不是 pop"
    assert a.used == {0, 1, 2, 3}, f"分出去的块要记进 self.used，现在 self.used = {a.used}"
    assert call(a.can_allocate, 1) is False, "块全分完了，can_allocate(1) 应该是 False"

    # 没有空闲块时
    try:
        a.allocate()
    except NotImplementedError:
        raise
    except RuntimeError:
        pass
    except IndexError:
        raise AssertionError(
            "块分完了再 allocate，报的是 deque 自己的 IndexError：要先检查有没有空闲块，自己 raise RuntimeError"
        ) from None
    else:
        raise AssertionError("块分完了再 allocate 应该 raise RuntimeError，但没有报错")

    # 释放后可以再分配，而且能分到刚还回来的块
    call(a.free, ids[1])
    assert a.num_free == 1, f"还回 1 个块后应该有 1 个空闲块，实际有 {a.num_free} 个"
    assert ids[1] not in a.used, "还回去的块要从 self.used 里去掉"
    again = call(a.allocate)
    assert again == ids[1], f"唯一的空闲块是 {ids[1]} 号，再 allocate 应该分到它，实际分到 {again}"

    # 重复释放、释放没分配过的块
    call(a.free, ids[2])
    for bad, why in [(ids[2], "同一个块还了两次"), (99, "释放一个从没分配过的块号")]:
        try:
            a.free(bad)
        except NotImplementedError:
            raise
        except ValueError:
            pass
        except KeyError:
            raise AssertionError(
                f"{why}，报的是 set 自己的 KeyError：要先检查 block_id 在不在 self.used 里，自己 raise ValueError"
            ) from None
        else:
            raise AssertionError(f"{why}，应该 raise ValueError，但没有报错（这种 bug 会让两个请求共用一个块）")
    assert a.num_free == 1, f"出错的 free 不应该改动空闲块列表：现在空闲块有 {a.num_free} 个，应该是 1 个"

    # 还回去的块排在末尾：先还的先被再分出去
    b = new_allocator(3)
    x, y, z = b.allocate(), b.allocate(), b.allocate()
    b.free(z)
    b.free(x)
    assert [b.allocate(), b.allocate()] == [z, x], "还回来的块要放到 free_blocks 末尾（append），先还的先被分出去"


def ref_slot(table, pos, bs):
    return table[pos // bs] * bs + pos % bs


def check_slot_mapping():
    table, bs = [7, 2, 5], 4
    positions = [0, 1, 3, 4, 5, 8, 9]
    want = [ref_slot(table, p, bs) for p in positions]        # [28, 29, 31, 8, 9, 20, 21]
    got = call(P.slot_mapping, table, positions, bs)
    assert got is not None, "slot_mapping 没有返回值：是不是忘了 return？"
    assert isinstance(got, list), f"slot_mapping 应该返回 Python 列表，实际是 {type(got).__name__}"
    assert len(got) == len(positions), f"传进 {len(positions)} 个位置，应该返回 {len(positions)} 个槽位号，实际 {len(got)} 个"
    if got != want:
        no_table = [p for p in positions]                      # 没查块表：槽位号 = 位置本身
        if got == no_table:
            raise AssertionError(
                f"算出的槽位号就是位置本身 {got}：要先查块表（block_table[pos // block_size]），得到池子里真正的块号"
            )
        for p, g, w in zip(positions, got, want):
            if g != w:
                raise AssertionError(
                    f"块表 {table}、block_size={bs}：第 {p} 个 token 在请求的第 {p // bs} 个块（池子里的 {table[p // bs]} 号块）"
                    f"、块内第 {p % bs} 个，槽位应该是 {table[p // bs]} × {bs} + {p % bs} = {w}，你算出 {g}"
                )
    assert call(P.slot_mapping, table, [], bs) == [], "positions 为空时应该返回空列表"


def check_ensure_blocks():
    # prefill 时一次就要好几个块
    t0 = []
    call(P.ensure_blocks, t0, 9, 4, P.BlockAllocator(10))
    assert len(t0) == 3, (
        f"空块表、一次装 9 个 token（block_size=4）要 3 块，实际分了 {len(t0)} 块：是不是用 if 只补了一块？要一直补到够为止"
    )

    a = P.BlockAllocator(10)
    a.allocate()                                   # 先占掉 0 号块，确认是从分配器拿的，不是自己编号
    table = []
    for n, want_len in [(0, 0), (1, 1), (4, 1), (5, 2), (8, 2), (9, 3), (9, 3), (16, 4)]:
        before = list(table)
        r = call(P.ensure_blocks, table, n, 4, a)
        assert r is None or r is table, "ensure_blocks 直接改传进来的 block_table 就行，不用返回新列表"
        assert len(table) == want_len, (
            f"block_size=4，装 {n} 个 token 要 {want_len} 块；调用前块表有 {len(before)} 块，调用后变成 {len(table)} 块"
            + ("：已经够了的时候不应该再分配" if len(table) > want_len and len(before) >= want_len else "")
        )
        assert table[:len(before)] == before, f"已经分到的块不能变：调用前 {before}，调用后 {table}"
    assert table == [1, 2, 3, 4], f"新块要用 allocator.allocate() 分配、按顺序 append 到末尾：期望 [1, 2, 3, 4]，实际 {table}"
    assert a.num_free == 5, f"一共分出去 5 个块，分配器里应该剩 5 个空闲块，实际剩 {a.num_free} 个"


def check_pool():
    import torch
    from miniserve.model import GPTConfig

    cfg = GPTConfig(block_size=64, vocab_size=100, n_layer=3, n_head=2, n_embd=8)
    pool = P.KVCachePool(cfg, num_blocks=6, block_size=4)
    k_id, v_id = id(pool.k), id(pool.v)
    g = torch.Generator().manual_seed(0)

    # 两个请求交替写：A 先 prefill 5 个 token，B prefill 3 个，然后两个都 decode 4 步
    tables = {"A": [], "B": []}
    lengths = {"A": 0, "B": 0}
    written = {(r, layer): [] for r in "AB" for layer in range(cfg.n_layer)}   # 每层按顺序写过的 (k, v)

    def write_tokens(r, n):
        positions = list(range(lengths[r], lengths[r] + n))
        P.ensure_blocks(tables[r], lengths[r] + n, pool.block_size, pool.allocator)
        slots = torch.tensor(P.slot_mapping(tables[r], positions, pool.block_size))
        for layer in range(cfg.n_layer):
            k = torch.randn(n, cfg.n_embd, generator=g)
            v = torch.randn(n, cfg.n_embd, generator=g)
            call(pool.write, layer, slots, k, v)
            written[(r, layer)].append((k, v))
        lengths[r] += n

    write_tokens("A", 5)
    write_tokens("B", 3)
    for _ in range(4):
        write_tokens("A", 1)
        write_tokens("B", 1)

    assert id(pool.k) == k_id and id(pool.v) == v_id, (
        "write 把 self.k / self.v 换成了新张量：应该直接往原来的池子里写（self.k[...] = ...），不要重新赋值整个 self.k"
    )
    for r in "AB":
        slots = torch.tensor(P.slot_mapping(tables[r], list(range(lengths[r])), pool.block_size))
        for layer in range(cfg.n_layer):
            want_k = torch.cat([k for k, _ in written[(r, layer)]])
            want_v = torch.cat([v for _, v in written[(r, layer)]])
            got_k, got_v = pool.read(layer, slots)
            if torch.equal(got_k, want_v) and torch.equal(got_v, want_k):
                raise AssertionError("k 和 v 写反了：k 要写进 self.k，v 写进 self.v")
            if torch.count_nonzero(got_k) == 0:
                raise AssertionError(
                    f"第 {layer} 层读出来全是 0：k 没写进池子。是不是写进了一个副本？（比如先 clone 再改）"
                )
            if torch.count_nonzero(got_v) == 0:
                raise AssertionError(f"第 {layer} 层的 v 读出来全是 0：v 没写进 self.v")
            assert torch.equal(got_k, want_k) and torch.equal(got_v, want_v), (
                f"请求 {r} 第 {layer} 层读出来的 k / v 和写进去的不一样：检查 write 有没有写到第 layer 层、槽位 slots 上"
            )
    used_slots = set()
    for r in "AB":
        used_slots |= set(P.slot_mapping(tables[r], list(range(lengths[r])), pool.block_size))
    unused = torch.tensor(sorted(set(range(pool.num_slots)) - used_slots))
    assert torch.count_nonzero(pool.k[:, unused]) == 0, "没用到的槽位也被写了：write 只能写 slots 指定的那几个槽位、第 layer 层"


def call_build_slot_table(*args, **kwargs):
    """调用 build_slot_table；每行没补齐时 torch.tensor 会报 ValueError，单独给提示"""
    try:
        return P.build_slot_table(*args, **kwargs)
    except NotImplementedError:
        raise
    except ValueError as e:
        if "expected sequence of length" in str(e):
            traceback.print_exc()
            raise AssertionError(
                "torch.tensor 报错说各行长度不一样：每一行都要补齐到 Lmax 个元素，才能拼成 (B, Lmax) 的矩阵（完整报错见上面）"
            ) from None
        traceback.print_exc()
        raise UserCodeError("build_slot_table 运行时报错了（完整报错见上面）") from None
    except Exception:
        traceback.print_exc()
        raise UserCodeError("build_slot_table 运行时报错了（完整报错见上面）") from None


def ref_slot_table(tables, lens, bs):
    import torch
    Lmax = max(lens)
    slots = [[ref_slot(t, p, bs) for p in range(L)] + [0] * (Lmax - L) for t, L in zip(tables, lens)]
    valid = [[True] * L + [False] * (Lmax - L) for L in lens]
    return torch.tensor(slots), torch.tensor(valid)


def check_one_slot_table(tables, lens, bs):
    import torch

    want_slots, want_valid = ref_slot_table(tables, lens, bs)
    B, Lmax = len(lens), max(lens)
    case = f"块表 {tables}、seq_lens {lens}、block_size={bs}"

    got = call_build_slot_table(tables, lens, bs)
    assert got is not None, "build_slot_table 没有返回值：要 return slot_table, valid"
    assert isinstance(got, tuple) and len(got) == 2, (
        f"build_slot_table 应该返回两个东西（return slot_table, valid），实际返回的是 {type(got).__name__}"
    )
    slots, valid = got
    for name, t in [("slot_table", slots), ("valid", valid)]:
        assert isinstance(t, torch.Tensor), (
            f"{name} 应该是张量，实际是 {type(t).__name__}：最后要用 torch.tensor(...) 把列表的列表变成张量"
        )
        assert t.dim() == 2, f"{name} 应该是二维的 (B, Lmax)，实际形状是 {tuple(t.shape)}"
        if t.shape[1] != Lmax and t.shape[0] == B:
            raise AssertionError(
                f"{case}：{name} 的形状应该是 ({B}, {Lmax})，实际是 {tuple(t.shape)}。"
                f"每一行要正好 Lmax = max(seq_lens) = {Lmax} 个元素：前 L 个是真 token，后面补 Lmax - L 个"
                "（range(L) 是 0 ~ L-1，正好 L 个）"
            )
        assert tuple(t.shape) == (B, Lmax), f"{case}：{name} 的形状应该是 ({B}, {Lmax})，实际是 {tuple(t.shape)}"

    assert not slots.is_floating_point(), (
        f"slot_table 的类型是 {slots.dtype}，应该是整数：槽位号要拿来当下标用，浮点数不能当下标"
    )
    assert valid.dtype == torch.bool, (
        f"valid 的类型是 {valid.dtype}，应该是 torch.bool：valid 那一行要用 True / False 来建，不要用 1 / 0"
    )

    if not torch.equal(valid, want_valid):
        if torch.equal(valid, ~want_valid):
            raise AssertionError("valid 正好反了：真 token 的格子是 True，补齐的格子才是 False")
        if torch.equal(valid, slots != 0):
            raise AssertionError(
                "valid 看起来是用 slot_table != 0 算的：0 号槽位也可能放着真 token（块表里有 0 号块时），"
                "要按每个请求的长度 L 来建：L 个 True + (Lmax - L) 个 False"
            )
        for b in range(B):
            if not torch.equal(valid[b], want_valid[b]):
                raise AssertionError(
                    f"{case}：第 {b} 个请求有 {lens[b]} 个 token，valid 第 {b} 行应该是 {want_valid[b].tolist()}，"
                    f"实际是 {valid[b].tolist()}"
                )

    for b in range(B):
        L = lens[b]
        g, w = slots[b, :L].tolist(), want_slots[b, :L].tolist()
        if g != w:
            hint = ""
            if g == list(range(L)):
                hint = "：槽位号就是位置本身，没查块表 —— 要用 slot_mapping(table, 第 0 ~ L-1 个位置, block_size)"
            raise AssertionError(
                f"{case}：第 {b} 个请求的前 {L} 个槽位应该是 {w}，实际是 {g}{hint}"
            )
    pad = slots[~want_valid]
    assert torch.count_nonzero(pad) == 0, (
        f"{case}：补齐的格子要填 0，现在 slot_table = {slots.tolist()}"
    )


def check_slot_table():
    import torch

    # 0 号块也在用：请求 1 的第 0 个 token 就在 0 号槽位
    check_one_slot_table([[7, 2, 5], [0], [3, 4]], [10, 1, 5], 4)
    # 长度都一样：不用补齐
    check_one_slot_table([[2], [0]], [3, 3], 4)
    # 只有一个请求、正好装满两个块
    check_one_slot_table([[3, 1]], [8], 4)
    # 补齐的长度超过一个块
    check_one_slot_table([[5, 6, 7, 8], [9]], [13, 2], 4)

    # 有没有把张量放到传进来的 device 上（"meta" 是 PyTorch 自带的假设备，不需要显卡）
    try:
        slots, valid = call_build_slot_table([[7, 2, 5], [1]], [10, 1], 4, "meta")
    except NotImplementedError:          # 前面几项都过了，说明已经写完；这里是 meta 设备不支持某个操作
        traceback.print_exc()
        raise UserCodeError(
            "build_slot_table 在 device='meta' 上运行时报错了（完整报错见上面）：要先用 Python 列表准备好数据，"
            "最后一步再 torch.tensor(..., device=device)"
        ) from None
    assert slots.device.type == "meta" and valid.device.type == "meta", (
        f"传了 device='meta'，但返回的张量在 {slots.device} / {valid.device} 上：要写 torch.tensor(..., device=device)，"
        "否则以后在 GPU 上跑时，槽位表会留在 CPU 上"
    )


def ref_decode_attention(q, k, v, lens, nh, scale=True):
    """逐个请求单独算（只用前 L 个真 token），当作标准答案"""
    import math
    import torch
    B, Lmax, C = k.shape
    hs = C // nh
    outs = []
    for b in range(B):
        L = lens[b]
        qb = q[b].view(nh, 1, hs)                           # (nh, 1, hs)
        kb = k[b, :L].view(L, nh, hs).transpose(0, 1)       # (nh, L, hs)
        vb = v[b, :L].view(L, nh, hs).transpose(0, 1)
        s = qb @ kb.transpose(-2, -1)
        if scale:
            s = s / math.sqrt(hs)
        outs.append((s.softmax(-1) @ vb).transpose(0, 1).reshape(C))
    return torch.stack(outs)


def make_decode_inputs(lens, nh, C, g):
    """随机的 q、k、v；补齐的格子里故意放很大的数：没盖住的话结果会差很多"""
    import torch
    B, Lmax = len(lens), max(lens)
    q = torch.randn(B, C, generator=g, dtype=torch.float64)
    k = torch.randn(B, Lmax, C, generator=g, dtype=torch.float64)
    v = torch.randn(B, Lmax, C, generator=g, dtype=torch.float64)
    valid = torch.tensor([[True] * L + [False] * (Lmax - L) for L in lens])
    for b, L in enumerate(lens):
        k[b, L:] = 10 * q[b]          # 和 q 的点积很大：没盖住的话 softmax 几乎全给它
        v[b, L:] = 100.0
    return q, k, v, valid


def call_decode_attention(*args):
    import torch  # noqa: F401
    try:
        return P.paged_decode_attention(*args)
    except NotImplementedError:
        raise
    except RuntimeError as e:
        traceback.print_exc()
        msg = str(e)
        if "broadcast" in msg or "must match the size" in msg or "expand" in msg:
            raise AssertionError(
                "形状对不上（完整报错见上面）：scores 是 (B, nh, 1, Lmax)，valid 是 (B, Lmax)，"
                "要先变成 (B, 1, 1, Lmax) 才能和 scores 对齐 —— valid[:, None, None, :]"
            ) from None
        raise UserCodeError("paged_decode_attention 运行时报错了（完整报错见上面）") from None
    except Exception:
        traceback.print_exc()
        raise UserCodeError("paged_decode_attention 运行时报错了（完整报错见上面）") from None


def check_one_decode(lens, nh, C, g):
    import torch

    q, k, v, valid = make_decode_inputs(lens, nh, C, g)
    B, Lmax = len(lens), max(lens)
    case = f"seq_lens {lens}、n_head={nh}、C={C}"
    backup = [t.clone() for t in (q, k, v, valid)]

    got = call_decode_attention(q, k, v, valid, nh)
    assert got is not None, "paged_decode_attention 没有返回值：是不是忘了 return？"
    assert isinstance(got, torch.Tensor), f"应该返回张量，实际返回的是 {type(got).__name__}"
    for name, before, after in zip(("q", "k", "v", "valid"), backup, (q, k, v, valid)):
        assert torch.equal(before, after), (
            f"传进来的 {name} 被改掉了：不要原地修改输入（比如对 {name} 用带下划线的 masked_fill_ / logical_not_），"
            "只能原地改你自己算出来的 scores"
        )
    if got.shape != (B, C):
        hint = ""
        if got.numel() == B * C:
            hint = "：元素个数对，形状不对 —— 最后要 .view(B, C)"
        raise AssertionError(f"{case}：返回的形状应该是 ({B}, {C})，实际是 {tuple(got.shape)}{hint}")
    assert got.dtype == q.dtype, f"输入是 {q.dtype}，返回的却是 {got.dtype}：中间不要换数据类型"

    want = ref_decode_attention(q, k, v, lens, nh)
    if torch.allclose(got, want, atol=1e-10):
        return

    full = torch.tensor([L == Lmax for L in lens])
    nan_rows = torch.isnan(got).any(dim=-1)
    if nan_rows[full].any():
        raise AssertionError(
            f"{case}：没有补齐的请求（每个格子都是真 token）算出了 NaN —— 说明把真 token 盖住了。"
            "要填 -inf 的是【不是】真 token 的格子：用 ~valid，不是 valid"
        )
    if nan_rows.any():
        raise AssertionError(
            f"{case}：有补齐的请求算出了 NaN。检查 softmax 是不是沿最后一维（dim=-1，也就是 Lmax 那一维）"
        )
    if torch.allclose(got, ref_decode_attention(q, k, v, [Lmax] * B, nh), atol=1e-10):
        raise AssertionError(
            f"{case}：结果和【不盖补齐格子】时一模一样 —— 补齐的格子没有填 -inf，"
            "masked_fill 的结果有没有接住？（不带下划线的 masked_fill 返回新张量，要写 scores = scores.masked_fill(...)）"
        )
    if torch.allclose(got, ref_decode_attention(q, k, v, lens, nh, scale=False), atol=1e-10):
        raise AssertionError(f"{case}：结果和不除以 sqrt(hs) 时一样 —— scores 忘了除以 math.sqrt(hs)")
    bad = [b for b in range(B) if not torch.allclose(got[b], want[b], atol=1e-10)]
    raise AssertionError(
        f"{case}：第 {bad} 个请求的结果和逐个请求单独算的不一样（最大误差 {(got - want).abs().max().item():.3g}）。"
        "检查拆头：q 是 view(B, 1, nh, hs).transpose(1, 2)，k、v 是 view(B, Lmax, nh, hs).transpose(1, 2)"
    )


def check_decode_attention():
    import torch

    g = torch.Generator().manual_seed(0)
    check_one_decode([10, 1, 5], 2, 8, g)          # 有长有短，有的只有 1 个 token
    check_one_decode([4, 4], 4, 8, g)              # 一样长，不用补齐；4 个头
    check_one_decode([1], 2, 8, g)                 # 只有一个请求、只有 1 个 token
    check_one_decode([3, 7, 2, 7], 2, 12, g)       # hs = 6


def call_decode_layer(*args):
    try:
        return P.paged_decode_layer(*args)
    except NotImplementedError:
        raise
    except ValueError as e:
        traceback.print_exc()
        if "not enough values to unpack" in str(e):
            raise AssertionError(
                "paged_decode_attention 里 B, Lmax, C = k.shape 拆不开（完整报错见上面）：传给它的 k、v 应该是 (B, Lmax, C) 的全部历史，"
                "也就是 ③ pool.read 读出来的 k_all、v_all，不是 ① 新 token 的 k、v"
            ) from None
        raise UserCodeError("paged_decode_layer 运行时报错了（完整报错见上面）") from None
    except Exception:
        traceback.print_exc()
        raise UserCodeError("paged_decode_layer 运行时报错了（完整报错见上面）") from None


def check_decode_layer():
    """和 M2 的 packed_attention（已经和 HF 对齐过）比：同样的历史、同样的新 token，结果要一样"""
    import torch
    from miniserve.model import GPT2, GPTConfig, LayerKVCache
    from miniserve.packed import packed_attention

    torch.manual_seed(0)
    cfg = GPTConfig(block_size=64, vocab_size=50, n_layer=2, n_head=2, n_embd=8)
    model = GPT2(cfg).double().eval()
    layer = 1                                  # 故意用第 1 层：检查有没有写到 / 读自第 layer 层
    attn = model.h[layer].attn
    C, bs = cfg.n_embd, 4
    pool = P.KVCachePool(cfg, num_blocks=12, block_size=bs, dtype=torch.float64)
    pool.allocator.allocate()                  # 占掉 0 号块：补齐的格子读 0 号槽位，读到的是全 0

    hist_lens = [9, 0, 4]                      # 每个请求之前已经有几个 token（第 1 个请求是刚 prefill 完的空历史）
    B = len(hist_lens)
    tables, caches = [], []
    for L in hist_lens:
        table = []
        P.ensure_blocks(table, L + 1, bs, pool.allocator)     # 连新 token 的位置一起分好块
        k_hist = torch.randn(L, C, dtype=torch.float64)
        v_hist = torch.randn(L, C, dtype=torch.float64)
        cache = LayerKVCache()
        if L > 0:
            pool.write(layer, torch.tensor(P.slot_mapping(table, list(range(L)), bs)), k_hist, v_hist)
            cache.k = k_hist.view(1, L, cfg.n_head, C // cfg.n_head).transpose(1, 2)
            cache.v = v_hist.view(1, L, cfg.n_head, C // cfg.n_head).transpose(1, 2)
        tables.append(table)
        caches.append(cache)

    new_slots = torch.tensor([P.slot_mapping(t, [L], bs)[0] for t, L in zip(tables, hist_lens)])
    slot_table, valid = P.build_slot_table(tables, [L + 1 for L in hist_lens], bs)
    x = torch.randn(B, C, dtype=torch.float64)

    with torch.no_grad():
        want = packed_attention(attn, x, [1] * B, caches)
        q_new, k_new, v_new = attn.c_attn(x).split(C, dim=-1)
        # "先读后写"会得到的结果：新 token 的槽位还是 0
        k_old, v_old = pool.k[layer, slot_table].clone(), pool.v[layer, slot_table].clone()
        read_first = attn.c_proj(P.paged_decode_attention(q_new, k_old, v_old, valid, cfg.n_head))
        layer0_before = pool.k[0].clone()
        got = call_decode_layer(attn, x, layer, pool, new_slots, slot_table, valid)

    assert got is not None, "paged_decode_layer 没有返回值：是不是忘了 return？"
    assert isinstance(got, torch.Tensor), f"应该返回张量，实际返回的是 {type(got).__name__}"
    assert tuple(got.shape) == (B, C), f"返回的形状应该是 ({B}, {C})，实际是 {tuple(got.shape)}"

    # 新 token 的 k、v 有没有写进池子
    if not torch.equal(pool.k[0], layer0_before):
        raise AssertionError("第 0 层的池子被改了：新 token 的 k、v 要写进第 layer 层（pool.write(layer, ...)）")
    got_k, got_v = pool.k[layer, new_slots], pool.v[layer, new_slots]
    if torch.count_nonzero(got_k) == 0:
        raise AssertionError("新 token 的 k 没有写进池子：要用 pool.write(layer, new_slots, k, v)")
    if torch.allclose(got_k, v_new) and torch.allclose(got_v, k_new):
        raise AssertionError("写进池子的 k、v 反了：pool.write(layer, new_slots, k, v)，k 在前")
    if torch.allclose(got_k, x):
        raise AssertionError("写进池子的是 x 本身：要先过 attn.c_attn(x) 再 split 出 k、v，写的是 k、v")
    assert torch.allclose(got_k, k_new) and torch.allclose(got_v, v_new), (
        "写进池子的 k、v 不对：应该是 attn.c_attn(x).split(C, dim=-1) 拆出来的第 2、3 份"
    )

    if torch.allclose(got, want, atol=1e-10):
        return
    if torch.allclose(got, read_first, atol=1e-10):
        raise AssertionError(
            "结果和【先读后写】时一样：读出来的历史里没有新 token 自己。要先 pool.write，再 pool.read"
        )
    if torch.allclose(attn.c_proj(got), want, atol=1e-10):
        raise AssertionError("差一步输出投影：最后要 return attn.c_proj(...)")
    bad = [b for b in range(B) if not torch.allclose(got[b], want[b], atol=1e-10)]
    raise AssertionError(
        f"第 {bad} 个请求的结果和 packed_attention 不一样（最大误差 {(got - want).abs().max().item():.3g}）。"
        "检查：pool.read 读的是第 layer 层、用的是 slot_table；paged_decode_attention 传的是 ① 的 q 和 ③ 读出来的 k_all、v_all"
    )


def call_decode_forward(*args):
    try:
        return P.paged_decode_forward(*args)
    except NotImplementedError:
        raise
    except IndexError as e:
        traceback.print_exc()
        if "list index out of range" in str(e):
            raise AssertionError(
                "列表下标越界（完整报错见上面），很可能是块表不够长：新 token 在第 pos 个位置，"
                "写它之前要先让块表够装下 pos + 1 个 token（第 2b 步的 ensure_blocks）"
            ) from None
        raise UserCodeError("paged_decode_forward 运行时报错了（完整报错见上面）") from None
    except RuntimeError as e:
        traceback.print_exc()
        if "same device" in str(e):
            raise AssertionError(
                "有张量和模型不在同一个设备上（完整报错见上面）：你新建的张量（位置、槽位……）要放到 tokens.device 上"
            ) from None
        raise UserCodeError("paged_decode_forward 运行时报错了（完整报错见上面）") from None
    except Exception:
        traceback.print_exc()
        raise UserCodeError("paged_decode_forward 运行时报错了（完整报错见上面）") from None


def run_decode_forward(device):
    """先用 M2 的 packed_forward 做 prefill、把缓存搬进池子，再两边一起 decode 6 轮，每轮对比 logits"""
    import torch
    from miniserve.model import GPT2, GPTConfig, new_kv_cache
    from miniserve.packed import packed_forward

    torch.manual_seed(0)
    cfg = GPTConfig(block_size=64, vocab_size=50, n_layer=2, n_head=2, n_embd=8)
    model = GPT2(cfg).double().eval().to(device)
    C, bs = cfg.n_embd, 4
    pool = P.KVCachePool(cfg, num_blocks=16, block_size=bs, device=device, dtype=torch.float64)
    pool.allocator.allocate()                  # 占掉 0 号块

    prompt_lens = [5, 1, 8]                    # 8 个 token 正好装满 2 块：第一轮 decode 就要补新块
    B = len(prompt_lens)
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, cfg.vocab_size, (L,), generator=g) for L in prompt_lens]

    kv_caches = [new_kv_cache(cfg) for _ in range(B)]
    with torch.no_grad():
        logits = packed_forward(model, torch.cat(prompts).to(device),
                                torch.cat([torch.arange(L) for L in prompt_lens]).to(device), prompt_lens, kv_caches)

    def cache_kv(b, i):                        # packed 路径里请求 b 第 i 层的缓存，变回 (L, C)
        c = kv_caches[b][i]
        L = c.k.size(2)
        return c.k[0].transpose(0, 1).reshape(L, C), c.v[0].transpose(0, 1).reshape(L, C)

    tables = []
    for b, L in enumerate(prompt_lens):
        table = []
        P.ensure_blocks(table, L, bs, pool.allocator)
        slots = torch.tensor(P.slot_mapping(table, list(range(L)), bs), device=device)
        for i in range(cfg.n_layer):
            pool.write(i, slots, *cache_kv(b, i))
        tables.append(table)

    lens = list(prompt_lens)
    for step in range(6):
        where = f"（{device} 上第 {step + 1} 轮 decode，新 token 的位置 {lens}）"
        next_tokens = logits.argmax(dim=-1)
        positions = list(lens)
        with torch.no_grad():
            want = packed_forward(model, next_tokens, torch.tensor(positions, device=device), [1] * B, kv_caches)
            before = [list(t) for t in tables]
            got = call_decode_forward(model, next_tokens, positions, tables, pool)

        assert got is not None, "paged_decode_forward 没有返回值：是不是忘了 return？"
        assert isinstance(got, torch.Tensor), f"应该返回张量，实际返回的是 {type(got).__name__}"
        assert tuple(got.shape) == (B, cfg.vocab_size), (
            f"返回的形状应该是 ({B}, {cfg.vocab_size})，实际是 {tuple(got.shape)}{where}"
        )

        # 块表：原来的块不能变，够装新 token，新块来自分配器
        for b in range(B):
            assert tables[b][:len(before[b])] == before[b], f"请求 {b} 原来的块被改了：调用前 {before[b]}，调用后 {tables[b]}"
            need = P.blocks_needed(positions[b] + 1, bs)
            assert len(tables[b]) >= need, (
                f"请求 {b} 的新 token 在位置 {positions[b]}，块表要至少 {need} 块，实际 {tables[b]}{where}：要先 ensure_blocks"
            )
        assert all(blk in pool.allocator.used for t in tables for blk in t), (
            "块表里有不是分配器分出来的块号：补块要用 ensure_blocks（它会调用 pool.allocator.allocate()）"
        )

        # 池子：逐层对比 packed 路径的缓存，找出第一个出错的地方
        for i in range(cfg.n_layer):
            for b in range(B):
                pos = positions[b]
                want_k, want_v = cache_kv(b, i)
                hist = torch.tensor(P.slot_mapping(tables[b], list(range(pos)), bs), device=device, dtype=torch.long)
                new_slot = torch.tensor(P.slot_mapping(tables[b], [pos], bs), device=device)
                hk, hv = pool.read(i, hist)
                if not (torch.allclose(hk, want_k[:pos], atol=1e-10) and torch.allclose(hv, want_v[:pos], atol=1e-10)):
                    raise AssertionError(
                        f"请求 {b} 第 {i} 层：之前的旧 token 的 k、v 被改掉了{where}。"
                        "新 token 写错了槽位，覆盖了旧 token —— 要写进【第 pos 个位置】对应的槽位"
                    )
                nk, nv = pool.read(i, new_slot)
                if torch.count_nonzero(nk) == 0:
                    raise AssertionError(
                        f"请求 {b} 第 {i} 层：第 pos 个位置的槽位上没有东西{where}。新 token 的 k、v 没写到这里：检查 new_slots 是不是用第 pos 个位置算的"
                    )
                if torch.allclose(nk, want_k[pos:], atol=1e-10) and torch.allclose(nv, want_v[pos:], atol=1e-10):
                    continue
                if i == 0:
                    raise AssertionError(
                        f"请求 {b}：第 0 层新 token 的 k、v 就不对{where} —— 说明进第 0 层之前的 x 就错了。"
                        "检查 x = 词嵌入 + 位置嵌入，位置嵌入要用传进来的 positions（新 token 在各自请求里的位置）"
                    )
                raise AssertionError(
                    f"请求 {b}：{'第 0 层' if i == 1 else f'第 0 ~ {i - 1} 层'}都对，第 {i} 层新 token 的 k、v 不对{where} —— 问题出在层与层之间。检查："
                    "① 每个请求的长度要算上新 token（pos + 1），不然新 token 看不到自己"
                    "② 每层有两个残差：x = x + attention(...)，x = x + mlp(ln_2(x))"
                )

        if not torch.allclose(got, want, atol=1e-8):
            bad = [b for b in range(B) if not torch.allclose(got[b], want[b], atol=1e-8)]
            raise AssertionError(
                f"第 {bad} 个请求的 logits 和 packed_forward 不一样{where}，最大误差 {(got - want).abs().max().item():.3g}。"
                "每一层池子里的 k、v 都已经确认是对的，问题在最后一层的 attention 之后：检查最后一层的 MLP 残差，以及最后的 model.ln_f 和 model.lm_head"
            )
        lens = [L + 1 for L in lens]
        logits = want


def check_decode_forward():
    import torch
    run_decode_forward("cpu")
    if torch.cuda.is_available():
        run_decode_forward("cuda")             # 在 GPU 上再跑一遍：检查新建的张量有没有放对设备


STEPS = [
    ("M3 第 1a 步：blocks_needed", check_blocks_needed),
    ("M3 第 1b 步：BlockAllocator", check_allocator),
    ("M3 第 2a 步：slot_mapping", check_slot_mapping),
    ("M3 第 2b 步：ensure_blocks", check_ensure_blocks),
    ("M3 第 2c 步：KVCachePool.write", check_pool),
    ("M3 第 3a 步：build_slot_table", check_slot_table),
    ("M3 第 3b 步：paged_decode_attention", check_decode_attention),
    ("M3 第 3c-1 步：paged_decode_layer", check_decode_layer),
    ("M3 第 3c-2 步：paged_decode_forward", check_decode_forward),
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
        except Exception as e:
            traceback.print_exc()
            print(f"[ERROR ] {label}：{type(e).__name__}: {e}")
            break
        else:
            print(f"[ PASS ] {label}")


if __name__ == "__main__":
    main()
