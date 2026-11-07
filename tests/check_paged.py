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


STEPS = [
    ("M3 第 1a 步：blocks_needed", check_blocks_needed),
    ("M3 第 1b 步：BlockAllocator", check_allocator),
    ("M3 第 2a 步：slot_mapping", check_slot_mapping),
    ("M3 第 2b 步：ensure_blocks", check_ensure_blocks),
    ("M3 第 2c 步：KVCachePool.write", check_pool),
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
