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


STEPS = [
    ("M3 第 1a 步：blocks_needed", check_blocks_needed),
    ("M3 第 1b 步：BlockAllocator", check_allocator),
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
