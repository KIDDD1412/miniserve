"""
M3：Paged KV Cache —— 像操作系统管理内存那样管理 KV Cache。

M1 / M2 的 KV Cache 有两个问题：
    ① 每个请求的缓存是一个不断 torch.cat 变长的张量：每生成一个 token，就要把整个缓存复制一遍
    ② 每个请求的缓存是各自独立的张量，attention 只能对每个请求写 for 循环（M2 第 4 步测出来的瓶颈）

Paged KV Cache 的做法：
    - 开工前，在显存里一次性开好一大块"缓存池"，切成很多个固定大小的【块】（block），
      每个块能放 block_size 个 token 的 k、v
    - 请求需要多少，就给它分配几个块；块不用连续，用一张【块表】（block table）记下
      "这个请求的第 0、1、2 …… 个块，分别是池子里的几号块"
    - 请求结束，把它的块还回池子，给别的请求用

和操作系统的虚拟内存分页一模一样：
    块 ⇔ 物理页，块表 ⇔ 页表，"请求里第几个 token" ⇔ 虚拟地址

例子（block_size = 4）：请求 A 有 10 个 token，需要 3 个块，分到了池子里的 7 号、2 号、5 号块
    块表 = [7, 2, 5]
    第 0~3 个 token 放在 7 号块，第 4~7 个放在 2 号块，第 8~9 个放在 5 号块（5 号块还剩 2 个空位）

第 1 步：块分配器 —— 只管"哪些块空着、分给谁、收回来"，还不碰张量
第 2 步：缓存池 —— 真正存 k、v 的大张量，以及"第几个 token 存在池子里哪个位置"的换算
检查："D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" tests/check_paged.py
"""
from collections import deque

import torch

from miniserve.model import GPTConfig


# ---------------------------------------------------------------------------
# M3 第 1a 步：装下 n 个 token 需要几个块
# ---------------------------------------------------------------------------
def blocks_needed(num_tokens: int, block_size: int) -> int:
    """
    装下 num_tokens 个 token 需要几个块（向上取整：最后一个块没装满也要占一整块）
    例：block_size = 4 时，4 个 token 要 1 块，5 个 token 要 2 块，0 个 token 要 0 块
    """
    # TODO 一行，只用整数运算（不要用 float）
    return (num_tokens + block_size -1) // block_size


# ---------------------------------------------------------------------------
# M3 第 1b 步：块分配器
# ---------------------------------------------------------------------------
class BlockAllocator:
    """管理缓存池里 num_blocks 个块（编号 0 ~ num_blocks-1）的分配和回收"""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.free_blocks: deque[int] = deque(range(num_blocks))   # 空闲的块号，一开始全部空闲
        self.used: set[int] = set()                   # 已经分配出去的块号

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, n: int) -> bool:
        """现在还能不能一次分出 n 个块"""
        # TODO 一行
        return self.num_free >= n

    def allocate(self) -> int:
        """分配 1 个块，返回它的块号"""
        # TODO ① 没有空闲块时：raise RuntimeError("没有空闲块了")
        #      ② 从 free_blocks 取出最前面的一个块号（popleft）
        #      ③ 记进 self.used，再返回它
        if self.num_free == 0:
            raise RuntimeError("无空闲块")
        num = self.free_blocks.popleft()
        self.used.add(num)
        return num

    def free(self, block_id: int) -> None:
        """把一个块还回池子"""
        # TODO ① block_id 不在 self.used 里（没分配过，或者已经还过一次了）：
        #         raise ValueError(f"块 {block_id} 没有被分配，不能释放")
        #      ② 从 self.used 里去掉，再放回 free_blocks 的末尾
        if block_id not in self.used:
            raise ValueError(f"块 {block_id} 没有被分配，不能释放")
        self.used.remove(block_id)
        self.free_blocks.append(block_id)


# ---------------------------------------------------------------------------
# M3 第 2 步：缓存池
#
# 池子里一共有 num_blocks × block_size 个【槽位】（slot），每个槽位放 1 个 token 的 k（或 v），
# 槽位按块排好：0 号块是槽位 0 ~ block_size-1，1 号块是槽位 block_size ~ 2*block_size-1，……
#
# 例子（block_size = 4，块表 = [7, 2, 5]）：
#     第 5 个 token → 在请求的第 5 // 4 = 1 个块里，块内第 5 % 4 = 1 个位置
#                   → 块表[1] = 2 号块 → 槽位 2 × 4 + 1 = 9
# ---------------------------------------------------------------------------
def slot_mapping(block_table: list[int], positions: list[int], block_size: int) -> list[int]:
    """
    第 2a 步：一个请求里第 positions[i] 个 token，存在池子里的几号槽位
    block_table: 这个请求的块表，比如 [7, 2, 5]
    positions:   要换算的 token 位置（Python 列表），比如 [4, 5, 6]
    返回：        对应的槽位号（Python 列表），和 positions 一一对应
    """
    # TODO 对每个位置 pos：先算它在请求的第几个块、块内第几个，再查块表得到真正的块号，最后换算成槽位号
    #   提示：用列表推导式，一行就能写完
    return [block_table[p // block_size] * block_size + p % block_size for p in positions]

def ensure_blocks(block_table: list[int], num_tokens: int, block_size: int,
                  allocator: BlockAllocator) -> None:
    """
    第 2b 步：让 block_table 里的块够装下 num_tokens 个 token
        不够：调用 allocator.allocate() 分配新块，append 到 block_table 末尾，直到够为止
        够了：什么都不做
    block_table 是 Python 列表，直接在它上面改（不用返回）
    """
    # TODO 用你写的 blocks_needed 算出一共要几个块，再和 len(block_table) 比
    need = blocks_needed(num_tokens, block_size)
    while need > len(block_table):
        block_table.append(allocator.allocate())



class KVCachePool:
    """
    所有请求共用的 k、v 缓存池，开工前一次性开好，之后只往里写、不再申请显存。
        self.k、self.v：形状 (n_layer, num_blocks * block_size, C)
                        第 0 维是层，第 1 维是槽位号，第 2 维是这个 token 的 k（或 v），C = n_embd
    """

    def __init__(self, config: GPTConfig, num_blocks: int, block_size: int,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.block_size = block_size
        self.num_slots = num_blocks * block_size
        shape = (config.n_layer, self.num_slots, config.n_embd)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.allocator = BlockAllocator(num_blocks)

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        第 2c 步：把一批 token 的 k、v 写进第 layer 层的对应槽位
        slots: (N,) 整数张量，每个 token 的槽位号
        k, v:  (N, C)，第 i 行要写进槽位 slots[i]
        """
        # TODO 两行：用"下标张量"一次写进所有槽位（直接改 self.k、self.v，不要新建张量）
        self.k[layer, slots] = k
        self.v[layer, slots] = v

    def read(self, layer: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """取出第 layer 层、这些槽位上的 k、v，返回两个 (len(slots), C) 的张量"""
        return self.k[layer, slots], self.v[layer, slots]
