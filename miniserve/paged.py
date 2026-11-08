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
第 2 步：缓存池 —— 真正存 k、v 的大张量，以及"第几个 token 存在池子里哪个槽位"的换算
第 3 步：用块表一次取出所有请求的 k、v，一次算完 decode attention（去掉 packed_attention 的 for 循环）
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
#     第 5 个 token → 在请求的第 5 // 4 = 1 个块里，块内第 5 % 4 = 1 格
#                   → 块表[1] = 2 号块 → 槽位 2 × 4 + 1 = 9
# ---------------------------------------------------------------------------
def slot_mapping(block_table: list[int], positions: list[int], block_size: int) -> list[int]:
    """
    第 2a 步：一个请求里第 positions[i] 个 token，存在池子里的几号槽位
    block_table: 这个请求的块表，比如 [7, 2, 5]
    positions:   要换算的 token 位置（Python 列表）。
                 位置 = 这个 token 是请求里的第几个，每个请求都从 0 开始数，和别的请求、和池子都无关。
                 可以只换算其中一部分，所以不一定从 0 开始：
                     读这个请求的全部历史（第 3a 步）：list(range(L))，即 [0, 1, ..., L-1]
                     只写新 token：已经有 4 个 token、又来了 3 个 → [4, 5, 6]
    返回：        对应的槽位号（Python 列表），和 positions 一一对应。
                 槽位号 = 在池子里的第几格，所有请求共用一个池子，所以不从 0 开始、也不连续。
    例子（block_size = 4）：
        slot_mapping([7, 2, 5], [0, 1, 2, 3, 4], 4) → [28, 29, 30, 31, 8]
        slot_mapping([7, 2, 5], [4, 5, 6], 4)       → [8, 9, 10]
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


# ---------------------------------------------------------------------------
# M3 第 3 步：用块表一次算出所有请求的 decode attention（去掉 packed_attention 的 for 循环）
#
# decode 时每个请求只送 1 个新 token，但它要看自己的全部历史 k、v，而每个请求的历史长度不一样。
# 做法：
#     把每个请求的历史槽位排成一行，短的在后面补齐到同样长度 Lmax，拼成一张 (B, Lmax) 的【槽位表】，
#     再配一张同样形状的 valid 表，标出哪些格子是真 token（True）、哪些是补齐出来的（False）。
#     之后用槽位表一次从池子里取出所有请求的 k、v → (B, Lmax, C)，
#     softmax 之前把补齐的格子填成 -inf，结果就和每个请求单独算完全一样。
#
# 例子（block_size = 4）：
#     请求 0：块表 [7, 2, 5]，一共 10 个 token → 槽位 28 29 30 31 8 9 10 11 20 21
#     请求 1：块表 [1]，     一共 1 个 token  → 槽位 4
#     请求 2：块表 [3, 4]，  一共 5 个 token  → 槽位 12 13 14 15 16
#
#     slot_table（Lmax = 10，补齐的格子填 0）          valid
#       [[28, 29, 30, 31,  8,  9, 10, 11, 20, 21],     [[T, T, T, T, T, T, T, T, T, T],
#        [ 4,  0,  0,  0,  0,  0,  0,  0,  0,  0],      [T, F, F, F, F, F, F, F, F, F],
#        [12, 13, 14, 15, 16,  0,  0,  0,  0,  0]]      [T, T, T, T, T, F, F, F, F, F]]
#
# 补齐的格子为什么填 0？—— 0 号槽位一定存在，按它去池子里取不会越界；
#     取出来的是不相干的数据，但会被 valid 盖掉，不影响结果
# ---------------------------------------------------------------------------
def build_slot_table(block_tables: list[list[int]], seq_lens: list[int], block_size: int,
                     device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    第 3a 步：把所有请求的槽位排成一张补齐的表
    block_tables: 每个请求的块表，比如 [[7, 2, 5], [1], [3, 4]]
    seq_lens:     每个请求现在一共有几个 token（decode 时包括这一轮刚写进池子的新 token），比如 [10, 1, 5]
    返回 (slot_table, valid)，形状都是 (B, Lmax)，B = 请求数，Lmax = max(seq_lens)
        slot_table: 整数张量，第 b 行的前 seq_lens[b] 个数是请求 b 第 0 ~ seq_lens[b]-1 个 token 的槽位号，后面补 0
        valid:      布尔张量，真 token 的格子是 True，补齐的格子是 False
    """
    # TODO ① 算出 Lmax
    #      ② 准备两个空列表 slot_rows、valid_rows，然后对每个请求（for table, L in zip(block_tables, seq_lens)）：
    #           槽位那一行：用你写的 slot_mapping 算出第 0 ~ L-1 个 token 的槽位（长度 L 的列表），后面补 Lmax - L 个 0
    #           valid 那一行：L 个 True，后面 Lmax - L 个 False
    #      ③ 用 torch.tensor(..., device=device) 把两个"列表的列表"分别变成 (B, Lmax) 的张量，返回这两个张量
    Lmax = max(seq_lens)
    slot_rows = []
    valid_rows = []
    for table, L in zip(block_tables, seq_lens):
        slot_row = slot_mapping(table, list(range(L)), block_size) + [0] * (Lmax - L)
        valid_row = [True] * L + [False] * (Lmax - L)
        slot_rows.append(slot_row)
        valid_rows.append(valid_row)
    return torch.tensor(slot_rows, device=device), torch.tensor(valid_rows, device=device)




