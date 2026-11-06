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

第 1 步（这个文件现在的内容）：块分配器 —— 只管"哪些块空着、分给谁、收回来"，还不碰张量
检查："D:\\Program Files\\Anaconda\\envs\\miniserve\\python.exe" tests/check_paged.py
"""
from collections import deque


# ---------------------------------------------------------------------------
# M3 第 1a 步：装下 n 个 token 需要几个块
# ---------------------------------------------------------------------------
def blocks_needed(num_tokens, block_size):
    """
    装下 num_tokens 个 token 需要几个块（向上取整：最后一个块没装满也要占一整块）
    例：block_size = 4 时，4 个 token 要 1 块，5 个 token 要 2 块，0 个 token 要 0 块
    """
    # TODO 一行，只用整数运算（不要用 float）
    return (num_tokens + block_size -1) // block_size + 1


# ---------------------------------------------------------------------------
# M3 第 1b 步：块分配器
# ---------------------------------------------------------------------------
class BlockAllocator:
    """管理缓存池里 num_blocks 个块（编号 0 ~ num_blocks-1）的分配和回收"""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free_blocks = deque(range(num_blocks))   # 空闲的块号，一开始全部空闲
        self.used = set()                             # 已经分配出去的块号

    @property
    def num_free(self):
        return len(self.free_blocks)

    def can_allocate(self, n):
        """现在还能不能一次分出 n 个块"""
        # TODO 一行
        return self.num_free >= n

    def allocate(self):
        """分配 1 个块，返回它的块号"""
        # TODO ① 没有空闲块时：raise RuntimeError("没有空闲块了")
        #      ② 从 free_blocks 取出最前面的一个块号（popleft）
        #      ③ 记进 self.used，再返回它
        if self.num_free == 0:
            raise RuntimeError("无空闲块")
        num = self.free_blocks.popleft()
        self.used.add(num)
        return num

    def free(self, block_id):
        """把一个块还回池子"""
        # TODO ① block_id 不在 self.used 里（没分配过，或者已经还过一次了）：
        #         raise ValueError(f"块 {block_id} 没有被分配，不能释放")
        #      ② 从 self.used 里去掉，再放回 free_blocks 的末尾
        if block_id not in self.used:
            raise ValueError(f"块 {block_id} 没有被分配，不能释放")
        self.used.remove(block_id)
        self.free_blocks.append(block_id)
