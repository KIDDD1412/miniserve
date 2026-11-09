# MiniServe

从零手写的轻量级 LLM 推理引擎（PyTorch）。目前已完成：手写 GPT-2 并与 HuggingFace 逐 token 对齐、KV Cache、支持 static / continuous batching 的调度引擎、Paged KV Cache 与批量 decode attention；每一步都有数值对比检查和 GPU 实测数据。

> 本项目以理解模型结构和推理原理为目标，不是生产级推理框架。

## 项目目标

- 不直接调用 HuggingFace 的模型实现，逐模块手写 GPT-2。
- 使用 HuggingFace 作为参考实现，对权重、logits 和生成结果进行对齐。
- 在正确性的基础上，逐步实现 KV Cache、Continuous Batching 和 Paged KV Cache。
- 通过可复现的基准测试理解不同优化对延迟、吞吐和显存的影响，并用 profiler 定位瓶颈。

## 当前功能

- **GPT-2 模型（M0）**：因果多头自注意力、Pre-LN Block、GELU MLP、权重共享；加载 HuggingFace 权重（处理 `Conv1D` 与 `nn.Linear` 的布局差异）；贪心解码。
- **KV Cache（M1）**：每层缓存历史 token 的 k、v；prefill 一次处理整个 prompt，decode 每步只送 1 个新 token，位置编号从缓存长度接续。
- **压平前向（M2）**：多个请求的 token 首尾相接排成 `(N, C)` 一起过线性层和 MLP，不做 padding；attention 按 `seq_lens` 切开，每个请求用自己的 KV Cache，因此 prefill 和 decode 可以混在同一批；只对每个请求的最后一个 token 计算 logits。
- **调度引擎（M2）**：waiting / running 两个队列，每轮 `step()` 调度 → 一起前向 → 移出结束的请求；支持 static batching 和 continuous batching（iteration-level scheduling）。
- **Paged KV Cache（M3）**：块分配器（空闲块列表 + 已分配集合）、每个请求一张块表、开工前一次开好的缓存池 `(n_layer, 槽位数, C)`；新 token 的 k、v 按槽位直接写进池子，不再 `torch.cat`；请求结束时块马上还回去给别的请求用。
- **批量 decode attention（M3）**：按块表拼出补齐的槽位表 `(B, Lmax)` 和 valid 表，一次从池子里取出所有请求的历史 k、v，用 padding + mask 一次算完所有请求的 attention，没有逐请求循环。
- **PagedEngine（M3）**：在 M2 调度引擎上换成共用缓存池，按空闲块数量决定能否放新请求进来。
- **分层正确性检查**：每一步都和 HuggingFace 或已验证的旧实现逐 token 对比。

## 测速结果

测量条件：RTX 3060 Laptop（6 GB），float32，贪心解码，Windows 11，插电 + 最佳性能模式；计时前先预热，计时前后 `torch.cuda.synchronize()`。

### M1：KV Cache（batch=1，prompt 16 个 token）

| 新 token 数 | 无缓存 | 有缓存 | 提速 | 有缓存每 token |
|---|---|---|---|---|
| 50 | 0.42 s | 0.39 s | 1.1x | 7.7 ms |
| 200 | 1.85 s | 1.42 s | 1.3x | 7.1 ms |
| 500 | 8.40 s | 3.50 s | 2.4x | 7.0 ms |
| 1000 | 35.04 s | 7.26 s | 4.8x | 7.3 ms |

### M2：Continuous Batching 吞吐（`benchmarks/bench_batching.py`）

32 个请求同时到达（离线吞吐），prompt 长度 16~252、max_new_tokens 29~498 随机，共生成 7788 个新 token。吞吐 = 新 token 总数 / 总耗时。

| 跑法 | 耗时 (s) | 轮数 | 吞吐 (tok/s) | 加速比 | 结果和逐个生成一致 | 峰值显存 (MB) |
|---|---|---|---|---|---|---|
| 逐个 generate_cached | 56.9 | - | 137 | 1.00x | - | 561 |
| static（batch=4） | 40.5 | 3037 | 192 | 1.41x | 32/32 | 674 |
| continuous（batch=4） | 35.5 | 2002 | 219 | 1.60x | 32/32 | 667 |
| static（batch=8） | 34.7 | 1776 | 224 | 1.64x | 32/32 | 748 |
| continuous（batch=8） | 31.9 | 1148 | 244 | 1.79x | 32/32 | 779 |
| static（batch=16） | 31.9 | 917 | 244 | 1.78x | 32/32 | 924 |
| continuous（batch=16） | 30.6 | 699 | 255 | 1.86x | 32/32 | 980 |

### M2：瓶颈分析（`benchmarks/bench_step.py`）

加速比远低于预期，continuous 的轮数比 static 少 24%，吞吐却只高 4%。用上表拟合，每轮耗时 ≈ 4.3 ms + 3.55 ms × 请求数：每多一个请求，这一轮就多花 3.55 ms，所以加速上限约 2.1 倍。

用 PyTorch profiler 测 decode 一轮里 GPU 真正执行 kernel 的时间（所有请求都在 decode，上下文约 128）：

| 请求数 | 每轮墙钟 (ms) | 每轮 GPU 计算 (ms) | GPU 忙碌占比 |
|---|---|---|---|
| 1 | 8.2 | 2.3 | 29% |
| 2 | 12.8 | 2.9 | 22% |
| 4 | 20.4 | 3.6 | 17% |
| 8 | 34.8 | 5.5 | 16% |
| 16 | 62.2 | 8.2 | 13% |
| 32 | 121.2 | 13.8 | 11% |

结论：瓶颈是 **CPU 端的 kernel launch 开销**，不是 GPU 计算。`packed_attention` 对每个请求循环计算 attention，每个请求每轮约 130 个小 kernel；每多一个请求，墙钟增加约 3.65 ms，GPU 计算只增加约 0.37 ms。下一步（M3）用 block table 一次算出所有请求的 attention，消除这部分随请求数线性增长的开销。

### M3：Paged KV Cache（`benchmarks/bench_paged.py`）

和 M2 同一批 32 个请求，KV 块大小 16。M2 和 M3 在同一次运行里测，方便直接对比（这次运行整体比上面 M2 那次慢约 5%~10%，所以 M2 的数字和上表略有不同）。

**吞吐**

| 跑法 | 耗时 (s) | 轮数 | 吞吐 (tok/s) | 相对逐个 | 相对 M2（同 batch） | 结果和逐个生成一致 | 峰值显存 (MB) |
|---|---|---|---|---|---|---|---|
| 逐个 generate_cached | 58.5 | - | 133 | 1.00x | - | - | 561 |
| M2 continuous（batch=4） | 36.9 | 2002 | 211 | 1.58x | 1.00x | 32/32 | 664 |
| **M3 paged（batch=4）** | 20.8 | 2002 | 374 | 2.81x | 1.77x | 32/32 | 701 |
| M2 continuous（batch=8） | 35.9 | 1148 | 217 | 1.63x | 1.00x | 32/32 | 783 |
| **M3 paged（batch=8）** | 12.4 | 1148 | 627 | 4.71x | 2.89x | 32/32 | 919 |
| M2 continuous（batch=16） | 34.0 | 699 | 229 | 1.72x | 1.00x | 32/32 | 979 |
| **M3 paged（batch=16）** | 8.9 | 699 | 873 | 6.56x | 3.81x | 32/32 | 1334 |
| M2 continuous（batch=32） | 33.6 | 498 | 232 | 1.74x | 1.00x | 32/32 | 待重测 |
| **M3 paged（batch=32）** | 7.3 | 498 | 1068 | 8.03x | 4.60x | 32/32 | 待重测 |

- M2 的吞吐几乎不随 batch 增长（211 → 232 tok/s），M3 随 batch 稳定增长（374 → 1068 tok/s）；轮数完全相同，差别全在每轮的耗时。
- 峰值显存 M3 更高：测速时缓存池按最坏情况一次开好（batch=16 开了 688 块，约 774 MB，实际同时最多只用了 371 块，约 417 MB）。M2 的 `torch.cat` 缓存按实际长度增长、不预留，但每步都要复制整个缓存。
- batch=32 两行的峰值显存第一次测量时，多算了 batch=16 那次还没释放的缓存池（约 774 MB，测速脚本的 bug，已修复），待重测。

**decode 一轮的时间**（所有请求都在 decode，上下文约 128，每个 B 连续 30 轮取平均）

| 请求数 | M2 墙钟 (ms) | M2 GPU (ms) | M2 GPU 忙碌 | M3 墙钟 (ms) | M3 GPU (ms) | M3 GPU 忙碌 | 墙钟加速 |
|---|---|---|---|---|---|---|---|
| 1 | 9.2 | 2.2 | 23% | 10.4 | 2.2 | 21% | 0.88x |
| 2 | 14.7 | 2.8 | 19% | 11.7 | 2.8 | 24% | 1.25x |
| 4 | 22.7 | 3.6 | 16% | 11.8 | 3.4 | 29% | 1.93x |
| 8 | 40.3 | 5.5 | 14% | 12.5 | 4.7 | 37% | 3.21x |
| 16 | 72.9 | 8.3 | 11% | 12.2 | 6.2 | 51% | 6.00x |
| 32 | 139.9 | 14.0 | 10% | 14.3 | 9.8 | 69% | 9.78x |

拟合：M2 每轮 ≈ 5.9 ms + 4.19 ms × 请求数，M3 每轮 ≈ 11.1 ms + 0.10 ms × 请求数。每多一个请求的边际开销从 4.19 ms 降到 0.10 ms；GPU 忙碌占比随请求数上升（21% → 69%），M2 则是下降（23% → 10%）。代价是固定开销变大：B=1 时 M3 比 M2 慢 12%（每轮要用 Python 拼槽位表、把几个列表转成 GPU 张量，每层还多了读写缓存池的 kernel）。

**KV 缓存的显存利用率**（上面 M3 batch=16 那次，每轮平均：真正存着 token 的位置 / 分出去的位置）

| 分配方式 | 利用率 |
|---|---|
| 按块分配（块大小 16，M3） | 97.4% |
| 每个请求预留 prompt + max_new_tokens（要事先知道生成多长） | 63.0% |
| 每个请求预留模型最大长度 1024 | 27.7% |

按块分配时每个请求只浪费最后一块里没用上的位置；预留一整段连续显存时，请求刚开始的大部分时间里预留的位置都是空的。同样大小的 KV 显存，按块分配能同时放下的请求数约是"按最大长度预留"的 3.5 倍（97.4% / 27.7%）。

## 项目结构

```text
miniserve/
├── miniserve/
│   ├── model.py              # GPT-2 模型、权重加载、贪心解码、KV Cache
│   ├── packed.py             # 多请求压平前向：packed_attention、packed_forward
│   ├── engine.py             # 调度引擎：Request、Engine（static / continuous）
│   ├── paged.py              # 块分配器、缓存池、槽位表、批量 decode attention、paged 前向
│   └── paged_engine.py       # PagedEngine：调度引擎换成共用缓存池
├── tests/                    # 正确性检查脚本
│   ├── check_model.py        # M0：与 HuggingFace 对齐
│   ├── check_kv_cache.py     # M1：KV Cache
│   ├── check_packed.py       # M2：压平前向
│   ├── check_engine.py       # M2：调度引擎
│   ├── check_bench_batching.py
│   ├── check_bench_step.py
│   ├── check_paged.py        # M3：块分配器、缓存池、批量 attention、paged 前向
│   └── check_paged_engine.py # M3：PagedEngine
├── benchmarks/
│   ├── bench_no_cache.py     # M1：无缓存时一次前向耗时 vs 序列长度
│   ├── bench_kv_cache.py     # M1：有无 KV Cache 的生成耗时
│   ├── bench_batching.py     # M2：逐个 / static / continuous 吞吐对比
│   ├── bench_step.py         # M2：一轮 step 的墙钟时间 vs GPU 计算时间
│   └── bench_paged.py        # M3：和 M2 对比吞吐、一轮 decode 的时间、KV 显存利用率
├── exercises/
│   └── tensor_exercises.py   # Tensor 基础练习
├── docs/
│   ├── 八股.md                # 项目涉及的面试知识点
│   └── 踩坑记录.md            # 实现过程中的问题与解决方法
└── README.md
```

## 环境准备

建议使用独立的 Conda 环境：

```bash
conda activate miniserve
python -m pip install torch transformers
```

首次运行需要联网下载 `openai-community/gpt2` 的模型权重和分词器。测速脚本需要 CUDA GPU。

## 运行检查和测速

在项目根目录执行，例如：

```bash
python tests/check_model.py
python tests/check_engine.py
python tests/check_paged.py
python tests/check_paged_engine.py
python benchmarks/bench_batching.py
python benchmarks/bench_paged.py
```

`check_model.py` 中除生成结果外，数值检查使用 `1e-5` 作为相对误差阈值；生成结果要求逐 token 一致。

## 快速使用

```python
from transformers import GPT2TokenizerFast

from miniserve.model import from_pretrained, generate_cached

model_name = "openai-community/gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = from_pretrained(model_name).eval()

prompt = "Hello, I'm a language model,"
inputs = tokenizer(prompt, return_tensors="pt")
output = generate_cached(model, inputs.input_ids, max_new_tokens=20)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 开发路线

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 手写 GPT-2 推理，与 HuggingFace 输出逐 token 对齐 | 已完成 |
| M1 | 实现 KV Cache，并对比无 Cache 推理的性能 | 已完成 |
| M2 | 实现 Continuous Batching，测吞吐并用 profiler 定位瓶颈 | 已完成 |
| M3 | 实现 Paged KV Cache 和批量 decode attention，对比吞吐和显存利用率 | 已完成 |
| M4 | 支持 Qwen2.5，并与 vLLM 进行对比 | 未开始 |

## 当前限制

- 仅支持 GPT-2 模型结构和贪心解码。
- prefill 的 attention 仍按请求逐个计算（每个请求只做一次，影响较小）；decode 已批量计算，但每轮仍有约 11 ms 的固定开销（kernel launch、Python 准备工作），请求数少时比 M2 还慢。
- 批量 decode attention 先按槽位表把 k、v 从池子里取出来，拼成补齐的 `(B, Lmax, C)` 张量（多一次复制），没有实现直接按块表读池子的专用 kernel。
- 块不够时没有抢占 / 换出：只在请求加入时按空闲块数量准入，decode 途中块用完会报错，所以缓存池要按最坏情况开够。
- 尚未提供 HTTP 服务、流式输出和采样策略。
- 基于 PyTorch 通用算子实现，未包含自定义 CUDA/Triton 算子。
- 检查脚本通过 `python tests/check_xxx.py` 运行，尚不是标准 pytest 测试集。

## 学习记录

- [踩坑记录](docs/踩坑记录.md)：Tensor 操作、因果 Mask、设备、权重加载、显存等问题。
- [八股](docs/八股.md)：项目涉及的面试知识点，每题都对应项目里的代码和实测数据。
