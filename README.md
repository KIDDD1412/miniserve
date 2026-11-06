# MiniServe

从零手写的轻量级 LLM 推理引擎（PyTorch）。目前已完成：手写 GPT-2 并与 HuggingFace 逐 token 对齐、KV Cache、支持 static / continuous batching 的调度引擎；每一步都有数值对比检查和 GPU 实测数据。

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

## 项目结构

```text
miniserve/
├── miniserve/
│   ├── model.py              # GPT-2 模型、权重加载、贪心解码、KV Cache
│   ├── packed.py             # 多请求压平前向：packed_attention、packed_forward
│   └── engine.py             # 调度引擎：Request、Engine（static / continuous）
├── tests/                    # 正确性检查脚本
│   ├── check_model.py        # M0：与 HuggingFace 对齐
│   ├── check_kv_cache.py     # M1：KV Cache
│   ├── check_packed.py       # M2：压平前向
│   ├── check_engine.py       # M2：调度引擎
│   ├── check_bench_batching.py
│   └── check_bench_step.py
├── benchmarks/
│   ├── bench_no_cache.py     # M1：无缓存时一次前向耗时 vs 序列长度
│   ├── bench_kv_cache.py     # M1：有无 KV Cache 的生成耗时
│   ├── bench_batching.py     # M2：逐个 / static / continuous 吞吐对比
│   └── bench_step.py         # M2：一轮 step 的墙钟时间 vs GPU 计算时间
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
python benchmarks/bench_batching.py
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
| M3 | 实现 Paged KV Cache 和批量 attention | 未开始 |
| M4 | 支持 Qwen2.5，并与 vLLM 进行对比 | 未开始 |

## 当前限制

- 仅支持 GPT-2 模型结构和贪心解码。
- attention 按请求逐个计算，decode 阶段受 kernel launch 开销限制（见瓶颈分析）。
- KV Cache 用 `torch.cat` 逐步增长，每步都会复制整个缓存。
- 尚未提供 HTTP 服务、流式输出和采样策略。
- 基于 PyTorch 通用算子实现，未包含自定义 CUDA/Triton 算子。
- 检查脚本通过 `python tests/check_xxx.py` 运行，尚不是标准 pytest 测试集。

## 学习记录

- [踩坑记录](docs/踩坑记录.md)：Tensor 操作、因果 Mask、设备、权重加载、显存等问题。
- [八股](docs/八股.md)：项目涉及的面试知识点，每题都对应项目里的代码和实测数据。
