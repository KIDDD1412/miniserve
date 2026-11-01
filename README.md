# MiniServe

基于 PyTorch 的轻量级 LLM 推理学习项目。当前已完成 GPT-2 模型、HuggingFace 权重加载和贪心解码，并通过分层数值对比验证实现正确性。

> 本项目以理解模型结构和推理原理为目标，不是生产级推理框架。

## 项目目标

- 不直接调用 HuggingFace 的模型实现，逐模块手写 GPT-2。
- 使用 HuggingFace 作为参考实现，对权重、logits 和生成结果进行对齐。
- 在正确性的基础上，逐步实现 KV Cache、连续批处理和 Paged KV Cache。
- 通过可复现的基准测试理解不同优化对延迟、吞吐和显存的影响。

## 当前功能

- 因果多头自注意力：QKV 投影、多头拆分、因果 Mask 和输出投影。
- GPT-2 Transformer Block：Pre-LayerNorm、残差连接和 GELU MLP。
- 完整 GPT-2：Token/Position Embedding、12 层 Block、LM Head 和权重共享。
- HuggingFace 权重加载：处理 `Conv1D` 与 `nn.Linear` 的权重布局差异。
- 贪心解码：逐步选择概率最高的 token，生成指定数量的新 token。
- 分层正确性检查：覆盖 Attention、MLP、Block、完整模型、权重加载和文本生成。

## 项目结构

```text
miniserve/
├── miniserve/
│   └── model.py              # GPT-2 模型、权重加载与贪心解码
├── tests/
│   └── check_model.py        # 与 HuggingFace 对齐的检查脚本
├── benchmarks/
│   └── bench_no_cache.py     # 无 KV Cache 性能测试（开发中）
├── exercises/
│   └── tensor_exercises.py   # Tensor 基础练习
├── docs/
│   └── 踩坑记录.md            # 实现过程中的问题与解决方法
└── README.md
```

## 环境准备

建议使用独立的 Conda 环境：

```bash
conda activate miniserve
python -m pip install torch transformers
```

首次运行需要联网下载 `openai-community/gpt2` 的模型权重和分词器。

## 运行正确性检查

在项目根目录执行：

```bash
python tests/check_model.py
```

脚本会依次检查：

1. Attention 输出及因果性。
2. MLP 输出及 GPT-2 所需的 tanh 近似 GELU。
3. Transformer Block 输出。
4. 完整模型的参数量和 logits。
5. HuggingFace 权重是否完整、正确地加载。
6. 贪心生成结果是否与 HuggingFace 逐 token 一致。

除生成结果外，数值检查使用 `1e-5` 作为相对误差阈值。

## 快速使用

```python
from transformers import GPT2TokenizerFast

from miniserve.model import from_pretrained, generate

model_name = "openai-community/gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = from_pretrained(model_name).eval()

prompt = "Hello, I'm a language model,"
inputs = tokenizer(prompt, return_tensors="pt")
output = generate(model, inputs.input_ids, max_new_tokens=20)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

当前 `generate` 使用贪心解码，每生成一个 token 都会重新计算完整序列，尚未启用 KV Cache。

## 开发路线

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 手写 GPT-2 推理，与 HuggingFace 输出逐 token 对齐 | 已完成 |
| M1 | 实现 KV Cache，并对比无 Cache 推理的性能 | 进行中 |
| M2 | 实现 Continuous Batching | 未开始 |
| M3 | 实现 Paged KV Cache | 未开始 |
| M4 | 支持 Qwen2.5，并与 vLLM 进行对比 | 未开始 |

## 当前限制

- 仅支持 GPT-2 模型结构和贪心解码。
- 尚未实现 KV Cache，长序列生成会产生大量重复计算。
- 尚未提供连续批处理、HTTP 服务、流式输出和并发请求管理。
- 基于 PyTorch 通用算子实现，未包含自定义 CUDA/Triton 算子。
- 当前检查脚本需通过 `python tests/check_model.py` 运行，尚不是标准 pytest 测试集。

## 学习记录

实现过程中遇到的 Tensor 操作、因果 Mask、设备和权重加载问题，记录在[踩坑记录](docs/踩坑记录.md)中。
