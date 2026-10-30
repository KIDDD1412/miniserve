# MiniServe

从零实现的 LLM 推理引擎（学习项目）。

## 路线

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 从零手写 GPT-2 推理，与 HuggingFace 输出逐 token 对齐 | 进行中 |
| M1 | KV Cache | |
| M2 | Continuous Batching | |
| M3 | Paged KV Cache | |
| M4 | 迁移到 Qwen2.5 + 与 vLLM 对比 | |

## 环境

```bash
conda activate miniserve
python -m pytest tests
```
