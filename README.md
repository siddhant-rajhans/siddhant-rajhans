# Siddhant Rajhans

ML engineer working on AI security and LLM evaluation, plus the training and inference infrastructure they run on. Grad student at Stevens. I fix bugs in the libraries people train and serve models with, and build tools around brain encoding and model evaluation.

## Merged upstream

| Project | Pull request |
|---|---|
| EleutherAI lm-evaluation-harness | [Add InfiniteBench: long-context evaluation beyond 100K tokens](https://github.com/EleutherAI/lm-evaluation-harness/pull/3662) |
| NVIDIA Megatron-Bridge | [Dequantize fine-grained FP8 weights on import for every bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6882) |
| NVIDIA Megatron-Bridge | [Keep the checkpoint's RoPE in config-only HF export](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6883) |
| NVIDIA Megatron-Bridge | [Pad the vocabulary when conversion TP does not divide it](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6896) |
| NVIDIA TransformerEngine | [Fix RecursionError for ops on CPU-resident QuantizedTensors](https://github.com/NVIDIA/TransformerEngine/pull/3812) |
| NVIDIA TensorRT-LLM | [Read raw logprobs from the logits before penalties and token bans](https://github.com/NVIDIA/TensorRT-LLM/pull/22303) |
| NVIDIA NeMo Agent Toolkit | [Keep red teaming results when the evaluator fails](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2610), [judge parser score recovery](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2612), [target_field validation](https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2620) |
| NVIDIA Dynamo | [Render Hugging Face named chat templates](https://github.com/ai-dynamo/frontend-crates/pull/609) |

In review: vLLM, SGLang, TensorRT-LLM, Megatron-LM, PyTorch, Triton, kvpress, PyRIT.

## Projects

- **[cortexlab](https://github.com/siddhant-rajhans/cortexlab)**: multimodal fMRI brain encoding toolkit built on Meta's TRIBE v2, with streaming inference, brain-alignment benchmarks and cognitive load scoring.
- **[moe-lora-stride-aliasing](https://github.com/siddhant-rajhans/moe-lora-stride-aliasing)**: silent cross-adapter contamination in fused MoE LoRA Triton kernels, with the stride arithmetic and a reproduction.
- **[llm-serving-benchmark](https://github.com/siddhant-rajhans/llm-serving-benchmark)**: vLLM vs SGLang latency, throughput and cost per million tokens under realistic traffic.
- **[bloomberg-sentiment-finbert](https://github.com/siddhant-rajhans/bloomberg-sentiment-finbert)**: Bloomberg's news-sentiment signal against open-source FinBERT on 30 S&P 100 stocks, 2018 to 2026.
- **[blackwell-proxmox-passthrough](https://github.com/siddhant-rajhans/blackwell-proxmox-passthrough)**: running an RTX 5060 Ti for ML inside a Proxmox VM.

## Stack

Python, PyTorch, CUDA and Triton, Go, FastAPI, PostgreSQL, Docker, Linux.

## Contact

[LinkedIn](https://www.linkedin.com/in/siddhant-rajhans/) · [X](https://twitter.com/RajhansSid) · [Medium](https://medium.com/@sidrajhans2000/about)
