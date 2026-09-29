### Siddhant Rajhans

ML Engineer. Inference optimization, GPU kernels, evaluation infrastructure.

MS Machine Learning, Stevens Institute of Technology (Dec 2026). Published researcher ([Google Scholar](https://scholar.google.com/citations?user=icezVIYAAAAJ)).

#### Open source

**EleutherAI/lm-evaluation-harness**
- [Merged: InfiniteBench long-context evaluation](https://github.com/EleutherAI/lm-evaluation-harness/pull/3662) (+799/-0, 15 files). Long-context LLM eval beyond 100K tokens, now live upstream.

**sgl-project/sglang**
- [Found and patched a stride-aliasing bug](https://github.com/sgl-project/sglang/pull/40240) in the fused MoE LoRA Triton kernel. When shared experts are fused into the routed set, `expert_id * stride_be` overflows the LoRA weight tensor and silently reads the next adapter's weights. Bit-exact repro, deterministic test across dtypes.

**siddhant-rajhans/cortexlab** (21 stars)
- [Multimodal fMRI brain-encoding toolkit](https://github.com/siddhant-rajhans/cortexlab) extending Meta's TRIBE v2. GPU voxelwise ridge regression via custom Triton kernels (matches scikit-learn RidgeCV to 1e-5). Published on [PyPI](https://pypi.org/project/cortexlab/) and [HuggingFace](https://huggingface.co/siddhant-rajhans).

#### Publications

1. CNN-Based Detection Mechanism for Deepfake Image | IEEE ICCE 2025
2. Cloud Computing: AI Integration, Threats, Challenges and Security Concerns | Springer ICRTC 2025
3. Machine Learning and AI in Cybersecurity: Insights and Solutions | Springer ICSPN 2023

#### Focus areas

Triton kernels, MoE/LoRA inference, long-context evaluation, multimodal RAG
