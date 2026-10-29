# Siddhant Rajhans

**ML Engineer** | Inference optimization, GPU kernels, evaluation infrastructure

MS Machine Learning, Stevens Institute of Technology (Dec 2026). Published researcher ([Google Scholar](https://scholar.google.com/citations?user=icezVIYAAAAJ)).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/siddhant-rajhans/)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=icezVIYAAAAJ)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/@sidrajhans2000/about)
[![X](https://img.shields.io/badge/X-black.svg?logo=X&logoColor=white)](https://twitter.com/RajhansSid)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=YouTube&logoColor=white)](https://www.youtube.com/channel/UCgz7R3pDODLT88P4abzwHcg)

---

### Open Source

**EleutherAI/lm-evaluation-harness**
- [Merged: InfiniteBench long-context evaluation](https://github.com/EleutherAI/lm-evaluation-harness/pull/3662) (+799/-0, 15 files). Long-context LLM eval beyond 100K tokens, now live upstream.

**sgl-project/sglang**
- [Found and patched a stride-aliasing bug](https://github.com/sgl-project/sglang/pull/40240) in the fused MoE LoRA Triton kernel. When shared experts are fused into the routed set, `expert_id * stride_be` overflows the LoRA weight tensor and silently reads the next adapter's weights. Bit-exact repro, deterministic test across dtypes. [Full writeup](https://github.com/siddhant-rajhans/moe-lora-stride-aliasing).

**siddhant-rajhans/cortexlab** (21 stars)
- [Multimodal fMRI brain-encoding toolkit](https://github.com/siddhant-rajhans/cortexlab) extending Meta's TRIBE v2. GPU voxelwise ridge regression via custom Triton kernels (matches scikit-learn RidgeCV to 1e-5). Published on [PyPI](https://pypi.org/project/cortexlab/) and [HuggingFace](https://huggingface.co/siddhant-rajhans).

---

### Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Go](https://img.shields.io/badge/Go-00ADD8?style=for-the-badge&logo=go&logoColor=white)
![Kotlin](https://img.shields.io/badge/Kotlin-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-6C3483?style=for-the-badge&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

---

### Publications

1. CNN-Based Detection Mechanism for Deepfake Image | **IEEE ICCE 2025**
2. Cloud Computing: AI Integration, Threats, Challenges and Security Concerns | **Springer ICRTC 2025**
3. Machine Learning and AI in Cybersecurity: Insights and Solutions | **Springer ICSPN 2023**

---

### GitHub Stats

<p>
  <img src="https://github-readme-stats.vercel.app/api?username=siddhant-rajhans&theme=dark&hide_border=true&include_all_commits=true&count_private=false" alt="GitHub Stats" />
</p>
<p>
  <img src="https://github-readme-streak-stats.herokuapp.com/?user=siddhant-rajhans&theme=dark&hide_border=true" alt="GitHub Streak" />
</p>
<p>
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=siddhant-rajhans&theme=dark&hide_border=true&include_all_commits=true&count_private=false&layout=compact" alt="Top Languages" />
</p>

### Trophies

![](https://github-profile-trophy.vercel.app/?username=siddhant-rajhans&theme=gitdimmed&no-frame=true&no-bg=true&margin-w=4)

---

#### Focus areas

Triton kernels, MoE/LoRA inference, long-context evaluation, multimodal RAG
