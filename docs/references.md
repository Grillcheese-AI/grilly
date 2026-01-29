Here are the foundational / primary references and links for each item you listed.

## 1. GCU (Growing Cosine Unit)

- Paper: **“Growing Cosine Unit: A Novel Oscillatory Activation Function That Can Speedup Training and Reduce Parameters in Convolutional Neural Networks”** by M. M. Noel (2021). [arxiv](https://arxiv.org/abs/2108.12943)
- arXiv: https://arxiv.org/abs/2108.12943 [arxiv](https://arxiv.org/abs/2108.12943)
- PDF direct: https://arxiv.org/pdf/2108.12943.pdf [arxiv](https://arxiv.org/pdf/2108.12943.pdf)

This paper introduces the GCU activation \(C(z) = z \cos z\) and demonstrates its oscillatory behavior and performance on CNN benchmarks. [arxiv](https://arxiv.org/pdf/2108.12943.pdf)

## 2. RoSwish (Rotating Swish)

- Paper: **“RoSwish: A novel Rotating Swish activation function with adaptive rotation around zero”** (2025). [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/40749309/)
- PubMed record (with full text / publisher link): https://pubmed.ncbi.nlm.nih.gov/40749309/ [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/40749309/)

RoSwish is defined as \(f(x) = (x + \alpha)\,\text{sigmoid}(\beta x) - 0.5\,\alpha\) with learnable \(\alpha, \beta\) to adaptively “rotate” Swish around zero. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/40749309/)

## 3. SwiGLU for Transformer FFNs

SwiGLU is a gated FFN activation introduced as part of large transformer work rather than in a standalone activation-only paper.

- **Origin in PaLM** (widely cited as first major use):  
  - The PaLM paper from Google introduces SwiGLU as a Swish-based gated linear unit in the transformer feed-forward blocks (you’ll find the definition in the feedforward / architecture section of that paper—search within for “SwiGLU”). [github](https://github.com/huggingface/transformers/issues/20403)
  - (You’ll need to open the PaLM arXiv entry directly; the activation is described there. The Emergent Mind article below summarizes it.)

- **Explanatory / secondary reference** (good technical summary):  
  - “SwiGLU Activation in Transformer Models” (Emergent Mind): https://www.emergentmind.com/topics/swiglu-activation [emergentmind](https://www.emergentmind.com/topics/swiglu-activation)

In practice, SwiGLU is of the form \(\text{SwiGLU}(x) = (xW_1) \odot \text{Swish}(xW_2)\) in the FFN, i.e. a GLU variant where the gate uses Swish. [emergentmind](https://www.emergentmind.com/topics/swiglu-activation)

## 4. Fused Linear + Activation Operations

There is no single canonical “activation function” paper here; this is a systems / kernels topic with multiple foundational references. For modern GPU-oriented fused linear+activation (Conv/MatMul + activation) you can cite:

- **DirectML fused activations (API-level reference, good conceptual explanation):**  
  - “Using fused operators to improve performance” (Microsoft DirectML docs): https://learn.microsoft.com/en-us/windows/ai/directml/dml-fused-activations [learn.microsoft](https://learn.microsoft.com/en-us/windows/ai/directml/dml-fused-activations)

  This document defines operator fusion in the context of fusing activations like ReLU into convolution or GEMM to avoid extra memory round-trips. [learn.microsoft](https://learn.microsoft.com/en-us/windows/ai/directml/dml-fused-activations)

- **Fused-tiled execution for deep learning pipelines (research-style reference):**  
  - Y. Xu et al., “Training of Deep Learning Pipelines on Memory-Constrained GPUs by Fused-Tiled Execution” (2022). Open-access article: https://pmc.ncbi.nlm.nih.gov/articles/PMC9302555/ [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9302555/)

  This work describes a fused-tiled execution strategy that fuses multiple operators in forward and backward passes to reduce activation memory and improve performance, which is a more general form of fused linear+activation segments. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9302555/)

For a neuromorphic / custom-kernel context, these two give you a solid combination of conceptual definition (DirectML) and a more formal fused-operator execution model (Xu et al.). [learn.microsoft](https://learn.microsoft.com/en-us/windows/ai/directml/dml-fused-activations)

