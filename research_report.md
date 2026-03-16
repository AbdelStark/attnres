# attnres-rs: Research Report

## Strategic Analysis of the Attention Residuals Opportunity

---

## 1. Why This Paper Matters

### 1.1 The Core Innovation

Attention Residuals is one of those papers that makes you think "why didn't anyone do this before?" The insight is simple and elegant: residual connections have been doing depth-wise linear attention since 2015 (He et al.). AttnRes upgrades them to depth-wise softmax attention. Same transition that transformed sequence modeling (RNN to Transformer), now applied to depth.

The results speak for themselves: a 1.25x compute advantage as a drop-in replacement with less than 2% inference overhead. In an era where every fraction of a percent matters for training efficiency, this is significant.

### 1.2 Why a Rust Implementation Is Valuable

There is currently NO implementation of Attention Residuals outside of Kimi's internal codebase. The GitHub repo (MoonshotAI/Attention-Residuals) contains only the paper PDF and a README with pseudocode. No runnable code.

A Rust implementation using burn would be:

1. **The first open implementation.** Anyone who wants to experiment with AttnRes currently has to reimplement from pseudocode. attnres-rs would be the reference.
2. **Production-grade from day one.** Burn compiles to multiple backends (CUDA, Metal, WASM, CPU). This means AttnRes running everywhere, not just NVIDIA GPUs.
3. **A showcase for Rust ML.** The burn ecosystem needs high-profile implementations of cutting-edge papers to prove Rust is viable for ML research. AttnRes is perfect because the algorithm is clean, well-specified, and the pseudocode maps naturally to Rust traits.
4. **Credibility builder.** Implementing a fresh paper from a major lab (Kimi raised $1B+, competes with OpenAI in China) demonstrates you can read, understand, and implement frontier ML research. This is exactly the signal that AMI Labs, Skild AI, or any ML hiring manager looks for.

### 1.3 Timing

The paper was released days ago (March 2026). The repo has 41 stars and 2 forks. There is zero competition for the "first Rust implementation" title. Every day that passes increases the risk someone else does it.

The Kimi team has strong social media presence. A quote-tweet of their announcement with "I built a Rust implementation of Attention Residuals" will get engagement from the ML community AND the Rust community simultaneously.

---

## 2. Technical Feasibility Assessment

### 2.1 Algorithm Complexity

AttnRes is architecturally simple. The core operation is:

1. Stack block representations into a value matrix V (N+1 tensors)
2. Apply RMSNorm to get keys K
3. Dot product of learned pseudo-query w with K to get logits
4. Softmax over the depth dimension
5. Weighted sum of V

This is just a single-head attention operation over the depth dimension with a learned query. No multi-head. No KV projection. No causal masking. It's simpler than standard self-attention.

The Block AttnRes variant adds:
- Block boundary tracking (counter)
- Partial sum accumulation (running sum of layer outputs within a block)
- Two AttnRes operations per transformer layer (one before attention, one before MLP)
- Online softmax merge for the two-phase inference strategy

**Assessment: Highly feasible in Rust. The algorithm is well-specified, the pseudocode is clear, and the operations (einsum, softmax, RMSNorm) are standard tensor operations available in burn.**

### 2.2 Burn Framework Suitability

burn (github.com/tracel-ai/burn) is the right choice:

- Multi-backend: ndarray (CPU), wgpu (cross-platform GPU), CUDA (NVIDIA), candle (HuggingFace)
- Tensor operations: all required ops exist (matmul, softmax, layer_norm, einsum equivalent)
- Autodiff: burn has automatic differentiation, needed for training
- Module system: burn's Module derive macro works like PyTorch's nn.Module
- Serialization: supports safetensors for weight loading/saving

**One consideration:** burn's einsum support may require manual decomposition into matmul/reshape operations. The paper's pseudocode uses `torch.einsum('d, n b t d -> n b t', ...)` which in burn would be implemented as a broadcast multiply + sum.

### 2.3 Tract for Inference Optimization

tract (github.com/sonos/tract) is a Rust neural network inference library optimized for production deployment. It could be useful for:
- Optimized inference on CPU/edge devices
- ONNX model import (load PyTorch-trained models)
- Quantized inference (INT8)

However, for the initial implementation, burn alone is sufficient. Tract becomes relevant when optimizing the inference path for deployment. Include it as a Phase 2 optimization target.

### 2.4 What We Can and Cannot Do

**Can do (Phase 1):**
- Implement Full AttnRes and Block AttnRes as burn Modules
- Train a small Transformer with AttnRes on a toy dataset (WikiText, CIFAR-10 text)
- Differential testing against the PyTorch pseudocode
- Benchmark: compare training loss of standard residuals vs AttnRes on same model/data

**Can do (Phase 2):**
- Two-phase inference optimization (Algorithm 1 from the paper)
- Online softmax merge
- WASM compilation for browser demo
- Safetensors weight loading (load Kimi's weights if released)

**Cannot do (resource constraints):**
- Full scaling law reproduction (requires thousands of GPU hours)
- Training at Kimi Linear scale (48B params, 1.4T tokens)
- Pipeline parallelism optimizations (requires multi-node cluster)

**The goal is not to reproduce the full paper. It's to provide a clean, correct, well-tested implementation that the community can build on.**

---

## 3. Competitive Analysis

### 3.1 Existing Implementations

| Implementation | Language | Status | Quality |
|---------------|----------|--------|---------|
| Kimi internal | Python (PyTorch) | Private | Production (48B model) |
| MoonshotAI/Attention-Residuals | N/A | Paper + pseudocode only | No runnable code |
| Community reimplementations | None found | N/A | N/A |

**There is zero competition.** We would be first.

### 3.2 Related Rust ML Projects

| Project | Stars | What | Relevance |
|---------|-------|------|-----------|
| burn | 9.5K | Rust deep learning framework | Our framework |
| candle | 16K | HuggingFace Rust ML | Alternative backend for burn |
| llama.cpp | 78K | C++ LLM inference | Proved non-Python ML can go viral |
| mistral.rs | 5K | Rust LLM inference | Proved Rust ML gets stars |
| dfdx | 1.8K | Rust tensor library | Earlier Rust ML attempt |

**Pattern:** Rust ML projects consistently get stars because the Rust community is hungry for ML tools and the ML community is curious about Rust. A well-executed paper implementation will get attention from both.

---

## 4. Impact Projections

### 4.1 GitHub Stars Potential

Based on comparable projects:
- mistral.rs (Rust LLM inference): 5K stars
- llm (Rust LLM inference): 6K stars  
- candle (HuggingFace Rust ML): 16K stars

attnres-rs is more niche (specific paper implementation, not a general tool), so realistic targets:
- Month 1: 200-500 stars
- Month 3: 500-1,500 stars (if paper gains traction)
- Month 6: 1,000-3,000 stars (if adopted by researchers)

### 4.2 Credibility Signal

Implementing a fresh paper from a top Chinese AI lab in Rust signals:
1. You can read and implement frontier ML research
2. You're comfortable with both Rust and ML
3. You move fast (first implementation of a brand-new paper)
4. You care about production quality (Rust, not a Jupyter notebook hack)

This directly supports your AMI Labs application (shows ML depth), your WorldForge thesis (shows you can implement novel architectures), and your "Karpathy of world models" positioning (shows you teach by building).

### 4.3 Community Engagement Opportunity

The Kimi/MoonshotAI team is active on X. A quote-tweet saying "I built the first Rust implementation of your Attention Residuals paper" is likely to get engagement from:
- The Kimi team (they'll appreciate the implementation and retweet)
- The Rust ML community (hungry for novel implementations)
- The broader ML community (curious about the paper itself)
- burn framework maintainers (they love showcases)

---

## 5. Recommendation

**Build this. Ship it fast. It's a weekend-to-week project for the core implementation, and the timing is perfect.**

Priority: Ship after awesome-world-models and jepa-rs initial commit, but before the larger projects (WorldForge, worldplay). This is a quick credibility win that compounds with everything else.
