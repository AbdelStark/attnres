# Attention Residuals — Paper Digest

**Authors:** Kimi Team (Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, + 30 co-authors)
**Organization:** MoonshotAI (Kimi)
**Date:** 2026
**Paper:** [Attention_Residuals.pdf](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf)

---

## Core Problem

Standard residual connections in Transformers accumulate all layer outputs with fixed unit weights. Unrolling the recurrence `h_l = h_{l-1} + f_{l-1}(h_{l-1})` reveals that every layer receives an equally-weighted sum of all prior outputs. This creates three problems:

1. **Hidden-state growth:** With PreNorm, magnitudes grow as O(L) with depth, progressively diluting each layer's relative contribution.
2. **No selective access:** Different layer types (attention vs MLP) receive the same aggregated state despite potentially benefiting from different weightings.
3. **Irreversible information loss:** Information lost through uniform aggregation cannot be recovered by deeper layers.

## Key Insight: Duality of Time and Depth

The authors observe a formal duality between depth-wise accumulation in residual networks and sequential recurrence in RNNs. Just as Transformers improved upon RNNs by replacing recurrence over time with attention, AttnRes replaces fixed accumulation over depth with learned attention.

The transition: **standard residuals perform depth-wise linear attention; AttnRes generalizes them to depth-wise softmax attention.**

## Full Attention Residuals

Replace the fixed accumulation with softmax attention over preceding layer outputs:

```
h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
```

Where:
- `v_0 = h_1` (token embedding)
- `v_i = f_i(h_i)` for i ≥ 1 (layer outputs)
- `α_{i→l}` are softmax attention weights computed via a single learned pseudo-query `w_l ∈ R^d` per layer
- Keys are RMSNorm'd values: `k_i = RMSNorm(v_i)`
- `α_{i→l} = softmax(w_l^T · k_i)` over all i

**Cost:** O(L²d) arithmetic, O(Ld) memory. Since depth L is modest (< 1000), this is computationally feasible.

**Parameters added:** Only one d-dimensional vector `w_l` per layer + one RMSNorm per layer. Negligible fraction of total parameters.

**Critical initialization:** All pseudo-query vectors must be initialized to **zero**. This ensures initial attention weights are uniform (equivalent to equal-weight average), preventing training volatility.

## Block Attention Residuals

For large-scale training with pipeline parallelism, Full AttnRes's O(Ld) communication overhead becomes impractical. Block AttnRes partitions L layers into N blocks:

**Intra-block:** Standard residual accumulation (sum layer outputs within block)
```
b_n = Σ_{j ∈ B_n} f_j(h_j)
```

**Inter-block:** Softmax attention over N block-level representations + token embedding
```
h_l = Σ α_{i→l} · [b_0, b_1, ..., b_{n-1}, b_n^i]
```

Where `b_n^i` is the partial sum within the current block up to layer i.

**Efficiency:** Memory and communication drop from O(Ld) to O(Nd). With N ≈ 8 blocks, this recovers most of Full AttnRes's gains.

## The Algorithm (Block AttnRes Forward Pass)

For each transformer layer within a block:

1. **Inter-block attention:** Compute softmax attention over completed block representations + current partial sum using the layer's pseudo-query `w_l`
2. **Check block boundary:** If at boundary, append partial sum to block list, reset partial sum
3. **Self-attention:** Apply attention sublayer to the AttnRes output
4. **Update partial sum:** Add attention output to partial sum
5. **Inter-block attention again:** Compute AttnRes before MLP (separate pseudo-query)
6. **MLP:** Apply MLP sublayer
7. **Update partial sum:** Add MLP output to partial sum

Each transformer layer has TWO AttnRes operations: one before attention, one before MLP. Each has its own pseudo-query and RMSNorm.

## Infrastructure Optimizations

### Two-Phase Inference Strategy

Because pseudo-queries are decoupled from forward computation, all S queries within a block can be batched:

- **Phase 1 (parallel):** Batch all S pseudo-queries against cached block representations. Single matrix multiply. Returns outputs + softmax statistics (max, log-sum-exp).
- **Phase 2 (sequential):** For each layer, compute intra-block attention against evolving partial sum, then merge with Phase 1 outputs via online softmax.

Result: per-layer memory I/O of only (N/S + 5)d reads + 2d writes. With typical N=8, S=16: 5.5d total, vs 3d for standard residuals (only 83% overhead).

### Cross-Stage Caching (Pipeline Parallelism)

Each physical stage caches blocks received during earlier virtual stages. Subsequent transitions transmit only incremental blocks. Reduces peak per-transition cost from O(C) to O(P), a V× improvement.

### Memory-Efficient Prefilling

For long-context sequences, block representations are sharded along the sequence dimension across tensor-parallel devices. Combined with chunked prefill (16K chunks), overhead drops to < 0.3 GB per device.

## Results

### Scaling Laws (5 model sizes, 194M-528M activated params)

Fitted power-law curves:
- Baseline: `L = 1.891 × C^{-0.057}`
- Block AttnRes: `L = 1.870 × C^{-0.058}`
- Full AttnRes: `L = 1.865 × C^{-0.057}`

AttnRes consistently outperforms across all compute budgets. Block AttnRes matches baseline trained with **1.25× more compute**.

### Downstream (Kimi Linear 48B / 3B activated, 1.4T tokens)

| Benchmark | Baseline | AttnRes | Delta |
|-----------|----------|---------|-------|
| MMLU | 73.5 | 74.6 | +1.1 |
| GPQA-Diamond | 36.9 | 44.4 | **+7.5** |
| BBH | 76.3 | 78.0 | +1.7 |
| Math | 53.5 | 57.1 | +3.6 |
| HumanEval | 59.1 | 62.2 | +3.1 |
| MBPP | 72.0 | 73.9 | +1.9 |
| CMMLU | 82.0 | 82.9 | +0.9 |
| C-Eval | 79.6 | 82.5 | +2.9 |

Largest gains on multi-step reasoning (GPQA-Diamond: +7.5) and math (Math: +3.6).

### Training Dynamics

- **Output magnitudes:** Baseline grows monotonically with depth. AttnRes confines growth within blocks, yielding bounded periodic pattern.
- **Gradient distribution:** Baseline has disproportionately large gradients in early layers. AttnRes produces substantially more uniform gradient distribution.
- **Validation loss:** AttnRes achieves consistently lower loss, with gap widening during decay phase.

### Ablation Highlights

- Content-dependent selection (softmax) > fixed weights > uniform average
- Zero initialization of pseudo-queries is critical for stability
- N ≈ 8 blocks recovers most of Full AttnRes's gains
- RMSNorm on keys prevents large-magnitude layers from dominating
