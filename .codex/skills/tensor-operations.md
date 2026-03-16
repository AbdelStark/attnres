---
name: tensor-operations
description: Tensor manipulation patterns for the AttnRes algorithm using burn's Tensor API. Activate when implementing forward passes, attention computations, or any operation involving multi-dimensional tensors. Especially relevant for the stack-normalize-attend-aggregate pipeline.
prerequisites: burn crate, understanding of AttnRes algorithm
---

# Tensor Operations

<purpose>
Reference for the specific tensor operations used in attnres. Maps paper equations to burn API calls with explicit shape annotations.
</purpose>

<context>
— burn's Tensor API differs from PyTorch in some conventions
— Key operations: stack, unsqueeze, sum_dim, softmax, element-wise multiply, broadcast
— Dimension ordering matters critically — AttnRes operates over the depth dimension (dim=0 after stacking)
</context>

<procedure>
AttnRes forward pass tensor pipeline:
1. STACK blocks + partial into V: `Tensor::stack(sources, 0)` → [N+1, B, T, D]
2. NORMALIZE to get K: `rms_norm(V)` → [N+1, B, T, D]
3. DOT PRODUCT for logits: `(K * w.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum_dim(3)` → [N+1, B, T]
4. SOFTMAX over depth: `softmax(logits, 0)` → [N+1, B, T]
5. WEIGHT and SUM: `(V * alpha.unsqueeze(3)).sum_dim(0)` → [B, T, D]
</procedure>

<patterns>
<do>
  — Always annotate shapes in comments: `// [B, T, D]`
  — Use `Tensor::stack(vec, dim)` to combine tensors along a new dimension
  — Use `.unsqueeze(dim)` to add broadcast dimensions
  — Use `.sum_dim(dim)` to reduce along a dimension
  — Use `.squeeze(dim)` to remove size-1 dimensions after reduction
  — Use `softmax(tensor, dim)` from burn — it handles numerical stability
  — Clone tensors before consuming them if needed again: `v.clone()`
</do>
<dont>
  — Don't confuse dim=0 (depth/block) with dim=1 (batch) or dim=2 (sequence)
  — Don't manually implement softmax — use burn's built-in (handles max-subtract for stability)
  — Don't forget to unsqueeze the pseudo-query vector for broadcasting: [D] → [1, 1, 1, D]
  — Don't use `.reshape()` when `.unsqueeze()` or `.squeeze()` suffices — reshape can silently reorder data
</dont>
</patterns>

<examples>
Example: AttnRes attention computation

```rust
// Given:
// blocks: Vec<Tensor<B, 3>>  — each [B, T, D], length N
// partial: Tensor<B, 3>      — [B, T, D]
// pseudo_query: Tensor<B, 1> — [D]

// Step 1: Stack all sources
let mut sources = blocks.clone();
sources.push(partial.clone());
let v = Tensor::stack(sources, 0);  // [N+1, B, T, D]

// Step 2: RMSNorm for keys
let k = rms_norm_4d(v.clone());     // [N+1, B, T, D]

// Step 3: Attention logits via dot product
let w = pseudo_query.clone()
    .unsqueeze::<2>(0)   // [1, D]
    .unsqueeze::<3>(0)   // [1, 1, D]
    .unsqueeze::<4>(0);  // [1, 1, 1, D]
let logits = (k * w).sum_dim(3);    // [N+1, B, T]

// Step 4: Softmax over depth (dim=0)
let alpha = softmax(logits, 0);     // [N+1, B, T]

// Step 5: Weighted sum
let alpha_exp = alpha.unsqueeze(3); // [N+1, B, T, 1]
let h = (v * alpha_exp).sum_dim(0); // [B, T, D]
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| "dimension mismatch" on multiply | Broadcasting not aligned | Check unsqueeze adds dimension at correct position |
| sum_dim returns wrong shape | Wrong dimension index | Remember: after stack dim 0=depth, 1=batch, 2=seq, 3=hidden |
| softmax output doesn't sum to 1 | Wrong softmax dimension | Must be dim=0 for depth attention, verify with `.sum_dim(0)` |
| Broadcast error on pseudo_query | Not enough unsqueezes | [D] needs 3 unsqueezes to become [1, 1, 1, D] for [N+1, B, T, D] |
</troubleshooting>

<references>
— spec.md §3.1: AttnResOp forward algorithm with full tensor shapes
— spec.md §4: RMSNorm 4D implementation
— burn Tensor API docs
</references>
