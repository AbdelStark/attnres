---
name: debugging
description: Debugging guide for attnres-rs covering numerical issues (NaN, Inf, gradient explosion), tensor shape mismatches, burn framework errors, and algorithm correctness verification. Activate when encountering errors, unexpected outputs, or numerical instability.
prerequisites: cargo test, understanding of spec.md
---

# Debugging

<purpose>
Systematic debugging procedures for the attnres-rs codebase. Covers the most common failure modes in tensor-based ML code and burn-specific issues.
</purpose>

<context>
— Most bugs in this codebase will be: wrong tensor dimensions, incorrect axis for operations, missing normalization, or algorithm misimplementation vs spec.md
— The spec.md contains the canonical algorithm — always compare against it
— burn errors are usually type-level (caught at compile time) or shape-level (caught at runtime)
</context>

<procedure>
General debugging cascade:
1. READ the full error message — Rust and burn errors are usually precise
2. CHECK tensor shapes at the failing operation — add temporary shape assertions
3. COMPARE your code line-by-line against spec.md pseudocode
4. VERIFY axis/dimension arguments — the most common source of subtle bugs
5. TEST with minimal inputs (batch=1, seq_len=1) to simplify
6. PRINT intermediate values if numerical issues suspected
7. If still stuck, write a minimal reproducing test case
</procedure>

<patterns>
<do>
  — Add shape comments on every tensor operation: `// [N+1, B, T, D]`
  — Use `tensor.shape()` assertions liberally during debugging
  — Compare against spec.md equation-by-equation when correctness is in question
  — Test edge cases: N=1 block (should equal standard residual), first layer, last layer
  — Use `Tensor::from_data` with known values for reproducible debugging
</do>
<dont>
  — Don't guess at tensor shapes — print or assert them
  — Don't assume burn operations match PyTorch exactly — check burn docs for axis conventions
  — Don't ignore compiler warnings — they often indicate real bugs in generic code
  — Don't debug on GPU backends — use NdArray first, then verify on GPU
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| NaN in output | Division by zero in RMSNorm | Check eps value; ensure variance computation doesn't produce zero |
| NaN in attention weights | Logits overflow before softmax | Add numerical stability: subtract max before exp (burn's softmax should handle this) |
| Inf in gradients | Exploding gradients through deep AttnRes chain | Verify zero-init on pseudo-queries; check learning rate |
| Wrong output values (not NaN) | Softmax on wrong dimension | AttnRes uses dim=0 (depth), NOT dim for sequence or batch |
| Shape mismatch in stack | Blocks have different batch/seq dims | Ensure all blocks and partial_block have identical [B, T, D] shape |
| "index out of bounds" | Block boundary logic error | Check `layer_idx % (block_size / 2) == 0` and `layer_idx > 0` |
| Compilation error on Backend bounds | Missing trait bound | Add `B: burn::tensor::backend::AutodiffBackend` if using gradients |
| Test passes but training doesn't improve | AttnRes weights stuck at uniform | Check gradient flow to pseudo_query; verify it's wrapped in Param |
</troubleshooting>

<references>
— spec.md §3: Core algorithm pseudocode
— spec.md §4: RMSNorm specification
— spec.md §6: Expected test behaviors
</references>
