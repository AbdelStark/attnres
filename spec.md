# attnres-rs — Technical Specification

## A Rust Implementation of Attention Residuals using Burn

**Version 0.1.0-draft | March 2026**

---

## 1. Project Overview

attnres-rs is the first open-source Rust implementation of the Attention Residuals paper from MoonshotAI/Kimi. It provides both Full AttnRes and Block AttnRes as drop-in replacement modules for standard residual connections in Transformer architectures, built on the burn deep learning framework.

### 1.1 Goals

1. **Correct:** Differential tests against the PyTorch pseudocode from the paper
2. **Clean:** Idiomatic Rust, well-documented, easy to read and extend
3. **Fast:** Leverages burn's multi-backend support (CUDA, Metal, wgpu, CPU)
4. **Educational:** Clear mapping from paper equations to code
5. **Practical:** Can be used as a library in other burn-based Transformer implementations

### 1.2 Non-Goals (Phase 1)

- Reproducing full scaling law experiments (requires massive compute)
- Pipeline parallelism optimizations (requires multi-node infrastructure)
- Training at Kimi Linear scale (48B parameters)

---

## 2. Core Data Structures

### 2.1 Configuration

```rust
/// Configuration for Attention Residuals.
#[derive(Config, Debug)]
pub struct AttnResConfig {
    /// Hidden dimension (d_model)
    pub d_model: usize,
    /// Total number of layers (L). Each attention + MLP counts as 2 layers.
    pub num_layers: usize,
    /// Number of blocks for Block AttnRes. Set to num_layers for Full AttnRes.
    pub num_blocks: usize,
    /// Epsilon for RMSNorm
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
}

impl AttnResConfig {
    /// Layers per block (S = L / N)
    pub fn block_size(&self) -> usize {
        self.num_layers / self.num_blocks
    }

    /// Whether this is Full AttnRes (N = L) or Block AttnRes (N < L)
    pub fn is_full(&self) -> bool {
        self.num_blocks == self.num_layers
    }
}
```

### 2.2 The AttnRes Module

```rust
/// A single Attention Residual operation.
///
/// Computes softmax attention over block representations using a learned
/// pseudo-query vector w_l. This is the core building block.
///
/// Paper reference: Equation 2-4, Figure 2 (block_attn_res function)
#[derive(Module, Debug)]
pub struct AttnResOp<B: Backend> {
    /// Learned pseudo-query vector w_l ∈ R^d
    /// CRITICAL: Must be initialized to zero for training stability.
    pseudo_query: Param<Tensor<B, 1>>,
    /// RMSNorm applied to values before computing attention logits
    norm: RmsNorm<B>,
}
```

### 2.3 The Block AttnRes Layer

```rust
/// A Transformer layer augmented with Block Attention Residuals.
///
/// Each transformer layer has TWO AttnRes operations:
/// - attn_res: applied before the self-attention sublayer
/// - mlp_res: applied before the MLP sublayer
///
/// Paper reference: Figure 2 (forward function)
#[derive(Module, Debug)]
pub struct AttnResLayer<B: Backend> {
    /// Layer index (0-based)
    layer_idx: usize,
    /// Block size (number of layers per block)
    block_size: usize,

    // AttnRes operations
    /// AttnRes before self-attention
    attn_res: AttnResOp<B>,
    /// AttnRes before MLP
    mlp_res: AttnResOp<B>,

    // Standard transformer sublayers
    /// Pre-attention LayerNorm
    attn_norm: RmsNorm<B>,
    /// Self-attention module
    attn: MultiHeadAttention<B>,
    /// Pre-MLP LayerNorm
    mlp_norm: RmsNorm<B>,
    /// Feed-forward MLP
    mlp: FeedForward<B>,
}
```

### 2.4 Block State

```rust
/// Tracks the state of block accumulation across layers.
///
/// This is passed between layers during the forward pass.
/// It maintains:
/// - Completed block representations (b_0, b_1, ..., b_{n-1})
/// - The current partial sum within the active block (b_n^i)
#[derive(Clone, Debug)]
pub struct BlockState<B: Backend> {
    /// Completed block representations.
    /// blocks[0] = token embedding (b_0 = h_1).
    /// blocks[n] = sum of layer outputs in block n.
    pub blocks: Vec<Tensor<B, 3>>,  // Each: [batch, seq_len, d_model]

    /// Partial sum of layer outputs within the current (incomplete) block.
    /// None at the start of a new block.
    pub partial_block: Option<Tensor<B, 3>>,  // [batch, seq_len, d_model]
}

impl<B: Backend> BlockState<B> {
    /// Initialize with token embeddings as the first block.
    pub fn new(token_embeddings: Tensor<B, 3>) -> Self {
        Self {
            blocks: vec![token_embeddings],
            partial_block: None,
        }
    }
}
```

---

## 3. Core Algorithm

### 3.1 AttnRes Operation (block_attn_res)

This is the heart of the implementation. Maps directly to the pseudocode in Figure 2.

```rust
impl<B: Backend> AttnResOp<B> {
    /// Compute attention residual over block representations.
    ///
    /// # Arguments
    /// * `blocks` - Completed block representations [N tensors of shape [B, T, D]]
    /// * `partial_block` - Current intra-block partial sum [B, T, D]
    ///
    /// # Returns
    /// * Attention-weighted combination of all sources [B, T, D]
    ///
    /// # Algorithm (from paper pseudocode):
    /// ```text
    /// V = stack(blocks + [partial_block])   // [N+1, B, T, D]
    /// K = RMSNorm(V)                        // [N+1, B, T, D]
    /// logits = einsum('d, n b t d -> n b t', w, K)  // [N+1, B, T]
    /// alpha = softmax(logits, dim=0)        // [N+1, B, T]
    /// h = einsum('n b t, n b t d -> b t d', alpha, V) // [B, T, D]
    /// ```
    pub fn forward(
        &self,
        blocks: &[Tensor<B, 3>],
        partial_block: &Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        // Step 1: Stack all sources into value matrix
        // V: [N+1, B, T, D]
        let mut sources: Vec<Tensor<B, 3>> = blocks.to_vec();
        sources.push(partial_block.clone());
        let v = Tensor::stack(sources, 0);  // [N+1, B, T, D]

        // Step 2: Apply RMSNorm to get keys
        // K: [N+1, B, T, D]
        let k = self.norm.forward(v.clone());

        // Step 3: Compute attention logits
        // w: [D] -> broadcast multiply with K -> sum over D
        // logits: [N+1, B, T]
        let w = self.pseudo_query.val();  // [D]
        let logits = (k * w.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            .sum_dim(3);  // [N+1, B, T]

        // Step 4: Softmax over the depth dimension (dim=0)
        let alpha = softmax(logits, 0);  // [N+1, B, T]

        // Step 5: Weighted sum of values
        // alpha: [N+1, B, T] -> [N+1, B, T, 1]
        // v: [N+1, B, T, D]
        // result: [B, T, D]
        let alpha_expanded = alpha.unsqueeze(3);  // [N+1, B, T, 1]
        let weighted = v * alpha_expanded;        // [N+1, B, T, D]
        let h = weighted.sum_dim(0);              // [B, T, D]

        h
    }
}
```

### 3.2 Layer Forward Pass

```rust
impl<B: Backend> AttnResLayer<B> {
    /// Forward pass for a single transformer layer with Block AttnRes.
    ///
    /// Maps directly to the `forward` function in Figure 2 of the paper.
    ///
    /// # Arguments
    /// * `state` - Current block state (completed blocks + partial sum)
    /// * `mask` - Optional attention mask for self-attention
    ///
    /// # Returns
    /// * Updated block state
    pub fn forward(
        &self,
        mut state: BlockState<B>,
        mask: Option<&Tensor<B, 2>>,
    ) -> BlockState<B> {
        let hidden_states = state.partial_block
            .take()
            .unwrap_or_else(|| {
                // At start of a new block, partial_block is None.
                // Use the last completed block as the starting point.
                // (This handles the very first layer where partial_block hasn't been set)
                Tensor::zeros_like(state.blocks.last().unwrap())
            });

        let partial_block = hidden_states.clone();

        // === AttnRes before self-attention ===
        let h = self.attn_res.forward(&state.blocks, &partial_block);

        // === Check block boundary ===
        // block_size counts ATTN + MLP; each transformer layer has 2 sublayers
        // So we check if layer_idx is at a block boundary
        let at_boundary = self.layer_idx > 0
            && self.layer_idx % (self.block_size / 2) == 0;

        let mut partial_block = if at_boundary {
            state.blocks.push(partial_block);
            None
        } else {
            Some(partial_block)
        };

        // === Self-attention sublayer ===
        let normed = self.attn_norm.forward(h);
        let attn_out = self.attn.forward(normed, mask);
        partial_block = Some(match partial_block {
            Some(pb) => pb + attn_out,
            None => attn_out,
        });

        // === AttnRes before MLP ===
        let h = self.mlp_res.forward(
            &state.blocks,
            partial_block.as_ref().unwrap(),
        );

        // === MLP sublayer ===
        let normed = self.mlp_norm.forward(h);
        let mlp_out = self.mlp.forward(normed);
        let partial_block = partial_block.unwrap() + mlp_out;

        state.partial_block = Some(partial_block);
        state
    }
}
```

### 3.3 Full Model

```rust
/// A Transformer model with Attention Residuals.
#[derive(Module, Debug)]
pub struct AttnResTransformer<B: Backend> {
    /// Token embedding layer
    embedding: Embedding<B>,
    /// Stack of transformer layers with AttnRes
    layers: Vec<AttnResLayer<B>>,
    /// Final RMSNorm
    final_norm: RmsNorm<B>,
    /// Output projection (language model head)
    lm_head: Linear<B>,
    /// Configuration
    config: AttnResConfig,
}

impl<B: Backend> AttnResTransformer<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,  // [B, T]
        mask: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 3> {  // [B, T, vocab_size]
        // 1. Token embedding
        let embeddings = self.embedding.forward(input_ids);  // [B, T, D]

        // 2. Initialize block state with embeddings as b_0
        let mut state = BlockState::new(embeddings);

        // 3. Forward through all layers
        for layer in &self.layers {
            state = layer.forward(state, mask);
        }

        // 4. Final aggregation
        // Attend over all blocks + final partial sum for the output
        // (The paper uses the last partial sum directly or a final AttnRes)
        let output = state.partial_block.unwrap();

        // 5. Final norm + LM head
        let normed = self.final_norm.forward(output);
        self.lm_head.forward(normed)
    }
}
```

---

## 4. RMSNorm Implementation

```rust
/// Root Mean Square Layer Normalization.
///
/// Used in AttnRes to normalize keys before computing attention logits.
/// This prevents layers with large-magnitude outputs from dominating.
///
/// RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learnable scale parameter
    gamma: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // x: [N, B, T, D]
        let variance = x.clone().powf_scalar(2.0).mean_dim(3);  // [N, B, T, 1]
        let rms = (variance + self.eps).sqrt();
        let normed = x / rms;
        normed * self.gamma.val().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    }
}
```

---

## 5. Two-Phase Inference (Algorithm 1)

For optimized inference, implement the two-phase strategy from Algorithm 1 of the paper.

```rust
/// Two-phase inference for a single block of layers.
///
/// Phase 1: Parallel inter-block attention (batch all S queries)
/// Phase 2: Sequential intra-block attention + online softmax merge
pub struct TwoPhaseInference<B: Backend> {
    block_layers: Vec<AttnResLayer<B>>,
}

impl<B: Backend> TwoPhaseInference<B> {
    pub fn forward(
        &self,
        blocks: &[Tensor<B, 3>],  // Completed block representations
        initial_state: Tensor<B, 3>,  // Input to this block
    ) -> Vec<Tensor<B, 3>> {
        let s = self.block_layers.len();
        let n = blocks.len();

        // === Phase 1: Parallel inter-block attention ===
        // Batch all S pseudo-queries into a single matrix
        // Q: [S, D]  (stack all pseudo-queries)
        // K, V: [N, B, T, D]  (block representations)
        // Returns: outputs, max values, log-sum-exp for each query

        let queries: Vec<Tensor<B, 1>> = self.block_layers.iter()
            .map(|l| l.attn_res.pseudo_query.val())
            .collect();
        let q_matrix = Tensor::stack(queries, 0);  // [S, D]

        let v_stack = Tensor::stack(blocks.to_vec(), 0);  // [N, B, T, D]
        let k_stack = rms_norm_4d(&v_stack);

        // Batched attention: [S, N, B, T]
        let logits = batched_query_key_dot(q_matrix, k_stack);  // [S, N, B, T]

        // Compute per-query softmax statistics
        let (phase1_outputs, phase1_max, phase1_lse) =
            attention_with_stats(logits, v_stack);

        // === Phase 2: Sequential intra-block + online softmax merge ===
        let mut partial_sum = Tensor::zeros_like(&initial_state);
        let mut layer_inputs = Vec::with_capacity(s);

        for (i, layer) in self.block_layers.iter().enumerate() {
            if i == 0 {
                // First layer: inter-block only (no partial sum yet)
                let h = phase1_outputs[i].clone()
                    / phase1_lse[i].clone();
                layer_inputs.push(h);
            } else {
                // Subsequent layers: merge inter-block + intra-block via online softmax
                let intra_logit = layer.attn_res.pseudo_query.val()
                    .dot(&rms_norm_1d(&partial_sum));
                let (intra_out, intra_max, intra_lse) = (
                    partial_sum.clone() * softmax_scalar(intra_logit),
                    intra_logit,
                    intra_logit.exp(),
                );

                // Online softmax merge (Algorithm 1, line 12)
                let m = phase1_max[i].clone().max(intra_max.clone());
                let numerator = (phase1_max[i].clone() - m.clone()).exp()
                    * phase1_outputs[i].clone()
                    + (intra_max - m.clone()).exp() * intra_out;
                let denominator = (phase1_max[i].clone() - m.clone()).exp()
                    * phase1_lse[i].clone()
                    + (intra_max - m).exp() * intra_lse;
                let h = numerator / denominator;

                layer_inputs.push(h);
            }

            // Execute the layer's actual computation (attn + MLP)
            let attn_out = layer.attn.forward(
                layer.attn_norm.forward(layer_inputs[i].clone()), None
            );
            partial_sum = partial_sum + attn_out;

            // (Repeat for MLP with mlp_res pseudo-query — similar logic)
            // ...
        }

        layer_inputs
    }
}
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    // Test 1: AttnRes with zero-initialized queries produces uniform weights
    #[test]
    fn test_zero_init_uniform_weights() {
        // When pseudo-query is all zeros, dot product with any key is zero
        // softmax([0, 0, ..., 0]) = [1/N, 1/N, ..., 1/N]
        // Result should be the mean of all block representations
    }

    // Test 2: AttnRes with single block reduces to identity (standard residual)
    #[test]
    fn test_single_block_is_identity() {
        // With N=1, AttnRes should behave like standard residual connections
    }

    // Test 3: AttnRes output shape is correct
    #[test]
    fn test_output_shape() {
        // Input: blocks=[3 tensors of [2, 16, 64]], partial=[2, 16, 64]
        // Output: [2, 16, 64]
    }

    // Test 4: Softmax weights sum to 1
    #[test]
    fn test_weights_sum_to_one() {
        // The attention weights alpha should sum to 1 over the depth dimension
    }

    // Test 5: Gradient flows through AttnRes
    #[test]
    fn test_gradient_flow() {
        // Verify that gradients propagate through the AttnRes operation
        // Both to the pseudo-query and to the input blocks
    }

    // Test 6: Block boundary tracking is correct
    #[test]
    fn test_block_boundaries() {
        // With 12 layers and 3 blocks (block_size=4):
        // Boundary at layers 4, 8. Not at 0, 1, 2, 3, 5, 6, 7, 9, 10, 11.
    }

    // Test 7: RMSNorm prevents magnitude domination
    #[test]
    fn test_rmsnorm_normalization() {
        // A block with 10x larger magnitude should not get 10x the attention weight
    }
}
```

### 6.2 Differential Tests (vs PyTorch Reference)

```rust
// Generate reference outputs from PyTorch pseudocode, save as JSON/safetensors
// Load in Rust, compare within tolerance

#[test]
fn test_differential_vs_pytorch() {
    // 1. Load reference inputs from fixtures/
    // 2. Run AttnRes forward pass
    // 3. Compare output to reference within 1e-5 tolerance
    // 4. Compare attention weights to reference
}
```

### 6.3 Property-Based Tests (proptest)

```rust
proptest! {
    // AttnRes output should always be a convex combination of inputs
    // (since softmax weights are non-negative and sum to 1)
    #[test]
    fn output_is_convex_combination(/* random inputs */) {
        // For any set of blocks, the output h should satisfy:
        // min(blocks) <= h <= max(blocks) element-wise
    }

    // AttnRes should be equivariant to permutation of tokens within a batch
    #[test]
    fn token_permutation_equivariance(/* random inputs + permutation */) {
        // Permuting tokens in the input should permute tokens in the output
    }
}
```

### 6.4 Integration Tests

```rust
// Train a small model (4 layers, d=128) on a tiny dataset
// Verify that AttnRes model achieves lower loss than standard residuals
// after N steps of training
#[test]
fn test_training_improvement() {
    let standard_loss = train_model(AttnResConfig { num_blocks: 1, .. });
    let attnres_loss = train_model(AttnResConfig { num_blocks: 4, .. });
    assert!(attnres_loss < standard_loss);
}
```

---

## 7. Module Structure

```
attnres-rs/
├── Cargo.toml
├── README.md
├── LICENSE                          # Apache 2.0
├── src/
│   ├── lib.rs                       # Public API
│   ├── config.rs                    # AttnResConfig
│   ├── attn_res_op.rs               # Core AttnRes operation
│   ├── block_state.rs               # Block state tracking
│   ├── layer.rs                     # AttnResLayer (transformer layer + AttnRes)
│   ├── model.rs                     # Full AttnResTransformer
│   ├── rms_norm.rs                  # RMSNorm implementation
│   ├── two_phase.rs                 # Two-phase inference optimization
│   ├── attention.rs                 # Standard multi-head attention
│   ├── feed_forward.rs              # MLP / feed-forward module
│   └── utils.rs                     # Helper functions
├── tests/
│   ├── unit_tests.rs
│   ├── differential_tests.rs
│   ├── property_tests.rs
│   └── integration_tests.rs
├── fixtures/                        # Reference outputs from PyTorch
│   ├── attn_res_forward.json
│   └── block_state_tracking.json
├── examples/
│   ├── train_tiny.rs                # Train a tiny model with AttnRes
│   ├── compare_residuals.rs         # Compare standard vs AttnRes
│   └── visualize_weights.rs         # Visualize learned attention weights over depth
└── benches/
    └── attn_res_benchmark.rs        # Benchmark AttnRes vs standard residuals
```

---

## 8. Public API

```rust
// Users of attnres-rs interact with these types:
pub use config::AttnResConfig;
pub use attn_res_op::AttnResOp;
pub use block_state::BlockState;
pub use layer::AttnResLayer;
pub use model::AttnResTransformer;

// Example usage:
use attnres_rs::{AttnResConfig, AttnResTransformer};
use burn::prelude::*;

let config = AttnResConfig::new(
    768,   // d_model
    24,    // num_layers (12 transformer blocks × 2)
    8,     // num_blocks (Block AttnRes with 8 blocks)
);

let model = config.init::<MyBackend>(&device);
let output = model.forward(input_ids, Some(&mask));
```
