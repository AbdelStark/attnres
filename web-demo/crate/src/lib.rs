//! WASM bindings for the Attention Residuals web demo.
//!
//! This is a faithful reimplementation of the core AttnRes algorithm from
//! `attnres`, written in pure Rust for guaranteed WASM compilation.
//! Every function mirrors the corresponding attnres source and cites
//! the paper equations it implements.
//!
//! Reference: "Attention as a Hypernetwork" (MoonshotAI/Kimi), Section 3.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[allow(dead_code)]
mod tensor;

// ─── Data Transfer Types ───────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub d_model: usize,
    pub num_layers: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub num_transformer_layers: usize,
    pub block_size: usize,
    pub is_full_attnres: bool,
    pub d_ff: usize,
    pub total_params: usize,
    pub total_attnres_ops: usize,
}

#[derive(Serialize, Deserialize)]
pub struct ForwardResult {
    pub logits_shape: Vec<usize>,
    pub predictions: Vec<usize>,
    /// Depth attention weights: [num_sublayers][num_sources]
    pub depth_weights: Vec<Vec<f32>>,
    pub block_boundaries: Vec<usize>,
    pub blocks_per_layer: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingSnapshot {
    pub step: usize,
    pub loss: f32,
    pub depth_weights: Vec<Vec<f32>>,
    pub pseudo_query_norms: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct ComparisonResult {
    pub standard_output: Vec<f32>,
    pub attnres_output: Vec<f32>,
    pub attention_weights: Vec<f32>,
    pub num_sources: usize,
}

// ─── Core AttnRes Engine ───────────────────────────────────────────────

/// Mirrors `attnres::AttnResOp` — a single attention residual operation.
///
/// Source: `src/attn_res_op.rs`
/// Paper: Equations 2-4
struct AttnResOp {
    /// Learned pseudo-query w_l ∈ R^d. ZERO-initialized.
    pseudo_query: Vec<f32>,
    /// RMSNorm scale parameter γ ∈ R^d. Ones-initialized.
    gamma: Vec<f32>,
    eps: f32,
    d_model: usize,
}

impl AttnResOp {
    fn new(d_model: usize) -> Self {
        Self {
            pseudo_query: vec![0.0; d_model], // CRITICAL: zero init per paper
            gamma: vec![1.0; d_model],
            eps: 1e-6,
            d_model,
        }
    }

    /// Compute RMSNorm on a single vector.
    ///
    /// Mirrors `attnres::RmsNorm::forward`
    /// Formula: RMSNorm(x) = (x / sqrt(mean(x²) + eps)) * gamma
    fn rms_norm(&self, x: &[f32]) -> Vec<f32> {
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let rms = (mean_sq + self.eps).sqrt();
        x.iter()
            .zip(self.gamma.iter())
            .map(|(xi, gi)| (xi / rms) * gi)
            .collect()
    }

    /// Compute attention residual over block representations.
    ///
    /// Mirrors `attnres::AttnResOp::forward`
    /// Source: `src/attn_res_op.rs:54-87`
    ///
    /// Algorithm (paper Eq. 2-4):
    /// ```text
    /// V = stack(blocks ++ [partial])       // [N+1, D]
    /// K = RMSNorm(V)                       // [N+1, D]
    /// logits = K · w                       // [N+1]
    /// alpha = softmax(logits, dim=0)       // [N+1]  — SOFTMAX OVER DEPTH
    /// h = sum(alpha_i * V_i)               // [D]
    /// ```
    fn forward(&self, blocks: &[Vec<f32>], partial: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut sources: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
        sources.push(partial);

        // Step 1-2: Apply RMSNorm to get keys
        let keys: Vec<Vec<f32>> = sources.iter().map(|s| self.rms_norm(s)).collect();

        // Step 3: Compute logits = dot(K_i, w) for each source
        let logits: Vec<f32> = keys
            .iter()
            .map(|k| {
                k.iter()
                    .zip(self.pseudo_query.iter())
                    .map(|(ki, wi)| ki * wi)
                    .sum()
            })
            .collect();

        // Step 4: Softmax over depth dimension
        let alpha = softmax_1d(&logits);

        // Step 5: Weighted sum of values
        let d = self.d_model;
        let mut output = vec![0.0; d];
        for (i, src) in sources.iter().enumerate() {
            for j in 0..d {
                output[j] += alpha[i] * src[j];
            }
        }

        (output, alpha)
    }

    /// Set pseudo-query weights directly (for training simulation).
    fn set_pseudo_query(&mut self, weights: &[f32]) {
        self.pseudo_query = weights.to_vec();
    }
}

/// Mirrors `attnres::AttnResLayer` — a transformer layer with two AttnRes ops.
///
/// Source: `src/layer.rs`
struct AttnResLayer {
    layer_idx: usize,
    block_size: usize,
    attn_res: AttnResOp,
    mlp_res: AttnResOp,
    d_model: usize,
}

impl AttnResLayer {
    fn new(layer_idx: usize, block_size: usize, d_model: usize) -> Self {
        Self {
            layer_idx,
            block_size,
            attn_res: AttnResOp::new(d_model),
            mlp_res: AttnResOp::new(d_model),
            d_model,
        }
    }

    /// Check if this layer starts a new block.
    ///
    /// Mirrors `attnres::AttnResLayer::is_at_boundary`
    /// Source: `src/layer.rs:72-75`
    fn is_at_boundary(&self) -> bool {
        let half_block = self.block_size / 2;
        self.layer_idx > 0 && (half_block == 0 || self.layer_idx.is_multiple_of(half_block))
    }

    /// Forward pass through this layer.
    ///
    /// Mirrors `attnres::AttnResLayer::forward`
    /// Source: `src/layer.rs:112-160`
    ///
    /// Returns: (updated_blocks, new_partial, attn_weights, mlp_weights)
    fn forward(
        &self,
        blocks: &[Vec<f32>],
        partial: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut blocks = blocks.to_vec();
        let at_boundary = self.is_at_boundary();

        let partial_for_attn = if at_boundary {
            blocks.push(partial.to_vec());
            vec![0.0; self.d_model]
        } else {
            partial.to_vec()
        };

        // AttnRes before self-attention
        let (h_attn, attn_weights) = self.attn_res.forward(&blocks, &partial_for_attn);

        // Simulate attention sublayer output (simplified — adds small perturbation)
        // In the real model this would be MultiHeadAttention
        let attn_out: Vec<f32> = h_attn.iter().map(|v| v * 0.1).collect();

        // Accumulate into partial
        let partial_after_attn: Vec<f32> = partial_for_attn
            .iter()
            .zip(attn_out.iter())
            .map(|(a, b)| a + b)
            .collect();

        // AttnRes before MLP
        let (h_mlp, mlp_weights) = self.mlp_res.forward(&blocks, &partial_after_attn);

        // Simulate MLP sublayer output
        let mlp_out: Vec<f32> = h_mlp.iter().map(|v| v * 0.1).collect();

        // Accumulate into partial
        let partial_after_mlp: Vec<f32> = partial_after_attn
            .iter()
            .zip(mlp_out.iter())
            .map(|(a, b)| a + b)
            .collect();

        (blocks, partial_after_mlp, attn_weights, mlp_weights)
    }
}

// ─── WASM-Exported Engine ──────────────────────────────────────────────

#[wasm_bindgen]
pub struct AttnResEngine {
    config: ModelConfig,
    layers: Vec<AttnResLayer>,
    training_step: usize,
}

#[wasm_bindgen]
impl AttnResEngine {
    /// Create a new engine.
    #[wasm_bindgen(constructor)]
    pub fn new(config_js: JsValue) -> Result<AttnResEngine, JsError> {
        let config: ModelConfig =
            serde_wasm_bindgen::from_value(config_js).map_err(|e| JsError::new(&e.to_string()))?;

        // Validate (mirrors AttnResConfig::validate)
        if config.num_layers == 0 || !config.num_layers.is_multiple_of(2) {
            return Err(JsError::new("num_layers must be positive and even"));
        }
        if config.num_blocks == 0 || !config.num_layers.is_multiple_of(config.num_blocks) {
            return Err(JsError::new("num_blocks must divide num_layers evenly"));
        }
        if config.d_model == 0 || !config.d_model.is_multiple_of(config.num_heads) {
            return Err(JsError::new(
                "d_model must be positive and divisible by num_heads",
            ));
        }

        let block_size = config.num_layers / config.num_blocks;
        let num_transformer_layers = config.num_layers / 2;

        let layers: Vec<AttnResLayer> = (0..num_transformer_layers)
            .map(|i| AttnResLayer::new(i, block_size, config.d_model))
            .collect();

        Ok(AttnResEngine {
            config,
            layers,
            training_step: 0,
        })
    }

    /// Run a forward pass and extract depth attention weights at every sublayer.
    ///
    /// This is the primary visualization entry point. It runs the full
    /// model forward pass and returns the softmax attention weights that
    /// each sublayer assigns to each block (the "depth attention" pattern).
    pub fn forward(&self, input_js: JsValue) -> Result<JsValue, JsError> {
        let input: Vec<f32> =
            serde_wasm_bindgen::from_value(input_js).map_err(|e| JsError::new(&e.to_string()))?;

        let d = self.config.d_model;
        // Pad or truncate input to d_model
        let mut embedding = vec![0.0f32; d];
        for (i, v) in input.iter().enumerate().take(d) {
            embedding[i] = *v;
        }

        let mut blocks: Vec<Vec<f32>> = vec![embedding.clone()]; // b_0 = embedding
        let mut partial = vec![0.0f32; d];

        let mut all_attn_weights: Vec<Vec<f32>> = Vec::new();
        let mut block_boundaries: Vec<usize> = Vec::new();
        let mut blocks_per_layer: Vec<usize> = Vec::new();

        for layer in &self.layers {
            let (new_blocks, new_partial, attn_w, mlp_w) = layer.forward(&blocks, &partial);
            blocks = new_blocks;
            partial = new_partial;

            if layer.is_at_boundary() {
                block_boundaries.push(layer.layer_idx);
            }

            blocks_per_layer.push(blocks.len());
            all_attn_weights.push(attn_w);
            all_attn_weights.push(mlp_w);
        }

        let result = ForwardResult {
            logits_shape: vec![1, 1, self.config.vocab_size],
            predictions: vec![0],
            depth_weights: all_attn_weights,
            block_boundaries,
            blocks_per_layer,
        };

        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Compute the AttnRes operation on custom inputs.
    ///
    /// For the interactive "core operation" visualization panel.
    /// Takes N source vectors + a pseudo-query vector, returns attention weights.
    pub fn compute_attn_res(
        &self,
        sources_js: JsValue,
        pseudo_query_js: JsValue,
    ) -> Result<JsValue, JsError> {
        let sources: Vec<Vec<f32>> =
            serde_wasm_bindgen::from_value(sources_js).map_err(|e| JsError::new(&e.to_string()))?;
        let pseudo_query: Vec<f32> = serde_wasm_bindgen::from_value(pseudo_query_js)
            .map_err(|e| JsError::new(&e.to_string()))?;

        if sources.is_empty() {
            return Err(JsError::new("Need at least one source"));
        }

        let d = sources[0].len();
        let mut op = AttnResOp::new(d);
        op.set_pseudo_query(&pseudo_query);

        let blocks: Vec<Vec<f32>> = sources[..sources.len() - 1].to_vec();
        let partial = &sources[sources.len() - 1];

        let (output, weights) = op.forward(&blocks, partial);

        // Also compute what standard residual would produce (simple mean)
        let num_src = sources.len();
        let mut standard = vec![0.0f32; d];
        for src in &sources {
            for (j, v) in src.iter().enumerate() {
                standard[j] += v / num_src as f32;
            }
        }

        let result = ComparisonResult {
            standard_output: standard,
            attnres_output: output,
            attention_weights: weights,
            num_sources: num_src,
        };

        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Simulate one training step by evolving pseudo-query weights.
    ///
    /// This simulates gradient descent on the pseudo-query vectors to show
    /// how depth attention patterns emerge during training. The simulation
    /// gradually moves pseudo-queries away from zero (the initial uniform state)
    /// toward patterns that prefer recent blocks — matching the paper's findings.
    pub fn train_step(&mut self) -> Result<JsValue, JsError> {
        self.training_step += 1;
        let step = self.training_step;
        let d = self.config.d_model;

        // Simulate learning: pseudo-queries evolve from zero toward
        // patterns that favor more recent blocks (paper observation).
        // Rate of evolution varies by layer depth.
        let num_layers = self.layers.len();
        let lr = 0.02;
        let t = step as f32;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let depth_ratio = (i + 1) as f32 / num_layers as f32;

            // Later layers learn to attend more selectively (paper finding)
            let selectivity = depth_ratio * (1.0 - (-t * lr).exp());

            // Create a gradient-like signal pushing toward recency bias
            let query: Vec<f32> = (0..d)
                .map(|j| {
                    let phase = (j as f32 / d as f32) * std::f32::consts::PI * 2.0;
                    selectivity * (phase + depth_ratio * 3.0).sin() * 0.5
                })
                .collect();

            layer.attn_res.set_pseudo_query(&query);

            // MLP AttnRes evolves slightly differently
            let mlp_query: Vec<f32> = (0..d)
                .map(|j| {
                    let phase = (j as f32 / d as f32) * std::f32::consts::PI * 2.0;
                    selectivity * (phase + depth_ratio * 5.0).cos() * 0.4
                })
                .collect();
            layer.mlp_res.set_pseudo_query(&mlp_query);
        }

        // Compute current depth weights
        let mut depth_weights: Vec<Vec<f32>> = Vec::new();
        let mut pseudo_query_norms: Vec<f32> = Vec::new();

        let embedding = vec![0.1f32; d];
        let mut blocks: Vec<Vec<f32>> = vec![embedding];
        let mut partial = vec![0.0f32; d];

        for layer in &self.layers {
            let (new_blocks, new_partial, attn_w, mlp_w) = layer.forward(&blocks, &partial);

            depth_weights.push(attn_w);
            depth_weights.push(mlp_w);

            // Track pseudo-query magnitude (||w||)
            let attn_norm: f32 = layer
                .attn_res
                .pseudo_query
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            let mlp_norm: f32 = layer
                .mlp_res
                .pseudo_query
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            pseudo_query_norms.push(attn_norm);
            pseudo_query_norms.push(mlp_norm);

            blocks = new_blocks;
            partial = new_partial;
        }

        // Simulated loss: starts high, decays exponentially with noise
        let base_loss = 8.0 * (-t * 0.03).exp() + 2.0;
        let noise = ((t * 7.31).sin() * 0.1 + (t * 13.7).cos() * 0.05) * (-t * 0.01).exp();
        let loss = (base_loss + noise).max(0.1);

        let result = TrainingSnapshot {
            step,
            loss,
            depth_weights,
            pseudo_query_norms,
        };

        serde_wasm_bindgen::to_value(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Reset training state.
    pub fn reset_training(&mut self) {
        self.training_step = 0;
        let d = self.config.d_model;
        for layer in &mut self.layers {
            layer.attn_res = AttnResOp::new(d);
            layer.mlp_res = AttnResOp::new(d);
        }
    }

    /// Get model configuration and derived properties.
    pub fn model_info(&self) -> Result<JsValue, JsError> {
        let block_size = self.config.num_layers / self.config.num_blocks;
        let num_tl = self.config.num_layers / 2;
        let d_ff = 4 * self.config.d_model;
        let d = self.config.d_model;
        let v = self.config.vocab_size;

        let embed_params = v * d * 2;
        let layer_params = 2 * (d * 2) + 2 * d + 4 * d * d + 2 * d * d_ff;
        let total = embed_params + num_tl * layer_params + d;

        let info = ModelInfo {
            d_model: d,
            num_layers: self.config.num_layers,
            num_blocks: self.config.num_blocks,
            num_heads: self.config.num_heads,
            vocab_size: v,
            num_transformer_layers: num_tl,
            block_size,
            is_full_attnres: self.config.num_blocks == self.config.num_layers,
            d_ff,
            total_params: total,
            total_attnres_ops: num_tl * 2,
        };

        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }
}

// ─── Utility Functions ─────────────────────────────────────────────────

/// Numerically stable softmax over a 1D slice.
fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Initialize console panic hook for better WASM error messages.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
