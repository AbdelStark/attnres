use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::kimi::cache::KimiMlaCache;
use crate::kimi::config::KimiAttentionRuntimeConfig;

#[derive(Debug)]
enum KimiQueryProjection<B: Backend> {
    Direct(Linear<B>),
    LowRank { down: Linear<B>, up: Linear<B> },
}

impl<B: Backend> KimiQueryProjection<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Direct(linear) => linear.forward(x),
            Self::LowRank { down, up } => up.forward(down.forward(x)),
        }
    }
}

/// MLA full-attention path for the local RFC 0002 baseline.
///
/// The local scaffold preserves the published projection and cache surface, but
/// it does not claim reference-parity rotary handling or optimized kernels.
#[derive(Debug)]
pub struct KimiMlaAttention<B: Backend> {
    q_proj: KimiQueryProjection<B>,
    kv_down: Linear<B>,
    k_up: Linear<B>,
    v_up: Linear<B>,
    out_proj: Linear<B>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    qk_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    mla_use_nope: bool,
    kv_repeat_factor: usize,
}

impl KimiAttentionRuntimeConfig {
    pub fn init_mla<B: Backend>(&self, device: &B::Device) -> KimiMlaAttention<B> {
        let qk_head_dim = self.mla_qk_head_dim();
        let q_proj = match self.q_lora_rank {
            Some(rank) => KimiQueryProjection::LowRank {
                down: LinearConfig::new(self.hidden_size, rank).init(device),
                up: LinearConfig::new(rank, self.num_attention_heads * qk_head_dim).init(device),
            },
            None => KimiQueryProjection::Direct(
                LinearConfig::new(self.hidden_size, self.num_attention_heads * qk_head_dim)
                    .init(device),
            ),
        };

        KimiMlaAttention {
            q_proj,
            kv_down: LinearConfig::new(self.hidden_size, self.kv_lora_rank).init(device),
            k_up: LinearConfig::new(self.kv_lora_rank, self.num_key_value_heads * qk_head_dim)
                .init(device),
            v_up: LinearConfig::new(
                self.kv_lora_rank,
                self.num_key_value_heads * self.v_head_dim,
            )
            .init(device),
            out_proj: LinearConfig::new(
                self.num_attention_heads * self.v_head_dim,
                self.hidden_size,
            )
            .init(device),
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            qk_head_dim,
            qk_nope_head_dim: self.qk_nope_head_dim,
            v_head_dim: self.v_head_dim,
            mla_use_nope: self.mla_use_nope,
            kv_repeat_factor: self.kv_repeat_factor(),
        }
    }
}

impl<B: Backend> KimiMlaAttention<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        cache: Option<&KimiMlaCache<B>>,
    ) -> (Tensor<B, 3>, KimiMlaCache<B>) {
        let [batch, seq_len, _hidden_size] = hidden.dims();
        let device = hidden.device();

        let q = self
            .apply_nope_policy(self.q_proj.forward(hidden.clone()).reshape([
                batch,
                seq_len,
                self.num_attention_heads,
                self.qk_head_dim,
            ]))
            .swap_dims(1, 2);

        let kv_latent = self.kv_down.forward(hidden);
        let current_k = self
            .apply_nope_policy(self.k_up.forward(kv_latent.clone()).reshape([
                batch,
                seq_len,
                self.num_key_value_heads,
                self.qk_head_dim,
            ]))
            .swap_dims(1, 2);
        let current_v = self
            .v_up
            .forward(kv_latent)
            .reshape([batch, seq_len, self.num_key_value_heads, self.v_head_dim])
            .swap_dims(1, 2);

        let next_cache = match cache {
            Some(previous) => previous
                .append(current_k.clone(), current_v.clone())
                .expect("valid internal MLA append should never fail"),
            None => KimiMlaCache::try_new(
                current_k.clone(),
                current_v.clone(),
                self.qk_head_dim,
                self.v_head_dim,
            )
            .expect("valid internal MLA cache should never fail"),
        };

        let all_k = next_cache.keys().clone();
        let all_v = next_cache.values().clone();
        let total_len = next_cache.processed_tokens();
        let prev_len = total_len - seq_len;

        let k = self.expand_kv_heads(all_k);
        let v = self.expand_kv_heads(all_v);

        let scores = q.matmul(k.swap_dims(2, 3)) / (self.qk_head_dim as f64).sqrt();
        let scores = scores
            + mla_chunk_causal_mask::<B>(batch, seq_len, total_len, prev_len, &device)
                .unsqueeze_dim::<4>(1);
        let weights = softmax(scores, 3);
        let output = weights.matmul(v).swap_dims(1, 2).reshape([
            batch,
            seq_len,
            self.num_attention_heads * self.v_head_dim,
        ]);

        (self.out_proj.forward(output), next_cache)
    }

    fn apply_nope_policy(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.mla_use_nope || self.qk_nope_head_dim == 0 {
            return tensor;
        }

        let [batch, seq_len, heads, _] = tensor.dims();
        let rope = tensor.clone().slice([
            0..batch,
            0..seq_len,
            0..heads,
            self.qk_nope_head_dim..self.qk_head_dim,
        ]);
        Tensor::cat(
            vec![
                Tensor::zeros(
                    [batch, seq_len, heads, self.qk_nope_head_dim],
                    &tensor.device(),
                ),
                rope,
            ],
            3,
        )
    }

    fn expand_kv_heads(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.kv_repeat_factor == 1 {
            tensor
        } else {
            tensor.repeat_dim(1, self.kv_repeat_factor)
        }
    }
}

fn mla_chunk_causal_mask<B: Backend>(
    batch_size: usize,
    query_len: usize,
    total_len: usize,
    prefix_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let query_positions =
        Tensor::<B, 1, Int>::arange(prefix_len as i64..(prefix_len + query_len) as i64, device)
            .unsqueeze_dim::<2>(1);
    let key_positions =
        Tensor::<B, 1, Int>::arange(0..total_len as i64, device).unsqueeze_dim::<2>(0);
    let disallowed = query_positions.greater_equal(key_positions).bool_not();

    Tensor::<B, 2>::zeros([query_len, total_len], device)
        .mask_fill(disallowed, -1e9)
        .unsqueeze_dim::<3>(0)
        .repeat_dim(0, batch_size)
}
