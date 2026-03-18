use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::kimi::cache::KimiMlaCache;
use crate::kimi::config::KimiAttentionRuntimeConfig;
use crate::kimi::payload::{load_param_tensor, KimiBaselinePayloadError, KimiDecodedTensor};
use crate::rms_norm::{RmsNorm, RmsNormConfig};

#[derive(Module, Debug)]
struct KimiLowRankQueryProjection<B: Backend> {
    down: Linear<B>,
    up: Linear<B>,
}

#[derive(Module, Debug)]
enum KimiQueryProjection<B: Backend> {
    Direct(Linear<B>),
    LowRank(KimiLowRankQueryProjection<B>),
}

impl<B: Backend> KimiQueryProjection<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Direct(linear) => linear.forward(x),
            Self::LowRank(low_rank) => low_rank.up.forward(low_rank.down.forward(x)),
        }
    }

    fn direct_weight_mut(&mut self) -> Option<&mut Param<Tensor<B, 2>>> {
        match self {
            Self::Direct(linear) => Some(&mut linear.weight),
            Self::LowRank(_) => None,
        }
    }
}

/// MLA full-attention path for the local RFC 0002 baseline.
///
/// This now matches the public Kimi latent-KV surface closely enough for
/// checkpoint-backed slice parity: `kv_a_proj_with_mqa`, `kv_a_layernorm`,
/// `kv_b_proj`, and the public eager attention layout.
#[derive(Module, Debug)]
pub struct KimiMlaAttention<B: Backend> {
    q_proj: KimiQueryProjection<B>,
    kv_a_proj_with_mqa: Linear<B>,
    kv_a_layernorm: RmsNorm<B>,
    kv_b_proj: Linear<B>,
    out_proj: Linear<B>,
    num_attention_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    q_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    use_nope: bool,
}

impl KimiAttentionRuntimeConfig {
    pub fn init_mla<B: Backend>(&self, device: &B::Device) -> KimiMlaAttention<B> {
        let q_head_dim = self.mla_qk_head_dim();
        let q_proj = match self.q_lora_rank {
            Some(rank) => KimiQueryProjection::LowRank(KimiLowRankQueryProjection {
                down: LinearConfig::new(self.hidden_size, rank)
                    .with_bias(false)
                    .init(device),
                up: LinearConfig::new(rank, self.num_attention_heads * q_head_dim)
                    .with_bias(false)
                    .init(device),
            }),
            None => KimiQueryProjection::Direct(
                LinearConfig::new(self.hidden_size, self.num_attention_heads * q_head_dim)
                    .with_bias(false)
                    .init(device),
            ),
        };

        KimiMlaAttention {
            q_proj,
            kv_a_proj_with_mqa: LinearConfig::new(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
            .with_bias(false)
            .init(device),
            kv_a_layernorm: RmsNormConfig::new(self.kv_lora_rank)
                .with_eps(1e-6)
                .init(device),
            kv_b_proj: LinearConfig::new(
                self.kv_lora_rank,
                self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            )
            .with_bias(false)
            .init(device),
            out_proj: LinearConfig::new(
                self.num_attention_heads * self.v_head_dim,
                self.hidden_size,
            )
            .with_bias(false)
            .init(device),
            num_attention_heads: self.num_attention_heads,
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            q_head_dim,
            v_head_dim: self.v_head_dim,
            kv_lora_rank: self.kv_lora_rank,
            use_nope: self.mla_use_nope,
        }
    }
}

impl<B: Backend> KimiMlaAttention<B> {
    pub(crate) fn try_apply_tensor_payload(
        &mut self,
        tensor_name: &str,
        leaf: &str,
        payload: &KimiDecodedTensor,
    ) -> Result<(), KimiBaselinePayloadError> {
        match leaf {
            "q_proj.weight" => {
                let Some(weight) = self.q_proj.direct_weight_mut() else {
                    return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: "low-rank MLA query payload loading remains deferred".to_string(),
                    });
                };
                load_param_tensor(weight, tensor_name, payload)
            }
            "kv_a_proj_with_mqa.weight" => load_param_tensor(
                &mut self.kv_a_proj_with_mqa.weight,
                tensor_name,
                payload,
            ),
            "kv_a_layernorm.weight" => {
                load_param_tensor(self.kv_a_layernorm.gamma_param_mut(), tensor_name, payload)
            }
            "kv_b_proj.weight" => {
                load_param_tensor(&mut self.kv_b_proj.weight, tensor_name, payload)
            }
            "o_proj.weight" => load_param_tensor(&mut self.out_proj.weight, tensor_name, payload),
            _ => Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                tensor_name: tensor_name.to_string(),
                detail: format!("unsupported MLA tensor leaf '{leaf}'"),
            }),
        }
    }

    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        cache: Option<&KimiMlaCache<B>>,
    ) -> (Tensor<B, 3>, KimiMlaCache<B>) {
        let [batch, seq_len, _hidden_size] = hidden.dims();
        let device = hidden.device();

        let q_states = self.q_proj.forward(hidden.clone()).reshape([
            batch,
            seq_len,
            self.num_attention_heads,
            self.q_head_dim,
        ]);
        let q_states = q_states.swap_dims(1, 2);
        let q_pass = q_states.clone().slice([
            0..batch,
            0..self.num_attention_heads,
            0..seq_len,
            0..self.qk_nope_head_dim,
        ]);
        let q_rot = q_states.slice([
            0..batch,
            0..self.num_attention_heads,
            0..seq_len,
            self.qk_nope_head_dim..self.q_head_dim,
        ]);

        let compressed_kv = self.kv_a_proj_with_mqa.forward(hidden);
        let latent_kv = compressed_kv.clone().slice([
            0..batch,
            0..seq_len,
            0..self.kv_lora_rank,
        ]);
        let k_rot = compressed_kv.slice([
            0..batch,
            0..seq_len,
            self.kv_lora_rank..self.kv_lora_rank + self.qk_rope_head_dim,
        ]);
        let kv_states = self.kv_b_proj.forward(self.kv_a_layernorm.forward(latent_kv)).reshape([
            batch,
            seq_len,
            self.num_attention_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ]);
        let kv_states = kv_states.swap_dims(1, 2);
        let k_pass = kv_states.clone().slice([
            0..batch,
            0..self.num_attention_heads,
            0..seq_len,
            0..self.qk_nope_head_dim,
        ]);
        let value_states = kv_states.slice([
            0..batch,
            0..self.num_attention_heads,
            0..seq_len,
            self.qk_nope_head_dim..self.qk_nope_head_dim + self.v_head_dim,
        ]);

        let k_rot = k_rot
            .reshape([batch, seq_len, 1, self.qk_rope_head_dim])
            .swap_dims(1, 2)
            .repeat_dim(1, self.num_attention_heads);
        let query_states = if self.use_nope {
            Tensor::cat(vec![q_pass, q_rot], 3)
        } else {
            Tensor::cat(
                vec![
                    Tensor::zeros(
                        [batch, self.num_attention_heads, seq_len, self.qk_nope_head_dim],
                        &device,
                    ),
                    q_rot,
                ],
                3,
            )
        };
        let current_key_states = if self.use_nope {
            Tensor::cat(vec![k_pass, k_rot], 3)
        } else {
            Tensor::cat(
                vec![
                    Tensor::zeros(
                        [batch, self.num_attention_heads, seq_len, self.qk_nope_head_dim],
                        &device,
                    ),
                    k_rot,
                ],
                3,
            )
        };

        let next_cache = match cache {
            Some(previous) => previous
                .append(current_key_states.clone(), value_states.clone())
                .expect("valid internal MLA append should never fail"),
            None => KimiMlaCache::try_new(
                current_key_states.clone(),
                value_states.clone(),
                self.q_head_dim,
                self.v_head_dim,
            )
            .expect("valid internal MLA cache should never fail"),
        };

        let all_keys = next_cache.keys().clone();
        let all_values = next_cache.values().clone();
        let total_len = next_cache.processed_tokens();
        let prev_len = total_len - seq_len;
        let scores = query_states.matmul(all_keys.swap_dims(2, 3)) / (self.q_head_dim as f64).sqrt();
        let scores = scores
            + mla_chunk_causal_mask::<B>(batch, seq_len, total_len, prev_len, &device)
                .unsqueeze_dim::<4>(1);
        let weights = softmax(scores, 3);
        let output = weights.matmul(all_values).swap_dims(1, 2).reshape([
            batch,
            seq_len,
            self.num_attention_heads * self.v_head_dim,
        ]);

        (self.out_proj.forward(output), next_cache)
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
