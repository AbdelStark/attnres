use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softplus;

use crate::kimi::cache::KimiKdaCache;
use crate::kimi::config::KimiAttentionRuntimeConfig;

/// Local KDA scaffold for RFC 0002.
///
/// This is a slow, explicit linear-attention implementation with short-conv
/// state so the repository has typed KDA execution and cache separation. It is
/// not an optimized `fla-core` kernel binding and is not presented as a
/// checkpoint-parity claim.
#[derive(Debug)]
pub struct KimiKdaAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    linear_head_dim: usize,
    value_head_dim: usize,
    short_conv_kernel_size: usize,
}

impl KimiAttentionRuntimeConfig {
    pub fn init_kda<B: Backend>(&self, device: &B::Device) -> KimiKdaAttention<B> {
        KimiKdaAttention {
            q_proj: LinearConfig::new(
                self.hidden_size,
                self.linear_attention_num_heads * self.linear_attention_head_dim,
            )
            .init(device),
            k_proj: LinearConfig::new(
                self.hidden_size,
                self.linear_attention_num_heads * self.linear_attention_head_dim,
            )
            .init(device),
            v_proj: LinearConfig::new(
                self.hidden_size,
                self.linear_attention_num_heads * self.v_head_dim,
            )
            .init(device),
            out_proj: LinearConfig::new(
                self.linear_attention_num_heads * self.v_head_dim,
                self.hidden_size,
            )
            .init(device),
            num_heads: self.linear_attention_num_heads,
            linear_head_dim: self.linear_attention_head_dim,
            value_head_dim: self.v_head_dim,
            short_conv_kernel_size: self.linear_attention_short_conv_kernel_size,
        }
    }
}

impl<B: Backend> KimiKdaAttention<B> {
    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        cache: Option<&KimiKdaCache<B>>,
    ) -> (Tensor<B, 3>, KimiKdaCache<B>) {
        let [batch, seq_len, _hidden_size] = hidden.dims();
        let q = softplus(
            self.q_proj
                .forward(hidden.clone())
                .reshape([batch, seq_len, self.num_heads, self.linear_head_dim])
                .swap_dims(1, 2),
            1.0,
        )
        .add_scalar(1e-6);
        let k = softplus(
            self.k_proj
                .forward(hidden.clone())
                .reshape([batch, seq_len, self.num_heads, self.linear_head_dim])
                .swap_dims(1, 2),
            1.0,
        )
        .add_scalar(1e-6);
        let v = self
            .v_proj
            .forward(hidden)
            .reshape([batch, seq_len, self.num_heads, self.value_head_dim])
            .swap_dims(1, 2);

        let kv = k.clone().unsqueeze_dim::<5>(4) * v.clone().unsqueeze_dim::<5>(3);
        let mut prefix_kv = kv.cumsum(2);
        let mut prefix_k = k.clone().cumsum(2);

        if let Some(previous) = cache {
            prefix_kv = prefix_kv + previous.recurrent_state().clone().unsqueeze_dim::<5>(2);
            prefix_k = prefix_k + previous.running_key_sum().clone().unsqueeze_dim::<4>(2);
        }

        let numerator = (q.clone().unsqueeze_dim::<5>(4) * prefix_kv.clone())
            .sum_dim(3)
            .reshape([batch, self.num_heads, seq_len, self.value_head_dim]);
        let denominator = (q.clone() * prefix_k.clone())
            .sum_dim(3)
            .reshape([batch, self.num_heads, seq_len, 1])
            .add_scalar(1e-6);
        let linear_out = numerator / denominator;
        let conv_out = self.short_conv_context(k.clone(), cache);
        let merged = linear_out + conv_out;

        let output = self
            .out_proj
            .forward(merged.clone().swap_dims(1, 2).reshape([
                batch,
                seq_len,
                self.num_heads * self.value_head_dim,
            ]));

        let updated_conv_state = self.updated_conv_state(k, cache);
        let updated_recurrent_state = prefix_kv.slice([
            0..batch,
            0..self.num_heads,
            seq_len - 1..seq_len,
            0..self.linear_head_dim,
            0..self.value_head_dim,
        ]);
        let updated_running_key = prefix_k.slice([
            0..batch,
            0..self.num_heads,
            seq_len - 1..seq_len,
            0..self.linear_head_dim,
        ]);
        let next_cache = match cache {
            Some(previous) => previous
                .advance(
                    updated_conv_state,
                    updated_recurrent_state.reshape([
                        batch,
                        self.num_heads,
                        self.linear_head_dim,
                        self.value_head_dim,
                    ]),
                    updated_running_key.reshape([batch, self.num_heads, self.linear_head_dim]),
                    seq_len,
                )
                .expect("valid internal KDA cache advance should never fail"),
            None => KimiKdaCache::try_new(
                updated_conv_state,
                updated_recurrent_state.reshape([
                    batch,
                    self.num_heads,
                    self.linear_head_dim,
                    self.value_head_dim,
                ]),
                updated_running_key.reshape([batch, self.num_heads, self.linear_head_dim]),
                seq_len,
                self.short_conv_kernel_size.saturating_sub(1),
                self.linear_head_dim,
                self.value_head_dim,
            )
            .expect("valid internal KDA cache should never fail"),
        };

        (output, next_cache)
    }

    fn short_conv_context(
        &self,
        projected_keys: Tensor<B, 4>,
        cache: Option<&KimiKdaCache<B>>,
    ) -> Tensor<B, 4> {
        let [batch, heads, seq_len, head_dim] = projected_keys.dims();
        if self.short_conv_kernel_size <= 1 {
            return Tensor::zeros([batch, heads, seq_len, head_dim], &projected_keys.device());
        }

        let prefix_len = cache.map(|state| state.conv_state().dims()[2]).unwrap_or(0);
        let combined = match cache {
            Some(previous) if prefix_len > 0 => Tensor::cat(
                vec![previous.conv_state().clone(), projected_keys.clone()],
                2,
            ),
            _ => projected_keys.clone(),
        };

        let windows = (0..seq_len)
            .map(|token_offset| {
                let end = prefix_len + token_offset + 1;
                let start = end.saturating_sub(self.short_conv_kernel_size);
                combined
                    .clone()
                    .slice([0..batch, 0..heads, start..end, 0..head_dim])
                    .mean_dim(2)
            })
            .collect();

        Tensor::cat(windows, 2)
    }

    fn updated_conv_state(
        &self,
        projected_keys: Tensor<B, 4>,
        cache: Option<&KimiKdaCache<B>>,
    ) -> Tensor<B, 4> {
        let [batch, heads, _seq_len, head_dim] = projected_keys.dims();
        let history_len = self.short_conv_kernel_size.saturating_sub(1);

        if history_len == 0 {
            return Tensor::zeros([batch, heads, 0, head_dim], &projected_keys.device());
        }

        let combined = match cache {
            Some(previous) if previous.conv_state().dims()[2] > 0 => {
                Tensor::cat(vec![previous.conv_state().clone(), projected_keys], 2)
            }
            _ => projected_keys,
        };

        let total_tokens = combined.dims()[2];
        let start = total_tokens.saturating_sub(history_len);
        combined.slice([0..batch, 0..heads, start..total_tokens, 0..head_dim])
    }
}
