use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, silu};

use crate::kimi::cache::KimiKdaCache;
use crate::kimi::config::KimiAttentionRuntimeConfig;
use crate::kimi::payload::{load_param_tensor, KimiBaselinePayloadError, KimiDecodedTensor};
use crate::rms_norm::{RmsNorm, RmsNormConfig};

/// Local KDA scaffold for RFC 0002.
///
/// This path now follows the public Kimi Delta Attention surface closely
/// enough for checkpoint-backed slice parity: bias-free projections,
/// depthwise short convolutions, learned forget-gate parameters, recurrent
/// state updates, and the gated RMS output normalization.
#[derive(Module, Debug)]
pub struct KimiKdaAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    q_conv1d_weight: Param<Tensor<B, 3>>,
    k_conv1d_weight: Param<Tensor<B, 3>>,
    v_conv1d_weight: Param<Tensor<B, 3>>,
    a_log: Param<Tensor<B, 4>>,
    f_a_proj: Linear<B>,
    f_b_proj: Linear<B>,
    dt_bias: Param<Tensor<B, 1>>,
    b_proj: Linear<B>,
    g_a_proj: Linear<B>,
    g_b_proj: Linear<B>,
    output_norm: RmsNorm<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    linear_head_dim: usize,
    value_head_dim: usize,
    short_conv_kernel_size: usize,
}

impl KimiAttentionRuntimeConfig {
    pub fn init_kda<B: Backend>(&self, device: &B::Device) -> KimiKdaAttention<B> {
        let projection_k_size = self.linear_attention_num_heads * self.linear_attention_head_dim;
        let projection_v_size = self.linear_attention_num_heads * self.v_head_dim;

        KimiKdaAttention {
            q_proj: LinearConfig::new(self.hidden_size, projection_k_size)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, projection_k_size)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, projection_v_size)
                .with_bias(false)
                .init(device),
            q_conv1d_weight: Param::from_tensor(Tensor::zeros(
                [
                    projection_k_size,
                    1,
                    self.linear_attention_short_conv_kernel_size,
                ],
                device,
            )),
            k_conv1d_weight: Param::from_tensor(Tensor::zeros(
                [
                    projection_k_size,
                    1,
                    self.linear_attention_short_conv_kernel_size,
                ],
                device,
            )),
            v_conv1d_weight: Param::from_tensor(Tensor::zeros(
                [
                    projection_v_size,
                    1,
                    self.linear_attention_short_conv_kernel_size,
                ],
                device,
            )),
            a_log: Param::from_tensor(Tensor::zeros(
                [1, 1, self.linear_attention_num_heads, 1],
                device,
            )),
            f_a_proj: LinearConfig::new(self.hidden_size, self.linear_attention_head_dim)
                .with_bias(false)
                .init(device),
            f_b_proj: LinearConfig::new(self.linear_attention_head_dim, projection_v_size)
                .with_bias(false)
                .init(device),
            dt_bias: Param::from_tensor(Tensor::zeros([projection_v_size], device)),
            b_proj: LinearConfig::new(self.hidden_size, self.linear_attention_num_heads)
                .with_bias(false)
                .init(device),
            g_a_proj: LinearConfig::new(self.hidden_size, self.linear_attention_head_dim)
                .with_bias(false)
                .init(device),
            g_b_proj: LinearConfig::new(self.linear_attention_head_dim, projection_v_size)
                .with_bias(false)
                .init(device),
            output_norm: RmsNormConfig::new(self.v_head_dim)
                .with_eps(self.rms_norm_eps)
                .init(device),
            out_proj: LinearConfig::new(projection_v_size, self.hidden_size)
                .with_bias(false)
                .init(device),
            num_heads: self.linear_attention_num_heads,
            linear_head_dim: self.linear_attention_head_dim,
            value_head_dim: self.v_head_dim,
            short_conv_kernel_size: self.linear_attention_short_conv_kernel_size,
        }
    }
}

impl<B: Backend> KimiKdaAttention<B> {
    pub(crate) fn try_apply_tensor_payload(
        &mut self,
        tensor_name: &str,
        leaf: &str,
        payload: &KimiDecodedTensor,
    ) -> Result<(), KimiBaselinePayloadError> {
        match leaf {
            "q_proj.weight" => load_param_tensor(&mut self.q_proj.weight, tensor_name, payload),
            "k_proj.weight" => load_param_tensor(&mut self.k_proj.weight, tensor_name, payload),
            "v_proj.weight" => load_param_tensor(&mut self.v_proj.weight, tensor_name, payload),
            "q_conv1d.weight" => load_param_tensor(&mut self.q_conv1d_weight, tensor_name, payload),
            "k_conv1d.weight" => load_param_tensor(&mut self.k_conv1d_weight, tensor_name, payload),
            "v_conv1d.weight" => load_param_tensor(&mut self.v_conv1d_weight, tensor_name, payload),
            "A_log" => load_param_tensor(&mut self.a_log, tensor_name, payload),
            "f_a_proj.weight" => load_param_tensor(&mut self.f_a_proj.weight, tensor_name, payload),
            "f_b_proj.weight" => load_param_tensor(&mut self.f_b_proj.weight, tensor_name, payload),
            "dt_bias" => load_param_tensor(&mut self.dt_bias, tensor_name, payload),
            "b_proj.weight" => load_param_tensor(&mut self.b_proj.weight, tensor_name, payload),
            "g_a_proj.weight" => load_param_tensor(&mut self.g_a_proj.weight, tensor_name, payload),
            "g_b_proj.weight" => load_param_tensor(&mut self.g_b_proj.weight, tensor_name, payload),
            "o_norm.weight" => {
                load_param_tensor(self.output_norm.gamma_param_mut(), tensor_name, payload)
            }
            "o_proj.weight" => load_param_tensor(&mut self.out_proj.weight, tensor_name, payload),
            _ => Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                tensor_name: tensor_name.to_string(),
                detail: format!("unsupported KDA tensor leaf '{leaf}'"),
            }),
        }
    }

    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        cache: Option<&KimiKdaCache<B>>,
    ) -> (Tensor<B, 3>, KimiKdaCache<B>) {
        let [batch, seq_len, _hidden_size] = hidden.dims();

        let q_projected = self
            .q_proj
            .forward(hidden.clone())
            .reshape([batch, seq_len, self.num_heads, self.linear_head_dim])
            .swap_dims(1, 2);
        let k_projected = self
            .k_proj
            .forward(hidden.clone())
            .reshape([batch, seq_len, self.num_heads, self.linear_head_dim])
            .swap_dims(1, 2);
        let v_projected = self
            .v_proj
            .forward(hidden.clone())
            .reshape([batch, seq_len, self.num_heads, self.value_head_dim])
            .swap_dims(1, 2);

        let (q, next_q_conv_state) = self.apply_short_conv(
            q_projected,
            self.q_conv1d_weight.val(),
            cache.map(|state| state.q_conv_state()),
        );
        let (k, next_k_conv_state) = self.apply_short_conv(
            k_projected,
            self.k_conv1d_weight.val(),
            cache.map(|state| state.k_conv_state()),
        );
        let (v, next_v_conv_state) = self.apply_short_conv(
            v_projected,
            self.v_conv1d_weight.val(),
            cache.map(|state| state.v_conv_state()),
        );

        let gate = self
            .f_b_proj
            .forward(self.f_a_proj.forward(hidden.clone()))
            .reshape([batch, seq_len, self.num_heads, self.linear_head_dim])
            .swap_dims(1, 2);
        let gate = self.kda_gate(gate);
        let beta = sigmoid(
            self.b_proj
                .forward(hidden.clone())
                .reshape([batch, seq_len, self.num_heads])
                .swap_dims(1, 2),
        );

        let (recurrent_output, next_recurrent_state) = self.recurrent_kda(
            q,
            k,
            v,
            gate,
            beta,
            cache.map(|state| state.recurrent_state()),
        );

        let output_gate = self
            .g_b_proj
            .forward(self.g_a_proj.forward(hidden))
            .reshape([batch, seq_len, self.num_heads, self.value_head_dim])
            .swap_dims(1, 2);
        let normalized = self
            .output_norm
            .forward(recurrent_output.clone().reshape([
                batch * self.num_heads,
                seq_len,
                self.value_head_dim,
            ]))
            .reshape([batch, self.num_heads, seq_len, self.value_head_dim]);
        let gated = normalized * sigmoid(output_gate);
        let output = self.out_proj.forward(gated.swap_dims(1, 2).reshape([
            batch,
            seq_len,
            self.num_heads * self.value_head_dim,
        ]));

        let next_cache = match cache {
            Some(previous) => previous
                .advance(
                    next_q_conv_state,
                    next_k_conv_state,
                    next_v_conv_state,
                    next_recurrent_state,
                    seq_len,
                )
                .expect("valid internal KDA cache advance should never fail"),
            None => KimiKdaCache::try_new(
                next_q_conv_state,
                next_k_conv_state,
                next_v_conv_state,
                next_recurrent_state,
                seq_len,
                self.short_conv_kernel_size.saturating_sub(1),
                self.linear_head_dim,
                self.value_head_dim,
            )
            .expect("valid internal KDA cache should never fail"),
        };

        (output, next_cache)
    }

    fn apply_short_conv(
        &self,
        projected: Tensor<B, 4>,
        weights: Tensor<B, 3>,
        cache: Option<&Tensor<B, 4>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, heads, seq_len, head_dim] = projected.dims();
        let history_len = cache.map(|state| state.dims()[2]).unwrap_or(0);
        let combined = match cache {
            Some(previous) if history_len > 0 => {
                Tensor::cat(vec![previous.clone(), projected.clone()], 2)
            }
            _ => projected.clone(),
        };
        let kernel_size = self.short_conv_kernel_size;
        let reshaped_weights = weights.reshape([heads, head_dim, kernel_size]);
        let outputs = (0..seq_len)
            .map(|token_offset| {
                let end = history_len + token_offset + 1;
                let start = end.saturating_sub(kernel_size);
                let window = combined
                    .clone()
                    .slice([0..batch, 0..heads, start..end, 0..head_dim]);
                let window_len = end - start;
                let weight_slice = reshaped_weights
                    .clone()
                    .slice([0..heads, 0..head_dim, kernel_size - window_len..kernel_size])
                    .swap_dims(1, 2)
                    .unsqueeze_dim::<4>(0);
                silu(
                    (window * weight_slice)
                        .sum_dim(2)
                        .reshape([batch, heads, head_dim]),
                )
                .unsqueeze_dim::<4>(2)
            })
            .collect::<Vec<_>>();
        let output = Tensor::cat(outputs, 2);
        let next_history = self.updated_conv_state(combined);
        (output, next_history)
    }

    fn updated_conv_state(&self, combined: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, heads, total_tokens, head_dim] = combined.dims();
        let history_len = self.short_conv_kernel_size.saturating_sub(1);
        if history_len == 0 {
            return Tensor::zeros([batch, heads, 0, head_dim], &combined.device());
        }

        let start = total_tokens.saturating_sub(history_len);
        combined.slice([0..batch, 0..heads, start..total_tokens, 0..head_dim])
    }

    fn kda_gate(&self, gate: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, heads, seq_len, head_dim] = gate.dims();
        let a_log = self.a_log.val().reshape([heads]);
        let dt_bias = self
            .dt_bias
            .val()
            .reshape([heads, head_dim])
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2)
            .repeat_dim(0, batch)
            .repeat_dim(2, seq_len);
        let a = a_log
            .exp()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3)
            .repeat_dim(0, batch)
            .repeat_dim(2, seq_len)
            .repeat_dim(3, head_dim);

        -(a * (gate + dt_bias).exp().add_scalar(1.0).log())
    }

    fn recurrent_kda(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        g: Tensor<B, 4>,
        beta: Tensor<B, 3>,
        initial_state: Option<&Tensor<B, 4>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, heads, seq_len, head_dim] = q.dims();
        let scale = (head_dim as f64).sqrt().recip();
        let q = l2_normalize_last_dim(q) * scale;
        let k = l2_normalize_last_dim(k);
        let mut state = match initial_state {
            Some(previous) => previous.clone(),
            None => Tensor::zeros([batch, heads, head_dim, self.value_head_dim], &q.device()),
        };

        let mut outputs = Vec::with_capacity(seq_len);
        for token_offset in 0..seq_len {
            let q_i = q
                .clone()
                .slice([
                    0..batch,
                    0..heads,
                    token_offset..token_offset + 1,
                    0..head_dim,
                ])
                .reshape([batch, heads, head_dim]);
            let k_i = k
                .clone()
                .slice([
                    0..batch,
                    0..heads,
                    token_offset..token_offset + 1,
                    0..head_dim,
                ])
                .reshape([batch, heads, head_dim]);
            let v_i = v
                .clone()
                .slice([
                    0..batch,
                    0..heads,
                    token_offset..token_offset + 1,
                    0..self.value_head_dim,
                ])
                .reshape([batch, heads, self.value_head_dim]);
            let g_i = g
                .clone()
                .slice([
                    0..batch,
                    0..heads,
                    token_offset..token_offset + 1,
                    0..head_dim,
                ])
                .reshape([batch, heads, head_dim]);
            let beta_i = beta
                .clone()
                .slice([0..batch, 0..heads, token_offset..token_offset + 1])
                .reshape([batch, heads]);

            state = state * g_i.exp().unsqueeze_dim::<4>(3);
            let projected_value = (state.clone() * k_i.clone().unsqueeze_dim::<4>(3))
                .sum_dim(2)
                .reshape([batch, heads, self.value_head_dim]);
            let delta_value = (v_i - projected_value) * beta_i.unsqueeze_dim::<3>(2);
            state = state + k_i.unsqueeze_dim::<4>(3) * delta_value.clone().unsqueeze_dim::<4>(2);
            let output_i = (state.clone() * q_i.unsqueeze_dim::<4>(3))
                .sum_dim(2)
                .reshape([batch, heads, self.value_head_dim])
                .unsqueeze_dim::<4>(2);
            outputs.push(output_i);
        }

        (Tensor::cat(outputs, 2), state)
    }
}

fn l2_normalize_last_dim<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    let norm = tensor
        .clone()
        .powf_scalar(2.0)
        .sum_dim(3)
        .add_scalar(1e-6)
        .sqrt();
    tensor / norm
}
