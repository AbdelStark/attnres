use burn::prelude::*;

use crate::attn_res_op::AttnResOp;
use crate::kimi::attention::{KimiKdaAttention, KimiMlaAttention};
use crate::kimi::attn_res_state::KimiAttnResBlockState;
use crate::kimi::cache::{KimiCacheError, KimiDecodeCache};
use crate::kimi::config::{KimiArtifactConfig, KimiAttnResConfig, KimiAttnResConfigError};
use crate::kimi::mlp::KimiDenseMlp;
use crate::kimi::moe::KimiSparseMoe;
use crate::kimi::payload::{load_param_tensor, KimiBaselinePayloadError, KimiDecodedTensor};
use crate::kimi::schedule::{KimiAttentionLayerKind, KimiFeedForwardLayerKind};
use crate::rms_norm::{RmsNorm, RmsNormConfig};

#[derive(Debug)]
enum KimiAttnResAttentionBlock<B: Backend> {
    Mla(Box<KimiMlaAttention<B>>),
    Kda(Box<KimiKdaAttention<B>>),
}

impl<B: Backend> KimiAttnResAttentionBlock<B> {
    fn try_apply_tensor_payload(
        &mut self,
        tensor_name: &str,
        leaf: &str,
        payload: &KimiDecodedTensor,
    ) -> Result<(), KimiBaselinePayloadError> {
        match self {
            Self::Mla(attention) => attention.try_apply_tensor_payload(tensor_name, leaf, payload),
            Self::Kda(attention) => attention.try_apply_tensor_payload(tensor_name, leaf, payload),
        }
    }
}

#[derive(Debug)]
enum KimiAttnResFeedForwardBlock<B: Backend> {
    Dense(Box<KimiDenseMlp<B>>),
    SparseMoe(Box<KimiSparseMoe<B>>),
}

impl<B: Backend> KimiAttnResFeedForwardBlock<B> {
    fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden),
            Self::SparseMoe(moe) => moe.forward(hidden),
        }
    }

    fn try_apply_tensor_payload(
        &mut self,
        tensor_name: &str,
        leaf: &str,
        payload: &KimiDecodedTensor,
    ) -> Result<(), KimiBaselinePayloadError> {
        match self {
            Self::Dense(mlp) => mlp.try_apply_tensor_payload(tensor_name, leaf, payload),
            Self::SparseMoe(moe) => moe.try_apply_tensor_payload(tensor_name, leaf, payload),
        }
    }
}

/// One RFC 0004 AttnRes-Kimi decoder layer.
#[derive(Debug)]
pub struct KimiAttnResDecoderLayer<B: Backend> {
    layer_idx: usize,
    block_size: usize,
    attention_kind: KimiAttentionLayerKind,
    feed_forward_kind: KimiFeedForwardLayerKind,
    attn_res: AttnResOp<B>,
    input_norm: RmsNorm<B>,
    attention: KimiAttnResAttentionBlock<B>,
    mlp_res: AttnResOp<B>,
    post_attention_norm: RmsNorm<B>,
    feed_forward: KimiAttnResFeedForwardBlock<B>,
}

impl KimiArtifactConfig {
    pub fn try_init_attn_res_layer<B: Backend>(
        &self,
        layer_idx: usize,
        num_blocks: usize,
        device: &B::Device,
    ) -> Result<KimiAttnResDecoderLayer<B>, KimiAttnResConfigError> {
        self.try_attn_res_config(num_blocks)?
            .try_init_layer(layer_idx, device)
    }

    pub fn init_attn_res_layer<B: Backend>(
        &self,
        layer_idx: usize,
        num_blocks: usize,
        device: &B::Device,
    ) -> KimiAttnResDecoderLayer<B> {
        self.try_init_attn_res_layer(layer_idx, num_blocks, device)
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

impl KimiAttnResConfig {
    pub fn try_init_layer<B: Backend>(
        &self,
        layer_idx: usize,
        device: &B::Device,
    ) -> Result<KimiAttnResDecoderLayer<B>, KimiAttnResConfigError> {
        let attn_res = self.try_attn_res_core_config()?;
        attn_res.try_validate_layer_idx(layer_idx)?;

        let layer = self
            .baseline
            .layer_schedule
            .try_layer(layer_idx)
            .map_err(crate::kimi::config::KimiArtifactConfigError::from)?;

        let attention = match layer.attention_kind {
            KimiAttentionLayerKind::FullAttention => {
                KimiAttnResAttentionBlock::Mla(Box::new(self.baseline.attention.init_mla(device)))
            }
            KimiAttentionLayerKind::LinearAttentionKda => {
                KimiAttnResAttentionBlock::Kda(Box::new(self.baseline.attention.init_kda(device)))
            }
        };
        let feed_forward = match layer.feed_forward_kind {
            KimiFeedForwardLayerKind::DenseMlp => {
                KimiAttnResFeedForwardBlock::Dense(Box::new(self.baseline.dense_mlp.init(device)))
            }
            KimiFeedForwardLayerKind::SparseMoe => KimiAttnResFeedForwardBlock::SparseMoe(
                Box::new(self.baseline.sparse_moe.init(device)),
            ),
        };

        Ok(KimiAttnResDecoderLayer {
            layer_idx,
            block_size: attn_res.block_size(),
            attention_kind: layer.attention_kind,
            feed_forward_kind: layer.feed_forward_kind,
            attn_res: attn_res.init_op(device),
            input_norm: RmsNormConfig::new(self.hidden_size())
                .with_eps(self.rms_norm_eps())
                .init(device),
            attention,
            mlp_res: attn_res.init_op(device),
            post_attention_norm: RmsNormConfig::new(self.hidden_size())
                .with_eps(self.rms_norm_eps())
                .init(device),
            feed_forward,
        })
    }

    pub fn init_layer<B: Backend>(
        &self,
        layer_idx: usize,
        device: &B::Device,
    ) -> KimiAttnResDecoderLayer<B> {
        self.try_init_layer(layer_idx, device)
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

impl<B: Backend> KimiAttnResDecoderLayer<B> {
    pub(crate) fn try_apply_tensor_payload(
        &mut self,
        tensor_name: &str,
        remainder: &str,
        payload: &KimiDecodedTensor,
    ) -> Result<(), KimiBaselinePayloadError> {
        match remainder {
            "input_layernorm.weight" => {
                load_param_tensor(self.input_norm.gamma_param_mut(), tensor_name, payload)
            }
            "post_attention_layernorm.weight" => load_param_tensor(
                self.post_attention_norm.gamma_param_mut(),
                tensor_name,
                payload,
            ),
            _ => {
                if let Some(leaf) = remainder.strip_prefix("self_attn.") {
                    return self
                        .attention
                        .try_apply_tensor_payload(tensor_name, leaf, payload);
                }

                if let Some(leaf) = remainder.strip_prefix("mlp.") {
                    return match self.feed_forward_kind {
                        KimiFeedForwardLayerKind::DenseMlp => self
                            .feed_forward
                            .try_apply_tensor_payload(tensor_name, leaf, payload),
                        KimiFeedForwardLayerKind::SparseMoe => {
                            Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                                tensor_name: tensor_name.to_string(),
                                detail: "dense MLP tensor cannot be applied to sparse MoE layer"
                                    .to_string(),
                            })
                        }
                    };
                }

                if let Some(leaf) = remainder.strip_prefix("block_sparse_moe.") {
                    return match self.feed_forward_kind {
                        KimiFeedForwardLayerKind::DenseMlp => {
                            Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                                tensor_name: tensor_name.to_string(),
                                detail: "sparse MoE tensor cannot be applied to dense MLP layer"
                                    .to_string(),
                            })
                        }
                        KimiFeedForwardLayerKind::SparseMoe => self
                            .feed_forward
                            .try_apply_tensor_payload(tensor_name, leaf, payload),
                    };
                }

                Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                    tensor_name: tensor_name.to_string(),
                    detail: format!("unsupported decoder-layer tensor leaf '{remainder}'"),
                })
            }
        }
    }

    fn attn_sublayer_idx(&self) -> usize {
        self.layer_idx * 2
    }

    fn mlp_sublayer_idx(&self) -> usize {
        self.attn_sublayer_idx() + 1
    }

    fn starts_new_block_before_sublayer(&self, sublayer_idx: usize) -> bool {
        sublayer_idx > 0 && sublayer_idx.is_multiple_of(self.block_size)
    }

    pub(crate) fn starts_new_block_before_attn(&self) -> bool {
        self.starts_new_block_before_sublayer(self.attn_sublayer_idx())
    }

    pub(crate) fn starts_new_block_before_mlp(&self) -> bool {
        self.starts_new_block_before_sublayer(self.mlp_sublayer_idx())
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn attention_kind(&self) -> KimiAttentionLayerKind {
        self.attention_kind
    }

    pub fn feed_forward_kind(&self) -> KimiFeedForwardLayerKind {
        self.feed_forward_kind
    }

    pub fn uses_moe(&self) -> bool {
        self.feed_forward_kind == KimiFeedForwardLayerKind::SparseMoe
    }

    pub fn is_at_boundary(&self) -> bool {
        self.starts_new_block_before_attn()
    }

    pub fn attn_res_ops(&self) -> (&AttnResOp<B>, &AttnResOp<B>) {
        (&self.attn_res, &self.mlp_res)
    }

    pub fn forward_attn_sublayer(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.input_norm.forward(hidden);
        match &self.attention {
            KimiAttnResAttentionBlock::Mla(attention) => attention.forward(normed, None).0,
            KimiAttnResAttentionBlock::Kda(attention) => attention.forward(normed, None).0,
        }
    }

    pub fn forward_mlp_sublayer(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.post_attention_norm.forward(hidden);
        self.feed_forward.forward(normed)
    }

    pub fn forward(&self, mut state: KimiAttnResBlockState<B>) -> KimiAttnResBlockState<B> {
        state.validate_or_panic();

        let current_partial = state.take_partial_block();
        let h = self
            .attn_res
            .forward_optional_partial(state.blocks(), current_partial.as_ref());

        let mut partial_for_attn =
            current_partial.unwrap_or_else(|| Tensor::zeros_like(state.last_completed_block()));
        if self.starts_new_block_before_attn() {
            state.push_completed_block(partial_for_attn.clone());
            partial_for_attn = Tensor::zeros_like(state.last_completed_block());
        }

        let partial_after_attn = partial_for_attn + self.forward_attn_sublayer(h);

        let h = self
            .mlp_res
            .forward_optional_partial(state.blocks(), Some(&partial_after_attn));

        let mut partial_for_mlp = partial_after_attn;
        if self.starts_new_block_before_mlp() {
            state.push_completed_block(partial_for_mlp.clone());
            partial_for_mlp = Tensor::zeros_like(state.last_completed_block());
        }

        state.set_partial_block(partial_for_mlp + self.forward_mlp_sublayer(h));
        state
    }

    pub fn try_forward_cached(
        &self,
        mut state: KimiAttnResBlockState<B>,
        cache: &mut KimiDecodeCache<B>,
    ) -> Result<KimiAttnResBlockState<B>, KimiCacheError> {
        state.validate_or_panic();

        let current_partial = state.take_partial_block();
        let h = self
            .attn_res
            .forward_optional_partial(state.blocks(), current_partial.as_ref());

        let mut partial_for_attn =
            current_partial.unwrap_or_else(|| Tensor::zeros_like(state.last_completed_block()));
        if self.starts_new_block_before_attn() {
            state.push_completed_block(partial_for_attn.clone());
            partial_for_attn = Tensor::zeros_like(state.last_completed_block());
        }

        let normed = self.input_norm.forward(h);
        let attention_out = match &self.attention {
            KimiAttnResAttentionBlock::Mla(attention) => {
                let (output, next_state) = {
                    let previous = cache.try_mla(self.layer_idx)?;
                    attention.forward(normed, previous)
                };
                cache.update_mla(self.layer_idx, next_state)?;
                output
            }
            KimiAttnResAttentionBlock::Kda(attention) => {
                let (output, next_state) = {
                    let previous = cache.try_kda(self.layer_idx)?;
                    attention.forward(normed, previous)
                };
                cache.update_kda(self.layer_idx, next_state)?;
                output
            }
        };
        let partial_after_attn = partial_for_attn + attention_out;

        let h = self
            .mlp_res
            .forward_optional_partial(state.blocks(), Some(&partial_after_attn));

        let mut partial_for_mlp = partial_after_attn;
        if self.starts_new_block_before_mlp() {
            state.push_completed_block(partial_for_mlp.clone());
            partial_for_mlp = Tensor::zeros_like(state.last_completed_block());
        }

        let normed = self.post_attention_norm.forward(h);
        state.set_partial_block(partial_for_mlp + self.feed_forward.forward(normed));
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type TestBackend = NdArray;

    fn reduced_attn_res_config() -> KimiAttnResConfig {
        crate::kimi::config::KimiArtifactConfig::from_json_str(
            r#"{
                "model_type": "kimi_linear",
                "dtype": "float32",
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "moe_intermediate_size": 24,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "kv_lora_rank": 8,
                "q_lora_rank": null,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 4,
                "v_head_dim": 8,
                "mla_use_nope": true,
                "hidden_act": "silu",
                "first_k_dense_replace": 1,
                "moe_layer_freq": 2,
                "num_experts": 4,
                "num_experts_per_token": 2,
                "num_shared_experts": 1,
                "tie_word_embeddings": false,
                "use_cache": true,
                "rms_norm_eps": 1e-5,
                "linear_attn_config": {
                    "full_attn_layers": [2],
                    "kda_layers": [1, 3],
                    "num_heads": 4,
                    "head_dim": 8,
                    "short_conv_kernel_size": 3
                }
            }"#,
        )
        .unwrap()
        .try_attn_res_config(2)
        .unwrap()
    }

    #[test]
    fn attn_res_kimi_layer_preserves_hidden_shape() {
        let device = Default::default();
        let config = reduced_attn_res_config();
        let layer = config.init_layer::<TestBackend>(0, &device);

        let emb = Tensor::random([1, 4, 16], Distribution::Normal(0.0, 1.0), &device);
        let state = KimiAttnResBlockState::new(emb);
        let new_state = layer.forward(state);

        assert_eq!(new_state.partial_block().unwrap().dims(), [1, 4, 16]);
    }

    #[test]
    fn attn_res_kimi_layer_can_boundary_between_attention_and_mlp() {
        let device = Default::default();
        let config = reduced_attn_res_config()
            .baseline
            .try_attn_res_config(6)
            .unwrap();
        let layer0 = config.init_layer::<TestBackend>(0, &device);

        assert_eq!(layer0.block_size(), 1);
        assert!(!layer0.starts_new_block_before_attn());
        assert!(layer0.starts_new_block_before_mlp());
    }

    #[test]
    #[should_panic(expected = "completed block 1 shape mismatch")]
    fn attn_res_kimi_layer_panics_loudly_when_previous_layer_state_is_corrupted() {
        let device = Default::default();
        let config = reduced_attn_res_config();
        let layer1 = config.init_layer::<TestBackend>(1, &device);
        let corrupted_state = KimiAttnResBlockState::from_parts_unchecked(
            vec![
                Tensor::zeros([1, 4, 16], &device),
                Tensor::zeros([1, 5, 16], &device),
            ],
            Some(Tensor::zeros([1, 4, 16], &device)),
        );

        let _ = layer1.forward(corrupted_state);
    }
}
