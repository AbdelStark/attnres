use burn::module::Module;
use burn::prelude::*;

use crate::kimi::attention::{KimiKdaAttention, KimiMlaAttention};
use crate::kimi::cache::{KimiCacheError, KimiDecodeCache};
use crate::kimi::config::{KimiArtifactConfig, KimiArtifactConfigError, KimiBaselineConfig};
use crate::kimi::mlp::KimiDenseMlp;
use crate::kimi::moe::KimiSparseMoe;
use crate::kimi::payload::{load_param_tensor, KimiBaselinePayloadError, KimiDecodedTensor};
use crate::kimi::schedule::{KimiAttentionLayerKind, KimiFeedForwardLayerKind};
use crate::rms_norm::{RmsNorm, RmsNormConfig};

#[derive(Module, Debug)]
enum KimiAttentionBlock<B: Backend> {
    Mla(KimiMlaAttention<B>),
    Kda(KimiKdaAttention<B>),
}

#[derive(Module, Debug)]
enum KimiFeedForwardBlock<B: Backend> {
    Dense(KimiDenseMlp<B>),
    SparseMoe(KimiSparseMoe<B>),
}

impl<B: Backend> KimiFeedForwardBlock<B> {
    fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden),
            Self::SparseMoe(moe) => moe.forward(hidden),
        }
    }
}

/// One baseline Kimi decoder layer.
#[derive(Module, Debug)]
pub struct KimiDecoderLayer<B: Backend> {
    layer_idx: usize,
    attention_kind: KimiAttentionLayerKind,
    feed_forward_kind: KimiFeedForwardLayerKind,
    input_norm: RmsNorm<B>,
    attention: KimiAttentionBlock<B>,
    post_attention_norm: RmsNorm<B>,
    feed_forward: KimiFeedForwardBlock<B>,
}

impl KimiArtifactConfig {
    pub fn try_init_layer<B: Backend>(
        &self,
        layer_idx: usize,
        device: &B::Device,
    ) -> Result<KimiDecoderLayer<B>, KimiArtifactConfigError> {
        self.try_baseline_config()?
            .try_init_layer(layer_idx, device)
    }

    pub fn init_layer<B: Backend>(
        &self,
        layer_idx: usize,
        device: &B::Device,
    ) -> KimiDecoderLayer<B> {
        self.try_init_layer(layer_idx, device)
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

impl KimiBaselineConfig {
    pub fn try_init_layer<B: Backend>(
        &self,
        layer_idx: usize,
        device: &B::Device,
    ) -> Result<KimiDecoderLayer<B>, KimiArtifactConfigError> {
        self.try_validate()?;
        let layer = self
            .layer_schedule
            .try_layer(layer_idx)
            .map_err(KimiArtifactConfigError::from)?;

        let attention = match layer.attention_kind {
            KimiAttentionLayerKind::FullAttention => KimiAttentionBlock::Mla(self.attention.init_mla(device)),
            KimiAttentionLayerKind::LinearAttentionKda => {
                KimiAttentionBlock::Kda(self.attention.init_kda(device))
            }
        };
        let feed_forward = match layer.feed_forward_kind {
            KimiFeedForwardLayerKind::DenseMlp => KimiFeedForwardBlock::Dense(self.dense_mlp.init(device)),
            KimiFeedForwardLayerKind::SparseMoe => {
                KimiFeedForwardBlock::SparseMoe(self.sparse_moe.init(device))
            }
        };

        Ok(KimiDecoderLayer {
            layer_idx,
            attention_kind: layer.attention_kind,
            feed_forward_kind: layer.feed_forward_kind,
            input_norm: RmsNormConfig::new(self.hidden_size())
                .with_eps(self.rms_norm_eps())
                .init(device),
            attention,
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
    ) -> KimiDecoderLayer<B> {
        self.try_init_layer(layer_idx, device)
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

impl<B: Backend> KimiDecoderLayer<B> {
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
                    return match &mut self.attention {
                        KimiAttentionBlock::Mla(attention) => {
                            attention.try_apply_tensor_payload(tensor_name, leaf, payload)
                        }
                        KimiAttentionBlock::Kda(attention) => {
                            attention.try_apply_tensor_payload(tensor_name, leaf, payload)
                        }
                    };
                }

                if let Some(leaf) = remainder.strip_prefix("mlp.") {
                    return match &mut self.feed_forward {
                        KimiFeedForwardBlock::Dense(mlp) => {
                            mlp.try_apply_tensor_payload(tensor_name, leaf, payload)
                        }
                        KimiFeedForwardBlock::SparseMoe(_) => {
                            Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                                tensor_name: tensor_name.to_string(),
                                detail: "dense MLP tensor cannot be applied to sparse MoE layer"
                                    .to_string(),
                            })
                        }
                    };
                }

                if let Some(leaf) = remainder.strip_prefix("block_sparse_moe.") {
                    return match &mut self.feed_forward {
                        KimiFeedForwardBlock::Dense(_) => {
                            Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                                tensor_name: tensor_name.to_string(),
                                detail: "sparse MoE tensor cannot be applied to dense MLP layer"
                                    .to_string(),
                            })
                        }
                        KimiFeedForwardBlock::SparseMoe(moe) => {
                            moe.try_apply_tensor_payload(tensor_name, leaf, payload)
                        }
                    };
                }

                Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                    tensor_name: tensor_name.to_string(),
                    detail: format!("unsupported decoder-layer tensor leaf '{remainder}'"),
                })
            }
        }
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
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

    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let normed = self.input_norm.forward(hidden);
        let attention_out = match &self.attention {
            KimiAttentionBlock::Mla(attention) => attention.forward(normed, None).0,
            KimiAttentionBlock::Kda(attention) => attention.forward(normed, None).0,
        };
        let hidden = residual + attention_out;

        let residual = hidden.clone();
        let normed = self.post_attention_norm.forward(hidden);
        residual + self.feed_forward.forward(normed)
    }

    pub fn try_forward_cached(
        &self,
        hidden: Tensor<B, 3>,
        cache: &mut KimiDecodeCache<B>,
    ) -> Result<Tensor<B, 3>, KimiCacheError> {
        let residual = hidden.clone();
        let normed = self.input_norm.forward(hidden);
        let attention_out = match &self.attention {
            KimiAttentionBlock::Mla(attention) => {
                let (output, next_state) = {
                    let previous = cache.try_mla(self.layer_idx)?;
                    attention.forward(normed, previous)
                };
                cache.update_mla(self.layer_idx, next_state)?;
                output
            }
            KimiAttentionBlock::Kda(attention) => {
                let (output, next_state) = {
                    let previous = cache.try_kda(self.layer_idx)?;
                    attention.forward(normed, previous)
                };
                cache.update_kda(self.layer_idx, next_state)?;
                output
            }
        };
        let hidden = residual + attention_out;

        let residual = hidden.clone();
        let normed = self.post_attention_norm.forward(hidden);
        Ok(residual + self.feed_forward.forward(normed))
    }
}
