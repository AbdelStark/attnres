use burn::prelude::*;

use crate::kimi::attention::{KimiKdaAttention, KimiMlaAttention};
use crate::kimi::cache::{KimiCacheError, KimiDecodeCache};
use crate::kimi::config::{KimiArtifactConfig, KimiArtifactConfigError, KimiBaselineConfig};
use crate::kimi::mlp::KimiDenseMlp;
use crate::kimi::moe::KimiSparseMoe;
use crate::kimi::schedule::{KimiAttentionLayerKind, KimiFeedForwardLayerKind};
use crate::rms_norm::{RmsNorm, RmsNormConfig};

#[derive(Debug)]
enum KimiAttentionBlock<B: Backend> {
    Mla(Box<KimiMlaAttention<B>>),
    Kda(Box<KimiKdaAttention<B>>),
}

#[derive(Debug)]
enum KimiFeedForwardBlock<B: Backend> {
    Dense(Box<KimiDenseMlp<B>>),
    SparseMoe(Box<KimiSparseMoe<B>>),
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
#[derive(Debug)]
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
            KimiAttentionLayerKind::FullAttention => {
                KimiAttentionBlock::Mla(Box::new(self.attention.init_mla(device)))
            }
            KimiAttentionLayerKind::LinearAttentionKda => {
                KimiAttentionBlock::Kda(Box::new(self.attention.init_kda(device)))
            }
        };
        let feed_forward = match layer.feed_forward_kind {
            KimiFeedForwardLayerKind::DenseMlp => {
                KimiFeedForwardBlock::Dense(Box::new(self.dense_mlp.init(device)))
            }
            KimiFeedForwardLayerKind::SparseMoe => {
                KimiFeedForwardBlock::SparseMoe(Box::new(self.sparse_moe.init(device)))
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
