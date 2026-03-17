use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::kimi::cache::{KimiCacheError, KimiDecodeCache};
use crate::kimi::config::{KimiArtifactConfig, KimiArtifactConfigError, KimiBaselineConfig};
use crate::kimi::layer::KimiDecoderLayer;
use crate::kimi::schedule::KimiLayerSchedule;
use crate::rms_norm::{RmsNorm, RmsNormConfig};

/// Baseline Kimi Linear model scaffold from RFC 0002.
#[derive(Debug)]
pub struct KimiLinearModel<B: Backend> {
    embedding: Embedding<B>,
    layers: Vec<KimiDecoderLayer<B>>,
    final_norm: RmsNorm<B>,
    lm_head: Linear<B>,
    layer_schedule: KimiLayerSchedule,
    use_cache: bool,
}

impl KimiArtifactConfig {
    pub fn try_init_model<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<KimiLinearModel<B>, KimiArtifactConfigError> {
        self.try_baseline_config()?.try_init_model(device)
    }

    pub fn init_model<B: Backend>(&self, device: &B::Device) -> KimiLinearModel<B> {
        self.try_init_model(device)
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

impl KimiBaselineConfig {
    pub fn try_init_model<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<KimiLinearModel<B>, KimiArtifactConfigError> {
        self.try_validate()?;

        let layers = (0..self.num_hidden_layers())
            .map(|layer_idx| self.try_init_layer(layer_idx, device))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(KimiLinearModel {
            embedding: EmbeddingConfig::new(self.vocab_size(), self.hidden_size()).init(device),
            layers,
            final_norm: RmsNormConfig::new(self.hidden_size())
                .with_eps(self.rms_norm_eps())
                .init(device),
            lm_head: LinearConfig::new(self.hidden_size(), self.vocab_size()).init(device),
            layer_schedule: self.layer_schedule.clone(),
            use_cache: self.use_cache(),
        })
    }

    pub fn init_model<B: Backend>(&self, device: &B::Device) -> KimiLinearModel<B> {
        self.try_init_model(device)
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

impl<B: Backend> KimiLinearModel<B> {
    pub fn layers(&self) -> &[KimiDecoderLayer<B>] {
        &self.layers
    }

    pub fn supports_cache(&self) -> bool {
        self.use_cache
    }

    pub fn embed_tokens(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(input_ids)
    }

    pub fn try_new_cache(&self) -> Result<KimiDecodeCache<B>, KimiCacheError> {
        if !self.use_cache {
            return Err(KimiCacheError::CachingDisabledByConfig);
        }

        Ok(KimiDecodeCache::from_schedule(&self.layer_schedule))
    }

    pub fn new_cache(&self) -> KimiDecodeCache<B> {
        self.try_new_cache().unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn forward_hidden(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut hidden = self.embedding.forward(input_ids);

        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }

        self.final_norm.forward(hidden)
    }

    pub fn try_forward_hidden_cached(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: &mut KimiDecodeCache<B>,
    ) -> Result<Tensor<B, 3>, KimiCacheError> {
        if !self.use_cache {
            return Err(KimiCacheError::CachingDisabledByConfig);
        }

        let mut hidden = self.embedding.forward(input_ids);
        for layer in &self.layers {
            hidden = layer.try_forward_cached(hidden, cache)?;
        }

        Ok(self.final_norm.forward(hidden))
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.lm_head.forward(self.forward_hidden(input_ids))
    }

    pub fn try_forward_cached(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: &mut KimiDecodeCache<B>,
    ) -> Result<Tensor<B, 3>, KimiCacheError> {
        Ok(self
            .lm_head
            .forward(self.try_forward_hidden_cached(input_ids, cache)?))
    }
}
