use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use std::path::Path;

use crate::kimi::cache::{KimiCacheError, KimiDecodeCache};
use crate::kimi::config::{KimiArtifactConfig, KimiArtifactConfigError, KimiBaselineConfig};
use crate::kimi::import::{KimiArtifactUnderstanding, KimiImportSelection};
use crate::kimi::layer::KimiDecoderLayer;
use crate::kimi::payload::{
    load_param_tensor, load_selected_tensor_payloads, KimiBaselinePayloadError, KimiDecodedTensor,
};
use crate::kimi::schedule::KimiLayerSchedule;
use crate::rms_norm::{RmsNorm, RmsNormConfig};

/// Baseline Kimi Linear model scaffold from RFC 0002.
#[derive(Module, Debug)]
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
            lm_head: LinearConfig::new(self.hidden_size(), self.vocab_size())
                .with_bias(false)
                .init(device),
            layer_schedule: self.layer_schedule.clone(),
            use_cache: self.use_cache(),
        })
    }

    pub fn init_model<B: Backend>(&self, device: &B::Device) -> KimiLinearModel<B> {
        self.try_init_model(device)
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

impl KimiArtifactUnderstanding {
    pub fn try_init_baseline_model_from_dir<B: Backend, P: AsRef<Path>>(
        &self,
        dir: P,
        selection: KimiImportSelection,
        device: &B::Device,
    ) -> Result<KimiLinearModel<B>, KimiBaselinePayloadError> {
        let mut model = self.config.try_init_model(device)?;
        model.try_load_baseline_payloads_from_dir(self, dir, selection)?;
        Ok(model)
    }
}

impl<B: Backend> KimiLinearModel<B> {
    pub fn try_from_artifact_dir<P: AsRef<Path>>(
        dir: P,
        selection: KimiImportSelection,
        device: &B::Device,
    ) -> Result<Self, KimiBaselinePayloadError> {
        let dir = dir.as_ref();
        let understanding = KimiArtifactUnderstanding::load_from_dir(dir)?;
        understanding.try_init_baseline_model_from_dir(dir, selection, device)
    }

    pub fn try_load_baseline_payloads_from_dir<P: AsRef<Path>>(
        &mut self,
        understanding: &KimiArtifactUnderstanding,
        dir: P,
        selection: KimiImportSelection,
    ) -> Result<(), KimiBaselinePayloadError> {
        let plan = understanding.try_slice_plan(selection)?;
        plan.try_require_loadable()?;

        let payloads =
            load_selected_tensor_payloads(&plan, dir.as_ref(), &understanding.config.dtype)?;
        for mapping in &plan.coverage.mapped_tensors {
            let payload = payloads
                .get(&mapping.tensor_name)
                .expect("complete payload coverage should be validated before application");
            self.try_apply_tensor_payload(&mapping.tensor_name, payload)?;
        }

        Ok(())
    }

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

    fn try_apply_tensor_payload(
        &mut self,
        tensor_name: &str,
        payload: &KimiDecodedTensor,
    ) -> Result<(), KimiBaselinePayloadError> {
        match tensor_name {
            "model.embed_tokens.weight" => {
                load_param_tensor(&mut self.embedding.weight, tensor_name, payload)
            }
            "model.norm.weight" => {
                load_param_tensor(self.final_norm.gamma_param_mut(), tensor_name, payload)
            }
            "lm_head.weight" => load_param_tensor(&mut self.lm_head.weight, tensor_name, payload),
            _ => {
                let prefix = "model.layers.";
                let Some(remainder) = tensor_name.strip_prefix(prefix) else {
                    return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: "tensor name is outside the baseline Kimi payload map".to_string(),
                    });
                };
                let Some((layer_idx, remainder)) = remainder.split_once('.') else {
                    return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: "tensor name is missing the decoder-layer suffix".to_string(),
                    });
                };
                let layer_idx = layer_idx.parse::<usize>().map_err(|_| {
                    KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: "decoder layer index is not a valid usize".to_string(),
                    }
                })?;
                let Some(layer) = self.layers.get_mut(layer_idx) else {
                    return Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                        tensor_name: tensor_name.to_string(),
                        detail: format!("decoder layer {layer_idx} is out of range"),
                    });
                };
                layer.try_apply_tensor_payload(tensor_name, remainder, payload)
            }
        }
    }
}
