use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use std::path::Path;

use crate::kimi::attn_res_layer::KimiAttnResDecoderLayer;
use crate::kimi::attn_res_state::KimiAttnResBlockState;
use crate::kimi::cache::{KimiCacheError, KimiDecodeCache};
use crate::kimi::config::{KimiArtifactConfig, KimiAttnResConfig, KimiAttnResConfigError};
use crate::kimi::import::{KimiArtifactUnderstanding, KimiImportSelection};
use crate::kimi::payload::{
    load_param_tensor, load_selected_tensor_payloads, KimiBaselinePayloadError, KimiDecodedTensor,
};
use crate::rms_norm::{RmsNorm, RmsNormConfig};
use crate::two_phase::{
    compute_intra_logit, normalize_inter_output, online_softmax_merge, phase1_batched,
};

/// RFC 0004 AttnRes-Kimi model scaffold.
#[derive(Debug)]
pub struct KimiAttnResModel<B: Backend> {
    embedding: Embedding<B>,
    layers: Vec<KimiAttnResDecoderLayer<B>>,
    final_norm: RmsNorm<B>,
    lm_head: Linear<B>,
    layer_schedule: crate::kimi::schedule::KimiLayerSchedule,
    use_cache: bool,
}

impl KimiArtifactConfig {
    pub fn try_init_attn_res_model<B: Backend>(
        &self,
        num_blocks: usize,
        device: &B::Device,
    ) -> Result<KimiAttnResModel<B>, KimiAttnResConfigError> {
        self.try_attn_res_config(num_blocks)?.try_init_model(device)
    }

    pub fn init_attn_res_model<B: Backend>(
        &self,
        num_blocks: usize,
        device: &B::Device,
    ) -> KimiAttnResModel<B> {
        self.try_init_attn_res_model(num_blocks, device)
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

impl KimiAttnResConfig {
    pub fn try_init_model<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<KimiAttnResModel<B>, KimiAttnResConfigError> {
        self.try_validate()?;

        let layers = (0..self.num_hidden_layers())
            .map(|layer_idx| self.try_init_layer(layer_idx, device))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(KimiAttnResModel {
            embedding: EmbeddingConfig::new(self.vocab_size(), self.hidden_size()).init(device),
            layers,
            final_norm: RmsNormConfig::new(self.hidden_size())
                .with_eps(self.rms_norm_eps())
                .init(device),
            lm_head: LinearConfig::new(self.hidden_size(), self.vocab_size()).init(device),
            layer_schedule: self.baseline.layer_schedule.clone(),
            use_cache: self.use_cache(),
        })
    }

    pub fn init_model<B: Backend>(&self, device: &B::Device) -> KimiAttnResModel<B> {
        self.try_init_model(device)
            .unwrap_or_else(|err| panic!("{err}"))
    }
}

impl KimiArtifactUnderstanding {
    pub fn try_init_attn_res_model_from_dir<B: Backend, P: AsRef<Path>>(
        &self,
        dir: P,
        num_blocks: usize,
        selection: KimiImportSelection,
        device: &B::Device,
    ) -> Result<KimiAttnResModel<B>, KimiBaselinePayloadError> {
        let mut model = self.config.try_init_attn_res_model(num_blocks, device)?;
        model.try_load_baseline_payloads_from_dir(self, dir, selection)?;
        Ok(model)
    }
}

impl<B: Backend> KimiAttnResModel<B> {
    pub fn try_from_artifact_dir<P: AsRef<Path>>(
        dir: P,
        num_blocks: usize,
        selection: KimiImportSelection,
        device: &B::Device,
    ) -> Result<Self, KimiBaselinePayloadError> {
        let dir = dir.as_ref();
        let understanding = KimiArtifactUnderstanding::load_from_dir(dir)?;
        understanding.try_init_attn_res_model_from_dir(dir, num_blocks, selection, device)
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

    pub fn layers(&self) -> &[KimiAttnResDecoderLayer<B>] {
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
        let mut state = KimiAttnResBlockState::new(self.embedding.forward(input_ids));

        for layer in &self.layers {
            state = layer.forward(state);
        }

        self.final_norm.forward(state.into_partial_block())
    }

    pub fn forward_hidden_two_phase(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let embeddings = self.embedding.forward(input_ids);
        let mut completed_blocks = vec![embeddings];
        let mut current_block: Option<Tensor<B, 3>> = None;

        let block_size = self.layers[0].block_size();
        let total_sublayers = self.layers.len() * 2;
        let mut block_start = 0;

        while block_start < total_sublayers {
            if let Some(previous_block) = current_block.take() {
                completed_blocks.push(previous_block);
            }

            let block_end = (block_start + block_size).min(total_sublayers);
            let mut ops = Vec::with_capacity(block_end - block_start);
            for sublayer_idx in block_start..block_end {
                let layer = &self.layers[sublayer_idx / 2];
                let (attn_op, mlp_op) = layer.attn_res_ops();
                ops.push(if sublayer_idx % 2 == 0 {
                    attn_op
                } else {
                    mlp_op
                });
            }

            let phase1 = phase1_batched(&ops, &completed_blocks);
            let mut partial: Option<Tensor<B, 3>> = None;

            for (offset, sublayer_idx) in (block_start..block_end).enumerate() {
                let layer = &self.layers[sublayer_idx / 2];
                let op = ops[offset];

                let h = if offset == 0 {
                    normalize_inter_output(
                        phase1.outputs[offset].clone(),
                        phase1.sum_exp[offset].clone(),
                    )
                } else {
                    let partial_ref = partial.as_ref().expect(
                        "missing AttnRes-Kimi intra-block partial during two-phase forward",
                    );
                    let intra_logit = compute_intra_logit(op, partial_ref);
                    online_softmax_merge(
                        phase1.outputs[offset].clone(),
                        phase1.max_logits[offset].clone(),
                        phase1.sum_exp[offset].clone(),
                        intra_logit,
                        partial_ref.clone(),
                    )
                };

                let sublayer_out = if sublayer_idx % 2 == 0 {
                    layer.forward_attn_sublayer(h)
                } else {
                    layer.forward_mlp_sublayer(h)
                };

                partial = Some(match partial {
                    Some(current_partial) => current_partial + sublayer_out,
                    None => sublayer_out,
                });
            }

            current_block = partial;
            block_start = block_end;
        }

        let output = current_block.expect(
            "missing final block after AttnRes-Kimi two-phase forward; this is a bug in RFC 0004 execution",
        );
        self.final_norm.forward(output)
    }

    pub fn try_forward_hidden_cached(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: &mut KimiDecodeCache<B>,
    ) -> Result<Tensor<B, 3>, KimiCacheError> {
        if !self.use_cache {
            return Err(KimiCacheError::CachingDisabledByConfig);
        }

        let mut state = KimiAttnResBlockState::new(self.embedding.forward(input_ids));
        for layer in &self.layers {
            state = layer.try_forward_cached(state, cache)?;
        }

        Ok(self.final_norm.forward(state.into_partial_block()))
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.lm_head.forward(self.forward_hidden(input_ids))
    }

    pub fn forward_two_phase(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.lm_head
            .forward(self.forward_hidden_two_phase(input_ids))
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
                        detail: "tensor name is outside the AttnRes-Kimi payload map".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    fn reduced_attn_res_config() -> KimiAttnResConfig {
        KimiArtifactConfig::from_json_str(
            r#"{
                "model_type": "kimi_linear",
                "dtype": "float32",
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "moe_intermediate_size": 24,
                "num_hidden_layers": 4,
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
                "first_k_dense_replace": 2,
                "moe_layer_freq": 2,
                "num_experts": 4,
                "num_experts_per_token": 2,
                "num_shared_experts": 1,
                "tie_word_embeddings": false,
                "use_cache": true,
                "rms_norm_eps": 1e-5,
                "linear_attn_config": {
                    "full_attn_layers": [2, 4],
                    "kda_layers": [1, 3],
                    "num_heads": 4,
                    "head_dim": 8,
                    "short_conv_kernel_size": 3
                }
            }"#,
        )
        .unwrap()
        .try_attn_res_config(4)
        .unwrap()
    }

    #[test]
    fn attn_res_kimi_model_forward_shapes_match_expectations() {
        let device = Default::default();
        let model = reduced_attn_res_config().init_model::<TestBackend>(&device);
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 5], &device);

        assert_eq!(model.forward_hidden(input_ids.clone()).dims(), [2, 5, 16]);
        assert_eq!(model.forward(input_ids).dims(), [2, 5, 64]);
    }

    #[test]
    fn attn_res_kimi_two_phase_matches_standard_hidden_forward() {
        let device = Default::default();
        let model = reduced_attn_res_config().init_model::<TestBackend>(&device);
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);

        let standard = model.forward_hidden(input_ids.clone());
        let two_phase = model.forward_hidden_two_phase(input_ids);

        let diff: f32 = (standard - two_phase).abs().max().into_scalar();
        assert!(diff < 1e-3, "two-phase hidden mismatch diff={diff}");
    }
}
