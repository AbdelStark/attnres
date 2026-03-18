use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;

use crate::kimi::config::KimiDenseMlpRuntimeConfig;
use crate::kimi::payload::{load_param_tensor, KimiBaselinePayloadError, KimiDecodedTensor};

#[derive(Module, Debug)]
pub(crate) struct KimiMlpExpert<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> KimiMlpExpert<B> {
    pub(crate) fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(false)
                .init(device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }

    pub(crate) fn try_apply_tensor_payload(
        &mut self,
        tensor_name: &str,
        leaf: &str,
        payload: &KimiDecodedTensor,
    ) -> Result<(), KimiBaselinePayloadError> {
        match leaf {
            "gate_proj.weight" => {
                load_param_tensor(&mut self.gate_proj.weight, tensor_name, payload)
            }
            "up_proj.weight" => load_param_tensor(&mut self.up_proj.weight, tensor_name, payload),
            "down_proj.weight" => {
                load_param_tensor(&mut self.down_proj.weight, tensor_name, payload)
            }
            _ => Err(KimiBaselinePayloadError::UnsupportedTensorApplication {
                tensor_name: tensor_name.to_string(),
                detail: format!("unsupported MLP tensor leaf '{leaf}'"),
            }),
        }
    }
}

/// Dense SiLU-gated MLP used on dense Kimi layers.
#[derive(Module, Debug)]
pub struct KimiDenseMlp<B: Backend> {
    inner: KimiMlpExpert<B>,
}

impl KimiDenseMlpRuntimeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KimiDenseMlp<B> {
        debug_assert_eq!(self.hidden_act, "silu");
        KimiDenseMlp {
            inner: KimiMlpExpert::new(self.hidden_size, self.intermediate_size, device),
        }
    }
}

impl<B: Backend> KimiDenseMlp<B> {
    pub(crate) fn try_apply_tensor_payload(
        &mut self,
        tensor_name: &str,
        leaf: &str,
        payload: &KimiDecodedTensor,
    ) -> Result<(), KimiBaselinePayloadError> {
        self.inner
            .try_apply_tensor_payload(tensor_name, leaf, payload)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.inner.forward(x)
    }
}
