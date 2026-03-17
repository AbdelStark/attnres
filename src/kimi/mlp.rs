use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::silu;

use crate::kimi::config::KimiDenseMlpRuntimeConfig;

#[derive(Debug)]
pub(crate) struct KimiMlpExpert<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> KimiMlpExpert<B> {
    pub(crate) fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(hidden_size, intermediate_size).init(device),
            up_proj: LinearConfig::new(hidden_size, intermediate_size).init(device),
            down_proj: LinearConfig::new(intermediate_size, hidden_size).init(device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Dense SiLU-gated MLP used on dense Kimi layers.
#[derive(Debug)]
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
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.inner.forward(x)
    }
}
