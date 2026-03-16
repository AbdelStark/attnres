//! Minimal tensor utilities for the WASM demo.
//!
//! Not a full tensor library — just enough for the AttnRes visualization.
//! The real implementation uses `burn::Tensor`.

/// A simple 1D tensor (vector) wrapper for clarity.
pub struct Tensor {
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn dot(&self, other: &Tensor) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn norm(&self) -> f32 {
        self.data.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    pub fn scale(&self, s: f32) -> Tensor {
        Tensor {
            data: self.data.iter().map(|v| v * s).collect(),
        }
    }
}
