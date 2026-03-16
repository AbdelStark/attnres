---
name: rust-burn-development
description: Patterns for building ML modules with the burn deep learning framework. Activate when creating or modifying burn Modules, Configs, tensor operations, or backend-generic code. Also relevant when setting up Cargo.toml dependencies for burn.
prerequisites: Rust 1.80+, burn crate
---

# Rust + Burn Development

<purpose>
Guides implementation of burn-based ML modules for the attnres project. Covers Module/Config patterns, parameter initialization, backend generics, and burn API idioms.
</purpose>

<context>
— burn uses a `Module` derive macro for neural network layers
— burn uses a `Config` derive macro for hyperparameter structs that can `init()` modules
— All modules are generic over `Backend`: `struct Foo<B: Backend>`
— Learnable parameters wrap tensors: `Param<Tensor<B, N>>`
— burn provides `Embedding`, `Linear`, and basic ops; AttnRes builds custom modules on top
</context>

<procedure>
1. Define config struct with `#[derive(Config, Debug)]`
2. Add `#[config(default = value)]` for fields with defaults
3. Define module struct with `#[derive(Module, Debug)]`
4. Implement `ConfigType::init(&self, device: &B::Device) -> ModuleType<B>` on the config
5. Implement `forward()` method on the module
6. Add `pub mod module_name;` to lib.rs
7. Add public re-export if part of public API
8. Write unit tests using NdArray backend
</procedure>

<patterns>
<do>
  — Use `#[derive(Module, Debug)]` for all neural network layers
  — Use `Param<Tensor<B, N>>` for learnable parameters
  — Initialize via Config::init pattern:
    ```rust
    impl AttnResConfig {
        pub fn init<B: Backend>(&self, device: &B::Device) -> AttnResOp<B> {
            let pseudo_query = Tensor::zeros([self.d_model], device);
            AttnResOp {
                pseudo_query: Param::from_tensor(pseudo_query),
                norm: RmsNormConfig::new(self.d_model, self.rms_norm_eps).init(device),
            }
        }
    }
    ```
  — Keep Backend generic everywhere — never hardcode a specific backend
  — Use `Tensor::zeros`, `Tensor::ones`, `Tensor::random` for initialization
  — Use `.clone()` on tensors when needed in computation graphs (burn tensors are reference-counted)
</do>
<dont>
  — Don't implement `Module` manually — always use derive macro
  — Don't store non-parameter tensors as `Param` — use plain `Tensor` or store in config
  — Don't use `unwrap()` on tensor operations — handle shape mismatches explicitly
  — Don't forget `device` parameter in init — all tensors must be on the same device
</dont>
</patterns>

<examples>
Example: Creating a burn Module with Config

```rust
use burn::prelude::*;
use burn::module::Module;
use burn::config::Config;

#[derive(Config, Debug)]
pub struct RmsNormConfig {
    pub d_model: usize,
    #[config(default = 1e-6)]
    pub eps: f64,
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    eps: f64,
}

impl RmsNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        RmsNorm {
            gamma: Param::from_tensor(Tensor::ones([self.d_model], device)),
            eps: self.eps,
        }
    }
}
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| "trait bound `B: Backend` is not satisfied" | Missing generic parameter | Add `<B: Backend>` to impl block and function signatures |
| "cannot move out of borrowed content" | Tensor consumed in computation | Use `.clone()` before the consuming operation |
| "expected Tensor<B, 3>, found Tensor<B, 4>" | Wrong number of dimensions | Check tensor shape; use `.squeeze()` or `.unsqueeze()` to adjust |
| Config field not settable | Missing pub on config field | Add `pub` to field in `#[derive(Config)]` struct |
</troubleshooting>

<references>
— spec.md: Full module specifications with Rust pseudocode
— burn docs: https://burn.dev/docs
— burn examples: https://github.com/tracel-ai/burn/tree/main/examples
</references>
