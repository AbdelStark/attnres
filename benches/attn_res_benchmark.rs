//! Benchmarks for AttnRes operations.
//!
//! Compares AttnRes forward pass performance at various model sizes.
//! Run with: `cargo bench`
//!
//! Kimi benchmark ids intentionally encode backend/model/mode/seq_len/layers
//! and, for AttnRes-Kimi, `num_blocks`. Any published number should still add
//! the host CPU/GPU, burn backend version, and dtype alongside that id.

use attnres::kimi::{KimiArtifactConfig, KimiAttnResModel, KimiLinearModel};
use attnres::{causal_mask, AttnResConfig, AttnResTransformer};
use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::Distribution;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

type B = NdArray;

#[derive(Clone, Copy)]
struct LocalKimiBenchMetadata {
    backend: &'static str,
    model: &'static str,
    mode: &'static str,
    seq_len: usize,
    num_hidden_layers: usize,
    num_blocks: Option<usize>,
}

impl LocalKimiBenchMetadata {
    fn id(&self) -> String {
        match self.num_blocks {
            Some(num_blocks) => format!(
                "backend={}/model={}/mode={}/seq_len={}/layers={}/blocks={}",
                self.backend,
                self.model,
                self.mode,
                self.seq_len,
                self.num_hidden_layers,
                num_blocks
            ),
            None => format!(
                "backend={}/model={}/mode={}/seq_len={}/layers={}",
                self.backend, self.model, self.mode, self.seq_len, self.num_hidden_layers
            ),
        }
    }
}

fn reduced_local_kimi_config() -> KimiArtifactConfig {
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
}

fn bench_attn_res_op(c: &mut Criterion) {
    let device = Default::default();

    let mut group = c.benchmark_group("attn_res_op_forward");
    for num_blocks in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("blocks", num_blocks),
            &num_blocks,
            |b, &num_blocks| {
                let config = AttnResConfig::new(64, 12, num_blocks);
                let op = config.init_op::<B>(&device);
                let blocks: Vec<_> = (0..num_blocks)
                    .map(|_| Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device))
                    .collect();
                let partial = Tensor::random([2, 16, 64], Distribution::Normal(0.0, 1.0), &device);

                b.iter(|| {
                    black_box(op.forward(&blocks, &partial));
                });
            },
        );
    }
    group.finish();
}

fn bench_model_forward(c: &mut Criterion) {
    let device = Default::default();

    let mut group = c.benchmark_group("model_forward");
    for (d_model, layers, blocks, heads) in
        [(32, 4, 2, 4), (64, 8, 2, 4), (64, 8, 4, 4), (128, 8, 2, 8)]
    {
        let label = format!("d{d_model}_l{layers}_b{blocks}");
        group.bench_with_input(
            BenchmarkId::new("config", &label),
            &(d_model, layers, blocks, heads),
            |b, &(d_model, layers, blocks, heads)| {
                let config = AttnResConfig::new(d_model, layers, blocks)
                    .with_num_heads(heads)
                    .with_vocab_size(100);
                let model: AttnResTransformer<B> = config.init_model(&device);
                let input_ids = Tensor::<B, 2, Int>::zeros([1, 16], &device);

                b.iter(|| {
                    black_box(model.forward(input_ids.clone(), None));
                });
            },
        );
    }
    group.finish();
}

fn bench_model_with_mask(c: &mut Criterion) {
    let device = Default::default();
    let config = AttnResConfig::new(64, 8, 2)
        .with_num_heads(4)
        .with_vocab_size(100);
    let model: AttnResTransformer<B> = config.init_model(&device);
    let mask = causal_mask::<B>(2, 16, &device);
    let input_ids = Tensor::<B, 2, Int>::zeros([2, 16], &device);

    c.bench_function("model_forward_with_mask", |b| {
        b.iter(|| {
            black_box(model.forward(input_ids.clone(), Some(&mask)));
        });
    });
}

fn bench_seq_len_scaling(c: &mut Criterion) {
    let device = Default::default();

    let mut group = c.benchmark_group("seq_len_scaling");
    for seq_len in [8, 16, 32, 64] {
        let config = AttnResConfig::new(64, 8, 2)
            .with_num_heads(4)
            .with_vocab_size(100);
        let model: AttnResTransformer<B> = config.init_model(&device);

        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &seq_len| {
                let input_ids = Tensor::<B, 2, Int>::zeros([1, seq_len], &device);
                b.iter(|| {
                    black_box(model.forward(input_ids.clone(), None));
                });
            },
        );
    }
    group.finish();
}

fn bench_kimi_baseline_forward(c: &mut Criterion) {
    let device = Default::default();
    B::seed(&device, 0);
    let model: KimiLinearModel<B> = reduced_local_kimi_config().try_init_model(&device).unwrap();

    let mut group = c.benchmark_group("kimi_baseline_forward_local");
    for seq_len in [32, 128] {
        let metadata = LocalKimiBenchMetadata {
            backend: "ndarray",
            model: "baseline_kimi",
            mode: "forward",
            seq_len,
            num_hidden_layers: 4,
            num_blocks: None,
        };
        let input_ids = Tensor::<B, 2, Int>::zeros([1, seq_len], &device);

        group.bench_with_input(
            BenchmarkId::from_parameter(metadata.id()),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(model.forward(input_ids.clone()));
                });
            },
        );
    }
    group.finish();
}

fn bench_kimi_baseline_cached_forward(c: &mut Criterion) {
    let device = Default::default();
    B::seed(&device, 0);
    let model: KimiLinearModel<B> = reduced_local_kimi_config().try_init_model(&device).unwrap();

    let mut group = c.benchmark_group("kimi_baseline_cached_forward_local");
    for seq_len in [32, 128] {
        let metadata = LocalKimiBenchMetadata {
            backend: "ndarray",
            model: "baseline_kimi",
            mode: "cached_forward",
            seq_len,
            num_hidden_layers: 4,
            num_blocks: None,
        };
        let token = Tensor::<B, 2, Int>::zeros([1, 1], &device);

        group.bench_with_input(
            BenchmarkId::from_parameter(metadata.id()),
            &seq_len,
            |b, &seq_len| {
                b.iter(|| {
                    let mut cache = model.new_cache();
                    for _ in 0..seq_len {
                        black_box(model.try_forward_cached(token.clone(), &mut cache).unwrap());
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_kimi_attn_res_forward(c: &mut Criterion) {
    let device = Default::default();
    B::seed(&device, 0);
    let model: KimiAttnResModel<B> = reduced_local_kimi_config()
        .try_init_attn_res_model(4, &device)
        .unwrap();

    let mut group = c.benchmark_group("kimi_attn_res_forward_local");
    for seq_len in [32, 128] {
        let metadata = LocalKimiBenchMetadata {
            backend: "ndarray",
            model: "attn_res_kimi",
            mode: "forward",
            seq_len,
            num_hidden_layers: 4,
            num_blocks: Some(4),
        };
        let input_ids = Tensor::<B, 2, Int>::zeros([1, seq_len], &device);

        group.bench_with_input(
            BenchmarkId::from_parameter(metadata.id()),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(model.forward(input_ids.clone()));
                });
            },
        );
    }
    group.finish();
}

fn bench_kimi_attn_res_two_phase(c: &mut Criterion) {
    let device = Default::default();
    B::seed(&device, 0);
    let model: KimiAttnResModel<B> = reduced_local_kimi_config()
        .try_init_attn_res_model(4, &device)
        .unwrap();

    let mut group = c.benchmark_group("kimi_attn_res_two_phase_local");
    for seq_len in [32, 128] {
        let metadata = LocalKimiBenchMetadata {
            backend: "ndarray",
            model: "attn_res_kimi",
            mode: "two_phase",
            seq_len,
            num_hidden_layers: 4,
            num_blocks: Some(4),
        };
        let input_ids = Tensor::<B, 2, Int>::zeros([1, seq_len], &device);

        group.bench_with_input(
            BenchmarkId::from_parameter(metadata.id()),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(model.forward_two_phase(input_ids.clone()));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_attn_res_op,
    bench_model_forward,
    bench_model_with_mask,
    bench_seq_len_scaling,
    bench_kimi_baseline_forward,
    bench_kimi_baseline_cached_forward,
    bench_kimi_attn_res_forward,
    bench_kimi_attn_res_two_phase,
);
criterion_main!(benches);
