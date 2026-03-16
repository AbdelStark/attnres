//! Benchmarks for AttnRes operations.
//!
//! Compares AttnRes forward pass performance at various model sizes.
//! Run with: `cargo bench`

use attnres::{causal_mask, AttnResConfig, AttnResTransformer};
use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::Distribution;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

type B = NdArray;

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

criterion_group!(
    benches,
    bench_attn_res_op,
    bench_model_forward,
    bench_model_with_mask,
    bench_seq_len_scaling,
);
criterion_main!(benches);
