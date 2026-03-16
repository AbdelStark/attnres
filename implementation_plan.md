# attnres-rs — Implementation Plan

## Phased Build Schedule

---

## Phase 1: Core Algorithm (Days 1-3)

### Day 1: Project Setup + Core Types

**Morning:**
- [ ] Initialize Cargo project with burn dependencies
- [ ] Set up CI (GitHub Actions: cargo test, clippy, fmt)
- [ ] Implement `AttnResConfig` with validation
- [ ] Implement `BlockState` struct with `new()` and `push_block()`

**Afternoon:**
- [ ] Implement `RmsNorm` module (4D tensor variant for stacked blocks)
- [ ] Unit tests: RmsNorm output shape, numerical stability with eps
- [ ] Implement `AttnResOp` struct with zero-initialized pseudo-query

**Evening:**
- [ ] Implement `AttnResOp::forward()` — the core algorithm
- [ ] Map each line of the paper's pseudocode to Rust code with comments
- [ ] Unit tests: zero-init produces uniform weights, output shape correct, weights sum to 1

### Day 2: Layer + Model

**Morning:**
- [ ] Implement basic `MultiHeadAttention` module (or use burn's built-in)
- [ ] Implement basic `FeedForward` (MLP) module
- [ ] Implement `AttnResLayer::forward()` with block boundary tracking

**Afternoon:**
- [ ] Implement `AttnResTransformer` (full model with embedding + layers + lm_head)
- [ ] Integration test: forward pass on random data, check output shape
- [ ] Test block boundary tracking with different num_layers / num_blocks configs

**Evening:**
- [ ] Implement `StandardResidualLayer` (baseline without AttnRes) for comparison
- [ ] Implement `StandardTransformer` (baseline model)
- [ ] Test: both models produce same output when AttnRes has zero queries (uniform weights)

### Day 3: Training Loop + Verification

**Morning:**
- [ ] Implement training loop with AdamW optimizer
- [ ] Data loading: WikiText-2 or a simple text dataset via burn's dataset utilities
- [ ] Train tiny model (4 layers, d=128, 2 blocks) on 1000 batches

**Afternoon:**
- [ ] Generate PyTorch reference outputs using paper's pseudocode
- [ ] Save reference as JSON fixtures
- [ ] Implement differential tests: compare Rust outputs to PyTorch within tolerance
- [ ] Fix any numerical discrepancies

**Evening:**
- [ ] Run comparison: train Standard vs AttnRes on same data/hyperparams
- [ ] Log training loss curves
- [ ] Verify AttnRes achieves lower loss (replicating the paper's core claim)

---

## Phase 2: Quality + Examples (Days 4-6)

### Day 4: Tests + Documentation

**Morning:**
- [ ] Property-based tests with proptest (convex combination, permutation equivariance)
- [ ] Gradient flow tests (verify gradients reach pseudo-queries and all blocks)
- [ ] Edge cases: single block (N=1 = standard residuals), full AttnRes (N=L)

**Afternoon:**
- [ ] Write comprehensive doc comments for all public types and functions
- [ ] Add paper equation references in doc comments (e.g., "See Eq. 4")
- [ ] Write README with:
  - What AttnRes is (one paragraph + the paper's overview figure)
  - Quick start code
  - Benchmark results
  - Link to paper, tweet, and repo

**Evening:**
- [ ] Create `examples/train_tiny.rs` — complete training example
- [ ] Create `examples/compare_residuals.rs` — standard vs AttnRes comparison
- [ ] Make sure `cargo run --example train_tiny` works on CPU (ndarray backend)

### Day 5: Visualization + Benchmarks

**Morning:**
- [ ] Implement attention weight extraction (expose alpha_{i->l} for visualization)
- [ ] Create `examples/visualize_weights.rs` — plot learned depth attention patterns
- [ ] Generate visualization: which blocks does each layer attend to? (replicating Fig 8 from paper)

**Afternoon:**
- [ ] Implement criterion benchmarks for AttnRes vs standard residuals
- [ ] Benchmark: forward pass latency
- [ ] Benchmark: backward pass latency
- [ ] Benchmark: memory usage
- [ ] Record results in README table

**Evening:**
- [ ] Test on wgpu backend (cross-platform GPU)
- [ ] Test on CUDA backend (if NVIDIA GPU available)
- [ ] Fix any backend-specific issues

### Day 6: Two-Phase Inference

**Morning:**
- [ ] Implement `TwoPhaseInference` struct
- [ ] Phase 1: batched inter-block attention
- [ ] Phase 2: sequential intra-block + online softmax merge

**Afternoon:**
- [ ] Verify two-phase output matches naive layer-by-layer output (within tolerance)
- [ ] Benchmark: two-phase vs naive inference latency
- [ ] Measure the < 2% latency overhead claim from the paper

**Evening:**
- [ ] Clean up code, run clippy, format
- [ ] Final test pass: `cargo test --all-features`
- [ ] Prepare for launch

---

## Phase 3: Launch (Day 7)

### Morning: Pre-Launch

- [ ] Final README polish
- [ ] Add badges (CI, license, crates.io, docs.rs)
- [ ] Create GitHub repo: `AbdelStark/attnres-rs`
- [ ] Push code
- [ ] Publish to crates.io: `cargo publish`
- [ ] Generate docs: `cargo doc --open`, verify on docs.rs

### Afternoon: Launch

- [ ] Write X thread (10-12 tweets):
  1. Hook: "I built the first Rust implementation of Attention Residuals from @kimi_moonshot's new paper."
  2. What AttnRes is (one-sentence, with diagram)
  3. The core insight: "Residual connections have been doing linear attention over depth. AttnRes upgrades to softmax attention."
  4. Key result: "1.25x compute advantage as a drop-in replacement"
  5. Why Rust: "Production deployment needs zero-cost abstractions. Burn compiles to CUDA, Metal, WASM, CPU."
  6. Code snippet showing the API
  7. Benchmark results (latency table)
  8. Visualization of learned attention weights
  9. Comparison: standard residuals vs AttnRes training curves
  10. Links: repo, paper, crates.io
  11. Quote-tweet of @kimi_moonshot's announcement

- [ ] Post to Hacker News: "Show HN: attnres-rs — Attention Residuals (Kimi) in Rust with Burn"
- [ ] Post to Reddit: r/rust, r/MachineLearning
- [ ] Post to LinkedIn (for AMI Labs visibility)
- [ ] Share in burn framework Discord (they'll appreciate the showcase)

### Evening: Engagement

- [ ] Respond to every comment, star, issue
- [ ] Tag @kimi_moonshot and @burn_rs on X
- [ ] Share in awesome-world-models and World Model Weekly (AttnRes is relevant to world model Transformers)

---

## Phase 4: Post-Launch (Week 2+)

### Ongoing Improvements

- [ ] Add WASM example (run AttnRes inference in the browser)
- [ ] Add safetensors weight loading (in case Kimi releases pre-trained weights)
- [ ] Add support for MoE (Mixture of Experts) layers to match Kimi Linear architecture
- [ ] Write blog post: "Implementing Attention Residuals in Rust: What I Learned"
- [ ] Explore integration with jepa-rs (AttnRes as the residual connection in JEPA's ViT encoder)

### Integration with Broader Ecosystem

attnres-rs strengthens your portfolio:
- **jepa-rs** uses standard ViT: attnres-rs could replace the residual connections
- **WorldForge** benefits from novel architectures in its provider models
- **awesome-world-models** lists it as a Rust ML resource
- **World Model Weekly** covers the paper and implementation

---

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Rust version | 1.80+ | nightly (for some burn features) |
| RAM | 8GB | 16GB |
| GPU (for CUDA backend) | RTX 3060 (12GB) | RTX 4070+ |
| GPU (for wgpu backend) | Any Vulkan/Metal GPU | Apple M-series recommended |
| CPU only | Works (ndarray backend) | Slow for training, fine for testing |
| Disk | 2GB (dependencies) | 5GB (with datasets) |
| Time to build | 5-7 days | 7-10 days (with polish) |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Core algorithm correct (differential test passes) | Day 3 |
| Training shows AttnRes < Standard loss | Day 3 |
| Published to crates.io | Day 7 |
| GitHub stars (week 1) | 100-300 |
| GitHub stars (month 1) | 300-800 |
| Cited/retweeted by Kimi team | Week 1-2 |
| Listed in burn framework showcase | Week 2-4 |
| External contributor | Month 1-2 |
