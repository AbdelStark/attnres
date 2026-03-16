/// Two-phase inference optimization for Block AttnRes.
///
/// Phase 1 (parallel): Batch all S pseudo-queries within a block against
/// cached block representations. Returns outputs + softmax statistics.
///
/// Phase 2 (sequential): For each layer, compute intra-block attention
/// against the evolving partial sum, then merge with Phase 1 outputs
/// via online softmax.
///
/// Paper reference: Algorithm 1, Section 4.1.
use burn::prelude::*;

use crate::attn_res_op::AttnResOp;

/// Result from Phase 1: batched inter-block attention.
pub struct Phase1Result<B: Backend> {
    /// Weighted outputs for each query [S tensors of [B, T, D]].
    pub outputs: Vec<Tensor<B, 3>>,
    /// Max logit per query for online softmax merge [S tensors of [B, T]].
    pub max_logits: Vec<Tensor<B, 2>>,
    /// Sum of exp(logits - max) per query [S tensors of [B, T]].
    pub sum_exp: Vec<Tensor<B, 2>>,
}

/// Compute Phase 1: batched inter-block attention for all layers in a block.
///
/// All S pseudo-queries are batched against the N cached block representations.
///
/// # Arguments
/// * `ops` - The S AttnResOp modules (one per sublayer in the block)
/// * `blocks` - The N completed block representations [each [B, T, D]]
///
/// # Returns
/// * Phase1Result with outputs and softmax statistics for online merge
pub fn phase1_batched<B: Backend>(
    ops: &[&AttnResOp<B>],
    blocks: &[Tensor<B, 3>],
) -> Phase1Result<B> {
    if blocks.is_empty() {
        // No inter-block context; return zeros
        let s = ops.len();
        return Phase1Result {
            outputs: Vec::with_capacity(s),
            max_logits: Vec::with_capacity(s),
            sum_exp: Vec::with_capacity(s),
        };
    }

    // Stack blocks: V = [N, B, T, D]
    let v = Tensor::stack(blocks.to_vec(), 0);

    let mut outputs = Vec::with_capacity(ops.len());
    let mut max_logits = Vec::with_capacity(ops.len());
    let mut sum_exp = Vec::with_capacity(ops.len());

    for op in ops {
        // Exact equivalence requires using each AttnResOp's own RMSNorm.
        let k = op.norm.forward_4d(v.clone());

        // Compute logits for this query against all blocks
        let w = op
            .pseudo_query
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0); // [1, 1, 1, D]
        let logits = (k * w).sum_dim(3).squeeze_dim::<3>(3); // [N, B, T]

        // Compute softmax statistics for online merge
        let max_l = logits.clone().max_dim(0).squeeze_dim::<2>(0); // [B, T]
        let shifted = logits.clone() - max_l.clone().unsqueeze_dim::<3>(0); // [N, B, T]
        let exp_shifted = shifted.exp(); // [N, B, T]
        let sum_e = exp_shifted.clone().sum_dim(0).squeeze_dim::<2>(0); // [B, T]

        // Weighted output (unnormalized)
        let alpha = exp_shifted.unsqueeze_dim::<4>(3); // [N, B, T, 1]
        let weighted = (v.clone() * alpha).sum_dim(0).squeeze_dim::<3>(0); // [B, T, D]

        outputs.push(weighted);
        max_logits.push(max_l);
        sum_exp.push(sum_e);
    }

    Phase1Result {
        outputs,
        max_logits,
        sum_exp,
    }
}

/// Normalize an inter-block Phase 1 output when there is no intra-block source.
///
/// This corresponds to Algorithm 1, line 8: `h_l <- o_l / l_l`.
pub fn normalize_inter_output<B: Backend>(
    inter_output: Tensor<B, 3>,
    inter_sum_exp: Tensor<B, 2>,
) -> Tensor<B, 3> {
    inter_output / inter_sum_exp.unsqueeze_dim::<3>(2)
}

/// Merge Phase 1 inter-block result with a new intra-block logit and value
/// using the online softmax technique.
///
/// Paper reference: Algorithm 1, line 12.
///
/// # Arguments
/// * `inter_output` - Unnormalized weighted output from Phase 1 [B, T, D]
/// * `inter_max` - Max logit from Phase 1 [B, T]
/// * `inter_sum_exp` - Sum of exp from Phase 1 [B, T]
/// * `intra_logit` - Logit for the intra-block value [B, T]
/// * `intra_value` - The intra-block partial sum [B, T, D]
///
/// # Returns
/// * Merged attention output [B, T, D]
pub fn online_softmax_merge<B: Backend>(
    inter_output: Tensor<B, 3>,
    inter_max: Tensor<B, 2>,
    inter_sum_exp: Tensor<B, 2>,
    intra_logit: Tensor<B, 2>,
    intra_value: Tensor<B, 3>,
) -> Tensor<B, 3> {
    // New global max
    let m = inter_max.clone().max_pair(intra_logit.clone()); // [B, T]

    // Re-scale inter-block contribution
    let inter_scale = (inter_max - m.clone()).exp(); // [B, T]
    let inter_scaled_sum = inter_sum_exp * inter_scale.clone(); // [B, T]
    let inter_scaled_out = inter_output * inter_scale.unsqueeze_dim::<3>(2); // [B, T, D]

    // Intra-block contribution
    let intra_scale = (intra_logit - m).exp(); // [B, T]
    let intra_scaled_out = intra_value * intra_scale.clone().unsqueeze_dim::<3>(2); // [B, T, D]

    // Normalize
    let total = inter_scaled_sum + intra_scale; // [B, T]
    (inter_scaled_out + intra_scaled_out) / total.unsqueeze_dim::<3>(2) // [B, T, D]
}

/// Compute the intra-block logit for a pseudo-query against a partial sum.
///
/// Returns one logit per batch element and token position: `[B, T]`.
pub fn compute_intra_logit<B: Backend>(op: &AttnResOp<B>, partial: &Tensor<B, 3>) -> Tensor<B, 2> {
    let normed = op.norm.forward(partial.clone()); // [B, T, D]
    let w = op
        .pseudo_query
        .val()
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0); // [1, 1, D]
    (normed * w).sum_dim(2).squeeze_dim::<2>(2) // [B, T]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type TestBackend = NdArray;

    #[test]
    fn test_phase1_output_count() {
        let device = Default::default();
        let config = crate::config::AttnResConfig::new(32, 4, 2);

        let op1 = config.init_op::<TestBackend>(&device);
        let op2 = config.init_op::<TestBackend>(&device);

        let blocks = vec![
            Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device),
            Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device),
        ];

        let result = phase1_batched(&[&op1, &op2], &blocks);
        assert_eq!(result.outputs.len(), 2);
        assert_eq!(result.max_logits.len(), 2);
        assert_eq!(result.sum_exp.len(), 2);
        assert_eq!(result.outputs[0].dims(), [1, 8, 32]);
    }

    #[test]
    fn test_compute_intra_logit_shape() {
        let device = Default::default();
        let config = crate::config::AttnResConfig::new(32, 4, 2);
        let op = config.init_op::<TestBackend>(&device);

        let partial = Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device);
        let logit = compute_intra_logit(&op, &partial);
        assert_eq!(logit.dims(), [1, 8]);
    }

    #[test]
    fn test_phase1_inter_only_matches_blocks_only_forward() {
        let device = Default::default();
        let config = crate::config::AttnResConfig::new(32, 4, 2);
        let op = config.init_op::<TestBackend>(&device);

        let blocks = vec![
            Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device),
            Tensor::random([1, 8, 32], Distribution::Normal(0.0, 1.0), &device),
        ];

        let standard = op.forward_optional_partial(&blocks, None);
        let phase1 = phase1_batched(&[&op], &blocks);
        let inter_only =
            normalize_inter_output(phase1.outputs[0].clone(), phase1.sum_exp[0].clone());

        let diff: f32 = (standard - inter_only).abs().max().into_scalar();
        assert!(
            diff < 1e-5,
            "Phase 1 inter-only output should match blocks-only forward, diff={diff}"
        );
    }
}
