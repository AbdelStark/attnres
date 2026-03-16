/// Helper functions for the attnres-rs crate.
use burn::prelude::*;

/// Create a causal attention mask [B, T, T].
///
/// Upper triangle (above diagonal) is filled with -1e9 (effectively -inf for softmax).
/// Lower triangle + diagonal is 0 (allows attention).
pub fn causal_mask<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mask = Tensor::<B, 2>::ones([seq_len, seq_len], device)
        .triu(1)
        .mul_scalar(-1e9);
    // Expand to [B, T, T]
    mask.unsqueeze_dim::<3>(0).repeat_dim(0, batch_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_causal_mask_shape() {
        let device = Default::default();
        let mask = causal_mask::<TestBackend>(2, 8, &device);
        assert_eq!(mask.dims(), [2, 8, 8]);
    }

    #[test]
    fn test_causal_mask_values() {
        let device = Default::default();
        let mask = causal_mask::<TestBackend>(1, 4, &device);
        let data: Vec<f32> = mask.reshape([16]).into_data().to_vec().unwrap();

        // Row 0: [0, -1e9, -1e9, -1e9]
        assert_eq!(data[0], 0.0);
        assert!(data[1] < -1e8);

        // Row 1: [0, 0, -1e9, -1e9]
        assert_eq!(data[4], 0.0);
        assert_eq!(data[5], 0.0);
        assert!(data[6] < -1e8);

        // Row 3: [0, 0, 0, 0]
        assert_eq!(data[12], 0.0);
        assert_eq!(data[13], 0.0);
        assert_eq!(data[14], 0.0);
        assert_eq!(data[15], 0.0);
    }
}
