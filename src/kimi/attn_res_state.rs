use burn::prelude::*;
use std::fmt::{Display, Formatter};

/// Typed validation failures for AttnRes-Kimi block state construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiAttnResStateError {
    CompletedBlocksMustNotBeEmpty,
    CompletedBlockShapeMismatch {
        block_idx: usize,
        expected: [usize; 3],
        found: [usize; 3],
    },
    PartialBlockShapeMismatch {
        expected: [usize; 3],
        found: [usize; 3],
    },
}

impl Display for KimiAttnResStateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CompletedBlocksMustNotBeEmpty => {
                write!(f, "AttnRes-Kimi block state requires at least one completed block")
            }
            Self::CompletedBlockShapeMismatch {
                block_idx,
                expected,
                found,
            } => write!(
                f,
                "AttnRes-Kimi completed block {block_idx} shape mismatch: expected {expected:?}, found {found:?}"
            ),
            Self::PartialBlockShapeMismatch { expected, found } => write!(
                f,
                "AttnRes-Kimi partial block shape mismatch: expected {expected:?}, found {found:?}"
            ),
        }
    }
}

impl std::error::Error for KimiAttnResStateError {}

/// Block-attention state for the RFC 0004 AttnRes-Kimi path.
#[derive(Clone, Debug)]
pub struct KimiAttnResBlockState<B: Backend> {
    blocks: Vec<Tensor<B, 3>>,
    partial_block: Option<Tensor<B, 3>>,
}

impl<B: Backend> KimiAttnResBlockState<B> {
    /// Initialize with token embeddings as the embedding block.
    pub fn new(token_embeddings: Tensor<B, 3>) -> Self {
        Self {
            blocks: vec![token_embeddings],
            partial_block: None,
        }
    }

    pub fn try_from_parts(
        blocks: Vec<Tensor<B, 3>>,
        partial_block: Option<Tensor<B, 3>>,
    ) -> Result<Self, KimiAttnResStateError> {
        let state = Self {
            blocks,
            partial_block,
        };
        state.try_validate()?;
        Ok(state)
    }

    #[cfg(test)]
    pub(crate) fn from_parts_unchecked(
        blocks: Vec<Tensor<B, 3>>,
        partial_block: Option<Tensor<B, 3>>,
    ) -> Self {
        Self {
            blocks,
            partial_block,
        }
    }

    pub fn try_validate(&self) -> Result<(), KimiAttnResStateError> {
        let Some(first_block) = self.blocks.first() else {
            return Err(KimiAttnResStateError::CompletedBlocksMustNotBeEmpty);
        };
        let expected = first_block.dims();

        for (block_idx, block) in self.blocks.iter().enumerate().skip(1) {
            let found = block.dims();
            if found != expected {
                return Err(KimiAttnResStateError::CompletedBlockShapeMismatch {
                    block_idx,
                    expected,
                    found,
                });
            }
        }

        if let Some(partial_block) = self.partial_block.as_ref() {
            let found = partial_block.dims();
            if found != expected {
                return Err(KimiAttnResStateError::PartialBlockShapeMismatch { expected, found });
            }
        }

        Ok(())
    }

    pub fn blocks(&self) -> &[Tensor<B, 3>] {
        &self.blocks
    }

    pub fn partial_block(&self) -> Option<&Tensor<B, 3>> {
        self.partial_block.as_ref()
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn into_partial_block(self) -> Tensor<B, 3> {
        self.partial_block.expect(
            "AttnRes-Kimi block state missing partial_block after forward pass; this is a bug in RFC 0004 execution",
        )
    }

    pub(crate) fn take_partial_block(&mut self) -> Option<Tensor<B, 3>> {
        self.partial_block.take()
    }

    pub(crate) fn set_partial_block(&mut self, partial_block: Tensor<B, 3>) {
        self.partial_block = Some(partial_block);
    }

    pub(crate) fn push_completed_block(&mut self, block: Tensor<B, 3>) {
        self.blocks.push(block);
    }

    pub(crate) fn last_completed_block(&self) -> &Tensor<B, 3> {
        self.blocks
            .last()
            .expect("AttnRes-Kimi block state lost the embedding block; this is a bug")
    }

    pub(crate) fn validate_or_panic(&self) {
        if let Err(err) = self.try_validate() {
            panic!("{err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn attn_res_kimi_state_rejects_empty_completed_blocks() {
        let state = KimiAttnResBlockState::<TestBackend>::try_from_parts(Vec::new(), None);
        assert_eq!(
            state.unwrap_err(),
            KimiAttnResStateError::CompletedBlocksMustNotBeEmpty
        );
    }

    #[test]
    fn attn_res_kimi_state_rejects_partial_shape_drift() {
        let device = Default::default();
        let state = KimiAttnResBlockState::<TestBackend>::try_from_parts(
            vec![Tensor::zeros([1, 2, 4], &device)],
            Some(Tensor::zeros([1, 3, 4], &device)),
        );

        assert_eq!(
            state.unwrap_err(),
            KimiAttnResStateError::PartialBlockShapeMismatch {
                expected: [1, 2, 4],
                found: [1, 3, 4],
            }
        );
    }

    #[test]
    #[should_panic(expected = "missing partial_block after forward pass")]
    fn attn_res_kimi_state_panics_when_final_partial_is_missing() {
        let device = Default::default();
        let state = KimiAttnResBlockState::<TestBackend>::new(Tensor::zeros([1, 2, 4], &device));
        let _ = state.into_partial_block();
    }

    #[test]
    #[should_panic(expected = "completed block 1 shape mismatch")]
    fn attn_res_kimi_state_validate_or_panic_reports_corrupted_completed_blocks() {
        let device = Default::default();
        let state = KimiAttnResBlockState::<TestBackend>::from_parts_unchecked(
            vec![
                Tensor::zeros([1, 2, 4], &device),
                Tensor::zeros([1, 3, 4], &device),
            ],
            Some(Tensor::zeros([1, 2, 4], &device)),
        );

        state.validate_or_panic();
    }
}
