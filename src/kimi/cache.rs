use burn::prelude::*;
use std::fmt::{Display, Formatter};

use crate::kimi::schedule::{KimiAttentionLayerKind, KimiLayerSchedule};

/// Typed failures for Kimi decode-cache construction and use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiCacheError {
    CachingDisabledByConfig,
    LayerIndexOutOfRange {
        layer_idx: usize,
        num_hidden_layers: usize,
    },
    ExpectedMlaLayer {
        layer_idx: usize,
        found: KimiAttentionLayerKind,
    },
    ExpectedKdaLayer {
        layer_idx: usize,
        found: KimiAttentionLayerKind,
    },
    MlaCacheMustTrackAtLeastOneToken,
    MlaCacheBatchMismatch {
        keys_batch: usize,
        values_batch: usize,
    },
    MlaCacheSequenceLengthMismatch {
        keys_seq_len: usize,
        values_seq_len: usize,
    },
    MlaCacheHeadMismatch {
        keys_heads: usize,
        values_heads: usize,
    },
    MlaCacheQkHeadDimMismatch {
        keys_head_dim: usize,
        expected_qk_head_dim: usize,
    },
    MlaCacheValueHeadDimMismatch {
        values_head_dim: usize,
        expected_v_head_dim: usize,
    },
    KdaCacheMustTrackAtLeastOneToken,
    KdaCacheConvStateTooLong {
        history_len: usize,
        max_history_len: usize,
    },
    KdaCacheBatchMismatch {
        conv_batch: usize,
        recurrent_batch: usize,
        running_batch: usize,
    },
    KdaCacheHeadCountMismatch {
        conv_heads: usize,
        recurrent_heads: usize,
        running_heads: usize,
    },
    KdaCacheConvHeadDimMismatch {
        conv_head_dim: usize,
        expected_head_dim: usize,
    },
    KdaCacheRecurrentHeadDimMismatch {
        recurrent_head_dim: usize,
        expected_head_dim: usize,
    },
    KdaCacheRecurrentValueDimMismatch {
        recurrent_value_dim: usize,
        expected_value_dim: usize,
    },
    KdaCacheRunningKeyHeadDimMismatch {
        running_key_head_dim: usize,
        expected_head_dim: usize,
    },
}

impl Display for KimiCacheError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CachingDisabledByConfig => {
                write!(f, "decode cache is disabled by this Kimi config")
            }
            Self::LayerIndexOutOfRange {
                layer_idx,
                num_hidden_layers,
            } => write!(
                f,
                "layer_idx ({layer_idx}) must be < num_hidden_layers ({num_hidden_layers})"
            ),
            Self::ExpectedMlaLayer { layer_idx, found } => write!(
                f,
                "layer {layer_idx} expects MLA cache access but is scheduled as {found:?}"
            ),
            Self::ExpectedKdaLayer { layer_idx, found } => write!(
                f,
                "layer {layer_idx} expects KDA cache access but is scheduled as {found:?}"
            ),
            Self::MlaCacheMustTrackAtLeastOneToken => {
                write!(f, "MLA cache must track at least one decoded token")
            }
            Self::MlaCacheBatchMismatch {
                keys_batch,
                values_batch,
            } => write!(
                f,
                "MLA cache keys batch ({keys_batch}) must match values batch ({values_batch})"
            ),
            Self::MlaCacheSequenceLengthMismatch {
                keys_seq_len,
                values_seq_len,
            } => write!(
                f,
                "MLA cache keys seq_len ({keys_seq_len}) must match values seq_len ({values_seq_len})"
            ),
            Self::MlaCacheHeadMismatch {
                keys_heads,
                values_heads,
            } => write!(
                f,
                "MLA cache keys heads ({keys_heads}) must match values heads ({values_heads})"
            ),
            Self::MlaCacheQkHeadDimMismatch {
                keys_head_dim,
                expected_qk_head_dim,
            } => write!(
                f,
                "MLA cache key head dim ({keys_head_dim}) must match expected qk head dim ({expected_qk_head_dim})"
            ),
            Self::MlaCacheValueHeadDimMismatch {
                values_head_dim,
                expected_v_head_dim,
            } => write!(
                f,
                "MLA cache value head dim ({values_head_dim}) must match expected value head dim ({expected_v_head_dim})"
            ),
            Self::KdaCacheMustTrackAtLeastOneToken => {
                write!(f, "KDA cache must track at least one decoded token")
            }
            Self::KdaCacheConvStateTooLong {
                history_len,
                max_history_len,
            } => write!(
                f,
                "KDA conv history ({history_len}) must be <= max_history_len ({max_history_len})"
            ),
            Self::KdaCacheBatchMismatch {
                conv_batch,
                recurrent_batch,
                running_batch,
            } => write!(
                f,
                "KDA cache batch mismatch: conv={conv_batch}, recurrent={recurrent_batch}, running_key={running_batch}"
            ),
            Self::KdaCacheHeadCountMismatch {
                conv_heads,
                recurrent_heads,
                running_heads,
            } => write!(
                f,
                "KDA cache head mismatch: conv={conv_heads}, recurrent={recurrent_heads}, running_key={running_heads}"
            ),
            Self::KdaCacheConvHeadDimMismatch {
                conv_head_dim,
                expected_head_dim,
            } => write!(
                f,
                "KDA conv-state head dim ({conv_head_dim}) must match expected head dim ({expected_head_dim})"
            ),
            Self::KdaCacheRecurrentHeadDimMismatch {
                recurrent_head_dim,
                expected_head_dim,
            } => write!(
                f,
                "KDA recurrent-state head dim ({recurrent_head_dim}) must match expected head dim ({expected_head_dim})"
            ),
            Self::KdaCacheRecurrentValueDimMismatch {
                recurrent_value_dim,
                expected_value_dim,
            } => write!(
                f,
                "KDA recurrent-state value dim ({recurrent_value_dim}) must match expected value dim ({expected_value_dim})"
            ),
            Self::KdaCacheRunningKeyHeadDimMismatch {
                running_key_head_dim,
                expected_head_dim,
            } => write!(
                f,
                "KDA running-key head dim ({running_key_head_dim}) must match expected head dim ({expected_head_dim})"
            ),
        }
    }
}

impl std::error::Error for KimiCacheError {}

/// Full-attention cache family for MLA layers.
#[derive(Debug, Clone)]
pub struct KimiMlaCache<B: Backend> {
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
    qk_head_dim: usize,
    v_head_dim: usize,
}

impl<B: Backend> KimiMlaCache<B> {
    pub fn try_new(
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
        expected_qk_head_dim: usize,
        expected_v_head_dim: usize,
    ) -> Result<Self, KimiCacheError> {
        let [keys_batch, _keys_heads, keys_seq_len, keys_head_dim] = keys.dims();
        let [values_batch, values_heads, values_seq_len, values_head_dim] = values.dims();

        if keys_seq_len == 0 || values_seq_len == 0 {
            return Err(KimiCacheError::MlaCacheMustTrackAtLeastOneToken);
        }
        if keys_batch != values_batch {
            return Err(KimiCacheError::MlaCacheBatchMismatch {
                keys_batch,
                values_batch,
            });
        }
        if keys_seq_len != values_seq_len {
            return Err(KimiCacheError::MlaCacheSequenceLengthMismatch {
                keys_seq_len,
                values_seq_len,
            });
        }
        if keys.dims()[1] != values_heads {
            return Err(KimiCacheError::MlaCacheHeadMismatch {
                keys_heads: keys.dims()[1],
                values_heads,
            });
        }
        if keys_head_dim != expected_qk_head_dim {
            return Err(KimiCacheError::MlaCacheQkHeadDimMismatch {
                keys_head_dim,
                expected_qk_head_dim,
            });
        }
        if values_head_dim != expected_v_head_dim {
            return Err(KimiCacheError::MlaCacheValueHeadDimMismatch {
                values_head_dim,
                expected_v_head_dim,
            });
        }

        Ok(Self {
            keys,
            values,
            qk_head_dim: expected_qk_head_dim,
            v_head_dim: expected_v_head_dim,
        })
    }

    pub fn append(&self, keys: Tensor<B, 4>, values: Tensor<B, 4>) -> Result<Self, KimiCacheError> {
        let next = Self::try_new(keys, values, self.qk_head_dim, self.v_head_dim)?;
        let [current_batch, current_heads, _, _] = self.keys.dims();
        let [next_batch, next_heads, _, _] = next.keys.dims();

        if current_batch != next_batch {
            return Err(KimiCacheError::MlaCacheBatchMismatch {
                keys_batch: current_batch,
                values_batch: next_batch,
            });
        }
        if current_heads != next_heads {
            return Err(KimiCacheError::MlaCacheHeadMismatch {
                keys_heads: current_heads,
                values_heads: next_heads,
            });
        }

        Self::try_new(
            Tensor::cat(vec![self.keys.clone(), next.keys], 2),
            Tensor::cat(vec![self.values.clone(), next.values], 2),
            self.qk_head_dim,
            self.v_head_dim,
        )
    }

    pub fn keys(&self) -> &Tensor<B, 4> {
        &self.keys
    }

    pub fn values(&self) -> &Tensor<B, 4> {
        &self.values
    }

    pub fn processed_tokens(&self) -> usize {
        self.keys.dims()[2]
    }
}

/// Linear-attention cache family for KDA layers.
#[derive(Debug, Clone)]
pub struct KimiKdaCache<B: Backend> {
    conv_state: Tensor<B, 4>,
    recurrent_state: Tensor<B, 4>,
    running_key_sum: Tensor<B, 3>,
    processed_tokens: usize,
    max_history_len: usize,
    head_dim: usize,
    value_dim: usize,
}

impl<B: Backend> KimiKdaCache<B> {
    pub fn try_new(
        conv_state: Tensor<B, 4>,
        recurrent_state: Tensor<B, 4>,
        running_key_sum: Tensor<B, 3>,
        processed_tokens: usize,
        max_history_len: usize,
        expected_head_dim: usize,
        expected_value_dim: usize,
    ) -> Result<Self, KimiCacheError> {
        let [conv_batch, conv_heads, history_len, conv_head_dim] = conv_state.dims();
        let [recurrent_batch, recurrent_heads, recurrent_head_dim, recurrent_value_dim] =
            recurrent_state.dims();
        let [running_batch, running_heads, running_key_head_dim] = running_key_sum.dims();

        if processed_tokens == 0 {
            return Err(KimiCacheError::KdaCacheMustTrackAtLeastOneToken);
        }
        if history_len > max_history_len {
            return Err(KimiCacheError::KdaCacheConvStateTooLong {
                history_len,
                max_history_len,
            });
        }
        if conv_batch != recurrent_batch || conv_batch != running_batch {
            return Err(KimiCacheError::KdaCacheBatchMismatch {
                conv_batch,
                recurrent_batch,
                running_batch,
            });
        }
        if conv_heads != recurrent_heads || conv_heads != running_heads {
            return Err(KimiCacheError::KdaCacheHeadCountMismatch {
                conv_heads,
                recurrent_heads,
                running_heads,
            });
        }
        if conv_head_dim != expected_head_dim {
            return Err(KimiCacheError::KdaCacheConvHeadDimMismatch {
                conv_head_dim,
                expected_head_dim,
            });
        }
        if recurrent_head_dim != expected_head_dim {
            return Err(KimiCacheError::KdaCacheRecurrentHeadDimMismatch {
                recurrent_head_dim,
                expected_head_dim,
            });
        }
        if recurrent_value_dim != expected_value_dim {
            return Err(KimiCacheError::KdaCacheRecurrentValueDimMismatch {
                recurrent_value_dim,
                expected_value_dim,
            });
        }
        if running_key_head_dim != expected_head_dim {
            return Err(KimiCacheError::KdaCacheRunningKeyHeadDimMismatch {
                running_key_head_dim,
                expected_head_dim,
            });
        }

        Ok(Self {
            conv_state,
            recurrent_state,
            running_key_sum,
            processed_tokens,
            max_history_len,
            head_dim: expected_head_dim,
            value_dim: expected_value_dim,
        })
    }

    pub fn advance(
        &self,
        conv_state: Tensor<B, 4>,
        recurrent_state: Tensor<B, 4>,
        running_key_sum: Tensor<B, 3>,
        new_tokens: usize,
    ) -> Result<Self, KimiCacheError> {
        Self::try_new(
            conv_state,
            recurrent_state,
            running_key_sum,
            self.processed_tokens + new_tokens,
            self.max_history_len,
            self.head_dim,
            self.value_dim,
        )
    }

    pub fn conv_state(&self) -> &Tensor<B, 4> {
        &self.conv_state
    }

    pub fn recurrent_state(&self) -> &Tensor<B, 4> {
        &self.recurrent_state
    }

    pub fn running_key_sum(&self) -> &Tensor<B, 3> {
        &self.running_key_sum
    }

    pub fn processed_tokens(&self) -> usize {
        self.processed_tokens
    }
}

/// Per-layer cache slot, typed from the schedule at construction time.
#[derive(Debug, Clone)]
pub enum KimiLayerCache<B: Backend> {
    Mla(Option<KimiMlaCache<B>>),
    Kda(Option<KimiKdaCache<B>>),
}

impl<B: Backend> KimiLayerCache<B> {
    pub fn kind(&self) -> KimiAttentionLayerKind {
        match self {
            Self::Mla(_) => KimiAttentionLayerKind::FullAttention,
            Self::Kda(_) => KimiAttentionLayerKind::LinearAttentionKda,
        }
    }

    pub fn is_initialized(&self) -> bool {
        match self {
            Self::Mla(state) => state.is_some(),
            Self::Kda(state) => state.is_some(),
        }
    }
}

/// Whole-model decode cache with MLA and KDA state kept in separate families.
#[derive(Debug, Clone)]
pub struct KimiDecodeCache<B: Backend> {
    layers: Vec<KimiLayerCache<B>>,
}

impl<B: Backend> KimiDecodeCache<B> {
    pub fn from_schedule(schedule: &KimiLayerSchedule) -> Self {
        let layers = schedule
            .layers()
            .iter()
            .map(|layer| match layer.attention_kind {
                KimiAttentionLayerKind::FullAttention => KimiLayerCache::Mla(None),
                KimiAttentionLayerKind::LinearAttentionKda => KimiLayerCache::Kda(None),
            })
            .collect();
        Self { layers }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_kind(&self, layer_idx: usize) -> Result<KimiAttentionLayerKind, KimiCacheError> {
        Ok(self.layer(layer_idx)?.kind())
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            match layer {
                KimiLayerCache::Mla(state) => *state = None,
                KimiLayerCache::Kda(state) => *state = None,
            }
        }
    }

    pub fn clear_layer(&mut self, layer_idx: usize) -> Result<(), KimiCacheError> {
        match self.layer_mut(layer_idx)? {
            KimiLayerCache::Mla(state) => *state = None,
            KimiLayerCache::Kda(state) => *state = None,
        }
        Ok(())
    }

    pub fn try_mla(&self, layer_idx: usize) -> Result<Option<&KimiMlaCache<B>>, KimiCacheError> {
        match self.layer(layer_idx)? {
            KimiLayerCache::Mla(state) => Ok(state.as_ref()),
            other => Err(KimiCacheError::ExpectedMlaLayer {
                layer_idx,
                found: other.kind(),
            }),
        }
    }

    pub fn try_kda(&self, layer_idx: usize) -> Result<Option<&KimiKdaCache<B>>, KimiCacheError> {
        match self.layer(layer_idx)? {
            KimiLayerCache::Kda(state) => Ok(state.as_ref()),
            other => Err(KimiCacheError::ExpectedKdaLayer {
                layer_idx,
                found: other.kind(),
            }),
        }
    }

    pub fn update_mla(
        &mut self,
        layer_idx: usize,
        state: KimiMlaCache<B>,
    ) -> Result<(), KimiCacheError> {
        match self.layer_mut(layer_idx)? {
            KimiLayerCache::Mla(slot) => {
                *slot = Some(state);
                Ok(())
            }
            other => Err(KimiCacheError::ExpectedMlaLayer {
                layer_idx,
                found: other.kind(),
            }),
        }
    }

    pub fn update_kda(
        &mut self,
        layer_idx: usize,
        state: KimiKdaCache<B>,
    ) -> Result<(), KimiCacheError> {
        match self.layer_mut(layer_idx)? {
            KimiLayerCache::Kda(slot) => {
                *slot = Some(state);
                Ok(())
            }
            other => Err(KimiCacheError::ExpectedKdaLayer {
                layer_idx,
                found: other.kind(),
            }),
        }
    }

    fn layer(&self, layer_idx: usize) -> Result<&KimiLayerCache<B>, KimiCacheError> {
        self.layers
            .get(layer_idx)
            .ok_or(KimiCacheError::LayerIndexOutOfRange {
                layer_idx,
                num_hidden_layers: self.layers.len(),
            })
    }

    fn layer_mut(&mut self, layer_idx: usize) -> Result<&mut KimiLayerCache<B>, KimiCacheError> {
        let num_hidden_layers = self.layers.len();
        self.layers
            .get_mut(layer_idx)
            .ok_or(KimiCacheError::LayerIndexOutOfRange {
                layer_idx,
                num_hidden_layers,
            })
    }
}
