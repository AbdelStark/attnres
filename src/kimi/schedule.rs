use burn::module::Module;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};

/// Attention path used by a given Kimi decoder layer.
#[derive(Module, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiAttentionLayerKind {
    /// Full-attention MLA layer.
    FullAttention,
    /// KDA linear-attention layer.
    LinearAttentionKda,
}

/// Feed-forward path used by a given Kimi decoder layer.
#[derive(Module, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KimiFeedForwardLayerKind {
    /// Dense MLP layer.
    DenseMlp,
    /// Sparse MoE layer.
    SparseMoe,
}

/// Typed schedule entry for one zero-based Kimi decoder layer.
#[derive(Module, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiScheduledLayer {
    pub layer_idx: usize,
    pub attention_kind: KimiAttentionLayerKind,
    pub feed_forward_kind: KimiFeedForwardLayerKind,
}

/// Zero-based internal layer schedule derived from the external artifact config.
#[derive(Module, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiLayerSchedule {
    layers: Vec<KimiScheduledLayer>,
    full_attention_layers_zero_based: Vec<usize>,
    kda_layers_zero_based: Vec<usize>,
    first_k_dense_replace: usize,
    moe_layer_freq: usize,
}

/// Typed validation failures for [`KimiLayerSchedule`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiLayerScheduleError {
    NumHiddenLayersMustBePositive,
    MoeLayerFreqMustBePositive,
    FirstKDenseReplaceOutOfRange {
        first_k_dense_replace: usize,
        num_hidden_layers: usize,
    },
    OneBasedLayerIndexMustBePositive {
        schedule_name: &'static str,
        position: usize,
    },
    LayerIndexOutOfRange {
        schedule_name: &'static str,
        one_based_layer_idx: usize,
        num_hidden_layers: usize,
    },
    DuplicateLayerIndex {
        schedule_name: &'static str,
        one_based_layer_idx: usize,
    },
    LayerAssignedTwice {
        one_based_layer_idx: usize,
    },
    MissingLayerAssignment {
        one_based_layer_idx: usize,
    },
    LayerIndexOutOfRangeZeroBased {
        layer_idx: usize,
        num_hidden_layers: usize,
    },
}

impl Display for KimiLayerScheduleError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NumHiddenLayersMustBePositive => {
                write!(f, "num_hidden_layers must be positive, got 0")
            }
            Self::MoeLayerFreqMustBePositive => write!(f, "moe_layer_freq must be positive, got 0"),
            Self::FirstKDenseReplaceOutOfRange {
                first_k_dense_replace,
                num_hidden_layers,
            } => write!(
                f,
                "first_k_dense_replace ({first_k_dense_replace}) must be <= num_hidden_layers ({num_hidden_layers})"
            ),
            Self::OneBasedLayerIndexMustBePositive {
                schedule_name,
                position,
            } => write!(
                f,
                "{schedule_name}[{position}] must be 1-based, got 0"
            ),
            Self::LayerIndexOutOfRange {
                schedule_name,
                one_based_layer_idx,
                num_hidden_layers,
            } => write!(
                f,
                "{schedule_name} layer index ({one_based_layer_idx}) must be <= num_hidden_layers ({num_hidden_layers})"
            ),
            Self::DuplicateLayerIndex {
                schedule_name,
                one_based_layer_idx,
            } => write!(
                f,
                "{schedule_name} contains duplicate layer index {one_based_layer_idx}"
            ),
            Self::LayerAssignedTwice { one_based_layer_idx } => write!(
                f,
                "layer {one_based_layer_idx} is assigned to both full-attention and KDA schedules"
            ),
            Self::MissingLayerAssignment { one_based_layer_idx } => write!(
                f,
                "layer {one_based_layer_idx} is missing from both full-attention and KDA schedules"
            ),
            Self::LayerIndexOutOfRangeZeroBased {
                layer_idx,
                num_hidden_layers,
            } => write!(
                f,
                "layer_idx ({layer_idx}) must be < num_hidden_layers ({num_hidden_layers})"
            ),
        }
    }
}

impl std::error::Error for KimiLayerScheduleError {}

impl KimiLayerSchedule {
    /// Build the internal zero-based schedule from the external 1-based artifact lists.
    pub fn try_from_one_based_lists(
        num_hidden_layers: usize,
        full_attention_layers_one_based: &[usize],
        kda_layers_one_based: &[usize],
        first_k_dense_replace: usize,
        moe_layer_freq: usize,
    ) -> Result<Self, KimiLayerScheduleError> {
        if num_hidden_layers == 0 {
            return Err(KimiLayerScheduleError::NumHiddenLayersMustBePositive);
        }
        if moe_layer_freq == 0 {
            return Err(KimiLayerScheduleError::MoeLayerFreqMustBePositive);
        }
        if first_k_dense_replace > num_hidden_layers {
            return Err(KimiLayerScheduleError::FirstKDenseReplaceOutOfRange {
                first_k_dense_replace,
                num_hidden_layers,
            });
        }

        let full_attention_layers_zero_based = decode_one_based_layers(
            "full_attn_layers",
            full_attention_layers_one_based,
            num_hidden_layers,
        )?;
        let kda_layers_zero_based =
            decode_one_based_layers("kda_layers", kda_layers_one_based, num_hidden_layers)?;

        let full_set: BTreeSet<_> = full_attention_layers_zero_based.iter().copied().collect();
        let kda_set: BTreeSet<_> = kda_layers_zero_based.iter().copied().collect();

        if let Some(layer_idx) = full_set.intersection(&kda_set).next() {
            return Err(KimiLayerScheduleError::LayerAssignedTwice {
                one_based_layer_idx: layer_idx + 1,
            });
        }

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for layer_idx in 0..num_hidden_layers {
            let attention_kind = if full_set.contains(&layer_idx) {
                KimiAttentionLayerKind::FullAttention
            } else if kda_set.contains(&layer_idx) {
                KimiAttentionLayerKind::LinearAttentionKda
            } else {
                return Err(KimiLayerScheduleError::MissingLayerAssignment {
                    one_based_layer_idx: layer_idx + 1,
                });
            };

            let feed_forward_kind =
                if layer_idx >= first_k_dense_replace && layer_idx % moe_layer_freq == 0 {
                    KimiFeedForwardLayerKind::SparseMoe
                } else {
                    KimiFeedForwardLayerKind::DenseMlp
                };

            layers.push(KimiScheduledLayer {
                layer_idx,
                attention_kind,
                feed_forward_kind,
            });
        }

        Ok(Self {
            layers,
            full_attention_layers_zero_based,
            kda_layers_zero_based,
            first_k_dense_replace,
            moe_layer_freq,
        })
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layers(&self) -> &[KimiScheduledLayer] {
        &self.layers
    }

    pub fn full_attention_layers_zero_based(&self) -> &[usize] {
        &self.full_attention_layers_zero_based
    }

    pub fn kda_layers_zero_based(&self) -> &[usize] {
        &self.kda_layers_zero_based
    }

    pub fn first_k_dense_replace(&self) -> usize {
        self.first_k_dense_replace
    }

    pub fn moe_layer_freq(&self) -> usize {
        self.moe_layer_freq
    }

    pub fn try_layer(
        &self,
        layer_idx: usize,
    ) -> Result<&KimiScheduledLayer, KimiLayerScheduleError> {
        self.layers
            .get(layer_idx)
            .ok_or(KimiLayerScheduleError::LayerIndexOutOfRangeZeroBased {
                layer_idx,
                num_hidden_layers: self.layers.len(),
            })
    }

    pub fn try_attention_kind(
        &self,
        layer_idx: usize,
    ) -> Result<KimiAttentionLayerKind, KimiLayerScheduleError> {
        Ok(self.try_layer(layer_idx)?.attention_kind)
    }

    pub fn try_feed_forward_kind(
        &self,
        layer_idx: usize,
    ) -> Result<KimiFeedForwardLayerKind, KimiLayerScheduleError> {
        Ok(self.try_layer(layer_idx)?.feed_forward_kind)
    }
}

fn decode_one_based_layers(
    schedule_name: &'static str,
    one_based_layers: &[usize],
    num_hidden_layers: usize,
) -> Result<Vec<usize>, KimiLayerScheduleError> {
    let mut seen = BTreeSet::new();
    let mut decoded = Vec::with_capacity(one_based_layers.len());

    for (position, &one_based_layer_idx) in one_based_layers.iter().enumerate() {
        if one_based_layer_idx == 0 {
            return Err(KimiLayerScheduleError::OneBasedLayerIndexMustBePositive {
                schedule_name,
                position,
            });
        }
        if one_based_layer_idx > num_hidden_layers {
            return Err(KimiLayerScheduleError::LayerIndexOutOfRange {
                schedule_name,
                one_based_layer_idx,
                num_hidden_layers,
            });
        }
        if !seen.insert(one_based_layer_idx) {
            return Err(KimiLayerScheduleError::DuplicateLayerIndex {
                schedule_name,
                one_based_layer_idx,
            });
        }
        decoded.push(one_based_layer_idx - 1);
    }

    Ok(decoded)
}
