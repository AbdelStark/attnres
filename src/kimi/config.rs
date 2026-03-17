use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::Path;

use crate::kimi::schedule::{KimiLayerSchedule, KimiLayerScheduleError};

/// Typed subset of `linear_attn_config` from the public Kimi artifact config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiLinearAttentionConfig {
    pub full_attn_layers: Vec<usize>,
    pub kda_layers: Vec<usize>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub short_conv_kernel_size: usize,
}

/// Typed subset of Hugging Face `config.json` needed for RFC 0001 staging.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiArtifactConfig {
    pub model_type: String,
    pub dtype: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub mla_use_nope: bool,
    pub hidden_act: String,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: usize,
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub num_shared_experts: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub linear_attn_config: KimiLinearAttentionConfig,
}

/// Typed validation failures for [`KimiArtifactConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiArtifactConfigError {
    ReadFailed {
        path: String,
        detail: String,
    },
    ParseFailed {
        detail: String,
    },
    UnsupportedModelType {
        model_type: String,
    },
    DtypeMustNotBeEmpty,
    HiddenSizeMustBePositive,
    IntermediateSizeMustBePositive,
    MoeIntermediateSizeMustBePositive,
    NumHiddenLayersMustBePositive,
    NumAttentionHeadsMustBePositive,
    NumKeyValueHeadsMustBePositive,
    NumAttentionHeadsMustBeDivisibleByNumKeyValueHeads {
        num_attention_heads: usize,
        num_key_value_heads: usize,
    },
    HeadDimMustBePositive,
    KvLoraRankMustBePositive,
    QkNopeHeadDimMustBePositive,
    QkRopeHeadDimMustBePositive,
    VHeadDimMustBePositive,
    HiddenActUnsupported {
        hidden_act: String,
    },
    TiedWordEmbeddingsUnsupported,
    NumExpertsMustBePositive,
    NumExpertsPerTokenMustBePositive,
    NumExpertsPerTokenExceedsNumExperts {
        num_experts_per_token: usize,
        num_experts: usize,
    },
    NumSharedExpertsExceedsNumExperts {
        num_shared_experts: usize,
        num_experts: usize,
    },
    LinearAttentionNumHeadsMustBePositive,
    LinearAttentionHeadDimMustBePositive,
    LinearAttentionShortConvKernelSizeMustBePositive,
    LinearAttentionNumHeadsMustMatchAttentionHeads {
        linear_attn_num_heads: usize,
        num_attention_heads: usize,
    },
    LayerSchedule(KimiLayerScheduleError),
}

impl Display for KimiArtifactConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadFailed { path, detail } => {
                write!(f, "failed to read Kimi config '{path}': {detail}")
            }
            Self::ParseFailed { detail } => write!(f, "failed to parse Kimi config JSON: {detail}"),
            Self::UnsupportedModelType { model_type } => write!(
                f,
                "expected model_type = \"kimi_linear\", got \"{model_type}\""
            ),
            Self::DtypeMustNotBeEmpty => write!(f, "dtype must not be empty"),
            Self::HiddenSizeMustBePositive => write!(f, "hidden_size must be positive, got 0"),
            Self::IntermediateSizeMustBePositive => {
                write!(f, "intermediate_size must be positive, got 0")
            }
            Self::MoeIntermediateSizeMustBePositive => {
                write!(f, "moe_intermediate_size must be positive, got 0")
            }
            Self::NumHiddenLayersMustBePositive => {
                write!(f, "num_hidden_layers must be positive, got 0")
            }
            Self::NumAttentionHeadsMustBePositive => {
                write!(f, "num_attention_heads must be positive, got 0")
            }
            Self::NumKeyValueHeadsMustBePositive => {
                write!(f, "num_key_value_heads must be positive, got 0")
            }
            Self::NumAttentionHeadsMustBeDivisibleByNumKeyValueHeads {
                num_attention_heads,
                num_key_value_heads,
            } => write!(
                f,
                "num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"
            ),
            Self::HeadDimMustBePositive => write!(f, "head_dim must be positive, got 0"),
            Self::KvLoraRankMustBePositive => write!(f, "kv_lora_rank must be positive, got 0"),
            Self::QkNopeHeadDimMustBePositive => {
                write!(f, "qk_nope_head_dim must be positive, got 0")
            }
            Self::QkRopeHeadDimMustBePositive => {
                write!(f, "qk_rope_head_dim must be positive, got 0")
            }
            Self::VHeadDimMustBePositive => write!(f, "v_head_dim must be positive, got 0"),
            Self::HiddenActUnsupported { hidden_act } => {
                write!(f, "hidden_act must be \"silu\" for Kimi Linear, got \"{hidden_act}\"")
            }
            Self::TiedWordEmbeddingsUnsupported => {
                write!(f, "tie_word_embeddings = true is out of scope for Kimi Linear RFC 0001")
            }
            Self::NumExpertsMustBePositive => write!(f, "num_experts must be positive, got 0"),
            Self::NumExpertsPerTokenMustBePositive => {
                write!(f, "num_experts_per_token must be positive, got 0")
            }
            Self::NumExpertsPerTokenExceedsNumExperts {
                num_experts_per_token,
                num_experts,
            } => write!(
                f,
                "num_experts_per_token ({num_experts_per_token}) must be <= num_experts ({num_experts})"
            ),
            Self::NumSharedExpertsExceedsNumExperts {
                num_shared_experts,
                num_experts,
            } => write!(
                f,
                "num_shared_experts ({num_shared_experts}) must be <= num_experts ({num_experts})"
            ),
            Self::LinearAttentionNumHeadsMustBePositive => {
                write!(f, "linear_attn_config.num_heads must be positive, got 0")
            }
            Self::LinearAttentionHeadDimMustBePositive => {
                write!(f, "linear_attn_config.head_dim must be positive, got 0")
            }
            Self::LinearAttentionShortConvKernelSizeMustBePositive => write!(
                f,
                "linear_attn_config.short_conv_kernel_size must be positive, got 0"
            ),
            Self::LinearAttentionNumHeadsMustMatchAttentionHeads {
                linear_attn_num_heads,
                num_attention_heads,
            } => write!(
                f,
                "linear_attn_config.num_heads ({linear_attn_num_heads}) must match num_attention_heads ({num_attention_heads})"
            ),
            Self::LayerSchedule(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for KimiArtifactConfigError {}

impl From<KimiLayerScheduleError> for KimiArtifactConfigError {
    fn from(err: KimiLayerScheduleError) -> Self {
        Self::LayerSchedule(err)
    }
}

impl KimiArtifactConfig {
    pub fn from_json_str(json: &str) -> Result<Self, KimiArtifactConfigError> {
        let config: Self =
            serde_json::from_str(json).map_err(|err| KimiArtifactConfigError::ParseFailed {
                detail: err.to_string(),
            })?;
        config.try_validate()?;
        Ok(config)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, KimiArtifactConfigError> {
        let path = path.as_ref();
        let json =
            std::fs::read_to_string(path).map_err(|err| KimiArtifactConfigError::ReadFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            })?;
        Self::from_json_str(&json)
    }

    pub fn try_validate(&self) -> Result<(), KimiArtifactConfigError> {
        if self.model_type != "kimi_linear" {
            return Err(KimiArtifactConfigError::UnsupportedModelType {
                model_type: self.model_type.clone(),
            });
        }
        if self.dtype.trim().is_empty() {
            return Err(KimiArtifactConfigError::DtypeMustNotBeEmpty);
        }
        if self.hidden_size == 0 {
            return Err(KimiArtifactConfigError::HiddenSizeMustBePositive);
        }
        if self.intermediate_size == 0 {
            return Err(KimiArtifactConfigError::IntermediateSizeMustBePositive);
        }
        if self.moe_intermediate_size == 0 {
            return Err(KimiArtifactConfigError::MoeIntermediateSizeMustBePositive);
        }
        if self.num_hidden_layers == 0 {
            return Err(KimiArtifactConfigError::NumHiddenLayersMustBePositive);
        }
        if self.num_attention_heads == 0 {
            return Err(KimiArtifactConfigError::NumAttentionHeadsMustBePositive);
        }
        if self.num_key_value_heads == 0 {
            return Err(KimiArtifactConfigError::NumKeyValueHeadsMustBePositive);
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(
                KimiArtifactConfigError::NumAttentionHeadsMustBeDivisibleByNumKeyValueHeads {
                    num_attention_heads: self.num_attention_heads,
                    num_key_value_heads: self.num_key_value_heads,
                },
            );
        }
        if self.head_dim == 0 {
            return Err(KimiArtifactConfigError::HeadDimMustBePositive);
        }
        if self.kv_lora_rank == 0 {
            return Err(KimiArtifactConfigError::KvLoraRankMustBePositive);
        }
        if self.qk_nope_head_dim == 0 {
            return Err(KimiArtifactConfigError::QkNopeHeadDimMustBePositive);
        }
        if self.qk_rope_head_dim == 0 {
            return Err(KimiArtifactConfigError::QkRopeHeadDimMustBePositive);
        }
        if self.v_head_dim == 0 {
            return Err(KimiArtifactConfigError::VHeadDimMustBePositive);
        }
        if self.hidden_act != "silu" {
            return Err(KimiArtifactConfigError::HiddenActUnsupported {
                hidden_act: self.hidden_act.clone(),
            });
        }
        if self.tie_word_embeddings {
            return Err(KimiArtifactConfigError::TiedWordEmbeddingsUnsupported);
        }
        if self.num_experts == 0 {
            return Err(KimiArtifactConfigError::NumExpertsMustBePositive);
        }
        if self.num_experts_per_token == 0 {
            return Err(KimiArtifactConfigError::NumExpertsPerTokenMustBePositive);
        }
        if self.num_experts_per_token > self.num_experts {
            return Err(
                KimiArtifactConfigError::NumExpertsPerTokenExceedsNumExperts {
                    num_experts_per_token: self.num_experts_per_token,
                    num_experts: self.num_experts,
                },
            );
        }
        if self.num_shared_experts > self.num_experts {
            return Err(KimiArtifactConfigError::NumSharedExpertsExceedsNumExperts {
                num_shared_experts: self.num_shared_experts,
                num_experts: self.num_experts,
            });
        }
        if self.linear_attn_config.num_heads == 0 {
            return Err(KimiArtifactConfigError::LinearAttentionNumHeadsMustBePositive);
        }
        if self.linear_attn_config.head_dim == 0 {
            return Err(KimiArtifactConfigError::LinearAttentionHeadDimMustBePositive);
        }
        if self.linear_attn_config.short_conv_kernel_size == 0 {
            return Err(KimiArtifactConfigError::LinearAttentionShortConvKernelSizeMustBePositive);
        }
        if self.linear_attn_config.num_heads != self.num_attention_heads {
            return Err(
                KimiArtifactConfigError::LinearAttentionNumHeadsMustMatchAttentionHeads {
                    linear_attn_num_heads: self.linear_attn_config.num_heads,
                    num_attention_heads: self.num_attention_heads,
                },
            );
        }

        self.try_layer_schedule()?;
        Ok(())
    }

    pub fn try_layer_schedule(&self) -> Result<KimiLayerSchedule, KimiArtifactConfigError> {
        Ok(KimiLayerSchedule::try_from_one_based_lists(
            self.num_hidden_layers,
            &self.linear_attn_config.full_attn_layers,
            &self.linear_attn_config.kda_layers,
            self.first_k_dense_replace,
            self.moe_layer_freq,
        )?)
    }
}
