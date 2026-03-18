use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::Path;

use crate::config::{AttnResConfig, ConfigError};
use crate::kimi::schedule::{
    KimiAttentionLayerKind, KimiFeedForwardLayerKind, KimiLayerSchedule, KimiLayerScheduleError,
};

fn default_moe_renormalize() -> bool {
    true
}

fn default_moe_router_activation_func() -> String {
    "sigmoid".to_string()
}

fn default_routed_scaling_factor() -> f64 {
    1.0
}

fn default_use_grouped_topk() -> bool {
    true
}

fn default_num_expert_group() -> usize {
    1
}

fn default_topk_group() -> usize {
    1
}

/// Typed subset of `linear_attn_config` from the public Kimi artifact config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiLinearAttentionConfig {
    pub full_attn_layers: Vec<usize>,
    pub kda_layers: Vec<usize>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub short_conv_kernel_size: usize,
}

/// Typed subset of Hugging Face `config.json` needed for RFC 0002 baseline Kimi.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiArtifactConfig {
    pub model_type: String,
    pub dtype: String,
    pub vocab_size: usize,
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
    #[serde(default = "default_moe_renormalize")]
    pub moe_renormalize: bool,
    #[serde(default = "default_moe_router_activation_func")]
    pub moe_router_activation_func: String,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    #[serde(default = "default_use_grouped_topk")]
    pub use_grouped_topk: bool,
    #[serde(default = "default_num_expert_group")]
    pub num_expert_group: usize,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub rms_norm_eps: f64,
    pub linear_attn_config: KimiLinearAttentionConfig,
}

/// Runtime view of the Kimi attention parameter surface used by the local baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttentionRuntimeConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub mla_use_nope: bool,
    pub rms_norm_eps: f64,
    pub linear_attention_num_heads: usize,
    pub linear_attention_head_dim: usize,
    pub linear_attention_short_conv_kernel_size: usize,
}

impl KimiAttentionRuntimeConfig {
    pub fn mla_qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    pub fn kv_repeat_factor(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn kda_conv_history_len(&self) -> usize {
        self.linear_attention_short_conv_kernel_size
            .saturating_sub(1)
    }
}

/// Dense SiLU MLP runtime parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiDenseMlpRuntimeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
}

/// Sparse MoE runtime parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiSparseMoeRuntimeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub num_shared_experts: usize,
    pub moe_renormalize: bool,
    pub moe_router_activation_func: String,
    pub routed_scaling_factor: f64,
    pub use_grouped_topk: bool,
    pub num_expert_group: usize,
    pub topk_group: usize,
    pub hidden_act: String,
}

/// Typed, validated runtime config used by the local RFC 0002 baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiBaselineConfig {
    pub artifact: KimiArtifactConfig,
    pub layer_schedule: KimiLayerSchedule,
    pub attention: KimiAttentionRuntimeConfig,
    pub dense_mlp: KimiDenseMlpRuntimeConfig,
    pub sparse_moe: KimiSparseMoeRuntimeConfig,
}

/// Typed runtime config for the RFC 0004 AttnRes-Kimi execution scaffold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KimiAttnResConfig {
    pub baseline: KimiBaselineConfig,
    pub num_blocks: usize,
}

impl KimiBaselineConfig {
    pub fn try_validate(&self) -> Result<(), KimiArtifactConfigError> {
        self.artifact.try_validate()?;
        if self.layer_schedule != self.artifact.try_layer_schedule()? {
            return Err(KimiArtifactConfigError::LayerScheduleMismatchWithArtifact);
        }
        Ok(())
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.artifact.num_hidden_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.artifact.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.artifact.vocab_size
    }

    pub fn rms_norm_eps(&self) -> f64 {
        self.artifact.rms_norm_eps
    }

    pub fn use_cache(&self) -> bool {
        self.artifact.use_cache
    }

    pub fn try_layer_attention_kind(
        &self,
        layer_idx: usize,
    ) -> Result<KimiAttentionLayerKind, KimiArtifactConfigError> {
        self.layer_schedule
            .try_attention_kind(layer_idx)
            .map_err(KimiArtifactConfigError::from)
    }

    pub fn try_layer_feed_forward_kind(
        &self,
        layer_idx: usize,
    ) -> Result<KimiFeedForwardLayerKind, KimiArtifactConfigError> {
        self.layer_schedule
            .try_feed_forward_kind(layer_idx)
            .map_err(KimiArtifactConfigError::from)
    }

    pub fn try_attn_res_config(
        &self,
        num_blocks: usize,
    ) -> Result<KimiAttnResConfig, KimiAttnResConfigError> {
        let config = KimiAttnResConfig {
            baseline: self.clone(),
            num_blocks,
        };
        config.try_validate()?;
        Ok(config)
    }
}

impl KimiAttnResConfig {
    pub fn try_validate(&self) -> Result<(), KimiAttnResConfigError> {
        self.baseline.try_validate()?;
        self.try_attn_res_core_config()?;
        Ok(())
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.baseline.num_hidden_layers()
    }

    pub fn num_sublayers(&self) -> usize {
        self.num_hidden_layers() * 2
    }

    pub fn hidden_size(&self) -> usize {
        self.baseline.hidden_size()
    }

    pub fn vocab_size(&self) -> usize {
        self.baseline.vocab_size()
    }

    pub fn rms_norm_eps(&self) -> f64 {
        self.baseline.rms_norm_eps()
    }

    pub fn use_cache(&self) -> bool {
        self.baseline.use_cache()
    }

    pub fn try_attn_res_core_config(&self) -> Result<AttnResConfig, KimiAttnResConfigError> {
        let attention_heads = self.baseline.attention.num_attention_heads;
        let config = AttnResConfig::new(self.hidden_size(), self.num_sublayers(), self.num_blocks)
            .with_num_heads(attention_heads)
            .with_vocab_size(self.vocab_size())
            .with_rms_norm_eps(self.rms_norm_eps());
        config.try_validate()?;
        Ok(config)
    }

    pub fn attn_res_core_config(&self) -> AttnResConfig {
        self.try_attn_res_core_config()
            .unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn block_size(&self) -> usize {
        self.attn_res_core_config().block_size()
    }
}

/// Typed validation failures for [`KimiArtifactConfig`].
#[derive(Debug, Clone, PartialEq)]
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
    VocabSizeMustBePositive,
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
    MoeRouterActivationUnsupported {
        moe_router_activation_func: String,
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
    RoutedScalingFactorMustBePositive {
        routed_scaling_factor: f64,
    },
    NumExpertGroupMustBePositive,
    NumExpertsMustBeDivisibleByNumExpertGroup {
        num_experts: usize,
        num_expert_group: usize,
    },
    TopkGroupMustBePositive,
    TopkGroupExceedsNumExpertGroup {
        topk_group: usize,
        num_expert_group: usize,
    },
    RmsNormEpsMustBePositive {
        rms_norm_eps: f64,
    },
    LinearAttentionNumHeadsMustBePositive,
    LinearAttentionHeadDimMustBePositive,
    LinearAttentionShortConvKernelSizeMustBePositive,
    LinearAttentionNumHeadsMustMatchAttentionHeads {
        linear_attn_num_heads: usize,
        num_attention_heads: usize,
    },
    LinearAttentionHeadDimMustMatchValueHeadDim {
        linear_attn_head_dim: usize,
        value_head_dim: usize,
    },
    LayerSchedule(KimiLayerScheduleError),
    LayerScheduleMismatchWithArtifact,
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
            Self::VocabSizeMustBePositive => write!(f, "vocab_size must be positive, got 0"),
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
            Self::MoeRouterActivationUnsupported {
                moe_router_activation_func,
            } => write!(
                f,
                "moe_router_activation_func must be \"sigmoid\" or \"softmax\", got \"{moe_router_activation_func}\""
            ),
            Self::TiedWordEmbeddingsUnsupported => {
                write!(f, "tie_word_embeddings = true is out of scope for Kimi Linear RFC 0002")
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
            Self::RoutedScalingFactorMustBePositive {
                routed_scaling_factor,
            } => write!(
                f,
                "routed_scaling_factor must be positive, got {routed_scaling_factor}"
            ),
            Self::NumExpertGroupMustBePositive => {
                write!(f, "num_expert_group must be positive, got 0")
            }
            Self::NumExpertsMustBeDivisibleByNumExpertGroup {
                num_experts,
                num_expert_group,
            } => write!(
                f,
                "num_experts ({num_experts}) must be divisible by num_expert_group ({num_expert_group})"
            ),
            Self::TopkGroupMustBePositive => {
                write!(f, "topk_group must be positive, got 0")
            }
            Self::TopkGroupExceedsNumExpertGroup {
                topk_group,
                num_expert_group,
            } => write!(
                f,
                "topk_group ({topk_group}) must be <= num_expert_group ({num_expert_group})"
            ),
            Self::RmsNormEpsMustBePositive { rms_norm_eps } => {
                write!(f, "rms_norm_eps must be positive, got {rms_norm_eps}")
            }
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
            Self::LinearAttentionHeadDimMustMatchValueHeadDim {
                linear_attn_head_dim,
                value_head_dim,
            } => write!(
                f,
                "linear_attn_config.head_dim ({linear_attn_head_dim}) must match v_head_dim ({value_head_dim}) in the RFC 0002 local KDA scaffold"
            ),
            Self::LayerSchedule(err) => write!(f, "{err}"),
            Self::LayerScheduleMismatchWithArtifact => write!(
                f,
                "baseline layer schedule drifted from artifact-derived schedule; this is a bug in KimiBaselineConfig construction"
            ),
        }
    }
}

impl std::error::Error for KimiArtifactConfigError {}

impl From<KimiLayerScheduleError> for KimiArtifactConfigError {
    fn from(err: KimiLayerScheduleError) -> Self {
        Self::LayerSchedule(err)
    }
}

/// Typed validation failures for [`KimiAttnResConfig`].
#[derive(Debug, Clone, PartialEq)]
pub enum KimiAttnResConfigError {
    Baseline(KimiArtifactConfigError),
    AttnRes(ConfigError),
}

impl Display for KimiAttnResConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Baseline(err) => write!(f, "{err}"),
            Self::AttnRes(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for KimiAttnResConfigError {}

impl From<KimiArtifactConfigError> for KimiAttnResConfigError {
    fn from(err: KimiArtifactConfigError) -> Self {
        Self::Baseline(err)
    }
}

impl From<ConfigError> for KimiAttnResConfigError {
    fn from(err: ConfigError) -> Self {
        Self::AttnRes(err)
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
        if self.vocab_size == 0 {
            return Err(KimiArtifactConfigError::VocabSizeMustBePositive);
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
        if !matches!(self.moe_router_activation_func.as_str(), "sigmoid" | "softmax") {
            return Err(KimiArtifactConfigError::MoeRouterActivationUnsupported {
                moe_router_activation_func: self.moe_router_activation_func.clone(),
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
        if !self.routed_scaling_factor.is_finite() || self.routed_scaling_factor <= 0.0 {
            return Err(KimiArtifactConfigError::RoutedScalingFactorMustBePositive {
                routed_scaling_factor: self.routed_scaling_factor,
            });
        }
        if self.num_expert_group == 0 {
            return Err(KimiArtifactConfigError::NumExpertGroupMustBePositive);
        }
        if !self.num_experts.is_multiple_of(self.num_expert_group) {
            return Err(
                KimiArtifactConfigError::NumExpertsMustBeDivisibleByNumExpertGroup {
                    num_experts: self.num_experts,
                    num_expert_group: self.num_expert_group,
                },
            );
        }
        if self.topk_group == 0 {
            return Err(KimiArtifactConfigError::TopkGroupMustBePositive);
        }
        if self.topk_group > self.num_expert_group {
            return Err(KimiArtifactConfigError::TopkGroupExceedsNumExpertGroup {
                topk_group: self.topk_group,
                num_expert_group: self.num_expert_group,
            });
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(KimiArtifactConfigError::RmsNormEpsMustBePositive {
                rms_norm_eps: self.rms_norm_eps,
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
        if self.linear_attn_config.head_dim != self.v_head_dim {
            return Err(
                KimiArtifactConfigError::LinearAttentionHeadDimMustMatchValueHeadDim {
                    linear_attn_head_dim: self.linear_attn_config.head_dim,
                    value_head_dim: self.v_head_dim,
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

    pub fn try_baseline_config(&self) -> Result<KimiBaselineConfig, KimiArtifactConfigError> {
        self.try_validate()?;

        Ok(KimiBaselineConfig {
            artifact: self.clone(),
            layer_schedule: self.try_layer_schedule()?,
            attention: KimiAttentionRuntimeConfig {
                hidden_size: self.hidden_size,
                num_attention_heads: self.num_attention_heads,
                num_key_value_heads: self.num_key_value_heads,
                head_dim: self.head_dim,
                kv_lora_rank: self.kv_lora_rank,
                q_lora_rank: self.q_lora_rank,
                qk_nope_head_dim: self.qk_nope_head_dim,
                qk_rope_head_dim: self.qk_rope_head_dim,
                v_head_dim: self.v_head_dim,
                mla_use_nope: self.mla_use_nope,
                rms_norm_eps: self.rms_norm_eps,
                linear_attention_num_heads: self.linear_attn_config.num_heads,
                linear_attention_head_dim: self.linear_attn_config.head_dim,
                linear_attention_short_conv_kernel_size: self
                    .linear_attn_config
                    .short_conv_kernel_size,
            },
            dense_mlp: KimiDenseMlpRuntimeConfig {
                hidden_size: self.hidden_size,
                intermediate_size: self.intermediate_size,
                hidden_act: self.hidden_act.clone(),
            },
            sparse_moe: KimiSparseMoeRuntimeConfig {
                hidden_size: self.hidden_size,
                intermediate_size: self.moe_intermediate_size,
                num_experts: self.num_experts,
                num_experts_per_token: self.num_experts_per_token,
                num_shared_experts: self.num_shared_experts,
                moe_renormalize: self.moe_renormalize,
                moe_router_activation_func: self.moe_router_activation_func.clone(),
                routed_scaling_factor: self.routed_scaling_factor,
                use_grouped_topk: self.use_grouped_topk,
                num_expert_group: self.num_expert_group,
                topk_group: self.topk_group,
                hidden_act: self.hidden_act.clone(),
            },
        })
    }

    pub fn try_attn_res_config(
        &self,
        num_blocks: usize,
    ) -> Result<KimiAttnResConfig, KimiAttnResConfigError> {
        self.try_baseline_config()?.try_attn_res_config(num_blocks)
    }
}
