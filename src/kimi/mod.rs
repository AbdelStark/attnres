//! Staged Kimi Linear support.
//!
//! The repository now contains two concrete Kimi delivery slices:
//! - RFC 0001 Phase A artifact-understanding surfaces for config/index/import
//!   metadata.
//! - RFC 0002 baseline Kimi Linear execution scaffolding with typed MLA/KDA
//!   layers, dense-vs-MoE placement, and separate decode-cache families.
//! - RFC 0003 sharded checkpoint-import scaffolding: tensor locators, explicit
//!   tensor-to-module coverage reports, dtype policy, selected-layer shard
//!   planning, and shard-path resolution without fake tensor payload loading.
//! - RFC 0004 AttnRes-Kimi execution scaffolding with a separate model/layer
//!   path that preserves baseline Kimi sublayer selection while inserting two
//!   AttnRes operations per decoder layer.
//! - RFC 0005 local validation/benchmark scaffolding: baseline-only Gate 1
//!   fixture-backed parity for a deterministic tiny-random Kimi-style bundle,
//!   Gate 4 functional tests, reduced-config Gate 5 hidden/logit agreement,
//!   and reduced local benchmark groups for baseline Kimi plus AttnRes-Kimi.
//!
//! Public-checkpoint tensor parity, full payload loading, optimized kernels,
//! and public Hugging Face/Python parity work remain deferred.

pub mod attention;
pub mod attn_res_layer;
pub mod attn_res_model;
pub mod attn_res_state;
pub mod cache;
pub mod config;
pub mod import;
pub mod index;
pub mod layer;
pub mod mlp;
pub mod model;
pub mod moe;
pub mod phase;
pub mod schedule;

pub use attention::{KimiKdaAttention, KimiMlaAttention};
pub use attn_res_layer::KimiAttnResDecoderLayer;
pub use attn_res_model::KimiAttnResModel;
pub use attn_res_state::{KimiAttnResBlockState, KimiAttnResStateError};
pub use cache::{KimiCacheError, KimiDecodeCache, KimiKdaCache, KimiLayerCache, KimiMlaCache};
pub use config::{
    KimiArtifactConfig, KimiArtifactConfigError, KimiAttentionRuntimeConfig, KimiAttnResConfig,
    KimiAttnResConfigError, KimiBaselineConfig, KimiDenseMlpRuntimeConfig,
    KimiLinearAttentionConfig, KimiSparseMoeRuntimeConfig,
};
pub use import::{
    KimiArtifactUnderstanding, KimiDuplicateTensor, KimiImportCoverageError,
    KimiImportCoverageReport, KimiImportDtypeAction, KimiImportDtypePolicy, KimiImportError,
    KimiImportMode, KimiImportPlan, KimiImportReport, KimiImportRuntimeDtype, KimiImportSelection,
    KimiLayerModuleRef, KimiMissingTensor, KimiModuleCoverage, KimiModuleRef, KimiResolvedShard,
    KimiShardResolver, KimiShardResolverError, KimiTensorMapping, KimiTensorMappingStatus,
    KimiUnsupportedTensorReason,
};
pub use index::{
    KimiShardIndex, KimiShardIndexError, KimiShardIndexMetadata, KimiTensorLocation,
    KimiTensorLocator, KimiTensorLocatorError,
};
pub use layer::KimiDecoderLayer;
pub use mlp::KimiDenseMlp;
pub use model::KimiLinearModel;
pub use moe::KimiSparseMoe;
pub use phase::{KimiMilestonePhase, KIMI_ARTIFACT_UNDERSTANDING_PHASE, KIMI_IMPLEMENTED_PHASE};
pub use schedule::{
    KimiAttentionLayerKind, KimiFeedForwardLayerKind, KimiLayerSchedule, KimiLayerScheduleError,
    KimiScheduledLayer,
};
