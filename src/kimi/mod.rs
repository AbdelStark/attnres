//! Staged Kimi Linear support.
//!
//! The repository now contains two concrete Kimi delivery slices:
//! - RFC 0001 Phase A artifact-understanding surfaces for config/index/import
//!   metadata.
//! - RFC 0002 baseline Kimi Linear execution scaffolding with typed MLA/KDA
//!   layers, dense-vs-MoE placement, and separate decode-cache families.
//! - RFC 0003 sharded checkpoint-import scaffolding: tensor locators, explicit
//!   tensor-to-module coverage reports, dtype policy, selected-layer shard
//!   planning, and shard-path resolution.
//! - RFC 0004 AttnRes-Kimi execution scaffolding with a separate model/layer
//!   path that preserves baseline Kimi sublayer selection while inserting two
//!   AttnRes operations per decoder layer.
//! - RFC 0005 local validation/benchmark scaffolding: baseline-only Gate 1
//!   fixture-backed parity for a deterministic tiny-random Kimi-style bundle,
//!   local sharded-payload loading for the supported baseline tensor subset
//!   into both `KimiLinearModel` and `KimiAttnResModel`, baseline-only Gate 2
//!   external slice-fixture consumption for local sharded artifacts,
//!   baseline-only Gate 2 external-generator handoff manifests for that same
//!   supported subset, public-checkpoint module-probe parity against the
//!   official Hugging Face remote-code path for one KDA layer, one MLA layer,
//!   final norm, and LM head, an honest full-checkpoint smoke harness with
//!   blocked-state reporting, Gate 4 functional tests, reduced-config Gate 5
//!   hidden/logit agreement, Gate 6 reduced training-stability validation, and
//!   reduced local benchmark groups for baseline Kimi plus AttnRes-Kimi.
//!
//! Full 48B end-to-end smoke success on real hardware, real-checkpoint
//! AttnRes quality evaluation, optimized kernels, and reportable benchmark
//! conclusions remain deferred.

pub mod attention;
pub mod attn_res_layer;
pub mod attn_res_model;
pub mod attn_res_state;
pub mod bootstrap;
pub mod cache;
pub mod config;
pub mod import;
pub mod index;
pub mod layer;
pub mod mlp;
pub mod model;
pub mod module_probe;
pub mod moe;
pub mod payload;
pub mod phase;
pub mod schedule;
pub mod slice_parity;

pub use attention::{KimiKdaAttention, KimiMlaAttention};
pub use attn_res_layer::KimiAttnResDecoderLayer;
pub use attn_res_model::KimiAttnResModel;
pub use attn_res_state::{KimiAttnResBlockState, KimiAttnResStateError};
pub use bootstrap::{
    KimiAttnResBootstrapLoadResult, KimiAttnResBootstrapParityStatus, KimiAttnResBootstrapPolicy,
    KimiAttnResBootstrapReport,
};
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
pub use module_probe::{
    build_default_module_probe_request, compare_module_probe_fixture_from_dir,
    generate_module_probe_fixture_from_dir, KimiModuleProbeCache, KimiModuleProbeDecodeStep,
    KimiModuleProbeError, KimiModuleProbeFingerprint, KimiModuleProbeFixture,
    KimiModuleProbeFixtureCase, KimiModuleProbeRequest, KimiModuleProbeRequestCase,
    KimiModuleProbeTarget, KimiModuleProbeToleranceSpec, KIMI_MODULE_PROBE_DEFAULT_SEED,
    KIMI_MODULE_PROBE_DEFAULT_SEQUENCE_LEN, KIMI_MODULE_PROBE_FIXTURE_KIND,
    KIMI_MODULE_PROBE_REQUEST_KIND, KIMI_MODULE_PROBE_RUNTIME_DTYPE, KIMI_MODULE_PROBE_VERSION,
};
pub use moe::KimiSparseMoe;
pub use payload::KimiBaselinePayloadError;
pub use phase::{KimiMilestonePhase, KIMI_ARTIFACT_UNDERSTANDING_PHASE, KIMI_IMPLEMENTED_PHASE};
pub use schedule::{
    KimiAttentionLayerKind, KimiFeedForwardLayerKind, KimiLayerSchedule, KimiLayerScheduleError,
    KimiScheduledLayer,
};
pub use slice_parity::{
    compare_baseline_slice_parity_fixture, compare_baseline_slice_parity_fixture_from_dir,
    compare_baseline_slice_parity_fixture_with_manifest,
    compare_baseline_slice_parity_fixture_with_manifest_from_dir,
    KimiBaselineSliceParityArtifactSpec, KimiBaselineSliceParityError,
    KimiBaselineSliceParityFixture, KimiBaselineSliceParityHiddenState,
    KimiBaselineSliceParityPromptResult, KimiBaselineSliceParityPromptSpec,
    KimiBaselineSliceParitySliceSpec, KimiBaselineSliceParityTensor,
    KimiBaselineSliceParityToleranceSpec, KimiBaselineSliceRequestManifest,
    KimiBaselineSliceRequestSpec, KIMI_BASELINE_SLICE_PARITY_FILENAME,
    KIMI_BASELINE_SLICE_PARITY_KIND, KIMI_BASELINE_SLICE_PARITY_VERSION,
    KIMI_BASELINE_SLICE_REQUEST_FILENAME, KIMI_BASELINE_SLICE_REQUEST_KIND,
    KIMI_BASELINE_SLICE_REQUEST_VERSION,
};
