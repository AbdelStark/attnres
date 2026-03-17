//! RFC 0001 Kimi milestone scaffolding.
//!
//! This module currently implements Phase A "artifact understanding":
//! typed config parsing, typed layer-schedule decoding, shard-index metadata,
//! and explicit import planning/report surfaces.
//!
//! It intentionally does not provide runnable Kimi execution, checkpoint
//! loading, or AttnRes-Kimi model paths yet. Those are deferred to later RFCs.

pub mod config;
pub mod import;
pub mod index;
pub mod phase;
pub mod schedule;

pub use config::{KimiArtifactConfig, KimiArtifactConfigError, KimiLinearAttentionConfig};
pub use import::{
    KimiArtifactUnderstanding, KimiImportError, KimiImportMode, KimiImportPlan, KimiImportReport,
    KimiImportSelection,
};
pub use index::{KimiShardIndex, KimiShardIndexError, KimiShardIndexMetadata};
pub use phase::{KimiMilestonePhase, KIMI_IMPLEMENTED_PHASE};
pub use schedule::{
    KimiAttentionLayerKind, KimiFeedForwardLayerKind, KimiLayerSchedule, KimiLayerScheduleError,
    KimiScheduledLayer,
};
