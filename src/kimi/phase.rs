use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

/// RFC 0001 milestone phases for the Kimi real-model program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KimiMilestonePhase {
    /// Parse and validate artifact metadata before any execution claims.
    ArtifactUnderstanding,
    /// Build the baseline Kimi architecture locally.
    BaselineImplementation,
    /// Validate baseline numerical parity against the public reference.
    BaselineParity,
    /// Add AttnRes to the Kimi architecture.
    AttnResIntegration,
    /// Run benchmark and training validation with explicit failure criteria.
    BenchmarksAndResearch,
}

impl KimiMilestonePhase {
    pub const fn rfc_label(self) -> &'static str {
        match self {
            Self::ArtifactUnderstanding => "Phase A",
            Self::BaselineImplementation => "Phase B",
            Self::BaselineParity => "Phase C",
            Self::AttnResIntegration => "Phase D",
            Self::BenchmarksAndResearch => "Phase E",
        }
    }

    pub const fn summary(self) -> &'static str {
        match self {
            Self::ArtifactUnderstanding => "artifact understanding",
            Self::BaselineImplementation => "baseline implementation",
            Self::BaselineParity => "baseline parity",
            Self::AttnResIntegration => "AttnRes-Kimi integration",
            Self::BenchmarksAndResearch => "benchmarks and research validation",
        }
    }
}

impl Display for KimiMilestonePhase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.rfc_label(), self.summary())
    }
}

/// Current repository-backed Kimi slice.
pub const KIMI_IMPLEMENTED_PHASE: KimiMilestonePhase = KimiMilestonePhase::ArtifactUnderstanding;
