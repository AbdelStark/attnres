/// Model serialization: save and load AttnRes model weights.
///
/// Uses burn's record system with NamedMpk format (default) or compact binary.
/// Provides convenience methods on `AttnResTransformer` for saving/loading.
///
/// # Formats
///
/// - **Default** (`DefaultRecorder`): NamedMpk format, full precision, human-debuggable.
///   File extension: `.mpk`
/// - **Compact** (`CompactRecorder`): NamedMpk format, half precision, smaller files.
///   File extension: `.mpk`
/// - **Binary** (`BinFileRecorder`): Bincode format, full precision, fast.
///   File extension: `.bin`
///
/// # Example
///
/// ```rust,no_run
/// use attnres_rs::{AttnResConfig, AttnResTransformer};
/// use burn::backend::NdArray;
///
/// type B = NdArray;
///
/// let device = Default::default();
/// let config = AttnResConfig::new(128, 8, 2)
///     .with_num_heads(4)
///     .with_vocab_size(1000);
///
/// let model: AttnResTransformer<B> = config.init_model(&device);
///
/// // Save with default recorder
/// model.save("my_model", &device).expect("Failed to save");
///
/// // Load into a fresh model
/// let loaded: AttnResTransformer<B> = AttnResTransformer::load("my_model", &config, &device)
///     .expect("Failed to load");
/// ```
use std::path::PathBuf;

use burn::module::Module;
use burn::prelude::*;
use burn::record::{
    BinFileRecorder, CompactRecorder, DefaultRecorder, FullPrecisionSettings,
    HalfPrecisionSettings, NamedMpkFileRecorder, Recorder,
};

use crate::config::AttnResConfig;
use crate::model::AttnResTransformer;

/// Errors that can occur during model serialization.
#[derive(Debug)]
pub enum SerializationError {
    /// Error from the burn recorder.
    RecorderError(String),
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RecorderError(msg) => write!(f, "Serialization error: {msg}"),
        }
    }
}

impl std::error::Error for SerializationError {}

impl From<burn::record::RecorderError> for SerializationError {
    fn from(err: burn::record::RecorderError) -> Self {
        Self::RecorderError(format!("{err:?}"))
    }
}

impl<B: Backend> AttnResTransformer<B> {
    /// Save model weights using the default recorder (NamedMpk, full precision).
    ///
    /// Creates a file at `{path}.mpk`.
    ///
    /// # Arguments
    /// * `path` - Base path for the output file (without extension)
    /// * `device` - Device (unused but kept for API symmetry with load)
    pub fn save(&self, path: &str, _device: &B::Device) -> Result<(), SerializationError> {
        let recorder = DefaultRecorder::default();
        recorder.record(self.clone().into_record(), PathBuf::from(path))?;
        Ok(())
    }

    /// Load model weights using the default recorder (NamedMpk, full precision).
    ///
    /// Reads from `{path}.mpk`.
    ///
    /// # Arguments
    /// * `path` - Base path for the input file (without extension)
    /// * `config` - Model configuration (must match the saved model's architecture)
    /// * `device` - Device to load tensors onto
    pub fn load(
        path: &str,
        config: &AttnResConfig,
        device: &B::Device,
    ) -> Result<Self, SerializationError> {
        let recorder = DefaultRecorder::default();
        let record = recorder.load(PathBuf::from(path), device)?;
        let model = config.init_model::<B>(device).load_record(record);
        Ok(model)
    }

    /// Save model weights using the compact recorder (NamedMpk, half precision).
    ///
    /// Creates a smaller file at `{path}.mpk` using f16 precision.
    pub fn save_compact(&self, path: &str) -> Result<(), SerializationError> {
        let recorder = CompactRecorder::default();
        recorder.record(self.clone().into_record(), PathBuf::from(path))?;
        Ok(())
    }

    /// Load model weights saved with the compact recorder (half precision).
    pub fn load_compact(
        path: &str,
        config: &AttnResConfig,
        device: &B::Device,
    ) -> Result<Self, SerializationError> {
        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
        let record = recorder.load(PathBuf::from(path), device)?;
        let model = config.init_model::<B>(device).load_record(record);
        Ok(model)
    }

    /// Save model weights using the binary recorder (bincode, full precision).
    ///
    /// Creates a file at `{path}.bin`. Fastest format for save/load.
    pub fn save_binary(&self, path: &str) -> Result<(), SerializationError> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder.record(self.clone().into_record(), PathBuf::from(path))?;
        Ok(())
    }

    /// Load model weights saved with the binary recorder.
    pub fn load_binary(
        path: &str,
        config: &AttnResConfig,
        device: &B::Device,
    ) -> Result<Self, SerializationError> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(PathBuf::from(path), device)?;
        let model = config.init_model::<B>(device).load_record(record);
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_save_load_roundtrip() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 2)
            .with_num_heads(4)
            .with_vocab_size(50);

        let model: AttnResTransformer<TestBackend> = config.init_model(&device);

        // Forward pass before save
        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
        let out_before = model.forward(input.clone(), None);

        // Save and load
        let path = std::env::temp_dir().join("attnres_test_save_load");
        let path_str = path.to_str().unwrap();
        model.save(path_str, &device).expect("Failed to save");

        let loaded: AttnResTransformer<TestBackend> =
            AttnResTransformer::load(path_str, &config, &device).expect("Failed to load");
        let out_after = loaded.forward(input, None);

        let diff: f32 = (out_before - out_after).abs().max().into_scalar();
        assert!(
            diff < 1e-6,
            "Loaded model should produce identical output, diff={diff}"
        );

        // Cleanup
        let _ = std::fs::remove_file(format!("{path_str}.mpk"));
    }

    #[test]
    fn test_save_load_binary_roundtrip() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 2)
            .with_num_heads(4)
            .with_vocab_size(50);

        let model: AttnResTransformer<TestBackend> = config.init_model(&device);

        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
        let out_before = model.forward(input.clone(), None);

        let path = std::env::temp_dir().join("attnres_test_save_load_bin");
        let path_str = path.to_str().unwrap();
        model.save_binary(path_str).expect("Failed to save binary");

        let loaded: AttnResTransformer<TestBackend> =
            AttnResTransformer::load_binary(path_str, &config, &device)
                .expect("Failed to load binary");
        let out_after = loaded.forward(input, None);

        let diff: f32 = (out_before - out_after).abs().max().into_scalar();
        assert!(
            diff < 1e-6,
            "Binary-loaded model should produce identical output, diff={diff}"
        );

        // Cleanup
        let _ = std::fs::remove_file(format!("{path_str}.bin"));
    }
}
