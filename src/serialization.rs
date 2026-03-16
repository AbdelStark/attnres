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
/// use attnres::{AttnResConfig, AttnResTransformer};
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
use std::path::{Path, PathBuf};

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
    /// Failed to save model weights to disk.
    SaveFailed {
        /// Base path that was being written to.
        path: String,
        /// Underlying recorder error details.
        detail: String,
    },
    /// Failed to load model weights from disk.
    LoadFailed {
        /// Base path that was being read from.
        path: String,
        /// Underlying recorder error details.
        detail: String,
    },
    /// Error from the burn recorder (generic, for From impl).
    RecorderError(String),
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SaveFailed { path, detail } => {
                write!(f, "Failed to save model to '{path}': {detail}")
            }
            Self::LoadFailed { path, detail } => {
                write!(f, "Failed to load model from '{path}': {detail}")
            }
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

fn display_path(path: &Path) -> String {
    path.display().to_string()
}

impl<B: Backend> AttnResTransformer<B> {
    /// Save model weights using the default recorder (NamedMpk, full precision).
    ///
    /// Creates a file at `{path}.mpk`.
    ///
    /// # Arguments
    /// * `path` - Base path for the output file (without extension)
    /// * `device` - Device (unused but kept for API symmetry with load)
    ///
    /// # Errors
    /// Returns [`SerializationError::SaveFailed`] if the file cannot be written.
    pub fn save<P: AsRef<Path>>(
        &self,
        path: P,
        _device: &B::Device,
    ) -> Result<(), SerializationError> {
        let path = path.as_ref();
        let recorder = DefaultRecorder::default();
        recorder
            .record(self.clone().into_record(), PathBuf::from(path))
            .map_err(|e| SerializationError::SaveFailed {
                path: display_path(path),
                detail: format!("{e:?}"),
            })?;
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
    ///
    /// # Errors
    /// Returns [`SerializationError::LoadFailed`] if the file cannot be read or
    /// the record is incompatible with the given config.
    pub fn load<P: AsRef<Path>>(
        path: P,
        config: &AttnResConfig,
        device: &B::Device,
    ) -> Result<Self, SerializationError> {
        let path = path.as_ref();
        let recorder = DefaultRecorder::default();
        let record = recorder.load(PathBuf::from(path), device).map_err(|e| {
            SerializationError::LoadFailed {
                path: display_path(path),
                detail: format!("{e:?}"),
            }
        })?;
        let model = config.init_model::<B>(device).load_record(record);
        Ok(model)
    }

    /// Save model weights using the compact recorder (NamedMpk, half precision).
    ///
    /// Creates a smaller file at `{path}.mpk` using f16 precision.
    ///
    /// # Errors
    /// Returns [`SerializationError::SaveFailed`] if the file cannot be written.
    pub fn save_compact<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let path = path.as_ref();
        let recorder = CompactRecorder::default();
        recorder
            .record(self.clone().into_record(), PathBuf::from(path))
            .map_err(|e| SerializationError::SaveFailed {
                path: display_path(path),
                detail: format!("{e:?}"),
            })?;
        Ok(())
    }

    /// Load model weights saved with the compact recorder (half precision).
    ///
    /// # Errors
    /// Returns [`SerializationError::LoadFailed`] if the file cannot be read.
    pub fn load_compact<P: AsRef<Path>>(
        path: P,
        config: &AttnResConfig,
        device: &B::Device,
    ) -> Result<Self, SerializationError> {
        let path = path.as_ref();
        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
        let record = recorder.load(PathBuf::from(path), device).map_err(|e| {
            SerializationError::LoadFailed {
                path: display_path(path),
                detail: format!("{e:?}"),
            }
        })?;
        let model = config.init_model::<B>(device).load_record(record);
        Ok(model)
    }

    /// Save model weights using the binary recorder (bincode, full precision).
    ///
    /// Creates a file at `{path}.bin`. Fastest format for save/load.
    ///
    /// # Errors
    /// Returns [`SerializationError::SaveFailed`] if the file cannot be written.
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), SerializationError> {
        let path = path.as_ref();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .record(self.clone().into_record(), PathBuf::from(path))
            .map_err(|e| SerializationError::SaveFailed {
                path: display_path(path),
                detail: format!("{e:?}"),
            })?;
        Ok(())
    }

    /// Load model weights saved with the binary recorder.
    ///
    /// # Errors
    /// Returns [`SerializationError::LoadFailed`] if the file cannot be read.
    pub fn load_binary<P: AsRef<Path>>(
        path: P,
        config: &AttnResConfig,
        device: &B::Device,
    ) -> Result<Self, SerializationError> {
        let path = path.as_ref();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(PathBuf::from(path), device).map_err(|e| {
            SerializationError::LoadFailed {
                path: display_path(path),
                detail: format!("{e:?}"),
            }
        })?;
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
        model.save(&path, &device).expect("Failed to save");

        let loaded: AttnResTransformer<TestBackend> =
            AttnResTransformer::load(&path, &config, &device).expect("Failed to load");
        let out_after = loaded.forward(input, None);

        let diff: f32 = (out_before - out_after).abs().max().into_scalar();
        assert!(
            diff < 1e-6,
            "Loaded model should produce identical output, diff={diff}"
        );

        // Cleanup
        let _ = std::fs::remove_file(path.with_extension("mpk"));
    }

    #[test]
    fn test_save_load_compact_roundtrip() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 2)
            .with_num_heads(4)
            .with_vocab_size(50);

        let model: AttnResTransformer<TestBackend> = config.init_model(&device);

        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
        let out_before = model.forward(input.clone(), None);

        let path = std::env::temp_dir().join("attnres_test_save_load_compact");
        model.save_compact(&path).expect("Failed to save compact");

        let loaded: AttnResTransformer<TestBackend> =
            AttnResTransformer::load_compact(&path, &config, &device)
                .expect("Failed to load compact");
        let out_after = loaded.forward(input, None);

        // Half precision causes some loss, so tolerance is larger
        let diff: f32 = (out_before - out_after).abs().max().into_scalar();
        assert!(
            diff < 1e-2,
            "Compact-loaded model should produce similar output, diff={diff}"
        );

        // Cleanup
        let _ = std::fs::remove_file(path.with_extension("mpk"));
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
        model.save_binary(&path).expect("Failed to save binary");

        let loaded: AttnResTransformer<TestBackend> =
            AttnResTransformer::load_binary(&path, &config, &device)
                .expect("Failed to load binary");
        let out_after = loaded.forward(input, None);

        let diff: f32 = (out_before - out_after).abs().max().into_scalar();
        assert!(
            diff < 1e-6,
            "Binary-loaded model should produce identical output, diff={diff}"
        );

        // Cleanup
        let _ = std::fs::remove_file(path.with_extension("bin"));
    }

    #[test]
    fn test_load_nonexistent_returns_error() {
        let device = Default::default();
        let config = AttnResConfig::new(32, 4, 2)
            .with_num_heads(4)
            .with_vocab_size(50);

        let result = AttnResTransformer::<TestBackend>::load(
            "/tmp/nonexistent_attnres_model_xyz",
            &config,
            &device,
        );
        assert!(result.is_err(), "Loading nonexistent file should fail");

        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("nonexistent_attnres_model_xyz"),
            "Error should contain the path, got: {msg}"
        );
    }

    #[test]
    fn test_serialization_error_display() {
        let err = SerializationError::SaveFailed {
            path: "test/path".to_string(),
            detail: "disk full".to_string(),
        };
        assert_eq!(
            format!("{err}"),
            "Failed to save model to 'test/path': disk full"
        );

        let err = SerializationError::LoadFailed {
            path: "model.mpk".to_string(),
            detail: "corrupted".to_string(),
        };
        assert_eq!(
            format!("{err}"),
            "Failed to load model from 'model.mpk': corrupted"
        );
    }
}
