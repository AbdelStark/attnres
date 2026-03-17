use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::path::Path;

/// Parsed `metadata` section from `model.safetensors.index.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiShardIndexMetadata {
    #[serde(rename = "total_parameters")]
    pub total_parameter_count: u64,
    #[serde(rename = "total_size")]
    pub total_size_bytes: u64,
}

/// Parsed shard index for a Hugging Face Kimi checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiShardIndex {
    pub metadata: KimiShardIndexMetadata,
    pub weight_map: BTreeMap<String, String>,
}

/// Typed validation failures for [`KimiShardIndex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiShardIndexError {
    ReadFailed { path: String, detail: String },
    ParseFailed { detail: String },
    TotalParameterCountMustBePositive,
    TotalSizeMustBePositive,
    WeightMapMustNotBeEmpty,
    EmptyTensorName,
    EmptyShardPath { tensor_name: String },
}

impl Display for KimiShardIndexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadFailed { path, detail } => {
                write!(f, "failed to read Kimi shard index '{path}': {detail}")
            }
            Self::ParseFailed { detail } => {
                write!(f, "failed to parse Kimi shard index JSON: {detail}")
            }
            Self::TotalParameterCountMustBePositive => {
                write!(f, "metadata.total_parameters must be positive, got 0")
            }
            Self::TotalSizeMustBePositive => {
                write!(f, "metadata.total_size must be positive, got 0")
            }
            Self::WeightMapMustNotBeEmpty => write!(f, "weight_map must not be empty"),
            Self::EmptyTensorName => write!(f, "weight_map contains an empty tensor name"),
            Self::EmptyShardPath { tensor_name } => {
                write!(
                    f,
                    "weight_map tensor '{tensor_name}' has an empty shard path"
                )
            }
        }
    }
}

impl std::error::Error for KimiShardIndexError {}

/// Resolved location for one tensor in a sharded Kimi checkpoint index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiTensorLocation {
    pub tensor_name: String,
    pub shard_path: String,
}

/// Name-to-shard lookup surface built from [`KimiShardIndex`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiTensorLocator {
    weight_map: BTreeMap<String, String>,
}

/// Typed lookup failures for [`KimiTensorLocator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiTensorLocatorError {
    EmptyTensorName,
    MissingTensor { tensor_name: String },
}

impl Display for KimiTensorLocatorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTensorName => write!(f, "tensor name must not be empty"),
            Self::MissingTensor { tensor_name } => {
                write!(
                    f,
                    "tensor '{tensor_name}' is not present in the shard index"
                )
            }
        }
    }
}

impl std::error::Error for KimiTensorLocatorError {}

impl KimiShardIndex {
    pub fn from_json_str(json: &str) -> Result<Self, KimiShardIndexError> {
        let index: Self =
            serde_json::from_str(json).map_err(|err| KimiShardIndexError::ParseFailed {
                detail: err.to_string(),
            })?;
        index.try_validate()?;
        Ok(index)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, KimiShardIndexError> {
        let path = path.as_ref();
        let json =
            std::fs::read_to_string(path).map_err(|err| KimiShardIndexError::ReadFailed {
                path: path.display().to_string(),
                detail: err.to_string(),
            })?;
        Self::from_json_str(&json)
    }

    pub fn try_validate(&self) -> Result<(), KimiShardIndexError> {
        if self.metadata.total_parameter_count == 0 {
            return Err(KimiShardIndexError::TotalParameterCountMustBePositive);
        }
        if self.metadata.total_size_bytes == 0 {
            return Err(KimiShardIndexError::TotalSizeMustBePositive);
        }
        if self.weight_map.is_empty() {
            return Err(KimiShardIndexError::WeightMapMustNotBeEmpty);
        }

        for (tensor_name, shard_path) in &self.weight_map {
            if tensor_name.trim().is_empty() {
                return Err(KimiShardIndexError::EmptyTensorName);
            }
            if shard_path.trim().is_empty() {
                return Err(KimiShardIndexError::EmptyShardPath {
                    tensor_name: tensor_name.clone(),
                });
            }
        }

        Ok(())
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shard_count(&self) -> usize {
        self.weight_map.values().collect::<BTreeSet<_>>().len()
    }

    pub fn shard_paths(&self) -> Vec<&str> {
        self.weight_map
            .values()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn shard_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    pub fn tensor_locator(&self) -> KimiTensorLocator {
        KimiTensorLocator {
            weight_map: self.weight_map.clone(),
        }
    }
}

impl KimiTensorLocator {
    pub fn from_index(index: &KimiShardIndex) -> Self {
        index.tensor_locator()
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shard_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    pub fn locate(&self, tensor_name: &str) -> Result<KimiTensorLocation, KimiTensorLocatorError> {
        if tensor_name.trim().is_empty() {
            return Err(KimiTensorLocatorError::EmptyTensorName);
        }

        let shard_path = self.weight_map.get(tensor_name).ok_or_else(|| {
            KimiTensorLocatorError::MissingTensor {
                tensor_name: tensor_name.to_string(),
            }
        })?;
        Ok(KimiTensorLocation {
            tensor_name: tensor_name.to_string(),
            shard_path: shard_path.clone(),
        })
    }

    pub fn required_shards_for_tensors<I, S>(
        &self,
        tensor_names: I,
    ) -> Result<Vec<String>, KimiTensorLocatorError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut shards = BTreeSet::new();

        for tensor_name in tensor_names {
            let location = self.locate(tensor_name.as_ref())?;
            shards.insert(location.shard_path);
        }

        Ok(shards.into_iter().collect())
    }

    pub fn weight_map(&self) -> &BTreeMap<String, String> {
        &self.weight_map
    }
}
