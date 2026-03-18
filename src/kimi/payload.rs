use burn::module::Param;
use burn::prelude::*;
use burn::tensor::TensorData;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::path::Path;

use crate::kimi::config::KimiArtifactConfigError;
use crate::kimi::import::{
    KimiImportCoverageError, KimiImportError, KimiImportPlan, KimiModuleRef, KimiShardResolver,
    KimiShardResolverError,
};

/// Typed failures for baseline-only tensor payload loading from local sharded artifacts.
#[derive(Debug, Clone, PartialEq)]
pub enum KimiBaselinePayloadError {
    Import(KimiImportError),
    Coverage(KimiImportCoverageError),
    ShardResolver(KimiShardResolverError),
    ReadFailed {
        path: String,
        detail: String,
    },
    ParseFailed {
        path: String,
        detail: String,
    },
    UnsupportedTensorDtype {
        tensor_name: String,
        dtype: String,
    },
    TensorDtypeMismatch {
        tensor_name: String,
        expected: String,
        actual: String,
    },
    TensorShapeMismatch {
        tensor_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    IncompleteModulePayload {
        module: KimiModuleRef,
        missing_tensors: Vec<String>,
    },
    UnsupportedTensorApplication {
        tensor_name: String,
        detail: String,
    },
}

impl Display for KimiBaselinePayloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Import(err) => write!(f, "{err}"),
            Self::Coverage(err) => write!(f, "{err}"),
            Self::ShardResolver(err) => write!(f, "{err}"),
            Self::ReadFailed { path, detail } => {
                write!(f, "failed to read safetensors shard '{path}': {detail}")
            }
            Self::ParseFailed { path, detail } => {
                write!(f, "failed to parse safetensors shard '{path}': {detail}")
            }
            Self::UnsupportedTensorDtype { tensor_name, dtype } => write!(
                f,
                "tensor '{tensor_name}' uses unsupported safetensors dtype '{dtype}'"
            ),
            Self::TensorDtypeMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                f,
                "tensor '{tensor_name}' expected payload dtype '{expected}', got '{actual}'"
            ),
            Self::TensorShapeMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                f,
                "tensor '{tensor_name}' expected shape {:?}, got {:?}",
                expected, actual
            ),
            Self::IncompleteModulePayload {
                module,
                missing_tensors,
            } => write!(
                f,
                "selected module {module:?} is missing tensor payloads in shard files: {:?}",
                missing_tensors
            ),
            Self::UnsupportedTensorApplication {
                tensor_name,
                detail,
            } => write!(
                f,
                "tensor '{tensor_name}' cannot be applied to the local baseline model: {detail}"
            ),
        }
    }
}

impl std::error::Error for KimiBaselinePayloadError {}

impl From<KimiArtifactConfigError> for KimiBaselinePayloadError {
    fn from(err: KimiArtifactConfigError) -> Self {
        Self::Import(KimiImportError::Config(err))
    }
}

impl From<KimiImportError> for KimiBaselinePayloadError {
    fn from(err: KimiImportError) -> Self {
        Self::Import(err)
    }
}

impl From<KimiImportCoverageError> for KimiBaselinePayloadError {
    fn from(err: KimiImportCoverageError) -> Self {
        Self::Coverage(err)
    }
}

impl From<KimiShardResolverError> for KimiBaselinePayloadError {
    fn from(err: KimiShardResolverError) -> Self {
        Self::ShardResolver(err)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KimiDecodedTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SafetensorHeader {
    #[serde(rename = "__metadata__")]
    _metadata: Option<BTreeMap<String, String>>,
    #[serde(flatten)]
    tensors: BTreeMap<String, SafetensorTensorHeader>,
}

#[derive(Debug, Deserialize)]
struct SafetensorTensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub(crate) fn load_selected_tensor_payloads(
    plan: &KimiImportPlan,
    root_dir: &Path,
    artifact_dtype: &str,
) -> Result<BTreeMap<String, KimiDecodedTensor>, KimiBaselinePayloadError> {
    let expected_dtype = expected_safetensors_dtype(artifact_dtype)?;
    let resolver = KimiShardResolver::new(root_dir);
    let resolved_shards = resolver.try_resolve_plan(plan)?;
    let tensors_by_shard = plan_mapped_tensors_by_shard(plan);
    let mut loaded = BTreeMap::new();

    for shard in resolved_shards {
        let requested = tensors_by_shard
            .get(&shard.shard_path)
            .expect("resolved shards must come from the mapped tensor set");
        load_shard_payloads(
            &shard.resolved_path,
            requested,
            artifact_dtype,
            expected_dtype,
            &mut loaded,
        )?;
    }

    ensure_complete_module_payloads(plan, &loaded)?;
    Ok(loaded)
}

pub(crate) fn load_param_tensor<const D: usize, B: Backend>(
    target: &mut Param<Tensor<B, D>>,
    tensor_name: &str,
    payload: &KimiDecodedTensor,
) -> Result<(), KimiBaselinePayloadError> {
    let expected = target.lazy_shape().dims;
    if payload.shape != expected {
        return Err(KimiBaselinePayloadError::TensorShapeMismatch {
            tensor_name: tensor_name.to_string(),
            expected,
            actual: payload.shape.clone(),
        });
    }

    let param_id = target.id;
    let device = target.lazy_device();
    let tensor = Tensor::<B, D>::from_data(
        TensorData::new(payload.values.clone(), payload.shape.clone()),
        &device,
    );
    *target = target.clone().transform_for_load(tensor, param_id);
    Ok(())
}

fn expected_safetensors_dtype(
    artifact_dtype: &str,
) -> Result<&'static str, KimiBaselinePayloadError> {
    match artifact_dtype {
        "float32" => Ok("F32"),
        "bfloat16" => Ok("BF16"),
        other => Err(KimiImportError::UnsupportedArtifactDtype {
            dtype: other.to_string(),
        }
        .into()),
    }
}

fn plan_mapped_tensors_by_shard(plan: &KimiImportPlan) -> BTreeMap<String, Vec<String>> {
    let mut by_shard = BTreeMap::new();

    for mapping in &plan.coverage.mapped_tensors {
        by_shard
            .entry(mapping.shard_path.clone())
            .or_insert_with(Vec::new)
            .push(mapping.tensor_name.clone());
    }

    by_shard
}

fn load_shard_payloads(
    path: &Path,
    requested: &[String],
    artifact_dtype: &str,
    expected_dtype: &str,
    loaded: &mut BTreeMap<String, KimiDecodedTensor>,
) -> Result<(), KimiBaselinePayloadError> {
    let bytes = std::fs::read(path).map_err(|err| KimiBaselinePayloadError::ReadFailed {
        path: path.display().to_string(),
        detail: err.to_string(),
    })?;
    let (header, data) = parse_safetensor_file(path, &bytes)?;

    for tensor_name in requested {
        let Some(tensor) = header.tensors.get(tensor_name) else {
            continue;
        };

        let decoded = decode_tensor_payload(
            path,
            tensor_name,
            tensor,
            data,
            artifact_dtype,
            expected_dtype,
        )?;
        loaded.insert(tensor_name.clone(), decoded);
    }

    Ok(())
}

fn parse_safetensor_file<'a>(
    path: &Path,
    bytes: &'a [u8],
) -> Result<(SafetensorHeader, &'a [u8]), KimiBaselinePayloadError> {
    const HEADER_PREFIX_BYTES: usize = 8;

    if bytes.len() < HEADER_PREFIX_BYTES {
        return Err(KimiBaselinePayloadError::ParseFailed {
            path: path.display().to_string(),
            detail: "file is smaller than the 8-byte safetensors header prefix".to_string(),
        });
    }

    let header_len = u64::from_le_bytes(
        bytes[..HEADER_PREFIX_BYTES]
            .try_into()
            .expect("8-byte prefix should convert to u64"),
    ) as usize;
    let header_end = HEADER_PREFIX_BYTES.checked_add(header_len).ok_or_else(|| {
        KimiBaselinePayloadError::ParseFailed {
            path: path.display().to_string(),
            detail: "header length overflowed usize".to_string(),
        }
    })?;

    if header_end > bytes.len() {
        return Err(KimiBaselinePayloadError::ParseFailed {
            path: path.display().to_string(),
            detail: "header length extends past the end of the file".to_string(),
        });
    }

    let header_json =
        std::str::from_utf8(&bytes[HEADER_PREFIX_BYTES..header_end]).map_err(|err| {
            KimiBaselinePayloadError::ParseFailed {
                path: path.display().to_string(),
                detail: format!("header is not valid UTF-8: {err}"),
            }
        })?;
    let header: SafetensorHeader =
        serde_json::from_str(header_json).map_err(|err| KimiBaselinePayloadError::ParseFailed {
            path: path.display().to_string(),
            detail: format!("header is not valid JSON: {err}"),
        })?;

    Ok((header, &bytes[header_end..]))
}

fn decode_tensor_payload(
    path: &Path,
    tensor_name: &str,
    tensor: &SafetensorTensorHeader,
    data: &[u8],
    artifact_dtype: &str,
    expected_dtype: &str,
) -> Result<KimiDecodedTensor, KimiBaselinePayloadError> {
    let dtype = tensor.dtype.as_str();
    let element_size = match dtype {
        "F32" => 4,
        "BF16" => 2,
        other => {
            return Err(KimiBaselinePayloadError::UnsupportedTensorDtype {
                tensor_name: tensor_name.to_string(),
                dtype: other.to_string(),
            });
        }
    };

    if dtype != expected_dtype {
        return Err(KimiBaselinePayloadError::TensorDtypeMismatch {
            tensor_name: tensor_name.to_string(),
            expected: artifact_dtype.to_string(),
            actual: render_tensor_dtype(dtype),
        });
    }

    let numel = tensor
        .shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| KimiBaselinePayloadError::ParseFailed {
            path: path.display().to_string(),
            detail: format!("tensor '{tensor_name}' shape overflowed usize"),
        })?;
    let expected_bytes =
        numel
            .checked_mul(element_size)
            .ok_or_else(|| KimiBaselinePayloadError::ParseFailed {
                path: path.display().to_string(),
                detail: format!("tensor '{tensor_name}' byte length overflowed usize"),
            })?;
    let [start, end] = tensor.data_offsets;

    if start > end || end > data.len() {
        return Err(KimiBaselinePayloadError::ParseFailed {
            path: path.display().to_string(),
            detail: format!("tensor '{tensor_name}' has out-of-bounds data offsets"),
        });
    }

    let raw = &data[start..end];
    if raw.len() != expected_bytes {
        return Err(KimiBaselinePayloadError::ParseFailed {
            path: path.display().to_string(),
            detail: format!(
                "tensor '{tensor_name}' expected {expected_bytes} bytes from shape {:?} and dtype {dtype}, got {}",
                tensor.shape,
                raw.len()
            ),
        });
    }

    let values = match dtype {
        "F32" => decode_f32(raw),
        "BF16" => decode_bf16(raw),
        _ => unreachable!("unsupported dtype should have returned above"),
    };

    Ok(KimiDecodedTensor {
        shape: tensor.shape.clone(),
        values,
    })
}

fn decode_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|chunk| {
            f32::from_le_bytes(
                chunk
                    .try_into()
                    .expect("exact 4-byte chunk should decode as f32"),
            )
        })
        .collect()
}

fn decode_bf16(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes(
                chunk
                    .try_into()
                    .expect("exact 2-byte chunk should decode as bf16"),
            );
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn ensure_complete_module_payloads(
    plan: &KimiImportPlan,
    loaded: &BTreeMap<String, KimiDecodedTensor>,
) -> Result<(), KimiBaselinePayloadError> {
    let loaded_tensors = loaded.keys().cloned().collect::<BTreeSet<_>>();

    for module in &plan.coverage.module_coverage {
        let missing_tensors = module
            .required_tensors
            .iter()
            .filter(|tensor_name| !loaded_tensors.contains(tensor_name.as_str()))
            .cloned()
            .collect::<Vec<_>>();

        if !missing_tensors.is_empty() {
            return Err(KimiBaselinePayloadError::IncompleteModulePayload {
                module: module.module.clone(),
                missing_tensors,
            });
        }
    }

    Ok(())
}

fn render_tensor_dtype(dtype: &str) -> String {
    match dtype {
        "F32" => "float32".to_string(),
        "BF16" => "bfloat16".to_string(),
        other => other.to_string(),
    }
}
