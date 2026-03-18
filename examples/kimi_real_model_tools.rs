use attnres::kimi::{
    build_default_module_probe_request, compare_baseline_slice_parity_fixture_with_manifest_from_dir,
    compare_module_probe_fixture_from_dir, generate_module_probe_fixture_from_dir,
    KimiArtifactUnderstanding, KimiBaselineSliceRequestSpec, KimiModuleProbeFixture,
    KimiModuleProbeRequest, KIMI_BASELINE_SLICE_REQUEST_FILENAME,
};
use burn::backend::NdArray;
use std::error::Error;
use std::fs;
use std::path::Path;

type BackendImpl = NdArray;

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    match args.as_slice() {
        [_, command, artifact_dir, output_path] if command == "emit-module-probe-request" => {
            emit_module_probe_request(Path::new(artifact_dir), Path::new(output_path))?;
        }
        [_, command, artifact_dir, request_path, output_path]
            if command == "generate-module-probe-fixture-rust" =>
        {
            generate_module_probe_fixture_rust(
                Path::new(artifact_dir),
                Path::new(request_path),
                Path::new(output_path),
            )?;
        }
        [_, command, artifact_dir, request_path, fixture_path]
            if command == "validate-module-probe-fixture" =>
        {
            validate_module_probe_fixture(
                Path::new(artifact_dir),
                Path::new(request_path),
                Path::new(fixture_path),
            )?;
        }
        [_, command, artifact_dir, request_spec_path, output_dir]
            if command == "emit-baseline-slice-request-bundle-from-spec" =>
        {
            emit_baseline_slice_request_bundle_from_spec(
                Path::new(artifact_dir),
                Path::new(request_spec_path),
                Path::new(output_dir),
            )?;
        }
        [_, command, artifact_dir, manifest_path, fixture_path]
            if command == "validate-baseline-slice-fixture" =>
        {
            validate_baseline_slice_fixture(
                Path::new(artifact_dir),
                Path::new(manifest_path),
                Path::new(fixture_path),
            )?;
        }
        _ => {
            eprintln!(
                "usage:\n  cargo run --example kimi_real_model_tools -- emit-module-probe-request <artifact-dir> <output-path>\n  cargo run --example kimi_real_model_tools -- generate-module-probe-fixture-rust <artifact-dir> <request-path> <output-path>\n  cargo run --example kimi_real_model_tools -- validate-module-probe-fixture <artifact-dir> <request-path> <fixture-path>\n  cargo run --example kimi_real_model_tools -- emit-baseline-slice-request-bundle-from-spec <artifact-dir> <request-spec-path> <output-dir>\n  cargo run --example kimi_real_model_tools -- validate-baseline-slice-fixture <artifact-dir> <manifest-path> <fixture-path>"
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

fn emit_module_probe_request(
    artifact_dir: &Path,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact_dir)?;
    let request = build_default_module_probe_request(&understanding.config)?;
    write_json(output_path, &request)?;
    Ok(())
}

fn generate_module_probe_fixture_rust(
    artifact_dir: &Path,
    request_path: &Path,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let request: KimiModuleProbeRequest = read_json(request_path)?;
    let device = Default::default();
    let fixture =
        generate_module_probe_fixture_from_dir::<BackendImpl, _>(artifact_dir, &request, &device)?;
    write_json(output_path, &fixture)?;
    Ok(())
}

fn validate_module_probe_fixture(
    artifact_dir: &Path,
    request_path: &Path,
    fixture_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let request: KimiModuleProbeRequest = read_json(request_path)?;
    let fixture: KimiModuleProbeFixture = read_json(fixture_path)?;
    let device = Default::default();
    compare_module_probe_fixture_from_dir::<BackendImpl, _>(
        artifact_dir,
        &request,
        &fixture,
        &device,
    )?;
    Ok(())
}

fn emit_baseline_slice_request_bundle_from_spec(
    artifact_dir: &Path,
    request_spec_path: &Path,
    output_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let understanding = KimiArtifactUnderstanding::load_from_dir(artifact_dir)?;
    let request_spec: KimiBaselineSliceRequestSpec = read_json(request_spec_path)?;
    let manifest = understanding.try_build_baseline_slice_request_manifest(request_spec)?;

    if output_dir.exists() {
        fs::remove_dir_all(output_dir)?;
    }
    fs::create_dir_all(output_dir)?;
    manifest.write(output_dir.join(KIMI_BASELINE_SLICE_REQUEST_FILENAME))?;
    Ok(())
}

fn validate_baseline_slice_fixture(
    artifact_dir: &Path,
    manifest_path: &Path,
    fixture_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let device = Default::default();
    compare_baseline_slice_parity_fixture_with_manifest_from_dir::<BackendImpl, _, _, _>(
        artifact_dir,
        manifest_path,
        fixture_path,
        &device,
    )?;
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}
