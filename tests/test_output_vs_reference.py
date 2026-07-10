"""Tests for converter.verify: structural verification of output vs reference GGUF.

Two groups of checks, since the real converter output does not exist yet:

1. Self-verification: ``verify_structure(reference, reference)`` must pass
   every check -- this validates the checking logic itself against a known
   (large, real) GGUF file.
2. Synthetic mismatch detection: small GGUF files are built in-memory with
   ``gguf.GGUFWriter`` (a handful of tensors each), one deliberately mutated
   from a shared baseline (type diff / shape diff / missing KV key / invalid
   ``config``), to confirm each mismatch is actually caught -- and that nothing
   *else* is spuriously flagged in the process.

A production test (real output GGUF vs real reference GGUF, from config.toml's
``[output]``/``[reference]``) is included but skipped until the converter has
actually produced ``output/Sulphur-2-base-distil-Q4_K_M.gguf``.
"""

from __future__ import annotations

import json
import os

import gguf
import numpy as np
import pytest

from converter.verify import _load_config, resolve_paths, verify_structure

REFERENCE_GGUF = (
    r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\models\ltx-2.3-gguf"
    r"\LTX-2.3-distilled-1.1\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
)

requires_reference_gguf = pytest.mark.skipif(
    not os.path.isfile(REFERENCE_GGUF),
    reason="reference GGUF not available in this environment",
)


# ---------------------------------------------------------------------------
# 1. self-verification against the real reference GGUF
# ---------------------------------------------------------------------------


@requires_reference_gguf
def test_self_verification_reference_vs_reference_passes():
    report = verify_structure(REFERENCE_GGUF, REFERENCE_GGUF)

    assert report.mismatches == {name: [] for name in report.mismatches}, report.summary()
    assert report.passed, report.summary()
    assert report.tensor_count_output == report.tensor_count_reference == 4444


@requires_reference_gguf
def test_self_verification_dequant_sanity_actually_samples_k_quant_tensors():
    # Guards against _check_dequant_sanity silently no-op'ing (e.g. if the
    # K-quant type-name filter ever stops matching anything): the real
    # reference file is known to contain Q4_K/Q5_K/Q6_K tensors, so the
    # sanity check must have exercised at least one of them.
    from converter.verify import _sample_indices
    from gguf import GGUFReader

    reader = GGUFReader(REFERENCE_GGUF)
    k_quant_tensors = [t for t in reader.tensors if t.tensor_type.name.endswith("_K")]
    assert len(k_quant_tensors) > 0
    assert len(_sample_indices(len(k_quant_tensors), 8)) > 0


# ---------------------------------------------------------------------------
# 2. synthetic mismatch detection
# ---------------------------------------------------------------------------

DEFAULT_TENSORS = {
    # Sized (tens of KB) so that the file_size_ratio check's +/-1% tolerance
    # comfortably exceeds the byte-level noise a KV-only mutation introduces
    # (a changed/removed string is a handful of bytes; a truly tiny GGUF
    # would make that noise look like a multi-percent size regression).
    "a.weight": np.arange(100 * 100, dtype=np.float32).reshape(100, 100),  # 40000 bytes
    "b.weight": np.arange(60 * 70, dtype=np.float32).reshape(60, 70),  # 16800 bytes
    "c.bias": np.arange(64, dtype=np.float32),  # 256 bytes
}

DEFAULT_METADATA = {
    "config": json.dumps({"transformer": {"hidden_size": 8}, "_class_name": "Dummy"}),
    "license": "test-license-text",
    "model_version": "0.0.1-test",
    "encrypted_wandb_properties": "opaque-test-blob",
}


def _write_gguf(
    path,
    tensors=None,
    metadata=None,
    *,
    arch="ltxv",
    quantization_version=2,
    file_type=15,
    include_alignment=False,
    skip_metadata_keys=(),
):
    """Build a small GGUF file at ``path`` with GGUFWriter and return its path.

    ``tensors`` maps tensor name -> numpy array (dtype determines the GGML
    type GGUFWriter tags it with, e.g. float32 -> F32, float16 -> F16).
    ``metadata`` maps the four string KV keys to their values.
    """
    tensors = dict(DEFAULT_TENSORS if tensors is None else tensors)
    metadata = dict(DEFAULT_METADATA if metadata is None else metadata)

    writer = gguf.GGUFWriter(str(path), arch=arch)
    writer.add_uint32("general.quantization_version", quantization_version)
    writer.add_uint32("general.file_type", file_type)
    if include_alignment:
        writer.add_uint32("general.alignment", 32)

    for key in ("config", "license", "model_version", "encrypted_wandb_properties"):
        if key in skip_metadata_keys:
            continue
        value = metadata.get(key)
        if value:
            writer.add_string(key, value)

    for name, arr in tensors.items():
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


def _only_check_failed(report, expected_check, *, also_allow=()):
    """Assert that exactly ``expected_check`` (plus any in ``also_allow``) failed."""
    allowed = {expected_check, *also_allow}
    failed = set(report.failed_checks)
    assert expected_check in failed, report.summary()
    assert failed <= allowed, (
        f"unexpected additional failing check(s) {failed - allowed}\n{report.summary()}"
    )


def test_synthetic_identical_structure_passes(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    out_path = _write_gguf(tmp_path / "out.gguf")  # same structure, same values

    report = verify_structure(out_path, ref_path)

    assert report.passed, report.summary()
    assert report.tensor_count_output == report.tensor_count_reference == 3


def test_synthetic_tensor_type_mismatch_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    mutated_tensors = dict(DEFAULT_TENSORS)
    mutated_tensors["c.bias"] = mutated_tensors["c.bias"].astype(np.float16)
    out_path = _write_gguf(tmp_path / "out.gguf", tensors=mutated_tensors)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    _only_check_failed(report, "tensor_types_shapes")
    assert any("c.bias" in issue and "tensor_type mismatch" in issue for issue in report.mismatches["tensor_types_shapes"])


def test_synthetic_tensor_shape_mismatch_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    mutated_tensors = dict(DEFAULT_TENSORS)
    mutated_tensors["b.weight"] = np.arange(60 * 71, dtype=np.float32).reshape(60, 71)  # was (60, 70)
    out_path = _write_gguf(tmp_path / "out.gguf", tensors=mutated_tensors)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    _only_check_failed(report, "tensor_types_shapes", also_allow=("file_size_ratio",))
    assert any("b.weight" in issue and "shape mismatch" in issue for issue in report.mismatches["tensor_types_shapes"])


def test_synthetic_missing_kv_key_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    out_path = _write_gguf(tmp_path / "out.gguf", skip_metadata_keys=("license",))

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    _only_check_failed(report, "kv_keys")
    assert any("license" in issue for issue in report.mismatches["kv_keys"])


def test_synthetic_invalid_config_json_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    bad_metadata = dict(DEFAULT_METADATA)
    bad_metadata["config"] = "{this is not valid json"
    out_path = _write_gguf(tmp_path / "out.gguf", metadata=bad_metadata)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    _only_check_failed(report, "kv_config_json")
    assert any("not valid JSON" in issue for issue in report.mismatches["kv_config_json"])


def test_synthetic_config_missing_transformer_key_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    bad_metadata = dict(DEFAULT_METADATA)
    bad_metadata["config"] = json.dumps({"not_transformer": {"hidden_size": 8}})
    out_path = _write_gguf(tmp_path / "out.gguf", metadata=bad_metadata)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    _only_check_failed(report, "kv_config_json")
    assert any("transformer" in issue for issue in report.mismatches["kv_config_json"])


def test_synthetic_config_differs_but_valid_still_passes(tmp_path):
    # config's *value* is allowed to differ from the reference (Sulphur's own
    # metadata) as long as it stays valid JSON with a top-level "transformer".
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    differing_metadata = dict(DEFAULT_METADATA)
    differing_metadata["config"] = json.dumps({"transformer": {"hidden_size": 999, "extra": True}})
    out_path = _write_gguf(tmp_path / "out.gguf", metadata=differing_metadata)

    report = verify_structure(out_path, ref_path)

    assert report.passed, report.summary()


def test_synthetic_tensor_order_mismatch_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    reordered_tensors = {
        "b.weight": DEFAULT_TENSORS["b.weight"],
        "a.weight": DEFAULT_TENSORS["a.weight"],
        "c.bias": DEFAULT_TENSORS["c.bias"],
    }
    out_path = _write_gguf(tmp_path / "out.gguf", tensors=reordered_tensors)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    _only_check_failed(report, "tensor_names_order")


def test_synthetic_missing_tensor_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    fewer_tensors = {k: v for k, v in DEFAULT_TENSORS.items() if k != "c.bias"}
    out_path = _write_gguf(tmp_path / "out.gguf", tensors=fewer_tensors)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    assert "tensor_count" in report.failed_checks
    assert "tensor_names_order" in report.failed_checks
    assert any("c.bias" in issue for issue in report.mismatches["tensor_names_order"])


def test_synthetic_general_alignment_present_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    out_path = _write_gguf(tmp_path / "out.gguf", include_alignment=True)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    # general.alignment is both a forbidden key specifically (caught by
    # general_alignment_absent) and simply an extra/unexpected KV key not
    # present in the reference (caught by kv_keys) -- both legitimately fire.
    _only_check_failed(report, "general_alignment_absent", also_allow=("kv_keys",))
    assert "general.alignment" in report.mismatches["kv_keys"][0]


def test_synthetic_wrong_architecture_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    out_path = _write_gguf(tmp_path / "out.gguf", arch="not-ltxv")

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    _only_check_failed(report, "kv_fixed_values")
    assert any("general.architecture" in issue for issue in report.mismatches["kv_fixed_values"])


def test_synthetic_wrong_quantization_version_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    out_path = _write_gguf(tmp_path / "out.gguf", quantization_version=1)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    _only_check_failed(report, "kv_fixed_values")
    assert any("general.quantization_version" in issue for issue in report.mismatches["kv_fixed_values"])


def test_synthetic_wrong_file_type_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    out_path = _write_gguf(tmp_path / "out.gguf", file_type=1)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    _only_check_failed(report, "kv_fixed_values")
    assert any("general.file_type" in issue for issue in report.mismatches["kv_fixed_values"])


def test_synthetic_file_size_ratio_detected(tmp_path):
    ref_path = _write_gguf(tmp_path / "ref.gguf")
    out_path = _write_gguf(tmp_path / "out.gguf")

    # Files are tiny (a handful of small tensors), so padding a couple KB onto
    # the output blows well past the 1% tolerance without touching structure.
    with open(out_path, "ab") as f:
        f.write(b"\0" * 4096)

    report = verify_structure(out_path, ref_path)

    assert not report.passed
    assert "file_size_ratio" in report.failed_checks


# ---------------------------------------------------------------------------
# 3. production test: real output GGUF vs real reference GGUF
# ---------------------------------------------------------------------------


def _real_output_gguf_path():
    project_root_config = _load_config()
    output_path, reference_path = resolve_paths(project_root_config)
    if output_path.is_file() and reference_path.is_file():
        return output_path, reference_path
    return None


_REAL_PATHS = _real_output_gguf_path()

requires_real_output_gguf = pytest.mark.skipif(
    _REAL_PATHS is None,
    reason="converted output GGUF (or reference GGUF) not available yet",
)


@requires_real_output_gguf
def test_real_output_gguf_matches_reference_structure():
    output_path, reference_path = _REAL_PATHS
    report = verify_structure(output_path, reference_path)
    assert report.passed, report.summary()
