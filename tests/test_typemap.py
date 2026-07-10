"""Tests for converter.typemap: reference-GGUF typemap extraction and auditing."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from converter.typemap import (
    EXPECTED_TOTAL,
    EXPECTED_TYPE_COUNTS,
    audit,
    classify_by_rule,
    extract_typemap,
    load_typemap,
    save_typemap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.toml"


def _reference_gguf_path() -> Path | None:
    if not CONFIG_PATH.exists():
        return None
    with CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)
    path = Path(config["reference"]["gguf_path"])
    return path if path.exists() else None


def _typemap_json_path() -> Path:
    with CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)
    return PROJECT_ROOT / config["reference"]["typemap_path"]


REFERENCE_GGUF = _reference_gguf_path()
requires_reference_gguf = pytest.mark.skipif(
    REFERENCE_GGUF is None,
    reason="reference GGUF file not available in this environment",
)


@pytest.fixture(scope="module")
def records():
    assert REFERENCE_GGUF is not None
    return extract_typemap(REFERENCE_GGUF)


@requires_reference_gguf
def test_extract_typemap_total_count(records):
    assert len(records) == EXPECTED_TOTAL == 4444


@requires_reference_gguf
def test_extract_typemap_type_counts(records):
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["ggml_type"]] = counts.get(rec["ggml_type"], 0) + 1
    assert counts == EXPECTED_TYPE_COUNTS


@requires_reference_gguf
def test_extract_typemap_preserves_order(records):
    # The first and last tensors of the known reference file, to catch any
    # accidental re-ordering (e.g. sorting) of the extracted records.
    assert records[0]["name"] == "adaln_single.emb.timestep_embedder.linear_1.bias"
    assert records[-1]["name"] == "video_embeddings_connector.transformer_1d_blocks.7.ff.net.2.weight"


@requires_reference_gguf
def test_extract_typemap_record_shape(records):
    for rec in records[:5]:
        assert set(rec.keys()) == {
            "name",
            "ggml_type",
            "shape_gguf",
            "shape_logical",
            "n_elements",
            "nbytes",
        }
        assert rec["shape_logical"] == list(reversed(rec["shape_gguf"]))
        assert rec["n_elements"] > 0
        assert rec["nbytes"] > 0


@requires_reference_gguf
def test_save_and_load_typemap_roundtrip(records, tmp_path):
    out_path = tmp_path / "typemap.json"
    save_typemap(records, out_path, source="test-source.gguf")

    loaded = load_typemap(out_path)
    assert loaded == records

    import json

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source"] == "test-source.gguf"
    assert payload["total"] == EXPECTED_TOTAL
    assert payload["type_counts"] == EXPECTED_TYPE_COUNTS


@requires_reference_gguf
def test_audit_has_no_mismatches(records):
    mismatches = audit(records)
    assert mismatches == []


@requires_reference_gguf
def test_generated_typemap_json_matches_reference(records):
    """The JSON artifact checked into typemap/ should match a fresh extraction."""
    typemap_json = _typemap_json_path()
    if not typemap_json.exists():
        pytest.skip("typemap JSON artifact not generated yet")
    loaded = load_typemap(typemap_json)
    assert loaded == records


def test_classify_by_rule_1d_is_f32():
    assert classify_by_rule("adaln_single.emb.timestep_embedder.linear_1.bias", [4096]) == "F32"
    assert classify_by_rule("transformer_blocks.10.attn1.to_v.bias", [4096]) == "F32"


def test_classify_by_rule_embeddings_connector_is_bf16():
    assert (
        classify_by_rule(
            "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_v.weight",
            [2048, 2048],
        )
        == "BF16"
    )
    assert (
        classify_by_rule(
            "video_embeddings_connector.transformer_1d_blocks.7.ff.net.2.weight",
            [4096, 16384],
        )
        == "BF16"
    )


def test_classify_by_rule_special_f32_names():
    assert classify_by_rule("patchify_proj.weight", [4096, 128]) == "F32"
    assert classify_by_rule("proj_out.weight", [128, 4096]) == "F32"
    assert classify_by_rule("audio_patchify_proj.weight", [2048, 128]) == "F32"
    assert classify_by_rule("audio_proj_out.weight", [128, 2048]) == "F32"
    assert classify_by_rule("scale_shift_table", [2, 4096]) == "F32"
    assert classify_by_rule("transformer_blocks.5.audio_scale_shift_table", [9, 2048]) == "F32"
    assert classify_by_rule("video_embeddings_connector.learnable_registers", [128, 4096]) == "F32"
    assert classify_by_rule("adaln_single.linear.weight", [36864, 4096]) == "F32"


def test_classify_by_rule_boundary_blocks_are_q5k():
    # All 2D tensors in the boundary blocks (0 and 47) are Q5_K, including ones
    # that would otherwise match the to_v/ff.net.2 -> Q6_K rule for "regular" blocks.
    assert classify_by_rule("transformer_blocks.0.attn1.to_v.weight", [4096, 4096]) == "Q5_K"
    assert classify_by_rule("transformer_blocks.47.ff.net.2.weight", [4096, 16384]) == "Q5_K"
    assert classify_by_rule("transformer_blocks.0.ff.net.0.proj.weight", [16384, 4096]) == "Q5_K"


def test_classify_by_rule_middle_blocks_to_v_and_ff2_are_q6k():
    assert classify_by_rule("transformer_blocks.10.attn1.to_v.weight", [4096, 4096]) == "Q6_K"
    assert classify_by_rule("transformer_blocks.10.ff.net.2.weight", [4096, 16384]) == "Q6_K"


def test_classify_by_rule_default_is_q4k():
    assert classify_by_rule("transformer_blocks.10.attn1.to_q.weight", [4096, 4096]) == "Q4_K"
    assert classify_by_rule("transformer_blocks.10.ff.net.0.proj.weight", [16384, 4096]) == "Q4_K"
