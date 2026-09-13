"""Fixture coverage for the ltx25-comfyquant admission/type-policy contract.

The profile admits a *community* ComfyUI-quantized redistribution of the LTX 2.5
transformer, so there is no file-identity lock to test.  What is tested here is
the structural admission that replaces it: payload completeness, the sidecar
fold (``weight`` + ``weight_scale``/``weight_s_rel``/... + ``comfy_quant``),
the official key/shape oracle match, and the read-only type policy taken from
the approved official map.

The synthetic source is a 14-raw-tensor / 8-logical-tensor miniature of the real
file: one ``int8_tensorwise`` layer, one ``asym_w4a8_int8`` connector layer and
six plain rows covering both dtype-widening cases the community file has (BF16
source -> F32 map row, F32 source -> quantized map row).  Fixture writers are
imported from ``test_ltx25`` rather than copied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFReader
from gguf.quants import dequantize

import converter
from converter import cli
from converter import comfy_dequant
from converter import ltx25
from converter import ltx25_comfyquant as comfyquant

from test_ltx25 import (
    FIXTURE_GEMMA_SOURCE_CHECKPOINT,
    _bf16,
    _fixture_lock,
    _fixture_metadata,
    _write_safetensors,
)


PREFIX = ltx25.RAW_TRANSFORMER_PREFIX
CONFIG = '{"transformer":{"width":256}}'

INT8_LAYER = f"{PREFIX}transformer_blocks.0.attn1.to_q"
W4A8_LAYER = f"{PREFIX}audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k"

INT8_MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
W4A8_MARKER = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}

OUT_FEATURES = 4
IN_FEATURES = 256

LOGICAL_NAMES = (
    "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.bias",
    "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.weight",
    "transformer_blocks.0.attn1.to_q.bias",
    "transformer_blocks.0.attn1.to_q.weight",
    "transformer_blocks.0.scale_shift_table",
    "transformer_blocks.0.to_gate_logits.weight",
    "video_embeddings_connector.proj.bias",
    "video_embeddings_connector.proj.weight",
)

# The official map's own columns for the eight fixture rows.  ``source_dtype``
# describes the *official* bf16 file (which is why two rows disagree with the
# community source on purpose) and is never compared by this profile.
POLICY_ROWS = (
    {
        "name": "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.bias",
        "source_dtype": "BF16",
        "shape_logical": [OUT_FEATURES],
        "ggml_type": "BF16",
    },
    {
        # Connector weights stay BF16 for the LTX bundle loader even though the
        # community file ships them as w4a8: they are dequantized, not requantized.
        "name": "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.weight",
        "source_dtype": "BF16",
        "shape_logical": [OUT_FEATURES, IN_FEATURES],
        "ggml_type": "BF16",
    },
    {
        "name": "transformer_blocks.0.attn1.to_q.bias",
        "source_dtype": "BF16",
        "shape_logical": [OUT_FEATURES],
        "ggml_type": "BF16",
    },
    {
        "name": "transformer_blocks.0.attn1.to_q.weight",
        "source_dtype": "BF16",
        "shape_logical": [OUT_FEATURES, IN_FEATURES],
        "ggml_type": "Q4_K",
    },
    {
        # Official F32 row; the community file stores it as BF16 (lossless widening).
        "name": "transformer_blocks.0.scale_shift_table",
        "source_dtype": "F32",
        "shape_logical": [2, OUT_FEATURES],
        "ggml_type": "F32",
    },
    {
        # Official BF16 row; the community file stores it as F32 and both quantize.
        "name": "transformer_blocks.0.to_gate_logits.weight",
        "source_dtype": "BF16",
        "shape_logical": [OUT_FEATURES, IN_FEATURES],
        "ggml_type": "Q4_K",
    },
    {
        "name": "video_embeddings_connector.proj.bias",
        "source_dtype": "BF16",
        "shape_logical": [OUT_FEATURES],
        "ggml_type": "BF16",
    },
    {
        "name": "video_embeddings_connector.proj.weight",
        "source_dtype": "BF16",
        "shape_logical": [OUT_FEATURES, IN_FEATURES],
        "ggml_type": "BF16",
    },
)

ORACLE_SHAPES = {
    "transformer": {
        "transformer_blocks.0.attn1.to_q.weight": [OUT_FEATURES, IN_FEATURES],
        "transformer_blocks.0.attn1.to_q.bias": [OUT_FEATURES],
        "transformer_blocks.0.scale_shift_table": [2, OUT_FEATURES],
        "transformer_blocks.0.to_gate_logits.weight": [OUT_FEATURES, IN_FEATURES],
    },
    "audio_embeddings_connector": {
        "transformer_1d_blocks.0.attn1.to_k.weight": [OUT_FEATURES, IN_FEATURES],
        "transformer_1d_blocks.0.attn1.to_k.bias": [OUT_FEATURES],
    },
    "video_embeddings_connector": {
        "proj.weight": [OUT_FEATURES, IN_FEATURES],
        "proj.bias": [OUT_FEATURES],
    },
}

_COMPONENT_FIXED = {
    "transformer": ("emit", "transformer", ""),
    "audio_embeddings_connector": (
        "emit",
        "gemma-audio-embeddings-connector",
        "audio_embeddings_connector.",
    ),
    "video_embeddings_connector": (
        "emit",
        "gemma-video-embeddings-connector",
        "video_embeddings_connector.",
    ),
}


# --------------------------------------------------------------------------
# fixture writers
# --------------------------------------------------------------------------
def _f32(values) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes()


def _i8(count: int) -> bytes:
    return (np.arange(count, dtype=np.int64) % 251 - 125).astype(np.int8).tobytes()


def _marker_bytes(marker: dict) -> bytes:
    return json.dumps(marker).encode("utf-8")


def _default_tensors() -> list[tuple[str, str, bytes, list[int]]]:
    """The 14 raw tensors of the admissible miniature source, in payload order."""
    int8_marker = _marker_bytes(INT8_MARKER)
    w4a8_marker = _marker_bytes(W4A8_MARKER)
    packed = IN_FEATURES // 2
    return [
        (f"{INT8_LAYER}.weight", "I8", _i8(OUT_FEATURES * IN_FEATURES), [OUT_FEATURES, IN_FEATURES]),
        (f"{INT8_LAYER}.weight_scale", "F32", _f32([0.01, 0.02, 0.03, 0.04]), [OUT_FEATURES, 1]),
        (f"{INT8_LAYER}.comfy_quant", "U8", int8_marker, [len(int8_marker)]),
        (f"{INT8_LAYER}.bias", "BF16", _bf16(np.zeros(OUT_FEATURES, np.float32)), [OUT_FEATURES]),
        (
            f"{PREFIX}transformer_blocks.0.scale_shift_table",
            "BF16",
            _bf16(np.linspace(-1.0, 1.0, 2 * OUT_FEATURES, dtype=np.float32)),
            [2, OUT_FEATURES],
        ),
        (
            f"{PREFIX}transformer_blocks.0.to_gate_logits.weight",
            "F32",
            _f32(np.linspace(-1.0, 1.0, OUT_FEATURES * IN_FEATURES, dtype=np.float32)),
            [OUT_FEATURES, IN_FEATURES],
        ),
        (f"{W4A8_LAYER}.weight", "I8", _i8(OUT_FEATURES * packed), [OUT_FEATURES, packed]),
        (
            f"{W4A8_LAYER}.weight_s_rel",
            "F8_E4M3",
            bytes([0x38] * (OUT_FEATURES * IN_FEATURES // 16)),
            [OUT_FEATURES, IN_FEATURES // 16],
        ),
        (f"{W4A8_LAYER}.weight_s_channel", "F32", _f32([0.1, 0.2, 0.3, 0.4]), [OUT_FEATURES]),
        (
            f"{W4A8_LAYER}.weight_codebook",
            "F32",
            _f32(np.linspace(-8.0, 7.0, 16, dtype=np.float32)),
            [16],
        ),
        (f"{W4A8_LAYER}.comfy_quant", "U8", w4a8_marker, [len(w4a8_marker)]),
        (f"{W4A8_LAYER}.bias", "BF16", _bf16(np.zeros(OUT_FEATURES, np.float32)), [OUT_FEATURES]),
        (
            f"{PREFIX}video_embeddings_connector.proj.weight",
            "BF16",
            _bf16(np.zeros(OUT_FEATURES * IN_FEATURES, np.float32)),
            [OUT_FEATURES, IN_FEATURES],
        ),
        (
            f"{PREFIX}video_embeddings_connector.proj.bias",
            "BF16",
            _bf16(np.zeros(OUT_FEATURES, np.float32)),
            [OUT_FEATURES],
        ),
    ]


def _drop(tensors, name):
    kept = [row for row in tensors if row[0] != name]
    assert len(kept) == len(tensors) - 1, name
    return kept


def _replace(tensors, name, dtype, raw, shape):
    index = [row[0] for row in tensors].index(name)
    updated = list(tensors)
    updated[index] = (name, dtype, raw, shape)
    return updated


def _write_source(path: Path, tensors=None, *, config: str = CONFIG, metadata=None) -> Path:
    _write_safetensors(
        path,
        _default_tensors() if tensors is None else tensors,
        _fixture_metadata(
            config,
            license="fixture",
            model_version="2.5.0",
            quant_format="mixed:w4a8+int8",
            quant_mixed_hi_layers="1",
        )
        if metadata is None
        else metadata,
    )
    return path


def _write_oracle(path: Path, *, config: str = CONFIG, shapes=None) -> Path:
    shapes = ORACLE_SHAPES if shapes is None else shapes
    components = {}
    for name, (classification, component_id, native_prefix) in _COMPONENT_FIXED.items():
        components[name] = {
            "classification": classification,
            "component_id": component_id,
            "native_prefix": native_prefix,
            "state_dict_shapes": shapes[name],
        }
    payload = {
        "format": ltx25.BUILDER_ORACLE_FORMAT,
        "profile": "ltx25",
        "official_code_commit": ltx25.OFFICIAL_CODE_COMMIT,
        "config_bytes_sha256": ltx25._sha256_bytes(config.encode("utf-8")),
        "components": components,
    }
    payload["oracle_sha256"] = ltx25._sha256_bytes(ltx25._canonical_json_bytes(payload))
    path.write_bytes(ltx25._canonical_json_bytes(payload) + b"\n")
    return path


def _write_policy_map(path: Path, rows=None) -> dict:
    """Write an approved ltx25-format map: the type policy this profile reads."""
    rows = POLICY_ROWS if rows is None else rows
    payload = {
        "format": ltx25.MAP_FORMAT,
        "profile": "ltx25",
        "status": "approved",
        "inventory_sha256": "0" * 64,
        "tensors": [
            {
                **row,
                "shape_gguf": list(reversed(row["shape_logical"])),
                "nbytes": ltx25._map_nbytes(row["shape_logical"], row["ggml_type"]),
                "rule_id": "fixture-policy",
                "reason": "Fixture row mirroring the approved official map columns.",
                "evidence_ids": ["E4"],
            }
            for row in rows
        ],
    }
    payload["type_counts"] = ltx25._type_counts(payload["tensors"])
    payload["map_sha256"] = ltx25._sha256_bytes(ltx25._canonical_json_bytes(payload))
    path.write_bytes(ltx25._canonical_json_bytes(payload) + b"\n")
    return payload


def _artifacts(tmp_path, tensors=None, **kwargs):
    source = _write_source(tmp_path / "fixture.safetensors", tensors, **kwargs)
    oracle = _write_oracle(tmp_path / "oracle.json")
    return source, oracle


def _inspect(tmp_path, tensors=None, **kwargs):
    source, oracle = _artifacts(tmp_path, tensors, **kwargs)
    return comfyquant.inspect_comfyquant(source, builder_oracle_path=oracle)


# --------------------------------------------------------------------------
# acceptance
# --------------------------------------------------------------------------
def test_miniature_source_is_admitted_and_folded(tmp_path):
    source, oracle = _artifacts(tmp_path)
    inventory_path = tmp_path / "fixture.gguf.inventory.json"
    inventory = comfyquant.inspect_comfyquant(
        source, builder_oracle_path=oracle, inventory_path=inventory_path
    )

    assert tuple(row["name"] for row in inventory.logical_tensors) == LOGICAL_NAMES
    assert len(inventory.raw_tensors) == 14
    assert inventory.source_size == source.stat().st_size
    assert inventory.source_sha256 == ltx25.sha256_of_file(source)
    assert inventory.config_text == CONFIG
    assert inventory.builder_oracle_sha256 == ltx25.load_builder_oracle(oracle)["oracle_sha256"]

    summary = inventory.quant_summary
    assert summary["logical_tensor_count"] == 8
    assert summary["raw_tensor_count"] == 14
    assert summary["logical_kind_counts"] == {
        "asym_w4a8_int8": 1,
        "int8_tensorwise": 1,
        "plain": 6,
    }
    assert summary["logical_source_dtype_counts"] == {"BF16": 5, "F32": 1, "I8": 2}
    assert summary["raw_dtype_counts"] == {
        "BF16": 5,
        "F32": 4,
        "F8_E4M3": 1,
        "I8": 2,
        "U8": 2,
    }
    assert [variant["count"] for variant in summary["marker_variants"]] == [1, 1]
    assert sorted(variant["marker"]["format"] for variant in summary["marker_variants"]) == [
        "asym_w4a8_int8",
        "int8_tensorwise",
    ]

    rows = comfyquant.logical_index(inventory)
    int8_row = rows["transformer_blocks.0.attn1.to_q.weight"]
    assert int8_row["kind"] == "int8_tensorwise"
    assert int8_row["source_dtype"] == "I8"
    assert int8_row["shape_logical"] == [OUT_FEATURES, IN_FEATURES]
    assert sorted(int8_row["sidecars"]) == ["comfy_quant", "weight_scale"]
    assert int8_row["marker"] == {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
        "group_size": None,
    }

    w4a8_row = rows["audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.weight"]
    assert w4a8_row["kind"] == "asym_w4a8_int8"
    # The stored I8 payload is half as wide; the logical width is the real one.
    assert w4a8_row["sidecars"]["weight_s_rel"]["shape"] == [OUT_FEATURES, IN_FEATURES // 16]
    assert w4a8_row["shape_logical"] == [OUT_FEATURES, IN_FEATURES]
    assert w4a8_row["marker"]["group_size"] == 16

    plain_row = rows["transformer_blocks.0.scale_shift_table"]
    assert plain_row["kind"] == "plain"
    assert plain_row["sidecars"] == {} and plain_row["marker"] is None

    # Unknown metadata keys are recorded verbatim, never rejected.
    assert inventory.source_metadata_extra["quant_format"] == "mixed:w4a8+int8"
    assert inventory.source_metadata_extra["quant_mixed_hi_layers"] == "1"
    assert "config" not in inventory.source_metadata_extra
    assert any("quant_format" in note for note in inventory.diagnostics)

    written = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert written["format"] == "nz-ltx25-comfyquant-inventory-v1"
    assert written["profile"] == "ltx25-comfyquant"
    assert written["inventory_sha256"] == inventory.inventory_sha256
    skeleton = {k: v for k, v in written.items() if k != "inventory_sha256"}
    assert ltx25._sha256_bytes(ltx25._canonical_json_bytes(skeleton)) == inventory.inventory_sha256


def test_official_map_q4_k_rows_follow_the_selected_quant_type(tmp_path):
    inventory = _inspect(tmp_path)
    map_path = tmp_path / "map.json"
    written = _write_policy_map(map_path)
    policy, map_sha256 = comfyquant.load_official_type_policy(map_path)
    assert map_sha256 == written["map_sha256"]
    assert map_path.read_bytes() == ltx25._canonical_json_bytes(written) + b"\n"

    default_records = comfyquant.records_for_inventory_and_policy(inventory, policy, "Q6_K")
    assert comfyquant.type_counts(default_records) == {"BF16": 5, "F32": 1, "Q6_K": 2}
    q4_records = comfyquant.records_for_inventory_and_policy(inventory, policy, "Q4_K")
    assert comfyquant.type_counts(q4_records) == {"BF16": 5, "F32": 1, "Q4_K": 2}

    by_name = {record["name"]: record for record in default_records}
    assert [record["name"] for record in default_records] == sorted(LOGICAL_NAMES)
    # BF16/F32 rows are left exactly as the map has them.
    assert by_name["transformer_blocks.0.scale_shift_table"]["ggml_type"] == "F32"
    assert (
        by_name["audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.weight"]["ggml_type"]
        == "BF16"
    )
    quantized = by_name["transformer_blocks.0.attn1.to_q.weight"]
    assert quantized["ggml_type"] == "Q6_K"
    assert quantized["shape_gguf"] == [IN_FEATURES, OUT_FEATURES]
    assert quantized["nbytes"] == ltx25._map_nbytes([OUT_FEATURES, IN_FEATURES], "Q6_K")
    assert quantized["source_kind"] == "int8_tensorwise"
    assert by_name["transformer_blocks.0.to_gate_logits.weight"]["source_kind"] == "plain"

    # The post-assertion is opt-in so a fixture can state its own histogram.
    comfyquant.records_for_inventory_and_policy(
        inventory, policy, "Q6_K", expected_type_counts={"BF16": 5, "F32": 1, "Q6_K": 2}
    )
    with pytest.raises(comfyquant.ComfyQuantPolicyError) as exc:
        comfyquant.records_for_inventory_and_policy(
            inventory, policy, "Q6_K", expected_type_counts=comfyquant.official_type_counts("Q6_K")
        )
    assert "type_counts" in str(exc.value)
    assert comfyquant.official_type_counts("Q6_K") == {"BF16": 2401, "F32": 290, "Q6_K": 1658}
    assert comfyquant.official_type_counts("Q4_K") == {"BF16": 2401, "F32": 290, "Q4_K": 1658}


def test_policy_rejects_unknown_quant_type_and_unmatched_names(tmp_path):
    inventory = _inspect(tmp_path)
    map_path = tmp_path / "map.json"
    _write_policy_map(map_path)
    policy, _ = comfyquant.load_official_type_policy(map_path)

    with pytest.raises(comfyquant.ComfyQuantPolicyError) as exc:
        comfyquant.records_for_inventory_and_policy(inventory, policy, "Q8_0")
    assert "quant type must be one of" in str(exc.value)

    short = {name: entry for name, entry in policy.items() if "to_gate_logits" not in name}
    with pytest.raises(comfyquant.ComfyQuantPolicyError) as exc:
        comfyquant.records_for_inventory_and_policy(inventory, short, "Q6_K")
    assert "extra=1" in str(exc.value)

    widened = dict(policy)
    widened["transformer_blocks.0.attn1.to_q.weight"] = {
        "shape_logical": [OUT_FEATURES, 512],
        "ggml_type": "Q4_K",
    }
    with pytest.raises(comfyquant.ComfyQuantPolicyError) as exc:
        comfyquant.records_for_inventory_and_policy(inventory, widened, "Q6_K")
    assert "shape_mismatch=1" in str(exc.value)

    narrowing = dict(policy)
    narrowing["transformer_blocks.0.to_gate_logits.weight"] = {
        "shape_logical": [OUT_FEATURES, IN_FEATURES],
        "ggml_type": "BF16",
    }
    with pytest.raises(comfyquant.ComfyQuantPolicyError) as exc:
        comfyquant.records_for_inventory_and_policy(inventory, narrowing, "Q6_K")
    assert "narrow an F32 source row to BF16" in str(exc.value)

    unsupported = dict(policy)
    unsupported["transformer_blocks.0.attn1.to_q.weight"] = {
        "shape_logical": [OUT_FEATURES, IN_FEATURES],
        "ggml_type": "Q5_K",
    }
    with pytest.raises(comfyquant.ComfyQuantPolicyError) as exc:
        comfyquant.records_for_inventory_and_policy(inventory, unsupported, "Q6_K")
    assert "neither passes through" in str(exc.value)


def test_draft_official_map_is_not_an_admissible_type_policy(tmp_path):
    map_path = tmp_path / "map.json"
    payload = _write_policy_map(map_path)
    payload["status"] = "draft"
    payload["map_sha256"] = ltx25._sha256_bytes(
        ltx25._canonical_json_bytes({k: v for k, v in payload.items() if k != "map_sha256"})
    )
    map_path.write_bytes(ltx25._canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ltx25.PolicyMapError):
        comfyquant.load_official_type_policy(map_path)


# --------------------------------------------------------------------------
# rejections (plan item 9-17 plus the marker/identity guards)
# --------------------------------------------------------------------------
def test_item9_truncated_payload_is_named_as_an_incomplete_download(tmp_path):
    source, oracle = _artifacts(tmp_path)
    full = source.read_bytes()
    source.write_bytes(full[:-1])
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        comfyquant.inspect_comfyquant(source, builder_oracle_path=oracle)
    message = str(exc.value)
    assert "incomplete payload" in message
    assert "truncated or incomplete download" in message
    assert "short by 1 bytes" in message
    assert "1 of 14 tensors end beyond the end of the file" in message
    assert f"expected file size {len(full)}, actual {len(full) - 1}" in message

    # A file with unexplained trailing bytes is equally inadmissible.
    source.write_bytes(full + b"\x00" * 32)
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        comfyquant.inspect_comfyquant(source, builder_oracle_path=oracle)
    assert "32 unused trailing bytes" in str(exc.value)


def test_item10_missing_weight_scale_is_rejected(tmp_path):
    tensors = _drop(_default_tensors(), f"{INT8_LAYER}.weight_scale")
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        _inspect(tmp_path, tensors)
    message = str(exc.value)
    assert "wrong sidecar set" in message
    assert "missing=['weight_scale']" in message


def test_item11_sidecar_without_a_parent_weight_is_rejected(tmp_path):
    tensors = _default_tensors() + [
        (f"{PREFIX}transformer_blocks.0.orphan.weight_scale", "F32", _f32([1.0]), [1, 1])
    ]
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        _inspect(tmp_path, tensors)
    assert "has no parent weight tensor" in str(exc.value)
    assert "transformer_blocks.0.orphan.weight'" in str(exc.value)


def test_item12_tampered_weight_s_rel_shape_is_rejected(tmp_path):
    tensors = _replace(
        _default_tensors(),
        f"{W4A8_LAYER}.weight_s_rel",
        "F8_E4M3",
        bytes([0x38] * (OUT_FEATURES * 8)),
        [OUT_FEATURES, 8],
    )
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        _inspect(tmp_path, tensors)
    assert "must be F8_E4M3 [4, 16]" in str(exc.value)


def test_item13_key_absent_from_the_oracle_is_rejected(tmp_path):
    tensors = _default_tensors() + [
        (
            f"{PREFIX}transformer_blocks.0.unexpected.weight",
            "BF16",
            _bf16(np.zeros(OUT_FEATURES, np.float32)),
            [OUT_FEATURES],
        )
    ]
    with pytest.raises(comfyquant.ComfyQuantInventoryMismatchError) as exc:
        _inspect(tmp_path, tensors)
    assert "transformer extra=1" in str(exc.value)
    assert "transformer_blocks.0.unexpected.weight" in str(exc.value)


def test_item14_key_required_by_the_oracle_but_absent_is_rejected(tmp_path):
    tensors = _drop(_default_tensors(), f"{PREFIX}video_embeddings_connector.proj.bias")
    with pytest.raises(comfyquant.ComfyQuantInventoryMismatchError) as exc:
        _inspect(tmp_path, tensors)
    assert "video_embeddings_connector missing=1 ['proj.bias']" in str(exc.value)


def test_item15_shape_disagreement_with_the_oracle_is_rejected(tmp_path):
    shapes = {name: dict(rows) for name, rows in ORACLE_SHAPES.items()}
    shapes["transformer"]["transformer_blocks.0.attn1.to_q.weight"] = [OUT_FEATURES, 512]
    source = _write_source(tmp_path / "fixture.safetensors")
    oracle = _write_oracle(tmp_path / "oracle.json", shapes=shapes)
    with pytest.raises(comfyquant.ComfyQuantInventoryMismatchError) as exc:
        comfyquant.inspect_comfyquant(source, builder_oracle_path=oracle)
    assert "transformer shape_mismatch=1" in str(exc.value)


def test_item16_config_digest_disagreement_is_rejected(tmp_path):
    source = _write_source(tmp_path / "fixture.safetensors")
    oracle = _write_oracle(tmp_path / "oracle.json", config='{"transformer":{"width":512}}')
    with pytest.raises(comfyquant.ComfyQuantInventoryMismatchError) as exc:
        comfyquant.inspect_comfyquant(source, builder_oracle_path=oracle)
    assert "config digest differs" in str(exc.value)


def test_item17_missing_gemma_source_checkpoint_is_rejected(tmp_path):
    with pytest.raises(ltx25.SourceRejectedError) as exc:
        _inspect(tmp_path, metadata={"config": CONFIG, "license": "fixture"})
    assert "gemma_source_checkpoint" in str(exc.value)


def test_unknown_marker_format_is_rejected(tmp_path):
    raw = _marker_bytes({"format": "nvfp4", "convrot_groupsize": 256})
    tensors = _replace(_default_tensors(), f"{INT8_LAYER}.comfy_quant", "U8", raw, [len(raw)])
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        _inspect(tmp_path, tensors)
    message = str(exc.value)
    assert "not an admissible ComfyUI quantization marker" in message
    assert "nvfp4" in message
    assert list(comfy_dequant.SUPPORTED_FORMATS) == ["int8_tensorwise", "asym_w4a8_int8"]


def test_missing_marker_and_unquantized_dtype_are_rejected(tmp_path):
    tensors = _drop(_default_tensors(), f"{INT8_LAYER}.comfy_quant")
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        _inspect(tmp_path, tensors)
    assert "no 'comfy_quant' format marker" in str(exc.value)

    tensors = _replace(
        _default_tensors(),
        f"{PREFIX}video_embeddings_connector.proj.bias",
        "F8_E4M3",
        bytes(OUT_FEATURES),
        [OUT_FEATURES],
    )
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        _inspect(tmp_path, tensors)
    assert "outside the allowlist ['BF16', 'F32']" in str(exc.value)


def test_foreign_prefix_is_rejected(tmp_path):
    tensors = _default_tensors() + [
        ("vae.decoder.weight", "BF16", _bf16(np.zeros(OUT_FEATURES, np.float32)), [OUT_FEATURES])
    ]
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        _inspect(tmp_path, tensors)
    assert "unknown component key 'vae.decoder.weight'" in str(exc.value)


def test_expect_sha256_is_an_optional_identity_pin(tmp_path):
    source, oracle = _artifacts(tmp_path)
    digest = ltx25.sha256_of_file(source)
    inventory = comfyquant.inspect_comfyquant(
        source, builder_oracle_path=oracle, expect_sha256=digest.upper()
    )
    assert inventory.source_sha256 == digest
    with pytest.raises(comfyquant.ComfyQuantSourceRejectedError) as exc:
        comfyquant.inspect_comfyquant(source, builder_oracle_path=oracle, expect_sha256="ab" * 32)
    assert f"SHA-256 {digest} != expected {'ab' * 32}" in str(exc.value)


def test_transformer_only_oracle_is_not_enough(tmp_path):
    source = _write_source(tmp_path / "fixture.safetensors")
    oracle_path = tmp_path / "legacy-oracle.json"
    payload = {
        "format": ltx25.BUILDER_ORACLE_FORMAT,
        "profile": "ltx25",
        "official_code_commit": ltx25.OFFICIAL_CODE_COMMIT,
        "config_bytes_sha256": ltx25._sha256_bytes(CONFIG.encode("utf-8")),
        "builder_state_dict_shapes": ORACLE_SHAPES["transformer"],
    }
    payload["oracle_sha256"] = ltx25._sha256_bytes(ltx25._canonical_json_bytes(payload))
    oracle_path.write_bytes(ltx25._canonical_json_bytes(payload) + b"\n")
    with pytest.raises(comfyquant.ComfyQuantInventoryMismatchError) as exc:
        comfyquant.inspect_comfyquant(source, builder_oracle_path=oracle_path)
    assert "three-component" in str(exc.value)


# --------------------------------------------------------------------------
# CLI (plan items 20-23)
# --------------------------------------------------------------------------
def test_item20_unsupported_commands_error_explicitly(tmp_path, capsys):
    """Every non-comfyquant command must say so, not fall into the ltx23 path."""
    expectations = {
        "download": "download is not available with --model ltx25-comfyquant",
        "extract-typemap": "extract-typemap is ltx23-only",
        "build-map": "build-map is available only with --model ltx25/gemma4-ltx25",
        "all": "all is ltx23-only",
    }
    for command, expected in expectations.items():
        assert cli.main([command, "--model", "ltx25-comfyquant"]) == 1
        captured = capsys.readouterr()
        assert expected in captured.err, (command, captured.err)
        assert "Traceback" not in captured.err
        # The ltx23 tables must not have been consulted at all.
        assert "SulphurAI/Sulphur-2-base" not in captured.out


def test_item21_profile_defaults_come_from_config_toml():
    namespace = cli.argparse.Namespace(model="ltx25-comfyquant", quant_workers=None)
    assert (
        cli._quant_worker_count(
            namespace, {"profiles": {"ltx25-comfyquant": {"quant_workers": 4}}}
        )
        == 4
    )
    config = cli._load_config()
    assert cli._quant_worker_count(namespace, config) == 4
    defaults = comfyquant.default_paths_from_config(config, cli._PROJECT_ROOT)
    assert defaults["quant_type"] == "Q6_K"
    assert defaults["quant_workers"] == 4
    assert defaults["map"] == cli._PROJECT_ROOT / "typemap" / "ltx25_conversion_map.json"
    assert defaults["oracle"] == cli._PROJECT_ROOT / "typemap" / "ltx25_builder_oracle.json"
    assert defaults["output_dir"] == cli._PROJECT_ROOT / "output"
    # No source lock and no source path exist for a community file.
    assert "source" not in defaults


def test_cli_inspect_writes_the_inventory_and_projects_the_type_counts(tmp_path, capsys):
    source, oracle = _artifacts(tmp_path)
    map_path = tmp_path / "map.json"
    _write_policy_map(map_path)
    config = {
        "profiles": {
            "ltx25-comfyquant": {
                "official_map_path": str(map_path),
                "builder_oracle_path": str(oracle),
                "output_dir": str(tmp_path / "output"),
                "quant_workers": 4,
                "quant_type": "Q6_K",
            }
        }
    }
    parser = cli.build_parser()

    args = parser.parse_args(
        ["inspect", "--model", "ltx25-comfyquant", "--st-path", str(source)]
    )
    assert cli.cmd_inspect(args, config) == 0
    out = capsys.readouterr().out
    expected_inventory = tmp_path / "output" / "fixture-Q6_K.gguf.inventory.json"
    assert expected_inventory.is_file()
    assert "14 {'BF16': 5, 'F32': 4, 'F8_E4M3': 1, 'I8': 2, 'U8': 2}" in out
    assert "8 {'asym_w4a8_int8': 1, 'int8_tensorwise': 1, 'plain': 6}" in out
    assert "{'BF16': 5, 'F32': 1, 'Q6_K': 2}" in out
    assert str(tmp_path / "output" / "fixture-Q6_K.gguf") in out

    args = parser.parse_args(
        [
            "inspect",
            "--model",
            "ltx25-comfyquant",
            "--st-path",
            str(source),
            "--quant-type",
            "Q4_K",
        ]
    )
    assert cli.cmd_inspect(args, config) == 0
    assert "{'BF16': 5, 'F32': 1, 'Q4_K': 2}" in capsys.readouterr().out

    args = parser.parse_args(["inspect", "--model", "ltx25-comfyquant"])
    assert cli.cmd_inspect(args, config) == 1
    assert "--st-path is required" in capsys.readouterr().err


def test_item22_ltx25_profile_still_refuses_f8_e4m3(tmp_path):
    """The shared _DTYPE_BITS entry must not widen the ltx25 source allowlist."""
    assert ltx25._DTYPE_BITS["F8_E4M3"] == 8
    path = tmp_path / "f8.safetensors"
    _write_safetensors(
        path,
        [(f"{PREFIX}transformer.weight", "F8_E4M3", bytes(256), [1, 256])],
        _fixture_metadata(CONFIG, license="fixture"),
    )
    with pytest.raises(ltx25.SourceRejectedError) as exc:
        ltx25.inspect_ltx25(
            path, source_lock=_fixture_lock(path), require_builder_oracle=False
        )
    assert "outside the E3 source-dtype allowlist BF16/F32" in str(exc.value)


def test_item23_checked_in_typemap_artifacts_are_unchanged():
    """ltx25-comfyquant reads typemap/ but must never write to it."""
    expected = {
        "gemma4_ltx25_builder_oracle.json": "7df6b93a33122caa3cae5b19ae747cba4712bf89c59c11aa6c18602f82523329",
        "gemma4_ltx25_conversion_map.draft.json": "fa1b68f134cc8f2803c03f26b67dfedaa94a93d26b5e781cb1c26842279a6226",
        "gemma4_ltx25_conversion_map.json": "a30d81973656a5f790d02c496b5bc62a33c6a00b48a500d79848314d00726a14",
        "gemma4_ltx25_inventory.json": "816ec1b295215b404311bcbfcd04f12fea94631bb729af15a8e624b877c2004d",
        "ltx23_q4km_typemap.json": "3d67c6c93662249a382398f2fc5307b07be7303ccd3e6fb1219b0edd6eebb485",
        "ltx25_builder_oracle.json": "aef9ed6ec412dff5b99209bbc0d977f1c15f7d67113ac78af12647047cbc0805",
        "ltx25_conversion_map.draft.json": "dd74d045039c9fdf7ed4234bdc51a0900ca9b34212298e619bff43c2690f2a91",
        "ltx25_conversion_map.json": "c9096c7d13bdd164e5aba2c1b07da57cf5c23ce6ec87bc6c8abf3ce132af73de",
        "ltx25_inventory.json": "7316b660c787d99dcefde3b2bcddf2ca928e38e030d8cfba6356e5a44d0722ef",
    }
    typemap_dir = Path(__file__).resolve().parents[1] / "typemap"
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(typemap_dir.glob("*.json"))
    }
    assert actual == expected


# --------------------------------------------------------------------------
# end-to-end conversion (plan items 18-19 plus the output-contract guards)
# --------------------------------------------------------------------------
def _policy_artifacts(tmp_path, tensors=None, **kwargs):
    """Source + oracle + approved official-format type policy for one conversion."""
    source, oracle = _artifacts(tmp_path, tensors, **kwargs)
    map_path = tmp_path / "map.json"
    _write_policy_map(map_path)
    return source, oracle, map_path


def _convert(tmp_path, output, *, quant_type="Q6_K", tensors=None, **kwargs):
    source, oracle, map_path = _policy_artifacts(tmp_path, tensors)
    manifest = comfyquant.convert_comfyquant(
        source,
        output,
        builder_oracle_path=oracle,
        map_path=map_path,
        quant_type=quant_type,
        quant_workers=1,
        **kwargs,
    )
    return manifest, source, oracle, map_path


def _kv_keys(reader) -> set[str]:
    return {key for key in reader.fields if not key.startswith("GGUF.")}


def _tensors_by_name(output) -> dict:
    return {tensor.name: tensor for tensor in GGUFReader(str(output)).tensors}


def _w4a8_tensors(layer: str) -> list:
    """The five raw tensors of one ``asym_w4a8_int8`` layer, values fixed."""
    marker = _marker_bytes(W4A8_MARKER)
    return [
        (f"{layer}.weight", "I8", _i8(OUT_FEATURES * (IN_FEATURES // 2)), [OUT_FEATURES, IN_FEATURES // 2]),
        (
            f"{layer}.weight_s_rel",
            "F8_E4M3",
            bytes([0x38] * (OUT_FEATURES * IN_FEATURES // 16)),
            [OUT_FEATURES, IN_FEATURES // 16],
        ),
        (f"{layer}.weight_s_channel", "F32", _f32([0.1, 0.2, 0.3, 0.4]), [OUT_FEATURES]),
        (
            f"{layer}.weight_codebook",
            "F32",
            _f32(np.linspace(-8.0, 7.0, 16, dtype=np.float32)),
            [16],
        ),
        (f"{layer}.comfy_quant", "U8", marker, [len(marker)]),
    ]


def _w4a8_reference() -> np.ndarray:
    """What :mod:`converter.comfy_dequant` alone makes of :func:`_w4a8_tensors`."""
    return comfy_dequant.dequantize_layer(
        comfy_dequant.parse_quant_marker(_marker_bytes(W4A8_MARKER)),
        {
            "weight": np.frombuffer(
                _i8(OUT_FEATURES * (IN_FEATURES // 2)), dtype=np.int8
            ).reshape(OUT_FEATURES, IN_FEATURES // 2),
            "weight_s_rel": np.frombuffer(
                bytes([0x38] * (OUT_FEATURES * IN_FEATURES // 16)), dtype=np.uint8
            ).reshape(OUT_FEATURES, IN_FEATURES // 16),
            "weight_s_channel": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "weight_codebook": np.linspace(-8.0, 7.0, 16, dtype=np.float32),
        },
        IN_FEATURES,
    )


@pytest.mark.parametrize("quant_type", ["Q6_K", "Q4_K"])
def test_item18_full_pipeline_converts_and_self_verifies(tmp_path, quant_type):
    """inspect -> convert -> self-verify on the miniature, for both type policies."""
    output = tmp_path / f"fixture-{quant_type}.gguf"
    manifest, source, oracle, map_path = _convert(tmp_path, output, quant_type=quant_type)

    reader = GGUFReader(str(output))
    assert [tensor.name for tensor in reader.tensors] == sorted(LOGICAL_NAMES)
    assert {tensor.name: tensor.tensor_type.name for tensor in reader.tensors} == {
        "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.bias": "BF16",
        # The w4a8 connector weight is dequantized, not requantized.
        "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.weight": "BF16",
        "transformer_blocks.0.attn1.to_q.bias": "BF16",
        "transformer_blocks.0.attn1.to_q.weight": quant_type,
        "transformer_blocks.0.scale_shift_table": "F32",
        "transformer_blocks.0.to_gate_logits.weight": quant_type,
        "video_embeddings_connector.proj.bias": "BF16",
        "video_embeddings_connector.proj.weight": "BF16",
    }

    # Exactly seven KV entries: the source's own quantization metadata is dropped.
    assert _kv_keys(reader) == {
        "general.architecture",
        "general.quantization_version",
        "general.file_type",
        "config",
        "license",
        "model_version",
        "gemma_source_checkpoint",
    }
    assert len(_kv_keys(reader)) == 7
    assert reader.fields["config"].contents().encode("utf-8") == CONFIG.encode("utf-8")
    assert reader.fields["gemma_source_checkpoint"].contents().encode(
        "utf-8"
    ) == FIXTURE_GEMMA_SOURCE_CHECKPOINT.encode("utf-8")

    inventory = comfyquant.inspect_comfyquant(source, builder_oracle_path=oracle)
    policy, map_sha256 = comfyquant.load_official_type_policy(map_path)
    records = comfyquant.records_for_inventory_and_policy(inventory, policy, quant_type)
    with ltx25._SafetensorsRaw(source) as raw:
        metadata = raw.metadata()
    # The size is the records' own prediction, not a hard-coded constant.
    assert output.stat().st_size == ltx25.estimate_gguf_size(records, metadata)

    assert manifest == {
        "format": "nz-ltx25-comfyquant-manifest-v1",
        "profile": "ltx25-comfyquant",
        "source_path": str(source),
        "source_size": source.stat().st_size,
        "source_sha256": ltx25.sha256_of_file(source),
        "source_quant_format": "mixed:w4a8+int8",
        "source_quant_mixed_hi_layers": "1",
        "quant_kind_counts": {"asym_w4a8_int8": 1, "int8_tensorwise": 1, "plain": 6},
        "official_map_path": str(map_path),
        "official_map_sha256": map_sha256,
        "builder_oracle_sha256": ltx25.load_builder_oracle(oracle)["oracle_sha256"],
        "inventory_sha256": inventory.inventory_sha256,
        "quant_type": quant_type,
        "dequant_layout": {
            "nibble_order": "low-first",
            "s_rel_op": "multiply",
            "int8_grid_rounding": True,
            "hadamard": "regular-H4-kron/sqrt(size)",
            "rotation_dtype": "float32",
        },
        "output_sha256": ltx25.sha256_of_file(output),
        "output_size": output.stat().st_size,
        "tensor_count": 8,
        "type_counts": {"BF16": 5, "F32": 1, quant_type: 2},
        "tool_version": converter.__version__,
        "general_file_type_note": comfyquant.GENERAL_FILE_TYPE_NOTE,
    }
    assert json.loads(Path(f"{output}.manifest.json").read_text(encoding="utf-8")) == manifest
    written_inventory = json.loads(
        comfyquant.default_inventory_path(output).read_text(encoding="utf-8")
    )
    assert written_inventory["inventory_sha256"] == inventory.inventory_sha256

    assert (
        comfyquant.verify_comfyquant(
            source, output, builder_oracle_path=oracle, map_path=map_path
        )
        == manifest
    )


def test_item19_source_quantization_metadata_never_reaches_the_output_kv(tmp_path):
    output = tmp_path / "fixture.gguf"
    manifest, *_ = _convert(tmp_path, output)
    fields = GGUFReader(str(output)).fields
    assert "quant_format" not in fields
    assert "quant_mixed_hi_layers" not in fields
    # It is not lost, only kept out of the GGUF: the manifest carries the provenance.
    assert manifest["source_quant_format"] == "mixed:w4a8+int8"
    assert manifest["source_quant_mixed_hi_layers"] == "1"


def test_an_existing_output_is_replaced_only_with_force(tmp_path):
    output = tmp_path / "fixture.gguf"
    first, source, oracle, map_path = _convert(tmp_path, output)

    with pytest.raises(ltx25.OutputExistsError) as exc:
        comfyquant.convert_comfyquant(
            source,
            output,
            builder_oracle_path=oracle,
            map_path=map_path,
            quant_type="Q6_K",
            quant_workers=1,
        )
    assert "--force" in str(exc.value)

    again = comfyquant.convert_comfyquant(
        source,
        output,
        builder_oracle_path=oracle,
        map_path=map_path,
        quant_type="Q6_K",
        quant_workers=1,
        force=True,
    )
    assert again == first
    assert ltx25.sha256_of_file(output) == first["output_sha256"]


def test_a_failure_during_the_write_leaves_no_output_and_no_temporary(tmp_path, monkeypatch):
    source, oracle, map_path = _policy_artifacts(tmp_path)
    output = tmp_path / "fixture.gguf"

    def _boom(*args, **kwargs):
        raise comfy_dequant.ComfyDequantError("synthetic sidecar failure")

    monkeypatch.setattr(comfyquant.comfy_dequant, "dequantize_layer", _boom)
    # The dequantiser's own error is wrapped in a profile error, so the CLI
    # prints it as a message instead of a traceback.
    with pytest.raises(comfyquant.ComfyQuantError) as exc:
        comfyquant.convert_comfyquant(
            source,
            output,
            builder_oracle_path=oracle,
            map_path=map_path,
            quant_type="Q6_K",
            quant_workers=1,
        )
    assert "synthetic sidecar failure" in str(exc.value)
    assert not output.exists()
    assert not Path(f"{output}.manifest.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_layer_that_cannot_be_dequantized_names_itself_without_a_traceback(tmp_path):
    """A non-finite dequantisation result is a profile error, not a raw ValueError.

    The source passes every structural check -- a NaN ``weight_scale`` is a
    well-formed F32 ``[out, 1]`` tensor -- so the failure can only surface
    during the write, where it must still arrive as an ``Ltx25Error`` carrying
    the offending layer's key.
    """
    tensors = _replace(
        _default_tensors(),
        f"{INT8_LAYER}.weight_scale",
        "F32",
        _f32([0.01, float("nan"), 0.03, 0.04]),
        [OUT_FEATURES, 1],
    )
    source, oracle = _artifacts(tmp_path, tensors)
    map_path = tmp_path / "map.json"
    _write_policy_map(map_path)
    output = tmp_path / "fixture.gguf"

    with pytest.raises(comfyquant.ComfyQuantError) as exc:
        comfyquant.convert_comfyquant(
            source,
            output,
            builder_oracle_path=oracle,
            map_path=map_path,
            quant_type="Q6_K",
            quant_workers=1,
        )
    message = str(exc.value)
    assert isinstance(exc.value, ltx25.Ltx25Error)
    assert f"{INT8_LAYER}.weight" in message
    assert "int8_tensorwise" in message
    assert "non-finite" in message
    assert not output.exists()
    assert not Path(f"{output}.manifest.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_tampered_manifest_quant_type_fails_self_verification(tmp_path):
    output = tmp_path / "fixture.gguf"
    _, source, oracle, map_path = _convert(tmp_path, output, quant_type="Q6_K")
    manifest_path = Path(f"{output}.manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["quant_type"] = "Q4_K"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ltx25.ManifestError) as exc:
        comfyquant.verify_comfyquant(
            source, output, builder_oracle_path=oracle, map_path=map_path
        )
    message = str(exc.value)
    assert "type_counts" in message
    assert "'Q4_K'" in message


def test_a_bf16_target_row_is_byte_identical_to_f32_to_bf16_u16(tmp_path):
    """The dequantized connector row must round exactly the way the converter rounds."""
    output = tmp_path / "fixture.gguf"
    _convert(tmp_path, output)
    tensor = _tensors_by_name(output)[
        "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_k.weight"
    ]
    assert tensor.tensor_type.name == "BF16"
    expected = comfy_dequant.f32_to_bf16_u16(_w4a8_reference()).reshape(-1)
    actual = np.ascontiguousarray(np.asarray(tensor.data)).view(np.uint16).reshape(-1)
    assert np.array_equal(actual, expected)


def test_a_quantized_w4a8_row_stays_within_the_q6_k_error_budget(tmp_path):
    """The 513 transformer w4a8 rows take dequantize -> Q6_K; measure that path.

    The miniature's own w4a8 layer is a connector (BF16 out), so this variant
    moves the same layer into the transformer, where the official map's Q4_K row
    becomes the selected ``--quant-type``.
    """
    layer = INT8_LAYER
    dropped = {f"{layer}.weight", f"{layer}.weight_scale", f"{layer}.comfy_quant"}
    tensors = [row for row in _default_tensors() if row[0] not in dropped]
    tensors += _w4a8_tensors(layer)
    output = tmp_path / "w4a8-Q6_K.gguf"
    _convert(tmp_path, output, quant_type="Q6_K", tensors=tensors)

    expected = _w4a8_reference()
    tensor = _tensors_by_name(output)["transformer_blocks.0.attn1.to_q.weight"]
    assert tensor.tensor_type.name == "Q6_K"
    actual = dequantize(
        np.ascontiguousarray(np.asarray(tensor.data)).view(np.uint8).reshape(-1),
        tensor.tensor_type,
    ).reshape(expected.shape)
    rel_rmse = float(
        np.sqrt(np.mean((actual - expected) ** 2)) / np.sqrt(np.mean(expected**2))
    )
    assert rel_rmse < 0.03, rel_rmse


def test_an_extra_known_metadata_key_breaks_the_kv_contract(tmp_path, monkeypatch):
    """The KV contract is a set, so even a key ``apply_kv`` knows is a failure.

    ``encrypted_wandb_properties`` is one of the optional strings the shared
    metadata transcoder copies; the community file must not carry it, and if a
    future one does the conversion stops instead of shipping an eighth KV entry.

    The contract is decided from the source metadata, so it stops *before* the
    write: on the real artifact the alternative is discovering it after roughly
    19.6 GB has been written.  Booby-trapping ``_temp_path`` is what proves the
    write never started -- no temporary file can be created without it.
    """
    metadata = _fixture_metadata(
        CONFIG,
        license="fixture",
        model_version="2.5.0",
        encrypted_wandb_properties="{}",
    )
    source, oracle = _artifacts(tmp_path, metadata=metadata)
    map_path = tmp_path / "map.json"
    _write_policy_map(map_path)
    output = tmp_path / "fixture.gguf"

    def _must_not_be_reached(final_path):
        raise AssertionError(f"the write phase started for {final_path}")

    monkeypatch.setattr(comfyquant.ltx25, "_temp_path", _must_not_be_reached)

    with pytest.raises(comfyquant.ComfyQuantOutputError) as exc:
        comfyquant.convert_comfyquant(
            source,
            output,
            builder_oracle_path=oracle,
            map_path=map_path,
            quant_type="Q6_K",
            quant_workers=1,
        )
    assert "unexpected=['encrypted_wandb_properties']" in str(exc.value)
    assert not output.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_missing_known_metadata_key_also_breaks_the_kv_contract_before_the_write(
    tmp_path, monkeypatch
):
    """The projection fails in both directions: a key the contract needs is gone."""
    metadata = _fixture_metadata(CONFIG, model_version="2.5.0")
    assert "license" not in metadata
    source, oracle = _artifacts(tmp_path, metadata=metadata)
    map_path = tmp_path / "map.json"
    _write_policy_map(map_path)
    output = tmp_path / "fixture.gguf"

    def _must_not_be_reached(final_path):
        raise AssertionError(f"the write phase started for {final_path}")

    monkeypatch.setattr(comfyquant.ltx25, "_temp_path", _must_not_be_reached)

    with pytest.raises(comfyquant.ComfyQuantOutputError) as exc:
        comfyquant.convert_comfyquant(
            source,
            output,
            builder_oracle_path=oracle,
            map_path=map_path,
            quant_type="Q6_K",
            quant_workers=1,
        )
    assert "missing=['license']" in str(exc.value)
    assert not output.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_the_approved_official_map_projects_the_official_type_counts():
    """The policy-derived post-assertion is the official histogram on the real map."""
    map_path = Path(__file__).resolve().parents[1] / "typemap" / "ltx25_conversion_map.json"
    policy, _ = comfyquant.load_official_type_policy(map_path)
    assert len(policy) == comfyquant.OFFICIAL_ROWS == 4349
    for quant_type in comfyquant.QUANT_TYPES:
        assert comfyquant.expected_type_counts_for_policy(
            policy, quant_type
        ) == comfyquant.official_type_counts(quant_type)
        # A conversion off this map arms the official histogram as well, not
        # only the map's own projection of itself.
        assert comfyquant._expected_type_counts_for_conversion(
            policy, quant_type
        ) == comfyquant.official_type_counts(quant_type)


def test_a_full_size_policy_is_held_to_the_official_histogram():
    """4,349 rows means "this is the official map"; its histogram is not negotiable.

    ``expected_type_counts_for_policy`` reads the histogram off the policy it is
    about to check, so an edited full-size map would simply assert its own edit.
    One row fewer is a miniature, which keeps stating its own histogram.
    """
    edited = {
        f"transformer_blocks.{index}.weight": {"shape_logical": [4, 4], "ggml_type": "BF16"}
        for index in range(comfyquant.OFFICIAL_ROWS)
    }
    assert comfyquant.expected_type_counts_for_policy(edited, "Q6_K") == {
        "BF16": comfyquant.OFFICIAL_ROWS
    }
    with pytest.raises(comfyquant.ComfyQuantPolicyError) as exc:
        comfyquant._expected_type_counts_for_conversion(edited, "Q6_K")
    assert str(comfyquant.OFFICIAL_ROWS) in str(exc.value)
    assert "2401" in str(exc.value)

    smaller = dict(list(edited.items())[:-1])
    assert comfyquant._expected_type_counts_for_conversion(smaller, "Q6_K") == {
        "BF16": comfyquant.OFFICIAL_ROWS - 1
    }


# --------------------------------------------------------------------------
# CLI convert / self-verify
# --------------------------------------------------------------------------
def _cli_config(tmp_path, oracle, map_path) -> dict:
    return {
        "profiles": {
            "ltx25-comfyquant": {
                "official_map_path": str(map_path),
                "builder_oracle_path": str(oracle),
                "output_dir": str(tmp_path / "output"),
                "quant_workers": 4,
                "quant_type": "Q6_K",
            }
        }
    }


def test_cli_convert_then_self_verify_round_trips(tmp_path, capsys):
    source, oracle, map_path = _policy_artifacts(tmp_path)
    config = _cli_config(tmp_path, oracle, map_path)
    parser = cli.build_parser()
    output = tmp_path / "output" / "fixture-Q6_K.gguf"

    args = parser.parse_args(["convert", "--model", "ltx25-comfyquant", "--st-path", str(source)])
    assert cli.cmd_convert(args, config) == 0
    out = capsys.readouterr().out
    assert output.is_file()
    assert f"conversion/self-verify complete: {output}" in out
    assert "{'BF16': 5, 'F32': 1, 'Q6_K': 2}" in out
    assert f"{output}.manifest.json" in out

    args = parser.parse_args(
        ["self-verify", "--model", "ltx25-comfyquant", "--st-path", str(source)]
    )
    assert cli.cmd_verify(args, config) == 0
    assert f"static self-verification passed: {output}" in capsys.readouterr().out


def test_cli_convert_and_self_verify_require_an_explicit_source(tmp_path, capsys):
    source, oracle, map_path = _policy_artifacts(tmp_path)
    config = _cli_config(tmp_path, oracle, map_path)
    parser = cli.build_parser()
    for command, handler in (("convert", cli.cmd_convert), ("self-verify", cli.cmd_verify)):
        args = parser.parse_args([command, "--model", "ltx25-comfyquant"])
        assert handler(args, config) == 1
        captured = capsys.readouterr()
        assert "--st-path is required" in captured.err, command
        assert "Traceback" not in captured.err

    # self-verify also reports a missing output rather than falling through.
    args = parser.parse_args(
        ["self-verify", "--model", "ltx25-comfyquant", "--st-path", str(source)]
    )
    assert cli.cmd_verify(args, config) == 1
    assert "output GGUF not found" in capsys.readouterr().err
