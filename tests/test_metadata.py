"""Tests for converter.metadata: safetensors __metadata__ -> GGUF KV transcription."""

from __future__ import annotations

import json
import os

import gguf
import numpy as np
import pytest
from safetensors.numpy import save_file

from converter.metadata import apply_kv, read_st_metadata, validate_metadata

REFERENCE_GGUF = (
    r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\models\ltx-2.3-gguf"
    r"\LTX-2.3-distilled-1.1\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
)

FULL_METADATA = {
    "config": json.dumps({"model_type": "ltxv", "hidden_size": 4096}),
    "license": "some-license-text",
    "model_version": "2.3.0",
    "encrypted_wandb_properties": "opaque-blob-of-wandb-data",
}


def _make_safetensors(tmp_path, metadata):
    path = os.path.join(str(tmp_path), "dummy.safetensors")
    tensors = {"weight": np.zeros((2, 2), dtype=np.float32)}
    save_file(tensors, path, metadata=metadata)
    return path


# --- read_st_metadata -------------------------------------------------


def test_read_st_metadata_round_trip(tmp_path):
    path = _make_safetensors(tmp_path, FULL_METADATA)
    meta = read_st_metadata(path)
    assert meta == FULL_METADATA


def test_read_st_metadata_no_metadata_block(tmp_path):
    path = os.path.join(str(tmp_path), "no_meta.safetensors")
    save_file({"weight": np.zeros((2, 2), dtype=np.float32)}, path)
    meta = read_st_metadata(path)
    assert meta == {}


# --- validate_metadata: happy path ------------------------------------


def test_validate_metadata_accepts_full_metadata():
    validate_metadata(dict(FULL_METADATA))  # should not raise


def test_validate_metadata_warns_but_passes_when_optional_keys_missing(caplog):
    meta = {"config": FULL_METADATA["config"]}
    with caplog.at_level("WARNING"):
        validate_metadata(meta)  # should not raise
    warned_keys = "\n".join(r.message for r in caplog.records)
    assert "license" in warned_keys
    assert "model_version" in warned_keys
    assert "encrypted_wandb_properties" in warned_keys


# --- validate_metadata: error cases ------------------------------------


def test_validate_metadata_missing_config_raises():
    meta = {"license": "x"}
    with pytest.raises(ValueError, match="config"):
        validate_metadata(meta)


def test_validate_metadata_empty_config_raises():
    meta = {"config": ""}
    with pytest.raises(ValueError, match="empty"):
        validate_metadata(meta)


def test_validate_metadata_invalid_json_config_raises():
    meta = {"config": "{not valid json"}
    with pytest.raises(ValueError, match="JSON"):
        validate_metadata(meta)


# --- apply_kv -----------------------------------------------------------


def test_apply_kv_writes_expected_keys(tmp_path):
    out_path = os.path.join(str(tmp_path), "out.gguf")

    writer = gguf.GGUFWriter(out_path, arch="ltxv")
    apply_kv(writer, dict(FULL_METADATA))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()

    reader = gguf.GGUFReader(out_path)
    fields = reader.fields

    arch_field = fields["general.architecture"]
    assert arch_field.types == [gguf.GGUFValueType.STRING]
    assert arch_field.contents() == "ltxv"

    qv_field = fields["general.quantization_version"]
    assert qv_field.types == [gguf.GGUFValueType.UINT32]
    assert qv_field.contents() == 2

    ft_field = fields["general.file_type"]
    assert ft_field.types == [gguf.GGUFValueType.UINT32]
    assert ft_field.contents() == 15

    for key in ("config", "license", "model_version", "encrypted_wandb_properties"):
        field = fields[key]
        assert field.types == [gguf.GGUFValueType.STRING]
        assert field.contents() == FULL_METADATA[key]


def test_apply_kv_skips_missing_optional_keys(tmp_path):
    out_path = os.path.join(str(tmp_path), "out_partial.gguf")

    meta = {"config": FULL_METADATA["config"]}  # no license/model_version/wandb

    writer = gguf.GGUFWriter(out_path, arch="ltxv")
    apply_kv(writer, meta)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()

    reader = gguf.GGUFReader(out_path)
    fields = reader.fields

    assert "config" in fields
    for key in ("license", "model_version", "encrypted_wandb_properties"):
        assert key not in fields

    # required KVs are still present regardless
    assert fields["general.quantization_version"].contents() == 2
    assert fields["general.file_type"].contents() == 15


# --- reference GGUF cross-check -----------------------------------------


@pytest.mark.skipif(
    not os.path.isfile(REFERENCE_GGUF),
    reason="reference GGUF not available in this environment",
)
def test_reference_gguf_kv_layout_matches_spec():
    reader = gguf.GGUFReader(REFERENCE_GGUF)
    fields = reader.fields

    assert fields["general.architecture"].types == [gguf.GGUFValueType.STRING]
    assert fields["general.architecture"].contents() == "ltxv"

    assert fields["general.quantization_version"].types == [gguf.GGUFValueType.UINT32]
    assert fields["general.quantization_version"].contents() == 2

    assert fields["general.file_type"].types == [gguf.GGUFValueType.UINT32]
    assert fields["general.file_type"].contents() == 15

    for key in ("config", "license", "model_version", "encrypted_wandb_properties"):
        assert fields[key].types == [gguf.GGUFValueType.STRING]

    # general.alignment is deliberately not part of the reference KV set.
    assert "general.alignment" not in fields
