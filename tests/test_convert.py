"""Tests for converter.convert: the two-pass streaming safetensors -> GGUF body.

The real source model is ~43 GB, so every test here works on tiny, self-made
fixtures that still exercise all code paths:

* an F32 target tensor (1D, promoted from a BF16 source),
* a BF16 target tensor (2D, byte-for-byte passthrough),
* Q4_K / Q5_K / Q6_K target tensors (2D, last logical dim a multiple of 256),
* skip-target components (``vae.*`` / ``audio_vae.*``).

Fixtures are written by hand because safetensors' NumPy framework cannot
represent BF16 (there is no NumPy bfloat16). We build the safetensors header +
payload ourselves so we can store genuine ``dtype="BF16"`` tensors.
"""

from __future__ import annotations

import json
import os
import struct

import gguf
import gguf.quants as gq
import numpy as np
import pytest

from converter.convert import _SafetensorsRaw, _register_tensor_info, _tensor_payload, convert
from converter.typemap import EXPECTED_TYPE_COUNTS, save_typemap

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
_TYPE_SIZE = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}


def _f32_to_bf16_bits(f32: np.ndarray) -> np.ndarray:
    """Round float32 -> bf16 bit pattern (round-half-to-even)."""
    u32 = np.ascontiguousarray(f32, np.float32).view(np.uint32)
    bias = ((u32 >> 16) & 1) + 0x7FFF
    return ((u32 + bias) >> 16).astype(np.uint16)


def _bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """Expand a bf16 bit pattern back to float32 (exact)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _nbytes(shape: list[int], ggml_type: str) -> int:
    n = 1
    for d in shape:
        n *= d
    if ggml_type == "F32":
        return n * 4
    if ggml_type == "BF16":
        return n * 2
    return (n // 256) * _TYPE_SIZE[ggml_type]


def _write_safetensors(path: str, tensors: list[tuple], metadata: dict | None) -> None:
    """Write a safetensors file from ``(name, dtype_str, raw_bytes, shape)`` tuples."""
    header: dict = {}
    blob = bytearray()
    off = 0
    for name, dtype, raw, shape in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [off, off + len(raw)],
        }
        blob += raw
        off += len(raw)
    if metadata is not None:
        header["__metadata__"] = metadata
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)


# Default fixture: (stripped_name, ggml_type, logical_shape)
_DEFAULT_SPECS = [
    ("adaln_single.linear.bias", "F32", [512]),
    ("transformer_blocks.0.embeddings_connector.weight", "BF16", [4, 8]),
    ("transformer_blocks.1.attn.to_q.weight", "Q4_K", [4, 512]),
    ("transformer_blocks.1.attn.to_v.weight", "Q6_K", [4, 512]),
    ("transformer_blocks.0.ff.net.0.weight", "Q5_K", [4, 256]),
]

_DEFAULT_METADATA = {
    "config": json.dumps({"model_type": "ltxv", "hidden_size": 4096}),
    "license": "fixture-license",
    "model_version": "0.0.1",
    "encrypted_wandb_properties": "opaque",
}

_DEFAULT_SKIPS = [
    ("vae.decoder.weight", "F32", np.zeros(4, np.float32).tobytes(), [2, 2]),
    ("audio_vae.enc.weight", "BF16", np.zeros(4, np.uint16).tobytes(), [2, 2]),
]


def _build(tmp_path, specs=_DEFAULT_SPECS, metadata=_DEFAULT_METADATA,
           skips=_DEFAULT_SKIPS, typemap_specs=None):
    """Build a fixture; returns (st_path, tm_path, out_path, ground_truth).

    ``typemap_specs`` defaults to ``specs`` but can differ to create
    source/typemap mismatches for error-path tests.
    """
    rng = np.random.default_rng(1234)
    st_tensors: list[tuple] = list(skips)
    ground: dict = {}
    for name, ggml_type, shape in specs:
        n = int(np.prod(shape))
        f32 = (rng.standard_normal(n).astype(np.float32) * np.float32(0.5))
        bits = _f32_to_bf16_bits(f32)
        canon = _bf16_bits_to_f32(bits).reshape(shape)
        ground[name] = {"type": ggml_type, "canon": canon, "bits": bits, "shape": shape}
        st_tensors.append(
            ("model.diffusion_model." + name, "BF16", bits.astype("<u2").tobytes(), shape)
        )

    tm_source = typemap_specs if typemap_specs is not None else specs
    recs = [
        {
            "name": name,
            "ggml_type": ggml_type,
            "shape_gguf": list(reversed(shape)),
            "shape_logical": list(shape),
            "n_elements": int(np.prod(shape)),
            "nbytes": _nbytes(list(shape), ggml_type),
        }
        for name, ggml_type, shape in tm_source
    ]

    st_path = os.path.join(str(tmp_path), "model.safetensors")
    tm_path = os.path.join(str(tmp_path), "typemap.json")
    out_path = os.path.join(str(tmp_path), "out.gguf")
    _write_safetensors(st_path, st_tensors, metadata)
    save_typemap(recs, tm_path, source="fixture")
    return st_path, tm_path, out_path, ground


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------
def test_convert_full_roundtrip(tmp_path):
    st, tm, out, ground = _build(tmp_path)
    result = convert(st, tm, out, reference_expected=False)
    assert str(result) == out
    assert os.path.isfile(out)

    rd = gguf.GGUFReader(out)

    # --- tensor count, order, type, shape (ne order) --------------------
    assert len(rd.tensors) == len(_DEFAULT_SPECS)
    for i, (name, ggml_type, shape) in enumerate(_DEFAULT_SPECS):
        t = rd.tensors[i]
        assert t.name == name
        assert t.tensor_type.name == ggml_type
        assert [int(d) for d in t.shape] == list(reversed(shape))

    # --- KV metadata ----------------------------------------------------
    fields = rd.fields
    assert fields["general.architecture"].contents() == "ltxv"
    assert fields["general.quantization_version"].contents() == 2
    assert fields["general.file_type"].contents() == 15
    assert json.loads(fields["config"].contents()) == {"model_type": "ltxv", "hidden_size": 4096}
    assert fields["license"].contents() == "fixture-license"
    assert fields["model_version"].contents() == "0.0.1"
    assert fields["encrypted_wandb_properties"].contents() == "opaque"

    # --- per-tensor value checks ----------------------------------------
    by_name = {t.name: t for t in rd.tensors}

    # F32: exact bf16->f32 promotion
    g = ground["adaln_single.linear.bias"]
    f32_data = np.asarray(by_name["adaln_single.linear.bias"].data).reshape(g["shape"])
    assert np.array_equal(f32_data, g["canon"])

    # BF16: byte-for-byte identical to the source representation
    g = ground["transformer_blocks.0.embeddings_connector.weight"]
    bf = by_name["transformer_blocks.0.embeddings_connector.weight"]
    assert bf.data.tobytes() == g["bits"].astype("<u2").tobytes()

    # K-quant: dequantize and compare within a tolerance appropriate to the type
    tolerances = {"Q4_K": 0.5, "Q5_K": 0.25, "Q6_K": 0.15}
    for name in (
        "transformer_blocks.1.attn.to_q.weight",   # Q4_K
        "transformer_blocks.1.attn.to_v.weight",   # Q6_K
        "transformer_blocks.0.ff.net.0.weight",    # Q5_K
    ):
        g = ground[name]
        t = by_name[name]
        deq = gq.dequantize(np.asarray(t.data), t.tensor_type).reshape(g["shape"])
        max_err = float(np.abs(deq - g["canon"]).max())
        assert max_err < tolerances[g["type"]], (name, g["type"], max_err)


def test_convert_reference_expected_true_ok_for_small_typemap(tmp_path):
    # reference_expected=True must not trip the full-reference count guard when
    # the typemap is not the full 4444-tensor reference.
    st, tm, out, _ = _build(tmp_path)
    convert(st, tm, out, reference_expected=True)
    assert os.path.isfile(out)


def test_convert_returns_pathlib_path(tmp_path):
    from pathlib import Path

    st, tm, out, _ = _build(tmp_path)
    result = convert(st, tm, out, reference_expected=False)
    assert isinstance(result, Path)


def test_register_tensor_info_uses_value_error_not_optimization_assertion():
    writer = gguf.GGUFWriter(None, arch="ltxv")
    record = {
        "ggml_type": "BF16",
        "shape_logical": [4],
        "nbytes": 999,
    }
    with pytest.raises(ValueError, match="computed nbytes"):
        _register_tensor_info(writer, "fixture.weight", record)


def test_i8_register_and_payload_are_byte_identical_passthrough(tmp_path):
    """I8 (added for gemma4-ltx25's U8 sidecars) is a raw, block-size-1 passthrough."""
    st = tmp_path / "u8.safetensors"
    payload = bytes(range(1, 17))  # 16 arbitrary bytes, avoids an all-zero false positive
    header = {"sidecar": {"dtype": "U8", "shape": [16], "data_offsets": [0, 16]}}
    header_bytes = json.dumps(header).encode("utf-8")
    st.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)

    record = {"name": "sidecar", "ggml_type": "I8", "shape_logical": [16], "nbytes": 16}
    writer = gguf.GGUFWriter(None, arch="ltxv")
    nbytes = _register_tensor_info(writer, "sidecar", record)
    assert nbytes == 16

    with _SafetensorsRaw(st) as reader:
        result = _tensor_payload(reader, "sidecar", record)
    assert result.dtype == np.int8
    assert result.nbytes == 16
    assert result.tobytes() == payload


def test_i8_output_gguf_roundtrips_byte_identical(tmp_path):
    """End-to-end: an I8 tensor written through the shared writer reads back verbatim."""
    st = tmp_path / "u8.safetensors"
    payload = bytes(range(1, 17))
    header = {"sidecar": {"dtype": "U8", "shape": [16], "data_offsets": [0, 16]}}
    header_bytes = json.dumps(header).encode("utf-8")
    st.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    record = {"name": "sidecar", "ggml_type": "I8", "shape_logical": [16], "nbytes": 16}

    out = tmp_path / "i8.gguf"
    with _SafetensorsRaw(st) as reader:
        writer = gguf.GGUFWriter(str(out), arch="ltxv")
        _register_tensor_info(writer, "sidecar", record)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        writer.write_tensor_data(_tensor_payload(reader, "sidecar", record))
        writer.close()

    rd = gguf.GGUFReader(str(out))
    assert len(rd.tensors) == 1
    tensor = rd.tensors[0]
    assert tensor.tensor_type.name == "I8"
    assert np.asarray(tensor.data).view(np.uint8).tobytes() == payload


def test_existing_ltx23_typemap_never_uses_i8():
    """Regression fence: adding I8 support must not change ltx23's own type mix."""
    assert "I8" not in EXPECTED_TYPE_COUNTS


# --------------------------------------------------------------------------
# error paths
# --------------------------------------------------------------------------
def test_source_key_missing_from_typemap_raises(tmp_path):
    # typemap references a tensor the source does not contain.
    typemap_specs = _DEFAULT_SPECS + [("does.not.exist.weight", "Q4_K", [4, 256])]
    st, tm, out, _ = _build(tmp_path, typemap_specs=typemap_specs)
    with pytest.raises(ValueError, match="missing from source"):
        convert(st, tm, out, reference_expected=False)


def test_typemap_missing_source_key_raises(tmp_path):
    # source has a diffusion tensor absent from the typemap.
    typemap_specs = _DEFAULT_SPECS[:-1]  # drop the last one from the typemap only
    st, tm, out, _ = _build(tmp_path, typemap_specs=typemap_specs)
    with pytest.raises(ValueError, match="not in typemap"):
        convert(st, tm, out, reference_expected=False)


def test_missing_config_metadata_raises(tmp_path):
    metadata = {"license": "x"}  # no 'config'
    st, tm, out, _ = _build(tmp_path, metadata=metadata)
    with pytest.raises(ValueError, match="config"):
        convert(st, tm, out, reference_expected=False)


def test_unexpected_prefix_key_raises(tmp_path):
    # A source key that is neither a diffusion tensor nor a known skip component.
    skips = _DEFAULT_SKIPS + [
        ("mystery.block.weight", "F32", np.zeros(4, np.float32).tobytes(), [2, 2]),
    ]
    st, tm, out, _ = _build(tmp_path, skips=skips)
    with pytest.raises(ValueError, match="neither the diffusion prefix"):
        convert(st, tm, out, reference_expected=False)


def test_skip_prefixes_are_ignored_not_written(tmp_path):
    # The default fixture already includes vae.* / audio_vae.* skip tensors;
    # confirm they never appear in the output GGUF.
    st, tm, out, _ = _build(tmp_path)
    convert(st, tm, out, reference_expected=False)
    rd = gguf.GGUFReader(out)
    names = {t.name for t in rd.tensors}
    assert not any(n.startswith(("vae.", "audio_vae.")) for n in names)
    assert len(names) == len(_DEFAULT_SPECS)
