"""Unit coverage for the ComfyUI dequantisation primitives.

The tests here deliberately re-derive every quantity a second, independent way:

* the FP8 E4M3FN table is rebuilt from the bit fields with ``math.ldexp`` on
  integer mantissas, a different formulation from the module's own construction;
* the int8 and w4a8 *forward* quantisers live here, so each round-trip test
  exercises the module's inverse against an encoder it did not write;
* a deliberately wrong decoder (nibbles swapped) is run through the same
  comparison to prove the round-trip metrics can actually tell right from wrong.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from converter import comfy_dequant as cd
from converter.convert import _SafetensorsRaw

from test_ltx25 import _write_safetensors


# The 16-level codebook every quantised layer of the real community checkpoint
# carries (all 609 ``weight_codebook`` tensors are byte-identical).  Used here
# so the synthetic round-trip exercises realistic level spacing.
REAL_CODEBOOK = np.array(
    [
        -0.9806020259857178,
        -0.7945290207862854,
        -0.6381649971008301,
        -0.5009859800338745,
        -0.3773210048675537,
        -0.2631869912147522,
        -0.1552100032567978,
        -0.05071999877691269,
        0.05254099890589714,
        0.15698499977588654,
        0.26528400182724,
        0.37953299283981323,
        0.5026360154151917,
        0.6389529705047607,
        0.794875979423523,
        0.9806709885597229,
    ],
    dtype=np.float32,
)


# --------------------------------------------------------------------------
# helpers: metrics
# --------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def _rel_rmse(a: np.ndarray, reference: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))


def _rotate(values: np.ndarray, groupsize: int) -> np.ndarray:
    """Apply the ConvRot rotation (its own inverse) to a [rows, cols] array."""
    rows, cols = values.shape
    matrix = cd.hadamard(groupsize)
    return np.ascontiguousarray(
        (values.reshape(rows, cols // groupsize, groupsize) @ matrix.T).reshape(rows, cols)
    )


# --------------------------------------------------------------------------
# helpers: independent FP8 E4M3FN reference
# --------------------------------------------------------------------------


def _reference_fp8_e4m3(byte: int) -> float:
    """Decode one E4M3FN byte using integer mantissas and ``math.ldexp``.

    Deliberately a different formulation from the module's table builder: the
    value is assembled as ``integer_mantissa * 2**k`` rather than
    ``2**exponent * (1 + mantissa/8)``.
    """
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    if exponent == 0:
        # subnormal: mantissa * 2**-6 / 8 == mantissa * 2**-9
        return math.copysign(math.ldexp(mantissa, -9), sign)
    # normal: (8 + mantissa) * 2**(exponent - 7) / 8 == (8 + mantissa) * 2**(exponent - 10)
    return math.copysign(math.ldexp(8 + mantissa, exponent - 10), sign)


def _nearest_fp8_bytes(values: np.ndarray) -> np.ndarray:
    """Snap non-negative values to the nearest representable E4M3FN byte."""
    finite = cd.FP8_E4M3_TABLE[:0x7F]  # bytes 0x00..0x7E: all finite, all >= 0
    diffs = np.abs(np.asarray(values, dtype=np.float32)[..., None] - finite)
    return diffs.argmin(axis=-1).astype(np.uint8)


# --------------------------------------------------------------------------
# helpers: forward quantisers (test-side encoders)
# --------------------------------------------------------------------------


def _forward_int8(weights: np.ndarray, convrot: bool, groupsize: int = 256):
    """Per-output-row absmax int8 quantisation, optionally after ConvRot."""
    rotated = _rotate(weights, groupsize) if convrot else np.ascontiguousarray(weights)
    scale = np.abs(rotated).max(axis=1) / 127.0
    codes = np.rint(rotated / scale[:, None])
    codes = np.clip(codes, -127.0, 127.0).astype(np.int8)
    return codes, scale.astype(np.float32).reshape(-1, 1)


def _forward_w4a8(
    weights: np.ndarray,
    codebook: np.ndarray,
    groupsize: int = 256,
    group: int = 16,
):
    """Codebook + two-stage (per-row, per-group FP8) scale quantisation."""
    rows, cols = weights.shape
    rotated = _rotate(weights, groupsize)

    s_channel = (np.abs(rotated).max(axis=1) / 127.0).astype(np.float32)
    on_int8_grid = rotated / s_channel[:, None]

    grouped = on_int8_grid.reshape(rows, cols // group, group)
    codebook_absmax = float(np.abs(codebook).max())
    wanted = np.abs(grouped).max(axis=2) / codebook_absmax
    wanted = np.maximum(wanted, np.finfo(np.float32).tiny)
    s_rel_bytes = _nearest_fp8_bytes(wanted)
    s_rel = cd.FP8_E4M3_TABLE[s_rel_bytes]

    normalised = grouped / s_rel[:, :, None]
    codes = np.abs(normalised[..., None] - codebook).argmin(axis=-1).astype(np.uint8)
    codes = codes.reshape(rows, cols)

    packed = ((codes[:, 1::2].astype(np.uint8) << 4) | codes[:, 0::2]).astype(np.uint8)
    return (
        packed.view(np.int8),
        np.ascontiguousarray(codebook, dtype=np.float32),
        s_channel,
        np.ascontiguousarray(s_rel_bytes, dtype=np.uint8),
    )


def _decode_w4a8_swapped_nibbles(
    packed: np.ndarray,
    codebook: np.ndarray,
    s_channel: np.ndarray,
    s_rel_bytes: np.ndarray,
    in_features: int,
    groupsize: int = 256,
    group: int = 16,
) -> np.ndarray:
    """A wrong decoder: high nibble first.  Used to prove detection power."""
    rows = packed.shape[0]
    raw = packed.view(np.uint8)
    codes = np.empty((rows, in_features), dtype=np.uint8)
    codes[:, 0::2] = raw >> 4  # wrong: this is element 2i+1
    codes[:, 1::2] = raw & 0x0F  # wrong: this is element 2i
    values = codebook[codes].reshape(rows, in_features // group, group)
    values = values * cd.FP8_E4M3_TABLE[s_rel_bytes][:, :, None]
    values = np.clip(np.rint(values), -127.0, 127.0).reshape(rows, in_features)
    values = values * s_channel.reshape(rows, 1)
    return _rotate(values.astype(np.float32), groupsize)


def _int8_marker(convrot: bool) -> dict:
    return {
        "format": "int8_tensorwise",
        "convrot": convrot,
        "convrot_groupsize": 256,
        "group_size": None,
    }


W4A8_MARKER = {
    "format": "asym_w4a8_int8",
    "convrot": True,
    "convrot_groupsize": 256,
    "group_size": 16,
}


# ==========================================================================
# 1. Hadamard
# ==========================================================================


def test_hadamard_256_is_symmetric_orthogonal_and_uniform():
    matrix = cd.hadamard(256)
    assert matrix.dtype == np.float32
    assert matrix.shape == (256, 256)
    assert np.array_equal(matrix, matrix.T)  # exact, not approximate
    identity = np.eye(256, dtype=np.float32)
    assert float(np.abs(matrix @ matrix - identity).max()) < 1e-6
    assert np.all(np.abs(matrix) == np.float32(1.0 / 16.0))


@pytest.mark.parametrize("size", [4, 16, 64])
def test_hadamard_smaller_powers_of_four_are_orthogonal(size):
    matrix = cd.hadamard(size)
    assert np.array_equal(matrix, matrix.T)
    identity = np.eye(size, dtype=np.float32)
    assert float(np.abs(matrix @ matrix - identity).max()) < 1e-6
    assert np.allclose(np.abs(matrix), 1.0 / math.sqrt(size))


@pytest.mark.parametrize("size", [0, 1, 2, 3, 8, 32, 128, 255, -4])
def test_hadamard_rejects_non_powers_of_four(size):
    with pytest.raises(cd.ComfyDequantError):
        cd.hadamard(size)


def test_hadamard_result_is_cached_and_read_only():
    assert cd.hadamard(256) is cd.hadamard(256)
    with pytest.raises(ValueError):
        cd.hadamard(256)[0, 0] = 0.0


# ==========================================================================
# 2. FP8 E4M3FN
# ==========================================================================


def test_fp8_table_matches_an_independent_implementation_for_all_256_bytes():
    for byte in range(256):
        got = float(cd.FP8_E4M3_TABLE[byte])
        want = _reference_fp8_e4m3(byte)
        if math.isnan(want):
            assert math.isnan(got), f"byte 0x{byte:02X} should decode to NaN"
        else:
            assert got == want, f"byte 0x{byte:02X}: got {got!r}, want {want!r}"
            # -0.0 must stay -0.0, not become +0.0
            assert math.copysign(1.0, got) == math.copysign(1.0, want)


def test_fp8_table_anchor_values():
    table = cd.FP8_E4M3_TABLE
    assert float(table[0x00]) == 0.0 and math.copysign(1.0, float(table[0x00])) == 1.0
    assert float(table[0x80]) == 0.0 and math.copysign(1.0, float(table[0x80])) == -1.0
    assert float(table[0x38]) == 1.0
    assert float(table[0x01]) == 2.0**-9
    assert float(table[0x07]) == 0.013671875
    assert float(table[0x7E]) == 448.0
    assert float(table[0xFE]) == -448.0
    assert math.isnan(float(table[0x7F]))
    assert math.isnan(float(table[0xFF]))


def test_fp8_table_has_no_infinities_and_exactly_two_nans():
    table = cd.FP8_E4M3_TABLE
    assert table.shape == (256,)
    assert table.dtype == np.float32
    assert int(np.isinf(table).sum()) == 0
    assert int(np.isnan(table).sum()) == 2
    assert float(np.nanmax(np.abs(table))) == 448.0


def test_decode_fp8_e4m3_preserves_shape_and_rejects_wrong_dtype():
    raw = np.array([[0x38, 0x00], [0x7E, 0xFE]], dtype=np.uint8)
    decoded = cd.decode_fp8_e4m3(raw)
    assert decoded.shape == (2, 2)
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, np.array([[1.0, 0.0], [448.0, -448.0]], dtype=np.float32))
    with pytest.raises(cd.ComfyDequantError):
        cd.decode_fp8_e4m3(raw.astype(np.int8))


# ==========================================================================
# 3. int8 round trip
# ==========================================================================


@pytest.mark.parametrize("convrot", [True, False])
def test_int8_tensorwise_round_trip(convrot):
    rng = np.random.default_rng(20260913)
    weights = rng.standard_normal((8, 512)).astype(np.float32)

    codes, scale = _forward_int8(weights, convrot=convrot)
    restored = cd.dequantize_layer(
        _int8_marker(convrot),
        {"weight": codes, "weight_scale": scale},
        512,
        layer_name="fixture.int8",
    )

    assert restored.dtype == np.float32
    assert restored.shape == (8, 512)
    assert restored.flags["C_CONTIGUOUS"]
    assert _rel_rmse(restored, weights) < 0.02
    assert _cosine(restored, weights) > 0.9999


def test_int8_scale_accepts_both_column_and_flat_layout():
    rng = np.random.default_rng(7)
    weights = rng.standard_normal((4, 256)).astype(np.float32)
    codes, scale = _forward_int8(weights, convrot=True)

    column = cd.dequantize_layer(
        _int8_marker(True), {"weight": codes, "weight_scale": scale}, 256
    )
    flat = cd.dequantize_layer(
        _int8_marker(True),
        {"weight": codes, "weight_scale": np.ascontiguousarray(scale.reshape(-1))},
        256,
    )
    assert np.array_equal(column, flat)


def test_int8_convrot_flag_actually_changes_the_result():
    rng = np.random.default_rng(11)
    weights = rng.standard_normal((4, 256)).astype(np.float32)
    codes, scale = _forward_int8(weights, convrot=True)
    tensors = {"weight": codes, "weight_scale": scale}
    with_rotation = cd.dequantize_layer(_int8_marker(True), tensors, 256)
    without_rotation = cd.dequantize_layer(_int8_marker(False), tensors, 256)
    assert not np.allclose(with_rotation, without_rotation)
    assert _cosine(without_rotation, weights) < 0.5


def test_int8_chunked_path_matches_a_single_chunk():
    """A layer wide enough to force several row chunks must give the same answer."""
    rng = np.random.default_rng(101)
    weights = rng.standard_normal((32, 256)).astype(np.float32)
    codes, scale = _forward_int8(weights, convrot=True)
    tensors = {"weight": codes, "weight_scale": scale}

    whole = cd.dequantize_layer(_int8_marker(True), tensors, 256)
    original = cd._CHUNK_BYTES
    try:
        cd._CHUNK_BYTES = 1  # forces one row per chunk
        chunked = cd.dequantize_layer(_int8_marker(True), tensors, 256)
    finally:
        cd._CHUNK_BYTES = original
    assert np.array_equal(whole, chunked)


# ==========================================================================
# 4./5. w4a8 round trip and the wrong-nibble control
# ==========================================================================


def test_asym_w4a8_int8_round_trip():
    rng = np.random.default_rng(20260914)
    weights = rng.standard_normal((8, 512)).astype(np.float32)

    packed, codebook, s_channel, s_rel = _forward_w4a8(weights, REAL_CODEBOOK)
    assert packed.shape == (8, 256)
    assert s_rel.shape == (8, 32)

    restored = cd.dequantize_layer(
        W4A8_MARKER,
        {
            "weight": packed,
            "weight_codebook": codebook,
            "weight_s_channel": s_channel,
            "weight_s_rel": s_rel,
        },
        512,
        layer_name="fixture.w4a8",
    )

    assert restored.dtype == np.float32
    assert restored.shape == (8, 512)
    assert restored.flags["C_CONTIGUOUS"]
    assert _cosine(restored, weights) > 0.99


def test_asym_w4a8_swapped_nibbles_destroys_the_signal():
    rng = np.random.default_rng(20260915)
    weights = rng.standard_normal((8, 512)).astype(np.float32)
    packed, codebook, s_channel, s_rel = _forward_w4a8(weights, REAL_CODEBOOK)

    good = cd.dequantize_layer(
        W4A8_MARKER,
        {
            "weight": packed,
            "weight_codebook": codebook,
            "weight_s_channel": s_channel,
            "weight_s_rel": s_rel,
        },
        512,
    )
    bad = _decode_w4a8_swapped_nibbles(packed, codebook, s_channel, s_rel, 512)

    assert _cosine(good, weights) > 0.99
    assert _cosine(bad, weights) < 0.3


def test_asym_w4a8_s_channel_accepts_both_layouts_and_chunking_is_invariant():
    rng = np.random.default_rng(3)
    weights = rng.standard_normal((16, 256)).astype(np.float32)
    packed, codebook, s_channel, s_rel = _forward_w4a8(weights, REAL_CODEBOOK)
    tensors = {
        "weight": packed,
        "weight_codebook": codebook,
        "weight_s_channel": s_channel,
        "weight_s_rel": s_rel,
    }

    flat = cd.dequantize_layer(W4A8_MARKER, tensors, 256)
    column = cd.dequantize_layer(
        W4A8_MARKER,
        {**tensors, "weight_s_channel": np.ascontiguousarray(s_channel.reshape(-1, 1))},
        256,
    )
    assert np.array_equal(flat, column)

    original = cd._CHUNK_BYTES
    try:
        cd._CHUNK_BYTES = 1
        chunked = cd.dequantize_layer(W4A8_MARKER, tensors, 256)
    finally:
        cd._CHUNK_BYTES = original
    assert np.array_equal(flat, chunked)


# ==========================================================================
# 6./7. marker parsing
# ==========================================================================


def test_parse_marker_int8_compact_and_spaced_forms():
    compact = b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'
    spaced = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
    expected = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
        "group_size": None,
    }
    assert cd.parse_quant_marker(compact) == expected
    assert cd.parse_quant_marker(spaced) == expected
    # The two spellings are exactly the 67- and 72-byte U8 payload lengths the
    # real community checkpoint records for its comfy_quant markers.
    assert (len(compact), len(spaced)) == (67, 72)


def test_parse_marker_int8_defaults_convrot_to_false():
    parsed = cd.parse_quant_marker(b'{"format":"int8_tensorwise"}')
    assert parsed == {
        "format": "int8_tensorwise",
        "convrot": False,
        "convrot_groupsize": 256,
        "group_size": None,
    }


def test_parse_marker_w4a8_defaults_and_normalisation():
    parsed = cd.parse_quant_marker(b'{"format":"asym_w4a8_int8"}')
    assert parsed == {
        "format": "asym_w4a8_int8",
        "convrot": True,
        "convrot_groupsize": 256,
        "group_size": 16,
    }
    explicit = cd.parse_quant_marker(
        b'{"format":"asym_w4a8_int8","group_size":16,"convrot":true,'
        b'"convrot_groupsize":256,"full_precision_matrix_mult":false}'
    )
    assert explicit == parsed


def test_parse_marker_accepts_the_nested_params_form():
    nested = json.dumps(
        {"format": "int8_tensorwise", "params": {"convrot": True, "convrot_groupsize": 256}}
    ).encode("utf-8")
    assert cd.parse_quant_marker(nested) == {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
        "group_size": None,
    }
    nested_w4a8 = json.dumps(
        {"format": "asym_w4a8_int8", "params": {"group_size": 16, "convrot_groupsize": 256}}
    ).encode("utf-8")
    assert cd.parse_quant_marker(nested_w4a8)["group_size"] == 16


def test_parse_marker_accepts_uint8_arrays_and_trailing_nul_padding():
    raw = b'{"format":"int8_tensorwise","convrot":true}'
    as_array = np.frombuffer(raw, dtype=np.uint8)
    assert cd.parse_quant_marker(as_array) == cd.parse_quant_marker(raw)
    assert cd.parse_quant_marker(raw + b"\x00\x00") == cd.parse_quant_marker(raw)


def test_parse_marker_rejects_unknown_format():
    with pytest.raises(cd.UnknownQuantFormatError) as excinfo:
        cd.parse_quant_marker(b'{"format":"nvfp4"}')
    assert "nvfp4" in str(excinfo.value)
    assert issubclass(cd.UnknownQuantFormatError, cd.ComfyDequantError)
    assert issubclass(cd.ComfyDequantError, ValueError)


@pytest.mark.parametrize(
    "marker",
    [
        b'{"format":"int8_tensorwise","scheme":"symmetric"}',  # unknown top-level key
        b'{"format":"int8_tensorwise","convrot_groupsize":128}',  # wrong ConvRot group
        b'{"format":"asym_w4a8_int8","group_size":32}',  # wrong w4a8 group
        b'{"format":"asym_w4a8_int8","convrot":false}',  # ConvRot is unconditional here
        b'{"format":"asym_w4a8_int8","params":{"nope":1}}',  # unknown nested key
        b'{"format":"int8_tensorwise","convrot":"yes"}',  # wrong type
        b'{"format":"int8_tensorwise","convrot_groupsize":"256"}',  # wrong type
        b'{"format":"int8_tensorwise","full_precision_matrix_mult":1}',  # wrong type
        b'{"convrot":true}',  # no format at all
        b'{"format":"int8_tensorwise","convrot":true,"params":{"convrot":false}}',  # self-contradiction
    ],
)
def test_parse_marker_rejects_unknown_or_ill_typed_keys(marker):
    with pytest.raises(cd.UnknownQuantFormatError):
        cd.parse_quant_marker(marker)


@pytest.mark.parametrize(
    "marker",
    [
        b"",
        b"not json at all",
        b'["int8_tensorwise"]',
        b"\xff\xfe{}",
        b'{"format":"int8_tensorwise","format":"asym_w4a8_int8"}',
    ],
)
def test_parse_marker_rejects_malformed_payloads(marker):
    with pytest.raises(cd.ComfyDequantError):
        cd.parse_quant_marker(marker)


# ==========================================================================
# 8. bf16 rounding parity with converter.convert
# ==========================================================================


def test_f32_to_bf16_u16_matches_get_bf16_bytes(tmp_path):
    rng = np.random.default_rng(20260916)
    values = np.concatenate(
        [
            rng.standard_normal(4096).astype(np.float32) * 1e-3,
            rng.standard_normal(4096).astype(np.float32) * 1e3,
            # ties and near-ties: exactly the cases round-half-to-even decides
            np.array(
                [0.0, -0.0, 1.0, -1.0, 1.5, -1.5, 2.5, -2.5, 65504.0, 1e-40, -1e-40],
                dtype=np.float32,
            ),
            (np.arange(1 << 12, dtype=np.uint32) << np.uint32(4)).view(np.float32)[:2048],
        ]
    ).astype(np.float32)
    values = values[np.isfinite(values)]

    path = tmp_path / "bf16_parity.safetensors"
    _write_safetensors(path, [("t", "F32", values.tobytes(), [values.size])])
    with _SafetensorsRaw(path) as raw:
        reference = raw.get_bf16_bytes("t")

    assert np.array_equal(cd.f32_to_bf16_u16(values), reference)


def test_f32_to_bf16_u16_preserves_shape():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = cd.f32_to_bf16_u16(values)
    assert out.shape == (3, 4)
    assert out.dtype == np.uint16


# ==========================================================================
# dequantize_layer: fail-closed behaviour
# ==========================================================================


def _valid_w4a8_tensors(rows=4, in_features=256):
    rng = np.random.default_rng(5)
    weights = rng.standard_normal((rows, in_features)).astype(np.float32)
    packed, codebook, s_channel, s_rel = _forward_w4a8(weights, REAL_CODEBOOK)
    return {
        "weight": packed,
        "weight_codebook": codebook,
        "weight_s_channel": s_channel,
        "weight_s_rel": s_rel,
    }


def test_dequantize_layer_rejects_unknown_format():
    with pytest.raises(cd.UnknownQuantFormatError):
        cd.dequantize_layer({"format": "nvfp4"}, {}, 256)


def test_dequantize_layer_rejects_wrong_key_set():
    tensors = _valid_w4a8_tensors()
    with pytest.raises(cd.ComfyDequantError) as excinfo:
        cd.dequantize_layer(
            W4A8_MARKER, {k: v for k, v in tensors.items() if k != "weight_s_rel"}, 256
        )
    assert "weight_s_rel" in str(excinfo.value)

    with pytest.raises(cd.ComfyDequantError):
        cd.dequantize_layer(W4A8_MARKER, {**tensors, "weight_correction": tensors["weight"]}, 256)


def test_dequantize_layer_rejects_missing_int8_scale():
    codes = np.zeros((4, 256), dtype=np.int8)
    with pytest.raises(cd.ComfyDequantError):
        cd.dequantize_layer(_int8_marker(True), {"weight": codes}, 256)


@pytest.mark.parametrize(
    "key,replacement",
    [
        ("weight_s_rel", np.zeros((4, 32), dtype=np.uint8)),  # [N, K/8], not [N, K/16]
        ("weight_s_rel", np.zeros((4, 8), dtype=np.uint8)),  # [N, K/32]: too few groups
        ("weight_s_channel", np.zeros((8,), dtype=np.float32)),  # wrong row count
        ("weight_codebook", np.zeros((8,), dtype=np.float32)),  # wrong codebook length
    ],
)
def test_dequantize_layer_rejects_bad_sidecar_shapes(key, replacement):
    tensors = {**_valid_w4a8_tensors(), key: replacement}
    with pytest.raises(cd.ComfyDequantError) as excinfo:
        cd.dequantize_layer(W4A8_MARKER, tensors, 256, layer_name="fixture.bad")
    assert "fixture.bad" in str(excinfo.value)


@pytest.mark.parametrize(
    "key,dtype",
    [
        ("weight", np.uint8),
        ("weight_codebook", np.float64),
        ("weight_s_channel", np.float64),
        ("weight_s_rel", np.int8),
    ],
)
def test_dequantize_layer_rejects_wrong_sidecar_dtypes(key, dtype):
    tensors = _valid_w4a8_tensors()
    tensors[key] = tensors[key].astype(dtype)
    with pytest.raises(cd.ComfyDequantError):
        cd.dequantize_layer(W4A8_MARKER, tensors, 256)


def test_dequantize_layer_rejects_wrong_in_features():
    tensors = _valid_w4a8_tensors()
    with pytest.raises(cd.ComfyDequantError):
        cd.dequantize_layer(W4A8_MARKER, tensors, 512)
    codes = np.zeros((4, 256), dtype=np.int8)
    scale = np.ones((4, 1), dtype=np.float32)
    with pytest.raises(cd.ComfyDequantError):
        cd.dequantize_layer(_int8_marker(True), {"weight": codes, "weight_scale": scale}, 128)


def test_dequantize_layer_rejects_in_features_not_divisible_by_the_convrot_group():
    codes = np.zeros((2, 128), dtype=np.int8)
    scale = np.ones((2, 1), dtype=np.float32)
    with pytest.raises(cd.ComfyDequantError) as excinfo:
        cd.dequantize_layer(
            _int8_marker(True), {"weight": codes, "weight_scale": scale}, 128, layer_name="narrow"
        )
    assert "narrow" in str(excinfo.value)


def test_dequantize_layer_rejects_fp8_nan_in_s_rel():
    tensors = _valid_w4a8_tensors()
    s_rel = tensors["weight_s_rel"].copy()
    s_rel[1, 2] = 0x7F  # the E4M3FN NaN encoding
    tensors["weight_s_rel"] = s_rel
    with pytest.raises(cd.ComfyDequantError) as excinfo:
        cd.dequantize_layer(W4A8_MARKER, tensors, 256, layer_name="fixture.nan")
    message = str(excinfo.value)
    assert "fixture.nan" in message
    assert "weight_s_rel" in message

    s_rel[1, 2] = 0xFF  # the negative NaN encoding
    tensors["weight_s_rel"] = s_rel
    with pytest.raises(cd.ComfyDequantError):
        cd.dequantize_layer(W4A8_MARKER, tensors, 256, layer_name="fixture.nan")


def test_dequantize_layer_rejects_non_finite_result():
    codes = np.full((2, 256), 100, dtype=np.int8)
    scale = np.array([[np.inf], [1.0]], dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        with pytest.raises(cd.ComfyDequantError) as excinfo:
            cd.dequantize_layer(
                _int8_marker(True),
                {"weight": codes, "weight_scale": scale},
                256,
                layer_name="fixture.inf",
            )
    assert "fixture.inf" in str(excinfo.value)


def test_dequantize_layer_rejects_non_array_inputs():
    with pytest.raises(cd.ComfyDequantError):
        cd.dequantize_layer(
            _int8_marker(False),
            {"weight": [[0, 1]], "weight_scale": np.ones((1, 1), dtype=np.float32)},
            2,
        )
    with pytest.raises(cd.ComfyDequantError):
        cd.dequantize_layer("int8_tensorwise", {}, 256)


def test_supported_formats_contract():
    assert cd.SUPPORTED_FORMATS == ("int8_tensorwise", "asym_w4a8_int8")
