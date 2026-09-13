"""Dequantisation of ComfyUI-style quantised Linear weights (NumPy only).

Community LTX 2.5 checkpoints produced by the ``ltx25-quant-lab`` tooling store
their Linear weights in one of two ComfyUI quantisation formats, each described
by a small UTF-8 JSON marker held in a ``<layer>.comfy_quant`` U8 tensor:

``int8_tensorwise``
    ``weight`` is ``I8 [out, in]`` and ``weight_scale`` is ``F32 [out, 1]``
    (a per-output-row scale).  When ``convrot`` is set the stored weight is
    additionally rotated by a normalised regular Hadamard matrix applied to
    contiguous groups of ``convrot_groupsize`` input channels.

``asym_w4a8_int8``
    ``weight`` is ``I8 [out, in/2]`` holding two 4-bit codes per byte,
    ``weight_codebook`` is ``F32 [16]``, ``weight_s_rel`` is ``F8_E4M3
    [out, in/16]`` (a per-group relative scale) and ``weight_s_channel`` is
    ``F32 [out]``.  ConvRot is always on for this format.

This module implements the *inverse* of both, in float32, with NumPy only.
The formats themselves are documented by the upstream ``comfy-kitchen``
project (Apache-2.0); no code was copied from it -- the maths below was
re-implemented from the format description and cross-checked numerically
against the official bf16 weights (see ``scripts/compare_comfyquant_vs_official.py``).

Design notes
------------
* The module knows nothing about safetensors, GGUF, type maps or the CLI.  It
  takes plain NumPy arrays and returns a plain NumPy array.
* It is *fail-closed*: an unknown format, an unknown marker key, an unexpected
  group size, a wrong dtype/shape or a non-finite result is an error, never a
  silently-ignored input.
* Memory is bounded.  The biggest Linear in this architecture is
  ``[16384, 4096]``, i.e. 256 MiB as float32.  The nibble expansion, codebook
  lookup, scale multiply and the inverse rotation are all done in the *same*
  row-chunk loop, writing straight into one pre-allocated float32 output array,
  so the transient working set stays in the tens of MiB rather than exceeding
  a gigabyte.
"""

from __future__ import annotations

import functools
import json
import math
from typing import Any

import numpy as np


#: Quantisation formats this module can invert.
SUPPORTED_FORMATS: tuple[str, ...] = ("int8_tensorwise", "asym_w4a8_int8")

#: The only ConvRot group size the community checkpoints use (and the only one
#: this module accepts -- a different value would silently change the maths).
CONVROT_GROUPSIZE = 256

#: The only ``asym_w4a8_int8`` group size this module accepts.
W4A8_GROUP_SIZE = 16

#: Number of entries in the ``asym_w4a8_int8`` codebook (4-bit codes).
CODEBOOK_SIZE = 16

#: Rough working-set budget for one row chunk, in bytes.  See module docstring.
_CHUNK_BYTES = 32 * 1024 * 1024

#: Bytes of transient working set per row, per input feature (empirical upper
#: bound over both code paths: nibbles + codes + float32 values + FP8 scales +
#: the matmul's own temporaries).
_CHUNK_BYTES_PER_ELEMENT = 12


class ComfyDequantError(ValueError):
    """A ComfyUI-quantised tensor could not be inverted as specified."""


class UnknownQuantFormatError(ComfyDequantError):
    """A ``comfy_quant`` marker names a format/parameter this module refuses."""


# --------------------------------------------------------------------------
# marker parsing
# --------------------------------------------------------------------------

# Keys each format's marker JSON is allowed to carry at the top level.  The
# ``params`` key is the older nesting ComfyUI also emits.
_MARKER_KEYS: dict[str, frozenset[str]] = {
    "int8_tensorwise": frozenset(
        {"format", "convrot", "convrot_groupsize", "full_precision_matrix_mult", "params"}
    ),
    "asym_w4a8_int8": frozenset(
        {
            "format",
            "group_size",
            "convrot",
            "convrot_groupsize",
            "full_precision_matrix_mult",
            "params",
        }
    ),
}

# Keys allowed inside the nested ``params`` object.
_PARAMS_KEYS: dict[str, frozenset[str]] = {
    "int8_tensorwise": frozenset({"convrot", "convrot_groupsize"}),
    "asym_w4a8_int8": frozenset({"convrot", "convrot_groupsize", "group_size"}),
}


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UnknownQuantFormatError(f"comfy_quant marker has duplicate JSON key {key!r}")
        result[key] = value
    return result


def _marker_bytes(raw: Any) -> bytes:
    """Coerce the marker payload to ``bytes``.

    Accepts ``bytes``/``bytearray``/``memoryview`` as well as the ``uint8``/
    ``int8`` NumPy array a safetensors ``U8`` tensor naturally reads back as.
    """
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    if isinstance(raw, np.ndarray):
        if raw.dtype not in (np.uint8, np.int8):
            raise ComfyDequantError(
                f"comfy_quant marker array must be uint8/int8 (got {raw.dtype})"
            )
        return raw.tobytes()
    raise ComfyDequantError(
        f"comfy_quant marker must be bytes or a uint8 array (got {type(raw).__name__})"
    )


def parse_quant_marker(raw: bytes) -> dict:
    """Parse and normalise a ``comfy_quant`` marker payload.

    ``raw`` is the raw UTF-8 JSON payload of the ``<layer>.comfy_quant`` U8
    tensor.  Trailing NUL/whitespace padding is tolerated; anything else is not.

    Returns a normalised dict with exactly these keys::

        {"format": str, "convrot": bool, "convrot_groupsize": int,
         "group_size": int | None}

    ``group_size`` is ``None`` for ``int8_tensorwise`` (the format has no
    sub-channel grouping) and ``16`` for ``asym_w4a8_int8``.  ``convrot`` is
    normalised to ``True`` for ``asym_w4a8_int8`` (ConvRot is unconditional
    there) and defaults to ``False`` for ``int8_tensorwise`` when the marker
    does not say.

    Raises :class:`UnknownQuantFormatError` for an unknown format, an unknown
    key, a ConvRot group size other than 256 or a w4a8 group size other than 16
    -- none of those are quietly ignored, because each of them would change the
    dequantisation maths.
    """
    data = _marker_bytes(raw)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ComfyDequantError(f"comfy_quant marker is not valid UTF-8 ({exc})") from exc
    text = text.rstrip("\x00").strip()
    if not text:
        raise ComfyDequantError("comfy_quant marker is empty")
    try:
        payload = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ComfyDequantError(
            f"comfy_quant marker is not valid JSON ({exc}): {text!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise ComfyDequantError(
            f"comfy_quant marker must be a JSON object (got {type(payload).__name__})"
        )

    fmt = payload.get("format")
    if not isinstance(fmt, str):
        raise UnknownQuantFormatError(
            f"comfy_quant marker has no string 'format' key: {text!r}"
        )
    if fmt not in SUPPORTED_FORMATS:
        raise UnknownQuantFormatError(
            f"unsupported comfy_quant format {fmt!r} "
            f"(supported: {', '.join(SUPPORTED_FORMATS)})"
        )

    unknown = sorted(set(payload) - _MARKER_KEYS[fmt])
    if unknown:
        raise UnknownQuantFormatError(
            f"comfy_quant marker for format {fmt!r} has unknown keys {unknown}"
        )

    flat: dict[str, Any] = {key: value for key, value in payload.items() if key != "params"}
    params = payload.get("params")
    if params is not None:
        if not isinstance(params, dict):
            raise UnknownQuantFormatError(
                f"comfy_quant marker 'params' must be a JSON object "
                f"(got {type(params).__name__})"
            )
        nested_unknown = sorted(set(params) - _PARAMS_KEYS[fmt])
        if nested_unknown:
            raise UnknownQuantFormatError(
                f"comfy_quant marker for format {fmt!r} has unknown 'params' keys "
                f"{nested_unknown}"
            )
        for key, value in params.items():
            if key in flat and flat[key] != value:
                raise UnknownQuantFormatError(
                    f"comfy_quant marker for format {fmt!r} disagrees with itself on "
                    f"{key!r}: top-level {flat[key]!r} vs params {value!r}"
                )
            flat[key] = value

    if "full_precision_matrix_mult" in flat and not isinstance(
        flat["full_precision_matrix_mult"], bool
    ):
        raise UnknownQuantFormatError(
            "comfy_quant marker 'full_precision_matrix_mult' must be a boolean "
            f"(got {flat['full_precision_matrix_mult']!r})"
        )

    groupsize = flat.get("convrot_groupsize", CONVROT_GROUPSIZE)
    if not _is_plain_int(groupsize):
        raise UnknownQuantFormatError(
            f"comfy_quant marker 'convrot_groupsize' must be an integer (got {groupsize!r})"
        )
    if groupsize != CONVROT_GROUPSIZE:
        raise UnknownQuantFormatError(
            f"comfy_quant marker 'convrot_groupsize' is {groupsize}; only "
            f"{CONVROT_GROUPSIZE} is supported"
        )

    convrot_raw = flat.get("convrot")
    if convrot_raw is not None and not isinstance(convrot_raw, bool):
        raise UnknownQuantFormatError(
            f"comfy_quant marker 'convrot' must be a boolean (got {convrot_raw!r})"
        )

    if fmt == "int8_tensorwise":
        convrot = bool(convrot_raw) if convrot_raw is not None else False
        group_size: int | None = None
    else:
        if convrot_raw is not None and convrot_raw is not True:
            raise UnknownQuantFormatError(
                "comfy_quant marker for format 'asym_w4a8_int8' sets convrot=false; "
                "ConvRot is unconditional for this format"
            )
        convrot = True
        group_size = flat.get("group_size", W4A8_GROUP_SIZE)
        if not _is_plain_int(group_size):
            raise UnknownQuantFormatError(
                f"comfy_quant marker 'group_size' must be an integer (got {group_size!r})"
            )
        if group_size != W4A8_GROUP_SIZE:
            raise UnknownQuantFormatError(
                f"comfy_quant marker 'group_size' is {group_size}; only "
                f"{W4A8_GROUP_SIZE} is supported"
            )

    return {
        "format": fmt,
        "convrot": convrot,
        "convrot_groupsize": int(groupsize),
        "group_size": group_size,
    }


# --------------------------------------------------------------------------
# ConvRot (normalised regular Hadamard)
# --------------------------------------------------------------------------

# The 4x4 regular Hadamard seed the ComfyUI ConvRot is built from.  Note this
# is *not* the Sylvester H4: it is symmetric with a -1 on the anti-diagonal,
# which makes every Kronecker power symmetric, orthogonal and involutory.
_H4 = np.array(
    [
        [1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


@functools.lru_cache(maxsize=None)
def hadamard(size: int) -> np.ndarray:
    """Return the normalised regular Hadamard matrix of ``size`` x ``size``.

    ``size`` must be a power of four (4, 16, 64, 256, ...).  The result is the
    Kronecker power of :data:`_H4` divided by ``sqrt(size)``; since ``size`` is
    a power of four the divisor is a power of two and the division is exact in
    float32.  Every element is therefore exactly ``+/- 1/sqrt(size)``.

    The matrix is symmetric, orthogonal and involutory (``H @ H == I``), so the
    forward rotation and its inverse are the *same* operation.

    The returned array is cached and read-only; copy it before mutating.
    """
    if not _is_plain_int(size) or size < 4:
        raise ComfyDequantError(
            f"hadamard size must be a power of four that is at least 4 (got {size!r})"
        )
    remainder = size
    while remainder % 4 == 0:
        remainder //= 4
    if remainder != 1:
        raise ComfyDequantError(f"hadamard size {size} is not a power of four")

    matrix = _H4
    while matrix.shape[0] < size:
        matrix = np.kron(matrix, _H4)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32) / np.float32(math.sqrt(size))
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    matrix.setflags(write=False)
    return matrix


# --------------------------------------------------------------------------
# FP8 E4M3 (the "fn" flavour: no infinities, NaN only at 0x7F/0xFF)
# --------------------------------------------------------------------------


def _build_fp8_e4m3_table() -> np.ndarray:
    """Build the 256-entry E4M3FN decode table from the format definition.

    Layout: 1 sign bit, 4 exponent bits, 3 mantissa bits, exponent bias 7.
    ``exponent == 0`` is subnormal (``2**-6 * mantissa/8``); the all-ones
    exponent is *not* reserved for infinities -- only the single pattern
    ``exponent == 15 and mantissa == 7`` is NaN, which makes ``0x7E`` the
    largest finite value, 448.
    """
    table = np.empty(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0x0F
        mantissa = byte & 0x07
        if exponent == 0x0F and mantissa == 0x07:
            table[byte] = math.nan
            continue
        if exponent == 0:
            magnitude = (2.0**-6) * (mantissa / 8.0)
        else:
            magnitude = (2.0 ** (exponent - 7)) * (1.0 + mantissa / 8.0)
        table[byte] = math.copysign(magnitude, sign)
    table.setflags(write=False)
    return table


#: 256-entry FP8 E4M3FN decode table (float32).  Read-only.
FP8_E4M3_TABLE: np.ndarray = _build_fp8_e4m3_table()


def decode_fp8_e4m3(raw_u8: np.ndarray) -> np.ndarray:
    """Decode raw FP8 E4M3FN bytes to float32, preserving shape."""
    arr = np.asarray(raw_u8)
    if arr.dtype != np.uint8:
        raise ComfyDequantError(
            f"decode_fp8_e4m3 expects a uint8 array of raw bytes (got dtype {arr.dtype})"
        )
    return FP8_E4M3_TABLE[arr]


# --------------------------------------------------------------------------
# bf16 rounding (must match converter.convert._SafetensorsRaw.get_bf16_bytes)
# --------------------------------------------------------------------------


def f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    """Round float32 values to bf16 and return the bit patterns as uint16.

    Round-half-to-even, bit-identical to
    ``converter.convert._SafetensorsRaw.get_bf16_bytes``'s F32 branch, so a
    dequantised tensor emitted as bf16 rounds exactly the way the rest of the
    converter rounds.
    """
    values = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = values.view(np.uint32)
    bias = ((u32 >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u32 + bias) >> np.uint32(16)).astype(np.uint16)


# --------------------------------------------------------------------------
# dequantisation
# --------------------------------------------------------------------------

_REQUIRED_TENSORS: dict[str, frozenset[str]] = {
    "int8_tensorwise": frozenset({"weight", "weight_scale"}),
    "asym_w4a8_int8": frozenset(
        {"weight", "weight_codebook", "weight_s_channel", "weight_s_rel"}
    ),
}


def _where(layer_name: str) -> str:
    return f" for layer {layer_name!r}" if layer_name else ""


def _as_array(value: Any, dtype: Any, key: str, layer_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ComfyDequantError(
            f"tensor {key!r}{_where(layer_name)} must be a numpy array "
            f"(got {type(value).__name__})"
        )
    if value.dtype != dtype:
        raise ComfyDequantError(
            f"tensor {key!r}{_where(layer_name)} must have dtype "
            f"{np.dtype(dtype).name} (got {value.dtype})"
        )
    return value


def _per_row_vector(arr: np.ndarray, rows: int, key: str, layer_name: str) -> np.ndarray:
    """Accept ``[rows]`` or ``[rows, 1]`` and return a contiguous ``[rows]``."""
    if arr.shape == (rows,):
        flat = arr
    elif arr.shape == (rows, 1):
        flat = arr.reshape(rows)
    else:
        raise ComfyDequantError(
            f"tensor {key!r}{_where(layer_name)} has shape {list(arr.shape)}, "
            f"expected [{rows}] or [{rows}, 1]"
        )
    return np.ascontiguousarray(flat, dtype=np.float32)


def _row_chunks(rows: int, in_features: int):
    per_row = max(1, in_features * _CHUNK_BYTES_PER_ELEMENT)
    step = max(1, _CHUNK_BYTES // per_row)
    for start in range(0, rows, step):
        yield start, min(start + step, rows)


def _check_finite(block: np.ndarray, layer_name: str, first_row: int) -> None:
    if not np.isfinite(block).all():
        raise ComfyDequantError(
            f"dequantised weights{_where(layer_name)} contain non-finite values "
            f"(first seen in the row chunk starting at row {first_row})"
        )


def _rotation_matrix(in_features: int, groupsize: int, layer_name: str) -> np.ndarray:
    if in_features % groupsize:
        raise ComfyDequantError(
            f"in_features {in_features}{_where(layer_name)} is not a multiple of the "
            f"ConvRot group size {groupsize}"
        )
    return hadamard(groupsize)


def _write_rows(
    out: np.ndarray,
    start: int,
    stop: int,
    block: np.ndarray,
    matrix: np.ndarray | None,
    groupsize: int,
) -> None:
    """Write ``block`` into ``out[start:stop]``, applying the inverse ConvRot.

    ``out`` is freshly allocated and C-contiguous, so ``out[start:stop]`` is a
    contiguous view and its ``reshape`` is a view too -- ``np.matmul(..., out=)``
    therefore writes straight into the output array with no extra full-size
    temporary.
    """
    rows = stop - start
    if matrix is None:
        out[start:stop] = block
        return
    groups = block.shape[1] // groupsize
    np.matmul(
        block.reshape(rows, groups, groupsize),
        matrix.T,
        out=out[start:stop].reshape(rows, groups, groupsize),
    )


def _dequantize_int8_tensorwise(
    marker: dict, tensors: dict, in_features: int, layer_name: str
) -> np.ndarray:
    weight = _as_array(tensors["weight"], np.int8, "weight", layer_name)
    if weight.ndim != 2 or weight.shape[1] != in_features:
        raise ComfyDequantError(
            f"tensor 'weight'{_where(layer_name)} has shape {list(weight.shape)}, "
            f"expected [out, {in_features}] for format 'int8_tensorwise'"
        )
    rows = int(weight.shape[0])
    scale = _per_row_vector(
        _as_array(tensors["weight_scale"], np.float32, "weight_scale", layer_name),
        rows,
        "weight_scale",
        layer_name,
    )

    groupsize = int(marker.get("convrot_groupsize", CONVROT_GROUPSIZE))
    matrix = (
        _rotation_matrix(in_features, groupsize, layer_name)
        if bool(marker.get("convrot", False))
        else None
    )

    out = np.empty((rows, in_features), dtype=np.float32)
    for start, stop in _row_chunks(rows, in_features):
        block = weight[start:stop].astype(np.float32)
        block *= scale[start:stop, None]
        _write_rows(out, start, stop, block, matrix, groupsize)
        _check_finite(out[start:stop], layer_name, start)
    return out


def _dequantize_asym_w4a8_int8(
    marker: dict, tensors: dict, in_features: int, layer_name: str
) -> np.ndarray:
    group_size = marker.get("group_size", W4A8_GROUP_SIZE)
    if group_size is None:
        group_size = W4A8_GROUP_SIZE
    if not _is_plain_int(group_size) or group_size != W4A8_GROUP_SIZE:
        raise ComfyDequantError(
            f"format 'asym_w4a8_int8'{_where(layer_name)} requires group_size "
            f"{W4A8_GROUP_SIZE} (got {group_size!r})"
        )
    if in_features % group_size:
        raise ComfyDequantError(
            f"in_features {in_features}{_where(layer_name)} is not a multiple of the "
            f"w4a8 group size {group_size}"
        )

    weight = _as_array(tensors["weight"], np.int8, "weight", layer_name)
    if weight.ndim != 2 or weight.shape[1] * 2 != in_features:
        raise ComfyDequantError(
            f"tensor 'weight'{_where(layer_name)} has shape {list(weight.shape)}, "
            f"expected [out, {in_features // 2}] (two 4-bit codes per byte) for "
            f"format 'asym_w4a8_int8'"
        )
    rows = int(weight.shape[0])
    # The nibble split below takes a uint8 view of a row slice, which requires a
    # contiguous buffer (a no-op for anything read straight out of safetensors).
    weight = np.ascontiguousarray(weight)

    codebook = _as_array(
        tensors["weight_codebook"], np.float32, "weight_codebook", layer_name
    )
    if codebook.shape != (CODEBOOK_SIZE,):
        raise ComfyDequantError(
            f"tensor 'weight_codebook'{_where(layer_name)} has shape "
            f"{list(codebook.shape)}, expected [{CODEBOOK_SIZE}]"
        )
    codebook = np.ascontiguousarray(codebook, dtype=np.float32)

    s_channel = _per_row_vector(
        _as_array(tensors["weight_s_channel"], np.float32, "weight_s_channel", layer_name),
        rows,
        "weight_s_channel",
        layer_name,
    )

    s_rel = _as_array(tensors["weight_s_rel"], np.uint8, "weight_s_rel", layer_name)
    groups = in_features // group_size
    if s_rel.shape != (rows, groups):
        raise ComfyDequantError(
            f"tensor 'weight_s_rel'{_where(layer_name)} has shape {list(s_rel.shape)}, "
            f"expected [{rows}, {groups}] (one FP8 E4M3 scale per group of "
            f"{group_size} input channels)"
        )

    groupsize = int(marker.get("convrot_groupsize", CONVROT_GROUPSIZE))
    # ConvRot is unconditional for this format, so the rotation is not optional
    # here: parse_quant_marker refuses a marker that claims otherwise.
    matrix = _rotation_matrix(in_features, groupsize, layer_name)

    out = np.empty((rows, in_features), dtype=np.float32)
    for start, stop in _row_chunks(rows, in_features):
        rows_here = stop - start
        packed = weight[start:stop].view(np.uint8)
        codes = np.empty((rows_here, in_features), dtype=np.uint8)
        # Low nibble first: byte i holds element 2i in bits 0-3 and element
        # 2i+1 in bits 4-7.
        np.bitwise_and(packed, np.uint8(0x0F), out=codes[:, 0::2])
        np.right_shift(packed, np.uint8(4), out=codes[:, 1::2])

        block = codebook[codes]

        scales = decode_fp8_e4m3(s_rel[start:stop])
        if not np.isfinite(scales).all():
            raise ComfyDequantError(
                f"tensor 'weight_s_rel'{_where(layer_name)} decodes to a non-finite "
                f"FP8 E4M3 value (NaN byte 0x7F/0xFF) in the row chunk starting at "
                f"row {start}"
            )

        grouped = block.reshape(rows_here, groups, group_size)
        grouped *= scales[:, :, None]
        # The format defines the intermediate as an int8 grid; round to it.
        np.rint(grouped, out=grouped)
        np.clip(grouped, -127.0, 127.0, out=grouped)

        block *= s_channel[start:stop, None]
        _write_rows(out, start, stop, block, matrix, groupsize)
        _check_finite(out[start:stop], layer_name, start)
    return out


def dequantize_layer(
    marker: dict,
    tensors: dict[str, np.ndarray],
    in_features: int,
    *,
    layer_name: str = "",
) -> np.ndarray:
    """Invert one ComfyUI-quantised Linear weight.

    ``marker`` is a normalised marker dict as returned by
    :func:`parse_quant_marker`.  ``tensors`` holds the layer's stored arrays
    under their safetensors suffixes (``weight``, ``weight_scale``,
    ``weight_codebook``, ``weight_s_channel``, ``weight_s_rel``) -- the key set
    must match the format's requirement exactly.  ``in_features`` is the
    *logical* input width (``weight.shape[1]`` for int8, twice that for w4a8).

    Returns a C-contiguous float32 array of shape ``[out, in_features]``.
    Raises :class:`ComfyDequantError` on any shape/dtype/key mismatch or if the
    result contains a non-finite value.
    """
    if not isinstance(marker, dict):
        raise ComfyDequantError(
            f"marker must be a dict{_where(layer_name)} (got {type(marker).__name__})"
        )
    fmt = marker.get("format")
    if fmt not in SUPPORTED_FORMATS:
        raise UnknownQuantFormatError(
            f"unsupported comfy_quant format {fmt!r}{_where(layer_name)} "
            f"(supported: {', '.join(SUPPORTED_FORMATS)})"
        )
    if not isinstance(tensors, dict):
        raise ComfyDequantError(
            f"tensors must be a dict{_where(layer_name)} (got {type(tensors).__name__})"
        )
    if not _is_plain_int(in_features) or in_features <= 0:
        raise ComfyDequantError(
            f"in_features must be a positive integer{_where(layer_name)} "
            f"(got {in_features!r})"
        )

    expected = _REQUIRED_TENSORS[fmt]
    present = set(tensors)
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ComfyDequantError(
            f"format {fmt!r}{_where(layer_name)} needs exactly "
            f"{sorted(expected)}; missing={missing} unexpected={extra}"
        )

    if fmt == "int8_tensorwise":
        return _dequantize_int8_tensorwise(marker, tensors, in_features, layer_name)
    return _dequantize_asym_w4a8_int8(marker, tensors, in_features, layer_name)
