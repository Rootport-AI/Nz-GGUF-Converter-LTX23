"""Community ComfyUI-quantized LTX 2.5 weights -> GGUF (``ltx25-comfyquant``).

A fourth opt-in profile alongside ``ltx23``, ``ltx25`` and ``gemma4-ltx25``.  It
does *not* extend the ``ltx25`` profile: that one is a fail-closed lock onto the
single authenticated official bf16 artifact and stays byte-for-byte unchanged.
This module admits a *community* redistribution of the same architecture whose
linear weights were pre-quantized by ComfyUI-style tooling
(``int8_tensorwise`` ConvRot and ``asym_w4a8_int8``), dequantizes them and
re-quantizes to the GGUF types the backend's fused kernels support.

Because the file is community-built there is no file-identity lock.  Admission
is *structural* instead (see ``Docs/LTX25_CONVERSION_MODE_SPEC.md``): the config
JSON digest must equal the official builder oracle's, every raw key must carry
the official ``model.diffusion_model.`` prefix, the folded logical tensor set
must match the official 4,349-key/shape oracle exactly, and every quantization
marker must name a format on the supported allowlist.  ``--expect-sha256``
remains available as an optional identity pin.

Stage layout mirrors ``ltx25.py``::

    inspect -> convert -> self-verify

There is no ``build-map`` stage: the *official* approved conversion map is the
type policy, read name/shape-only, with its ``Q4_K`` rows rewritten to the
selected ``--quant-type``.  The map file itself is never modified.

Header/digest/atomic-write primitives are reused from :mod:`converter.ltx25`
(the same "separate module, reuse the private helpers" arrangement
:mod:`converter.ltx25_gemma` uses); the numeric dequantization lives in
:mod:`converter.comfy_dequant`, which knows nothing about I/O or maps.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader, GGUFWriter

from . import __version__
from . import comfy_dequant
from . import ltx25
from .convert import _SafetensorsRaw, _tensor_payload
from .quant_kernels import quantize, validate_quant_workers


PROFILE_ID = "ltx25-comfyquant"
INVENTORY_FORMAT = "nz-ltx25-comfyquant-inventory-v1"
MANIFEST_FORMAT = "nz-ltx25-comfyquant-manifest-v1"

#: Selectable output policies for the map's ``Q4_K`` rows.  ``Q6_K`` is the
#: default (lower added error); ``Q4_K`` reproduces the official GGUF's type
#: layout exactly.  ``Q8_0`` is deliberately absent: the backend has no fused
#: dequantization kernel for it.
QUANT_TYPES = ("Q6_K", "Q4_K")
DEFAULT_QUANT_TYPE = "Q6_K"

#: Row counts of the approved official map, used as the post-assertion for a
#: full conversion (``{BF16: 2401, F32: 290, <quant_type>: 1658}``).
OFFICIAL_BF16_ROWS = 2401
OFFICIAL_F32_ROWS = 290
OFFICIAL_QUANT_ROWS = 1658

PLAIN_KIND = "plain"
#: Source dtypes admitted for a tensor that carries no quantization sidecars.
PLAIN_SOURCE_DTYPES = ("BF16", "F32")

#: Suffixes the quantizer appends next to ``<layer>.weight``.  Each is folded
#: into its parent row; none of them ever reaches the output GGUF.
SIDECAR_SUFFIXES = (
    "weight_scale",
    "weight_codebook",
    "weight_s_channel",
    "weight_s_rel",
    "comfy_quant",
)
MARKER_SUFFIX = "comfy_quant"
_PARENT_SUFFIX = "weight"

#: Sidecar sets each supported marker format requires, exactly (no more, no less).
_REQUIRED_SIDECARS = {
    "int8_tensorwise": frozenset({"weight_scale", MARKER_SUFFIX}),
    "asym_w4a8_int8": frozenset(
        {"weight_codebook", "weight_s_channel", "weight_s_rel", MARKER_SUFFIX}
    ),
}
#: 4-bit asymmetric codebook: one F32 level per nibble value.
_W4A8_CODEBOOK_SIZE = 16
_W4A8_NIBBLES_PER_BYTE = 2

_MAP_SUBSTITUTED_TYPE = "Q4_K"
_PASSTHROUGH_TARGET_TYPES = ("BF16", "F32")

#: Exactly the KV keys the output GGUF may carry.  ``apply_kv`` writes only
#: keys it knows, so the community file's extra ``__metadata__`` entries
#: (``quant_format``, ``quant_mixed_hi_layers``, ...) are dropped by
#: construction -- this set turns that into a checked contract rather than an
#: assumption, in both directions (a missing key fails just as loudly).
EXPECTED_KV_KEYS = frozenset(
    {
        "general.architecture",
        "general.quantization_version",
        "general.file_type",
        "config",
        "license",
        "model_version",
        ltx25.GEMMA_SOURCE_CHECKPOINT_KEY,
    }
)
#: ``GGUFReader`` exposes the file header itself as pseudo-fields under this
#: prefix; they are not KV entries and are excluded from the contract above.
_READER_PSEUDO_FIELD_PREFIX = "GGUF."

#: The dequantization conventions this build implements, recorded in the
#: manifest so a later output can be told apart from one written by a build
#: that read the format differently.  Every entry is a decision that would
#: silently change the weights if it were flipped.
DEQUANT_LAYOUT = {
    "nibble_order": "low-first",
    "s_rel_op": "multiply",
    "int8_grid_rounding": True,
    "hadamard": "regular-H4-kron/sqrt(size)",
    "rotation_dtype": "float32",
}

#: ``metadata.apply_kv`` writes the fixed ``general.file_type`` of the LTX 2.3
#: reference conversion, which names Q4_K_M regardless of ``--quant-type``.
#: The backend never reads ``general.*``; third-party tools may, so the real
#: type policy is recorded in the manifest instead.
GENERAL_FILE_TYPE_NOTE = (
    "general.file_type is fixed at 15 (Q4_K_M) for all quant types; "
    "consumers must read quant_type from this manifest"
)

#: How each admitted source dtype is reinterpreted from its raw bytes.  NumPy
#: has no FP8 dtype, so ``F8_E4M3`` is carried as raw ``uint8`` and decoded by
#: :func:`comfy_dequant.decode_fp8_e4m3`.
_RAW_READ_DTYPES = {
    "I8": np.dtype(np.int8),
    "F32": np.dtype("<f4"),
    "F8_E4M3": np.dtype(np.uint8),
}


# ---------------------------------------------------------------------------
# errors -- all subclass ltx25.Ltx25Error so cli.py's existing
# ``except ltx25_mod.Ltx25Error`` handler prints them without a traceback.
# ---------------------------------------------------------------------------
class ComfyQuantError(ltx25.Ltx25Error):
    """Base error for expected ltx25-comfyquant failures."""


class ComfyQuantSourceRejectedError(ComfyQuantError):
    """The source file does not satisfy the structural admission contract."""


class ComfyQuantInventoryMismatchError(ComfyQuantError):
    """The folded logical tensor set does not match the official oracle."""


class ComfyQuantPolicyError(ComfyQuantError):
    """The official type policy does not cover the source, or is unusable."""


class ComfyQuantOutputError(ComfyQuantError):
    """The written GGUF does not satisfy this profile's output contract."""


def _reject(message: str) -> ComfyQuantSourceRejectedError:
    return ComfyQuantSourceRejectedError(f"{PROFILE_ID} source rejected: {message}")


def _mismatch(message: str) -> ComfyQuantInventoryMismatchError:
    return ComfyQuantInventoryMismatchError(f"{PROFILE_ID} inventory mismatch: {message}")


def _policy_error(message: str) -> ComfyQuantPolicyError:
    return ComfyQuantPolicyError(f"{PROFILE_ID} policy mismatch: {message}")


def _output_error(message: str) -> ComfyQuantOutputError:
    return ComfyQuantOutputError(f"{PROFILE_ID} self-verify failed: {message}")


def _manifest_error(message: str) -> ltx25.ManifestError:
    return ltx25.ManifestError(f"{PROFILE_ID} manifest mismatch: {message}")


@dataclass(frozen=True)
class ComfyQuantInventory:
    """Folded logical inventory of one community-quantized source file."""

    source_path: Path
    source_sha256: str
    source_size: int
    config_text: str
    config_bytes_sha256: str
    builder_oracle_sha256: str
    logical_tensors: tuple[dict[str, Any], ...]
    raw_tensors: tuple[dict[str, Any], ...]
    quant_summary: dict[str, Any]
    source_metadata_extra: dict[str, str]
    diagnostics: tuple[str, ...]
    inventory_sha256: str


# ---------------------------------------------------------------------------
# stage 1: header completeness (front-loaded for a readable failure)
# ---------------------------------------------------------------------------
def _declared_payload_end(header: dict[str, Any]) -> tuple[int | None, int]:
    """Return (largest declared payload end, tensor count), or (None, 0).

    ``None`` means the header is structurally broken in a way
    :func:`ltx25._validate_tensor_entries` reports far more precisely, so the
    completeness check stands aside and lets that message through.
    """
    declared = 0
    count = 0
    for raw_key, entry in header.items():
        if raw_key == "__metadata__":
            continue
        if not isinstance(entry, dict):
            return None, 0
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)
        ):
            return None, 0
        count += 1
        declared = max(declared, offsets[1])
    return declared, count


def _require_complete_payload(header: dict[str, Any], payload_base: int, source_size: int) -> None:
    """Reject a file whose payload does not match what its header declares.

    ``_validate_tensor_entries`` would also stop a truncated download, but its
    message names one arbitrary out-of-range tensor.  This front-loaded check
    exists purely so the operator is told the actual cause: a partial download.
    """
    available = source_size - payload_base
    declared, total = _declared_payload_end(header)
    if declared is None or total == 0 or declared == available:
        return
    beyond = sum(
        1
        for raw_key, entry in header.items()
        if raw_key != "__metadata__" and entry["data_offsets"][1] > available
    )
    if declared > available:
        cause = "possible truncated or incomplete download"
        detail = f"short by {declared - available} bytes"
    else:
        cause = "possible appended or padded file"
        detail = f"{available - declared} unused trailing bytes"
    raise _reject(
        f"incomplete payload ({cause}): the header declares {declared} payload bytes but the "
        f"file provides {available} ({detail}); {beyond} of {total} tensors end beyond the end "
        f"of the file; expected file size {payload_base + declared}, actual {source_size}"
    )


# ---------------------------------------------------------------------------
# stage 2: fold the quantization sidecars into logical tensors
# ---------------------------------------------------------------------------
def _shape(entry: ltx25.TensorHeader) -> tuple[int, ...]:
    return tuple(entry.shape_logical)


def _read_markers(
    source: Path,
    payload_base: int,
    sidecars: dict[str, dict[str, ltx25.TensorHeader]],
) -> dict[str, dict[str, Any]]:
    """Read and normalize every ``comfy_quant`` marker payload."""
    pending: list[tuple[str, ltx25.TensorHeader]] = []
    for parent, group in sidecars.items():
        marker_entry = group.get(MARKER_SUFFIX)
        if marker_entry is None:
            raise _reject(
                f"quantized tensor {parent!r} carries sidecars {sorted(group)!r} but no "
                f"{MARKER_SUFFIX!r} format marker"
            )
        if marker_entry.source_dtype != "U8" or len(marker_entry.shape_logical) != 1:
            raise _reject(
                f"marker {marker_entry.raw_key!r} must be a 1-D U8 byte string, got "
                f"{marker_entry.source_dtype} {list(marker_entry.shape_logical)!r}"
            )
        pending.append((parent, marker_entry))
    markers: dict[str, dict[str, Any]] = {}
    if not pending:
        return markers
    # One pass in payload order: the markers are a tiny, contiguous tail region.
    with source.open("rb") as fh:
        for parent, entry in sorted(pending, key=lambda item: item[1].data_offsets[0]):
            start, end = entry.data_offsets
            fh.seek(payload_base + start)
            raw = fh.read(end - start)
            if len(raw) != end - start:
                raise _reject(f"marker {entry.raw_key!r} payload could not be read in full")
            try:
                markers[parent] = comfy_dequant.parse_quant_marker(raw)
            except comfy_dequant.ComfyDequantError as exc:
                raise _reject(
                    f"marker {entry.raw_key!r} is not an admissible ComfyUI quantization marker "
                    f"({exc}); supported formats are {list(comfy_dequant.SUPPORTED_FORMATS)!r}"
                ) from exc
    return markers


def _quant_logical_shape(
    raw_key: str,
    entry: ltx25.TensorHeader,
    group: dict[str, ltx25.TensorHeader],
    marker: dict[str, Any],
) -> list[int]:
    """Check one quantized layer's sidecar set and geometry; return its logical shape."""
    quant_format = marker["format"]
    required = _REQUIRED_SIDECARS.get(quant_format)
    if required is None:
        raise _reject(
            f"quantized tensor {raw_key!r} uses format {quant_format!r}, which has no admitted "
            "sidecar geometry in this profile"
        )
    present = frozenset(group)
    if present != required:
        missing = sorted(required - present)
        extra = sorted(present - required)
        parts = []
        if missing:
            parts.append(f"missing={missing!r}")
        if extra:
            parts.append(f"unexpected={extra!r}")
        raise _reject(
            f"quantized tensor {raw_key!r} with format {quant_format!r} has the wrong sidecar "
            f"set: " + "; ".join(parts)
        )
    if entry.source_dtype != "I8":
        raise _reject(
            f"quantized tensor {raw_key!r} has dtype {entry.source_dtype!r}, expected the packed I8 payload"
        )
    if len(entry.shape_logical) != 2:
        raise _reject(
            f"quantized tensor {raw_key!r} shape {list(entry.shape_logical)!r} is not a 2-D [out, in] weight"
        )
    out_features, packed_width = entry.shape_logical

    if quant_format == "int8_tensorwise":
        in_features = packed_width
        scale = group["weight_scale"]
        if scale.source_dtype != "F32" or _shape(scale) not in ((out_features, 1), (out_features,)):
            raise _reject(
                f"{scale.raw_key!r} must be F32 [{out_features}, 1] or [{out_features}], got "
                f"{scale.source_dtype} {list(scale.shape_logical)!r}"
            )
    else:  # asym_w4a8_int8: two 4-bit codes per stored byte
        in_features = packed_width * _W4A8_NIBBLES_PER_BYTE
        group_size = marker.get("group_size")
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
            raise _reject(
                f"marker for {raw_key!r} has an invalid group_size {group_size!r} for format {quant_format!r}"
            )
        if in_features % group_size:
            raise _reject(
                f"quantized tensor {raw_key!r} logical width {in_features} is not a multiple of "
                f"group_size {group_size}"
            )
        s_rel = group["weight_s_rel"]
        expected_rel = (out_features, in_features // group_size)
        if s_rel.source_dtype != "F8_E4M3" or _shape(s_rel) != expected_rel:
            raise _reject(
                f"{s_rel.raw_key!r} must be F8_E4M3 {list(expected_rel)!r}, got "
                f"{s_rel.source_dtype} {list(s_rel.shape_logical)!r}"
            )
        s_channel = group["weight_s_channel"]
        if s_channel.source_dtype != "F32" or _shape(s_channel) != (out_features,):
            raise _reject(
                f"{s_channel.raw_key!r} must be F32 [{out_features}], got "
                f"{s_channel.source_dtype} {list(s_channel.shape_logical)!r}"
            )
        codebook = group["weight_codebook"]
        if codebook.source_dtype != "F32" or _shape(codebook) != (_W4A8_CODEBOOK_SIZE,):
            raise _reject(
                f"{codebook.raw_key!r} must be F32 [{_W4A8_CODEBOOK_SIZE}], got "
                f"{codebook.source_dtype} {list(codebook.shape_logical)!r}"
            )

    convrot_groupsize = marker.get("convrot_groupsize")
    if marker.get("convrot"):
        if (
            not isinstance(convrot_groupsize, int)
            or isinstance(convrot_groupsize, bool)
            or convrot_groupsize <= 0
            or in_features % convrot_groupsize
        ):
            raise _reject(
                f"quantized tensor {raw_key!r} logical width {in_features} is not a multiple of "
                f"convrot_groupsize {convrot_groupsize!r}"
            )
    return [out_features, in_features]


def _sidecar_record(entry: ltx25.TensorHeader) -> dict[str, Any]:
    return {
        "raw_key": entry.raw_key,
        "source_dtype": entry.source_dtype,
        "shape": list(entry.shape_logical),
    }


def _fold_quant_sidecars(
    source: Path,
    payload_base: int,
    entries: list[ltx25.TensorHeader],
) -> list[dict[str, Any]]:
    """Collapse ``<layer>.weight`` + its quantization sidecars into logical rows.

    The stripped logical name is what the official type policy and the GGUF use;
    the raw key is kept alongside it because
    :func:`ltx25._classify_oracle_tensor` requires the ``model.diffusion_model.``
    prefix and rejects an already-stripped key.
    """
    by_key: dict[str, ltx25.TensorHeader] = {}
    for entry in entries:
        if not entry.raw_key.startswith(ltx25.RAW_TRANSFORMER_PREFIX):
            raise _reject(
                f"unknown component key {entry.raw_key!r}: every tensor must carry the official "
                f"{ltx25.RAW_TRANSFORMER_PREFIX!r} prefix"
            )
        if entry.raw_key == ltx25.RAW_TRANSFORMER_PREFIX:
            raise _reject("transformer key is empty after one prefix strip")
        by_key[entry.raw_key] = entry

    sidecars: dict[str, dict[str, ltx25.TensorHeader]] = {}
    for raw_key, entry in by_key.items():
        base, _, last = raw_key.rpartition(".")
        if last not in SIDECAR_SUFFIXES:
            continue
        parent = f"{base}.{_PARENT_SUFFIX}"
        if not base or parent not in by_key:
            raise _reject(
                f"quantization sidecar {raw_key!r} has no parent weight tensor {parent!r}"
            )
        sidecars.setdefault(parent, {})[last] = entry

    markers = _read_markers(source, payload_base, sidecars)

    rows: list[dict[str, Any]] = []
    for raw_key, entry in by_key.items():
        if raw_key.rpartition(".")[2] in SIDECAR_SUFFIXES:
            continue
        name = raw_key[len(ltx25.RAW_TRANSFORMER_PREFIX) :]
        group = sidecars.get(raw_key)
        if group is None:
            if entry.source_dtype not in PLAIN_SOURCE_DTYPES:
                raise _reject(
                    f"unquantized tensor {raw_key!r} has dtype {entry.source_dtype!r}, outside the "
                    f"allowlist {list(PLAIN_SOURCE_DTYPES)!r}"
                )
            rows.append(
                {
                    "name": name,
                    "raw_key": raw_key,
                    "kind": PLAIN_KIND,
                    "shape_logical": list(entry.shape_logical),
                    "source_dtype": entry.source_dtype,
                    "sidecars": {},
                    "marker": None,
                }
            )
            continue
        marker = markers[raw_key]
        rows.append(
            {
                "name": name,
                "raw_key": raw_key,
                "kind": marker["format"],
                "shape_logical": _quant_logical_shape(raw_key, entry, group, marker),
                "source_dtype": entry.source_dtype,
                "sidecars": {
                    suffix: _sidecar_record(group[suffix]) for suffix in sorted(group)
                },
                "marker": dict(marker),
            }
        )
    rows.sort(key=lambda row: row["name"])
    return rows


# ---------------------------------------------------------------------------
# stage 3: official key/shape oracle
# ---------------------------------------------------------------------------
def _describe_set_diff(missing: list[str], extra: list[str], shape_errors: list[str]) -> str:
    parts: list[str] = []
    if missing:
        parts.append(f"missing={len(missing)} {missing[:20]!r}")
    if extra:
        parts.append(f"extra={len(extra)} {extra[:20]!r}")
    if shape_errors:
        parts.append(f"shape_mismatch={len(shape_errors)} {shape_errors[:20]!r}")
    return "; ".join(parts)


def _match_builder_oracle(rows: list[dict[str, Any]], contracts: dict[str, dict[str, Any]]) -> None:
    """Require the folded logical set to equal the official three-component oracle."""
    actual: dict[str, dict[str, list[int]]] = {name: {} for name in contracts}
    for row in rows:
        _, component_name, component_key = ltx25._classify_oracle_tensor(row["raw_key"], contracts)
        component = actual[component_name]
        if component_key in component:
            raise _mismatch(f"duplicate {component_name} key {component_key!r}")
        component[component_key] = list(row["shape_logical"])
    for name in sorted(contracts):
        expected = contracts[name]["state_dict_shapes"]
        found = actual[name]
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        shape_errors = [
            key for key in sorted(set(expected) & set(found)) if list(expected[key]) != found[key]
        ]
        if missing or extra or shape_errors:
            raise _mismatch(f"{name} " + _describe_set_diff(missing, extra, shape_errors))


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------
def _inventory_payload(inventory: ComfyQuantInventory) -> dict[str, Any]:
    return {
        "format": INVENTORY_FORMAT,
        "profile": PROFILE_ID,
        "source_size": inventory.source_size,
        "source_sha256": inventory.source_sha256,
        "config_text": inventory.config_text,
        "config_bytes_sha256": inventory.config_bytes_sha256,
        "builder_oracle_sha256": inventory.builder_oracle_sha256,
        "logical_tensors": list(inventory.logical_tensors),
        "raw_tensors": list(inventory.raw_tensors),
        "quant_summary": inventory.quant_summary,
        "source_metadata_extra": inventory.source_metadata_extra,
        "diagnostics": list(inventory.diagnostics),
    }


def write_inventory(inventory: ComfyQuantInventory, path: str | Path) -> dict[str, Any]:
    """Write the inventory sidecar atomically and return the written payload."""
    payload = _inventory_payload(inventory)
    payload["inventory_sha256"] = inventory.inventory_sha256
    ltx25._write_json_atomic(Path(path), payload)
    return payload


def _quant_summary(rows: list[dict[str, Any]], entries: list[ltx25.TensorHeader]) -> dict[str, Any]:
    variants: dict[bytes, dict[str, Any]] = {}
    for row in rows:
        marker = row["marker"]
        if marker is None:
            continue
        key = ltx25._canonical_json_bytes(marker)
        variant = variants.setdefault(key, {"marker": dict(marker), "count": 0})
        variant["count"] += 1
    return {
        "logical_tensor_count": len(rows),
        "raw_tensor_count": len(entries),
        "logical_kind_counts": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "logical_source_dtype_counts": dict(
            sorted(Counter(row["source_dtype"] for row in rows).items())
        ),
        "raw_dtype_counts": dict(sorted(Counter(entry.source_dtype for entry in entries).items())),
        "marker_variants": [variants[key] for key in sorted(variants)],
    }


def _diagnostics(
    config_text: str,
    metadata_extra: dict[str, str],
    summary: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    quant_format = metadata_extra.get("quant_format")
    if quant_format:
        notes.append(f"source __metadata__.quant_format={quant_format!r}")
    hi_layers = metadata_extra.get("quant_mixed_hi_layers")
    if hi_layers:
        notes.append(f"source __metadata__.quant_mixed_hi_layers={hi_layers!r}")
    notes.append(f"logical kind counts {summary['logical_kind_counts']!r}")
    notes.append(f"raw dtype counts {summary['raw_dtype_counts']!r}")
    notes.append(f"{len(summary['marker_variants'])} distinct quantization marker payload(s)")
    config_markers = ltx25._diagnostic_markers("", config_text)
    if config_markers:
        notes.append(f"diagnostic marker(s) {','.join(config_markers)} in __metadata__.config")
    return sorted(notes)


def inspect_comfyquant(
    source_path: str | Path,
    *,
    builder_oracle_path: str | Path,
    inventory_path: str | Path | None = None,
    expect_sha256: str | None = None,
) -> ComfyQuantInventory:
    """Admit one community-quantized source structurally and fold its inventory."""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"{PROFILE_ID} source not found: {source}")

    header, payload_base, source_size = ltx25._read_validated_header(source)
    # Front-loaded so a partial download says so, instead of naming one tensor.
    _require_complete_payload(header, payload_base, source_size)

    source_sha = ltx25.sha256_of_file(source)
    if expect_sha256 is not None and source_sha.lower() != expect_sha256.strip().lower():
        raise _reject(f"SHA-256 {source_sha} != expected {expect_sha256.strip().lower()}")

    entries = ltx25._validate_tensor_entries(header, source_size - payload_base)
    if not entries:
        raise _reject("the file declares no tensors")

    config_text = ltx25._config_from_header(header)
    config_bytes = config_text.encode("utf-8")
    metadata: dict[str, Any] = header["__metadata__"]
    # Validated for the backend's Gemma pairing check; kept verbatim in extras.
    ltx25.gemma_source_checkpoint_from_metadata(metadata)
    # Unknown keys (quant_format, quant_mixed_hi_layers, ...) are recorded, not rejected.
    metadata_extra = {
        str(key): str(value) for key, value in sorted(metadata.items()) if key != "config"
    }

    rows = _fold_quant_sidecars(source, payload_base, entries)

    oracle = ltx25.load_builder_oracle(builder_oracle_path)
    if oracle["config_bytes_sha256"] != ltx25._sha256_bytes(config_bytes):
        raise _mismatch(
            "official builder oracle config digest differs from the source __metadata__.config; "
            "this file is not the same LTX 2.5 architecture"
        )
    contracts = oracle["component_contracts"]
    if set(contracts) != set(ltx25._LTX25_COMPONENT_RULES):
        raise _mismatch(
            "builder oracle must carry the three-component transformer/audio/video contract"
        )
    _match_builder_oracle(rows, contracts)

    summary = _quant_summary(rows, entries)
    skeleton = {
        "format": INVENTORY_FORMAT,
        "profile": PROFILE_ID,
        "source_size": source_size,
        "source_sha256": source_sha,
        "config_text": config_text,
        "config_bytes_sha256": ltx25._sha256_bytes(config_bytes),
        "builder_oracle_sha256": oracle["oracle_sha256"],
        "logical_tensors": rows,
        "raw_tensors": sorted(
            (
                {
                    "raw_key": entry.raw_key,
                    "source_dtype": entry.source_dtype,
                    "shape": list(entry.shape_logical),
                    "data_offsets": list(entry.data_offsets),
                }
                for entry in entries
            ),
            key=lambda record: record["raw_key"],
        ),
        "quant_summary": summary,
        "source_metadata_extra": metadata_extra,
        "diagnostics": _diagnostics(config_text, metadata_extra, summary),
    }
    inventory = ComfyQuantInventory(
        source_path=source,
        source_sha256=source_sha,
        source_size=source_size,
        config_text=config_text,
        config_bytes_sha256=skeleton["config_bytes_sha256"],
        builder_oracle_sha256=skeleton["builder_oracle_sha256"],
        logical_tensors=tuple(skeleton["logical_tensors"]),
        raw_tensors=tuple(skeleton["raw_tensors"]),
        quant_summary=summary,
        source_metadata_extra=metadata_extra,
        diagnostics=tuple(skeleton["diagnostics"]),
        inventory_sha256=ltx25._sha256_bytes(ltx25._canonical_json_bytes(skeleton)),
    )
    if inventory_path is not None:
        write_inventory(inventory, inventory_path)
    return inventory


def logical_index(inventory: ComfyQuantInventory) -> dict[str, dict[str, Any]]:
    """Return the inventory's logical rows keyed by their stripped GGUF name."""
    return {row["name"]: row for row in inventory.logical_tensors}


# ---------------------------------------------------------------------------
# type policy: the official approved map, read name/shape-only
# ---------------------------------------------------------------------------
def load_official_type_policy(map_path: str | Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Read the approved official map as a name -> (shape, type) policy.

    Only ``name``/``shape_logical``/``ggml_type`` are consumed.  The map's
    ``inventory_sha256`` binds it to the *official* bf16 inventory and its
    ``source_dtype`` column describes the *official* file, so neither applies to
    a community source; both are deliberately ignored.  The file is never written.
    """
    payload = ltx25.load_policy_map(map_path, require_approved=True)
    policy = {
        row["name"]: {
            "shape_logical": list(row["shape_logical"]),
            "ggml_type": row["ggml_type"],
        }
        for row in payload["tensors"]
    }
    return policy, payload["map_sha256"]


def validate_quant_type(quant_type: str) -> str:
    if quant_type not in QUANT_TYPES:
        raise _policy_error(f"quant type must be one of {list(QUANT_TYPES)!r}, got {quant_type!r}")
    return quant_type


def official_type_counts(quant_type: str) -> dict[str, int]:
    """Expected output type histogram for a full official-map conversion."""
    validate_quant_type(quant_type)
    return dict(
        sorted(
            {
                "BF16": OFFICIAL_BF16_ROWS,
                "F32": OFFICIAL_F32_ROWS,
                quant_type: OFFICIAL_QUANT_ROWS,
            }.items()
        )
    )


def type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Output type histogram of writer records (same helper the ltx25 map uses)."""
    return ltx25._type_counts(records)


def expected_type_counts_for_policy(
    policy: dict[str, dict[str, Any]], quant_type: str
) -> dict[str, int]:
    """Type histogram the emitted records must have, taken from the policy itself.

    This is the post-assertion :func:`convert_comfyquant` arms: the output's
    types are the policy's types with every ``Q4_K`` row rewritten to
    ``quant_type``, so any row that silently took another route is caught.  For
    the approved official map the result *is*
    ``official_type_counts(quant_type)`` (``{BF16: 2401, F32: 290, ...: 1658}``);
    deriving it from the policy rather than hard-coding those numbers is what
    lets a miniature fixture state its own histogram without a second code path.
    """
    validate_quant_type(quant_type)
    counts: Counter[str] = Counter()
    for entry in policy.values():
        mapped = entry["ggml_type"]
        counts[quant_type if mapped == _MAP_SUBSTITUTED_TYPE else mapped] += 1
    return dict(sorted(counts.items()))


def records_for_inventory_and_policy(
    inventory: ComfyQuantInventory,
    policy: dict[str, dict[str, Any]],
    quant_type: str,
    *,
    expected_type_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Join the folded inventory with the official policy into writer records.

    Names and logical shapes must match exactly.  The official ``Q4_K`` rows are
    rewritten to ``quant_type``; ``BF16``/``F32`` rows are left alone.  Source
    dtypes are intentionally not compared: the community file stores the 290
    ``*scale_shift_table*`` rows as BF16 where the official file has F32 (a
    lossless widening), and the 96 ``to_gate_logits.weight`` rows as F32 where
    the official file has BF16 (quantized either way).
    """
    validate_quant_type(quant_type)
    rows = logical_index(inventory)
    missing = sorted(set(policy) - set(rows))
    extra = sorted(set(rows) - set(policy))
    if missing or extra:
        raise _policy_error(_describe_set_diff(missing, extra, []))

    records: list[dict[str, Any]] = []
    shape_errors: list[str] = []
    for name in sorted(policy):
        row = rows[name]
        entry = policy[name]
        shape_logical = list(row["shape_logical"])
        if shape_logical != list(entry["shape_logical"]):
            shape_errors.append(name)
            continue
        mapped = entry["ggml_type"]
        if mapped == _MAP_SUBSTITUTED_TYPE:
            target = quant_type
        elif mapped in _PASSTHROUGH_TARGET_TYPES:
            target = mapped
        else:
            raise _policy_error(
                f"{name} uses map type {mapped!r}, which this profile neither passes through "
                f"{list(_PASSTHROUGH_TARGET_TYPES)!r} nor substitutes ({_MAP_SUBSTITUTED_TYPE})"
            )
        if row["source_dtype"] == "F32" and target == "BF16":
            raise _policy_error(
                f"{name} would narrow an F32 source row to BF16; narrowing casts are not admitted"
            )
        records.append(
            {
                "name": name,
                "shape_logical": shape_logical,
                "shape_gguf": list(reversed(shape_logical)),
                "ggml_type": target,
                "nbytes": ltx25._map_nbytes(shape_logical, target),
                "source_kind": row["kind"],
            }
        )
    if shape_errors:
        raise _policy_error(_describe_set_diff([], [], shape_errors))
    if expected_type_counts is not None:
        actual = ltx25._type_counts(records)
        if actual != dict(sorted(expected_type_counts.items())):
            raise _policy_error(
                f"type_counts {actual!r} != expected {dict(sorted(expected_type_counts.items()))!r}"
            )
    return records


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def default_output_path(output_dir: Path, source_path: str | Path, quant_type: str) -> Path:
    """``<output_dir>/<source stem>-<quant_type>.gguf``."""
    validate_quant_type(quant_type)
    return Path(output_dir) / f"{Path(source_path).stem}-{quant_type}.gguf"


def default_inventory_path(output_path: str | Path) -> Path:
    return Path(f"{output_path}.inventory.json")


def default_paths_from_config(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Resolve only the ltx25-comfyquant table; no source lock, no source path.

    The source file is community-built, so ``--st-path`` is always explicit and
    ``config.toml`` carries no ``source_*`` keys for this profile.
    """
    profile = config.get("profiles", {}).get(PROFILE_ID, {})
    if not isinstance(profile, dict):
        profile = {}
    return {
        "map": project_root / str(profile.get("official_map_path", "typemap/ltx25_conversion_map.json")),
        "oracle": project_root
        / str(profile.get("builder_oracle_path", "typemap/ltx25_builder_oracle.json")),
        "output_dir": project_root / str(profile.get("output_dir", "output")),
        "quant_workers": validate_quant_workers(profile.get("quant_workers", 4)),
        "quant_type": validate_quant_type(str(profile.get("quant_type", DEFAULT_QUANT_TYPE))),
    }


# ---------------------------------------------------------------------------
# payload production
# ---------------------------------------------------------------------------
def _read_typed(
    reader: _SafetensorsRaw,
    raw_key: str,
    expected_dtype: str,
    expected_shape: list[int],
) -> np.ndarray:
    """Read one raw tensor and reinterpret its bytes as the admitted dtype.

    The dtype/shape the fold recorded at inspect time are re-checked against the
    header the conversion pass actually opened, so a source edited between the
    two passes cannot slip a differently-shaped payload into the arithmetic.
    """
    actual_dtype = reader.dtype_of(raw_key)
    if actual_dtype != expected_dtype:
        raise _reject(
            f"{raw_key!r} is {actual_dtype!r} at conversion time but was {expected_dtype!r} "
            "when the source was inspected"
        )
    actual_shape = [int(dim) for dim in reader.shape_of(raw_key)]
    wanted = [int(dim) for dim in expected_shape]
    if actual_shape != wanted:
        raise _reject(
            f"{raw_key!r} has shape {actual_shape!r} at conversion time, expected {wanted!r}"
        )
    view_dtype = _RAW_READ_DTYPES.get(expected_dtype)
    if view_dtype is None:
        raise _reject(
            f"{raw_key!r} has dtype {expected_dtype!r}, which this profile cannot reinterpret "
            f"(supported: {sorted(_RAW_READ_DTYPES)!r})"
        )
    array = reader.get_raw_bytes(raw_key).view(view_dtype)
    if array.size != math.prod(wanted):
        raise _reject(
            f"{raw_key!r} holds {array.size} {expected_dtype} elements, expected "
            f"{math.prod(wanted)} for shape {wanted!r}"
        )
    return np.ascontiguousarray(array.reshape(wanted))


def _stored_weight_shape(kind: str, shape_logical: list[int]) -> list[int]:
    """Shape the quantized ``weight`` is *stored* with (w4a8 packs two per byte)."""
    out_features, in_features = shape_logical
    if kind == "asym_w4a8_int8":
        return [out_features, in_features // _W4A8_NIBBLES_PER_BYTE]
    return [out_features, in_features]


def _dequantized_weight(reader: _SafetensorsRaw, row: dict[str, Any]) -> np.ndarray:
    """Invert one quantized logical row into a float32 ``[out, in]`` array."""
    shape_logical = [int(dim) for dim in row["shape_logical"]]
    tensors = {
        "weight": _read_typed(
            reader,
            row["raw_key"],
            row["source_dtype"],
            _stored_weight_shape(row["kind"], shape_logical),
        )
    }
    for suffix, sidecar in row["sidecars"].items():
        if suffix == MARKER_SUFFIX:
            continue
        tensors[suffix] = _read_typed(
            reader, sidecar["raw_key"], sidecar["source_dtype"], sidecar["shape"]
        )
    return comfy_dequant.dequantize_layer(
        row["marker"], tensors, shape_logical[1], layer_name=row["name"]
    )


def _comfy_tensor_payload(
    reader: _SafetensorsRaw,
    row: dict[str, Any],
    record: dict[str, Any],
    *,
    quant_workers: int,
    q4_executor: Executor | None,
) -> np.ndarray:
    """Produce one tensor's exact output payload.

    Unquantized rows take the converter's ordinary path verbatim (a BF16 source
    emitted as BF16 is still a byte-for-byte copy).  A quantized row is first
    inverted to float32 and only then cast or re-quantized, so the dequantized
    values -- not the stored codes -- are what reaches the writer.
    """
    if row["kind"] == PLAIN_KIND:
        return _tensor_payload(
            reader,
            row["raw_key"],
            record,
            quant_workers=quant_workers,
            q4_executor=q4_executor,
        )
    weights = _dequantized_weight(reader, row)
    target = record["ggml_type"]
    if target == "BF16":
        return comfy_dequant.f32_to_bf16_u16(weights)
    if target == "F32":
        return weights
    return quantize(weights, target, q4_workers=quant_workers, q4_executor=q4_executor)


# ---------------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------------
def _output_kv_keys(output_path: str | Path) -> set[str]:
    """Return the output GGUF's KV key set, with the reader already released.

    ``GGUFReader`` memory-maps the file, and on Windows a live mapping blocks
    both ``os.replace`` and ``unlink``.  Reading the keys in their own frame
    means no reader survives in the caller -- or in the traceback of an
    exception raised there -- so the temporary file can still be committed or
    cleaned up afterwards.
    """
    reader = GGUFReader(str(output_path))
    return {key for key in reader.fields if not key.startswith(_READER_PSEUDO_FIELD_PREFIX)}


def _verify_kv_contract(output_path: str | Path) -> None:
    """Require the output's KV key set to be exactly :data:`EXPECTED_KV_KEYS`.

    Compared as a set, not a count, so the failure names the offending keys.
    """
    actual = _output_kv_keys(output_path)
    if actual == EXPECTED_KV_KEYS:
        return
    parts = []
    missing = sorted(EXPECTED_KV_KEYS - actual)
    unexpected = sorted(actual - EXPECTED_KV_KEYS)
    if missing:
        parts.append(f"missing={missing!r}")
    if unexpected:
        parts.append(f"unexpected={unexpected!r}")
    raise _output_error(
        f"output KV key set is not the expected {sorted(EXPECTED_KV_KEYS)!r}: "
        + "; ".join(parts)
    )


def _verify_output_size(output_path: Path, required_bytes: int) -> None:
    """Require the written file to be exactly the size the records predicted."""
    actual = os.path.getsize(output_path)
    if actual != required_bytes:
        raise _output_error(
            f"output is {actual} bytes but the records predict {required_bytes} "
            f"(difference {actual - required_bytes})"
        )


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------
def _manifest(
    *,
    source: Path,
    inventory: ComfyQuantInventory,
    map_path: str | Path,
    map_sha256: str,
    quant_type: str,
    records: list[dict[str, Any]],
    output: Path,
    output_sha256: str,
) -> dict[str, Any]:
    """Build the sidecar that carries everything the GGUF KV deliberately omits."""
    extra = inventory.source_metadata_extra
    return {
        "format": MANIFEST_FORMAT,
        "profile": PROFILE_ID,
        "source_path": str(source),
        "source_size": inventory.source_size,
        "source_sha256": inventory.source_sha256,
        "source_quant_format": extra.get("quant_format"),
        "source_quant_mixed_hi_layers": extra.get("quant_mixed_hi_layers"),
        "quant_kind_counts": inventory.quant_summary["logical_kind_counts"],
        "official_map_path": str(map_path),
        "official_map_sha256": map_sha256,
        "builder_oracle_sha256": inventory.builder_oracle_sha256,
        "inventory_sha256": inventory.inventory_sha256,
        "quant_type": quant_type,
        "dequant_layout": dict(DEQUANT_LAYOUT),
        "output_sha256": output_sha256,
        "output_size": os.path.getsize(output),
        "tensor_count": len(records),
        "type_counts": type_counts(records),
        "tool_version": __version__,
        "general_file_type_note": GENERAL_FILE_TYPE_NOTE,
    }


def convert_comfyquant(
    source_path: str | Path,
    output_path: str | Path,
    *,
    builder_oracle_path: str | Path,
    map_path: str | Path,
    quant_type: str,
    quant_workers: int,
    force: bool = False,
    expect_sha256: str | None = None,
) -> dict[str, Any]:
    """Convert one community-quantized source into a GGUF, and self-verify it.

    The source is admitted exactly once (:func:`inspect_comfyquant` walks the
    whole file to digest it, which costs minutes on the real 17 GB artifact), and
    everything downstream reuses that single inventory.

    Returns the written manifest.  The output GGUF, its ``.inventory.json`` and
    its ``.manifest.json`` are written only after the temporary file has passed
    :func:`ltx25._verify_output`, the KV contract and the size assertion.
    """
    quant_type = validate_quant_type(quant_type)
    quant_workers = validate_quant_workers(quant_workers)
    source = Path(source_path)
    output = Path(output_path)

    policy, map_sha256 = load_official_type_policy(map_path)
    inventory = inspect_comfyquant(
        source, builder_oracle_path=builder_oracle_path, expect_sha256=expect_sha256
    )
    records = records_for_inventory_and_policy(
        inventory,
        policy,
        quant_type,
        expected_type_counts=expected_type_counts_for_policy(policy, quant_type),
    )
    if output.exists() and not force:
        raise ltx25.OutputExistsError(
            f"{PROFILE_ID} output exists: {output}. Re-run with --force to replace it "
            "after temporary verification."
        )

    # Exactly the metadata the writer will see, so the size preflight is exact.
    with _SafetensorsRaw(source) as metadata_reader:
        metadata = metadata_reader.metadata()
    if metadata.get("config") != inventory.config_text:
        raise _mismatch("source config changed after header admission")
    # Admitted before the long write, not after it.
    gemma_source_checkpoint = ltx25.gemma_source_checkpoint_from_metadata(metadata)
    required_bytes = ltx25.estimate_gguf_size(records, metadata)
    ltx25._preflight_disk(output, required_bytes)

    rows = logical_index(inventory)
    temp = ltx25._temp_path(output)
    committed = False
    try:
        writer: GGUFWriter | None = None
        q4_executor = (
            ThreadPoolExecutor(max_workers=quant_workers, thread_name_prefix="q4-k")
            if quant_workers > 1
            else None
        )
        try:
            with _SafetensorsRaw(source) as reader:
                writer = ltx25._build_writer(records, reader.metadata(), temp)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_ti_data_to_file()
                for record in records:
                    payload = _comfy_tensor_payload(
                        reader,
                        rows[record["name"]],
                        record,
                        quant_workers=quant_workers,
                        q4_executor=q4_executor,
                    )
                    if payload.nbytes != record["nbytes"]:
                        raise _output_error(
                            f"{record['name']} payload {payload.nbytes} != policy "
                            f"{record['nbytes']}"
                        )
                    writer.write_tensor_data(payload)
                    del payload
                writer.close()
        finally:
            if writer is not None:
                writer.close()
            if q4_executor is not None:
                q4_executor.shutdown(wait=True, cancel_futures=True)
        ltx25._verify_output(
            temp,
            records,
            inventory.config_text,
            expected_gemma_source_checkpoint=gemma_source_checkpoint,
        )
        _verify_kv_contract(temp)
        _verify_output_size(temp, required_bytes)
        ltx25._replace_with_retry(temp, output)
        committed = True
        manifest = _manifest(
            source=source,
            inventory=inventory,
            map_path=map_path,
            map_sha256=map_sha256,
            quant_type=quant_type,
            records=records,
            output=output,
            output_sha256=ltx25.sha256_of_file(output),
        )
        write_inventory(inventory, default_inventory_path(output))
        ltx25._write_json_atomic(Path(f"{output}.manifest.json"), manifest)
        return manifest
    except Exception as exc:
        remaining = ltx25._cleanup_temp(temp)
        if committed:
            message = f"{PROFILE_ID} conversion committed GGUF but failed afterward: {exc}"
            if remaining:
                message += f"; temporary file retained at {remaining}"
            raise ltx25.ManifestError(message) from exc
        if remaining:
            raise ComfyQuantError(f"{exc}; temporary file retained at {remaining}") from exc
        raise


# ---------------------------------------------------------------------------
# self-verify
# ---------------------------------------------------------------------------
def verify_comfyquant(
    source_path: str | Path,
    output_path: str | Path,
    *,
    builder_oracle_path: str | Path,
    map_path: str | Path,
    expect_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-admit the source and re-check the committed GGUF against the manifest.

    Nothing is trusted on the strength of the sidecars alone: the source is
    inspected again from scratch, the records are re-derived from the official
    policy and the manifest's ``quant_type``, and the output is re-verified
    structurally.  Returns the manifest.
    """
    output = Path(output_path)
    if not output.is_file():
        raise FileNotFoundError(f"{PROFILE_ID} output not found: {output}")
    manifest_file = Path(f"{output}.manifest.json")
    if not manifest_file.is_file():
        raise ltx25.ManifestError(f"{PROFILE_ID} manifest-missing: {manifest_file}")
    manifest = ltx25._read_json(manifest_file)
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("profile") != PROFILE_ID:
        raise _manifest_error("manifest format/profile is invalid")
    if manifest.get("output_sha256") != ltx25.sha256_of_file(output):
        raise _manifest_error("output SHA-256 does not match")
    if manifest.get("output_size") != os.path.getsize(output):
        raise _manifest_error("output size does not match")
    quant_type = validate_quant_type(str(manifest.get("quant_type")))

    policy, map_sha256 = load_official_type_policy(map_path)
    if manifest.get("official_map_sha256") != map_sha256:
        raise _manifest_error("official map SHA-256 does not match")

    inventory_file = default_inventory_path(output)
    if not inventory_file.is_file():
        raise _mismatch(f"inventory-missing: {inventory_file}")
    stored = ltx25._read_json(inventory_file)
    if stored.get("format") != INVENTORY_FORMAT or stored.get("profile") != PROFILE_ID:
        raise _mismatch("saved inventory format/profile is invalid")
    skeleton = {key: value for key, value in stored.items() if key != "inventory_sha256"}
    if ltx25._sha256_bytes(ltx25._canonical_json_bytes(skeleton)) != stored.get(
        "inventory_sha256"
    ):
        raise _mismatch("saved inventory does not match its own recorded SHA-256")
    if manifest.get("inventory_sha256") != stored.get("inventory_sha256"):
        raise _manifest_error("inventory SHA-256 does not match the saved inventory")

    inventory = inspect_comfyquant(
        source_path, builder_oracle_path=builder_oracle_path, expect_sha256=expect_sha256
    )
    if inventory.inventory_sha256 != stored.get("inventory_sha256"):
        raise _mismatch("saved inventory differs from the re-admitted source")
    if manifest.get("source_sha256") != inventory.source_sha256:
        raise _manifest_error("source SHA-256 does not match the re-admitted source")
    if manifest.get("builder_oracle_sha256") != inventory.builder_oracle_sha256:
        raise _manifest_error("builder oracle SHA-256 does not match")

    records = records_for_inventory_and_policy(
        inventory,
        policy,
        quant_type,
        expected_type_counts=expected_type_counts_for_policy(policy, quant_type),
    )
    if manifest.get("tensor_count") != len(records) or manifest.get(
        "type_counts"
    ) != type_counts(records):
        raise _manifest_error(
            f"manifest tensor_count/type_counts do not match the records derived from "
            f"quant_type {quant_type!r}"
        )

    with _SafetensorsRaw(Path(source_path)) as metadata_reader:
        metadata = metadata_reader.metadata()
    ltx25._verify_output(
        output,
        records,
        inventory.config_text,
        expected_gemma_source_checkpoint=ltx25.gemma_source_checkpoint_from_metadata(metadata),
    )
    _verify_kv_contract(output)
    return manifest
