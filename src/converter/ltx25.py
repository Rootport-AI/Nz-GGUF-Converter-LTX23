"""LTX 2.5 transformer-only safetensors to GGUF conversion support.

This module deliberately implements only the two-profile LTX 2.3/LTX 2.5
boundary.  LTX 2.5 requires both an authenticated official source lock and a
checked official-code builder oracle before normal CLI inspection can proceed.
The public helpers accept an explicit test SourceLock so unit tests can
exercise the fail-closed path with small synthetic safetensors files.

The LTX 2.5 path has four explicit stages:

    inspect -> build-policy -> convert -> self-verify

It never reads a third-party GGUF as a typemap oracle.  A future authenticated
official artifact supplies the concrete inventory and reviewed policy map.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES, GGUFValueType
from gguf.quants import dequantize

from . import __version__
from .convert import (
    ARCH,
    _SafetensorsRaw,
    _register_tensor_info,
    _tensor_payload,
)
from .metadata import apply_kv
from .quant_kernels import validate_quant_workers


PROFILE_ID = "ltx25"
RAW_TRANSFORMER_PREFIX = "model.diffusion_model."
SOURCE_LOCK_UNCONFIRMED = "UNCONFIRMED_GATED_AUTH_REQUIRED"
MAP_FORMAT = "nz-ltx25-conversion-map-v1"
INVENTORY_FORMAT = "nz-ltx25-inventory-v1"
MANIFEST_FORMAT = "nz-ltx25-manifest-v1"
BUILDER_ORACLE_FORMAT = "nz-ltx25-builder-oracle-v1"
OFFICIAL_CODE_COMMIT = "400fd31054597515f47125691032c04b1c3ee24e"
# Provenance record the backend loader reads to check which Gemma text encoder
# this transformer was distilled against.  The official bf16 safetensors carries
# it in ``__metadata__``; the converter passes the string through verbatim into
# the output GGUF KV of the same name, and self-verification requires it.
GEMMA_SOURCE_CHECKPOINT_KEY = "gemma_source_checkpoint"
_ALLOWED_TARGET_TYPES = {"F32", "BF16", "Q4_K", "Q5_K", "Q6_K"}
_LTX25_COMPONENT_RULES = {
    "transformer": {"classification": "emit", "component_id": "transformer", "native_prefix": ""},
    "audio_embeddings_connector": {
        "classification": "emit",
        "component_id": "gemma-audio-embeddings-connector",
        "native_prefix": "audio_embeddings_connector.",
    },
    "video_embeddings_connector": {
        "classification": "emit",
        "component_id": "gemma-video-embeddings-connector",
        "native_prefix": "video_embeddings_connector.",
    },
}
_CONNECTOR_COMPONENTS = ("audio_embeddings_connector", "video_embeddings_connector")
_CONNECTOR_GGUF_PREFIXES = tuple(
    _LTX25_COMPONENT_RULES[name]["native_prefix"] for name in _CONNECTOR_COMPONENTS
)
_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0FNU": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
    "F4_E2M1FN": 4,
    "F4_E2M1FNUZ": 4,
    "F6_E2M3FN": 6,
    "F6_E3M2FN": 6,
}


class Ltx25Error(ValueError):
    """Base error for expected LTX 2.5 conversion failures."""


class SourceLockIncompleteError(Ltx25Error):
    """The official gated artifact has not supplied a trusted digest yet."""


class SourceRejectedError(Ltx25Error):
    """The source does not match the selected LTX 2.5 profile."""


class InventoryMismatchError(Ltx25Error):
    """The source/header inventory does not match an expected inventory/map."""


class PolicyMapError(Ltx25Error):
    """The concrete conversion map is malformed or does not cover the source."""


class OutputExistsError(Ltx25Error):
    """A final output exists and explicit replacement was not selected."""


class DiskPreflightError(Ltx25Error):
    """The destination filesystem lacks room for a temporary GGUF."""


class ManifestError(Ltx25Error):
    """The output GGUF was committed but the sidecar manifest is absent/bad."""


@dataclass(frozen=True)
class SourceLock:
    """Pinned facts required to admit an LTX 2.5 source file."""

    repo_id: str
    artifact_revision: str
    filename: str
    expected_size: int | None
    source_sha256: str | None
    lfs_sha256: str | None
    git_blob_oid: str
    gated: str = "auto"
    license_id: str = "ltx-2-community-license-agreement"

    @property
    def complete(self) -> bool:
        return bool(
            self.expected_size is not None
            and self.source_sha256
            and self.lfs_sha256
            and self.source_sha256 != SOURCE_LOCK_UNCONFIRMED
            and self.lfs_sha256 != SOURCE_LOCK_UNCONFIRMED
        )


DEFAULT_SOURCE_LOCK = SourceLock(
    repo_id="Lightricks/LTX-2.5",
    artifact_revision="dd53cc2cd45bbeaa3563dfb575cba3f49cf44761",
    filename="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    expected_size=42_018_190_584,
    source_sha256="31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4",
    lfs_sha256="31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4",
    git_blob_oid="3ba48d13c75fd5df2e868aa8c07b0c5119a6116f",
)


@dataclass(frozen=True)
class TensorHeader:
    """A validated safetensors tensor header entry."""

    raw_key: str
    source_dtype: str
    shape_logical: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class Inventory:
    """Validated LTX 2.5 header inventory used by the concrete policy map."""

    source_path: Path
    source_sha256: str
    source_size: int
    config_text: str
    config_bytes_sha256: str
    builder_oracle_sha256: str | None
    tensors: tuple[dict[str, Any], ...]
    diagnostics: tuple[str, ...]
    inventory_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_of_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceRejectedError(f"ltx25 source rejected: duplicate JSON key {key!r} in safetensors header")
        result[key] = value
    return result


def _read_validated_header(path: str | Path) -> tuple[dict[str, Any], int, int]:
    """Read and structurally validate a safetensors header without payload loading."""
    source = Path(path)
    size = source.stat().st_size
    if size < 8:
        raise SourceRejectedError("ltx25 source rejected: file is shorter than the safetensors header length")
    with source.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        if header_len <= 0 or header_len > size - 8:
            raise SourceRejectedError(
                f"ltx25 source rejected: invalid safetensors header length {header_len} for {size}-byte file"
            )
        if header_len > 128 * 1024 * 1024:
            raise SourceRejectedError(
                f"ltx25 source rejected: header length {header_len} exceeds the 128 MiB safety limit"
            )
        raw = fh.read(header_len)
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise SourceRejectedError(f"ltx25 source rejected: header is not UTF-8 ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise SourceRejectedError(f"ltx25 source rejected: header is not valid JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise SourceRejectedError("ltx25 source rejected: safetensors header must be a JSON object")
    return header, 8 + header_len, size


def _dtype_nbytes(dtype: str, shape: Iterable[int]) -> int:
    bits = _DTYPE_BITS.get(dtype)
    if bits is None:
        raise SourceRejectedError(f"ltx25 source rejected: unknown safetensors dtype {dtype!r}")
    elements = math.prod(shape)
    total_bits = elements * bits
    if total_bits % 8:
        raise SourceRejectedError(
            f"ltx25 source rejected: dtype {dtype!r} and shape {list(shape)!r} do not occupy whole bytes"
        )
    return total_bits // 8


def _validate_tensor_entries(header: dict[str, Any], payload_size: int) -> list[TensorHeader]:
    metadata = header.get("__metadata__")
    if metadata is not None and not isinstance(metadata, dict):
        raise SourceRejectedError("ltx25 source rejected: __metadata__ must be an object")
    entries: list[TensorHeader] = []
    ranges: list[tuple[int, int, str]] = []
    for raw_key, entry in header.items():
        if raw_key == "__metadata__":
            continue
        if not isinstance(raw_key, str) or not raw_key:
            raise SourceRejectedError("ltx25 source rejected: tensor names must be non-empty strings")
        if not isinstance(entry, dict):
            raise SourceRejectedError(f"ltx25 source rejected: tensor {raw_key!r} entry must be an object")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str):
            raise SourceRejectedError(f"ltx25 source rejected: tensor {raw_key!r} has no string dtype")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise SourceRejectedError(
                f"ltx25 source rejected: tensor {raw_key!r} has an invalid non-negative integer shape"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)
        ):
            raise SourceRejectedError(f"ltx25 source rejected: tensor {raw_key!r} has invalid data_offsets")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            raise SourceRejectedError(
                f"ltx25 source rejected: tensor {raw_key!r} offsets {offsets!r} are outside payload size {payload_size}"
            )
        expected_nbytes = _dtype_nbytes(dtype, shape)
        if end - start != expected_nbytes:
            raise SourceRejectedError(
                f"ltx25 source rejected: tensor {raw_key!r} payload is {end - start} bytes, "
                f"expected {expected_nbytes} for {dtype} {shape!r}"
            )
        ranges.append((start, end, raw_key))
        entries.append(TensorHeader(raw_key, dtype, tuple(shape), (start, end)))
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if current[0] < previous[1]:
            raise SourceRejectedError(
                f"ltx25 source rejected: overlapping tensor ranges {previous[2]!r} and {current[2]!r}"
            )
    return entries


def _config_from_header(header: dict[str, Any]) -> str:
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise SourceRejectedError("ltx25 source rejected: __metadata__.config is required")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()):
        raise SourceRejectedError("ltx25 source rejected: __metadata__ values must be strings")
    config_text = metadata.get("config")
    if not isinstance(config_text, str) or not config_text:
        raise SourceRejectedError("ltx25 source rejected: __metadata__.config must be a non-empty JSON string")
    try:
        parsed = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise SourceRejectedError(f"ltx25 source rejected: __metadata__.config is not JSON ({exc})") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("transformer"), dict):
        raise SourceRejectedError("ltx25 source rejected: config must contain a top-level transformer object")
    return config_text


def gemma_source_checkpoint_from_metadata(metadata: dict[str, Any]) -> str:
    """Return the admitted, verbatim ``gemma_source_checkpoint`` string.

    The backend loader reads ``gemma_source_checkpoint.gemma_version`` to decide
    which Gemma text encoder pairs with this transformer, so a missing or
    malformed value is a source rejection rather than a silently dropped KV.
    Only a copy is parsed for validation -- the returned value is the source
    string byte for byte, which is what the output GGUF KV must carry.
    """
    value = metadata.get(GEMMA_SOURCE_CHECKPOINT_KEY)
    if not isinstance(value, str) or not value:
        raise SourceRejectedError(
            f"ltx25 source rejected: __metadata__.{GEMMA_SOURCE_CHECKPOINT_KEY} must be a "
            "non-empty JSON string"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SourceRejectedError(
            f"ltx25 source rejected: __metadata__.{GEMMA_SOURCE_CHECKPOINT_KEY} is not JSON ({exc})"
        ) from exc
    if not isinstance(parsed, dict):
        raise SourceRejectedError(
            f"ltx25 source rejected: __metadata__.{GEMMA_SOURCE_CHECKPOINT_KEY} must be a JSON object"
        )
    gemma_version = parsed.get("gemma_version")
    if not isinstance(gemma_version, str) or not gemma_version:
        raise SourceRejectedError(
            f"ltx25 source rejected: __metadata__.{GEMMA_SOURCE_CHECKPOINT_KEY} lacks a non-empty "
            "gemma_version string"
        )
    return value


def _classify_tensor(raw_key: str, exclude_prefixes: dict[str, str] | None = None) -> tuple[str, str | None]:
    """Return exactly one of emit/native-key or exclude/component-id."""
    matches: list[tuple[str, str | None]] = []
    if raw_key.startswith(RAW_TRANSFORMER_PREFIX):
        native_key = raw_key[len(RAW_TRANSFORMER_PREFIX) :]
        if not native_key:
            raise SourceRejectedError("ltx25 source rejected: transformer key is empty after one prefix strip")
        matches.append(("emit", native_key))
    for prefix, component_id in (exclude_prefixes or {}).items():
        if raw_key.startswith(prefix):
            matches.append(("exclude", component_id))
    if not matches:
        raise SourceRejectedError(f"ltx25 source rejected: unknown component key {raw_key!r}")
    if len(matches) != 1:
        raise SourceRejectedError(
            f"ltx25 source rejected: key {raw_key!r} matched multiple component rules {matches!r}"
        )
    return matches[0]


def _classify_oracle_tensor(raw_key: str, contracts: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    """Classify exactly one raw key using the fixed three-component E2 contract."""
    if not raw_key.startswith(RAW_TRANSFORMER_PREFIX):
        raise SourceRejectedError(f"ltx25 source rejected: unknown component key {raw_key!r}")
    native_key = raw_key[len(RAW_TRANSFORMER_PREFIX) :]
    if not native_key:
        raise SourceRejectedError("ltx25 source rejected: transformer key is empty after one prefix strip")
    for name in ("audio_embeddings_connector", "video_embeddings_connector"):
        contract = contracts[name]
        prefix = contract["native_prefix"]
        if native_key.startswith(prefix):
            component_key = native_key[len(prefix) :]
            if not component_key:
                raise SourceRejectedError(
                    f"ltx25 source rejected: {name} key is empty after component prefix strip"
                )
            return contract["classification"], name, component_key
    return contracts["transformer"]["classification"], "transformer", native_key


def _diagnostic_markers(raw_key: str, config_text: str) -> list[str]:
    markers = ("int8", "convrot", "fp8", "e4m3", "e5m2", "nvfp4", "fp4", "quant", "prequantized")
    haystack = f"{raw_key}\n{config_text}".casefold()
    return [marker for marker in markers if marker in haystack]


def _require_source_lock(path: Path, source_lock: SourceLock, require_complete: bool) -> str:
    if require_complete and not source_lock.complete:
        raise SourceLockIncompleteError(
            "ltx25 source-lock-incomplete: authenticated official source SHA-256/LFS hash is "
            "required before source admission"
        )
    if source_lock.expected_size is not None and path.stat().st_size != source_lock.expected_size:
        raise SourceRejectedError(
            f"ltx25 source rejected: size {path.stat().st_size} != pinned {source_lock.expected_size}"
        )
    actual_sha = sha256_of_file(path)
    if source_lock.source_sha256 and source_lock.source_sha256 != SOURCE_LOCK_UNCONFIRMED:
        if actual_sha.lower() != source_lock.source_sha256.lower():
            raise SourceRejectedError(
                f"ltx25 source rejected: SHA-256 {actual_sha} != pinned {source_lock.source_sha256}"
            )
    if source_lock.lfs_sha256 and source_lock.lfs_sha256 != SOURCE_LOCK_UNCONFIRMED:
        if actual_sha.lower() != source_lock.lfs_sha256.lower():
            raise SourceRejectedError(
                f"ltx25 source rejected: SHA-256 {actual_sha} != pinned LFS/Xet {source_lock.lfs_sha256}"
            )
    return actual_sha


def _inventory_payload(inventory: Inventory) -> dict[str, Any]:
    return {
        "format": INVENTORY_FORMAT,
        "profile": PROFILE_ID,
        "source_size": inventory.source_size,
        "source_sha256": inventory.source_sha256,
        "config_bytes_sha256": inventory.config_bytes_sha256,
        "builder_oracle_sha256": inventory.builder_oracle_sha256,
        "tensors": list(inventory.tensors),
        "diagnostics": list(inventory.diagnostics),
    }


def _validated_oracle_shapes(value: Any, component: str) -> dict[str, list[int]]:
    if not isinstance(value, dict) or not value:
        raise InventoryMismatchError(
            f"ltx25 builder-oracle-missing: {component} key/shape table is missing"
        )
    normalized: dict[str, list[int]] = {}
    for name, shape in value.items():
        if not isinstance(name, str) or not name or not isinstance(shape, list):
            raise InventoryMismatchError(
                f"ltx25 builder-oracle-missing: invalid {component} key/shape row"
            )
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise InventoryMismatchError(
                f"ltx25 builder-oracle-missing: invalid shape for {component}.{name!r}"
            )
        if name in normalized:
            raise InventoryMismatchError(
                f"ltx25 builder-oracle-missing: duplicate {component} key {name!r}"
            )
        normalized[name] = list(shape)
    return normalized


def load_builder_oracle(path: str | Path) -> dict[str, Any]:
    """Load the E2 meta-builder result without importing Torch in converter CI.

    The checked-in/generated artifact is produced separately from the pinned
    official LTX code on meta tensors. Converter code only consumes its exact
    component role/key/shape result; E3's authenticated safetensors header is
    the source-dtype authority. The local backend and third-party GGUFs are
    never oracles.
    """
    payload = _read_json(path)
    if payload.get("format") != BUILDER_ORACLE_FORMAT or payload.get("profile") != PROFILE_ID:
        raise InventoryMismatchError(f"ltx25 builder-oracle-missing: unsupported oracle file {path}")
    if payload.get("official_code_commit") != OFFICIAL_CODE_COMMIT:
        raise InventoryMismatchError(
            "ltx25 builder-oracle-missing: oracle does not use the pinned official code commit"
        )
    config_sha = payload.get("config_bytes_sha256")
    if not isinstance(config_sha, str) or len(config_sha) != 64:
        raise InventoryMismatchError("ltx25 builder-oracle-missing: config bytes SHA-256 is invalid")
    components = payload.get("components")
    contracts: dict[str, dict[str, Any]] = {}
    if components is None:
        # Small synthetic test or legacy E2 fixture: this is transformer-only
        # and cannot describe the production Gemma connector components.
        contracts["transformer"] = {
            **_LTX25_COMPONENT_RULES["transformer"],
            "state_dict_shapes": _validated_oracle_shapes(
                payload.get("builder_state_dict_shapes"), "transformer"
            ),
        }
    else:
        if not isinstance(components, dict) or set(components) != set(_LTX25_COMPONENT_RULES):
            raise InventoryMismatchError(
                "ltx25 builder-oracle-missing: expected transformer and two connector contracts"
            )
        for name, fixed in _LTX25_COMPONENT_RULES.items():
            component = components[name]
            if not isinstance(component, dict):
                raise InventoryMismatchError(f"ltx25 builder-oracle-missing: {name} contract is invalid")
            for field, expected in fixed.items():
                # The original checked-in E2 recorded ``exclude`` for the two
                # Gemma-side component roles.  E2 is a key/shape oracle, not an
                # artifact-packaging policy: accept that legacy role spelling while
                # normalising output classification to the fixed bundle contract.
                if (
                    field == "classification"
                    and name in _CONNECTOR_COMPONENTS
                    and component.get(field) in {"emit", "exclude"}
                ):
                    continue
                if component.get(field) != expected:
                    raise InventoryMismatchError(
                        f"ltx25 builder-oracle-missing: {name} {field} does not match the fixed contract"
                    )
            shapes = _validated_oracle_shapes(component.get("state_dict_shapes"), name)
            contracts[name] = {
                **fixed,
                "state_dict_shapes": shapes,
            }
    recorded = payload.get("oracle_sha256")
    skeleton = dict(payload)
    skeleton.pop("oracle_sha256", None)
    actual = _sha256_bytes(_canonical_json_bytes(skeleton))
    if not isinstance(recorded, str) or recorded != actual:
        raise InventoryMismatchError("ltx25 builder-oracle-missing: oracle SHA-256 is absent or incorrect")
    payload["component_contracts"] = contracts
    payload["builder_state_dict_shapes"] = contracts["transformer"]["state_dict_shapes"]
    return payload


def inspect_ltx25(
    source_path: str | Path,
    *,
    source_lock: SourceLock = DEFAULT_SOURCE_LOCK,
    require_source_lock: bool = True,
    expected_builder_shapes: dict[str, list[int]] | None = None,
    builder_oracle_path: str | Path | None = None,
    require_builder_oracle: bool = True,
    audit_path: str | Path | None = None,
    exclude_prefixes: dict[str, str] | None = None,
) -> Inventory:
    """Perform E3-style header admission and return the canonical inventory."""
    source = Path(source_path)
    if require_source_lock and not source_lock.complete:
        raise SourceLockIncompleteError(
            "ltx25 source-lock-incomplete: authenticated official source SHA-256/LFS hash is "
            "required before conversion"
        )
    if not source.is_file():
        raise FileNotFoundError(f"ltx25 source not found: {source}")
    source_sha = _require_source_lock(source, source_lock, require_source_lock)
    header, payload_base, source_size = _read_validated_header(source)
    config_text = _config_from_header(header)
    config_bytes = config_text.encode("utf-8")
    builder_oracle_sha: str | None = None
    component_contracts: dict[str, dict[str, Any]] | None = None
    if builder_oracle_path is not None:
        oracle = load_builder_oracle(builder_oracle_path)
        if oracle["config_bytes_sha256"] != _sha256_bytes(config_bytes):
            raise InventoryMismatchError(
                "ltx25 inventory mismatch: official builder oracle config digest differs from source metadata"
            )
        oracle_shapes = oracle["builder_state_dict_shapes"]
        expected_builder_shapes = oracle_shapes
        builder_oracle_sha = oracle["oracle_sha256"]
        component_contracts = oracle["component_contracts"]
    elif require_builder_oracle:
        raise InventoryMismatchError(
            "ltx25 builder-oracle-missing: generate the E2 meta-builder key/shape artifact before inspection"
        )
    tensors = _validate_tensor_entries(header, source_size - payload_base)

    seen_builder_keys: set[str] = set()
    component_keys: dict[str, dict[str, list[int]]] = {}
    records: list[dict[str, Any]] = []
    diagnostics: set[str] = set()
    for entry in tensors:
        component_name: str | None = None
        component_key: str | None = None
        if component_contracts is not None and set(component_contracts) == set(_LTX25_COMPONENT_RULES):
            classification, component_name, component_key = _classify_oracle_tensor(
                entry.raw_key, component_contracts
            )
            # Transformer names are the official builder keys.  Connector
            # components are independently oracle-checked after their component
            # prefix strip, but must retain their bare component prefix in GGUF so
            # the existing LTX bundle loader can extract them.
            native_key = entry.raw_key[len(RAW_TRANSFORMER_PREFIX) :]
            value = component_key if component_name == "transformer" else native_key
        else:
            classification, value = _classify_tensor(entry.raw_key, exclude_prefixes)
        record: dict[str, Any] = {
            "raw_key": entry.raw_key,
            "classification": classification,
            "source_dtype": entry.source_dtype,
            "shape_logical": list(entry.shape_logical),
            "data_offsets": list(entry.data_offsets),
        }
        if component_name is not None and component_key is not None:
            expected_shapes = component_contracts[component_name]["state_dict_shapes"]
            if component_key in component_keys.setdefault(component_name, {}):
                raise InventoryMismatchError(
                    f"ltx25 inventory mismatch: duplicate {component_name} key {component_key!r}"
                )
            component_keys[component_name][component_key] = list(entry.shape_logical)
            if component_key not in expected_shapes:
                raise InventoryMismatchError(
                    f"ltx25 inventory mismatch: unknown {component_name} key {component_key!r}"
                )
        if classification == "emit":
            if value is None:
                raise InventoryMismatchError(
                    f"ltx25 inventory mismatch: emitted tensor {entry.raw_key!r} has no builder key"
                )
            builder_key = str(value)
            if builder_key in seen_builder_keys:
                raise InventoryMismatchError(
                    f"ltx25 inventory mismatch: normalized builder key collision {builder_key!r}"
                )
            seen_builder_keys.add(builder_key)
            if component_name in _CONNECTOR_COMPONENTS and entry.source_dtype != "BF16":
                raise SourceRejectedError(
                    f"ltx25 source rejected: connector tensor {entry.raw_key!r} has dtype "
                    f"{entry.source_dtype!r}, expected official E3 BF16"
                )
            if entry.source_dtype not in {"BF16", "F32"}:
                raise SourceRejectedError(
                    f"ltx25 source rejected: emitted tensor {entry.raw_key!r} has dtype "
                    f"{entry.source_dtype!r}, outside the E3 source-dtype allowlist BF16/F32"
                )
            record["builder_state_dict_key"] = builder_key
            if component_name is not None:
                record["component_id"] = component_contracts[component_name]["component_id"]
                if component_name in _CONNECTOR_COMPONENTS:
                    record["component_state_dict_key"] = component_key
        else:
            record["component_id"] = value
            if component_name is not None and component_key is not None:
                record["component_state_dict_key"] = component_key
                if entry.source_dtype not in {"BF16", "F32"}:
                    raise SourceRejectedError(
                        f"ltx25 source rejected: excluded {component_name} tensor {entry.raw_key!r} "
                        f"has dtype {entry.source_dtype!r}, outside the E3 source-dtype allowlist BF16/F32"
                    )
        marker_hits = _diagnostic_markers(entry.raw_key, config_text)
        if marker_hits:
            diagnostics.add(f"diagnostic marker(s) {','.join(marker_hits)} in {entry.raw_key}")
        records.append(record)

    emitted = {
        record["builder_state_dict_key"]: record["shape_logical"]
        for record in records
        if record["classification"] == "emit" and record.get("component_id") in {None, "transformer"}
    }
    if expected_builder_shapes is not None:
        expected = {str(name): list(shape) for name, shape in expected_builder_shapes.items()}
        missing = sorted(set(expected) - set(emitted))
        extra = sorted(set(emitted) - set(expected))
        shape_errors = [
            name
            for name in sorted(set(expected) & set(emitted))
            if list(expected[name]) != list(emitted[name])
        ]
        if missing or extra or shape_errors:
            parts: list[str] = []
            if missing:
                parts.append(f"missing={len(missing)} {missing[:20]!r}")
            if extra:
                parts.append(f"extra={len(extra)} {extra[:20]!r}")
            if shape_errors:
                parts.append(f"shape_mismatch={len(shape_errors)} {shape_errors[:20]!r}")
            raise InventoryMismatchError("ltx25 inventory mismatch: " + "; ".join(parts))
    if component_contracts is not None and set(component_contracts) == set(_LTX25_COMPONENT_RULES):
        for name in _LTX25_COMPONENT_RULES:
            expected = component_contracts[name]["state_dict_shapes"]
            actual = component_keys.get(name, {})
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            shape_errors = [
                key
                for key in sorted(set(expected) & set(actual))
                if expected[key] != actual[key]
            ]
            if missing or extra or shape_errors:
                parts: list[str] = []
                if missing:
                    parts.append(f"missing={len(missing)} {missing[:20]!r}")
                if extra:
                    parts.append(f"extra={len(extra)} {extra[:20]!r}")
                if shape_errors:
                    parts.append(f"shape_mismatch={len(shape_errors)} {shape_errors[:20]!r}")
                raise InventoryMismatchError(
                    f"ltx25 inventory mismatch: {name} " + "; ".join(parts)
                )

    skeleton = {
        "format": INVENTORY_FORMAT,
        "profile": PROFILE_ID,
        "source_size": source_size,
        "source_sha256": source_sha,
        "config_bytes_sha256": _sha256_bytes(config_bytes),
        "builder_oracle_sha256": builder_oracle_sha,
        "tensors": sorted(records, key=lambda record: record["raw_key"]),
        "diagnostics": sorted(diagnostics),
    }
    inventory_sha = _sha256_bytes(_canonical_json_bytes(skeleton))
    inventory = Inventory(
        source_path=source,
        source_sha256=source_sha,
        source_size=source_size,
        config_text=config_text,
        config_bytes_sha256=skeleton["config_bytes_sha256"],
        builder_oracle_sha256=builder_oracle_sha,
        tensors=tuple(skeleton["tensors"]),
        diagnostics=tuple(skeleton["diagnostics"]),
        inventory_sha256=inventory_sha,
    )
    if audit_path is not None:
        payload = _inventory_payload(inventory)
        payload["inventory_sha256"] = inventory.inventory_sha256
        _write_json_atomic(Path(audit_path), payload)
    return inventory


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            value = json.load(fh, object_pairs_hook=_json_object_without_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise Ltx25Error(f"ltx25 JSON is invalid: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise Ltx25Error(f"ltx25 expected a JSON object: {path}")
    return value


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Ltx25Error(f"ltx25 JSON has duplicate key {key!r}")
        value[key] = item
    return value


def load_inventory(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("format") != INVENTORY_FORMAT or payload.get("profile") != PROFILE_ID:
        raise InventoryMismatchError(f"ltx25 inventory mismatch: unsupported inventory file {path}")
    recorded = payload.get("inventory_sha256")
    skeleton = dict(payload)
    skeleton.pop("inventory_sha256", None)
    actual = _sha256_bytes(_canonical_json_bytes(skeleton))
    if not isinstance(recorded, str) or recorded != actual:
        raise InventoryMismatchError("ltx25 inventory mismatch: inventory SHA-256 is absent or incorrect")
    if not isinstance(payload.get("tensors"), list):
        raise InventoryMismatchError("ltx25 inventory mismatch: tensors must be a list")
    if not isinstance(payload.get("source_size"), int) or isinstance(payload["source_size"], bool) or payload["source_size"] < 8:
        raise InventoryMismatchError("ltx25 inventory mismatch: source_size is invalid")
    for field in ("source_sha256", "config_bytes_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise InventoryMismatchError(f"ltx25 inventory mismatch: {field} is invalid")
    oracle_sha = payload.get("builder_oracle_sha256")
    if oracle_sha is not None and (not isinstance(oracle_sha, str) or len(oracle_sha) != 64):
        raise InventoryMismatchError("ltx25 inventory mismatch: builder_oracle_sha256 is invalid")
    if not isinstance(payload.get("diagnostics"), list) or any(
        not isinstance(item, str) for item in payload["diagnostics"]
    ):
        raise InventoryMismatchError("ltx25 inventory mismatch: diagnostics must be a string list")
    raw_keys: set[str] = set()
    builder_keys: set[str] = set()
    for record in payload["tensors"]:
        if not isinstance(record, dict):
            raise InventoryMismatchError("ltx25 inventory mismatch: tensor record must be an object")
        raw_key = record.get("raw_key")
        classification = record.get("classification")
        shape = record.get("shape_logical")
        offsets = record.get("data_offsets")
        if not isinstance(raw_key, str) or not raw_key or raw_key in raw_keys:
            raise InventoryMismatchError("ltx25 inventory mismatch: duplicate or invalid raw key")
        raw_keys.add(raw_key)
        if classification not in {"emit", "exclude"}:
            raise InventoryMismatchError(f"ltx25 inventory mismatch: {raw_key!r} has invalid classification")
        if not isinstance(record.get("source_dtype"), str) or not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise InventoryMismatchError(f"ltx25 inventory mismatch: {raw_key!r} has invalid dtype or shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) for value in offsets
        ) or offsets[0] < 0 or offsets[1] < offsets[0] or offsets[1] > payload["source_size"]:
            raise InventoryMismatchError(f"ltx25 inventory mismatch: {raw_key!r} has invalid offsets")
        builder_key = record.get("builder_state_dict_key")
        component_id = record.get("component_id")
        if classification == "emit":
            if (
                not isinstance(builder_key, str)
                or not builder_key
                or builder_key in builder_keys
                or record["source_dtype"] not in {"BF16", "F32"}
            ):
                raise InventoryMismatchError(f"ltx25 inventory mismatch: {raw_key!r} emit classification is inconsistent")
            if component_id is not None and component_id not in {
                rule["component_id"] for rule in _LTX25_COMPONENT_RULES.values()
            }:
                raise InventoryMismatchError(f"ltx25 inventory mismatch: {raw_key!r} has invalid component_id")
            if builder_key.startswith(_CONNECTOR_GGUF_PREFIXES):
                if component_id not in {
                    _LTX25_COMPONENT_RULES[name]["component_id"] for name in _CONNECTOR_COMPONENTS
                } or record["source_dtype"] != "BF16":
                    raise InventoryMismatchError(
                        f"ltx25 inventory mismatch: {raw_key!r} connector emit classification is inconsistent"
                    )
            builder_keys.add(builder_key)
        elif not isinstance(component_id, str) or not component_id or builder_key is not None:
            raise InventoryMismatchError(f"ltx25 inventory mismatch: {raw_key!r} exclude classification is inconsistent")
    return payload


def _map_nbytes(shape_logical: list[int], ggml_type: str) -> int:
    if ggml_type == "F32":
        return math.prod(shape_logical) * 4
    if ggml_type == "BF16":
        return math.prod(shape_logical) * 2
    from .convert import _byte_shape_and_nbytes

    try:
        _, nbytes = _byte_shape_and_nbytes(shape_logical, ggml_type)
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyMapError(
            f"ltx25 map mismatch: cannot calculate {ggml_type} bytes for shape {shape_logical!r}"
        ) from exc
    return nbytes


def build_policy_map(
    inventory_path: str | Path,
    map_path: str | Path,
) -> dict[str, Any]:
    """Create the review-only concrete E4 draft from the authenticated E3 rows.

    The generated rows are concrete and direct-lookup based.  They are marked
    draft. A separate review must promote a checked-in E4 map; this CLI never
    self-approves a conversion policy.
    """
    inventory = load_inventory(inventory_path)
    if not isinstance(inventory.get("builder_oracle_sha256"), str):
        raise InventoryMismatchError(
            "ltx25 inventory mismatch: build-map requires a production E2 builder oracle"
        )
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for record in inventory["tensors"]:
        if record.get("classification") != "emit":
            continue
        name = record.get("builder_state_dict_key")
        dtype = record.get("source_dtype")
        shape = record.get("shape_logical")
        if not isinstance(name, str) or not isinstance(dtype, str) or not isinstance(shape, list):
            raise InventoryMismatchError("ltx25 inventory mismatch: emitted record is incomplete")
        if dtype not in {"BF16", "F32"}:
            raise InventoryMismatchError(
                f"ltx25 inventory mismatch: {name} has unsupported source dtype {dtype!r}"
            )
        if name.startswith(_CONNECTOR_GGUF_PREFIXES):
            if dtype != "BF16":
                raise InventoryMismatchError(
                    f"ltx25 inventory mismatch: connector {name} must have official BF16 source dtype"
                )
            default_ggml_type = "BF16"
            rule_id = "connector-bf16-bundle-preservation"
            reason = "LTX bundle connector remains BF16 under its bare GGUF key."
        elif dtype == "F32":
            default_ggml_type = "F32"
            rule_id = "f32-source-preservation"
            reason = "Official E3 F32 row remains F32."
        elif len(shape) == 2 and shape[-1] % 256 == 0 and name.endswith(".weight"):
            default_ggml_type = "Q4_K"
            rule_id = "q4-k-aligned-linear-weight"
            reason = "Official BF16 2-D .weight row has K-compatible input width."
        else:
            default_ggml_type = "BF16"
            rule_id = "bf16-nonmatrix-or-ineligible"
            reason = "Official BF16 row is not a K-compatible 2-D .weight tensor."
        if name in names:
            raise InventoryMismatchError(f"ltx25 inventory mismatch: duplicate emitted key {name!r}")
        names.add(name)
        rows.append(
            {
                "name": name,
                "source_dtype": dtype,
                "shape_logical": list(shape),
                "shape_gguf": list(reversed(shape)),
                "ggml_type": default_ggml_type,
                "nbytes": _map_nbytes(list(shape), default_ggml_type),
                "rule_id": rule_id,
                "reason": reason,
                "evidence_ids": ["E3", "E4"],
            }
        )
    if not rows:
        raise InventoryMismatchError("ltx25 inventory mismatch: no emitted tensors")
    payload: dict[str, Any] = {
        "format": MAP_FORMAT,
        "profile": PROFILE_ID,
        "status": "draft",
        "inventory_sha256": inventory["inventory_sha256"],
        "tensors": rows,
        "type_counts": dict(sorted(Counter(row["ggml_type"] for row in rows).items())),
    }
    payload["map_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    _write_json_atomic(Path(map_path), payload)
    return payload


def load_policy_map(path: str | Path, *, require_approved: bool = True) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("format") != MAP_FORMAT or payload.get("profile") != PROFILE_ID:
        raise PolicyMapError(f"ltx25 map mismatch: unsupported map file {path}")
    if require_approved and payload.get("status") != "approved":
        raise PolicyMapError("ltx25 map mismatch: conversion requires an approved concrete map")
    if payload.get("status") not in {"draft", "approved"}:
        raise PolicyMapError("ltx25 map mismatch: status must be draft or approved")
    inventory_sha = payload.get("inventory_sha256")
    if not isinstance(inventory_sha, str) or len(inventory_sha) != 64:
        raise PolicyMapError("ltx25 map mismatch: inventory SHA-256 is invalid")
    recorded = payload.get("map_sha256")
    skeleton = dict(payload)
    skeleton.pop("map_sha256", None)
    actual = _sha256_bytes(_canonical_json_bytes(skeleton))
    if not isinstance(recorded, str) or recorded != actual:
        raise PolicyMapError("ltx25 map mismatch: map SHA-256 is absent or incorrect")
    rows = payload.get("tensors")
    if not isinstance(rows, list) or not rows:
        raise PolicyMapError("ltx25 map mismatch: tensors must be a non-empty list")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise PolicyMapError("ltx25 map mismatch: tensor row must be an object")
        name = row.get("name")
        dtype = row.get("source_dtype")
        shape = row.get("shape_logical")
        ggml_type = row.get("ggml_type")
        if not isinstance(name, str) or name in names:
            raise PolicyMapError(f"ltx25 map mismatch: duplicate or invalid tensor name {name!r}")
        names.add(name)
        if dtype not in {"BF16", "F32"}:
            raise PolicyMapError(f"ltx25 map mismatch: {name} source dtype must be BF16 or F32")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise PolicyMapError(f"ltx25 map mismatch: {name} has invalid shape")
        if row.get("shape_gguf") != list(reversed(shape)):
            raise PolicyMapError(f"ltx25 map mismatch: {name} has wrong GGUF shape")
        if ggml_type not in _ALLOWED_TARGET_TYPES:
            raise PolicyMapError(f"ltx25 map mismatch: {name} uses unsupported type {ggml_type!r}")
        if dtype == "F32" and ggml_type != "F32":
            raise PolicyMapError(f"ltx25 map mismatch: {name} F32 source must remain F32")
        if name.startswith(_CONNECTOR_GGUF_PREFIXES) and (dtype != "BF16" or ggml_type != "BF16"):
            raise PolicyMapError(
                f"ltx25 map mismatch: connector {name} must remain BF16 for the LTX bundle loader"
            )
        if ggml_type.startswith("Q") and (not shape or shape[-1] % 256):
            raise PolicyMapError(f"ltx25 map mismatch: {name} K-quant last dimension is not divisible by 256")
        if not isinstance(row.get("nbytes"), int) or isinstance(row["nbytes"], bool) or row["nbytes"] < 0:
            raise PolicyMapError(f"ltx25 map mismatch: {name} has invalid nbytes")
        expected_nbytes = _map_nbytes(shape, ggml_type)
        if row.get("nbytes") != expected_nbytes:
            raise PolicyMapError(
                f"ltx25 map mismatch: {name} nbytes={row.get('nbytes')} != expected {expected_nbytes}"
            )
        if not isinstance(row.get("rule_id"), str) or not row["rule_id"].strip():
            raise PolicyMapError(f"ltx25 map mismatch: {name} has an empty rule_id")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise PolicyMapError(f"ltx25 map mismatch: {name} has an empty reason")
        evidence_ids = row.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids or any(
            not isinstance(item, str) or not item.startswith("E") or not item[1:].isdigit()
            for item in evidence_ids
        ):
            raise PolicyMapError(f"ltx25 map mismatch: {name} has invalid evidence_ids")
    expected_type_counts = _type_counts(rows)
    if payload.get("type_counts") != expected_type_counts:
        raise PolicyMapError("ltx25 map mismatch: type_counts does not match tensor rows")
    return payload


def _records_for_inventory_and_map(inventory: Inventory, policy_map: dict[str, Any]) -> list[dict[str, Any]]:
    emitted = [
        record for record in inventory.tensors if record["classification"] == "emit"
    ]
    source_by_key = {record["builder_state_dict_key"]: record for record in emitted}
    map_rows = policy_map["tensors"]
    map_by_key = {row["name"]: row for row in map_rows}
    if len(map_by_key) != len(map_rows):
        raise PolicyMapError("ltx25 map mismatch: duplicate map names")
    missing = sorted(set(source_by_key) - set(map_by_key))
    extra = sorted(set(map_by_key) - set(source_by_key))
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={len(missing)} {missing[:20]!r}")
        if extra:
            parts.append(f"extra={len(extra)} {extra[:20]!r}")
        raise PolicyMapError("ltx25 map mismatch: " + "; ".join(parts))
    records: list[dict[str, Any]] = []
    for source in sorted(emitted, key=lambda record: record["builder_state_dict_key"]):
        row = map_by_key[source["builder_state_dict_key"]]
        if row["source_dtype"] != source["source_dtype"] or row["shape_logical"] != source["shape_logical"]:
            raise PolicyMapError(f"ltx25 map mismatch: {row['name']} dtype or shape differs from inventory")
        records.append(dict(row))
    return records


def _build_writer(records: list[dict[str, Any]], metadata: dict[str, Any], path: str | Path | None) -> GGUFWriter:
    writer = GGUFWriter(str(path) if path is not None else None, arch=ARCH)
    apply_kv(writer, metadata)
    for record in records:
        _register_tensor_info(writer, record["name"], record)
    return writer


def estimate_gguf_size(records: list[dict[str, Any]], metadata: dict[str, Any]) -> int:
    """Calculate the exact one-shard GGUF byte count for this writer/version."""
    writer = _build_writer(records, metadata, path=None)
    header_nbytes = 4 + 4 + 8 + 8
    kv_nbytes = 0
    for key, value in writer.kv_data[0].items():
        kv_nbytes += len(writer._pack_val(key, GGUFValueType.STRING, add_vtype=False))
        kv_nbytes += len(writer._pack_val(value.value, value.type, add_vtype=True, sub_type=value.sub_type))
    ti_nbytes = 0
    payload_nbytes = 0
    for name, tensor_info in writer.tensors[0].items():
        ti_nbytes += len(writer._pack_val(name, GGUFValueType.STRING, add_vtype=False))
        ti_nbytes += 4 + 8 * len(tensor_info.shape) + 4 + 8
        payload_nbytes += GGUFWriter.ggml_pad(tensor_info.nbytes, writer.data_alignment)
    return GGUFWriter.ggml_pad(header_nbytes + kv_nbytes + ti_nbytes, writer.data_alignment) + payload_nbytes


def _preflight_disk(out_path: Path, required_bytes: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(out_path.parent).free
    if free_bytes < required_bytes:
        raise DiskPreflightError(
            f"ltx25 disk-preflight-failed: need {required_bytes} bytes, available {free_bytes} bytes"
        )


def _temp_path(final_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
        delete=False,
    ) as fh:
        return Path(fh.name)


def _cleanup_temp(path: Path) -> str | None:
    if not path.exists():
        return None
    for attempt in range(2):
        try:
            path.unlink()
            return None
        except PermissionError:
            if attempt == 0:
                time.sleep(0.1)
            else:
                return str(path)
        except OSError:
            return str(path)
    return str(path)


def _replace_with_retry(temp_path: Path, final_path: Path) -> None:
    """Commit the verified GGUF, allowing one short Windows handle retry."""
    for attempt in range(2):
        try:
            os.replace(temp_path, final_path)
            return
        except PermissionError as exc:
            if attempt == 0:
                time.sleep(0.1)
                continue
            raise Ltx25Error(
                f"ltx25 output-commit-failed: cannot replace {final_path}; close any process using it and retry"
            ) from exc


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_path(path)
    try:
        temp.write_bytes(_canonical_json_bytes(payload) + b"\n")
        os.replace(temp, path)
    except Exception:
        remaining = _cleanup_temp(temp)
        suffix = f"; temporary manifest retained at {remaining}" if remaining else ""
        raise ManifestError(f"ltx25 manifest write failed{suffix}")


def _type_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["ggml_type"] for record in records).items()))


_RAW_COMPARE_CHUNK_BYTES = 4 * 1024 * 1024


def _verify_connector_raw_bytes(
    reader: GGUFReader,
    source_path: Path,
    inventory: Inventory,
    *,
    chunk_bytes: int = _RAW_COMPARE_CHUNK_BYTES,
) -> None:
    """Compare every emitted BF16 bundle connector with source bytes in bounded chunks."""
    connector_ids = {
        _LTX25_COMPONENT_RULES[name]["component_id"] for name in _CONNECTOR_COMPONENTS
    }
    connectors = [
        record
        for record in inventory.tensors
        if record.get("component_id") in connector_ids
    ]
    if not connectors:
        return
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
        raise ValueError("ltx25 self-verify failed: raw compare chunk size must be positive")
    output_by_name = {tensor.name: tensor for tensor in reader.tensors}
    _, payload_base, _ = _read_validated_header(source_path)
    with source_path.open("rb") as source:
        for record in connectors:
            name = record["builder_state_dict_key"]
            raw_key = record["raw_key"]
            offsets = record["data_offsets"]
            if record.get("source_dtype") != "BF16":
                raise Ltx25Error(
                    f"ltx25 self-verify failed: connector {name} source dtype is not BF16"
                )
            tensor = output_by_name.get(name)
            if tensor is None or tensor.tensor_type.name != "BF16":
                raise Ltx25Error(
                    f"ltx25 self-verify failed: connector {name} is absent or not BF16 in GGUF"
                )
            output_bytes = np.asarray(tensor.data)
            if not output_bytes.flags.c_contiguous:
                raise Ltx25Error(f"ltx25 self-verify failed: connector {name} payload is not contiguous")
            output_raw = output_bytes.view(np.uint8).reshape(-1)
            expected_nbytes = offsets[1] - offsets[0]
            if output_raw.nbytes != expected_nbytes:
                raise Ltx25Error(
                    f"ltx25 self-verify failed: connector {name} payload bytes "
                    f"{output_raw.nbytes} != source {expected_nbytes}"
                )
            source.seek(payload_base + offsets[0])
            for start in range(0, expected_nbytes, chunk_bytes):
                count = min(chunk_bytes, expected_nbytes - start)
                source_chunk = source.read(count)
                if len(source_chunk) != count:
                    raise Ltx25Error(
                        f"ltx25 self-verify failed: connector {raw_key} source payload is truncated"
                    )
                if not np.array_equal(output_raw[start : start + count], np.frombuffer(source_chunk, dtype=np.uint8)):
                    raise Ltx25Error(
                        f"ltx25 self-verify failed: connector {name} raw payload differs from source at byte {start}"
                    )


def _verify_output(
    output_path: Path,
    records: list[dict[str, Any]],
    expected_config_text: str | None = None,
    *,
    source_path: Path | None = None,
    inventory: Inventory | None = None,
    expected_gemma_source_checkpoint: str | None = None,
) -> None:
    reader = GGUFReader(str(output_path))
    if len(reader.tensors) != len(records):
        raise Ltx25Error(
            f"ltx25 self-verify failed: tensor count {len(reader.tensors)} != {len(records)}"
        )
    for index, (tensor, record) in enumerate(zip(reader.tensors, records)):
        if tensor.name != record["name"]:
            raise Ltx25Error(
                f"ltx25 self-verify failed: tensor #{index} name {tensor.name!r} != {record['name']!r}"
            )
        if tensor.tensor_type.name != record["ggml_type"]:
            raise Ltx25Error(
                f"ltx25 self-verify failed: {tensor.name} type {tensor.tensor_type.name} != {record['ggml_type']}"
            )
        if [int(dim) for dim in tensor.shape] != list(record["shape_gguf"]):
            raise Ltx25Error(f"ltx25 self-verify failed: {tensor.name} shape does not match map")
    config_field = reader.fields.get("config")
    if config_field is None:
        raise Ltx25Error("ltx25 self-verify failed: output lacks config metadata")
    config_text = config_field.contents()
    try:
        parsed = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise Ltx25Error(f"ltx25 self-verify failed: output config is invalid JSON ({exc})") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("transformer"), dict):
        raise Ltx25Error("ltx25 self-verify failed: output config lacks transformer object")
    if expected_config_text is not None and config_text.encode("utf-8") != expected_config_text.encode("utf-8"):
        raise Ltx25Error("ltx25 self-verify failed: output config bytes differ from source metadata")
    if expected_gemma_source_checkpoint is not None:
        checkpoint_field = reader.fields.get(GEMMA_SOURCE_CHECKPOINT_KEY)
        if checkpoint_field is None:
            raise Ltx25Error(
                f"ltx25 self-verify failed: output lacks {GEMMA_SOURCE_CHECKPOINT_KEY} metadata"
            )
        if checkpoint_field.contents().encode("utf-8") != expected_gemma_source_checkpoint.encode("utf-8"):
            raise Ltx25Error(
                f"ltx25 self-verify failed: output {GEMMA_SOURCE_CHECKPOINT_KEY} bytes differ "
                "from source metadata"
            )
    for tensor in reader.tensors:
        if tensor.tensor_type.name.startswith("Q"):
            _verify_quant_tensor_finite(tensor)
    if source_path is not None and inventory is not None:
        _verify_connector_raw_bytes(reader, source_path, inventory)


def _verify_quant_tensor_finite(tensor: Any, blocks_per_chunk: int = 1024) -> None:
    """Dequantize a bounded number of K blocks at a time for E5 finiteness."""
    try:
        _, block_nbytes = GGML_QUANT_SIZES[tensor.tensor_type]
    except KeyError as exc:
        raise Ltx25Error(f"ltx25 self-verify failed: unsupported quant type {tensor.tensor_type.name}") from exc
    raw = np.ascontiguousarray(np.asarray(tensor.data)).view(np.uint8).reshape(-1)
    chunk_nbytes = max(int(block_nbytes), int(block_nbytes) * blocks_per_chunk)
    if raw.size % int(block_nbytes):
        raise Ltx25Error(f"ltx25 self-verify failed: malformed quant payload for {tensor.name}")
    for start in range(0, raw.size, chunk_nbytes):
        values = dequantize(raw[start : start + chunk_nbytes], tensor.tensor_type)
        if not np.isfinite(values).all():
            raise Ltx25Error(f"ltx25 self-verify failed: non-finite values after dequantizing {tensor.name}")


def convert_ltx25(
    source_path: str | Path,
    policy_map_path: str | Path,
    output_path: str | Path,
    *,
    source_lock: SourceLock = DEFAULT_SOURCE_LOCK,
    require_source_lock: bool = True,
    force: bool = False,
    expected_builder_shapes: dict[str, list[int]] | None = None,
    builder_oracle_path: str | Path | None = None,
    require_builder_oracle: bool = True,
    quant_workers: int = 4,
) -> Path:
    """Convert an admitted LTX 2.5 transformer using an approved concrete map."""
    quant_workers = validate_quant_workers(quant_workers)
    source = Path(source_path)
    output = Path(output_path)
    if require_source_lock and not source_lock.complete:
        raise SourceLockIncompleteError(
            "ltx25 source-lock-incomplete: authenticated official source SHA-256/LFS hash is "
            "required before conversion"
        )
    policy_map = load_policy_map(policy_map_path, require_approved=True)
    inventory = inspect_ltx25(
        source,
        source_lock=source_lock,
        require_source_lock=require_source_lock,
        expected_builder_shapes=expected_builder_shapes,
        builder_oracle_path=builder_oracle_path,
        require_builder_oracle=require_builder_oracle,
    )
    if policy_map.get("inventory_sha256") != inventory.inventory_sha256:
        raise PolicyMapError("ltx25 map mismatch: map inventory hash differs from inspected source")
    records = _records_for_inventory_and_map(inventory, policy_map)
    if output.exists() and not force:
        raise OutputExistsError(
            f"ltx25 output exists: {output}. Re-run with --force to replace it after temporary verification."
        )
    # Use exactly the metadata that the shared writer will see.  This keeps the
    # preflight calculation exact even when optional source metadata is present.
    with _SafetensorsRaw(source) as metadata_reader:
        metadata = metadata_reader.metadata()
    if metadata.get("config") != inventory.config_text:
        raise InventoryMismatchError("ltx25 inventory mismatch: source config changed after header admission")
    # Admitted before the multi-hour write, not after: a source without a usable
    # provenance record must fail immediately rather than at self-verify.
    gemma_source_checkpoint = gemma_source_checkpoint_from_metadata(metadata)
    required_bytes = estimate_gguf_size(records, metadata)
    _preflight_disk(output, required_bytes)
    temp = _temp_path(output)
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
                writer = _build_writer(records, reader.metadata(), temp)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_ti_data_to_file()
                raw_by_builder = {
                    record["builder_state_dict_key"]: record["raw_key"]
                    for record in inventory.tensors
                    if record["classification"] == "emit"
                }
                for record in records:
                    payload = _tensor_payload(
                        reader,
                        raw_by_builder[record["name"]],
                        record,
                        quant_workers=quant_workers,
                        q4_executor=q4_executor,
                    )
                    if payload.nbytes != record["nbytes"]:
                        raise Ltx25Error(
                            f"ltx25 conversion failed: {record['name']} payload {payload.nbytes} != map {record['nbytes']}"
                        )
                    writer.write_tensor_data(payload)
                    del payload
                writer.close()
        finally:
            if writer is not None:
                writer.close()
            if q4_executor is not None:
                q4_executor.shutdown(wait=True, cancel_futures=True)
        _verify_output(
            temp,
            records,
            inventory.config_text,
            source_path=source,
            inventory=inventory,
            expected_gemma_source_checkpoint=gemma_source_checkpoint,
        )
        _replace_with_retry(temp, output)
        committed = True
        output_sha = sha256_of_file(output)
        manifest = {
            "format": MANIFEST_FORMAT,
            "profile": PROFILE_ID,
            "source_sha256": inventory.source_sha256,
            "inventory_sha256": inventory.inventory_sha256,
            "map_sha256": policy_map["map_sha256"],
            "output_sha256": output_sha,
            "tensor_count": len(records),
            "type_counts": _type_counts(records),
            "tool_version": __version__,
        }
        _write_json_atomic(Path(f"{output}.manifest.json"), manifest)
        return output
    except Exception as exc:
        remaining = _cleanup_temp(temp)
        if committed:
            message = f"ltx25 conversion committed GGUF but failed afterward: {exc}"
            if remaining:
                message += f"; temporary file retained at {remaining}"
            raise ManifestError(message) from exc
        if remaining:
            raise Ltx25Error(f"{exc}; temporary file retained at {remaining}") from exc
        raise


def verify_ltx25(
    output_path: str | Path,
    policy_map_path: str | Path,
    *,
    inventory_path: str | Path,
    source_path: str | Path,
    source_lock: SourceLock,
    builder_oracle_path: str | Path,
    manifest_path: str | Path | None = None,
    require_source_lock: bool = True,
) -> dict[str, Any]:
    """Verify E1/E3/E4/E5 consistency without importing a backend or Torch.

    Source lock, canonical inventory, source, and E2 oracle are mandatory so
    self-verification re-admits E1/E2/E3 rather than trusting a sidecar alone.
    """
    if require_source_lock and not source_lock.complete:
        raise SourceLockIncompleteError(
            "ltx25 source-lock-incomplete: authenticated official source SHA-256/LFS hash is "
            "required before self-verification"
        )
    output = Path(output_path)
    if not output.is_file():
        raise FileNotFoundError(f"ltx25 output not found: {output}")
    policy_map = load_policy_map(policy_map_path, require_approved=True)
    records = sorted(
        (dict(row) for row in policy_map["tensors"]),
        key=lambda row: str(row["name"]),
    )
    manifest_file = Path(manifest_path) if manifest_path is not None else Path(f"{output}.manifest.json")
    if not manifest_file.is_file():
        raise ManifestError(f"ltx25 manifest-missing: {manifest_file}")
    manifest = _read_json(manifest_file)
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("profile") != PROFILE_ID:
        raise ManifestError("ltx25 manifest-missing: manifest format/profile is invalid")
    if manifest.get("map_sha256") != policy_map.get("map_sha256"):
        raise ManifestError("ltx25 manifest-missing: map SHA-256 does not match")
    if manifest.get("output_sha256") != sha256_of_file(output):
        raise ManifestError("ltx25 manifest-missing: output SHA-256 does not match")
    if manifest.get("tensor_count") != len(records) or manifest.get("type_counts") != _type_counts(records):
        raise ManifestError("ltx25 manifest-missing: tensor counts do not match")
    inventory_file = Path(inventory_path)
    if not inventory_file.is_file():
        raise InventoryMismatchError(f"ltx25 inventory-missing: {inventory_file}")
    loaded_inventory = load_inventory(inventory_file)
    inventory_sha = loaded_inventory["inventory_sha256"]
    if policy_map.get("inventory_sha256") != inventory_sha:
        raise InventoryMismatchError("ltx25 inventory mismatch: map does not match canonical inventory")
    if manifest.get("inventory_sha256") != inventory_sha:
        raise ManifestError("ltx25 manifest-missing: inventory SHA-256 does not match")
    inspected = inspect_ltx25(
        source_path,
        source_lock=source_lock,
        require_source_lock=require_source_lock,
        builder_oracle_path=builder_oracle_path,
    )
    if loaded_inventory.get("source_sha256") != inspected.source_sha256:
        raise InventoryMismatchError("ltx25 inventory mismatch: canonical inventory source SHA-256 differs")
    if manifest.get("source_sha256") != inspected.source_sha256:
        raise ManifestError("ltx25 manifest-missing: source SHA-256 does not match admitted source")
    if policy_map.get("inventory_sha256") != inspected.inventory_sha256:
        raise InventoryMismatchError("ltx25 inventory mismatch: map does not match admitted source")
    if manifest.get("inventory_sha256") != inspected.inventory_sha256:
        raise ManifestError("ltx25 manifest-missing: inventory SHA-256 does not match admitted source")
    if loaded_inventory.get("inventory_sha256") != inspected.inventory_sha256:
        raise InventoryMismatchError("ltx25 inventory mismatch: canonical inventory differs from admitted source")
    with _SafetensorsRaw(Path(source_path)) as metadata_reader:
        source_metadata = metadata_reader.metadata()
    _verify_output(
        output,
        records,
        inspected.config_text,
        source_path=Path(source_path),
        inventory=inspected,
        expected_gemma_source_checkpoint=gemma_source_checkpoint_from_metadata(source_metadata),
    )
    return manifest


def source_lock_from_config(config: dict[str, Any]) -> SourceLock:
    """Read the frozen LTX 2.5 source lock table without affecting legacy config."""
    profile = config.get("profiles", {}).get("ltx25", {})
    if not isinstance(profile, dict):
        return DEFAULT_SOURCE_LOCK
    return SourceLock(
        repo_id=str(profile.get("source_repo_id", DEFAULT_SOURCE_LOCK.repo_id)),
        artifact_revision=str(profile.get("artifact_revision", DEFAULT_SOURCE_LOCK.artifact_revision)),
        filename=str(profile.get("source_filename", DEFAULT_SOURCE_LOCK.filename)),
        expected_size=profile.get("source_expected_size", DEFAULT_SOURCE_LOCK.expected_size),
        source_sha256=str(profile.get("source_sha256", SOURCE_LOCK_UNCONFIRMED)),
        lfs_sha256=str(profile.get("lfs_sha256", SOURCE_LOCK_UNCONFIRMED)),
        git_blob_oid=str(profile.get("git_blob_oid", DEFAULT_SOURCE_LOCK.git_blob_oid)),
        gated=str(profile.get("gated", DEFAULT_SOURCE_LOCK.gated)),
        license_id=str(profile.get("license_id", DEFAULT_SOURCE_LOCK.license_id)),
    )


def default_paths_from_config(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """Resolve only ltx25 paths; legacy config resolution is intentionally untouched."""
    profile = config.get("profiles", {}).get("ltx25", {})
    if not isinstance(profile, dict):
        profile = {}
    local_dir = project_root / str(profile.get("source_local_dir", ".artifacts/official-ltx25"))
    filename = str(profile.get("source_filename", DEFAULT_SOURCE_LOCK.filename))
    return {
        "local_dir": local_dir,
        "source": local_dir / filename,
        "oracle": project_root / str(profile.get("builder_oracle_path", "typemap/ltx25_builder_oracle.json")),
        "inventory": project_root / str(profile.get("inventory_path", "typemap/ltx25_inventory.json")),
        "map": project_root / str(profile.get("map_path", "typemap/ltx25_conversion_map.json")),
        "draft_map": project_root
        / str(profile.get("draft_map_path", "typemap/ltx25_conversion_map.draft.json")),
        "output": project_root / str(profile.get("output_dir", "output"))
        / str(profile.get("output_filename", "LTX-2.5-22B-distilled-transformer-Q4_K_M.gguf")),
    }
