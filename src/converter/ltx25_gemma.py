"""LTX-fine-tuned Gemma 4 text-encoder safetensors -> GGUF conversion (gemma4-ltx25).

A second, independent opt-in profile alongside ``ltx25`` (see that module's
docstring for the LTX 2.5 *transformer*). This module converts a different
source file entirely: the standalone Gemma 4 text-encoder checkpoint bundled
by the same LTX 2.5 release. See ``Docs/LTX25_CONVERSION_MODE_SPEC.md``'s
"gemma4-ltx25 profile" section for the full contract this module implements.

Unlike ``ltx25.py``, there is no emit/exclude component routing here: the
source file bundles nothing outside scope, so every one of its 686 tensors is
emitted verbatim under its raw safetensors key -- no prefix strip, no
renaming. The stages are the same four as ``ltx25.py``::

    inspect -> build-policy -> convert -> self-verify

Reader/writer/streaming primitives (``_SafetensorsRaw``, ``_register_tensor_info``,
``_tensor_payload``, the safetensors header validator, atomic-write/disk-preflight
helpers) are reused from :mod:`converter.convert` and :mod:`converter.ltx25`
rather than reimplemented; only the things that are genuinely different for this
profile -- source lock, config-driven E2 derivation, the ordered E3/E4
classification rule table, and the KV set -- live here.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType

from . import ltx25
from .convert import ARCH, _SafetensorsRaw, _byte_shape_and_nbytes, _register_tensor_info, _tensor_payload
from .quant_kernels import validate_quant_workers


PROFILE_ID = "gemma4-ltx25"
GEMMA_CONFIG_METADATA_KEY = "gemma_config"
OFFICIAL_CODE_COMMIT = ltx25.OFFICIAL_CODE_COMMIT
EXPECTED_GEMMA_VERSION = "gemma4-12b-ltx-v1"

ORACLE_FORMAT = "nz-gemma4-ltx25-builder-oracle-v1"
INVENTORY_FORMAT = "nz-gemma4-ltx25-inventory-v1"
MAP_FORMAT = "nz-gemma4-ltx25-conversion-map-v1"
MANIFEST_FORMAT = "nz-gemma4-ltx25-manifest-v1"

_ALLOWED_SOURCE_DTYPES = {"BF16", "U8"}
_ALLOWED_TARGET_TYPES = {"BF16", "Q4_K", "Q6_K", "I8"}
EXPECTED_TOTAL = 686
EXPECTED_TYPE_COUNTS = {"BF16": 351, "I8": 5, "Q4_K": 328, "Q6_K": 2}

# ---------------------------------------------------------------------------
# E1: source lock (E1 SourceLock table values transcribed from
# .artifacts/official-ltx25/download-5files-report.json's text-encoder row).
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_LOCK = ltx25.SourceLock(
    repo_id="Lightricks/LTX-2.5",
    artifact_revision="dd53cc2cd45bbeaa3563dfb575cba3f49cf44761",
    filename="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    expected_size=26_263_860_594,
    source_sha256="1c647a94c0e902fb87f9a403cbca36a8b6d8e5867094442df1b41ae557cfd1c6",
    lfs_sha256="1c647a94c0e902fb87f9a403cbca36a8b6d8e5867094442df1b41ae557cfd1c6",
    # Not captured by download-5files-report.json (a SHA-256-only report); the
    # SourceLock.complete gate below never inspects this field.
    git_blob_oid="",
)


# ---------------------------------------------------------------------------
# errors -- all subclass ltx25.Ltx25Error so cli.py's existing
# ``except ltx25_mod.Ltx25Error`` handler catches them without modification.
# ---------------------------------------------------------------------------
class Gemma4Error(ltx25.Ltx25Error):
    """Base error for expected gemma4-ltx25 conversion failures."""


class Gemma4SourceLockIncompleteError(Gemma4Error):
    """The official gated artifact has not supplied a trusted digest yet."""


class Gemma4SourceRejectedError(Gemma4Error):
    """The source does not match the gemma4-ltx25 profile's admission rules."""


class Gemma4InventoryMismatchError(Gemma4Error):
    """The source/header inventory does not match an expected oracle/map."""


class Gemma4PolicyMapError(Gemma4Error):
    """The concrete conversion map is malformed or does not cover the source."""


class Gemma4OutputExistsError(Gemma4Error):
    """A final output exists and explicit replacement was not selected."""


class Gemma4DiskPreflightError(Gemma4Error):
    """The destination filesystem lacks room for a temporary GGUF."""


class Gemma4ManifestError(Gemma4Error):
    """The output GGUF was committed but the sidecar manifest is absent/bad."""


# ---------------------------------------------------------------------------
# E2: deterministic, torch-free key/shape derivation from the source file's
# own gemma_config.text_config, plus a small number of named constants
# carried over from the already-audited ltx25 (transformer) config.
# ---------------------------------------------------------------------------

# The seven backbone nn.Linear module-name suffixes eligible for Q4_K (see the
# ordered classification table below). ``v_proj`` is absent on full-attention
# layers (k=v shared -- attention_k_eq_v).
_BACKBONE_LINEAR_SUFFIXES = (
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.v_proj",
)
_BACKBONE_LINEAR_RE = re.compile(
    r"^model\.layers\.\d+\.(?:"
    r"mlp\.(?:down_proj|gate_proj|up_proj)"
    r"|self_attn\.(?:q_proj|k_proj|o_proj|v_proj)"
    r")\.weight$"
)

_AGGREGATE_Q6K_NAMES = (
    "text_embedding_projection.video_aggregate_embed.weight",
    "text_embedding_projection.audio_aggregate_embed.weight",
)

# Sidecar non-tensor payloads: tokenizer/config JSON and the chat-template text,
# stored as raw U8 byte blobs (not numeric tensors). All five are JSON except
# the .jinja template.
_SIDECAR_U8_JSON_NAMES = (
    "hf_asset__generation_config.json",
    "hf_asset__processor_config.json",
    "hf_asset__tokenizer_config.json",
    "tokenizer_json",
)
_SIDECAR_U8_TEXT_ONLY_NAMES = ("hf_asset__chat_template.jinja",)
_SIDECAR_U8_NAMES = _SIDECAR_U8_JSON_NAMES + _SIDECAR_U8_TEXT_ONLY_NAMES

# Exact byte sizes of the five sidecar payloads, as authenticated in the
# pinned E1 source (.artifacts/official-ltx25/text_encoders/gemma4-12b-with-
# proj-ltx-2.5-bf16.safetensors header, read 2026-08-21). Fixed release-snapshot
# facts, not derivable from gemma_config.
_SIDECAR_U8_SIZES = {
    "hf_asset__chat_template.jinja": 18683,
    "hf_asset__generation_config.json": 255,
    "hf_asset__processor_config.json": 1382,
    "hf_asset__tokenizer_config.json": 3736,
    "tokenizer_json": 32169626,
}

# video_inner_dim = num_attention_heads * attention_head_dim and
# audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim, per
# the official ltx-core encoder_configurator.py's _create_feature_extractor
# (packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py,
# ~lines 203-208 at the pinned OFFICIAL_CODE_COMMIT). These two values are the
# LTX 2.5 *transformer*'s own config fields (video: heads=32, head_dim=128;
# audio: heads=32, head_dim=64), already independently authenticated by the
# ltx25 profile's own E1/E3 (see Docs/LTX25_CONVERSION_MODE_SPEC.md's "Video
# dimensions" / "Audio dimensions" rows: cross-attention dim 4096 / 2048).
_VIDEO_AGGREGATE_OUT_DIM = 32 * 128  # 4096
_AUDIO_AGGREGATE_OUT_DIM = 32 * 64  # 2048

# Vision-tower internal dims not exposed by gemma_config (patch-conv fan-in
# geometry); pinned to the authenticated E3 header as fixed known-good
# checkpoint facts. E2 still independently catches a name/dtype/rank/other-
# shape drift; only this one geometric constant is not re-derived from config.
_VISION_PATCH_FAN_IN = 6912


class Gemma4InventoryDerivationError(Gemma4Error):
    """gemma_config.text_config is missing a field this derivation needs."""


def _config_field(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise Gemma4InventoryDerivationError(
            f"gemma4-ltx25 builder-oracle-missing: gemma_config is missing required field {key!r}"
        )
    return config[key]


def derive_backbone_entries(text_config: dict[str, Any]) -> dict[str, tuple[str, list[int]]]:
    """Derive the 664 backbone tensor name -> (dtype, shape) pairs from text_config.

    Formulas (confirmed against the authenticated E3 header, 2026-08-21):
    q_proj out = num_attention_heads * layer_head_dim; k_proj out =
    layer_kv_heads * layer_head_dim; o_proj in = q_proj out, out = hidden_size;
    v_proj (sliding layers only, sharing k_proj's shape) is omitted on
    full-attention layers because attention_k_eq_v shares k as v there.
    """
    if _config_field(text_config, "attention_k_eq_v") is not True:
        raise Gemma4InventoryDerivationError(
            "gemma4-ltx25 builder-oracle-missing: derivation assumes attention_k_eq_v=true "
            "(full-attention layers share k as v and omit v_proj)"
        )
    layer_types = _config_field(text_config, "layer_types")
    num_hidden_layers = _config_field(text_config, "num_hidden_layers")
    if not isinstance(layer_types, list) or len(layer_types) != num_hidden_layers:
        raise Gemma4InventoryDerivationError(
            "gemma4-ltx25 builder-oracle-missing: layer_types length must equal num_hidden_layers"
        )
    hidden_size = _config_field(text_config, "hidden_size")
    intermediate_size = _config_field(text_config, "intermediate_size")
    head_dim = _config_field(text_config, "head_dim")
    global_head_dim = _config_field(text_config, "global_head_dim")
    num_attention_heads = _config_field(text_config, "num_attention_heads")
    num_key_value_heads = _config_field(text_config, "num_key_value_heads")
    num_global_key_value_heads = _config_field(text_config, "num_global_key_value_heads")

    entries: dict[str, tuple[str, list[int]]] = {}
    for index, layer_type in enumerate(layer_types):
        is_full = layer_type == "full_attention"
        if not is_full and layer_type != "sliding_attention":
            raise Gemma4InventoryDerivationError(
                f"gemma4-ltx25 builder-oracle-missing: unknown layer_types[{index}]={layer_type!r}"
            )
        layer_head_dim = global_head_dim if is_full else head_dim
        kv_heads = num_global_key_value_heads if is_full else num_key_value_heads
        q_out = num_attention_heads * layer_head_dim
        kv_out = kv_heads * layer_head_dim
        prefix = f"model.layers.{index}."
        entries[prefix + "input_layernorm.weight"] = ("BF16", [hidden_size])
        entries[prefix + "layer_scalar"] = ("BF16", [1])
        entries[prefix + "mlp.down_proj.weight"] = ("BF16", [hidden_size, intermediate_size])
        entries[prefix + "mlp.gate_proj.weight"] = ("BF16", [intermediate_size, hidden_size])
        entries[prefix + "mlp.up_proj.weight"] = ("BF16", [intermediate_size, hidden_size])
        entries[prefix + "post_attention_layernorm.weight"] = ("BF16", [hidden_size])
        entries[prefix + "post_feedforward_layernorm.weight"] = ("BF16", [hidden_size])
        entries[prefix + "pre_feedforward_layernorm.weight"] = ("BF16", [hidden_size])
        entries[prefix + "self_attn.k_norm.weight"] = ("BF16", [layer_head_dim])
        entries[prefix + "self_attn.k_proj.weight"] = ("BF16", [kv_out, hidden_size])
        entries[prefix + "self_attn.o_proj.weight"] = ("BF16", [hidden_size, q_out])
        entries[prefix + "self_attn.q_norm.weight"] = ("BF16", [layer_head_dim])
        entries[prefix + "self_attn.q_proj.weight"] = ("BF16", [q_out, hidden_size])
        if not is_full:
            entries[prefix + "self_attn.v_proj.weight"] = ("BF16", [kv_out, hidden_size])
    return entries


def derive_aggregate_entries(text_config: dict[str, Any]) -> dict[str, tuple[str, list[int]]]:
    """Derive the 2 aggregate_embed weights + 2 biases (188,160-wide flattened features)."""
    hidden_size = _config_field(text_config, "hidden_size")
    num_hidden_layers = _config_field(text_config, "num_hidden_layers")
    flat_dim = hidden_size * (num_hidden_layers + 1)  # +1: the embedding-layer output
    return {
        "text_embedding_projection.video_aggregate_embed.weight": (
            "BF16",
            [_VIDEO_AGGREGATE_OUT_DIM, flat_dim],
        ),
        "text_embedding_projection.video_aggregate_embed.bias": ("BF16", [_VIDEO_AGGREGATE_OUT_DIM]),
        "text_embedding_projection.audio_aggregate_embed.weight": (
            "BF16",
            [_AUDIO_AGGREGATE_OUT_DIM, flat_dim],
        ),
        "text_embedding_projection.audio_aggregate_embed.bias": ("BF16", [_AUDIO_AGGREGATE_OUT_DIM]),
    }


def derive_fixed_entries(
    text_config: dict[str, Any],
    vision_config: dict[str, Any],
    audio_config: dict[str, Any],
) -> dict[str, tuple[str, list[int]]]:
    """The remaining 20 non-backbone, non-aggregate rows.

    embed_tokens/norm/projectors are derived from config fields; the
    nine-row vision tower's internal geometry is pinned (see
    ``_VISION_PATCH_FAN_IN``); the five sidecars are U8 byte blobs of a
    fixed, authenticated size.
    """
    hidden_size = _config_field(text_config, "hidden_size")
    vocab_size = _config_field(text_config, "vocab_size")
    audio_embed_dim = _config_field(audio_config, "audio_embed_dim")
    mm_embed_dim = _config_field(vision_config, "mm_embed_dim")
    mm_posemb_size = _config_field(vision_config, "mm_posemb_size")
    entries: dict[str, tuple[str, list[int]]] = {
        "model.embed_tokens.weight": ("BF16", [vocab_size, hidden_size]),
        "model.norm.weight": ("BF16", [hidden_size]),
        "audio_projector.embedding_projection.weight": ("BF16", [hidden_size, audio_embed_dim]),
        "multi_modal_projector.embedding_projection.weight": ("BF16", [hidden_size, mm_embed_dim]),
        "vision_model.patch_dense.weight": ("BF16", [mm_embed_dim, _VISION_PATCH_FAN_IN]),
        "vision_model.patch_dense.bias": ("BF16", [mm_embed_dim]),
        "vision_model.patch_ln1.weight": ("BF16", [_VISION_PATCH_FAN_IN]),
        "vision_model.patch_ln1.bias": ("BF16", [_VISION_PATCH_FAN_IN]),
        "vision_model.patch_ln2.weight": ("BF16", [mm_embed_dim]),
        "vision_model.patch_ln2.bias": ("BF16", [mm_embed_dim]),
        "vision_model.pos_embedding": ("BF16", [mm_posemb_size, 2, mm_embed_dim]),
        "vision_model.pos_norm.weight": ("BF16", [mm_embed_dim]),
        "vision_model.pos_norm.bias": ("BF16", [mm_embed_dim]),
    }
    for name, size in _SIDECAR_U8_SIZES.items():
        entries[name] = ("U8", [size])
    return entries


def derive_expected_tensor_table(parsed_config: dict[str, Any]) -> dict[str, tuple[str, list[int]]]:
    """Derive all 686 name -> (dtype, shape) pairs from a parsed gemma_config."""
    text_config = _config_field(parsed_config, "text_config")
    vision_config = _config_field(parsed_config, "vision_config")
    audio_config = _config_field(parsed_config, "audio_config")
    table: dict[str, tuple[str, list[int]]] = {}
    table.update(derive_backbone_entries(text_config))
    table.update(derive_aggregate_entries(text_config))
    table.update(derive_fixed_entries(text_config, vision_config, audio_config))
    return table


def build_builder_oracle_payload(config_text: str) -> dict[str, Any]:
    """Build the checked-in E2 oracle payload from the source's own gemma_config text.

    Pure/deterministic: no Torch, no file I/O beyond what the caller already
    did to obtain ``config_text``. Raises :class:`Gemma4InventoryDerivationError`
    if a required config field is absent.
    """
    parsed = json.loads(config_text)
    table = derive_expected_tensor_table(parsed)
    tensors = {name: {"dtype": dtype, "shape": list(shape)} for name, (dtype, shape) in sorted(table.items())}
    payload: dict[str, Any] = {
        "format": ORACLE_FORMAT,
        "profile": PROFILE_ID,
        "official_code_commit": OFFICIAL_CODE_COMMIT,
        "config_bytes_sha256": ltx25._sha256_bytes(config_text.encode("utf-8")),
        "tensors": tensors,
    }
    payload["oracle_sha256"] = ltx25._sha256_bytes(ltx25._canonical_json_bytes(payload))
    return payload


def load_builder_oracle(path: str | Path) -> dict[str, Any]:
    """Load and validate the checked-in E2 oracle artifact."""
    payload = ltx25._read_json(path)
    if payload.get("format") != ORACLE_FORMAT or payload.get("profile") != PROFILE_ID:
        raise Gemma4InventoryMismatchError(f"gemma4-ltx25 builder-oracle-missing: unsupported oracle file {path}")
    if payload.get("official_code_commit") != OFFICIAL_CODE_COMMIT:
        raise Gemma4InventoryMismatchError(
            "gemma4-ltx25 builder-oracle-missing: oracle does not use the pinned official code commit"
        )
    config_sha = payload.get("config_bytes_sha256")
    if not isinstance(config_sha, str) or len(config_sha) != 64:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 builder-oracle-missing: config bytes SHA-256 is invalid")
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 builder-oracle-missing: tensors table is missing")
    normalized: dict[str, dict[str, Any]] = {}
    for name, row in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict):
            raise Gemma4InventoryMismatchError(f"gemma4-ltx25 builder-oracle-missing: invalid tensor row {name!r}")
        dtype = row.get("dtype")
        shape = row.get("shape")
        if dtype not in _ALLOWED_SOURCE_DTYPES or not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise Gemma4InventoryMismatchError(f"gemma4-ltx25 builder-oracle-missing: invalid row for {name!r}")
        normalized[name] = {"dtype": dtype, "shape": list(shape)}
    recorded = payload.get("oracle_sha256")
    skeleton = dict(payload)
    skeleton.pop("oracle_sha256", None)
    actual = ltx25._sha256_bytes(ltx25._canonical_json_bytes(skeleton))
    if not isinstance(recorded, str) or recorded != actual:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 builder-oracle-missing: oracle SHA-256 is absent or incorrect")
    payload["tensors"] = normalized
    return payload


# ---------------------------------------------------------------------------
# E3: ordered, mutually-exclusive classification (used by both inspect's
# dtype-per-role check and build_policy_map's target-type assignment).
# ---------------------------------------------------------------------------
def classify_gemma4_tensor(name: str, shape: list[int]) -> tuple[str, str, str]:
    """Return (ggml_type, rule_id, reason) for one raw tensor name.

    Ordered, most-specific-first: three rows would otherwise multi-match a
    naive "2-D .weight with a 256-divisible last dim" Q4_K predicate --
    ``vision_model.patch_dense.weight`` (6912 = 27*256),
    ``multi_modal_projector.embedding_projection.weight`` (3840 = 15*256), and
    ``model.embed_tokens.weight`` (3840 = 15*256) -- so all three must be
    resolved by an earlier, more specific rule before the backbone rule runs.
    """
    if name in _SIDECAR_U8_NAMES:
        return (
            "I8",
            "i8-sidecar-passthrough",
            "Fixed non-tensor U8 sidecar payload; gguf-py has no U8 writer type, "
            "I8 is a byte-identical passthrough.",
        )
    if name.startswith("vision_model."):
        return (
            "BF16",
            "vision-tower-bf16",
            "Vision tower row kept BF16 (owner-adjudicated exclusion from the Q4_K "
            "backbone policy, not a K-alignment failure).",
        )
    if name.startswith("multi_modal_projector."):
        return (
            "BF16",
            "multimodal-projector-bf16",
            "Projector row kept BF16; not part of the reviewed backbone Linear "
            "allowlist even though its shape is K-alignment-eligible.",
        )
    if name.startswith("audio_projector."):
        return (
            "BF16",
            "audio-projector-bf16",
            "Projector's last dim (640) is not divisible by 256; K-quant is not possible.",
        )
    if name in _AGGREGATE_Q6K_NAMES:
        return (
            "Q6_K",
            "aggregate-embed-q6k",
            "Owner-adjudicated Q6_K: measured rel-RMSE 0.0173 (vs. 0.0656 for Q4_K); "
            "an nn.Linear weight, so this is real VRAM savings.",
        )
    if name == "model.embed_tokens.weight":
        return (
            "BF16",
            "embed-tokens-bf16-backend-expansion",
            "Backend always expands embeddings to BF16 at load; quantizing here saves no VRAM.",
        )
    if len(shape) == 2 and shape[-1] % 256 == 0 and _BACKBONE_LINEAR_RE.match(name):
        return (
            "Q4_K",
            "q4-k-backbone-linear",
            "Backbone nn.Linear weight: module-name allowlist match, rank-2, "
            "input width divisible by 256.",
        )
    return (
        "BF16",
        "no-rmsnorm-folding",
        "Norm/scalar/bias/q,k_norm row kept BF16 verbatim; this is not llama.cpp-style "
        "RMSNorm, its (1+w) fold must never be applied here.",
    )


# ---------------------------------------------------------------------------
# E3: inventory (header admission)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Inventory:
    """Validated gemma4-ltx25 header inventory used by the concrete policy map."""

    source_path: Path
    source_sha256: str
    source_size: int
    config_text: str
    config_bytes_sha256: str
    gemma_version: str
    builder_oracle_sha256: str | None
    tensors: tuple[dict[str, Any], ...]
    inventory_sha256: str


def _require_source_lock(path: Path, source_lock: ltx25.SourceLock, require_complete: bool) -> str:
    if require_complete and not source_lock.complete:
        raise Gemma4SourceLockIncompleteError(
            "gemma4-ltx25 source-lock-incomplete: authenticated official source SHA-256/LFS hash "
            "is required before source admission"
        )
    if source_lock.expected_size is not None and path.stat().st_size != source_lock.expected_size:
        raise Gemma4SourceRejectedError(
            f"gemma4-ltx25 source rejected: size {path.stat().st_size} != pinned {source_lock.expected_size}"
        )
    actual_sha = ltx25.sha256_of_file(path)
    if source_lock.source_sha256 and source_lock.source_sha256 != ltx25.SOURCE_LOCK_UNCONFIRMED:
        if actual_sha.lower() != source_lock.source_sha256.lower():
            raise Gemma4SourceRejectedError(
                f"gemma4-ltx25 source rejected: SHA-256 {actual_sha} != pinned {source_lock.source_sha256}"
            )
    if source_lock.lfs_sha256 and source_lock.lfs_sha256 != ltx25.SOURCE_LOCK_UNCONFIRMED:
        if actual_sha.lower() != source_lock.lfs_sha256.lower():
            raise Gemma4SourceRejectedError(
                f"gemma4-ltx25 source rejected: SHA-256 {actual_sha} != pinned LFS/Xet {source_lock.lfs_sha256}"
            )
    return actual_sha


def _gemma_config_from_header(header: dict[str, Any]) -> tuple[str, str]:
    """Return (config_text, gemma_version) after validating __metadata__.gemma_config."""
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise Gemma4SourceRejectedError("gemma4-ltx25 source rejected: __metadata__ is required")
    config_text = metadata.get(GEMMA_CONFIG_METADATA_KEY)
    if not isinstance(config_text, str) or not config_text:
        raise Gemma4SourceRejectedError(
            f"gemma4-ltx25 source rejected: __metadata__.{GEMMA_CONFIG_METADATA_KEY} "
            "must be a non-empty JSON string"
        )
    try:
        parsed = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise Gemma4SourceRejectedError(f"gemma4-ltx25 source rejected: gemma_config is not JSON ({exc})") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("text_config"), dict):
        raise Gemma4SourceRejectedError(
            "gemma4-ltx25 source rejected: gemma_config must contain a top-level text_config object"
        )
    gemma_version = parsed.get("gemma_version")
    if not isinstance(gemma_version, str) or not gemma_version:
        raise Gemma4SourceRejectedError(
            "gemma4-ltx25 source rejected: gemma_config.gemma_version must be a non-empty string"
        )
    if gemma_version != EXPECTED_GEMMA_VERSION:
        raise Gemma4SourceRejectedError(
            f"gemma4-ltx25 source rejected: gemma_config.gemma_version={gemma_version!r} != "
            f"expected {EXPECTED_GEMMA_VERSION!r}"
        )
    return config_text, gemma_version


def inspect_gemma4(
    source_path: str | Path,
    *,
    source_lock: ltx25.SourceLock = DEFAULT_SOURCE_LOCK,
    require_source_lock: bool = True,
    builder_oracle_path: str | Path | None = None,
    require_builder_oracle: bool = True,
    audit_path: str | Path | None = None,
) -> Inventory:
    """Perform E3-style header admission and return the canonical inventory."""
    source = Path(source_path)
    if require_source_lock and not source_lock.complete:
        raise Gemma4SourceLockIncompleteError(
            "gemma4-ltx25 source-lock-incomplete: authenticated official source SHA-256/LFS hash "
            "is required before conversion"
        )
    if not source.is_file():
        raise FileNotFoundError(f"gemma4-ltx25 source not found: {source}")
    source_sha = _require_source_lock(source, source_lock, require_source_lock)
    header, payload_base, source_size = ltx25._read_validated_header(source)
    config_text, gemma_version = _gemma_config_from_header(header)
    config_bytes = config_text.encode("utf-8")

    expected: dict[str, dict[str, Any]] | None = None
    oracle_sha: str | None = None
    if builder_oracle_path is not None:
        oracle = load_builder_oracle(builder_oracle_path)
        if oracle["config_bytes_sha256"] != ltx25._sha256_bytes(config_bytes):
            raise Gemma4InventoryMismatchError(
                "gemma4-ltx25 inventory mismatch: builder oracle config digest differs from source metadata"
            )
        expected = oracle["tensors"]
        oracle_sha = oracle["oracle_sha256"]
    elif require_builder_oracle:
        raise Gemma4InventoryMismatchError(
            "gemma4-ltx25 builder-oracle-missing: generate the E2 builder oracle before inspection"
        )

    tensor_headers = ltx25._validate_tensor_entries(header, source_size - payload_base)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in tensor_headers:
        name = entry.raw_key
        if name in seen:
            raise Gemma4InventoryMismatchError(f"gemma4-ltx25 inventory mismatch: duplicate key {name!r}")
        seen.add(name)
        is_sidecar = name in _SIDECAR_U8_NAMES
        expected_dtype = "U8" if is_sidecar else "BF16"
        if entry.source_dtype != expected_dtype:
            raise Gemma4SourceRejectedError(
                f"gemma4-ltx25 source rejected: {name!r} has dtype {entry.source_dtype!r}, "
                f"expected {expected_dtype!r} (E3 allowlist {{BF16, U8}}, role-pinned)"
            )
        records.append(
            {
                "raw_key": name,
                "source_dtype": entry.source_dtype,
                "shape_logical": list(entry.shape_logical),
                "data_offsets": list(entry.data_offsets),
            }
        )

    if expected is not None:
        actual = {record["raw_key"]: record for record in records}
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatches = [
            name
            for name in sorted(set(expected) & set(actual))
            if expected[name]["dtype"] != actual[name]["source_dtype"]
            or list(expected[name]["shape"]) != list(actual[name]["shape_logical"])
        ]
        if missing or extra or mismatches:
            parts: list[str] = []
            if missing:
                parts.append(f"missing={len(missing)} {missing[:20]!r}")
            if extra:
                parts.append(f"extra={len(extra)} {extra[:20]!r}")
            if mismatches:
                parts.append(f"dtype_or_shape_mismatch={len(mismatches)} {mismatches[:20]!r}")
            raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: " + "; ".join(parts))

    skeleton = {
        "format": INVENTORY_FORMAT,
        "profile": PROFILE_ID,
        "source_size": source_size,
        "source_sha256": source_sha,
        "config_bytes_sha256": ltx25._sha256_bytes(config_bytes),
        "gemma_version": gemma_version,
        "builder_oracle_sha256": oracle_sha,
        "tensors": sorted(records, key=lambda record: record["raw_key"]),
    }
    inventory_sha = ltx25._sha256_bytes(ltx25._canonical_json_bytes(skeleton))
    inventory = Inventory(
        source_path=source,
        source_sha256=source_sha,
        source_size=source_size,
        config_text=config_text,
        config_bytes_sha256=skeleton["config_bytes_sha256"],
        gemma_version=gemma_version,
        builder_oracle_sha256=oracle_sha,
        tensors=tuple(skeleton["tensors"]),
        inventory_sha256=inventory_sha,
    )
    if audit_path is not None:
        payload = dict(skeleton)
        payload["inventory_sha256"] = inventory_sha
        ltx25._write_json_atomic(Path(audit_path), payload)
    return inventory


def load_inventory(path: str | Path) -> dict[str, Any]:
    payload = ltx25._read_json(path)
    if payload.get("format") != INVENTORY_FORMAT or payload.get("profile") != PROFILE_ID:
        raise Gemma4InventoryMismatchError(f"gemma4-ltx25 inventory mismatch: unsupported inventory file {path}")
    recorded = payload.get("inventory_sha256")
    skeleton = dict(payload)
    skeleton.pop("inventory_sha256", None)
    actual = ltx25._sha256_bytes(ltx25._canonical_json_bytes(skeleton))
    if not isinstance(recorded, str) or recorded != actual:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: inventory SHA-256 is absent or incorrect")
    if not isinstance(payload.get("tensors"), list) or not payload["tensors"]:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: tensors must be a non-empty list")
    for field in ("source_sha256", "config_bytes_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise Gemma4InventoryMismatchError(f"gemma4-ltx25 inventory mismatch: {field} is invalid")
    if not isinstance(payload.get("gemma_version"), str) or not payload["gemma_version"]:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: gemma_version is invalid")
    raw_keys: set[str] = set()
    for record in payload["tensors"]:
        if not isinstance(record, dict):
            raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: tensor record must be an object")
        raw_key = record.get("raw_key")
        shape = record.get("shape_logical")
        offsets = record.get("data_offsets")
        if not isinstance(raw_key, str) or not raw_key or raw_key in raw_keys:
            raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: duplicate or invalid raw key")
        raw_keys.add(raw_key)
        if record.get("source_dtype") not in _ALLOWED_SOURCE_DTYPES:
            raise Gemma4InventoryMismatchError(f"gemma4-ltx25 inventory mismatch: {raw_key!r} has invalid dtype")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise Gemma4InventoryMismatchError(f"gemma4-ltx25 inventory mismatch: {raw_key!r} has invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload["source_size"]
        ):
            raise Gemma4InventoryMismatchError(f"gemma4-ltx25 inventory mismatch: {raw_key!r} has invalid offsets")
    return payload


# ---------------------------------------------------------------------------
# E4: concrete conversion map
# ---------------------------------------------------------------------------
def _map_nbytes(shape_logical: list[int], ggml_type: str) -> int:
    if ggml_type == "BF16":
        return math.prod(shape_logical) * 2
    if ggml_type == "I8":
        return math.prod(shape_logical) * 1
    try:
        _, nbytes = _byte_shape_and_nbytes(shape_logical, ggml_type)
    except (KeyError, TypeError, ValueError) as exc:
        raise Gemma4PolicyMapError(
            f"gemma4-ltx25 map mismatch: cannot calculate {ggml_type} bytes for shape {shape_logical!r}"
        ) from exc
    return nbytes


def _type_counts(records: Any) -> dict[str, int]:
    return dict(sorted(Counter(record["ggml_type"] for record in records).items()))


def build_policy_map(inventory_path: str | Path, map_path: str | Path) -> dict[str, Any]:
    """Create the review-only concrete E4 draft from the authenticated E3 rows.

    Every row's target type/rule_id/reason is decided by the ordered
    :func:`classify_gemma4_tensor` table. A separate review must promote a
    checked-in E4 map before ``convert`` will accept it (see ``status``).
    """
    inventory = load_inventory(inventory_path)
    rows: list[dict[str, Any]] = []
    for record in inventory["tensors"]:
        name = record["raw_key"]
        dtype = record["source_dtype"]
        shape = list(record["shape_logical"])
        ggml_type, rule_id, reason = classify_gemma4_tensor(name, shape)
        if ggml_type == "I8" and dtype != "U8":
            raise Gemma4InventoryMismatchError(
                f"gemma4-ltx25 inventory mismatch: {name} classified I8 but source dtype is {dtype!r}"
            )
        if ggml_type != "I8" and dtype != "BF16":
            raise Gemma4InventoryMismatchError(
                f"gemma4-ltx25 inventory mismatch: {name} classified {ggml_type} but source dtype is {dtype!r}"
            )
        rows.append(
            {
                "name": name,
                "source_dtype": dtype,
                "shape_logical": shape,
                "shape_gguf": list(reversed(shape)),
                "ggml_type": ggml_type,
                "nbytes": _map_nbytes(shape, ggml_type),
                "rule_id": rule_id,
                "reason": reason,
                "evidence_ids": ["E3", "E4"],
            }
        )
    if not rows:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: no tensors")
    type_counts = _type_counts(rows)
    if len(rows) == EXPECTED_TOTAL and type_counts != EXPECTED_TYPE_COUNTS:
        raise Gemma4PolicyMapError(
            f"gemma4-ltx25 map mismatch: type_counts {type_counts} != expected {EXPECTED_TYPE_COUNTS} "
            f"for the full {EXPECTED_TOTAL}-tensor bundle"
        )
    payload: dict[str, Any] = {
        "format": MAP_FORMAT,
        "profile": PROFILE_ID,
        "status": "draft",
        "inventory_sha256": inventory["inventory_sha256"],
        "tensors": rows,
        "type_counts": type_counts,
    }
    payload["map_sha256"] = ltx25._sha256_bytes(ltx25._canonical_json_bytes(payload))
    ltx25._write_json_atomic(Path(map_path), payload)
    return payload


def load_policy_map(path: str | Path, *, require_approved: bool = True) -> dict[str, Any]:
    payload = ltx25._read_json(path)
    if payload.get("format") != MAP_FORMAT or payload.get("profile") != PROFILE_ID:
        raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: unsupported map file {path}")
    if require_approved and payload.get("status") != "approved":
        raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: conversion requires an approved concrete map")
    if payload.get("status") not in {"draft", "approved"}:
        raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: status must be draft or approved")
    inventory_sha = payload.get("inventory_sha256")
    if not isinstance(inventory_sha, str) or len(inventory_sha) != 64:
        raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: inventory SHA-256 is invalid")
    recorded = payload.get("map_sha256")
    skeleton = dict(payload)
    skeleton.pop("map_sha256", None)
    actual = ltx25._sha256_bytes(ltx25._canonical_json_bytes(skeleton))
    if not isinstance(recorded, str) or recorded != actual:
        raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: map SHA-256 is absent or incorrect")
    rows = payload.get("tensors")
    if not isinstance(rows, list) or not rows:
        raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: tensors must be a non-empty list")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: tensor row must be an object")
        name = row.get("name")
        dtype = row.get("source_dtype")
        shape = row.get("shape_logical")
        ggml_type = row.get("ggml_type")
        if not isinstance(name, str) or name in names:
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: duplicate or invalid tensor name {name!r}")
        names.add(name)
        if dtype not in _ALLOWED_SOURCE_DTYPES:
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} source dtype must be BF16 or U8")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} has invalid shape")
        if row.get("shape_gguf") != list(reversed(shape)):
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} has wrong GGUF shape")
        if ggml_type not in _ALLOWED_TARGET_TYPES:
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} uses unsupported type {ggml_type!r}")
        if ggml_type == "I8" and dtype != "U8":
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} I8 target requires U8 source")
        if ggml_type != "I8" and dtype != "BF16":
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} non-I8 target requires BF16 source")
        if ggml_type.startswith("Q") and (not shape or shape[-1] % 256):
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} K-quant last dimension is not divisible by 256")
        if not isinstance(row.get("nbytes"), int) or isinstance(row["nbytes"], bool) or row["nbytes"] < 0:
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} has invalid nbytes")
        expected_nbytes = _map_nbytes(shape, ggml_type)
        if row.get("nbytes") != expected_nbytes:
            raise Gemma4PolicyMapError(
                f"gemma4-ltx25 map mismatch: {name} nbytes={row.get('nbytes')} != expected {expected_nbytes}"
            )
        if not isinstance(row.get("rule_id"), str) or not row["rule_id"].strip():
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} has an empty rule_id")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} has an empty reason")
        evidence_ids = row.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids or any(
            not isinstance(item, str) or not item.startswith("E") or not item[1:].isdigit()
            for item in evidence_ids
        ):
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} has invalid evidence_ids")
    expected_type_counts = _type_counts(rows)
    if payload.get("type_counts") != expected_type_counts:
        raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: type_counts does not match tensor rows")
    return payload


def _records_for_inventory_and_map(inventory: Inventory, policy_map: dict[str, Any]) -> list[dict[str, Any]]:
    source_by_key = {record["raw_key"]: record for record in inventory.tensors}
    map_rows = policy_map["tensors"]
    map_by_key = {row["name"]: row for row in map_rows}
    if len(map_by_key) != len(map_rows):
        raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: duplicate map names")
    missing = sorted(set(source_by_key) - set(map_by_key))
    extra = sorted(set(map_by_key) - set(source_by_key))
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={len(missing)} {missing[:20]!r}")
        if extra:
            parts.append(f"extra={len(extra)} {extra[:20]!r}")
        raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: " + "; ".join(parts))
    records: list[dict[str, Any]] = []
    for name in sorted(source_by_key):
        source = source_by_key[name]
        row = map_by_key[name]
        if row["source_dtype"] != source["source_dtype"] or row["shape_logical"] != source["shape_logical"]:
            raise Gemma4PolicyMapError(f"gemma4-ltx25 map mismatch: {name} dtype or shape differs from inventory")
        records.append(dict(row))
    return records


# ---------------------------------------------------------------------------
# writer / KV
# ---------------------------------------------------------------------------
def _build_writer(records: list[dict[str, Any]], config_text: str, gemma_version: str, path: str | Path | None) -> GGUFWriter:
    writer = GGUFWriter(str(path) if path is not None else None, arch=ARCH)
    # general.architecture is written by GGUFWriter.__init__ from arch=.
    writer.add_string("config", config_text)
    writer.add_string("ltx.component", "text_encoder")
    writer.add_string("ltx.text_encoder.gemma_version", gemma_version)
    for record in records:
        _register_tensor_info(writer, record["name"], record)
    return writer


def estimate_gguf_size(records: list[dict[str, Any]], config_text: str, gemma_version: str) -> int:
    """Calculate the exact one-shard GGUF byte count for this writer/version."""
    writer = _build_writer(records, config_text, gemma_version, path=None)
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


# ---------------------------------------------------------------------------
# E5: self-verification
# ---------------------------------------------------------------------------
_RAW_COMPARE_CHUNK_BYTES = 4 * 1024 * 1024


def _verify_raw_passthrough(
    reader: GGUFReader,
    source_path: Path,
    inventory: Inventory,
    records: list[dict[str, Any]],
    *,
    ggml_type_name: str,
    chunk_bytes: int = _RAW_COMPARE_CHUNK_BYTES,
) -> None:
    """Byte-compare every emitted row of one target type against source bytes.

    Used both for the 351 BF16 rows (verbatim copy, since every BF16-target
    row also has a BF16 source) and the 5 I8 rows (verbatim U8 passthrough).
    """
    names = {record["name"] for record in records if record["ggml_type"] == ggml_type_name}
    if not names:
        return
    source_by_name = {record["raw_key"]: record for record in inventory.tensors}
    output_by_name = {tensor.name: tensor for tensor in reader.tensors}
    _, payload_base, _ = ltx25._read_validated_header(source_path)
    with source_path.open("rb") as source:
        for name in sorted(names):
            tensor = output_by_name.get(name)
            if tensor is None or tensor.tensor_type.name != ggml_type_name:
                raise Gemma4Error(
                    f"gemma4-ltx25 self-verify failed: {name} is absent or not {ggml_type_name} in GGUF"
                )
            output_bytes = np.asarray(tensor.data)
            if not output_bytes.flags.c_contiguous:
                raise Gemma4Error(f"gemma4-ltx25 self-verify failed: {name} payload is not contiguous")
            output_raw = output_bytes.view(np.uint8).reshape(-1)
            offsets = source_by_name[name]["data_offsets"]
            expected_nbytes = offsets[1] - offsets[0]
            if output_raw.nbytes != expected_nbytes:
                raise Gemma4Error(
                    f"gemma4-ltx25 self-verify failed: {name} payload bytes "
                    f"{output_raw.nbytes} != source {expected_nbytes}"
                )
            source.seek(payload_base + offsets[0])
            for start in range(0, expected_nbytes, chunk_bytes):
                count = min(chunk_bytes, expected_nbytes - start)
                source_chunk = source.read(count)
                if len(source_chunk) != count:
                    raise Gemma4Error(f"gemma4-ltx25 self-verify failed: {name} source payload is truncated")
                if not np.array_equal(output_raw[start : start + count], np.frombuffer(source_chunk, dtype=np.uint8)):
                    raise Gemma4Error(
                        f"gemma4-ltx25 self-verify failed: {name} raw payload differs from source at byte {start}"
                    )


def _verify_sidecar_text(reader: GGUFReader, records: list[dict[str, Any]]) -> None:
    """The 5 I8 sidecars must UTF-8 decode; the 4 *.json-ish ones must JSON-parse."""
    output_by_name = {tensor.name: tensor for tensor in reader.tensors}
    for record in records:
        if record["ggml_type"] != "I8":
            continue
        name = record["name"]
        tensor = output_by_name.get(name)
        if tensor is None:
            raise Gemma4Error(f"gemma4-ltx25 self-verify failed: sidecar {name} is absent in GGUF")
        raw_bytes = np.asarray(tensor.data).view(np.uint8).tobytes()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Gemma4Error(f"gemma4-ltx25 self-verify failed: sidecar {name} is not UTF-8 ({exc})") from exc
        if name in _SIDECAR_U8_JSON_NAMES:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise Gemma4Error(f"gemma4-ltx25 self-verify failed: sidecar {name} is not valid JSON ({exc})") from exc


def _verify_output(
    output_path: Path,
    records: list[dict[str, Any]],
    expected_config_text: str,
    *,
    source_path: Path,
    inventory: Inventory,
) -> None:
    reader = GGUFReader(str(output_path))
    if len(reader.tensors) != len(records):
        raise Gemma4Error(f"gemma4-ltx25 self-verify failed: tensor count {len(reader.tensors)} != {len(records)}")
    for index, (tensor, record) in enumerate(zip(reader.tensors, records)):
        if tensor.name != record["name"]:
            raise Gemma4Error(f"gemma4-ltx25 self-verify failed: tensor #{index} name {tensor.name!r} != {record['name']!r}")
        if tensor.tensor_type.name != record["ggml_type"]:
            raise Gemma4Error(
                f"gemma4-ltx25 self-verify failed: {tensor.name} type {tensor.tensor_type.name} != {record['ggml_type']}"
            )
        if [int(dim) for dim in tensor.shape] != list(record["shape_gguf"]):
            raise Gemma4Error(f"gemma4-ltx25 self-verify failed: {tensor.name} shape does not match map")
    config_field = reader.fields.get("config")
    if config_field is None:
        raise Gemma4Error("gemma4-ltx25 self-verify failed: output lacks config metadata")
    config_text = config_field.contents()
    if config_text.encode("utf-8") != expected_config_text.encode("utf-8"):
        raise Gemma4Error("gemma4-ltx25 self-verify failed: output config bytes differ from source metadata")
    for tensor in reader.tensors:
        if tensor.tensor_type.name.startswith("Q"):
            ltx25._verify_quant_tensor_finite(tensor)
    _verify_raw_passthrough(reader, source_path, inventory, records, ggml_type_name="BF16")
    _verify_raw_passthrough(reader, source_path, inventory, records, ggml_type_name="I8")
    _verify_sidecar_text(reader, records)


# ---------------------------------------------------------------------------
# convert / verify
# ---------------------------------------------------------------------------
def convert_gemma4(
    source_path: str | Path,
    policy_map_path: str | Path,
    output_path: str | Path,
    *,
    source_lock: ltx25.SourceLock = DEFAULT_SOURCE_LOCK,
    require_source_lock: bool = True,
    force: bool = False,
    builder_oracle_path: str | Path | None = None,
    require_builder_oracle: bool = True,
    quant_workers: int = 4,
) -> Path:
    """Convert an admitted gemma4-ltx25 source using an approved concrete map."""
    quant_workers = validate_quant_workers(quant_workers)
    source = Path(source_path)
    output = Path(output_path)
    if require_source_lock and not source_lock.complete:
        raise Gemma4SourceLockIncompleteError(
            "gemma4-ltx25 source-lock-incomplete: authenticated official source SHA-256/LFS hash "
            "is required before conversion"
        )
    policy_map = load_policy_map(policy_map_path, require_approved=True)
    inventory = inspect_gemma4(
        source,
        source_lock=source_lock,
        require_source_lock=require_source_lock,
        builder_oracle_path=builder_oracle_path,
        require_builder_oracle=require_builder_oracle,
    )
    if policy_map.get("inventory_sha256") != inventory.inventory_sha256:
        raise Gemma4PolicyMapError("gemma4-ltx25 map mismatch: map inventory hash differs from inspected source")
    records = _records_for_inventory_and_map(inventory, policy_map)
    if output.exists() and not force:
        raise Gemma4OutputExistsError(
            f"gemma4-ltx25 output exists: {output}. Re-run with --force to replace it after temporary verification."
        )
    with _SafetensorsRaw(source) as metadata_reader:
        metadata = metadata_reader.metadata()
    if metadata.get(GEMMA_CONFIG_METADATA_KEY) != inventory.config_text:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: source config changed after header admission")
    required_bytes = estimate_gguf_size(records, inventory.config_text, inventory.gemma_version)
    ltx25._preflight_disk(output, required_bytes)
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
                writer = _build_writer(records, inventory.config_text, inventory.gemma_version, temp)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_ti_data_to_file()
                for record in records:
                    payload = _tensor_payload(
                        reader,
                        record["name"],
                        record,
                        quant_workers=quant_workers,
                        q4_executor=q4_executor,
                    )
                    if payload.nbytes != record["nbytes"]:
                        raise Gemma4Error(
                            f"gemma4-ltx25 conversion failed: {record['name']} payload {payload.nbytes} != map {record['nbytes']}"
                        )
                    writer.write_tensor_data(payload)
                    del payload
                writer.close()
        finally:
            if writer is not None:
                writer.close()
            if q4_executor is not None:
                q4_executor.shutdown(wait=True, cancel_futures=True)
        _verify_output(temp, records, inventory.config_text, source_path=source, inventory=inventory)
        ltx25._replace_with_retry(temp, output)
        committed = True
        output_sha = ltx25.sha256_of_file(output)
        manifest = {
            "format": MANIFEST_FORMAT,
            "profile": PROFILE_ID,
            "source_sha256": inventory.source_sha256,
            "inventory_sha256": inventory.inventory_sha256,
            "map_sha256": policy_map["map_sha256"],
            "output_sha256": output_sha,
            "tensor_count": len(records),
            "type_counts": _type_counts(records),
            "tool_version": ltx25.__version__,
        }
        ltx25._write_json_atomic(Path(f"{output}.manifest.json"), manifest)
        return output
    except Exception as exc:
        remaining = ltx25._cleanup_temp(temp)
        if committed:
            message = f"gemma4-ltx25 conversion committed GGUF but failed afterward: {exc}"
            if remaining:
                message += f"; temporary file retained at {remaining}"
            raise Gemma4ManifestError(message) from exc
        if remaining:
            raise Gemma4Error(f"{exc}; temporary file retained at {remaining}") from exc
        raise


def verify_gemma4(
    output_path: str | Path,
    policy_map_path: str | Path,
    *,
    inventory_path: str | Path,
    source_path: str | Path,
    source_lock: ltx25.SourceLock,
    builder_oracle_path: str | Path,
    manifest_path: str | Path | None = None,
    require_source_lock: bool = True,
) -> dict[str, Any]:
    """Verify E1/E3/E4/E5 consistency without importing a backend or Torch."""
    if require_source_lock and not source_lock.complete:
        raise Gemma4SourceLockIncompleteError(
            "gemma4-ltx25 source-lock-incomplete: authenticated official source SHA-256/LFS hash "
            "is required before self-verification"
        )
    output = Path(output_path)
    if not output.is_file():
        raise FileNotFoundError(f"gemma4-ltx25 output not found: {output}")
    policy_map = load_policy_map(policy_map_path, require_approved=True)
    records = sorted((dict(row) for row in policy_map["tensors"]), key=lambda row: str(row["name"]))
    manifest_file = Path(manifest_path) if manifest_path is not None else Path(f"{output}.manifest.json")
    if not manifest_file.is_file():
        raise Gemma4ManifestError(f"gemma4-ltx25 manifest-missing: {manifest_file}")
    manifest = ltx25._read_json(manifest_file)
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("profile") != PROFILE_ID:
        raise Gemma4ManifestError("gemma4-ltx25 manifest-missing: manifest format/profile is invalid")
    if manifest.get("map_sha256") != policy_map.get("map_sha256"):
        raise Gemma4ManifestError("gemma4-ltx25 manifest-missing: map SHA-256 does not match")
    if manifest.get("output_sha256") != ltx25.sha256_of_file(output):
        raise Gemma4ManifestError("gemma4-ltx25 manifest-missing: output SHA-256 does not match")
    if manifest.get("tensor_count") != len(records) or manifest.get("type_counts") != _type_counts(records):
        raise Gemma4ManifestError("gemma4-ltx25 manifest-missing: tensor counts do not match")
    inventory_file = Path(inventory_path)
    if not inventory_file.is_file():
        raise Gemma4InventoryMismatchError(f"gemma4-ltx25 inventory-missing: {inventory_file}")
    loaded_inventory = load_inventory(inventory_file)
    inventory_sha = loaded_inventory["inventory_sha256"]
    if policy_map.get("inventory_sha256") != inventory_sha:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: map does not match canonical inventory")
    if manifest.get("inventory_sha256") != inventory_sha:
        raise Gemma4ManifestError("gemma4-ltx25 manifest-missing: inventory SHA-256 does not match")
    inspected = inspect_gemma4(
        source_path,
        source_lock=source_lock,
        require_source_lock=require_source_lock,
        builder_oracle_path=builder_oracle_path,
    )
    if loaded_inventory.get("source_sha256") != inspected.source_sha256:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: canonical inventory source SHA-256 differs")
    if manifest.get("source_sha256") != inspected.source_sha256:
        raise Gemma4ManifestError("gemma4-ltx25 manifest-missing: source SHA-256 does not match admitted source")
    if policy_map.get("inventory_sha256") != inspected.inventory_sha256:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: map does not match admitted source")
    if loaded_inventory.get("inventory_sha256") != inspected.inventory_sha256:
        raise Gemma4InventoryMismatchError("gemma4-ltx25 inventory mismatch: canonical inventory differs from admitted source")
    _verify_output(output, records, inspected.config_text, source_path=Path(source_path), inventory=inspected)
    return manifest


# ---------------------------------------------------------------------------
# config.toml plumbing
# ---------------------------------------------------------------------------
def source_lock_from_config(config: dict[str, Any]) -> ltx25.SourceLock:
    profile = config.get("profiles", {}).get(PROFILE_ID, {})
    if not isinstance(profile, dict):
        return DEFAULT_SOURCE_LOCK
    return ltx25.SourceLock(
        repo_id=str(profile.get("source_repo_id", DEFAULT_SOURCE_LOCK.repo_id)),
        artifact_revision=str(profile.get("artifact_revision", DEFAULT_SOURCE_LOCK.artifact_revision)),
        filename=str(profile.get("source_filename", DEFAULT_SOURCE_LOCK.filename)),
        expected_size=profile.get("source_expected_size", DEFAULT_SOURCE_LOCK.expected_size),
        source_sha256=str(profile.get("source_sha256", ltx25.SOURCE_LOCK_UNCONFIRMED)),
        lfs_sha256=str(profile.get("lfs_sha256", ltx25.SOURCE_LOCK_UNCONFIRMED)),
        git_blob_oid=str(profile.get("git_blob_oid", DEFAULT_SOURCE_LOCK.git_blob_oid)),
        gated=str(profile.get("gated", DEFAULT_SOURCE_LOCK.gated)),
        license_id=str(profile.get("license_id", DEFAULT_SOURCE_LOCK.license_id)),
    )


def default_paths_from_config(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    profile = config.get("profiles", {}).get(PROFILE_ID, {})
    if not isinstance(profile, dict):
        profile = {}
    local_dir = project_root / str(profile.get("source_local_dir", ".artifacts/official-ltx25"))
    filename = str(profile.get("source_filename", DEFAULT_SOURCE_LOCK.filename))
    return {
        "local_dir": local_dir,
        "source": local_dir / filename,
        "oracle": project_root / str(profile.get("builder_oracle_path", "typemap/gemma4_ltx25_builder_oracle.json")),
        "inventory": project_root / str(profile.get("inventory_path", "typemap/gemma4_ltx25_inventory.json")),
        "map": project_root / str(profile.get("map_path", "typemap/gemma4_ltx25_conversion_map.json")),
        "draft_map": project_root
        / str(profile.get("draft_map_path", "typemap/gemma4_ltx25_conversion_map.draft.json")),
        "output": project_root / str(profile.get("output_dir", "output"))
        / str(profile.get("output_filename", "LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf")),
    }
