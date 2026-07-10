"""Transcribe safetensors ``__metadata__`` into GGUF key-value metadata.

The source safetensors file for LTX-2.3 (e.g. ``sulphur_distil_bf16.safetensors``)
carries a top-level ``__metadata__`` block with (at least) the following string
entries:

- ``config``: a JSON string describing the model. The backend requires this
  to be present and parseable -- without it, loading the converted GGUF
  raises a ``RuntimeError``.
- ``license``, ``model_version``, ``encrypted_wandb_properties``: opaque
  strings that are carried over as-is, best effort.

This module reads that block, validates it, and writes the relevant keys
onto a :class:`gguf.GGUFWriter` so the resulting GGUF matches the KV layout
of the reference conversion (``general.architecture``,
``general.quantization_version``, ``general.file_type``, plus the four
string keys above).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import gguf
import safetensors

logger = logging.getLogger(__name__)

# The one safetensors __metadata__ key that must be present and valid.
_REQUIRED_KEY = "config"

# Keys copied verbatim (as GGUF STRING values) from safetensors __metadata__.
# Missing/empty entries are skipped with a warning -- processing continues.
_OPTIONAL_STRING_KEYS = ("license", "model_version", "encrypted_wandb_properties")

# Fixed KV values matching the reference GGUF (see module docstring / Docs).
_QUANTIZATION_VERSION = 2
_FILE_TYPE = 15


def read_st_metadata(safetensors_path: str | Path) -> dict[str, str]:
    """Read the ``__metadata__`` block of a safetensors file.

    Only the header is parsed (via ``safe_open().metadata()``), so this is
    cheap even for multi-gigabyte tensor files -- the tensor data itself is
    never touched.

    Returns an empty dict if the file has no ``__metadata__`` block at all.
    """
    with safetensors.safe_open(str(safetensors_path), framework="numpy") as f:
        meta = f.metadata()
    return dict(meta) if meta else {}


def validate_metadata(meta: dict[str, Any]) -> None:
    """Validate that ``meta`` contains a usable ``config`` entry.

    ``config`` is required by the backend: missing, empty, or non-JSON
    values are treated as a hard error. The other known keys (``license``,
    ``model_version``, ``encrypted_wandb_properties``) are optional --
    if absent or empty, a warning is logged and conversion continues; the
    caller (``apply_kv``) will simply not write that key.

    Raises:
        ValueError: if ``config`` is missing, empty, or not valid JSON.
    """
    config_value = meta.get(_REQUIRED_KEY)

    if config_value is None:
        raise ValueError(
            f"safetensors __metadata__ is missing the required '{_REQUIRED_KEY}' key. "
            "The backend cannot load a GGUF converted without it (it raises "
            "RuntimeError at load time when 'config' is absent)."
        )

    if not config_value:
        raise ValueError(
            f"safetensors __metadata__['{_REQUIRED_KEY}'] is an empty string. "
            "The backend requires a non-empty JSON config string; refusing to "
            "convert a model whose config would be silently dropped (gguf's "
            "add_string ignores empty strings)."
        )

    try:
        json.loads(config_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"safetensors __metadata__['{_REQUIRED_KEY}'] is not valid JSON "
            f"(json.loads failed: {exc}). First 200 chars: {config_value[:200]!r}"
        ) from exc

    for key in _OPTIONAL_STRING_KEYS:
        if not meta.get(key):
            logger.warning(
                "safetensors __metadata__ has no (or empty) '%s' key; "
                "it will not be written to the output GGUF.",
                key,
            )


def apply_kv(writer: "gguf.GGUFWriter", meta: dict[str, Any]) -> None:
    """Write the transcoded metadata key-values onto ``writer``.

    Writes ``general.quantization_version`` (UINT32=2), ``general.file_type``
    (UINT32=15), and any of ``config``/``license``/``model_version``/
    ``encrypted_wandb_properties`` present (non-empty) in ``meta`` as GGUF
    STRING values. No other keys from ``meta`` are written.

    ``general.architecture`` is intentionally not written here: it is set
    automatically by ``GGUFWriter.__init__`` from its ``arch`` constructor
    argument. ``general.alignment`` is also intentionally never written
    (the reference GGUF does not carry it either).
    """
    writer.add_uint32("general.quantization_version", _QUANTIZATION_VERSION)
    writer.add_uint32("general.file_type", _FILE_TYPE)

    for key in (_REQUIRED_KEY, *_OPTIONAL_STRING_KEYS):
        value = meta.get(key)
        if value:
            # add_string() itself no-ops on falsy values (gguf 0.18.0), this
            # guard just avoids passing None and keeps intent explicit.
            writer.add_string(key, value)
