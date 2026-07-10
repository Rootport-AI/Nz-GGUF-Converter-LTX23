"""Structural verification of a converted GGUF against the reference GGUF.

This module answers one question: *does the output GGUF have exactly the same
structure as the reference GGUF?* It deliberately does **not** compare tensor
weight values -- the output is a fine-tune (Sulphur) of the architecture the
reference GGUF (stock LTX-2.3) describes, so the numbers are expected to
differ. What must match exactly is everything that the AviUtl2 backend relies
on to load and run the model:

1. Tensor count (output and reference must have the same number of tensors).
2. Tensor name set *and* order (an exact, position-by-position match).
3. Each tensor's GGML quantization type (``tensor_type``) and shape.
4. The KV metadata key set (order-independent), each key's GGUF value type,
   and the fixed values ``general.architecture == "ltxv"``,
   ``general.quantization_version == 2``, ``general.file_type == 15``.
   ``config``'s value is allowed to differ (it is Sulphur's own metadata) but
   must be valid JSON with a top-level ``"transformer"`` key -- the backend
   needs that to build the model.
5. ``general.alignment`` must be absent from both files.
6. Output file size must be within +/-1% of the reference file size.
7. A best-effort dequantization sanity check: a handful of K-quant tensors
   from the output are dequantized (``gguf.quants.dequantize``) and checked
   for exceptions / NaN / Inf.

See ``config.toml`` for where the output and reference GGUF paths come from.
"""

from __future__ import annotations

import json
import logging
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.toml"

# Fixed KV values the backend expects (see metadata.py's _QUANTIZATION_VERSION
# / _FILE_TYPE, and the reference GGUF's general.architecture).
_EXPECTED_ARCHITECTURE = "ltxv"
_EXPECTED_QUANTIZATION_VERSION = 2
_EXPECTED_FILE_TYPE = 15

# Pseudo-fields GGUFReader synthesizes from the file header (magic/version/
# counts) -- these are not real KV entries written by GGUFWriter.add_*(), so
# they are excluded from KV-key-set comparisons.
_HEADER_PSEUDO_FIELDS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}

# How many K-quant tensors to sample for the dequantization sanity check.
_MAX_DEQUANT_SAMPLES = 8

# Output file size must be within this fraction of the reference file size.
_FILE_SIZE_TOLERANCE = 0.01


@dataclass
class VerifyReport:
    """Result of :func:`verify_structure`.

    ``mismatches`` maps a check name to a list of human-readable issue
    strings; an empty list means that check passed. ``passed`` is True iff
    every check's issue list is empty.
    """

    output_path: str
    reference_path: str
    tensor_count_output: int
    tensor_count_reference: int
    mismatches: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(not issues for issues in self.mismatches.values())

    @property
    def failed_checks(self) -> list[str]:
        return [name for name, issues in self.mismatches.items() if issues]

    def summary(self) -> str:
        lines = [
            f"Output:    {self.output_path}",
            f"Reference: {self.reference_path}",
            f"Tensors:   output={self.tensor_count_output} reference={self.tensor_count_reference}",
            "",
        ]
        for name, issues in self.mismatches.items():
            status = "PASS" if not issues else f"FAIL ({len(issues)} issue(s))"
            lines.append(f"[{status}] {name}")
            for issue in issues[:20]:
                lines.append(f"    - {issue}")
            if len(issues) > 20:
                lines.append(f"    ... and {len(issues) - 20} more")
        lines.append("")
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def _kv_keys(reader: GGUFReader) -> set[str]:
    """Real KV metadata keys of ``reader`` (excludes header pseudo-fields)."""
    return set(reader.fields.keys()) - _HEADER_PSEUDO_FIELDS


# --- individual checks (each returns a list of issue strings; [] == pass) --


def _check_tensor_count(out_reader: GGUFReader, ref_reader: GGUFReader) -> list[str]:
    n_out, n_ref = len(out_reader.tensors), len(ref_reader.tensors)
    if n_out != n_ref:
        return [f"tensor count mismatch: output has {n_out}, reference has {n_ref}"]
    return []


def _check_tensor_names_order(out_reader: GGUFReader, ref_reader: GGUFReader) -> list[str]:
    out_names = [t.name for t in out_reader.tensors]
    ref_names = [t.name for t in ref_reader.tensors]

    issues: list[str] = []
    out_set, ref_set = set(out_names), set(ref_names)
    if out_set != ref_set:
        missing = sorted(ref_set - out_set)
        extra = sorted(out_set - ref_set)
        if missing:
            issues.append(f"{len(missing)} tensor name(s) missing from output, e.g. {missing[:5]}")
        if extra:
            issues.append(f"{len(extra)} unexpected tensor name(s) in output, e.g. {extra[:5]}")
        return issues  # position-by-position order comparison is meaningless here

    for i, (out_name, ref_name) in enumerate(zip(out_names, ref_names)):
        if out_name != ref_name:
            issues.append(f"order mismatch at index {i}: output={out_name!r} reference={ref_name!r}")
    return issues


def _check_tensor_types_shapes(out_reader: GGUFReader, ref_reader: GGUFReader) -> list[str]:
    issues: list[str] = []
    ref_by_name = {t.name: t for t in ref_reader.tensors}
    for out_t in out_reader.tensors:
        ref_t = ref_by_name.get(out_t.name)
        if ref_t is None:
            continue  # already reported by _check_tensor_names_order

        if out_t.tensor_type != ref_t.tensor_type:
            issues.append(
                f"{out_t.name}: tensor_type mismatch: "
                f"output={out_t.tensor_type.name} reference={ref_t.tensor_type.name}"
            )

        out_shape = [int(d) for d in out_t.shape]
        ref_shape = [int(d) for d in ref_t.shape]
        if out_shape != ref_shape:
            issues.append(f"{out_t.name}: shape mismatch: output={out_shape} reference={ref_shape}")
    return issues


def _check_kv_keys(out_reader: GGUFReader, ref_reader: GGUFReader) -> list[str]:
    out_keys = _kv_keys(out_reader)
    ref_keys = _kv_keys(ref_reader)

    issues: list[str] = []
    missing = sorted(ref_keys - out_keys)
    extra = sorted(out_keys - ref_keys)
    if missing:
        issues.append(f"KV key(s) missing from output: {missing}")
    if extra:
        issues.append(f"unexpected KV key(s) in output: {extra}")
    return issues


def _check_kv_types(out_reader: GGUFReader, ref_reader: GGUFReader) -> list[str]:
    issues: list[str] = []
    common_keys = _kv_keys(out_reader) & _kv_keys(ref_reader)
    for key in sorted(common_keys):
        out_types = out_reader.fields[key].types
        ref_types = ref_reader.fields[key].types
        if out_types != ref_types:
            issues.append(f"{key}: KV type mismatch: output={out_types} reference={ref_types}")
    return issues


def _check_kv_fixed_values(out_reader: GGUFReader) -> list[str]:
    issues: list[str] = []

    arch_field = out_reader.fields.get("general.architecture")
    if arch_field is not None:
        value = arch_field.contents()
        if value != _EXPECTED_ARCHITECTURE:
            issues.append(f"general.architecture: expected {_EXPECTED_ARCHITECTURE!r}, got {value!r}")

    qv_field = out_reader.fields.get("general.quantization_version")
    if qv_field is not None:
        value = qv_field.contents()
        if value != _EXPECTED_QUANTIZATION_VERSION:
            issues.append(
                f"general.quantization_version: expected {_EXPECTED_QUANTIZATION_VERSION}, got {value}"
            )

    ft_field = out_reader.fields.get("general.file_type")
    if ft_field is not None:
        value = ft_field.contents()
        if value != _EXPECTED_FILE_TYPE:
            issues.append(f"general.file_type: expected {_EXPECTED_FILE_TYPE}, got {value}")

    return issues


def _check_config_json(out_reader: GGUFReader) -> list[str]:
    config_field = out_reader.fields.get("config")
    if config_field is None:
        return []  # already reported by _check_kv_keys

    try:
        raw = config_field.contents()
    except Exception as exc:  # pragma: no cover - defensive
        return [f"config: failed to read STRING contents ({exc!r})"]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"config: not valid JSON (json.loads failed: {exc})"]

    if not isinstance(parsed, dict) or "transformer" not in parsed:
        return ["config: parsed JSON is missing the required top-level 'transformer' key"]

    return []


def _check_no_general_alignment(out_reader: GGUFReader, ref_reader: GGUFReader) -> list[str]:
    issues: list[str] = []
    if "general.alignment" in out_reader.fields:
        issues.append("general.alignment is present in the output GGUF but must not be")
    if "general.alignment" in ref_reader.fields:
        issues.append("general.alignment is present in the reference GGUF (unexpected)")
    return issues


def _check_file_size(output_path: Union[str, Path], reference_path: Union[str, Path]) -> list[str]:
    out_size = Path(output_path).stat().st_size
    ref_size = Path(reference_path).stat().st_size
    if ref_size == 0:
        return ["reference file size is 0 bytes; cannot compute a size ratio"]

    ratio = abs(out_size - ref_size) / ref_size
    if ratio > _FILE_SIZE_TOLERANCE:
        return [
            f"file size differs by {ratio:.2%} (tolerance {_FILE_SIZE_TOLERANCE:.0%}): "
            f"output={out_size} bytes, reference={ref_size} bytes"
        ]
    return []


def _sample_indices(n: int, max_samples: int) -> list[int]:
    """Pick up to ``max_samples`` roughly-evenly-spaced indices in ``range(n)``."""
    if n <= max_samples:
        return list(range(n))
    step = n / max_samples
    idxs = sorted({int(i * step) for i in range(max_samples)} | {n - 1})
    return idxs


def _check_dequant_sanity(out_reader: GGUFReader) -> list[str]:
    k_quant_tensors = [t for t in out_reader.tensors if t.tensor_type.name.endswith("_K")]
    if not k_quant_tensors:
        return []

    issues: list[str] = []
    for i in _sample_indices(len(k_quant_tensors), _MAX_DEQUANT_SAMPLES):
        tensor = k_quant_tensors[i]
        try:
            values = dequantize(tensor.data, tensor.tensor_type)
        except Exception as exc:
            issues.append(f"{tensor.name} ({tensor.tensor_type.name}): dequantize raised {exc!r}")
            continue

        arr = np.asarray(values, dtype=np.float32)
        if np.isnan(arr).any():
            issues.append(f"{tensor.name} ({tensor.tensor_type.name}): dequantized values contain NaN")
        if np.isinf(arr).any():
            issues.append(f"{tensor.name} ({tensor.tensor_type.name}): dequantized values contain Inf")
    return issues


# --- public API -------------------------------------------------------------


def verify_structure(
    output_gguf_path: Union[str, Path],
    reference_gguf_path: Union[str, Path],
) -> VerifyReport:
    """Compare ``output_gguf_path``'s structure against ``reference_gguf_path``.

    Opens both files with :class:`gguf.GGUFReader` (memory-mapped -- cheap
    even for multi-gigabyte files) and runs every check described in the
    module docstring. Returns a :class:`VerifyReport`; it never raises for
    structural mismatches (only for I/O errors opening the files themselves).
    """
    out_reader = GGUFReader(str(output_gguf_path))
    ref_reader = GGUFReader(str(reference_gguf_path))

    mismatches: dict[str, list[str]] = {
        "tensor_count": _check_tensor_count(out_reader, ref_reader),
        "tensor_names_order": _check_tensor_names_order(out_reader, ref_reader),
        "tensor_types_shapes": _check_tensor_types_shapes(out_reader, ref_reader),
        "kv_keys": _check_kv_keys(out_reader, ref_reader),
        "kv_types": _check_kv_types(out_reader, ref_reader),
        "kv_fixed_values": _check_kv_fixed_values(out_reader),
        "kv_config_json": _check_config_json(out_reader),
        "general_alignment_absent": _check_no_general_alignment(out_reader, ref_reader),
        "file_size_ratio": _check_file_size(output_gguf_path, reference_gguf_path),
        "dequant_sanity": _check_dequant_sanity(out_reader),
    }

    return VerifyReport(
        output_path=str(output_gguf_path),
        reference_path=str(reference_gguf_path),
        tensor_count_output=len(out_reader.tensors),
        tensor_count_reference=len(ref_reader.tensors),
        mismatches=mismatches,
    )


# --- CLI ---------------------------------------------------------------


def _load_config(config_path: Union[str, Path] = _DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def resolve_paths(config: dict) -> tuple[Path, Path]:
    """Resolve the output and reference GGUF paths described by ``config``."""
    output_path = _PROJECT_ROOT / config["output"]["dir"] / config["output"]["filename"]
    reference_path = Path(config["reference"]["gguf_path"])
    return output_path, reference_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = _load_config()
    output_path, reference_path = resolve_paths(config)

    if not reference_path.is_file():
        print(f"Reference GGUF not found: {reference_path}", file=sys.stderr)
        return 1
    if not output_path.is_file():
        print(f"Output GGUF not found: {output_path}", file=sys.stderr)
        return 1

    report = verify_structure(output_path, reference_path)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
