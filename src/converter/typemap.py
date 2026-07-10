"""Reference-GGUF typemap extraction for LTX-2.3 -> Q4_K_M conversion.

The reference GGUF (see ``config.toml`` -> ``[reference].gguf_path``) is the single
source of truth for how each tensor of the LTX-2.3 architecture should be quantized.
This module extracts an *ordered* "tensor name -> GGML quantization type" mapping
from that reference file and persists it as JSON so the converter never has to
re-derive quantization decisions on its own.

It also ships a rule-based classifier (``classify_by_rule``) that reproduces the
observed quantization pattern from tensor name/shape alone. The rules are not used
by the converter (the extracted typemap is authoritative) but are useful as an
audit tool to sanity-check the extracted data and to document *why* each tensor
got the type it did.

Known-good statistics for the current reference GGUF
(``LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf``, 4444 tensors total):

    F32=2700, Q4_K=1242, Q6_K=322, BF16=112, Q5_K=68
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from gguf import GGUFReader

EXPECTED_TOTAL = 4444
EXPECTED_TYPE_COUNTS: dict[str, int] = {
    "F32": 2700,
    "Q4_K": 1242,
    "Q6_K": 322,
    "BF16": 112,
    "Q5_K": 68,
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.toml"


def extract_typemap(reference_gguf_path: str | Path) -> list[dict[str, Any]]:
    """Extract an ordered tensor typemap from ``reference_gguf_path``.

    Returns a list of records (one per tensor, in the same order as they appear
    in the GGUF file) with the following keys:

    - ``name``: raw tensor name (GGUF key) as a string.
    - ``ggml_type``: GGML quantization type name (e.g. ``"F32"``, ``"Q4_K"``).
    - ``shape_gguf``: shape as reported by GGUFReader (``ne`` order), as a list of ints.
    - ``shape_logical``: ``shape_gguf`` reversed (torch/safetensors-style logical shape).
    - ``n_elements``: total element count.
    - ``nbytes``: number of bytes the tensor occupies in the GGUF file.
    """
    reader = GGUFReader(str(reference_gguf_path))
    records: list[dict[str, Any]] = []
    for tensor in reader.tensors:
        shape_gguf = [int(dim) for dim in tensor.shape]
        shape_logical = list(reversed(shape_gguf))
        records.append(
            {
                "name": tensor.name,
                "ggml_type": tensor.tensor_type.name,
                "shape_gguf": shape_gguf,
                "shape_logical": shape_logical,
                "n_elements": int(tensor.n_elements),
                "nbytes": int(tensor.n_bytes),
            }
        )
    return records


def _type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        ggml_type = rec["ggml_type"]
        counts[ggml_type] = counts.get(ggml_type, 0) + 1
    return counts


def save_typemap(
    records: list[dict[str, Any]],
    out_path: str | Path,
    source: str = "",
) -> None:
    """Save ``records`` (as produced by :func:`extract_typemap`) to ``out_path`` as JSON.

    The JSON has this top-level shape::

        {
          "source": "<reference gguf filename>",
          "total": 4444,
          "type_counts": {"F32": 2700, ...},
          "tensors": [ {record...}, ... ]
        }
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "total": len(records),
        "type_counts": _type_counts(records),
        "tensors": records,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_typemap(path: str | Path) -> list[dict[str, Any]]:
    """Load a typemap JSON file (as written by :func:`save_typemap`) and return the
    ordered list of tensor records (the ``"tensors"`` array)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["tensors"]


def classify_by_rule(name: str, shape_logical: list[int]) -> str:
    """Rule-based (fallback/audit) classification of a tensor's GGML type.

    Rules are evaluated top-to-bottom; the first matching rule wins. This
    reproduces the quantization pattern observed in the reference GGUF for the
    LTX-2.3 architecture:

    1. 1-dimensional tensors -> F32 (biases, 1D tables, etc.).
    2. AdaLN modules, patchify/proj_out projections, scale/shift tables and
       learnable registers -> F32 (kept at full precision regardless of shape,
       including ``*_embeddings_connector.learnable_registers``, which is why
       this rule must be checked before the "embeddings_connector" rule below).
    3. Any other 2D tensor belonging to an "embeddings_connector" block -> BF16.
    4. 2D tensors in transformer_blocks 0 or 47 (first/last block) -> Q5_K
       (higher precision at the network boundaries).
    5. ``*.to_v.weight`` / ``*.ff.net.2.weight`` 2D tensors (in the remaining
       "regular" transformer blocks) -> Q6_K.
    6. Everything else (regular 2D weight matrices) -> Q4_K.
    """
    if len(shape_logical) == 1:
        return "F32"

    if (
        "adaln" in name
        or name.endswith("patchify_proj.weight")
        or name.endswith("proj_out.weight")
        or "scale_shift_table" in name
        or "learnable_registers" in name
    ):
        return "F32"

    if "embeddings_connector" in name:
        return "BF16"

    if name.startswith("transformer_blocks.0.") or name.startswith("transformer_blocks.47."):
        return "Q5_K"

    if name.endswith(".to_v.weight") or name.endswith(".ff.net.2.weight"):
        return "Q6_K"

    return "Q4_K"


def audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-check every record's ``ggml_type`` against :func:`classify_by_rule`.

    Returns a list of mismatch records (empty if the rule set fully explains the
    typemap). Each mismatch has: ``name``, ``shape_logical``, ``actual`` (the
    type from the typemap) and ``predicted`` (the type from the rule).
    """
    mismatches: list[dict[str, Any]] = []
    for rec in records:
        predicted = classify_by_rule(rec["name"], rec["shape_logical"])
        if predicted != rec["ggml_type"]:
            mismatches.append(
                {
                    "name": rec["name"],
                    "shape_logical": rec["shape_logical"],
                    "actual": rec["ggml_type"],
                    "predicted": predicted,
                }
            )
    return mismatches


def _load_config(config_path: str | Path = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main() -> int:
    config = _load_config()
    reference_gguf_path = Path(config["reference"]["gguf_path"])
    typemap_path = _PROJECT_ROOT / config["reference"]["typemap_path"]

    if not reference_gguf_path.exists():
        print(f"Reference GGUF not found: {reference_gguf_path}", file=sys.stderr)
        return 1

    print(f"Reading reference GGUF: {reference_gguf_path}")
    records = extract_typemap(reference_gguf_path)
    save_typemap(records, typemap_path, source=reference_gguf_path.name)

    counts = _type_counts(records)
    print(f"Extracted {len(records)} tensors -> {typemap_path}")
    print(f"Type counts: {counts}")

    if len(records) != EXPECTED_TOTAL:
        print(
            f"WARNING: total tensor count {len(records)} != expected {EXPECTED_TOTAL}",
            file=sys.stderr,
        )
    if counts != EXPECTED_TYPE_COUNTS:
        print(
            f"WARNING: type counts {counts} != expected {EXPECTED_TYPE_COUNTS}",
            file=sys.stderr,
        )

    mismatches = audit(records)
    print(f"Audit: {len(mismatches)} mismatch(es) between typemap and classify_by_rule.")
    for m in mismatches[:50]:
        print(f"  {m['name']} shape={m['shape_logical']} actual={m['actual']} predicted={m['predicted']}")
    if len(mismatches) > 50:
        print(f"  ... and {len(mismatches) - 50} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
