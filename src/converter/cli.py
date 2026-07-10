"""Command-line interface for the Sulphur-2 (LTX-2.3 fine-tune) -> GGUF pipeline.

Wires together the four pipeline stages -- each implemented in its own module
and unchanged here -- into a single ``converter`` CLI with subcommands:

* ``download``        -- :mod:`converter.download`  (fetch the source safetensors)
* ``extract-typemap``  -- :mod:`converter.typemap`   (derive the quantization typemap
  from the reference GGUF)
* ``convert``          -- :mod:`converter.convert`   (stream-convert safetensors -> GGUF)
* ``verify``           -- :mod:`converter.verify`    (structural check vs. the reference)
* ``all``              -- runs the four steps above in order, stopping at the first
  failure. ``extract-typemap`` is skipped if its output already exists, unless
  ``--force-typemap`` is given.

Every subcommand falls back to ``config.toml`` for its default paths/values and
accepts CLI options to override them (see ``--help`` on each subcommand).

Exit codes: ``0`` on success, ``1`` on any failure. Expected/handled failures
(missing input files, structural verification mismatches, etc.) print a short
message to stderr with no traceback. Unexpected exceptions bubble up to
:func:`main`, which prints one clear summary line to stderr followed by the
full traceback (for debugging) before returning ``1``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
import tomllib
from pathlib import Path
from typing import Any

from . import convert as convert_mod
from . import download as download_mod
from . import typemap as typemap_mod
from . import verify as verify_mod

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.toml"


# --------------------------------------------------------------------------
# config / path resolution helpers
# --------------------------------------------------------------------------
def _load_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _resolve_path(value: str | None, default: Path) -> Path:
    """Return ``Path(value)`` if an override was given, else ``default``."""
    return Path(value) if value else default


# --------------------------------------------------------------------------
# subcommand handlers
# --------------------------------------------------------------------------
def cmd_download(args: argparse.Namespace, config: dict[str, Any]) -> int:
    source = config["source"]
    repo_id = getattr(args, "repo_id", None) or source["repo_id"]
    filename = getattr(args, "filename", None) or source["filename"]
    local_dir = _resolve_path(getattr(args, "local_dir", None), _PROJECT_ROOT / source["local_dir"])
    expected_size = getattr(args, "expected_size", None)
    if expected_size is None:
        expected_size = source["expected_size"]

    print(f"Repo    : {repo_id}")
    print(f"Filename: {filename}")
    print(f"Dest dir: {local_dir}")

    path = download_mod.download(repo_id, filename, local_dir, expected_size)
    print(f"Download complete: {path}")
    return 0


def cmd_extract_typemap(args: argparse.Namespace, config: dict[str, Any]) -> int:
    reference_gguf_path = _resolve_path(
        getattr(args, "reference", None), Path(config["reference"]["gguf_path"])
    )
    typemap_path = _resolve_path(
        getattr(args, "typemap", None), _PROJECT_ROOT / config["reference"]["typemap_path"]
    )

    if not reference_gguf_path.exists():
        print(f"Reference GGUF not found: {reference_gguf_path}", file=sys.stderr)
        return 1

    print(f"Reading reference GGUF: {reference_gguf_path}")
    records = typemap_mod.extract_typemap(reference_gguf_path)
    typemap_mod.save_typemap(records, typemap_path, source=reference_gguf_path.name)

    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["ggml_type"]] = counts.get(rec["ggml_type"], 0) + 1

    print(f"Extracted {len(records)} tensors -> {typemap_path}")
    print(f"Type counts: {counts}")

    if len(records) != typemap_mod.EXPECTED_TOTAL:
        print(
            f"WARNING: total tensor count {len(records)} != expected {typemap_mod.EXPECTED_TOTAL}",
            file=sys.stderr,
        )
    if counts != typemap_mod.EXPECTED_TYPE_COUNTS:
        print(
            f"WARNING: type counts {counts} != expected {typemap_mod.EXPECTED_TYPE_COUNTS}",
            file=sys.stderr,
        )

    mismatches = typemap_mod.audit(records)
    print(f"Audit: {len(mismatches)} mismatch(es) between typemap and classify_by_rule.")
    for m in mismatches[:50]:
        print(f"  {m['name']} shape={m['shape_logical']} actual={m['actual']} predicted={m['predicted']}")
    if len(mismatches) > 50:
        print(f"  ... and {len(mismatches) - 50} more")

    return 0


def cmd_convert(args: argparse.Namespace, config: dict[str, Any]) -> int:
    src = config["source"]
    default_st_path = _PROJECT_ROOT / src["local_dir"] / src["filename"]
    st_path = _resolve_path(getattr(args, "st_path", None), default_st_path)

    typemap_path = _resolve_path(
        getattr(args, "typemap", None), _PROJECT_ROOT / config["reference"]["typemap_path"]
    )

    out_cfg = config["output"]
    default_out_path = _PROJECT_ROOT / out_cfg["dir"] / out_cfg["filename"]
    out_path = _resolve_path(getattr(args, "out", None), default_out_path)

    reference_expected = getattr(args, "reference_expected", True)
    if reference_expected is None:
        reference_expected = True

    if not st_path.is_file():
        print(f"Source safetensors not found: {st_path}", file=sys.stderr)
        return 1
    if not typemap_path.is_file():
        print(f"Typemap not found: {typemap_path}", file=sys.stderr)
        return 1
    if out_path.exists():
        print(f"WARNING: output already exists and will be overwritten: {out_path}")

    print(f"Source : {st_path}")
    print(f"Typemap: {typemap_path}")
    print(f"Output : {out_path}")

    result = convert_mod.convert(st_path, typemap_path, out_path, reference_expected=reference_expected)
    print(f"Done: {result}")
    return 0


def cmd_verify(args: argparse.Namespace, config: dict[str, Any]) -> int:
    out_cfg = config["output"]
    default_out_path = _PROJECT_ROOT / out_cfg["dir"] / out_cfg["filename"]
    output_path = _resolve_path(getattr(args, "out", None), default_out_path)

    reference_path = _resolve_path(
        getattr(args, "reference", None), Path(config["reference"]["gguf_path"])
    )

    if not reference_path.is_file():
        print(f"Reference GGUF not found: {reference_path}", file=sys.stderr)
        return 1
    if not output_path.is_file():
        print(f"Output GGUF not found: {output_path}", file=sys.stderr)
        return 1

    report = verify_mod.verify_structure(output_path, reference_path)
    print(report.summary())
    return 0 if report.passed else 1


def cmd_all(args: argparse.Namespace, config: dict[str, Any]) -> int:
    print("=== [1/4] download ===")
    rc = cmd_download(args, config)
    if rc != 0:
        return rc

    print()
    print("=== [2/4] extract-typemap ===")
    typemap_path = _resolve_path(
        getattr(args, "typemap", None), _PROJECT_ROOT / config["reference"]["typemap_path"]
    )
    if typemap_path.is_file() and not getattr(args, "force_typemap", False):
        print(
            f"Typemap already exists, skipping extraction: {typemap_path} "
            "(use --force-typemap to re-extract)"
        )
    else:
        rc = cmd_extract_typemap(args, config)
        if rc != 0:
            return rc

    print()
    print("=== [3/4] convert ===")
    rc = cmd_convert(args, config)
    if rc != 0:
        return rc

    print()
    print("=== [4/4] verify ===")
    rc = cmd_verify(args, config)
    if rc != 0:
        return rc

    print()
    print("All steps completed successfully.")
    return 0


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="converter",
        description=(
            "Sulphur-2 (LTX-2.3 fine-tune) safetensors -> GGUF (Q4_K_M) conversion pipeline. "
            "All subcommands fall back to config.toml for their default paths; "
            "pass options to override them. Run via: run.bat <command> [options]"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    # -- download ----------------------------------------------------------
    p_download = subparsers.add_parser(
        "download",
        help="Download the source safetensors file from the Hugging Face Hub.",
        description=(
            "Download the source safetensors file straight into the local "
            "safetensors/ directory (never duplicated into the global "
            "Hugging Face cache). Skips the download if the file is already "
            "present with the expected size."
        ),
    )
    p_download.add_argument(
        "--repo-id", help="Hugging Face Hub repo id (default: config.toml [source].repo_id)"
    )
    p_download.add_argument(
        "--filename", help="File name within the repo (default: config.toml [source].filename)"
    )
    p_download.add_argument(
        "--local-dir", help="Destination directory (default: config.toml [source].local_dir)"
    )
    p_download.add_argument(
        "--expected-size",
        type=int,
        help="Expected file size in bytes (default: config.toml [source].expected_size)",
    )
    p_download.set_defaults(handler=cmd_download)

    # -- extract-typemap -----------------------------------------------------
    p_typemap = subparsers.add_parser(
        "extract-typemap",
        help="Extract the tensor quantization typemap from the reference GGUF.",
        description=(
            "Read the reference GGUF and write an ordered tensor-name -> "
            "GGML-quantization-type typemap JSON, used by 'convert' as the "
            "single source of truth for how each tensor should be quantized."
        ),
    )
    p_typemap.add_argument(
        "--reference", help="Path to the reference GGUF (default: config.toml [reference].gguf_path)"
    )
    p_typemap.add_argument(
        "--typemap",
        help="Output path for the typemap JSON (default: config.toml [reference].typemap_path)",
    )
    p_typemap.set_defaults(handler=cmd_extract_typemap)

    # -- convert -------------------------------------------------------------
    p_convert = subparsers.add_parser(
        "convert",
        help="Convert the source safetensors file to a Q4_K_M GGUF using the typemap.",
        description=(
            "Stream-convert the source safetensors file into a GGUF, one "
            "tensor at a time, following the ordered typemap."
        ),
    )
    p_convert.add_argument(
        "--st-path",
        help="Path to the source safetensors file (default: config.toml [source] local_dir/filename)",
    )
    p_convert.add_argument(
        "--typemap", help="Path to the typemap JSON (default: config.toml [reference].typemap_path)"
    )
    p_convert.add_argument(
        "--out", help="Output GGUF path (default: config.toml [output] dir/filename)"
    )
    p_convert.add_argument(
        "--reference-expected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Assert the typemap's per-type counts match the known LTX-2.3 "
            "reference distribution (default: true; use --no-reference-expected "
            "for a partial/fixture typemap)"
        ),
    )
    p_convert.set_defaults(handler=cmd_convert)

    # -- verify ----------------------------------------------------------
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify the converted GGUF's structure against the reference GGUF.",
        description=(
            "Compare tensor names/order/types/shapes and KV metadata between "
            "the output GGUF and the reference GGUF (structure only, not weight "
            "values -- the output is a fine-tune of the reference architecture)."
        ),
    )
    p_verify.add_argument(
        "--out", help="Path to the output GGUF to verify (default: config.toml [output] dir/filename)"
    )
    p_verify.add_argument(
        "--reference", help="Path to the reference GGUF (default: config.toml [reference].gguf_path)"
    )
    p_verify.set_defaults(handler=cmd_verify)

    # -- all -------------------------------------------------------------
    p_all = subparsers.add_parser(
        "all",
        help="Run download -> extract-typemap -> convert -> verify in sequence.",
        description=(
            "Run the full pipeline end-to-end, stopping immediately (non-zero "
            "exit) at the first failing step. The extract-typemap step is "
            "skipped when its output typemap JSON already exists, unless "
            "--force-typemap is given."
        ),
    )
    p_all.add_argument("--repo-id", help="[download] Hugging Face Hub repo id")
    p_all.add_argument("--filename", help="[download] File name within the repo")
    p_all.add_argument("--local-dir", help="[download] Destination directory")
    p_all.add_argument("--expected-size", type=int, help="[download] Expected file size in bytes")
    p_all.add_argument("--reference", help="[extract-typemap/verify] Path to the reference GGUF")
    p_all.add_argument("--typemap", help="[extract-typemap/convert] Path to the typemap JSON")
    p_all.add_argument("--st-path", help="[convert] Path to the source safetensors file")
    p_all.add_argument("--out", help="[convert/verify] Output GGUF path")
    p_all.add_argument(
        "--force-typemap",
        action="store_true",
        help="Re-extract the typemap even if it already exists",
    )
    p_all.set_defaults(handler=cmd_all, reference_expected=True)

    return parser


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = _load_config()
    except Exception as exc:
        print(f"Error: failed to load config.toml: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    try:
        return args.handler(args, config)
    except Exception as exc:
        print(f"Error: '{args.command}' failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
