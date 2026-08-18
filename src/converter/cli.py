"""Command-line interface for the Nz-LTX23 weight-conversion toolbox.

Wires together the pipeline stages -- each implemented in its own module and
unchanged here -- into a single ``converter`` CLI with subcommands.

The Sulphur-2 (LTX-2.3 fine-tune) -> GGUF pipeline:

* ``download``        -- :mod:`converter.download`  (fetch the source safetensors)
* ``extract-typemap``  -- :mod:`converter.typemap`   (derive the quantization typemap
  from the reference GGUF)
* ``convert``          -- :mod:`converter.convert`   (stream-convert safetensors -> GGUF)
* ``verify``           -- :mod:`converter.verify`    (structural check vs. the reference)
* ``all``              -- runs the four steps above in order, stopping at the first
  failure. ``extract-typemap`` is skipped if its output already exists, unless
  ``--force-typemap`` is given.

Standalone conversions:

* ``convert-vae``     -- :mod:`converter.convert_vae` (PrunaVAED pruned video VAE
  -> decoder-only safetensors, downloading the pinned upstream revision first if
  it is not already present)

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
from . import convert_vae as convert_vae_mod
from . import download as download_mod
from . import ltx25 as ltx25_mod
from . import quant_kernels as quant_kernels_mod
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


def _model(args: argparse.Namespace) -> str:
    """Return the selected frozen profile; omission remains legacy LTX 2.3."""
    return getattr(args, "model", "ltx23")


def _ltx25_paths(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path]:
    """Resolve LTX 2.5-only defaults without touching legacy path resolution."""
    defaults = ltx25_mod.default_paths_from_config(config, _PROJECT_ROOT)
    return {
        "source": _resolve_path(getattr(args, "st_path", None), defaults["source"]),
        "inventory": _resolve_path(getattr(args, "inventory", None), defaults["inventory"]),
        "oracle": _resolve_path(getattr(args, "builder_oracle", None), defaults["oracle"]),
        "map": _resolve_path(getattr(args, "map", None), defaults["map"]),
        "draft_map": defaults["draft_map"],
        "output": _resolve_path(getattr(args, "out", None), defaults["output"]),
    }


def _ltx25_only(args: argparse.Namespace, command: str) -> bool:
    if _model(args) == "ltx25":
        return True
    print(f"{command} is available only with --model ltx25", file=sys.stderr)
    return False


def _require_ltx25_source_lock(config: dict[str, Any]) -> ltx25_mod.SourceLock:
    lock = ltx25_mod.source_lock_from_config(config)
    if not lock.complete:
        raise ltx25_mod.SourceLockIncompleteError(
            "ltx25 source-lock-incomplete: authenticated official source SHA-256/LFS hash is "
            "required before this command"
        )
    return lock


def _quant_worker_count(args: argparse.Namespace, config: dict[str, Any]) -> int:
    requested = getattr(args, "quant_workers", None)
    if requested is not None:
        return quant_kernels_mod.validate_quant_workers(requested)
    if _model(args) == "ltx25":
        profile = config.get("profiles", {}).get("ltx25", {})
        if isinstance(profile, dict):
            return quant_kernels_mod.validate_quant_workers(profile.get("quant_workers", 4))
        return 4
    return 1


def _quant_worker_argument(text: str) -> int:
    try:
        return quant_kernels_mod.validate_quant_workers(int(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


# --------------------------------------------------------------------------
# subcommand handlers
# --------------------------------------------------------------------------
def cmd_download(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if _model(args) == "ltx25":
        lock = _require_ltx25_source_lock(config)
        if any(
            getattr(args, name, None) is not None
            for name in ("repo_id", "filename", "expected_size", "revision", "sha256")
        ):
            raise ltx25_mod.SourceRejectedError(
                "ltx25 source rejected: --repo-id/--filename/--expected-size/--revision/--sha256 "
                "cannot override the frozen source lock"
            )
        defaults = ltx25_mod.default_paths_from_config(config, _PROJECT_ROOT)
        local_dir = _resolve_path(getattr(args, "local_dir", None), defaults["local_dir"])
        print(f"Repo    : {lock.repo_id}")
        print(f"Filename: {lock.filename}")
        print(f"Dest dir: {local_dir}")
        print(f"Revision: {lock.artifact_revision}")
        print(f"SHA-256 : {lock.source_sha256}")
        path = download_mod.download(
            lock.repo_id,
            lock.filename,
            local_dir,
            int(lock.expected_size),
            lock.artifact_revision,
            lock.source_sha256,
        )
        print(f"Download complete: {path}")
        return 0

    source = config["source"]
    repo_id = getattr(args, "repo_id", None) or source["repo_id"]
    filename = getattr(args, "filename", None) or source["filename"]
    local_dir = _resolve_path(getattr(args, "local_dir", None), _PROJECT_ROOT / source["local_dir"])
    expected_size = getattr(args, "expected_size", None)
    if expected_size is None:
        expected_size = source["expected_size"]
    revision = getattr(args, "revision", None) or source.get("revision")
    sha256 = getattr(args, "sha256", None) or source.get("sha256")

    print(f"Repo    : {repo_id}")
    print(f"Filename: {filename}")
    print(f"Dest dir: {local_dir}")
    if revision:
        print(f"Revision: {revision}")
    if sha256:
        print(f"SHA-256 : {sha256}")

    path = download_mod.download(repo_id, filename, local_dir, expected_size, revision, sha256)
    print(f"Download complete: {path}")
    return 0


def cmd_extract_typemap(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if _model(args) == "ltx25":
        print(
            "extract-typemap is ltx23-only; LTX 2.5 uses inspect -> build-map -> convert -> self-verify.",
            file=sys.stderr,
        )
        return 1
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
    if _model(args) == "ltx25":
        lock = _require_ltx25_source_lock(config)
        quant_workers = _quant_worker_count(args, config)
        paths = _ltx25_paths(args, config)
        if not paths["source"].is_file():
            print(f"LTX 2.5 source safetensors not found: {paths['source']}", file=sys.stderr)
            return 1
        if not paths["map"].is_file():
            print(
                f"LTX 2.5 conversion map not found: {paths['map']} "
                "(run inspect then build-map after the source lock is complete)",
                file=sys.stderr,
            )
            return 1
        if not paths["oracle"].is_file():
            print(f"LTX 2.5 builder oracle not found: {paths['oracle']}", file=sys.stderr)
            return 1
        result = ltx25_mod.convert_ltx25(
            paths["source"],
            paths["map"],
            paths["output"],
            source_lock=lock,
            force=getattr(args, "force", False),
            builder_oracle_path=paths["oracle"],
            quant_workers=quant_workers,
        )
        print(f"LTX 2.5 conversion/self-verify complete: {result}")
        return 0

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

    result = convert_mod.convert(
        st_path,
        typemap_path,
        out_path,
        reference_expected=reference_expected,
        quant_workers=_quant_worker_count(args, config),
    )
    print(f"Done: {result}")
    return 0


def cmd_convert_vae(args: argparse.Namespace, config: dict[str, Any]) -> int:
    pv = config["prunavaed"]
    repo_id = getattr(args, "repo_id", None) or pv["repo_id"]
    revision = getattr(args, "revision", None) or pv["revision"]
    filename = getattr(args, "filename", None) or pv["filename"]
    expected_size = getattr(args, "expected_size", None)
    if expected_size is None:
        expected_size = pv["expected_size"]
    sha256 = getattr(args, "sha256", None) or pv["sha256"]
    local_dir = _resolve_path(getattr(args, "local_dir", None), _PROJECT_ROOT / pv["local_dir"])

    default_out_path = _PROJECT_ROOT / pv["out_dir"] / pv["out_filename"]
    out_path = _resolve_path(getattr(args, "out", None), default_out_path)

    reference_expected = getattr(args, "reference_expected", True)
    if reference_expected is None:
        reference_expected = True

    reference_vae = getattr(args, "reference_vae", None) or pv.get("reference_vae_path")
    reference_vae_path = Path(reference_vae) if reference_vae else None

    st_override = getattr(args, "st_path", None)
    src_path = Path(st_override) if st_override else local_dir / filename
    # An explicit --st-path means "use this file", so nothing is fetched.
    do_download = st_override is None and getattr(args, "download", True)

    print(f"Repo    : {repo_id}")
    print(f"Revision: {revision}")
    print(f"Filename: {filename}")
    print(f"Source  : {src_path}")
    print(f"Output  : {out_path}")
    if reference_vae_path is not None:
        print(f"Ref. VAE: {reference_vae_path}"
              f"{'' if reference_vae_path.is_file() else '  (absent -- item 6 will be skipped)'}")

    # Fetch on demand: the pinned revision and digest make this idempotent, and
    # an already-present file only costs one SHA-256 pass.
    if do_download:
        src_path = download_mod.download(
            repo_id, filename, local_dir, expected_size, revision, sha256
        )
    elif not src_path.is_file():
        print(f"Source safetensors not found: {src_path}", file=sys.stderr)
        return 1

    if out_path.exists():
        print(f"WARNING: output already exists and will be overwritten: {out_path}")

    try:
        report = convert_vae_mod.convert_vae(
            src_path,
            out_path,
            source_repo=repo_id,
            source_revision=revision,
            source_filename=filename,
            expected_sha256=sha256,
            reference_expected=reference_expected,
            reference_vae_path=reference_vae_path,
        )
    except convert_vae_mod.VaeConversionError as exc:
        # Expected/handled failure: a short message, no traceback (see the
        # module docstring's exit-code contract).
        print(f"convert-vae failed: {exc}", file=sys.stderr)
        return 1

    print()
    print(report.summary())
    return 0 if report.passed else 1


def cmd_verify(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if _model(args) == "ltx25":
        _require_ltx25_source_lock(config)
        paths = _ltx25_paths(args, config)
        if not paths["source"].is_file():
            print(f"LTX 2.5 source safetensors not found: {paths['source']}", file=sys.stderr)
            return 1
        if not paths["inventory"].is_file():
            print(f"LTX 2.5 canonical inventory not found: {paths['inventory']}", file=sys.stderr)
            return 1
        if not paths["map"].is_file():
            print(f"LTX 2.5 conversion map not found: {paths['map']}", file=sys.stderr)
            return 1
        if not paths["oracle"].is_file():
            print(f"LTX 2.5 builder oracle not found: {paths['oracle']}", file=sys.stderr)
            return 1
        if not paths["output"].is_file():
            print(f"LTX 2.5 output GGUF not found: {paths['output']}", file=sys.stderr)
            return 1
        manifest_path = _resolve_path(getattr(args, "manifest", None), Path(f"{paths['output']}.manifest.json"))
        manifest = ltx25_mod.verify_ltx25(
            paths["output"],
            paths["map"],
            manifest_path=manifest_path,
            inventory_path=paths["inventory"],
            source_path=paths["source"],
            source_lock=_require_ltx25_source_lock(config),
            builder_oracle_path=paths["oracle"],
        )
        print(
            f"LTX 2.5 static self-verification passed: {manifest['tensor_count']} tensors, "
            f"output SHA-256 {manifest['output_sha256']}"
        )
        return 0

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
    if _model(args) == "ltx25":
        print(
            "all is ltx23-only; LTX 2.5 requires inspect -> build-map -> convert -> self-verify.",
            file=sys.stderr,
        )
        return 1
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


def cmd_inspect(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Write the canonical E3 inventory for the selected LTX 2.5 source."""
    if not _ltx25_only(args, "inspect"):
        return 1
    lock = _require_ltx25_source_lock(config)
    paths = _ltx25_paths(args, config)
    if not paths["source"].is_file():
        print(f"LTX 2.5 source safetensors not found: {paths['source']}", file=sys.stderr)
        return 1
    if not paths["oracle"].is_file():
        print(f"LTX 2.5 builder oracle not found: {paths['oracle']}", file=sys.stderr)
        return 1
    inventory = ltx25_mod.inspect_ltx25(
        paths["source"],
        source_lock=lock,
        audit_path=paths["inventory"],
        builder_oracle_path=paths["oracle"],
    )
    emitted = sum(record["classification"] == "emit" for record in inventory.tensors)
    excluded = len(inventory.tensors) - emitted
    print(
        f"LTX 2.5 inventory written: {paths['inventory']} "
        f"({emitted} emitted, {excluded} excluded, SHA-256 {inventory.inventory_sha256})"
    )
    return 0


def cmd_build_map(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Build the direct-lookup E4 map; conversion requires explicit approval."""
    if not _ltx25_only(args, "build-map"):
        return 1
    _require_ltx25_source_lock(config)
    paths = _ltx25_paths(args, config)
    map_path = _resolve_path(getattr(args, "map", None), paths["draft_map"])
    if not paths["inventory"].is_file():
        print(
            f"LTX 2.5 inventory not found: {paths['inventory']} (run inspect first)",
            file=sys.stderr,
        )
        return 1
    if map_path.resolve() == paths["map"].resolve():
        print(
            f"LTX 2.5 approved E4 path is protected: {paths['map']} "
            "(build-map writes a review draft; select a distinct --map path)",
            file=sys.stderr,
        )
        return 1
    policy = ltx25_mod.build_policy_map(paths["inventory"], map_path)
    print(
        f"LTX 2.5 {policy['status']} conversion map written: {map_path} "
        f"({len(policy['tensors'])} tensors, SHA-256 {policy['map_sha256']})"
    )
    print("Review the draft and promote a separately reviewed checked-in E4 map before convert.")
    return 0


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="converter",
        description=(
            "Nz-LTX23 weight-conversion toolbox: the Sulphur-2 (LTX-2.3 fine-tune) "
            "safetensors -> GGUF (Q4_K_M) pipeline, plus the standalone PrunaVAED "
            "video-VAE decoder conversion. All subcommands fall back to config.toml "
            "for their default paths; pass options to override them. "
            "Run via: run.bat <command> [options]"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    def add_model_option(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--model",
            choices=("ltx23", "ltx25"),
            default="ltx23",
            help="Frozen conversion profile (default: ltx23; ltx25 is explicit and gated).",
        )

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
    p_download.add_argument(
        "--revision",
        help=(
            "Git revision (commit hash/branch/tag) to pin the download to "
            "(default: config.toml [source].revision if present, else the repo's "
            "default branch)"
        ),
    )
    p_download.add_argument(
        "--sha256",
        help=(
            "Pinned SHA-256 of the file's content; verified after download "
            "(default: config.toml [source].sha256 if present, else no check)"
        ),
    )
    add_model_option(p_download)
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
    add_model_option(p_typemap)
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
        "--map",
        help="LTX 2.5 concrete conversion map (default: config.toml [profiles.ltx25].map_path)",
    )
    p_convert.add_argument(
        "--builder-oracle",
        help="LTX 2.5 pinned E2 builder-oracle JSON path",
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
    p_convert.add_argument(
        "--force",
        action="store_true",
        help="With --model ltx25, replace an existing output only after temp self-verification.",
    )
    p_convert.add_argument(
        "--quant-workers",
        type=_quant_worker_argument,
        default=None,
        help="Q4_K workers: ltx23 default 1; ltx25 profile default 4 (range: 1-8)",
    )
    add_model_option(p_convert)
    p_convert.set_defaults(handler=cmd_convert)

    # -- verify ----------------------------------------------------------
    p_verify = subparsers.add_parser(
        "verify",
        aliases=["self-verify"],
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
    p_verify.add_argument(
        "--map",
        help="LTX 2.5 concrete conversion map (default: config.toml [profiles.ltx25].map_path)",
    )
    p_verify.add_argument(
        "--manifest",
        help="LTX 2.5 manifest path (default: <out>.manifest.json)",
    )
    p_verify.add_argument(
        "--st-path", help="LTX 2.5 source safetensors path for self-verification"
    )
    p_verify.add_argument("--inventory", help="LTX 2.5 canonical inventory JSON path")
    p_verify.add_argument("--builder-oracle", help="LTX 2.5 pinned E2 builder-oracle JSON path")
    add_model_option(p_verify)
    p_verify.set_defaults(handler=cmd_verify)

    # -- inspect (LTX 2.5) -----------------------------------------------
    p_inspect = subparsers.add_parser(
        "inspect",
        help="Inspect the official LTX 2.5 header and write its canonical inventory.",
        description=(
            "LTX 2.5 only: validates the frozen official source lock, safetensors "
            "header, metadata config, E2 component key/shape classification, and E3 BF16/F32 rows."
        ),
    )
    p_inspect.add_argument("--st-path", help="Official LTX 2.5 source safetensors path")
    p_inspect.add_argument("--inventory", help="Output canonical inventory JSON path")
    p_inspect.add_argument("--builder-oracle", help="Pinned E2 builder-oracle JSON path")
    add_model_option(p_inspect)
    p_inspect.set_defaults(handler=cmd_inspect)

    # -- build-map (LTX 2.5) ---------------------------------------------
    p_build_map = subparsers.add_parser(
        "build-map",
        aliases=["build-policy"],
        help="Build a review-only LTX 2.5 draft map from the canonical inventory.",
        description=(
            "LTX 2.5 only: creates a review-only one-row-per-tensor draft. It cannot "
            "write the approved E4 path, approve, or convert the map."
        ),
    )
    p_build_map.add_argument("--inventory", help="Canonical inventory JSON path")
    p_build_map.add_argument(
        "--map",
        help="Output review-draft map JSON path (default: [profiles.ltx25].draft_map_path)",
    )
    add_model_option(p_build_map)
    p_build_map.set_defaults(handler=cmd_build_map)

    # -- convert-vae ---------------------------------------------------------
    p_convert_vae = subparsers.add_parser(
        "convert-vae",
        help="Convert the PrunaVAED checkpoint into a decoder-only safetensors file.",
        description=(
            "Download (pinned revision + SHA-256) the upstream PrunaVAED "
            "diffusers-format VAE, extract just the decoder, rename its keys to "
            "the backend's flat ltx-core layout and write a ~690 MB BF16 "
            "safetensors file. Tensor bytes are copied verbatim -- no dtype "
            "conversion, no requantization. The command self-verifies what it "
            "wrote (tensor count, key set, shapes, parameter total, an "
            "exhaustive per-tensor MD5 passthrough proof, the latent "
            "statistics, a header round-trip and the total size) and exits "
            "non-zero if any item fails."
        ),
    )
    p_convert_vae.add_argument(
        "--repo-id", help="Hugging Face Hub repo id (default: config.toml [prunavaed].repo_id)"
    )
    p_convert_vae.add_argument(
        "--revision", help="Pinned git revision (default: config.toml [prunavaed].revision)"
    )
    p_convert_vae.add_argument(
        "--filename", help="File name within the repo (default: config.toml [prunavaed].filename)"
    )
    p_convert_vae.add_argument(
        "--expected-size",
        type=int,
        help="Expected source size in bytes (default: config.toml [prunavaed].expected_size)",
    )
    p_convert_vae.add_argument(
        "--sha256", help="Pinned source SHA-256 (default: config.toml [prunavaed].sha256)"
    )
    p_convert_vae.add_argument(
        "--local-dir",
        help="Download destination directory (default: config.toml [prunavaed].local_dir)",
    )
    p_convert_vae.add_argument(
        "--st-path",
        help=(
            "Use this source safetensors file as-is and skip the download "
            "(default: <local-dir>/<filename>)"
        ),
    )
    p_convert_vae.add_argument(
        "--out",
        help="Output safetensors path (default: config.toml [prunavaed] out_dir/out_filename)",
    )
    p_convert_vae.add_argument(
        "--reference-vae",
        help=(
            "Stock LTX23_video_vae_bf16.safetensors, used only to cross-check the "
            "latent statistics (default: config.toml [prunavaed].reference_vae_path; "
            "the check is skipped when the file is absent)"
        ),
    )
    p_convert_vae.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch the source from the Hub if needed (default: true)",
    )
    p_convert_vae.add_argument(
        "--reference-expected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Check the shape table, the 345,006,256 parameter total and the "
            "690,012,512-byte payload against the real PrunaVAED v2 model "
            "(default: true; use --no-reference-expected for a synthetic fixture)"
        ),
    )
    p_convert_vae.set_defaults(handler=cmd_convert_vae)

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
    add_model_option(p_all)
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
    except ltx25_mod.Ltx25Error as exc:
        # Admission/map/output failures are expected user-facing outcomes for
        # the gated profile.  Keep them concise; a traceback adds no remedy.
        print(f"Error: '{args.command}' failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: '{args.command}' failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
