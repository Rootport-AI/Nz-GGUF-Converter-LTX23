"""E2E stage A: streaming CPU load/dequant check of the converted GGUF, using the
AviUtl2 backend's own loader/dequant code paths.

This script is intentionally self-contained and READ-ONLY with respect to the
backend directory (``Nz-LTX23-backend``): it never edits or writes any file
under that tree. It only *imports* two backend modules --
``engine.gguf.loader_service`` and ``engine.gguf.quant_service`` -- by adding
the backend root to ``sys.path``, and it must be *run* with the backend's own
venv interpreter (``Nz-LTX23-backend\\.venv-engine\\Scripts\\python.exe``)
because that venv is the one with ``torch`` installed.

What it checks (streaming, one tensor at a time -- see the RAM-safety note
below):

  (a) KV metadata: the backend's ``GGUFStateDictLoader.metadata()`` (in
      ``loader_service.py``) is called against the output GGUF. That method
      is the *exact* code path the real backend uses at model-load time to
      find and ``json.loads`` the ``config`` KV entry (it searches, in order,
      the KV keys ``"config"``, ``"ltx.config"``, ``"general.config"``). This
      script calls that method verbatim -- no reimplementation.

  (b) Per-tensor dequantization: every tensor in the output GGUF (4444 for
      this model) is read via ``gguf.GGUFReader`` (memory-mapped -- cheap) and
      dequantized with the backend's ``quant_service.dequantize_ggml_tensor``
      -- the SAME function ``GGUFStateDictLoader.load()`` calls for every
      tensor during a real model load (via the thin wrapper
      ``loader_service._dequantize_tensor``). Each tensor's dequantized
      output is checked for NaN/Inf, then immediately discarded before the
      next tensor is processed.

RAM-safety design
------------------
The full model is ~22B parameters. Dequantized to bf16 (2 bytes/param) that
is ~44 GB -- fully materializing the whole state dict at once is only safe if
the machine has comfortably more than that free. Rather than gate on a RAM
check and branch into two code paths, this script *always* streams: it holds
at most one tensor's raw bytes + one tensor's dequantized output in memory at
a time, then frees both (``del`` + the loop moving on) before touching the
next tensor. Peak extra RAM beyond the base Python/torch/gguf-mmap footprint
is therefore bounded by the single largest tensor in the file (a few hundred
MB at most for this architecture's biggest Linear layers), not by the model
size. This is deliberately the safer of the two designs described in the
task brief, and it is used unconditionally -- the RAM probe below is
recorded for the report only, not used to pick a riskier full-materialization
path.

Fallback
--------
If importing the backend modules fails for any reason (e.g. a future venv
missing a dependency, or an import-time CUDA initialization error), this
script falls back to ``gguf.quants.dequantize`` from the ``gguf`` PyPI
package for the per-tensor dequant check, and does a local re-implementation
of the KV/config lookup (same three candidate keys, same ``json.loads``).
The project's own QUANT_KERNELS.md verification already established that the
backend's torch dequant kernels are a faithful, bit-matching port of
``gguf.quants`` (same reference oracle), so this fallback is numerically
equivalent -- it is not a weaker check, just a different call path.

Usage (must be run with the BACKEND's venv, not this project's .venv; run from
this project's root, e.g. assuming this repo and Nz-LTX23-backend are cloned
as sibling directories):

    ..\\Nz-LTX23-backend\\.venv-engine\\Scripts\\python.exe scripts\\e2e_load_check.py

Optional arguments:
    --gguf PATH       override the output GGUF path (default: config.toml's
                      [output] dir/filename, resolved relative to this
                      project's root)
    --max-tensors N   stop after N tensors (for a quick smoke test)

The backend root used to locate ``engine.gguf.loader_service`` /
``engine.gguf.quant_service`` (see ``_try_import_backend`` below) defaults to
a sibling ``Nz-LTX23-backend`` directory next to this project; override with
the ``NZKONV_BACKEND_ROOT`` environment variable if your layout differs. If
the backend cannot be found or imported, this falls back to the
``gguf.quants`` mode described above -- it is not a fatal error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = Path(
    os.environ.get("NZKONV_BACKEND_ROOT")
    or (_PROJECT_ROOT.parent / "Nz-LTX23-backend")
)


# --------------------------------------------------------------------------
# RAM probe (informational only -- see RAM-safety note in the module docstring)
# --------------------------------------------------------------------------
def probe_ram() -> dict[str, float | None]:
    """Best-effort total/free physical RAM in GiB via wmic. Never raises."""
    result: dict[str, float | None] = {"total_gib": None, "free_gib": None}
    try:
        out = subprocess.run(
            ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/format:list"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        kv: dict[str, int] = {}
        for line in out.splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                if v.isdigit():
                    kv[k.strip()] = int(v)
        if "TotalVisibleMemorySize" in kv:
            result["total_gib"] = kv["TotalVisibleMemorySize"] / (1024 * 1024)
        if "FreePhysicalMemory" in kv:
            result["free_gib"] = kv["FreePhysicalMemory"] / (1024 * 1024)
    except Exception as exc:  # pragma: no cover - best effort only
        print(f"[ram-probe] wmic query failed (non-fatal): {exc!r}")
    return result


def resolve_output_gguf(override: str | None) -> Path:
    if override:
        return Path(override)
    with open(_PROJECT_ROOT / "config.toml", "rb") as f:
        config = tomllib.load(f)
    out_cfg = config["output"]
    return _PROJECT_ROOT / out_cfg["dir"] / out_cfg["filename"]


# --------------------------------------------------------------------------
# Backend-code-path (primary) mode
# --------------------------------------------------------------------------
def _try_import_backend():
    """Import the two backend modules. Returns (loader_service, quant_service)
    or raises -- caller decides whether to fall back."""
    backend_str = str(_BACKEND_ROOT)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)
    from engine.gguf import loader_service, quant_service  # noqa: PLC0415
    return loader_service, quant_service


def check_metadata_via_backend(loader_service_mod, gguf_path: Path) -> dict:
    """Call the backend's own GGUFStateDictLoader.metadata() verbatim."""
    loader = loader_service_mod.GGUFStateDictLoader(gguf_path=str(gguf_path))
    config = loader.metadata(str(gguf_path))
    if not isinstance(config, dict) or "transformer" not in config:
        raise AssertionError(
            "backend metadata() returned JSON without a top-level 'transformer' key"
        )
    return config


def check_tensors_via_backend(
    quant_service_mod, gguf_path: Path, max_tensors: int | None
) -> tuple[int, int, list[str]]:
    """Dequantize every tensor via quant_service.dequantize_ggml_tensor (CPU).

    Returns (n_ok, n_fail, first_failures).
    """
    import numpy as np
    import torch
    import gguf as gguf_lib

    reader = gguf_lib.GGUFReader(str(gguf_path), mode="r")
    tensors = reader.tensors
    if max_tensors is not None:
        tensors = tensors[:max_tensors]

    n_ok = 0
    n_fail = 0
    failures: list[str] = []
    dequantize_ggml_tensor = quant_service_mod.dequantize_ggml_tensor

    for i, tensor in enumerate(tensors):
        name = tensor.name
        ggml_type = tensor.tensor_type.value
        shape = tuple(reversed(tensor.shape.tolist()))
        try:
            raw_np = np.array(tensor.data, copy=True)  # owned copy, breaks mmap alias
            raw_flat = torch.from_numpy(raw_np).reshape(-1)
            out = dequantize_ggml_tensor(raw_flat, ggml_type, shape, torch.bfloat16)
            if torch.isnan(out).any():
                raise AssertionError("dequantized output contains NaN")
            if torch.isinf(out).any():
                raise AssertionError("dequantized output contains Inf")
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            msg = f"{name} (type={tensor.tensor_type.name}, shape={shape}): {exc!r}"
            if len(failures) < 20:
                failures.append(msg)
        finally:
            # Explicitly drop references before the next iteration so the
            # single-tensor peak (not the whole-model peak) is what matters.
            raw_np = None
            raw_flat = None
            out = None

        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(tensors)} tensors processed "
                  f"({n_ok} ok, {n_fail} failed so far)")

    return n_ok, n_fail, failures


# --------------------------------------------------------------------------
# gguf.quants fallback mode (used only if the backend import fails)
# --------------------------------------------------------------------------
def check_metadata_fallback(gguf_path: Path) -> dict:
    import gguf as gguf_lib

    reader = gguf_lib.GGUFReader(str(gguf_path), mode="r")
    for field in reader.fields.values():
        if field.name in ("config", "ltx.config", "general.config"):
            raw = bytes(field.parts[-1])
            config = json.loads(raw.decode("utf-8"))
            if isinstance(config, dict) and "transformer" in config:
                return config
    raise AssertionError("no config/ltx.config/general.config KV key with a valid 'transformer' JSON found")


def check_tensors_fallback(gguf_path: Path, max_tensors: int | None) -> tuple[int, int, list[str]]:
    import numpy as np
    import gguf as gguf_lib
    from gguf.quants import dequantize

    reader = gguf_lib.GGUFReader(str(gguf_path), mode="r")
    tensors = reader.tensors
    if max_tensors is not None:
        tensors = tensors[:max_tensors]

    n_ok = 0
    n_fail = 0
    failures: list[str] = []

    for i, tensor in enumerate(tensors):
        name = tensor.name
        try:
            values = dequantize(tensor.data, tensor.tensor_type)
            arr = np.asarray(values, dtype=np.float32)
            if np.isnan(arr).any():
                raise AssertionError("dequantized output contains NaN")
            if np.isinf(arr).any():
                raise AssertionError("dequantized output contains Inf")
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            msg = f"{name} (type={tensor.tensor_type.name}): {exc!r}"
            if len(failures) < 20:
                failures.append(msg)
        finally:
            values = None
            arr = None

        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(tensors)} tensors processed "
                  f"({n_ok} ok, {n_fail} failed so far)")

    return n_ok, n_fail, failures


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", help="Override the output GGUF path")
    parser.add_argument("--max-tensors", type=int, default=None,
                         help="Stop after N tensors (smoke test)")
    args = parser.parse_args()

    gguf_path = resolve_output_gguf(args.gguf)
    print(f"Output GGUF: {gguf_path}")
    if not gguf_path.is_file():
        print(f"ERROR: output GGUF not found: {gguf_path}", file=sys.stderr)
        return 1

    ram = probe_ram()
    if ram["total_gib"] is not None:
        print(f"Host RAM: total={ram['total_gib']:.1f} GiB free={ram['free_gib']:.1f} GiB "
              "(informational only -- this script always streams one tensor at a time "
              "regardless of available RAM; see module docstring)")
    else:
        print("Host RAM: could not be determined (wmic unavailable) -- streaming mode used anyway")

    mode = "backend"
    loader_service_mod = None
    quant_service_mod = None
    try:
        loader_service_mod, quant_service_mod = _try_import_backend()
        print("Backend modules imported OK: engine.gguf.loader_service, engine.gguf.quant_service")
    except Exception as exc:
        mode = "fallback"
        print(f"WARNING: could not import backend engine.gguf modules ({exc!r})")
        print("Falling back to gguf.quants.dequantize (bit-equivalent per QUANT_KERNELS.md).")

    print()
    print(f"=== (a) KV metadata / config check [{mode} path] ===")
    t0 = time.time()
    try:
        if mode == "backend":
            config = check_metadata_via_backend(loader_service_mod, gguf_path)
        else:
            config = check_metadata_fallback(gguf_path)
        print(f"OK: config JSON parsed, top-level keys: {sorted(config.keys())[:10]}"
              f"{' ...' if len(config) > 10 else ''}")
        metadata_ok = True
    except Exception as exc:
        print(f"FAILED: {exc!r}")
        metadata_ok = False
    print(f"(elapsed: {time.time() - t0:.2f}s)")

    print()
    print(f"=== (b) per-tensor CPU dequantization check [{mode} path] ===")
    t0 = time.time()
    if mode == "backend":
        n_ok, n_fail, failures = check_tensors_via_backend(quant_service_mod, gguf_path, args.max_tensors)
    else:
        n_ok, n_fail, failures = check_tensors_fallback(gguf_path, args.max_tensors)
    elapsed = time.time() - t0

    print()
    print("=== Summary ===")
    print(f"Mode                 : {mode}")
    print(f"KV/config check      : {'PASS' if metadata_ok else 'FAIL'}")
    print(f"Tensors checked      : {n_ok + n_fail}")
    print(f"Tensors OK           : {n_ok}")
    print(f"Tensors FAILED       : {n_fail}")
    print(f"Elapsed              : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if n_ok + n_fail:
        print(f"Throughput           : {(n_ok + n_fail) / elapsed:.1f} tensors/s")
    if failures:
        print("First failures:")
        for f in failures:
            print(f"  - {f}")

    passed = metadata_ok and n_fail == 0
    print()
    print("RESULT: " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
