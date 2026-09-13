"""Numerically compare a ComfyUI-quantised community LTX 2.5 checkpoint against
the official bf16 safetensors, layer by layer.

This is the evidence gate for the ``ltx25-comfyquant`` converter profile: it
proves that :mod:`converter.comfy_dequant` inverts the community file's
quantisation correctly, without needing a GPU or the backend.  For every
quantised Linear layer it dequantises the community weight to float32 and
compares it with the official bf16 weight promoted to float32, reporting

    cos          cosine similarity of the flattened weights
    rel_rmse     ||dequantised - official|| / ||official||
    std_ratio    std(dequantised) / std(official)
    row_norm_r   Pearson correlation of the per-output-row L2 norms

plus, for the connector's *non*-quantised BF16 tensors, plain raw-byte identity.

Gates (thresholds from the conversion plan):

    G-0  coverage: every selected layer was compared, with a marker read from
         the file (no errored layer, no unreadable layer, no assumed marker)
    G-A  connector plain BF16 tensors are byte-identical to the official file
    G-B  connector w4a8 layers        cos >= 0.99
    G-C  every quantised layer        cos >= 0.50
    G-D  transformer quantised layers cos median >= 0.90 and 1st pct >= 0.60
    G-E  every quantised layer        0.50 <= std_ratio <= 2.00
    G-F  every quantised layer        row_norm_r >= 0.80

Truncated sources
-----------------
The community file currently on disk is an incomplete download: its header
declares more payload than the file holds.  Rather than refuse to run, this
script treats any tensor whose ``data_offsets`` end past the real end of file
as *unreadable*, counts it, and skips the layer.  All 1,440 ``comfy_quant``
marker tensors happen to live in the missing tail, so for those layers the
format is inferred from which sidecars exist (``weight_scale`` -> int8 with
ConvRot on, ``weight_s_rel`` -> asym w4a8) and the layer is flagged
``marker_assumed``.  Where a marker *is* readable it is parsed strictly with
:func:`converter.comfy_dequant.parse_quant_marker` instead.

Such a run still reports its per-layer metrics, but it cannot pass: G-0 counts
every skipped layer and every assumed marker as a failure, so the exit code
stays non-zero until the evidence is produced against a complete file.

Memory
------
Both files are read with ``seek``/``read`` one tensor at a time; neither is
loaded whole.  Peak resident memory is roughly two float32 copies of the
largest Linear (about 0.5 GiB for ``[16384, 4096]``), plus small chunks.

Usage::

    python scripts/compare_comfyquant_vs_official.py \\
        --community <community.safetensors> \\
        --official  <official-bf16.safetensors> \\
        --report    output/compare-report.json \\
        [--limit N] [--layers <regex>]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from converter.comfy_dequant import (  # noqa: E402
    ComfyDequantError,
    dequantize_layer,
    parse_quant_marker,
)

REPORT_FORMAT = "nz-comfyquant-compare-report-v1"

QUANT_SIDECAR_SUFFIXES = (
    ".weight_scale",
    ".weight_codebook",
    ".weight_s_channel",
    ".weight_s_rel",
    ".comfy_quant",
)

CONNECTOR_MARKERS = ("audio_embeddings_connector", "video_embeddings_connector")

# Gate thresholds (conversion plan section 4.2).
G_B_MIN_COS = 0.99
G_C_MIN_COS = 0.50
G_D_MIN_MEDIAN_COS = 0.90
G_D_MIN_P1_COS = 0.60
G_E_STD_RATIO_RANGE = (0.50, 2.00)
G_F_MIN_ROW_NORM_R = 0.80

_METRIC_CHUNK_ROWS = 512


# --------------------------------------------------------------------------
# safetensors streaming reader
# --------------------------------------------------------------------------


class SafetensorsStream:
    """Header-only safetensors reader that tolerates a truncated payload."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self._fh: BinaryIO = self.path.open("rb")
        header_len = struct.unpack("<Q", self._fh.read(8))[0]
        self.header: dict[str, Any] = json.loads(self._fh.read(header_len).decode("utf-8"))
        self.base = 8 + header_len
        self.payload_available = self.size - self.base
        self.metadata: dict[str, str] = dict(self.header.get("__metadata__", {}))
        self.entries: dict[str, dict[str, Any]] = {
            name: entry for name, entry in self.header.items() if name != "__metadata__"
        }
        self.max_data_end = max(
            (entry["data_offsets"][1] for entry in self.entries.values()), default=0
        )
        self.missing_bytes = max(0, self.max_data_end - self.payload_available)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "SafetensorsStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- introspection ---------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def dtype_of(self, name: str) -> str:
        return self.entries[name]["dtype"]

    def shape_of(self, name: str) -> list[int]:
        return list(self.entries[name]["shape"])

    def readable(self, name: str) -> bool:
        entry = self.entries.get(name)
        if entry is None:
            return False
        return entry["data_offsets"][1] <= self.payload_available

    # -- payload ---------------------------------------------------------
    def raw_bytes(self, name: str) -> bytes:
        entry = self.entries[name]
        start, end = entry["data_offsets"]
        if end > self.payload_available:
            raise EOFError(f"tensor {name!r} ends at {end}, past the {self.payload_available}-byte payload")
        self._fh.seek(self.base + start)
        data = self._fh.read(end - start)
        if len(data) != end - start:
            raise EOFError(f"short read for tensor {name!r}")
        return data

    def array(self, name: str) -> np.ndarray:
        """Return the tensor as a NumPy array in its *stored* representation.

        BF16 is promoted to float32 exactly; F8_E4M3 comes back as raw uint8
        (NumPy has no fp8 dtype -- :func:`decode_fp8_e4m3` interprets it).
        """
        entry = self.entries[name]
        dtype = entry["dtype"]
        shape = tuple(entry["shape"])
        raw = self.raw_bytes(name)
        if dtype == "BF16":
            # Promote in place: the biggest Linear here is 134 MiB of bf16, so
            # avoiding one extra full-size temporary is worth the two steps.
            u32 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            u32 <<= np.uint32(16)
            arr = u32.view(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype="<f4")
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dtype == "I8":
            arr = np.frombuffer(raw, dtype=np.int8)
        elif dtype in ("U8", "BOOL"):
            arr = np.frombuffer(raw, dtype=np.uint8)
        elif dtype.startswith("F8_"):
            arr = np.frombuffer(raw, dtype=np.uint8)
        else:
            raise ValueError(f"tensor {name!r} has unsupported dtype {dtype!r}")
        return np.ascontiguousarray(arr).reshape(shape)


# --------------------------------------------------------------------------
# layer discovery
# --------------------------------------------------------------------------


def collect_quant_layers(source: SafetensorsStream) -> dict[str, dict[str, str]]:
    """Map ``<layer>`` -> {sidecar suffix (without the dot): tensor name}."""
    layers: dict[str, dict[str, str]] = {}
    for name in source.entries:
        for suffix in QUANT_SIDECAR_SUFFIXES:
            if name.endswith(suffix):
                layers.setdefault(name[: -len(suffix)], {})[suffix[1:]] = name
                break
    return layers


def collect_plain_tensors(
    source: SafetensorsStream, quant_layers: dict[str, dict[str, str]]
) -> list[str]:
    """Tensor names that are stored verbatim (no quantisation sidecars)."""
    plain: list[str] = []
    for name in source.entries:
        if any(name.endswith(suffix) for suffix in QUANT_SIDECAR_SUFFIXES):
            continue
        if name.endswith(".weight") and name[: -len(".weight")] in quant_layers:
            continue
        plain.append(name)
    return sorted(plain)


def is_connector(layer: str) -> bool:
    return any(marker in layer for marker in CONNECTOR_MARKERS)


def infer_kind(sidecars: dict[str, str]) -> str:
    if "weight_scale" in sidecars:
        return "int8_tensorwise"
    if "weight_s_rel" in sidecars:
        return "asym_w4a8_int8"
    raise ValueError(f"cannot infer quantisation format from sidecars {sorted(sidecars)}")


def assumed_marker(kind: str) -> dict:
    """Marker the plan says to assume when the real one is in the missing tail."""
    if kind == "int8_tensorwise":
        return {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
            "group_size": None,
        }
    return {
        "format": "asym_w4a8_int8",
        "convrot": True,
        "convrot_groupsize": 256,
        "group_size": 16,
    }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def compare_matrices(got: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Chunked float64 accumulation of the four comparison metrics."""
    rows, cols = reference.shape
    sum_ab = sum_aa = sum_bb = sum_dd = sum_a = sum_b = 0.0
    norms_a = np.empty(rows, dtype=np.float64)
    norms_b = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, _METRIC_CHUNK_ROWS):
        stop = min(start + _METRIC_CHUNK_ROWS, rows)
        a = got[start:stop].astype(np.float64)
        b = reference[start:stop].astype(np.float64)
        diff = a - b
        sum_ab += float((a * b).sum())
        sum_aa += float((a * a).sum())
        sum_bb += float((b * b).sum())
        sum_dd += float((diff * diff).sum())
        sum_a += float(a.sum())
        sum_b += float(b.sum())
        norms_a[start:stop] = np.sqrt((a * a).sum(axis=1))
        norms_b[start:stop] = np.sqrt((b * b).sum(axis=1))

    count = float(rows * cols)
    cos = sum_ab / math_sqrt(sum_aa * sum_bb) if sum_aa > 0 and sum_bb > 0 else float("nan")
    rel_rmse = math_sqrt(sum_dd / sum_bb) if sum_bb > 0 else float("nan")
    var_a = max(sum_aa / count - (sum_a / count) ** 2, 0.0)
    var_b = max(sum_bb / count - (sum_b / count) ** 2, 0.0)
    std_a, std_b = math_sqrt(var_a), math_sqrt(var_b)
    std_ratio = std_a / std_b if std_b > 0 else float("nan")

    if rows >= 2 and norms_a.std() > 0 and norms_b.std() > 0:
        row_norm_r = float(np.corrcoef(norms_a, norms_b)[0, 1])
    else:
        row_norm_r = float("nan")

    return {
        "cos": float(cos),
        "rel_rmse": float(rel_rmse),
        "std_ratio": float(std_ratio),
        "row_norm_r": row_norm_r,
        "std_dequantised": std_a,
        "std_official": std_b,
    }


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


# --------------------------------------------------------------------------
# per-layer work
# --------------------------------------------------------------------------


def compare_quant_layer(
    layer: str,
    sidecars: dict[str, str],
    community: SafetensorsStream,
    official: SafetensorsStream,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "layer": layer,
        "component": "connector" if is_connector(layer) else "transformer",
        "status": "ok",
    }
    weight_name = f"{layer}.weight"
    try:
        kind = infer_kind(sidecars)
    except ValueError as exc:
        record.update(status="error", error=str(exc))
        return record
    record["format"] = kind

    needed = [weight_name] + [
        name for key, name in sidecars.items() if key != "comfy_quant"
    ]
    missing = [name for name in needed if name not in community]
    if missing:
        record.update(status="error", error=f"community file lacks {missing}")
        return record
    unreadable = [name for name in needed if not community.readable(name)]
    if unreadable:
        record.update(
            status="unreadable",
            unreadable_tensors=sorted(unreadable),
            note="tensor lies in the truncated tail of the community file",
        )
        return record
    if weight_name not in official:
        record.update(status="error", error="official file lacks the reference weight")
        return record

    marker_name = sidecars.get("comfy_quant")
    if marker_name is not None and community.readable(marker_name):
        try:
            marker = parse_quant_marker(community.array(marker_name))
        except ComfyDequantError as exc:
            record.update(status="error", error=f"comfy_quant marker rejected: {exc}")
            return record
        record["marker_assumed"] = False
        if marker["format"] != kind:
            record.update(
                status="error",
                error=f"marker says {marker['format']!r} but sidecars say {kind!r}",
            )
            return record
    else:
        marker = assumed_marker(kind)
        record["marker_assumed"] = True
    record["marker"] = marker

    weight = community.array(weight_name)
    in_features = int(weight.shape[1]) * (2 if kind == "asym_w4a8_int8" else 1)
    official_shape = official.shape_of(weight_name)
    record["shape"] = official_shape
    if list(official_shape) != [int(weight.shape[0]), in_features]:
        record.update(
            status="error",
            error=(
                f"shape mismatch: community implies {[int(weight.shape[0]), in_features]}, "
                f"official is {official_shape}"
            ),
        )
        return record

    tensors: dict[str, np.ndarray] = {"weight": weight}
    for key, name in sidecars.items():
        if key == "comfy_quant":
            continue
        tensors[key] = community.array(name)

    try:
        restored = dequantize_layer(marker, tensors, in_features, layer_name=layer)
    except ComfyDequantError as exc:
        record.update(status="error", error=f"dequantisation failed: {exc}")
        return record

    reference = official.array(weight_name)
    record["metrics"] = compare_matrices(restored, reference)
    return record


def compare_plain_tensor(
    name: str, community: SafetensorsStream, official: SafetensorsStream
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "tensor": name,
        "component": "connector" if is_connector(name) else "transformer",
        "dtype": community.dtype_of(name),
        "status": "ok",
    }
    if name not in official:
        record.update(status="error", error="official file lacks this tensor")
        return record
    if not community.readable(name):
        record.update(status="unreadable")
        return record
    record["dtype_official"] = official.dtype_of(name)
    if community.dtype_of(name) == official.dtype_of(name):
        record["byte_identical"] = community.raw_bytes(name) == official.raw_bytes(name)
        return record
    # The community tooling stores some of the official F32 tensors (the
    # scale_shift tables) as BF16.  Byte identity cannot hold there, so compare
    # values after promoting both sides to float32 instead.
    record["dtype_differs"] = True
    record["byte_identical"] = None
    got = community.array(name).astype(np.float32, copy=False)
    ref = official.array(name).astype(np.float32, copy=False)
    if got.shape != ref.shape:
        record.update(status="error", error=f"shape {got.shape} vs official {ref.shape}")
        return record
    record["max_abs_diff"] = float(np.abs(got - ref).max()) if got.size else 0.0
    record["value_identical"] = bool(record["max_abs_diff"] == 0.0)
    return record


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def _gate(status: str, checked: int, failures: list[dict], **detail: Any) -> dict[str, Any]:
    return {
        "status": status,
        "checked": checked,
        "failed": len(failures),
        "failures": failures[:50],
        **detail,
    }


def evaluate_gates(
    layer_records: list[dict[str, Any]],
    plain_records: list[dict[str, Any]],
    counts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    ok_layers = [r for r in layer_records if r["status"] == "ok"]
    connector_bf16 = [
        r for r in plain_records if r["component"] == "connector" and r["dtype"] == "BF16"
    ]
    gates: dict[str, dict[str, Any]] = {}

    # G-0 -- coverage.  Every gate below is evaluated over the layers that were
    # actually compared, so on an incomplete source they would all pass on a
    # handful of survivors.  This gate is what makes the run's exit code mean
    # "the whole selection was checked against the marker text the file really
    # carries": an errored layer, an unreadable layer or an assumed marker each
    # fail it.
    selected = counts["quant_layers_selected"]
    failures = [
        {"counter": name, "count": counts[name]}
        for name in ("quant_layers_error", "quant_layers_unreadable", "markers_assumed")
        if counts[name]
    ]
    gates["G-0"] = _gate(
        "skipped" if not selected else ("pass" if not failures else "fail"),
        selected,
        failures,
        errored=counts["quant_layers_error"],
        unreadable=counts["quant_layers_unreadable"],
        markers_assumed=counts["markers_assumed"],
        description=(
            "coverage: every selected layer compared, with its own readable comfy_quant marker"
        ),
    )

    # G-A -- connector plain BF16 byte identity
    checked = [r for r in connector_bf16 if r["status"] == "ok"]
    failures = [
        {"tensor": r["tensor"], "byte_identical": r.get("byte_identical"), "error": r.get("error")}
        for r in connector_bf16
        if r["status"] != "ok" or not r.get("byte_identical")
    ]
    gates["G-A"] = _gate(
        "skipped" if not checked else ("pass" if not failures else "fail"),
        len(checked),
        failures,
        description="connector plain BF16 tensors byte-identical to the official file",
    )

    # G-B -- connector w4a8 cos >= 0.99
    subset = [
        r
        for r in ok_layers
        if r["component"] == "connector" and r["format"] == "asym_w4a8_int8"
    ]
    failures = [
        {"layer": r["layer"], "cos": r["metrics"]["cos"]}
        for r in subset
        if not r["metrics"]["cos"] >= G_B_MIN_COS
    ]
    gates["G-B"] = _gate(
        "skipped" if not subset else ("pass" if not failures else "fail"),
        len(subset),
        failures,
        threshold=G_B_MIN_COS,
        description="connector w4a8 layers: cos >= 0.99",
        **_distribution([r["metrics"]["cos"] for r in subset]),
    )

    # G-C -- every quantised layer cos >= 0.5
    failures = [
        {"layer": r["layer"], "cos": r["metrics"]["cos"]}
        for r in ok_layers
        if not r["metrics"]["cos"] >= G_C_MIN_COS
    ]
    gates["G-C"] = _gate(
        "skipped" if not ok_layers else ("pass" if not failures else "fail"),
        len(ok_layers),
        failures,
        threshold=G_C_MIN_COS,
        description="every quantised layer: cos >= 0.5",
        **_distribution([r["metrics"]["cos"] for r in ok_layers]),
    )

    # G-D -- transformer quantised layers: median and 1st percentile of cos
    subset = [r for r in ok_layers if r["component"] == "transformer"]
    cosines = [r["metrics"]["cos"] for r in subset]
    median = statistics.median(cosines) if cosines else float("nan")
    p1 = _percentile(cosines, 1.0)
    failures = []
    if cosines:
        if not median >= G_D_MIN_MEDIAN_COS:
            failures.append({"statistic": "median", "value": median, "threshold": G_D_MIN_MEDIAN_COS})
        if not p1 >= G_D_MIN_P1_COS:
            failures.append({"statistic": "p1", "value": p1, "threshold": G_D_MIN_P1_COS})
    gates["G-D"] = _gate(
        "skipped" if not cosines else ("pass" if not failures else "fail"),
        len(subset),
        failures,
        description="transformer quantised layers: cos median >= 0.90, 1st percentile >= 0.60",
        **_distribution(cosines),
    )

    # G-E -- std ratio band
    low, high = G_E_STD_RATIO_RANGE
    failures = [
        {"layer": r["layer"], "std_ratio": r["metrics"]["std_ratio"]}
        for r in ok_layers
        if not (low <= r["metrics"]["std_ratio"] <= high)
    ]
    gates["G-E"] = _gate(
        "skipped" if not ok_layers else ("pass" if not failures else "fail"),
        len(ok_layers),
        failures,
        threshold=list(G_E_STD_RATIO_RANGE),
        description="every quantised layer: 0.5 <= std_ratio <= 2.0",
        **_distribution([r["metrics"]["std_ratio"] for r in ok_layers], prefix="std_ratio_"),
    )

    # G-F -- row-norm correlation
    failures = [
        {"layer": r["layer"], "row_norm_r": r["metrics"]["row_norm_r"]}
        for r in ok_layers
        if not r["metrics"]["row_norm_r"] >= G_F_MIN_ROW_NORM_R
    ]
    gates["G-F"] = _gate(
        "skipped" if not ok_layers else ("pass" if not failures else "fail"),
        len(ok_layers),
        failures,
        threshold=G_F_MIN_ROW_NORM_R,
        description="every quantised layer: row-norm correlation >= 0.8",
        **_distribution([r["metrics"]["row_norm_r"] for r in ok_layers], prefix="row_norm_r_"),
    )
    return gates


def _distribution(values: list[float], prefix: str = "cos_") -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}min": float(arr.min()),
        f"{prefix}p1": float(np.percentile(arr, 1.0)),
        f"{prefix}median": float(np.median(arr)),
        f"{prefix}max": float(arr.max()),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_summary(report: dict[str, Any]) -> None:
    counts = report["counts"]
    print("")
    print("=" * 78)
    print("comfyquant vs official -- summary")
    print("=" * 78)
    print(f"community : {report['community']['path']}")
    print(
        f"            {report['community']['size']:,} B on disk; header declares "
        f"{report['community']['max_data_end']:,} B of payload "
        f"({report['community']['missing_bytes']:,} B missing)"
    )
    print(f"official  : {report['official']['path']}")
    print("")
    print("layers")
    print(f"  quantised layers in file      : {counts['quant_layers_total']}")
    print(f"  selected by filters           : {counts['quant_layers_selected']}")
    print(f"  compared                      : {counts['quant_layers_compared']}")
    print(f"  skipped (truncated tail)      : {counts['quant_layers_unreadable']}")
    print(f"  errored                       : {counts['quant_layers_error']}")
    print(
        f"    int8_tensorwise / asym_w4a8  : {counts['compared_int8']} / "
        f"{counts['compared_w4a8']}"
    )
    print(
        f"    markers parsed / assumed     : {counts['markers_parsed']} / "
        f"{counts['markers_assumed']}"
    )
    print("plain tensors")
    print(f"  connector BF16 compared       : {counts['connector_plain_compared']}")
    print(f"  connector BF16 byte-identical : {counts['connector_plain_identical']}")
    print(f"  connector BF16 unreadable     : {counts['connector_plain_unreadable']}")
    print("")
    print("gates")
    for name in ("G-0", "G-A", "G-B", "G-C", "G-D", "G-E", "G-F"):
        gate = report["gates"][name]
        line = f"  {name}  {gate['status'].upper():8s} checked={gate['checked']:5d} failed={gate['failed']:4d}  {gate['description']}"
        print(line)
        if name == "G-0":
            print(
                f"        errored={gate['errored']}  unreadable={gate['unreadable']}  "
                f"markers_assumed={gate['markers_assumed']}"
            )
    print("")
    all_cos = report["distributions"]["cos"]
    all_std = report["distributions"]["std_ratio"]
    all_rr = report["distributions"]["row_norm_r"]
    all_rmse = report["distributions"]["rel_rmse"]
    if all_cos:
        print("distributions over every compared layer")
        print(
            f"  cos        median={all_cos['median']:.6f}  min={all_cos['min']:.6f}  "
            f"p1={all_cos['p1']:.6f}  max={all_cos['max']:.6f}"
        )
        print(
            f"  std_ratio  median={all_std['median']:.6f}  min={all_std['min']:.6f}  "
            f"max={all_std['max']:.6f}"
        )
        print(
            f"  row_norm_r median={all_rr['median']:.6f}  min={all_rr['min']:.6f}  "
            f"max={all_rr['max']:.6f}"
        )
        print(
            f"  rel_rmse   median={all_rmse['median']:.6f}  min={all_rmse['min']:.6f}  "
            f"max={all_rmse['max']:.6f}"
        )
    print("")
    worst = report["worst_layers"]
    if worst:
        print(f"lowest-cos layers (first {len(worst)})")
        for row in worst:
            print(
                f"  cos={row['cos']:.6f}  std_ratio={row['std_ratio']:.4f}  "
                f"row_norm_r={row['row_norm_r']:.4f}  {row['format']:15s} {row['layer']}"
            )
    failing = report["failing_layers"]
    print("")
    if failing:
        print(f"layers failing at least one per-layer gate ({len(failing)} total, first 20)")
        for row in failing[:20]:
            print(
                f"  {','.join(row['gates']):12s} cos={row['cos']:.6f} "
                f"std_ratio={row['std_ratio']:.4f} row_norm_r={row['row_norm_r']:.4f} {row['layer']}"
            )
    else:
        print("no layer failed a per-layer gate")
    print("")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare dequantised community LTX 2.5 weights with the official bf16 file."
    )
    parser.add_argument("--community", required=True, type=Path, help="community safetensors path")
    parser.add_argument("--official", required=True, type=Path, help="official bf16 safetensors path")
    parser.add_argument("--report", required=True, type=Path, help="JSON report output path")
    parser.add_argument("--limit", type=int, default=None, help="compare at most N quantised layers")
    parser.add_argument("--layers", type=str, default=None, help="regex filter on layer names")
    parser.add_argument(
        "--progress-every", type=int, default=50, help="print a progress line every N layers"
    )
    args = parser.parse_args()

    pattern = re.compile(args.layers) if args.layers else None
    started = time.time()

    with SafetensorsStream(args.community) as community, SafetensorsStream(
        args.official
    ) as official:
        quant_layers = collect_quant_layers(community)
        plain_names = collect_plain_tensors(community, quant_layers)

        selected = sorted(quant_layers)
        if pattern is not None:
            selected = [name for name in selected if pattern.search(name)]
        if args.limit is not None:
            selected = selected[: args.limit]

        layer_records: list[dict[str, Any]] = []
        for index, layer in enumerate(selected, start=1):
            layer_records.append(
                compare_quant_layer(layer, quant_layers[layer], community, official)
            )
            if args.progress_every and index % args.progress_every == 0:
                done = sum(1 for r in layer_records if r["status"] == "ok")
                print(
                    f"[{index}/{len(selected)}] compared={done} "
                    f"elapsed={time.time() - started:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )

        plain_selected = plain_names
        if pattern is not None:
            plain_selected = [name for name in plain_names if pattern.search(name)]
        plain_records = [
            compare_plain_tensor(name, community, official) for name in plain_selected
        ]

        community_info = {
            "path": str(community.path),
            "size": community.size,
            "header_bytes": community.base - 8,
            "payload_available": community.payload_available,
            "max_data_end": community.max_data_end,
            "missing_bytes": community.missing_bytes,
            "tensor_count": len(community.entries),
            "metadata_keys": sorted(community.metadata),
            "quant_format": community.metadata.get("quant_format"),
        }
        official_info = {
            "path": str(official.path),
            "size": official.size,
            "header_bytes": official.base - 8,
            "tensor_count": len(official.entries),
        }

    ok_layers = [r for r in layer_records if r["status"] == "ok"]
    counts = {
        "quant_layers_total": len(quant_layers),
        "quant_layers_selected": len(selected),
        "quant_layers_compared": len(ok_layers),
        "quant_layers_unreadable": sum(1 for r in layer_records if r["status"] == "unreadable"),
        "quant_layers_error": sum(1 for r in layer_records if r["status"] == "error"),
        "compared_int8": sum(1 for r in ok_layers if r["format"] == "int8_tensorwise"),
        "compared_w4a8": sum(1 for r in ok_layers if r["format"] == "asym_w4a8_int8"),
        "markers_parsed": sum(1 for r in ok_layers if r.get("marker_assumed") is False),
        "markers_assumed": sum(1 for r in ok_layers if r.get("marker_assumed") is True),
        "plain_tensors_total": len(plain_names),
        "plain_tensors_compared": sum(1 for r in plain_records if r["status"] == "ok"),
        "connector_plain_total": sum(1 for r in plain_records if r["component"] == "connector"),
        "connector_plain_compared": sum(
            1 for r in plain_records if r["component"] == "connector" and r["status"] == "ok"
        ),
        "connector_plain_identical": sum(
            1 for r in plain_records if r["component"] == "connector" and r.get("byte_identical")
        ),
        "connector_plain_unreadable": sum(
            1 for r in plain_records if r["component"] == "connector" and r["status"] == "unreadable"
        ),
    }

    gates = evaluate_gates(layer_records, plain_records, counts)

    distributions = {
        "cos": _distribution([r["metrics"]["cos"] for r in ok_layers], prefix=""),
        "std_ratio": _distribution([r["metrics"]["std_ratio"] for r in ok_layers], prefix=""),
        "row_norm_r": _distribution([r["metrics"]["row_norm_r"] for r in ok_layers], prefix=""),
        "rel_rmse": _distribution([r["metrics"]["rel_rmse"] for r in ok_layers], prefix=""),
    }

    worst = sorted(ok_layers, key=lambda r: r["metrics"]["cos"])[:20]
    worst_rows = [
        {
            "layer": r["layer"],
            "format": r["format"],
            "component": r["component"],
            "cos": r["metrics"]["cos"],
            "rel_rmse": r["metrics"]["rel_rmse"],
            "std_ratio": r["metrics"]["std_ratio"],
            "row_norm_r": r["metrics"]["row_norm_r"],
        }
        for r in worst
    ]

    low, high = G_E_STD_RATIO_RANGE
    failing_layers = []
    for record in ok_layers:
        metrics = record["metrics"]
        failed = []
        if not metrics["cos"] >= G_C_MIN_COS:
            failed.append("G-C")
        if record["component"] == "connector" and record["format"] == "asym_w4a8_int8":
            if not metrics["cos"] >= G_B_MIN_COS:
                failed.append("G-B")
        if not (low <= metrics["std_ratio"] <= high):
            failed.append("G-E")
        if not metrics["row_norm_r"] >= G_F_MIN_ROW_NORM_R:
            failed.append("G-F")
        if failed:
            failing_layers.append(
                {
                    "layer": record["layer"],
                    "gates": failed,
                    "cos": metrics["cos"],
                    "std_ratio": metrics["std_ratio"],
                    "row_norm_r": metrics["row_norm_r"],
                }
            )
    failing_layers.sort(key=lambda row: row["cos"])

    report = {
        "format": REPORT_FORMAT,
        "generated_unix": int(time.time()),
        "elapsed_seconds": round(time.time() - started, 1),
        "filters": {"limit": args.limit, "layers": args.layers},
        "community": community_info,
        "official": official_info,
        "counts": counts,
        "gates": gates,
        "distributions": distributions,
        "worst_layers": worst_rows,
        "failing_layers": failing_layers,
        "layers": layer_records,
        "plain_tensors": plain_records,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")

    print_summary(report)
    print(f"report written to {args.report}")

    hard_failures = [name for name, gate in gates.items() if gate["status"] == "fail"]
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
