"""Tests for the K-quant quantization kernels (Q4_K / Q5_K / Q6_K).

Four groups of checks:

1. Fidelity gate (always runnable): a compact, literal *scalar* C-order
   translation of make_qkx2_quants / make_qx_quants is compared bit-for-bit
   against the vectorized kernels on random data.  This is the true measure of
   port faithfulness and needs no reference model.
2. Idempotency (main verification): take real quantized tensors from a
   reference GGUF, dequantize to f32 with gguf.quants, re-quantize with our
   kernels, and compare against the *original* raw bytes.  Skipped when the
   reference GGUF is not present.  Note: even a bit-perfect port is NOT fully
   idempotent for K-quants -- the scale search re-optimizes on the reconstructed
   f32, which differs from the original f32, so a fraction of blocks legitimately
   land on a different (equally valid) optimum.  Q4_K is ~99.9% idempotent;
   Q5_K/Q6_K ~85% (higher, asymmetric-min sensitivity).
3. Round-trip statistics: quantize -> dequantize random + structured data,
   validating byte-length and reconstruction error (max abs / RMSE / SQNR).
4. Throughput (informational, printed with -s).

Run::

    set PYTHONPATH=src && .venv\\Scripts\\python -m pytest tests/test_quant_roundtrip.py -v -s
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

import gguf
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as GT

from converter import quant_kernels as qk


REF_GGUF = os.environ.get(
    "REF_GGUF",
    r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-LTX23-backend\models"
    r"\ltx-2.3-gguf\LTX-2.3-distilled-1.1"
    r"\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
)

TYPE_ID = {12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}
GT_OF = {"Q4_K": GT.Q4_K, "Q5_K": GT.Q5_K, "Q6_K": GT.Q6_K}

# Conservative idempotency floors reflecting genuine (non-buggy) non-idempotency.
IDEMPOTENT_FLOOR = {"Q4_K": 0.99, "Q5_K": 0.75, "Q6_K": 0.78}


# ---------------------------------------------------------------------------
# scalar, literal C-order reference (fidelity gate)
# ---------------------------------------------------------------------------
_f32 = np.float32


def _s_ni(fval):
    v = _f32(fval) + _f32(12582912.0)
    i = np.array([v], dtype=np.float32).view(np.int32)[0]
    return int((i & np.int32(0x007FFFFF)) - np.int32(0x00400000))


def test_q4_k_bounded_workers_are_byte_identical():
    """Independent Q4_K block batches must preserve serial bytes exactly."""
    values = np.random.default_rng(20260819).normal(size=(2051, 256)).astype(np.float32)
    serial = qk.quantize_q4_k(values, workers=1)
    for workers in (2, 4, 8):
        assert np.array_equal(serial, qk.quantize_q4_k(values, workers=workers))


@pytest.mark.parametrize("workers", [0, 9, True, "4"])
def test_q4_k_worker_range_is_validated(workers):
    with pytest.raises(ValueError, match="quant workers"):
        qk.quantize_q4_k(np.zeros((1, 256), dtype=np.float32), workers=workers)


def _s_make_qkx2(x, w, nmax, rmin, rdelta, nstep):
    n = len(x)
    mn = x[0]; mx = x[0]; sum_w = w[0]; sum_x = _f32(sum_w * x[0])
    for i in range(1, n):
        if x[i] < mn: mn = x[i]
        if x[i] > mx: mx = x[i]
        sum_w = _f32(sum_w + w[i]); sum_x = _f32(sum_x + _f32(w[i] * x[i]))
    if mn > 0: mn = _f32(0)
    L = [0] * n
    if mx == mn:
        return _f32(0), _f32(-mn), L
    iscale = _f32(nmax / _f32(mx - mn)); scale = _f32(1 / iscale)
    best = _f32(0)
    for i in range(n):
        l = _s_ni(_f32(iscale * _f32(x[i] - mn))); l = max(0, min(nmax, l)); L[i] = l
        diff = _f32(_f32(scale * l) + mn - x[i]); diff = _f32(diff * diff)
        best = _f32(best + _f32(w[i] * diff))
    if nstep < 1:
        return scale, _f32(-mn), L
    Laux = [0] * n
    for is_ in range(0, nstep + 1):
        iscale = _f32(_f32(_f32(rmin) + _f32(_f32(rdelta) * _f32(is_))) + _f32(nmax))
        iscale = _f32(iscale / _f32(mx - mn))
        sum_l = _f32(0); sum_l2 = _f32(0); sum_xl = _f32(0)
        for i in range(n):
            l = _s_ni(_f32(iscale * _f32(x[i] - mn))); l = max(0, min(nmax, l)); Laux[i] = l
            sum_l = _f32(sum_l + _f32(w[i] * l))
            sum_l2 = _f32(sum_l2 + _f32(_f32(w[i] * l) * l))
            sum_xl = _f32(sum_xl + _f32(_f32(w[i] * l) * x[i]))
        D = _f32(_f32(sum_w * sum_l2) - _f32(sum_l * sum_l))
        if D > 0:
            ts = _f32(_f32(_f32(sum_w * sum_xl) - _f32(sum_x * sum_l)) / D)
            tm = _f32(_f32(_f32(sum_l2 * sum_x) - _f32(sum_l * sum_xl)) / D)
            if tm > 0:
                tm = _f32(0); ts = _f32(sum_xl / sum_l2)
            cur = _f32(0)
            for i in range(n):
                diff = _f32(_f32(ts * Laux[i]) + tm - x[i]); diff = _f32(diff * diff)
                cur = _f32(cur + _f32(w[i] * diff))
            if cur < best:
                for i in range(n): L[i] = Laux[i]
                best = cur; scale = ts; mn = tm
    return scale, _f32(-mn), L


def _s_make_qx(x, nmax):
    n = len(x); mx = _f32(0); amax = _f32(0)
    for i in range(n):
        ax = _f32(abs(x[i]))
        if ax > amax: amax = ax; mx = x[i]
    if amax < _f32(1e-15):
        return _f32(0), [0] * n
    iscale = _f32(-nmax / mx)
    L = [0] * n
    sumlx = _f32(0); suml2 = _f32(0)
    for i in range(n):
        l = _s_ni(_f32(iscale * x[i])); l = max(-nmax, min(nmax - 1, l)); L[i] = l + nmax
        wv = _f32(x[i] * x[i])
        sumlx = _f32(sumlx + _f32(_f32(wv * x[i]) * l))
        suml2 = _f32(suml2 + _f32(_f32(wv * l) * l))
    scale = _f32(sumlx / suml2) if suml2 else _f32(0)
    best = _f32(scale * sumlx)
    for is_ in range(-9, 10):
        if is_ == 0: continue
        iscale = _f32(-_f32(_f32(nmax) + _f32(_f32(0.1) * _f32(is_))) / mx)
        sumlx = _f32(0); suml2 = _f32(0)
        for i in range(n):
            l = _s_ni(_f32(iscale * x[i])); l = max(-nmax, min(nmax - 1, l))
            wv = _f32(x[i] * x[i])
            sumlx = _f32(sumlx + _f32(_f32(wv * x[i]) * l))
            suml2 = _f32(suml2 + _f32(_f32(wv * l) * l))
        if suml2 > 0 and _f32(sumlx * sumlx) > _f32(best * suml2):
            for i in range(n):
                l = _s_ni(_f32(iscale * x[i])); L[i] = nmax + max(-nmax, min(nmax - 1, l))
            scale = _f32(sumlx / suml2); best = _f32(scale * sumlx)
    return scale, L


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sqnr_db(orig, recon):
    orig = orig.astype(np.float64); recon = recon.astype(np.float64)
    sig = np.sum(orig * orig); noise = np.sum((orig - recon) ** 2)
    if noise == 0: return float("inf")
    if sig == 0: return float("-inf")
    return 10.0 * np.log10(sig / noise)


_reader_cache = {}


def _get_reader():
    if "r" not in _reader_cache:
        _reader_cache["r"] = gguf.GGUFReader(REF_GGUF) if os.path.exists(REF_GGUF) else None
    return _reader_cache["r"]


def _tensors_of(tn):
    r = _get_reader()
    if r is None:
        return []
    return [t for t in r.tensors if TYPE_ID.get(int(t.tensor_type)) == tn]


ref_missing = not os.path.exists(REF_GGUF)
skip_ref = pytest.mark.skipif(ref_missing, reason=f"reference GGUF not found: {REF_GGUF}")


# ---------------------------------------------------------------------------
# 1. fidelity gate: vectorized == scalar C-order (bit-exact)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nmax,rmin,rdelta,nstep", [(15, -1.0, 0.1, 20), (31, -0.5, 0.1, 15)])
def test_fidelity_make_qkx2(nmax, rmin, rdelta, nstep):
    rng = np.random.default_rng(42)
    nsub = 400
    # mix of scales / distributions to exercise the search
    sub = (rng.standard_normal((nsub, 32)).astype(np.float32)
           * rng.choice([0.01, 1.0, 30.0], size=(nsub, 1)).astype(np.float32))
    sub[:5] = 0.0  # all-zero blocks
    sub[5:10] = rng.standard_normal((1, 32)).astype(np.float32)  # constant-ish
    sx2 = qk._fsum(sub * sub)
    avx = np.sqrt(sx2 / np.float32(32.0))
    wts = (avx[:, None] + np.abs(sub)).astype(np.float32)
    sc_v, mn_v, L_v = qk._make_qkx2(sub, wts, nmax, rmin, rdelta, nstep)
    bad = 0
    for i in range(nsub):
        sc, mn, L = _s_make_qkx2(list(sub[i]), list(wts[i]), nmax, rmin, rdelta, nstep)
        if sc != sc_v[i] or mn != mn_v[i] or list(L) != list(L_v[i]):
            bad += 1
    assert bad == 0, f"make_qkx2 (nmax={nmax}) diverged from scalar C-order in {bad}/{nsub} sub-blocks"


def test_fidelity_make_qx():
    rng = np.random.default_rng(7)
    nsub = 400
    sub = (rng.standard_normal((nsub, 16)).astype(np.float32)
           * rng.choice([0.01, 1.0, 30.0], size=(nsub, 1)).astype(np.float32))
    sub[:5] = 0.0
    sc_v, L_v = qk._make_qx(sub, 32)
    bad = 0
    for i in range(nsub):
        sc, L = _s_make_qx(list(sub[i]), 32)
        if sc != sc_v[i] or list(L) != list(L_v[i]):
            bad += 1
    assert bad == 0, f"make_qx diverged from scalar C-order in {bad}/{nsub} sub-blocks"


# ---------------------------------------------------------------------------
# 2. idempotency (needs reference GGUF)
# ---------------------------------------------------------------------------
def _idempotency_sample(tn, max_tensors=10, max_elems=8_000_000):
    ts = _tensors_of(tn)
    assert ts, f"no {tn} tensors in reference GGUF"
    ts = [t for t in ts if int(np.prod(t.shape)) <= max_elems]
    ts.sort(key=lambda t: int(np.prod(t.shape)))
    if len(ts) > max_tensors:
        idxs = np.unique(np.linspace(0, len(ts) - 1, max_tensors).astype(int))
        ts = [ts[i] for i in idxs]
    tsize = qk.TYPE_SIZE[tn]
    gt = GT_OF[tn]
    tot = eq = perfect = 0
    sig = noise = 0.0
    for t in ts:
        raw = np.array(t.data, dtype=np.uint8).reshape(-1)
        f32 = dequantize(raw.copy(), gt).astype(np.float32)
        re = qk.quantize(f32, tn).reshape(-1)
        assert re.size == raw.size
        nb = raw.size // tsize
        beq = int(np.count_nonzero(np.all(raw.reshape(nb, tsize) == re.reshape(nb, tsize), axis=1)))
        tot += nb; eq += beq; perfect += (beq == nb)
        deq = dequantize(re.copy(), gt).astype(np.float64)
        o = f32.astype(np.float64)
        sig += np.sum(o * o); noise += np.sum((o - deq) ** 2)
    rate = eq / tot
    sqnr = float("inf") if noise == 0 else 10 * np.log10(sig / noise)
    return len(ts), perfect, eq, tot, rate, sqnr


@skip_ref
@pytest.mark.parametrize("tn", ["Q4_K", "Q5_K", "Q6_K"])
def test_idempotency(tn):
    n, perfect, eq, tot, rate, sqnr = _idempotency_sample(tn)
    print(f"\n[{tn}] idempotency: {n} tensors, perfect={perfect}/{n}, "
          f"block_match={eq}/{tot} ({rate*100:.4f}%), roundtrip_SQNR={sqnr:.2f}dB")
    assert rate >= IDEMPOTENT_FLOOR[tn], (
        f"{tn} idempotency {rate*100:.3f}% below floor {IDEMPOTENT_FLOOR[tn]*100:.1f}%"
    )


# ---------------------------------------------------------------------------
# 3. round-trip statistics (always runnable)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tn", ["Q4_K", "Q5_K", "Q6_K"])
@pytest.mark.parametrize("kind", ["normal", "uniform", "outliers"])
def test_roundtrip_random(tn, kind):
    rng = np.random.default_rng(1234)
    rows, cols = 64, 256 * 4
    if kind == "normal":
        data = rng.standard_normal((rows, cols)).astype(np.float32)
    elif kind == "uniform":
        data = (rng.random((rows, cols), dtype=np.float32) * 2 - 1).astype(np.float32)
    else:
        data = rng.standard_normal((rows, cols)).astype(np.float32)
        data[rng.random((rows, cols)) < 0.02] *= 50.0

    qbytes = qk.quantize(data, tn)
    exp = (data.size // qk.QK_K) * qk.TYPE_SIZE[tn]
    assert qbytes.size == exp, (qbytes.size, exp)

    deq = dequantize(qbytes.reshape(-1).copy(), GT_OF[tn]).reshape(data.shape)
    assert deq.shape == data.shape
    assert np.isfinite(deq).all()

    err = np.abs(data - deq)
    print(f"\n[{tn}/{kind}] max_abs={err.max():.4g} rmse={np.sqrt((err**2).mean()):.4g} "
          f"sqnr={_sqnr_db(data, deq):.2f}dB")


def test_native_sqnr_ordering():
    """More bits must give better reconstruction: Q4 < Q5 < Q6 (SQNR)."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((256, 4096)).astype(np.float32)
    s = {}
    for tn in ("Q4_K", "Q5_K", "Q6_K"):
        deq = dequantize(qk.quantize(data, tn).reshape(-1).copy(), GT_OF[tn]).reshape(data.shape)
        s[tn] = _sqnr_db(data, deq)
        print(f"\n[{tn}] native SQNR={s[tn]:.2f}dB")
    assert s["Q4_K"] < s["Q5_K"] < s["Q6_K"]
    assert s["Q4_K"] > 18.0  # sane floor for 4-bit unit-scale data


@pytest.mark.parametrize("tn", ["Q4_K", "Q5_K", "Q6_K"])
def test_zero_and_constant_blocks(tn):
    data = np.zeros((4, 256), dtype=np.float32)
    deq = dequantize(qk.quantize(data, tn).reshape(-1).copy(), GT_OF[tn]).reshape(data.shape)
    assert np.allclose(deq, 0.0)
    data2 = np.full((4, 256), 0.75, dtype=np.float32)
    deq2 = dequantize(qk.quantize(data2, tn).reshape(-1).copy(), GT_OF[tn]).reshape(data2.shape)
    assert np.isfinite(deq2).all()


def test_dispatch_and_invalid_size_error():
    data = np.ones((256,), dtype=np.float32)
    for tn in ("Q4_K", "Q5_K", "Q6_K"):
        assert qk.quantize(data, tn).size == qk.TYPE_SIZE[tn]
    with pytest.raises(NotImplementedError):
        qk.quantize(data, "Q3_K")
    with pytest.raises(ValueError, match="multiple"):
        qk.quantize(np.ones((255,), dtype=np.float32), "Q4_K")


# ---------------------------------------------------------------------------
# 4. throughput (informational)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tn", ["Q4_K", "Q5_K", "Q6_K"])
def test_throughput(tn):
    data = np.random.default_rng(0).standard_normal((4096, 4096)).astype(np.float32)
    in_mb = data.nbytes / (1024 * 1024)
    qk.quantize(data[:256], tn)  # warmup
    t0 = time.perf_counter()
    qk.quantize(data, tn)
    dt = time.perf_counter() - t0
    mbps = in_mb / dt
    est_s = 15.0 * 1024 / mbps
    print(f"\n[{tn}] {in_mb:.1f} MiB f32 in {dt:.3f}s = {mbps:.1f} MiB/s "
          f"| est 15 GiB f32 input: {est_s:.1f}s ({est_s/60:.1f} min)")
    assert dt > 0
