"""K-quant (Q4_K / Q5_K / Q6_K) quantization kernels ported from llama.cpp.

gguf-py implements the *dequantization* of the K-quant families but not the
*quantization* (writing) side.  This module is a faithful NumPy port of the
reference C implementations in llama.cpp:

    ggml/src/ggml-quants.c
        quantize_row_q4_K_ref  (+ make_qkx2_quants, get_scale_min_k4 packing)
        quantize_row_q5_K_ref
        quantize_row_q6_K_ref  (+ make_qx_quants)

Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c
Ported: 2026-07-10 (llama.cpp master).  imatrix-free reference variant.

Portions derived from ggml-org/llama.cpp (MIT License, Copyright (c) 2023-2024
The ggml authors). See the repository's top-level ``LICENSE`` file for this
project's own license (Apache License 2.0); the MIT-licensed upstream algorithm
this module ports is unaffected by that choice.

Design notes
------------
* All arithmetic is done in float32 to mirror the C ``float`` code paths.  The
  scalar reduction sums in the C code are strictly sequential (no fast-math), so
  we accumulate reductions sequentially as well (``_fsum``) to reproduce the C
  rounding bit-for-bit.  NumPy's default ``sum`` uses an 8-accumulator unrolled
  reduction which can differ from the C order in the last ulp and occasionally
  flip a ``nearest_int`` decision at a tie boundary.
* ``nearest_int`` reproduces the ggml "magic number" round-half-to-even trick.
* fp16 storage uses ``astype(np.float16)`` which is round-half-to-even, matching
  ggml's reference GGML_FP32_TO_FP16.
* Everything is vectorized across super-blocks; the only Python loops are the
  fixed heuristic search steps (``nstep`` for qkx2, the -9..9 sweep for qx) and
  the sequential reduction helper.

Public API
----------
    quantize_q4_k(arr, workers=1, executor=None) -> np.ndarray[uint8]
    quantize_q5_k(arr) -> np.ndarray[uint8]
    quantize_q6_k(arr, workers=1, executor=None) -> np.ndarray[uint8]
    quantize(arr, ggml_type_name) -> np.ndarray[uint8]
"""

from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor

import numpy as np

QK_K = 256
K_SCALE_SIZE = 12
GROUP_MAX_EPS = np.float32(1e-15)
Q4_K_BLOCKS_PER_TASK = 1024
Q6_K_BLOCKS_PER_TASK = 1024
MAX_QUANT_WORKERS = 8

# Byte size of one 256-element super-block for each K-quant type.
TYPE_SIZE = {
    "Q4_K": 2 + 2 + K_SCALE_SIZE + QK_K // 2,            # 144
    "Q5_K": 2 + 2 + K_SCALE_SIZE + QK_K // 8 + QK_K // 2,  # 176
    "Q6_K": QK_K // 2 + QK_K // 4 + QK_K // 16 + 2,       # 210
}

_F32 = np.float32


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _nearest_int(fval: np.ndarray) -> np.ndarray:
    """Vectorized port of ggml ``nearest_int`` (round half to even).

    C reference::

        float val = fval + 12582912.f;      // 1.5 * 2^23
        int i; memcpy(&i, &val, sizeof(int));
        return (i & 0x007fffff) - 0x00400000;
    """
    val = np.ascontiguousarray(fval, dtype=np.float32) + np.float32(12582912.0)
    i = val.view(np.int32)
    return (i & np.int32(0x007FFFFF)) - np.int32(0x00400000)


def _fsum(a: np.ndarray) -> np.ndarray:
    """Sequential float32 reduction along the last axis (matches C accumulation).

    a: (..., n) float32 -> (...,) float32

    ``np.cumsum`` accumulates strictly left-to-right in float32 (rounding each
    partial sum to float32), so its final element is bit-identical to a naive
    sequential ``for``-loop accumulator -- i.e. exactly the C ``-ffast-math``-free
    reduction order -- but it runs in a single C call (~3x faster than a Python
    loop).  NumPy's plain ``sum`` uses an 8-accumulator unrolled reduction whose
    order differs and can flip ``nearest_int`` decisions at tie boundaries.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    return np.cumsum(a, axis=-1, dtype=np.float32)[..., -1]


def _f16_round(x: np.ndarray) -> np.ndarray:
    """Round a float32 array through fp16 and back (fp16 storage semantics)."""
    return x.astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# make_qkx2_quants  (used by Q4_K and Q5_K, use_mad = false)
# ---------------------------------------------------------------------------
def _make_qkx2(x, weights, nmax, rmin, rdelta, nstep):
    """Vectorized port of ``make_qkx2_quants`` (imatrix-free, use_mad=false).

    x, weights : (B, 32) float32
    returns (scale (B,), the_min (B,), L (B, 32) int32 in [0, nmax])
    """
    nmax_f = np.float32(nmax)
    xmin = np.minimum(x.min(axis=1), np.float32(0.0)).astype(np.float32)
    xmax = x.max(axis=1).astype(np.float32)

    sum_w = _fsum(weights)
    sum_x = _fsum(weights * x)

    eq = xmax == xmin  # "all equal" -> zero result

    denom0 = np.where(eq, np.float32(1.0), (xmax - xmin)).astype(np.float32)
    iscale = (nmax_f / denom0).astype(np.float32)
    scale = (np.float32(1.0) / iscale).astype(np.float32)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        l0 = _nearest_int(iscale[:, None] * (x - xmin[:, None]))
        L = np.clip(l0, 0, nmax).astype(np.int32)
        Lf = L.astype(np.float32)
        diff = (scale[:, None] * Lf + xmin[:, None] - x).astype(np.float32)
        best_error = _fsum(weights * diff * diff)

        min_arr = xmin.copy()
        scale_arr = scale.copy()
        L_arr = L.copy()

        for is_ in range(0, nstep + 1):
            r = np.float32(rdelta) * np.float32(is_)
            r = (np.float32(rmin) + r).astype(np.float32)
            r = (r + nmax_f).astype(np.float32)
            iscale2 = (r / (xmax - min_arr)).astype(np.float32)

            la = _nearest_int(iscale2[:, None] * (x - min_arr[:, None]))
            Laux_i = np.clip(la, 0, nmax).astype(np.int32)
            Laux = Laux_i.astype(np.float32)

            sum_l = _fsum(weights * Laux)
            sum_l2 = _fsum(weights * Laux * Laux)
            sum_xl = _fsum(weights * Laux * x)

            D = (sum_w * sum_l2 - sum_l * sum_l).astype(np.float32)
            Dpos = D > 0
            safeD = np.where(Dpos, D, np.float32(1.0)).astype(np.float32)

            this_scale = ((sum_w * sum_xl - sum_x * sum_l) / safeD).astype(np.float32)
            this_min = ((sum_l2 * sum_x - sum_l * sum_xl) / safeD).astype(np.float32)

            negmin = this_min > 0
            safe_l2 = np.where(sum_l2 == 0, np.float32(1.0), sum_l2).astype(np.float32)
            this_scale = np.where(negmin, (sum_xl / safe_l2).astype(np.float32), this_scale)
            this_min = np.where(negmin, np.float32(0.0), this_min).astype(np.float32)

            cdiff = (this_scale[:, None] * Laux + this_min[:, None] - x).astype(np.float32)
            cur_error = _fsum(weights * cdiff * cdiff)

            improve = Dpos & (cur_error < best_error)
            imcol = improve[:, None]
            L_arr = np.where(imcol, Laux_i, L_arr)
            best_error = np.where(improve, cur_error, best_error)
            scale_arr = np.where(improve, this_scale, scale_arr)
            min_arr = np.where(improve, this_min, min_arr)

    the_min = (-min_arr).astype(np.float32)

    if eq.any():
        scale_arr = np.where(eq, np.float32(0.0), scale_arr).astype(np.float32)
        the_min = np.where(eq, (-xmin).astype(np.float32), the_min).astype(np.float32)
        L_arr = np.where(eq[:, None], np.int32(0), L_arr)

    return scale_arr.astype(np.float32), the_min, L_arr.astype(np.int32)


# ---------------------------------------------------------------------------
# make_qx_quants  (used by Q6_K, rmse_type = 1, qw = NULL)
# ---------------------------------------------------------------------------
def _make_qx(x, nmax):
    """Vectorized port of ``make_qx_quants`` with rmse_type=1, qw=NULL.

    x : (B, 16) float32
    returns (scale (B,), L (B, 16) int32 in [0, 2*nmax-1])
    """
    nmax_i = int(nmax)
    ax = np.abs(x)
    amax = ax.max(axis=1).astype(np.float32)
    imax = ax.argmax(axis=1)
    maxv = np.take_along_axis(x, imax[:, None], axis=1)[:, 0].astype(np.float32)

    allzero = amax < GROUP_MAX_EPS
    safe_max = np.where(maxv == 0, np.float32(1.0), maxv).astype(np.float32)

    w = (x * x).astype(np.float32)  # rmse_type == 1

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        iscale = (np.float32(-nmax_i) / safe_max).astype(np.float32)

        l0 = np.clip(_nearest_int(iscale[:, None] * x), -nmax_i, nmax_i - 1)
        l0f = l0.astype(np.float32)
        sumlx = _fsum(w * x * l0f)
        suml2 = _fsum(w * l0f * l0f)
        safe_l2 = np.where(suml2 == 0, np.float32(1.0), suml2).astype(np.float32)
        scale = np.where(suml2 != 0, (sumlx / safe_l2).astype(np.float32), np.float32(0.0)).astype(np.float32)
        best = (scale * sumlx).astype(np.float32)
        L = (l0 + nmax_i).astype(np.int32)

        for is_ in range(-9, 10):
            if is_ == 0:
                continue
            t = (np.float32(0.1) * np.float32(is_)).astype(np.float32)
            t = (np.float32(nmax_i) + t).astype(np.float32)
            iscale2 = ((-t) / safe_max).astype(np.float32)

            lc = np.clip(_nearest_int(iscale2[:, None] * x), -nmax_i, nmax_i - 1)
            lcf = lc.astype(np.float32)
            sumlx2 = _fsum(w * x * lcf)
            suml22 = _fsum(w * lcf * lcf)

            cond = (suml22 > 0) & (sumlx2 * sumlx2 > best * suml22)
            safe_l22 = np.where(suml22 == 0, np.float32(1.0), suml22).astype(np.float32)
            newscale = (sumlx2 / safe_l22).astype(np.float32)

            Lupd = (lc + nmax_i).astype(np.int32)
            ccol = cond[:, None]
            L = np.where(ccol, Lupd, L)
            scale = np.where(cond, newscale, scale)
            best = np.where(cond, (newscale * sumlx2).astype(np.float32), best)

    scale = np.where(allzero, np.float32(0.0), scale).astype(np.float32)
    L = np.where(allzero[:, None], np.int32(0), L)
    return scale, L


# ---------------------------------------------------------------------------
# 6-bit scale/min packing shared by Q4_K and Q5_K
# ---------------------------------------------------------------------------
def _pack_scales_6bit(ls, lm):
    """Pack per-sub-block 6-bit scales (ls) and mins (lm) into 12 bytes.

    ls, lm : (B, 8) uint8 with values in [0, 63]
    returns (B, 12) uint8, matching get_scale_min_k4 layout.
    """
    B = ls.shape[0]
    out = np.zeros((B, 12), dtype=np.uint8)
    # j < 4
    out[:, 0:4] = ls[:, 0:4]
    out[:, 4:8] = lm[:, 0:4]
    # j = 4..7
    out[:, 8:12] = (ls[:, 4:8] & 0x0F) | ((lm[:, 4:8] & 0x0F) << 4)
    out[:, 0:4] |= ((ls[:, 4:8] >> 4) << 6).astype(np.uint8)
    out[:, 4:8] |= ((lm[:, 4:8] >> 4) << 6).astype(np.uint8)
    return out


def _quantize_scales(scales, mins):
    """Compute packed 6-bit ls/lm and the fp16 super-scales for Q4_K/Q5_K.

    scales, mins : (nb, 8) float32
    returns (scales12 (nb,12) uint8, ls (nb,8) uint8, lm (nb,8) uint8,
             d16 (nb,) float16, dmin16 (nb,) float16)
    """
    max_scale = np.maximum(np.float32(0.0), scales.max(axis=1)).astype(np.float32)
    max_min = np.maximum(np.float32(0.0), mins.max(axis=1)).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_scale = np.where(max_scale > 0, (np.float32(63.0) / max_scale), np.float32(0.0)).astype(np.float32)
        inv_min = np.where(max_min > 0, (np.float32(63.0) / max_min), np.float32(0.0)).astype(np.float32)

    ni_s = _nearest_int(inv_scale[:, None] * scales)
    ni_m = _nearest_int(inv_min[:, None] * mins)
    # C: cast int -> uint8 (mod 256), then MIN(63, .)
    ls = np.minimum(np.uint16(63), (ni_s & 0xFF).astype(np.uint16)).astype(np.uint8)
    lm = np.minimum(np.uint16(63), (ni_m & 0xFF).astype(np.uint16)).astype(np.uint8)

    scales12 = _pack_scales_6bit(ls, lm)
    d16 = (max_scale / np.float32(63.0)).astype(np.float16)
    dmin16 = (max_min / np.float32(63.0)).astype(np.float16)
    return scales12, ls, lm, d16, dmin16


# ---------------------------------------------------------------------------
# Q4_K
# ---------------------------------------------------------------------------
def _quantize_q4_k_blocks(blocks):
    nb = blocks.shape[0]
    sub = blocks.reshape(nb * 8, 32).astype(np.float32)

    sum_x2 = _fsum(sub * sub)
    av_x = np.sqrt(sum_x2 / np.float32(32.0)).astype(np.float32)
    weights = (av_x[:, None] + np.abs(sub)).astype(np.float32)

    scale, the_min, Lq = _make_qkx2(sub, weights, 15, -1.0, 0.1, 20)

    scales = scale.reshape(nb, 8)
    mins = the_min.reshape(nb, 8)
    Lq = Lq.reshape(nb, 8, 32)

    scales12, ls, lm, d16, dmin16 = _quantize_scales(scales, mins)

    d_f32 = d16.astype(np.float32)
    dmin_f32 = dmin16.astype(np.float32)
    d_j = (d_f32[:, None] * ls.astype(np.float32)).astype(np.float32)      # (nb,8)
    dm_j = (dmin_f32[:, None] * lm.astype(np.float32)).astype(np.float32)  # (nb,8)

    xsb = blocks.reshape(nb, 8, 32).astype(np.float32)
    dnz = d_j != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (xsb + dm_j[:, :, None]) / np.where(dnz[:, :, None], d_j[:, :, None], np.float32(1.0))
        lnew = np.clip(_nearest_int(val), 0, 15)
    Lfinal = np.where(dnz[:, :, None], lnew, Lq).astype(np.uint8)

    # pack qs: groups of 64 -> (nb,4,2,32)
    L4 = Lfinal.reshape(nb, 4, 2, 32)
    qs = ((L4[:, :, 0, :] & 0x0F) | ((L4[:, :, 1, :] & 0x0F) << 4)).astype(np.uint8).reshape(nb, 128)

    d_bytes = d16.view(np.uint8).reshape(nb, 2)
    dmin_bytes = dmin16.view(np.uint8).reshape(nb, 2)

    out = np.concatenate([d_bytes, dmin_bytes, scales12, qs], axis=1)
    if out.shape[1] != TYPE_SIZE["Q4_K"]:
        raise ValueError(f"Q4_K packed width is {out.shape[1]}, expected {TYPE_SIZE['Q4_K']}")
    return out


# ---------------------------------------------------------------------------
# Q5_K
# ---------------------------------------------------------------------------
def _quantize_q5_k_blocks(blocks):
    nb = blocks.shape[0]
    sub = blocks.reshape(nb * 8, 32).astype(np.float32)

    sum_x2 = _fsum(sub * sub)
    av_x = np.sqrt(sum_x2 / np.float32(32.0)).astype(np.float32)
    weights = (av_x[:, None] + np.abs(sub)).astype(np.float32)

    scale, the_min, Lq = _make_qkx2(sub, weights, 31, -0.5, 0.1, 15)

    scales = scale.reshape(nb, 8)
    mins = the_min.reshape(nb, 8)
    Lq = Lq.reshape(nb, 8, 32)

    scales12, ls, lm, d16, dmin16 = _quantize_scales(scales, mins)

    d_f32 = d16.astype(np.float32)
    dmin_f32 = dmin16.astype(np.float32)
    d_j = (d_f32[:, None] * ls.astype(np.float32)).astype(np.float32)
    dm_j = (dmin_f32[:, None] * lm.astype(np.float32)).astype(np.float32)

    xsb = blocks.reshape(nb, 8, 32).astype(np.float32)
    dnz = d_j != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (xsb + dm_j[:, :, None]) / np.where(dnz[:, :, None], d_j[:, :, None], np.float32(1.0))
        lnew = np.clip(_nearest_int(val), 0, 31)
    Lfinal = np.where(dnz[:, :, None], lnew, Lq).astype(np.int32)  # values 0..31

    L5 = Lfinal.reshape(nb, 4, 2, 32)
    ql = ((L5[:, :, 0, :] & 0x0F) | ((L5[:, :, 1, :] & 0x0F) << 4)).astype(np.uint8).reshape(nb, 128)

    # qh: 32 bytes, bit (2g) from part0>15, bit (2g+1) from part1>15
    hi0 = (L5[:, :, 0, :] > 15).astype(np.uint8)  # (nb,4,32)
    hi1 = (L5[:, :, 1, :] > 15).astype(np.uint8)
    qh = np.zeros((nb, 32), dtype=np.uint8)
    for g in range(4):
        qh |= (hi0[:, g, :] << np.uint8(2 * g))
        qh |= (hi1[:, g, :] << np.uint8(2 * g + 1))

    d_bytes = d16.view(np.uint8).reshape(nb, 2)
    dmin_bytes = dmin16.view(np.uint8).reshape(nb, 2)

    out = np.concatenate([d_bytes, dmin_bytes, scales12, qh, ql], axis=1)
    if out.shape[1] != TYPE_SIZE["Q5_K"]:
        raise ValueError(f"Q5_K packed width is {out.shape[1]}, expected {TYPE_SIZE['Q5_K']}")
    return out


# ---------------------------------------------------------------------------
# Q6_K
# ---------------------------------------------------------------------------
def _quantize_q6_k_blocks(blocks):
    nb = blocks.shape[0]
    sub = blocks.reshape(nb * 16, 16).astype(np.float32)

    scale, Lq = _make_qx(sub, 32)
    scales = scale.reshape(nb, 16)
    Lq = Lq.reshape(nb, 16, 16)

    ax = np.abs(scales)
    imax = ax.argmax(axis=1)
    max_scale = np.take_along_axis(scales, imax[:, None], axis=1)[:, 0].astype(np.float32)
    max_abs = np.abs(max_scale).astype(np.float32)
    zero_block = max_abs < GROUP_MAX_EPS

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_ms = np.where(max_scale == 0, np.float32(1.0), max_scale).astype(np.float32)
        iscale = (np.float32(-128.0) / safe_ms).astype(np.float32)
        d_super = (np.float32(1.0) / iscale).astype(np.float32)  # = -max_scale/128
    d16 = d_super.astype(np.float16)
    # zero blocks -> d = 0
    d16 = np.where(zero_block, np.float16(0.0), d16).astype(np.float16)

    ni = _nearest_int(iscale[:, None] * scales)
    scales_q = np.minimum(np.int32(127), ni).astype(np.int8)  # (nb,16)

    d_f32 = d16.astype(np.float32)
    d_j = (d_f32[:, None] * scales_q.astype(np.float32)).astype(np.float32)  # (nb,16)

    xsb = blocks.reshape(nb, 16, 16).astype(np.float32)
    dnz = d_j != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        val = xsb / np.where(dnz[:, :, None], d_j[:, :, None], np.float32(1.0))
        lnew = np.clip(_nearest_int(val), -32, 31) + 32
    Lfinal = np.where(dnz[:, :, None], lnew, Lq).astype(np.uint8)  # 0..63

    # pack ql/qh: linear L (nb,256) -> (nb,2,4,32)
    Llin = Lfinal.reshape(nb, 256)
    Lg = Llin.reshape(nb, 2, 4, 32)
    a = Lg[:, :, 0, :]
    b = Lg[:, :, 1, :]
    c = Lg[:, :, 2, :]
    dd = Lg[:, :, 3, :]
    ql_low = ((a & 0x0F) | ((c & 0x0F) << 4)).astype(np.uint8)   # (nb,2,32)
    ql_high = ((b & 0x0F) | ((dd & 0x0F) << 4)).astype(np.uint8)
    ql = np.concatenate([ql_low, ql_high], axis=-1).reshape(nb, 128)
    qh = ((a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) | ((dd >> 4) << 6)).astype(np.uint8).reshape(nb, 64)

    scales_bytes = scales_q.view(np.uint8).reshape(nb, 16)
    d_bytes = d16.view(np.uint8).reshape(nb, 2)

    # zero blocks: memset(ql, qh, scales) = 0 (d already 0)
    if zero_block.any():
        zb = zero_block[:, None]
        ql = np.where(zb, np.uint8(0), ql)
        qh = np.where(zb, np.uint8(0), qh)
        scales_bytes = np.where(zb, np.uint8(0), scales_bytes)

    out = np.concatenate([ql, qh, scales_bytes, d_bytes], axis=1)
    if out.shape[1] != TYPE_SIZE["Q6_K"]:
        raise ValueError(f"Q6_K packed width is {out.shape[1]}, expected {TYPE_SIZE['Q6_K']}")
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _prepare(arr):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if a.size % QK_K:
        raise ValueError(f"element count {a.size} is not a multiple of QK_K={QK_K}")
    return a


def validate_quant_workers(workers: int) -> int:
    """Validate the intentionally small Q4_K worker range."""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_QUANT_WORKERS:
        raise ValueError(f"quant workers must be an integer in [1, {MAX_QUANT_WORKERS}], got {workers!r}")
    return workers


def _quantize_q4_k_bounded(
    blocks: np.ndarray,
    *,
    workers: int,
    executor: Executor | None,
) -> np.ndarray:
    """Quantize bounded, input-ordered Q4_K batches without altering the kernel."""
    workers = validate_quant_workers(workers)
    if blocks.shape[0] <= Q4_K_BLOCKS_PER_TASK:
        return _quantize_q4_k_blocks(blocks)

    batches = (
        blocks[start : start + Q4_K_BLOCKS_PER_TASK]
        for start in range(0, blocks.shape[0], Q4_K_BLOCKS_PER_TASK)
    )
    if workers == 1:
        parts = [_quantize_q4_k_blocks(batch) for batch in batches]
    elif executor is not None:
        parts = list(executor.map(_quantize_q4_k_blocks, batches))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="q4-k") as local_executor:
            parts = list(local_executor.map(_quantize_q4_k_blocks, batches))
    return np.concatenate(parts, axis=0)


def quantize_q4_k(
    arr: np.ndarray,
    *,
    workers: int = 1,
    executor: Executor | None = None,
) -> np.ndarray:
    a = _prepare(arr)
    blocks = a.reshape(-1, QK_K)
    out = _quantize_q4_k_bounded(blocks, workers=workers, executor=executor)
    return out.reshape(*a.shape[:-1], -1) if a.ndim > 1 else out.reshape(-1)


def quantize_q5_k(arr: np.ndarray) -> np.ndarray:
    a = _prepare(arr)
    blocks = a.reshape(-1, QK_K)
    out = _quantize_q5_k_blocks(blocks)
    return out.reshape(*a.shape[:-1], -1) if a.ndim > 1 else out.reshape(-1)


def _quantize_q6_k_bounded(
    blocks: np.ndarray,
    *,
    workers: int,
    executor: Executor | None,
) -> np.ndarray:
    """Quantize bounded, input-ordered Q6_K batches without altering the kernel.

    Faithfully mirrors :func:`_quantize_q4_k_bounded`: each 1,024-block batch is
    independent (Q6_K super-blocks never span a batch boundary), so splitting is
    decision-preserving and the concatenated result is byte-identical to a single
    unsplit call, for any worker count. Splitting exists to bound peak memory --
    without it, a multi-million-block tensor (e.g. a 188,160-wide aggregate_embed
    row) would materialize dozens of full-width float32 intermediates at once.
    """
    workers = validate_quant_workers(workers)
    if blocks.shape[0] <= Q6_K_BLOCKS_PER_TASK:
        return _quantize_q6_k_blocks(blocks)

    batches = (
        blocks[start : start + Q6_K_BLOCKS_PER_TASK]
        for start in range(0, blocks.shape[0], Q6_K_BLOCKS_PER_TASK)
    )
    if workers == 1:
        parts = [_quantize_q6_k_blocks(batch) for batch in batches]
    elif executor is not None:
        parts = list(executor.map(_quantize_q6_k_blocks, batches))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="q6-k") as local_executor:
            parts = list(local_executor.map(_quantize_q6_k_blocks, batches))
    return np.concatenate(parts, axis=0)


def quantize_q6_k(
    arr: np.ndarray,
    *,
    workers: int = 1,
    executor: Executor | None = None,
) -> np.ndarray:
    a = _prepare(arr)
    blocks = a.reshape(-1, QK_K)
    out = _quantize_q6_k_bounded(blocks, workers=workers, executor=executor)
    return out.reshape(*a.shape[:-1], -1) if a.ndim > 1 else out.reshape(-1)


_DISPATCH = {
    "Q4_K": quantize_q4_k,
    "Q5_K": quantize_q5_k,
    "Q6_K": quantize_q6_k,
}


def quantize(
    arr: np.ndarray,
    ggml_type_name: str,
    *,
    q4_workers: int = 1,
    q4_executor: Executor | None = None,
) -> np.ndarray:
    """Dispatch quantization by ggml type name ("Q4_K" / "Q5_K" / "Q6_K").

    ``q4_workers``/``q4_executor`` bound both Q4_K and Q6_K block-batch tasks
    (the two K-quant kernels large enough to need splitting); the parameter
    names are kept for call-site compatibility (see ``convert._tensor_payload``).
    """
    try:
        fn = _DISPATCH[ggml_type_name]
    except KeyError:
        raise NotImplementedError(
            f"quantize: unsupported type {ggml_type_name!r}; "
            f"supported: {sorted(_DISPATCH)}"
        )
    if ggml_type_name == "Q4_K":
        return quantize_q4_k(arr, workers=q4_workers, executor=q4_executor)
    if ggml_type_name == "Q6_K":
        return quantize_q6_k(arr, workers=q4_workers, executor=q4_executor)
    return fn(arr)
