"""Two-pass, streaming safetensors -> GGUF conversion for LTX-2.3 (Q4_K_M).

This is the pipeline body that ties together the three helper modules:

* :mod:`converter.typemap`      -- the authoritative, *ordered* "tensor name ->
  GGML quantization type" mapping extracted from the reference GGUF.
* :mod:`converter.quant_kernels` -- NumPy K-quant (Q4_K / Q5_K / Q6_K) kernels.
* :mod:`converter.metadata`      -- safetensors ``__metadata__`` -> GGUF KV.

Design goals
------------
Memory safety is the primary constraint: the source safetensors file is ~43 GB
and must never be loaded whole.  The converter therefore streams **one tensor at
a time** and keeps RAM to a single tensor's worth (plus small constants):

* Pass 1 registers every tensor's *info* (name, shape, dtype, byte size) on the
  :class:`gguf.GGUFWriter` in typemap order -- no tensor data touched.
* The GGUF header / KV / tensor-info blocks are flushed to disk.
* Pass 2 walks the typemap again in the *same order*, reads each source tensor's
  raw bytes directly from the safetensors payload, converts/quantizes it, writes
  it with :meth:`GGUFWriter.write_tensor_data`, and immediately drops the
  reference so the garbage collector can reclaim it before the next tensor.

BF16 handling
-------------
safetensors 0.7.0's ``framework="numpy"`` cannot materialise BF16 tensors --
``safe_open(...).get_tensor(name)`` raises ``TypeError: data type 'bfloat16' not
understood`` (there is no NumPy bfloat16 and ``ml_dtypes`` is not installed).
``framework="pt"`` is unavailable (no torch).  We therefore bypass safetensors'
tensor materialisation entirely and read raw tensor bytes straight from the file
using the safetensors header's ``data_offsets``.  A BF16 value is exactly the top
16 bits of the corresponding float32, so:

* BF16 -> F32 promotion is exact: ``uint16 << 16`` viewed as float32.
* Writing a BF16 target is a verbatim copy of the 2-byte source representation
  (no value change), matching the reference conversion.

This raw-bytes approach is the fallback explicitly sanctioned by the spec and is
also the most robust: it is dtype-agnostic and never depends on NumPy growing a
bfloat16 dtype.
"""

from __future__ import annotations

import json
import struct
import sys
import tomllib
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import quant_shape_to_byte_shape
from tqdm import tqdm

from .metadata import apply_kv, validate_metadata
from .quant_kernels import quantize
from .typemap import EXPECTED_TOTAL, EXPECTED_TYPE_COUNTS, load_typemap

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
ARCH = "ltxv"
DIFFUSION_PREFIX = "model.diffusion_model."

# Source-model components that are *not* part of the diffusion GGUF and are
# therefore skipped (recorded, never written).
SKIP_PREFIXES = ("vae.", "audio_vae.", "vocoder.", "text_embedding_projection.")

# K-quant target types (require float32 -> quantize()).
_KQUANT_TYPES = ("Q4_K", "Q5_K", "Q6_K")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.toml"


# --------------------------------------------------------------------------
# minimal, memory-safe safetensors raw reader
# --------------------------------------------------------------------------
class _SafetensorsRaw:
    """Parse a safetensors header once and read tensor payloads on demand.

    Only the (small) JSON header is held in memory; tensor bytes are read
    lazily via ``seek``/``read`` so at most one tensor is resident at a time.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh: BinaryIO = open(self.path, "rb")
        header_len = struct.unpack("<Q", self._fh.read(8))[0]
        header_bytes = self._fh.read(header_len)
        self._header: dict[str, Any] = json.loads(header_bytes.decode("utf-8"))
        # payload starts right after the 8-byte length + header
        self._base = 8 + header_len
        self._metadata: dict[str, str] = dict(self._header.get("__metadata__", {}))

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "_SafetensorsRaw":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None  # type: ignore[assignment]

    # -- introspection ---------------------------------------------------
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    def keys(self) -> list[str]:
        return [k for k in self._header.keys() if k != "__metadata__"]

    def dtype_of(self, name: str) -> str:
        return self._header[name]["dtype"]

    def shape_of(self, name: str) -> list[int]:
        """Return the tensor's logical shape as recorded in the header."""
        return list(self._header[name]["shape"])

    def region_of(self, name: str) -> tuple[int, int]:
        """Return the tensor's ``(start, end)`` byte offsets *within the file*.

        The header's ``data_offsets`` are relative to the start of the payload;
        this adds the payload base so the result can be fed straight to
        ``seek``/``read`` on an independently opened handle (used by
        :mod:`converter.convert_vae` for its byte-level passthrough proof).
        """
        start, end = self._header[name]["data_offsets"]
        return self._base + start, self._base + end

    # -- payload access --------------------------------------------------
    def _read_raw(self, name: str) -> bytes:
        entry = self._header[name]
        start, end = entry["data_offsets"]
        self._fh.seek(self._base + start)
        return self._fh.read(end - start)

    def get_f32(self, name: str, logical_shape: list[int]) -> np.ndarray:
        """Return the tensor as a C-contiguous float32 array of ``logical_shape``.

        BF16 sources are promoted exactly (top-16-bits trick); F16/F32 sources
        are handled directly.
        """
        dtype = self.dtype_of(name)
        raw = self._read_raw(name)
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2")
            u32 = u16.astype(np.uint32) << np.uint32(16)
            arr = u32.view(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype="<f4")
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            raise ValueError(
                f"tensor {name!r} has unsupported source dtype {dtype!r} "
                "(expected BF16, F16 or F32)"
            )
        return np.ascontiguousarray(arr, dtype=np.float32).reshape(logical_shape)

    def get_bf16_bytes(self, name: str) -> np.ndarray:
        """Return the tensor as a uint16 array holding its BF16 bit pattern.

        A BF16 source is passed through verbatim (value-preserving).  An F32/F16
        source is rounded to BF16 (round-half-to-even) -- this branch is not
        exercised by the LTX-2.3 bf16 source model but keeps the writer correct.
        """
        dtype = self.dtype_of(name)
        raw = self._read_raw(name)
        if dtype == "BF16":
            return np.frombuffer(raw, dtype="<u2").copy()
        # F32/F16 -> BF16 round-half-to-even
        if dtype == "F32":
            f32 = np.frombuffer(raw, dtype="<f4")
        elif dtype == "F16":
            f32 = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            raise ValueError(f"tensor {name!r} has unsupported source dtype {dtype!r}")
        u32 = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
        bias = ((u32 >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
        return ((u32 + bias) >> np.uint32(16)).astype(np.uint16)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ggml_type(name: str) -> GGMLQuantizationType:
    return GGMLQuantizationType[name]


def _byte_shape_and_nbytes(shape_logical: list[int], ggml_type_name: str) -> tuple[tuple[int, ...], int]:
    """Return (byte_shape, nbytes) for a tensor of the given logical shape/type.

    Uses gguf's own ``quant_shape_to_byte_shape`` (block_size/type_size from
    ``GGML_QUANT_SIZES``) so the result is exactly what the GGUF format expects;
    this has been verified to match every ``nbytes`` in the reference typemap.
    """
    qt = _ggml_type(ggml_type_name)
    byte_shape = quant_shape_to_byte_shape(shape_logical, qt)
    nbytes = 1
    for d in byte_shape:
        nbytes *= int(d)
    return byte_shape, nbytes


def _classify_source_keys(
    source_keys: list[str],
) -> tuple[dict[str, str], list[str], list[str]]:
    """Split raw safetensors keys into diffusion / skipped / unexpected.

    Returns:
        diffusion: {stripped_name -> raw_key} for ``model.diffusion_model.*``.
        skipped:   raw keys matching a known non-diffusion component prefix.
        unexpected: raw keys matching neither (a hard error upstream).
    """
    diffusion: dict[str, str] = {}
    skipped: list[str] = []
    unexpected: list[str] = []
    for key in source_keys:
        if key.startswith(SKIP_PREFIXES):
            skipped.append(key)
        elif key.startswith(DIFFUSION_PREFIX):
            diffusion[key[len(DIFFUSION_PREFIX):]] = key
        else:
            unexpected.append(key)
    return diffusion, skipped, unexpected


def _validate_key_sets(
    diffusion: dict[str, str],
    unexpected: list[str],
    typemap_names: list[str],
) -> None:
    """Ensure the stripped diffusion key set matches the typemap set exactly."""
    errors: list[str] = []

    if unexpected:
        preview = ", ".join(sorted(unexpected)[:20])
        errors.append(
            f"{len(unexpected)} source key(s) match neither the diffusion prefix "
            f"{DIFFUSION_PREFIX!r} nor a known skip prefix {SKIP_PREFIXES}: {preview}"
        )

    tm_set = set(typemap_names)
    df_set = set(diffusion.keys())
    if len(tm_set) != len(typemap_names):
        errors.append("typemap contains duplicate tensor names")

    missing = tm_set - df_set  # in typemap, absent from source
    extra = df_set - tm_set    # in source, absent from typemap
    if missing:
        preview = ", ".join(sorted(missing)[:20])
        errors.append(f"{len(missing)} typemap tensor(s) missing from source: {preview}")
    if extra:
        preview = ", ".join(sorted(extra)[:20])
        errors.append(f"{len(extra)} source diffusion tensor(s) not in typemap: {preview}")

    if errors:
        raise ValueError(
            "safetensors <-> typemap key mismatch:\n  - " + "\n  - ".join(errors)
        )


def _register_tensor_info(writer: GGUFWriter, name: str, rec: dict[str, Any]) -> int:
    """Pass-1: register one tensor's info. Returns the registered byte size."""
    ggml_type_name = rec["ggml_type"]
    shape_logical = list(rec["shape_logical"])
    byte_shape, nbytes = _byte_shape_and_nbytes(shape_logical, ggml_type_name)

    # Cross-check against the typemap's own recorded values (belt & braces).
    assert nbytes == rec["nbytes"], (
        f"{name}: computed nbytes {nbytes} != typemap {rec['nbytes']}"
    )
    assert list(byte_shape[:-1]) == shape_logical[:-1], (
        f"{name}: byte_shape leading dims {byte_shape} vs logical {shape_logical}"
    )

    qt = _ggml_type(ggml_type_name)
    if ggml_type_name == "F32":
        writer.add_tensor_info(name, shape_logical, np.dtype(np.float32), nbytes)
    elif ggml_type_name == "BF16":
        # non-uint8 dtype -> shape is stored as-passed (logical). raw_dtype=BF16.
        writer.add_tensor_info(name, shape_logical, np.dtype(np.uint16), nbytes, raw_dtype=qt)
    elif ggml_type_name in _KQUANT_TYPES:
        # uint8 dtype -> writer recovers logical shape from the byte shape.
        writer.add_tensor_info(name, list(byte_shape), np.dtype(np.uint8), nbytes, raw_dtype=qt)
    else:
        raise ValueError(f"{name}: unsupported target GGML type {ggml_type_name!r}")
    return nbytes


def _tensor_payload(reader: _SafetensorsRaw, raw_key: str, rec: dict[str, Any]) -> np.ndarray:
    """Pass-2: produce the exact byte payload array for one tensor."""
    ggml_type_name = rec["ggml_type"]
    shape_logical = list(rec["shape_logical"])

    if ggml_type_name == "F32":
        return reader.get_f32(raw_key, shape_logical)
    if ggml_type_name == "BF16":
        return reader.get_bf16_bytes(raw_key)
    if ggml_type_name in _KQUANT_TYPES:
        f32 = reader.get_f32(raw_key, shape_logical)
        return quantize(f32, ggml_type_name)
    raise ValueError(f"{raw_key}: unsupported target GGML type {ggml_type_name!r}")


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def convert(
    st_path: str | Path,
    typemap_path: str | Path,
    out_path: str | Path,
    reference_expected: bool = True,
) -> Path:
    """Convert a safetensors LTX-2.3 diffusion model to a Q4_K_M GGUF.

    Args:
        st_path: source safetensors file.
        typemap_path: authoritative ordered typemap JSON (see converter.typemap).
        out_path: destination GGUF path.
        reference_expected: when True *and* the typemap describes the full known
            reference (``EXPECTED_TOTAL`` tensors), additionally assert that the
            per-type counts match the known reference distribution. Set False to
            convert an arbitrary (e.g. fixture or partial) typemap without that
            extra cross-check.

    Returns:
        The output GGUF path.
    """
    st_path = Path(st_path)
    typemap_path = Path(typemap_path)
    out_path = Path(out_path)

    # -- preparation ----------------------------------------------------
    records = load_typemap(typemap_path)
    typemap_names = [rec["name"] for rec in records]

    if reference_expected and len(records) == EXPECTED_TOTAL:
        counts: dict[str, int] = {}
        for rec in records:
            counts[rec["ggml_type"]] = counts.get(rec["ggml_type"], 0) + 1
        if counts != EXPECTED_TYPE_COUNTS:
            raise ValueError(
                f"typemap type counts {counts} != known reference "
                f"{EXPECTED_TYPE_COUNTS} (reference_expected=True)"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _SafetensorsRaw(st_path) as reader:
        meta = reader.metadata()
        validate_metadata(meta)

        diffusion, skipped, unexpected = _classify_source_keys(reader.keys())
        _validate_key_sets(diffusion, unexpected, typemap_names)

        if skipped:
            print(
                f"Skipping {len(skipped)} non-diffusion tensor(s) "
                f"(e.g. {', '.join(sorted(skipped)[:4])}...)"
            )

        # -- writer + KV -------------------------------------------------
        writer = GGUFWriter(str(out_path), arch=ARCH)
        apply_kv(writer, meta)

        # -- pass 1: register every tensor info, in typemap order -------
        registered_nbytes: list[int] = []
        for rec in records:
            registered_nbytes.append(_register_tensor_info(writer, rec["name"], rec))

        # -- flush header / KV / tensor-info ----------------------------
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        # -- pass 2: stream tensor data, one at a time ------------------
        for rec, exp_nbytes in tqdm(
            list(zip(records, registered_nbytes)),
            desc="Converting",
            unit="tensor",
        ):
            raw_key = diffusion[rec["name"]]
            payload = _tensor_payload(reader, raw_key, rec)
            assert payload.nbytes == exp_nbytes, (
                f"{rec['name']}: payload {payload.nbytes} bytes "
                f"!= registered {exp_nbytes}"
            )
            writer.write_tensor_data(payload)
            del payload  # drop the single resident tensor before the next one

        writer.close()

    # -- self-verification -------------------------------------------------
    _self_verify(out_path, records)
    return out_path


def _self_verify(out_path: Path, records: list[dict[str, Any]]) -> None:
    """Re-open the freshly written GGUF and sanity-check it against the typemap."""
    rd = GGUFReader(str(out_path))

    # (a) tensor count
    n = len(rd.tensors)
    if n != len(records):
        raise ValueError(
            f"self-check: output has {n} tensors, expected {len(records)}"
        )

    # (b) config KV present and JSON-parseable
    cfg = rd.fields.get("config")
    if cfg is None:
        raise ValueError("self-check: output GGUF is missing the 'config' KV")
    json.loads(cfg.contents())

    # (c) first & last few tensors: type + shape (ne order) match the typemap
    idxs = list(range(min(3, n))) + list(range(max(0, n - 3), n))
    for i in sorted(set(idxs)):
        t = rd.tensors[i]
        rec = records[i]
        if t.name != rec["name"]:
            raise ValueError(
                f"self-check: tensor #{i} name {t.name!r} != {rec['name']!r}"
            )
        if t.tensor_type.name != rec["ggml_type"]:
            raise ValueError(
                f"self-check: tensor {t.name!r} type {t.tensor_type.name} "
                f"!= {rec['ggml_type']}"
            )
        got_shape = [int(d) for d in t.shape]
        if got_shape != list(rec["shape_gguf"]):
            raise ValueError(
                f"self-check: tensor {t.name!r} shape {got_shape} "
                f"!= {rec['shape_gguf']}"
            )


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
def _load_config(config_path: str | Path = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main() -> int:
    config = _load_config()

    src = config["source"]
    st_path = _PROJECT_ROOT / src["local_dir"] / src["filename"]

    typemap_path = _PROJECT_ROOT / config["reference"]["typemap_path"]

    out = config["output"]
    out_path = _PROJECT_ROOT / out["dir"] / out["filename"]

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

    result = convert(st_path, typemap_path, out_path)
    print(f"Done: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
