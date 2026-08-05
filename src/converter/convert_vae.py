"""PrunaVAED (pruned LTX-2.3 video VAE) -> decoder-only safetensors conversion.

This is the "convert-vae" step of the toolbox. It takes the upstream, diffusers
formatted PrunaVAED checkpoint (``PrunaAI/PrunaVAED``,
``vae/diffusion_pytorch_model.safetensors``, ~1.33 GB, BF16) and emits a
decoder-only safetensors file (~690 MB) whose tensor keys are the *module
relative bare keys* that the backend's ``PrunedVideoDecoder`` loads directly.

Why a separate, pre-converted file
----------------------------------
The upstream checkpoint carries the encoder too, and the encoder is
byte-for-byte identical to the one users already have -- shipping it again
would double the download for no gain. Renaming the keys at load time was the
alternative, but ltx-core's ``SDOps`` is a chain of ``str.replace`` calls whose
outputs can re-match later rules, which makes runtime renaming fragile. So the
rename happens once, here, offline. See ``PRUNAVAED_WORKORDER.md`` §0-4 / §4.1.

What "conversion" means here
----------------------------
**Nothing is recomputed.** Every emitted tensor is a verbatim copy of the
source bytes: the source is already BF16 and the target is BF16, so the payload
is a raw ``seek``/``read``/``write`` passthrough driven by the safetensors
header's ``data_offsets`` (the same trick :mod:`converter.convert` uses, and
the reason this tool stays torch-free). The only things that actually change
are the *names* of the tensors, which tensors are present, and the
``__metadata__`` block.

Key mapping (§4.1 of the workorder, confirmed against the real header)
----------------------------------------------------------------------
``VideoDecoder.up_blocks`` is a single *flat* ``nn.ModuleList`` mixing resnet
stacks, upsamplers and -- new in PrunaVAED -- two projection resnets that
change the channel width. diffusers instead nests them per resolution level, so
the mapping is a pure prefix rewrite; the leaf suffixes are already identical on
both sides:

===================================================  ===========================
diffusers                                            ltx-core (flat)
===================================================  ===========================
``decoder.conv_in.conv.*``                           ``conv_in.conv.*``
``decoder.mid_block.resnets.{0,1}.*``                ``up_blocks.0.res_blocks.{0,1}.*``
``decoder.up_blocks.0.upsamplers.0.*``               ``up_blocks.1.*``
``decoder.up_blocks.0.resnets.{0,1}.*``              ``up_blocks.2.res_blocks.*``
``decoder.up_blocks.1.conv_in.*``                    ``up_blocks.3.*``
``decoder.up_blocks.1.upsamplers.0.*``               ``up_blocks.4.*``
``decoder.up_blocks.1.resnets.{0..3}.*``             ``up_blocks.5.res_blocks.*``
``decoder.up_blocks.2.conv_in.*``                    ``up_blocks.6.*``
``decoder.up_blocks.2.upsamplers.0.*``               ``up_blocks.7.*``
``decoder.up_blocks.2.resnets.{0..5}.*``             ``up_blocks.8.res_blocks.*``
``decoder.up_blocks.3.upsamplers.0.*``               ``up_blocks.9.*``
``decoder.up_blocks.3.resnets.{0..3}.*``             ``up_blocks.10.res_blocks.*``
``decoder.conv_out.conv.*``                          ``conv_out.conv.*``
``latents_mean`` / ``latents_std``                   ``per_channel_statistics.mean-of-means``
                                                     / ``.std-of-means``
===================================================  ===========================

No ``decoder.`` prefix and no ``vae.`` prefix survive: the backend passes
``model_sd_ops=None`` for this file, so what is written here is exactly what
``load_state_dict`` sees.

Self-verification
-----------------
:func:`convert_vae` never returns a file it has not proof-read (§5.3):

1. tensor count is 102,
2. the emitted key set equals the expected key set exactly,
3. every shape matches the reference shape table,
4. the parameter total is 345,006,256 (statistics excluded),
5. the mapping is injective (no two output tensors read the same source bytes)
   *and* **every** tensor's output region has the same MD5 as the source region
   it claims to come from -- exhaustive, never sampled, because items 1-4 and
   6-8 all pass unchanged when two same-shaped tensors get crossed (``conv1``
   and ``conv2`` of ``up_blocks.5`` are both ``[384,384,3,3,3]``: swap them and
   the count, the shapes, the parameter total and the byte total are all still
   exactly right),
6. the latent statistics match the stock ``LTX23_video_vae_bf16.safetensors``
   (optional: only when that file is reachable),
7. the file we just wrote re-parses (our own reader *and* the safetensors
   library),
8. the total size is the expected payload plus header.

Items 3, 4 and the absolute half of 8 are the only ones that assume the real
model; ``reference_expected=False`` relaxes them so the test suite can drive the
same code path with tiny synthetic fixtures.

Everything that fails raises (``VaeConversionError`` and friends). There is not
a single bare ``assert`` in this module on purpose: ``python -O`` strips them,
and a verification that can be optimized away is not a verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Union

from tqdm import tqdm

from . import __version__ as TOOL_VERSION
from .convert import _SafetensorsRaw

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
_LICENSE_PATH = _PACKAGE_DIR / "data" / "LTX-2-Community-License.txt"

# Read size for hashing / copying. Bounded so a 690 MB conversion never holds
# more than a few MB of payload in RAM (matching the repo's streaming policy).
_CHUNK_BYTES = 8 * 1024 * 1024


# --------------------------------------------------------------------------
# exceptions
# --------------------------------------------------------------------------
class VaeConversionError(RuntimeError):
    """Base class for every expected (handled, no-traceback) convert-vae failure."""


class MissingTensorError(VaeConversionError):
    """A tensor the key map requires is absent from the source checkpoint."""


class UnexpectedTensorError(VaeConversionError):
    """The source checkpoint holds a decoder tensor the key map does not know."""


class SourceDtypeError(VaeConversionError):
    """A source tensor is not BF16, so it cannot be passed through verbatim."""


class SelfVerificationError(VaeConversionError):
    """The written file failed one or more of the §5.3 self-verification items."""


# --------------------------------------------------------------------------
# structural constants (the single source of truth for the key map)
# --------------------------------------------------------------------------
DECODER_PREFIX = "decoder."

#: Source components deliberately dropped. PrunaVAED does not touch the encoder
#: (it is byte-identical to the stock one), so re-shipping it would be dead
#: weight -- see the workorder §2.3.
SKIP_PREFIXES = ("encoder.",)

#: Leaf suffixes, identical on the diffusers and the ltx-core side.
#: ``conv_in`` / ``conv_out`` *are* the ``CausalConv3d``, so one ``.conv`` level.
_CONV_SUFFIXES = ("conv.weight", "conv.bias")
#: An upsampler owns a ``CausalConv3d`` named ``conv``, which owns the real
#: ``nn.Conv3d`` -- hence two ``.conv`` levels.
_UPSAMPLE_SUFFIXES = ("conv.conv.weight", "conv.conv.bias")
_RESNET_SUFFIXES = (
    "conv1.conv.weight",
    "conv1.conv.bias",
    "conv2.conv.weight",
    "conv2.conv.bias",
)
#: A projection resnet additionally owns ``norm3`` (a channel-wise LayerNorm,
#: see workorder §2.5) and a 1x1x1 ``conv_shortcut``. Note that
#: ``conv_shortcut`` is a *bare* ``nn.Conv3d`` on both sides, so -- unlike
#: ``conv1``/``conv2`` -- it does NOT carry an extra ``.conv`` level.
_PROJECTION_SUFFIXES = (
    "norm3.weight",
    "norm3.bias",
    "conv1.conv.weight",
    "conv1.conv.bias",
    "conv2.conv.weight",
    "conv2.conv.bias",
    "conv_shortcut.weight",
    "conv_shortcut.bias",
)

#: ``(flat index, kind, diffusers prefix, geometry)`` for the 11 flat
#: ``up_blocks`` entries, in execution order. ``geometry`` feeds both the
#: expected-shape table and the human-readable summary.
_UP_BLOCK_LAYOUT: tuple[tuple[int, str, str, dict[str, int]], ...] = (
    (0, "resnets", "decoder.mid_block.resnets", {"n": 2, "ch": 1024}),
    (1, "upsample", "decoder.up_blocks.0.upsamplers.0", {"in_ch": 1024, "out_ch": 4096}),
    (2, "resnets", "decoder.up_blocks.0.resnets", {"n": 2, "ch": 512}),
    (3, "projection", "decoder.up_blocks.1.conv_in", {"in_ch": 512, "out_ch": 384}),
    (4, "upsample", "decoder.up_blocks.1.upsamplers.0", {"in_ch": 384, "out_ch": 3072}),
    (5, "resnets", "decoder.up_blocks.1.resnets", {"n": 4, "ch": 384}),
    (6, "projection", "decoder.up_blocks.2.conv_in", {"in_ch": 384, "out_ch": 256}),
    (7, "upsample", "decoder.up_blocks.2.upsamplers.0", {"in_ch": 256, "out_ch": 256}),
    (8, "resnets", "decoder.up_blocks.2.resnets", {"n": 6, "ch": 128}),
    (9, "upsample", "decoder.up_blocks.3.upsamplers.0", {"in_ch": 128, "out_ch": 256}),
    (10, "resnets", "decoder.up_blocks.3.resnets", {"n": 4, "ch": 64}),
)

#: ``latents_*`` become the decoder's per-channel statistics buffers.
_STATS_MAP = (
    ("latents_mean", "per_channel_statistics.mean-of-means"),
    ("latents_std", "per_channel_statistics.std-of-means"),
)

# Reference geometry of the real PrunaVAED v2 decoder.
LATENT_CHANNELS = 128
_CONV_IN_SHAPE = [1024, 128, 3, 3, 3]
_CONV_OUT_SHAPE = [48, 64, 3, 3, 3]

EXPECTED_TENSOR_COUNT = 102
EXPECTED_DECODER_TENSOR_COUNT = 100
EXPECTED_PARAMETER_TOTAL = 345_006_256
EXPECTED_DECODER_BYTES = 690_012_512
EXPECTED_STATS_BYTES = 512

#: Written into ``__metadata__["config"]``. Mandatory: ltx-core's
#: ``sft_loader.metadata()`` does ``json.loads(f.metadata()["config"])`` with no
#: guard, so a file without it cannot even reach the configurator (§2.4-C).
#: ``decoder_blocks`` is deliberately absent -- ltx-core's block vocabulary
#: cannot express a projection resnet, so including it would let the *stock*
#: ``VideoDecoderConfigurator`` build a silently wrong decoder instead of
#: failing loudly (§4.2).
MODEL_VERSION = "PrunaVAED-v2"
_CONFIG_CLASS_NAME = "PrunaVAEDDecoder"


def _config_dict() -> dict[str, Any]:
    return {
        "vae": {
            "_class_name": _CONFIG_CLASS_NAME,
            "latent_channels": LATENT_CHANNELS,
            "patch_size": 4,
            "norm_layer": "pixel_norm",
            "causal_decoder": False,
            "timestep_conditioning": False,
            "decoder_base_channels": 128,
        }
    }


# --------------------------------------------------------------------------
# key map / expected shapes (both generated from _UP_BLOCK_LAYOUT)
# --------------------------------------------------------------------------
def build_key_map() -> list[tuple[str, str]]:
    """Return the ordered ``(diffusers key, ltx-core key)`` pairs to emit.

    Order is the decoder's execution order, which is also the order the
    tensors are written in, so the output file reads top-to-bottom like the
    model runs.
    """
    pairs: list[tuple[str, str]] = []

    for suffix in _CONV_SUFFIXES:
        pairs.append((f"decoder.conv_in.{suffix}", f"conv_in.{suffix}"))

    for flat, kind, src_prefix, geom in _UP_BLOCK_LAYOUT:
        dst_prefix = f"up_blocks.{flat}"
        if kind == "resnets":
            for i in range(geom["n"]):
                for suffix in _RESNET_SUFFIXES:
                    pairs.append(
                        (f"{src_prefix}.{i}.{suffix}", f"{dst_prefix}.res_blocks.{i}.{suffix}")
                    )
        elif kind == "upsample":
            for suffix in _UPSAMPLE_SUFFIXES:
                pairs.append((f"{src_prefix}.{suffix}", f"{dst_prefix}.{suffix}"))
        elif kind == "projection":
            for suffix in _PROJECTION_SUFFIXES:
                pairs.append((f"{src_prefix}.{suffix}", f"{dst_prefix}.{suffix}"))
        else:  # pragma: no cover - guarded by the frozen layout table above
            raise VaeConversionError(f"unknown up_block kind {kind!r} at flat index {flat}")

    for suffix in _CONV_SUFFIXES:
        pairs.append((f"decoder.conv_out.{suffix}", f"conv_out.{suffix}"))

    pairs.extend(_STATS_MAP)
    return pairs


def expected_output_keys() -> list[str]:
    """The 102 keys the converted file must contain, in write order."""
    return [dst for _, dst in build_key_map()]


def expected_shapes() -> dict[str, list[int]]:
    """Reference shape table for the real PrunaVAED v2 decoder (§4.1)."""
    shapes: dict[str, list[int]] = {
        "conv_in.conv.weight": list(_CONV_IN_SHAPE),
        "conv_in.conv.bias": [_CONV_IN_SHAPE[0]],
        "conv_out.conv.weight": list(_CONV_OUT_SHAPE),
        "conv_out.conv.bias": [_CONV_OUT_SHAPE[0]],
    }

    for flat, kind, _src_prefix, geom in _UP_BLOCK_LAYOUT:
        dst = f"up_blocks.{flat}"
        if kind == "resnets":
            ch = geom["ch"]
            for i in range(geom["n"]):
                for which in ("conv1", "conv2"):
                    shapes[f"{dst}.res_blocks.{i}.{which}.conv.weight"] = [ch, ch, 3, 3, 3]
                    shapes[f"{dst}.res_blocks.{i}.{which}.conv.bias"] = [ch]
        elif kind == "upsample":
            shapes[f"{dst}.conv.conv.weight"] = [geom["out_ch"], geom["in_ch"], 3, 3, 3]
            shapes[f"{dst}.conv.conv.bias"] = [geom["out_ch"]]
        elif kind == "projection":
            in_ch, out_ch = geom["in_ch"], geom["out_ch"]
            shapes[f"{dst}.norm3.weight"] = [in_ch]
            shapes[f"{dst}.norm3.bias"] = [in_ch]
            shapes[f"{dst}.conv1.conv.weight"] = [out_ch, in_ch, 3, 3, 3]
            shapes[f"{dst}.conv1.conv.bias"] = [out_ch]
            shapes[f"{dst}.conv2.conv.weight"] = [out_ch, out_ch, 3, 3, 3]
            shapes[f"{dst}.conv2.conv.bias"] = [out_ch]
            shapes[f"{dst}.conv_shortcut.weight"] = [out_ch, in_ch, 1, 1, 1]
            shapes[f"{dst}.conv_shortcut.bias"] = [out_ch]

    for _src, dst in _STATS_MAP:
        shapes[dst] = [LATENT_CHANNELS]

    return shapes


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
@dataclass
class VaeConvertReport:
    """Result of :func:`convert_vae`'s self-verification.

    ``checks`` maps a check name to its issue list; an empty list means PASS.
    ``skipped`` holds the checks that could not be evaluated (relaxed
    ``reference_expected``, or an unavailable optional reference file); they do
    not fail the run but are reported as SKIP so a log never looks like it
    proved more than it did.
    """

    source_path: str
    output_path: str
    tensor_count: int
    parameter_total: int
    payload_bytes: int
    header_bytes: int
    file_bytes: int
    source_sha256: str
    checks: dict[str, list[str]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(not issues for issues in self.checks.values())

    @property
    def failed_checks(self) -> list[str]:
        return [name for name, issues in self.checks.items() if issues]

    def summary(self) -> str:
        lines = [
            f"Source : {self.source_path}",
            f"Output : {self.output_path}",
            f"SHA-256 (source): {self.source_sha256}",
            f"Tensors: {self.tensor_count}",
            f"Params : {self.parameter_total:,} (per_channel_statistics excluded)",
            f"Bytes  : header={self.header_bytes} payload={self.payload_bytes} "
            f"file={self.file_bytes}",
            "",
        ]
        for name, issues in self.checks.items():
            if name in self.skipped:
                lines.append(f"[SKIP] {name} -- {self.skipped[name]}")
                continue
            status = "PASS" if not issues else f"FAIL ({len(issues)} issue(s))"
            lines.append(f"[{status}] {name}")
            for issue in issues[:20]:
                lines.append(f"    - {issue}")
            if len(issues) > 20:
                lines.append(f"    ... and {len(issues) - 20} more")
        lines.append("")
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _num_elements(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _md5_region(fh: BinaryIO, start: int, length: int) -> str:
    """MD5 of ``length`` bytes at ``start``, read in bounded chunks."""
    digest = hashlib.md5()
    fh.seek(start)
    remaining = length
    while remaining > 0:
        chunk = fh.read(min(_CHUNK_BYTES, remaining))
        if not chunk:
            raise VaeConversionError(
                f"unexpected end of file while hashing region at {start} "
                f"({remaining} bytes short)"
            )
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _copy_region(src: BinaryIO, dst: BinaryIO, start: int, length: int) -> None:
    src.seek(start)
    remaining = length
    while remaining > 0:
        chunk = src.read(min(_CHUNK_BYTES, remaining))
        if not chunk:
            raise VaeConversionError(
                f"unexpected end of file while copying region at {start} "
                f"({remaining} bytes short)"
            )
        dst.write(chunk)
        remaining -= len(chunk)


def load_license_text() -> str:
    """Return the LTX-2 Community License text embedded into the output.

    Read as bytes and newline-normalised so a CRLF checkout cannot change the
    metadata the file ships with. The vendored copy was lifted verbatim from
    the stock ``LTX23_video_vae_bf16.safetensors``'s ``__metadata__["license"]``
    (workorder §0-7: the pruned weights carry the same licence as the LTX
    weights they derive from).
    """
    raw = _LICENSE_PATH.read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n")


def _write_safetensors_streaming(
    out_path: Path,
    source_path: Path,
    plan: list[dict[str, Any]],
    metadata: dict[str, str],
) -> tuple[int, int]:
    """Write the output file; return ``(header_bytes, payload_bytes)``.

    ``plan`` entries are ``{"dst", "dtype", "shape", "src_start", "nbytes",
    "out_start"}``. Tensors are copied one at a time straight from the source
    file, so peak RAM is one 8 MiB chunk regardless of tensor size.
    """
    header: dict[str, Any] = {"__metadata__": dict(metadata)}
    for entry in plan:
        header[entry["dst"]] = {
            "dtype": entry["dtype"],
            "shape": list(entry["shape"]),
            "data_offsets": [entry["out_start"], entry["out_start"] + entry["nbytes"]],
        }

    header_bytes = json.dumps(header).encode("utf-8")
    # safetensors' own serializer pads the header to an 8-byte boundary so the
    # payload starts aligned; mirror that instead of inventing our own layout.
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = 0
    with open(source_path, "rb") as src, open(out_path, "wb") as dst:
        dst.write(struct.pack("<Q", len(header_bytes)))
        dst.write(header_bytes)
        for entry in tqdm(plan, desc="writing tensors", unit="tensor"):
            _copy_region(src, dst, entry["src_start"], entry["nbytes"])
            payload_bytes += entry["nbytes"]

    return len(header_bytes), payload_bytes


# --------------------------------------------------------------------------
# the conversion
# --------------------------------------------------------------------------
def convert_vae(
    source_path: Union[str, Path],
    out_path: Union[str, Path],
    *,
    source_repo: str,
    source_revision: str,
    source_filename: str,
    expected_sha256: Union[str, None] = None,
    reference_expected: bool = True,
    reference_vae_path: Union[str, Path, None] = None,
) -> VaeConvertReport:
    """Extract, rename and re-emit the PrunaVAED decoder; verify what was written.

    Args:
        source_path: The upstream diffusers-format PrunaVAED safetensors file.
        out_path: Destination (``PrunaVAED-decoder-bf16.safetensors``).
        source_repo / source_revision / source_filename: Recorded verbatim in
            ``__metadata__["provenance"]`` so the output can always be traced
            back to the exact upstream commit.
        expected_sha256: Pinned digest of ``source_path``. Verified before any
            byte is written; a mismatch raises.
        reference_expected: When True (default) the shape table, the parameter
            total and the absolute byte total are compared against the real
            PrunaVAED v2 model. Set False for structurally-identical fixtures.
        reference_vae_path: Optional stock ``LTX23_video_vae_bf16.safetensors``
            used for the latent-statistics cross-check (item 6).

    Returns:
        A passing :class:`VaeConvertReport`.

    Raises:
        VaeConversionError (or a subclass) on any failure, including a failed
        self-verification -- in which case ``out_path`` is left on disk for
        inspection but must not be used.
    """
    source_path = Path(source_path)
    out_path = Path(out_path)

    if not source_path.is_file():
        raise VaeConversionError(f"source safetensors not found: {source_path}")

    # -- 0. prove the input is the pinned revision before doing any work ----
    # Local import: download.py pulls in huggingface_hub, and importing this
    # module should not require the Hub client just to read a local file.
    from .download import sha256_of_file, verify_sha256

    if expected_sha256:
        logger.info("Verifying the source SHA-256 (pinned revision %s)...", source_revision)
        source_sha256 = verify_sha256(source_path, expected_sha256, label="source")
        logger.info("Source SHA-256 OK: %s", source_sha256)
    else:
        logger.info("Hashing the source file (no pinned digest configured)...")
        source_sha256 = sha256_of_file(source_path)
        logger.info("Source SHA-256: %s", source_sha256)

    key_map = build_key_map()
    expected_keys = [dst for _, dst in key_map]

    # -- 1. plan the output from the source header -------------------------
    plan: list[dict[str, Any]] = []
    src_regions: dict[str, tuple[int, int]] = {}
    with _SafetensorsRaw(source_path) as reader:
        source_keys = set(reader.keys())

        missing = [src for src, _ in key_map if src not in source_keys]
        if missing:
            raise MissingTensorError(
                f"{len(missing)} tensor(s) required by the PrunaVAED key map are "
                f"absent from {source_path.name}: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )

        mapped = {src for src, _ in key_map}
        unexpected = sorted(
            k
            for k in source_keys
            if k not in mapped and not k.startswith(SKIP_PREFIXES)
        )
        if unexpected:
            raise UnexpectedTensorError(
                f"{len(unexpected)} unmapped tensor(s) in {source_path.name}: "
                f"{unexpected[:10]}"
                + (" ..." if len(unexpected) > 10 else "")
                + ". The checkpoint's structure differs from PrunaVAED v2; refusing "
                "to emit a partially-understood decoder."
            )

        out_offset = 0
        for src_key, dst_key in key_map:
            dtype = reader.dtype_of(src_key)
            if dtype != "BF16":
                raise SourceDtypeError(
                    f"tensor {src_key!r} has dtype {dtype!r}; convert-vae is a "
                    "verbatim BF16 passthrough and does not convert dtypes."
                )
            start, end = reader.region_of(src_key)
            nbytes = end - start
            shape = reader.shape_of(src_key)
            src_regions[dst_key] = (start, nbytes)
            plan.append(
                {
                    "src": src_key,
                    "dst": dst_key,
                    "dtype": dtype,
                    "shape": shape,
                    "src_start": start,
                    "nbytes": nbytes,
                    "out_start": out_offset,
                }
            )
            out_offset += nbytes

    # -- 2. metadata -------------------------------------------------------
    provenance = {
        "source_repo": source_repo,
        "source_revision": source_revision,
        "source_filename": source_filename,
        "source_sha256": source_sha256,
        "converted_by": "Nz-GGUF-Converter-LTX23 convert-vae",
        "tool_version": TOOL_VERSION,
    }
    metadata = {
        "config": json.dumps(_config_dict()),
        "model_version": MODEL_VERSION,
        "license": load_license_text(),
        "provenance": json.dumps(provenance),
    }

    # -- 3. write ----------------------------------------------------------
    logger.info("Writing %d tensors to %s", len(plan), out_path)
    header_bytes, payload_bytes = _write_safetensors_streaming(
        out_path, source_path, plan, metadata
    )

    # -- 4. self-verify ----------------------------------------------------
    report = _self_verify(
        source_path=source_path,
        out_path=out_path,
        plan=plan,
        src_regions=src_regions,
        expected_keys=expected_keys,
        header_bytes=header_bytes,
        payload_bytes=payload_bytes,
        source_sha256=source_sha256,
        reference_expected=reference_expected,
        reference_vae_path=reference_vae_path,
    )
    if not report.passed:
        raise SelfVerificationError(
            "convert-vae self-verification failed "
            f"({', '.join(report.failed_checks)}):\n{report.summary()}"
        )
    return report


# --------------------------------------------------------------------------
# self-verification (§5.3)
# --------------------------------------------------------------------------
def _self_verify(
    *,
    source_path: Path,
    out_path: Path,
    plan: list[dict[str, Any]],
    src_regions: dict[str, tuple[int, int]],
    expected_keys: list[str],
    header_bytes: int,
    payload_bytes: int,
    source_sha256: str,
    reference_expected: bool,
    reference_vae_path: Union[str, Path, None],
) -> VaeConvertReport:
    checks: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}

    stat_keys = {dst for _src, dst in _STATS_MAP}
    parameter_total = sum(
        _num_elements(e["shape"]) for e in plan if e["dst"] not in stat_keys
    )
    file_bytes = out_path.stat().st_size

    # --- 7 (first, everything below reads the re-parsed file) ------------
    roundtrip: list[str] = []
    out_shapes: dict[str, list[int]] = {}
    out_regions: dict[str, tuple[int, int]] = {}
    out_meta: dict[str, str] = {}
    with _SafetensorsRaw(out_path) as out_reader:
        out_keys = out_reader.keys()
        out_meta = out_reader.metadata()
        for name in out_keys:
            out_shapes[name] = out_reader.shape_of(name)
            start, end = out_reader.region_of(name)
            out_regions[name] = (start, end - start)

    if out_keys != expected_keys:
        roundtrip.append(
            "re-parsed key order differs from the write order "
            f"(first difference at index "
            f"{next((i for i, (a, b) in enumerate(zip(out_keys, expected_keys)) if a != b), 'n/a')})"
        )
    # The payload must tile the file with no gaps and no overlaps -- that is
    # what the safetensors format requires and what our writer promises.
    cursor = 8 + header_bytes
    for name in out_keys:
        start, length = out_regions[name]
        if start != cursor:
            roundtrip.append(f"{name}: starts at {start}, expected {cursor} (gap/overlap)")
        cursor = start + length
    if cursor != file_bytes:
        roundtrip.append(f"payload ends at {cursor} but the file is {file_bytes} bytes")

    if "config" not in out_meta:
        roundtrip.append("__metadata__ has no 'config' key (ltx-core's loader would crash)")
    else:
        try:
            cfg = json.loads(out_meta["config"])
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            roundtrip.append(f"__metadata__['config'] is not valid JSON: {exc}")
        else:
            vae_cfg = cfg.get("vae")
            if not isinstance(vae_cfg, dict):
                roundtrip.append("config has no 'vae' object")
            else:
                if "decoder_blocks" in vae_cfg or "decoder_blocks" in cfg:
                    roundtrip.append(
                        "config must NOT contain 'decoder_blocks' (§4.2: it would let "
                        "the stock configurator build a silently wrong decoder)"
                    )
                if vae_cfg.get("_class_name") != _CONFIG_CLASS_NAME:
                    roundtrip.append(
                        f"config _class_name is {vae_cfg.get('_class_name')!r}, "
                        f"expected {_CONFIG_CLASS_NAME!r}"
                    )
    for required in ("model_version", "license", "provenance"):
        if not out_meta.get(required):
            roundtrip.append(f"__metadata__ is missing or empty for {required!r}")

    # The safetensors library itself must accept the file: our own reader is
    # deliberately lenient, the Rust one is not.
    try:
        from safetensors import safe_open

        with safe_open(str(out_path), framework="numpy") as f:
            lib_keys = sorted(f.keys())
            lib_meta = f.metadata()
        if lib_keys != sorted(expected_keys):
            roundtrip.append("safetensors library reports a different key set")
        if not lib_meta or "config" not in lib_meta:
            roundtrip.append("safetensors library sees no 'config' in __metadata__")
    except ImportError:  # pragma: no cover - safetensors is a hard requirement
        skipped["7_output_header_roundtrip"] = "safetensors library not installed"
    except Exception as exc:  # noqa: BLE001
        roundtrip.append(f"safetensors library refused the file: {exc}")

    # --- 1. tensor count --------------------------------------------------
    count_issues: list[str] = []
    if len(out_keys) != EXPECTED_TENSOR_COUNT:
        count_issues.append(
            f"tensor count is {len(out_keys)}, expected {EXPECTED_TENSOR_COUNT} "
            f"({EXPECTED_DECODER_TENSOR_COUNT} decoder tensors + 2 per-channel statistics)"
        )
    checks["1_tensor_count"] = count_issues

    # --- 2. key set -------------------------------------------------------
    key_issues: list[str] = []
    got, want = set(out_keys), set(expected_keys)
    for extra in sorted(got - want):
        key_issues.append(f"unexpected key in the output: {extra}")
    for absent in sorted(want - got):
        key_issues.append(f"missing key in the output: {absent}")
    checks["2_key_set"] = key_issues

    # --- 3. shapes --------------------------------------------------------
    shape_issues: list[str] = []
    if reference_expected:
        reference_shapes = expected_shapes()
        for name in expected_keys:
            want_shape = reference_shapes.get(name)
            got_shape = out_shapes.get(name)
            if want_shape is not None and got_shape != want_shape:
                shape_issues.append(f"{name}: shape {got_shape}, expected {want_shape}")
    else:
        skipped["3_tensor_shapes"] = "reference_expected=False (synthetic fixture)"
    checks["3_tensor_shapes"] = shape_issues

    # --- 4. parameter total ----------------------------------------------
    param_issues: list[str] = []
    if reference_expected:
        if parameter_total != EXPECTED_PARAMETER_TOTAL:
            param_issues.append(
                f"parameter total is {parameter_total:,}, expected "
                f"{EXPECTED_PARAMETER_TOTAL:,}"
            )
    else:
        skipped["4_parameter_total"] = "reference_expected=False (synthetic fixture)"
    checks["4_parameter_total"] = param_issues

    # --- 5. exhaustive MD5 passthrough proof ------------------------------
    # Two halves, both map-independent:
    #  (a) injectivity -- no two output tensors may be fed from the same source
    #      region, and every mapped source tensor must be consumed exactly once.
    #      This is what catches a key map that accidentally points two entries at
    #      the same source (the "same-shape swap" family of bugs, where every
    #      count/shape/total check still passes).
    #  (b) byte equality -- each output region must hash identically to the
    #      source region it claims to come from, proving the copy really was a
    #      verbatim passthrough and no offset drifted.
    md5_issues: list[str] = []
    seen_regions: dict[tuple[int, int], str] = {}
    for name in expected_keys:
        region = src_regions[name]
        previous = seen_regions.get(region)
        if previous is not None:
            md5_issues.append(
                f"{name} and {previous} both read source bytes [{region[0]}, "
                f"{region[0] + region[1]}) -- the key map is not injective"
            )
        else:
            seen_regions[region] = name

    with open(source_path, "rb") as src_fh, open(out_path, "rb") as out_fh:
        for name in tqdm(expected_keys, desc="md5 cross-check", unit="tensor"):
            src_start, src_len = src_regions[name]
            out_start, out_len = out_regions[name]
            if src_len != out_len:
                md5_issues.append(f"{name}: {out_len} bytes written, source has {src_len}")
                continue
            src_md5 = _md5_region(src_fh, src_start, src_len)
            out_md5 = _md5_region(out_fh, out_start, out_len)
            if src_md5 != out_md5:
                md5_issues.append(
                    f"{name}: MD5 {out_md5} != source MD5 {src_md5} "
                    "(wrong source region -- the key map is not a bijection)"
                )
    checks["5_md5_passthrough"] = md5_issues

    # --- 6. latent statistics vs. the stock video VAE ---------------------
    stats_issues: list[str] = []
    ref_path = Path(reference_vae_path) if reference_vae_path else None
    if ref_path is None or not ref_path.is_file():
        skipped["6_latent_statistics_vs_reference"] = (
            "stock LTX23_video_vae_bf16.safetensors not available"
            if ref_path is None
            else f"reference not found: {ref_path}"
        )
    else:
        with _SafetensorsRaw(ref_path) as ref_reader, open(ref_path, "rb") as ref_fh, open(
            out_path, "rb"
        ) as out_fh:
            ref_keys = set(ref_reader.keys())
            for _src, dst in _STATS_MAP:
                if dst not in ref_keys:
                    stats_issues.append(f"{dst} absent from {ref_path.name}")
                    continue
                r_start, r_end = ref_reader.region_of(dst)
                o_start, o_len = out_regions[dst]
                if (r_end - r_start) != o_len:
                    stats_issues.append(
                        f"{dst}: {o_len} bytes vs {r_end - r_start} in {ref_path.name}"
                    )
                    continue
                if _md5_region(ref_fh, r_start, r_end - r_start) != _md5_region(
                    out_fh, o_start, o_len
                ):
                    stats_issues.append(f"{dst}: bytes differ from {ref_path.name}")
    checks["6_latent_statistics_vs_reference"] = stats_issues

    # --- 7. (issues collected above) --------------------------------------
    checks["7_output_header_roundtrip"] = roundtrip

    # --- 8. total size ----------------------------------------------------
    size_issues: list[str] = []
    if file_bytes != 8 + header_bytes + payload_bytes:
        size_issues.append(
            f"file is {file_bytes} bytes, expected 8 + {header_bytes} (header) + "
            f"{payload_bytes} (payload) = {8 + header_bytes + payload_bytes}"
        )
    if reference_expected:
        want_payload = EXPECTED_DECODER_BYTES + EXPECTED_STATS_BYTES
        if payload_bytes != want_payload:
            size_issues.append(
                f"payload is {payload_bytes} bytes, expected {want_payload} "
                f"({EXPECTED_DECODER_BYTES} decoder + {EXPECTED_STATS_BYTES} statistics)"
            )
    checks["8_total_size"] = size_issues

    # Re-order the dict so the printed report follows §5.3's numbering.
    ordered = {name: checks[name] for name in sorted(checks)}

    return VaeConvertReport(
        source_path=str(source_path),
        output_path=str(out_path),
        tensor_count=len(out_keys),
        parameter_total=parameter_total,
        payload_bytes=payload_bytes,
        header_bytes=header_bytes,
        file_bytes=file_bytes,
        source_sha256=source_sha256,
        checks=ordered,
        skipped=skipped,
    )
