"""Tests for converter.convert_vae: PrunaVAED -> decoder-only safetensors.

The real upstream checkpoint is ~1.33 GB, so every test here builds a tiny
synthetic stand-in that keeps the *structure* the converter cares about while
shrinking every channel count by 64x:

* the projection resnets (``norm3`` + ``conv_shortcut``, the two blocks the
  stock LTX-2.3 decoder does not have),
* the four upsamplers,
* the four resnet stacks (2 / 2 / 4 / 6 / 4 blocks),
* the ``latents_mean`` / ``latents_std`` statistics,
* an encoder that must be dropped.

Because the key names do not depend on any channel width, the tiny fixture
produces *exactly* the same 102 output keys as the real model, so the key-set
and MD5 checks are exercised for real. Only the three reference-value checks
(shape table, parameter total, absolute byte total) are relaxed via
``reference_expected=False``.

Fixtures are written by hand, as in ``test_convert.py``: safetensors' NumPy
framework cannot represent BF16, so we assemble the header + payload ourselves
to store genuine ``dtype="BF16"`` tensors.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

from converter import convert_vae as cv
from converter.convert_vae import (
    MissingTensorError,
    SelfVerificationError,
    SourceDtypeError,
    UnexpectedTensorError,
    build_key_map,
    convert_vae,
    expected_output_keys,
)

# --------------------------------------------------------------------------
# the golden key map -- an independent transcription of the workorder's §4.1
# table. Hand-written on purpose: comparing build_key_map() against a table
# generated the same way it is would prove nothing.
# --------------------------------------------------------------------------
GOLDEN_KEY_MAP: list[tuple[str, str]] = [
    ("decoder.conv_in.conv.weight", "conv_in.conv.weight"),
    ("decoder.conv_in.conv.bias", "conv_in.conv.bias"),
    # flat 0 -- res_x x2 @1024 (diffusers calls it the mid_block)
    ("decoder.mid_block.resnets.0.conv1.conv.weight", "up_blocks.0.res_blocks.0.conv1.conv.weight"),
    ("decoder.mid_block.resnets.0.conv1.conv.bias", "up_blocks.0.res_blocks.0.conv1.conv.bias"),
    ("decoder.mid_block.resnets.0.conv2.conv.weight", "up_blocks.0.res_blocks.0.conv2.conv.weight"),
    ("decoder.mid_block.resnets.0.conv2.conv.bias", "up_blocks.0.res_blocks.0.conv2.conv.bias"),
    ("decoder.mid_block.resnets.1.conv1.conv.weight", "up_blocks.0.res_blocks.1.conv1.conv.weight"),
    ("decoder.mid_block.resnets.1.conv1.conv.bias", "up_blocks.0.res_blocks.1.conv1.conv.bias"),
    ("decoder.mid_block.resnets.1.conv2.conv.weight", "up_blocks.0.res_blocks.1.conv2.conv.weight"),
    ("decoder.mid_block.resnets.1.conv2.conv.bias", "up_blocks.0.res_blocks.1.conv2.conv.bias"),
    # flat 1 -- compress_all m=2, 1024 -> 512
    ("decoder.up_blocks.0.upsamplers.0.conv.conv.weight", "up_blocks.1.conv.conv.weight"),
    ("decoder.up_blocks.0.upsamplers.0.conv.conv.bias", "up_blocks.1.conv.conv.bias"),
    # flat 2 -- res_x x2 @512
    ("decoder.up_blocks.0.resnets.0.conv1.conv.weight", "up_blocks.2.res_blocks.0.conv1.conv.weight"),
    ("decoder.up_blocks.0.resnets.0.conv1.conv.bias", "up_blocks.2.res_blocks.0.conv1.conv.bias"),
    ("decoder.up_blocks.0.resnets.0.conv2.conv.weight", "up_blocks.2.res_blocks.0.conv2.conv.weight"),
    ("decoder.up_blocks.0.resnets.0.conv2.conv.bias", "up_blocks.2.res_blocks.0.conv2.conv.bias"),
    ("decoder.up_blocks.0.resnets.1.conv1.conv.weight", "up_blocks.2.res_blocks.1.conv1.conv.weight"),
    ("decoder.up_blocks.0.resnets.1.conv1.conv.bias", "up_blocks.2.res_blocks.1.conv1.conv.bias"),
    ("decoder.up_blocks.0.resnets.1.conv2.conv.weight", "up_blocks.2.res_blocks.1.conv2.conv.weight"),
    ("decoder.up_blocks.0.resnets.1.conv2.conv.bias", "up_blocks.2.res_blocks.1.conv2.conv.bias"),
    # flat 3 -- projection resnet 512 -> 384 (new in PrunaVAED)
    ("decoder.up_blocks.1.conv_in.norm3.weight", "up_blocks.3.norm3.weight"),
    ("decoder.up_blocks.1.conv_in.norm3.bias", "up_blocks.3.norm3.bias"),
    ("decoder.up_blocks.1.conv_in.conv1.conv.weight", "up_blocks.3.conv1.conv.weight"),
    ("decoder.up_blocks.1.conv_in.conv1.conv.bias", "up_blocks.3.conv1.conv.bias"),
    ("decoder.up_blocks.1.conv_in.conv2.conv.weight", "up_blocks.3.conv2.conv.weight"),
    ("decoder.up_blocks.1.conv_in.conv2.conv.bias", "up_blocks.3.conv2.conv.bias"),
    ("decoder.up_blocks.1.conv_in.conv_shortcut.weight", "up_blocks.3.conv_shortcut.weight"),
    ("decoder.up_blocks.1.conv_in.conv_shortcut.bias", "up_blocks.3.conv_shortcut.bias"),
    # flat 4 -- compress_all m=1, 384 -> 384
    ("decoder.up_blocks.1.upsamplers.0.conv.conv.weight", "up_blocks.4.conv.conv.weight"),
    ("decoder.up_blocks.1.upsamplers.0.conv.conv.bias", "up_blocks.4.conv.conv.bias"),
    # flat 5 -- res_x x4 @384
    ("decoder.up_blocks.1.resnets.0.conv1.conv.weight", "up_blocks.5.res_blocks.0.conv1.conv.weight"),
    ("decoder.up_blocks.1.resnets.0.conv1.conv.bias", "up_blocks.5.res_blocks.0.conv1.conv.bias"),
    ("decoder.up_blocks.1.resnets.0.conv2.conv.weight", "up_blocks.5.res_blocks.0.conv2.conv.weight"),
    ("decoder.up_blocks.1.resnets.0.conv2.conv.bias", "up_blocks.5.res_blocks.0.conv2.conv.bias"),
    ("decoder.up_blocks.1.resnets.1.conv1.conv.weight", "up_blocks.5.res_blocks.1.conv1.conv.weight"),
    ("decoder.up_blocks.1.resnets.1.conv1.conv.bias", "up_blocks.5.res_blocks.1.conv1.conv.bias"),
    ("decoder.up_blocks.1.resnets.1.conv2.conv.weight", "up_blocks.5.res_blocks.1.conv2.conv.weight"),
    ("decoder.up_blocks.1.resnets.1.conv2.conv.bias", "up_blocks.5.res_blocks.1.conv2.conv.bias"),
    ("decoder.up_blocks.1.resnets.2.conv1.conv.weight", "up_blocks.5.res_blocks.2.conv1.conv.weight"),
    ("decoder.up_blocks.1.resnets.2.conv1.conv.bias", "up_blocks.5.res_blocks.2.conv1.conv.bias"),
    ("decoder.up_blocks.1.resnets.2.conv2.conv.weight", "up_blocks.5.res_blocks.2.conv2.conv.weight"),
    ("decoder.up_blocks.1.resnets.2.conv2.conv.bias", "up_blocks.5.res_blocks.2.conv2.conv.bias"),
    ("decoder.up_blocks.1.resnets.3.conv1.conv.weight", "up_blocks.5.res_blocks.3.conv1.conv.weight"),
    ("decoder.up_blocks.1.resnets.3.conv1.conv.bias", "up_blocks.5.res_blocks.3.conv1.conv.bias"),
    ("decoder.up_blocks.1.resnets.3.conv2.conv.weight", "up_blocks.5.res_blocks.3.conv2.conv.weight"),
    ("decoder.up_blocks.1.resnets.3.conv2.conv.bias", "up_blocks.5.res_blocks.3.conv2.conv.bias"),
    # flat 6 -- projection resnet 384 -> 256 (new in PrunaVAED)
    ("decoder.up_blocks.2.conv_in.norm3.weight", "up_blocks.6.norm3.weight"),
    ("decoder.up_blocks.2.conv_in.norm3.bias", "up_blocks.6.norm3.bias"),
    ("decoder.up_blocks.2.conv_in.conv1.conv.weight", "up_blocks.6.conv1.conv.weight"),
    ("decoder.up_blocks.2.conv_in.conv1.conv.bias", "up_blocks.6.conv1.conv.bias"),
    ("decoder.up_blocks.2.conv_in.conv2.conv.weight", "up_blocks.6.conv2.conv.weight"),
    ("decoder.up_blocks.2.conv_in.conv2.conv.bias", "up_blocks.6.conv2.conv.bias"),
    ("decoder.up_blocks.2.conv_in.conv_shortcut.weight", "up_blocks.6.conv_shortcut.weight"),
    ("decoder.up_blocks.2.conv_in.conv_shortcut.bias", "up_blocks.6.conv_shortcut.bias"),
    # flat 7 -- compress_time m=2, 256 -> 128
    ("decoder.up_blocks.2.upsamplers.0.conv.conv.weight", "up_blocks.7.conv.conv.weight"),
    ("decoder.up_blocks.2.upsamplers.0.conv.conv.bias", "up_blocks.7.conv.conv.bias"),
    # flat 8 -- res_x x6 @128
    ("decoder.up_blocks.2.resnets.0.conv1.conv.weight", "up_blocks.8.res_blocks.0.conv1.conv.weight"),
    ("decoder.up_blocks.2.resnets.0.conv1.conv.bias", "up_blocks.8.res_blocks.0.conv1.conv.bias"),
    ("decoder.up_blocks.2.resnets.0.conv2.conv.weight", "up_blocks.8.res_blocks.0.conv2.conv.weight"),
    ("decoder.up_blocks.2.resnets.0.conv2.conv.bias", "up_blocks.8.res_blocks.0.conv2.conv.bias"),
    ("decoder.up_blocks.2.resnets.1.conv1.conv.weight", "up_blocks.8.res_blocks.1.conv1.conv.weight"),
    ("decoder.up_blocks.2.resnets.1.conv1.conv.bias", "up_blocks.8.res_blocks.1.conv1.conv.bias"),
    ("decoder.up_blocks.2.resnets.1.conv2.conv.weight", "up_blocks.8.res_blocks.1.conv2.conv.weight"),
    ("decoder.up_blocks.2.resnets.1.conv2.conv.bias", "up_blocks.8.res_blocks.1.conv2.conv.bias"),
    ("decoder.up_blocks.2.resnets.2.conv1.conv.weight", "up_blocks.8.res_blocks.2.conv1.conv.weight"),
    ("decoder.up_blocks.2.resnets.2.conv1.conv.bias", "up_blocks.8.res_blocks.2.conv1.conv.bias"),
    ("decoder.up_blocks.2.resnets.2.conv2.conv.weight", "up_blocks.8.res_blocks.2.conv2.conv.weight"),
    ("decoder.up_blocks.2.resnets.2.conv2.conv.bias", "up_blocks.8.res_blocks.2.conv2.conv.bias"),
    ("decoder.up_blocks.2.resnets.3.conv1.conv.weight", "up_blocks.8.res_blocks.3.conv1.conv.weight"),
    ("decoder.up_blocks.2.resnets.3.conv1.conv.bias", "up_blocks.8.res_blocks.3.conv1.conv.bias"),
    ("decoder.up_blocks.2.resnets.3.conv2.conv.weight", "up_blocks.8.res_blocks.3.conv2.conv.weight"),
    ("decoder.up_blocks.2.resnets.3.conv2.conv.bias", "up_blocks.8.res_blocks.3.conv2.conv.bias"),
    ("decoder.up_blocks.2.resnets.4.conv1.conv.weight", "up_blocks.8.res_blocks.4.conv1.conv.weight"),
    ("decoder.up_blocks.2.resnets.4.conv1.conv.bias", "up_blocks.8.res_blocks.4.conv1.conv.bias"),
    ("decoder.up_blocks.2.resnets.4.conv2.conv.weight", "up_blocks.8.res_blocks.4.conv2.conv.weight"),
    ("decoder.up_blocks.2.resnets.4.conv2.conv.bias", "up_blocks.8.res_blocks.4.conv2.conv.bias"),
    ("decoder.up_blocks.2.resnets.5.conv1.conv.weight", "up_blocks.8.res_blocks.5.conv1.conv.weight"),
    ("decoder.up_blocks.2.resnets.5.conv1.conv.bias", "up_blocks.8.res_blocks.5.conv1.conv.bias"),
    ("decoder.up_blocks.2.resnets.5.conv2.conv.weight", "up_blocks.8.res_blocks.5.conv2.conv.weight"),
    ("decoder.up_blocks.2.resnets.5.conv2.conv.bias", "up_blocks.8.res_blocks.5.conv2.conv.bias"),
    # flat 9 -- compress_space m=2, 128 -> 64
    ("decoder.up_blocks.3.upsamplers.0.conv.conv.weight", "up_blocks.9.conv.conv.weight"),
    ("decoder.up_blocks.3.upsamplers.0.conv.conv.bias", "up_blocks.9.conv.conv.bias"),
    # flat 10 -- res_x x4 @64
    ("decoder.up_blocks.3.resnets.0.conv1.conv.weight", "up_blocks.10.res_blocks.0.conv1.conv.weight"),
    ("decoder.up_blocks.3.resnets.0.conv1.conv.bias", "up_blocks.10.res_blocks.0.conv1.conv.bias"),
    ("decoder.up_blocks.3.resnets.0.conv2.conv.weight", "up_blocks.10.res_blocks.0.conv2.conv.weight"),
    ("decoder.up_blocks.3.resnets.0.conv2.conv.bias", "up_blocks.10.res_blocks.0.conv2.conv.bias"),
    ("decoder.up_blocks.3.resnets.1.conv1.conv.weight", "up_blocks.10.res_blocks.1.conv1.conv.weight"),
    ("decoder.up_blocks.3.resnets.1.conv1.conv.bias", "up_blocks.10.res_blocks.1.conv1.conv.bias"),
    ("decoder.up_blocks.3.resnets.1.conv2.conv.weight", "up_blocks.10.res_blocks.1.conv2.conv.weight"),
    ("decoder.up_blocks.3.resnets.1.conv2.conv.bias", "up_blocks.10.res_blocks.1.conv2.conv.bias"),
    ("decoder.up_blocks.3.resnets.2.conv1.conv.weight", "up_blocks.10.res_blocks.2.conv1.conv.weight"),
    ("decoder.up_blocks.3.resnets.2.conv1.conv.bias", "up_blocks.10.res_blocks.2.conv1.conv.bias"),
    ("decoder.up_blocks.3.resnets.2.conv2.conv.weight", "up_blocks.10.res_blocks.2.conv2.conv.weight"),
    ("decoder.up_blocks.3.resnets.2.conv2.conv.bias", "up_blocks.10.res_blocks.2.conv2.conv.bias"),
    ("decoder.up_blocks.3.resnets.3.conv1.conv.weight", "up_blocks.10.res_blocks.3.conv1.conv.weight"),
    ("decoder.up_blocks.3.resnets.3.conv1.conv.bias", "up_blocks.10.res_blocks.3.conv1.conv.bias"),
    ("decoder.up_blocks.3.resnets.3.conv2.conv.weight", "up_blocks.10.res_blocks.3.conv2.conv.weight"),
    ("decoder.up_blocks.3.resnets.3.conv2.conv.bias", "up_blocks.10.res_blocks.3.conv2.conv.bias"),
    ("decoder.conv_out.conv.weight", "conv_out.conv.weight"),
    ("decoder.conv_out.conv.bias", "conv_out.conv.bias"),
    ("latents_mean", "per_channel_statistics.mean-of-means"),
    ("latents_std", "per_channel_statistics.std-of-means"),
]


# --------------------------------------------------------------------------
# fixture helpers
# --------------------------------------------------------------------------
def _write_safetensors(path: str, tensors: list[tuple], metadata: dict | None) -> None:
    """Write a safetensors file from ``(name, dtype_str, raw_bytes, shape)`` tuples."""
    header: dict = {}
    blob = bytearray()
    off = 0
    for name, dtype, raw, shape in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [off, off + len(raw)],
        }
        blob += raw
        off += len(raw)
    if metadata is not None:
        header["__metadata__"] = metadata
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)


def _shrink(shape: list[int]) -> list[int]:
    """Scale channel dims by 1/64, leaving the kernel dims untouched."""
    if len(shape) == 1:
        return [max(1, shape[0] // 64)]
    return [max(1, shape[0] // 64), max(1, shape[1] // 64)] + list(shape[2:])


def _tiny_source_shapes() -> dict[str, list[int]]:
    """Source-side (diffusers) shapes for the shrunk fixture."""
    reference = cv.expected_shapes()
    return {src: _shrink(reference[dst]) for src, dst in build_key_map()}


#: Encoder tensors that must be silently dropped (they are byte-identical to
#: the stock encoder upstream, which is why PrunaVAED does not re-ship one).
_ENCODER_SPECS = {
    "encoder.conv_in.conv.weight": [2, 1, 3, 3, 3],
    "encoder.conv_in.conv.bias": [2],
    "encoder.conv_out.conv.weight": [3, 16, 3, 3, 3],
}


def _build(tmp_path, *, drop=(), extra=None, dtype_overrides=None):
    """Build a shrunk PrunaVAED-shaped fixture; returns ``(src_path, out_path)``.

    ``drop`` removes source tensors (missing-tensor path), ``extra`` adds
    ``{name: shape}`` entries (unexpected-tensor path) and ``dtype_overrides``
    forces a non-BF16 dtype on named tensors.
    """
    rng = np.random.default_rng(20260805)
    dtype_overrides = dtype_overrides or {}
    specs = dict(_ENCODER_SPECS)
    specs.update(_tiny_source_shapes())
    if extra:
        specs.update(extra)

    tensors: list[tuple] = []
    for name, shape in specs.items():
        if name in drop:
            continue
        n = int(np.prod(shape))
        dtype = dtype_overrides.get(name, "BF16")
        if dtype == "BF16":
            # Random bit patterns: distinct per tensor, which is what makes the
            # MD5 cross-check able to tell two same-shaped tensors apart.
            raw = rng.integers(0, 0x10000, size=n, dtype=np.uint16).astype("<u2").tobytes()
        elif dtype == "F32":
            raw = rng.standard_normal(n).astype("<f4").tobytes()
        else:  # pragma: no cover - only the two above are used
            raise ValueError(dtype)
        tensors.append((name, dtype, raw, shape))

    src_path = os.path.join(str(tmp_path), "diffusion_pytorch_model.safetensors")
    out_path = os.path.join(str(tmp_path), "PrunaVAED-decoder-bf16.safetensors")
    _write_safetensors(src_path, tensors, {"format": "pt"})
    return src_path, out_path


def _run(src, out, **kwargs):
    params = dict(
        source_repo="PrunaAI/PrunaVAED",
        source_revision="4baacd7ef66a6131439542c1f05872afe042e128",
        source_filename="vae/diffusion_pytorch_model.safetensors",
        expected_sha256=None,
        reference_expected=False,
    )
    params.update(kwargs)
    return convert_vae(src, out, **params)


def _read_header(path: str) -> dict:
    with open(path, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hl))


# --------------------------------------------------------------------------
# the key map itself
# --------------------------------------------------------------------------
def test_key_map_matches_the_hand_written_table():
    assert build_key_map() == GOLDEN_KEY_MAP


def test_key_map_is_a_bijection():
    pairs = build_key_map()
    sources = [s for s, _ in pairs]
    targets = [t for _, t in pairs]
    assert len(set(sources)) == len(sources)
    assert len(set(targets)) == len(targets)
    assert len(pairs) == cv.EXPECTED_TENSOR_COUNT == 102


def test_output_keys_carry_no_prefix():
    # The backend loads this file with model_sd_ops=None, so whatever is written
    # here is exactly what load_state_dict sees.
    for key in expected_output_keys():
        assert not key.startswith("decoder.")
        assert not key.startswith("vae.")


def test_expected_shapes_cover_every_key_and_total_345m():
    shapes = cv.expected_shapes()
    keys = expected_output_keys()
    assert set(shapes) == set(keys)
    stats = {dst for _s, dst in cv._STATS_MAP}
    total = 0
    for name in keys:
        if name in stats:
            continue
        n = 1
        for d in shapes[name]:
            n *= d
        total += n
    assert total == cv.EXPECTED_PARAMETER_TOTAL == 345_006_256
    assert total * 2 == cv.EXPECTED_DECODER_BYTES


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------
def test_convert_vae_happy_path(tmp_path):
    src, out = _build(tmp_path)
    report = _run(src, out)

    assert report.passed, report.summary()
    assert os.path.isfile(out)
    assert report.tensor_count == 102
    # Every check either passed or was explicitly skipped, and nothing failed.
    assert report.failed_checks == []
    assert set(report.skipped) == {
        "3_tensor_shapes",
        "4_parameter_total",
        "6_latent_statistics_vs_reference",
    }
    summary = report.summary()
    for name in ("1_tensor_count", "2_key_set", "5_md5_passthrough",
                 "7_output_header_roundtrip", "8_total_size"):
        assert f"[PASS] {name}" in summary
    assert "RESULT: PASS" in summary


def test_output_key_set_is_exactly_the_expected_set(tmp_path):
    src, out = _build(tmp_path)
    _run(src, out)

    header = _read_header(out)
    keys = [k for k in header if k != "__metadata__"]
    assert keys == expected_output_keys()          # order too, not just the set
    assert len(keys) == 102
    assert "up_blocks.3.norm3.weight" in keys      # projection resnet #1
    assert "up_blocks.6.conv_shortcut.weight" in keys  # projection resnet #2
    assert "up_blocks.4.conv.conv.weight" in keys  # upsampler, no res_blocks level
    assert "per_channel_statistics.mean-of-means" in keys
    assert not any(k.startswith("encoder.") for k in keys)


def test_payload_is_a_verbatim_copy(tmp_path):
    src, out = _build(tmp_path)
    _run(src, out)

    src_header, out_header = _read_header(src), _read_header(out)
    with open(src, "rb") as sf, open(out, "rb") as of:
        src_base = 8 + struct.unpack("<Q", sf.read(8))[0]
        out_base = 8 + struct.unpack("<Q", of.read(8))[0]
        for source_key, target_key in build_key_map():
            s0, s1 = src_header[source_key]["data_offsets"]
            o0, o1 = out_header[target_key]["data_offsets"]
            sf.seek(src_base + s0)
            of.seek(out_base + o0)
            assert sf.read(s1 - s0) == of.read(o1 - o0), target_key
            assert out_header[target_key]["dtype"] == "BF16"
            assert out_header[target_key]["shape"] == src_header[source_key]["shape"]


def test_metadata_has_config_without_decoder_blocks(tmp_path):
    src, out = _build(tmp_path)
    _run(src, out)

    meta = _read_header(out)["__metadata__"]
    assert set(meta) == {"config", "model_version", "license", "provenance"}

    config = json.loads(meta["config"])
    # Present at all: ltx-core's sft_loader does json.loads(metadata()["config"])
    # with no guard, so a missing key is a crash before the configurator runs.
    assert set(config) == {"vae"}
    vae = config["vae"]
    assert vae["_class_name"] == "PrunaVAEDDecoder"
    assert vae["latent_channels"] == 128
    assert vae["patch_size"] == 4
    assert vae["norm_layer"] == "pixel_norm"
    assert vae["causal_decoder"] is False
    assert vae["timestep_conditioning"] is False
    assert vae["decoder_base_channels"] == 128
    # The whole point of §4.2: with decoder_blocks present the *stock*
    # configurator would happily build a decoder without the projection
    # resnets. Without it, that path raises instead of silently misbehaving.
    assert "decoder_blocks" not in vae
    assert "decoder_blocks" not in config

    assert meta["model_version"] == "PrunaVAED-v2"
    assert "LTX-2 Community License" in meta["license"]

    prov = json.loads(meta["provenance"])
    assert prov["source_repo"] == "PrunaAI/PrunaVAED"
    assert prov["source_revision"] == "4baacd7ef66a6131439542c1f05872afe042e128"
    assert len(prov["source_sha256"]) == 64
    assert prov["tool_version"]


def test_output_is_reproducible(tmp_path):
    """No timestamps anywhere: re-running must give a byte-identical file."""
    src, out_a = _build(tmp_path)
    out_b = os.path.join(str(tmp_path), "again.safetensors")
    _run(src, out_a)
    _run(src, out_b)
    with open(out_a, "rb") as a, open(out_b, "rb") as b:
        assert a.read() == b.read()


# --------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------
def test_missing_tensor_fails(tmp_path):
    src, out = _build(tmp_path, drop=("decoder.up_blocks.1.conv_in.norm3.weight",))
    with pytest.raises(MissingTensorError) as exc:
        _run(src, out)
    assert "norm3.weight" in str(exc.value)
    assert not os.path.exists(out)


def test_extra_decoder_tensor_fails(tmp_path):
    src, out = _build(tmp_path, extra={"decoder.up_blocks.4.resnets.0.conv1.conv.weight": [2, 2, 3, 3, 3]})
    with pytest.raises(UnexpectedTensorError) as exc:
        _run(src, out)
    assert "up_blocks.4" in str(exc.value)
    assert not os.path.exists(out)


def test_non_bf16_source_fails(tmp_path):
    src, out = _build(tmp_path, dtype_overrides={"decoder.conv_out.conv.weight": "F32"})
    with pytest.raises(SourceDtypeError) as exc:
        _run(src, out)
    assert "F32" in str(exc.value)


def test_pinned_sha256_mismatch_fails(tmp_path):
    from converter.download import Sha256MismatchError

    src, out = _build(tmp_path)
    with pytest.raises(Sha256MismatchError):
        _run(src, out, expected_sha256="0" * 64)
    assert not os.path.exists(out)


def test_md5_check_catches_a_same_shape_swap(tmp_path, monkeypatch):
    """A writer that crosses two identically-shaped tensors must be caught.

    ``up_blocks.5.res_blocks.0``'s ``conv1`` and ``conv2`` have the same shape,
    the same byte length and the same dtype, so the tensor count, the key set,
    the shape table, the parameter total and the file size are all still
    exactly right after a swap. Only the byte-level cross-check notices.
    """
    src, out = _build(tmp_path)
    a = "up_blocks.5.res_blocks.0.conv1.conv.weight"
    b = "up_blocks.5.res_blocks.0.conv2.conv.weight"

    real_writer = cv._write_safetensors_streaming

    def swapping_writer(out_path, source_path, plan, metadata):
        by_name = {e["dst"]: e for e in plan}
        assert by_name[a]["nbytes"] == by_name[b]["nbytes"]
        by_name[a]["src_start"], by_name[b]["src_start"] = (
            by_name[b]["src_start"],
            by_name[a]["src_start"],
        )
        return real_writer(out_path, source_path, plan, metadata)

    monkeypatch.setattr(cv, "_write_safetensors_streaming", swapping_writer)

    with pytest.raises(SelfVerificationError) as exc:
        _run(src, out)
    message = str(exc.value)
    assert "5_md5_passthrough" in message
    assert a in message and b in message
    # ...and nothing else noticed: the swap is invisible to every other item.
    assert "[FAIL] 1_tensor_count" not in message
    assert "[FAIL] 2_key_set" not in message
    assert "[FAIL] 8_total_size" not in message


def test_injectivity_check_catches_a_duplicated_source(tmp_path, monkeypatch):
    """A key map that reads the same source tensor twice must be caught."""
    a_src = "decoder.up_blocks.1.resnets.0.conv1.conv.weight"
    b_src = "decoder.up_blocks.1.resnets.0.conv2.conv.weight"
    # The fixture omits b_src so the broken map is *self-consistent*: nothing is
    # missing and nothing is unmapped, exactly like a real off-by-one in the
    # table would look. Only the injectivity half of item 5 can see it.
    src, out = _build(tmp_path, drop=(b_src,))
    broken = [(a_src if s == b_src else s, t) for s, t in build_key_map()]

    monkeypatch.setattr(cv, "build_key_map", lambda: broken)

    with pytest.raises(SelfVerificationError) as exc:
        _run(src, out)
    assert "not injective" in str(exc.value)


def test_reference_expected_rejects_a_shrunk_model(tmp_path):
    """The real-model checks must actually bite when they are switched on."""
    src, out = _build(tmp_path)
    with pytest.raises(SelfVerificationError) as exc:
        _run(src, out, reference_expected=True)
    message = str(exc.value)
    assert "3_tensor_shapes" in message
    assert "4_parameter_total" in message
    assert "8_total_size" in message


def test_latent_statistics_cross_check(tmp_path):
    """Item 6 passes against a reference holding the same statistics bytes."""
    src, out = _build(tmp_path)

    src_header = _read_header(src)
    with open(src, "rb") as f:
        base = 8 + struct.unpack("<Q", f.read(8))[0]
        blobs = {}
        for source_key, target_key in cv._STATS_MAP:
            s0, s1 = src_header[source_key]["data_offsets"]
            f.seek(base + s0)
            blobs[target_key] = f.read(s1 - s0)

    ref_path = os.path.join(str(tmp_path), "LTX23_video_vae_bf16.safetensors")
    _write_safetensors(
        ref_path,
        [(name, "BF16", raw, [len(raw) // 2]) for name, raw in blobs.items()],
        {"config": "{}"},
    )

    report = _run(src, out, reference_vae_path=ref_path)
    assert report.passed, report.summary()
    assert "6_latent_statistics_vs_reference" not in report.skipped
    assert "[PASS] 6_latent_statistics_vs_reference" in report.summary()


def test_latent_statistics_mismatch_is_reported(tmp_path):
    src, out = _build(tmp_path)
    ref_path = os.path.join(str(tmp_path), "LTX23_video_vae_bf16.safetensors")
    wrong = b"\x00\x11" * 2
    _write_safetensors(
        ref_path,
        [
            ("per_channel_statistics.mean-of-means", "BF16", wrong, [2]),
            ("per_channel_statistics.std-of-means", "BF16", wrong, [2]),
        ],
        {"config": "{}"},
    )
    with pytest.raises(SelfVerificationError) as exc:
        _run(src, out, reference_vae_path=ref_path)
    assert "6_latent_statistics_vs_reference" in str(exc.value)
