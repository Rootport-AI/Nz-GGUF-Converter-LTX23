"""Fixture coverage for the opt-in, fail-closed LTX 2.5 conversion profile."""

from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGUFReader

from converter import ltx25
from converter import cli
from converter import quant_kernels as qk


def _bf16(values: np.ndarray) -> bytes:
    u32 = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    bias = ((u32 >> 16) & 1) + 0x7FFF
    return ((u32 + bias) >> 16).astype("<u2").tobytes()


def _write_safetensors(path, tensors, metadata=None) -> None:
    """Write a tiny safetensors file without requiring NumPy BF16 support."""
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, raw, shape in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    if metadata is not None:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def _fixture_lock(path) -> ltx25.SourceLock:
    digest = ltx25.sha256_of_file(path)
    return ltx25.SourceLock(
        repo_id="fixture/LTX-2.5",
        artifact_revision="fixture-revision",
        filename="fixture.safetensors",
        expected_size=path.stat().st_size,
        source_sha256=digest,
        lfs_sha256=digest,
        git_blob_oid="fixture-blob",
    )


def _source(tmp_path, *, raw_key="model.diffusion_model.transformer.weight", dtype="BF16", metadata=None):
    path = tmp_path / "fixture.safetensors"
    if dtype == "BF16":
        raw = _bf16(np.linspace(-1.0, 1.0, 256, dtype=np.float32))
    elif dtype == "I8":
        raw = bytes(range(16))
    else:
        raise AssertionError(dtype)
    config = '{ "transformer" : { "width" : 256 }, "quant_note" : "int8 is diagnostic only" }'
    _write_safetensors(
        path,
        [(raw_key, dtype, raw, [1, 256] if dtype == "BF16" else [16])],
        metadata if metadata is not None else {"config": config, "license": "fixture"},
    )
    return path, config


def _approved_map(inventory_path, map_path):
    policy = ltx25.build_policy_map(inventory_path, map_path)
    assert policy["status"] == "draft"
    policy["status"] = "approved"  # Simulates separate human review of checked-in E4.
    policy["map_sha256"] = ltx25._sha256_bytes(
        ltx25._canonical_json_bytes({key: value for key, value in policy.items() if key != "map_sha256"})
    )
    map_path.write_bytes(ltx25._canonical_json_bytes(policy) + b"\n")
    return policy


def _write_oracle(path, config, shapes) -> None:
    payload = {
        "format": ltx25.BUILDER_ORACLE_FORMAT,
        "profile": "ltx25",
        "official_code_commit": ltx25.OFFICIAL_CODE_COMMIT,
        "config_bytes_sha256": ltx25._sha256_bytes(config.encode("utf-8")),
        "builder_state_dict_shapes": shapes,
    }
    payload["oracle_sha256"] = ltx25._sha256_bytes(ltx25._canonical_json_bytes(payload))
    path.write_bytes(ltx25._canonical_json_bytes(payload) + b"\n")


def _write_component_oracle(path, config, component_rows) -> None:
    fixed = {
        "transformer": ("emit", "transformer", ""),
        "audio_embeddings_connector": (
            "emit",
            "gemma-audio-embeddings-connector",
            "audio_embeddings_connector.",
        ),
        "video_embeddings_connector": (
            "emit",
            "gemma-video-embeddings-connector",
            "video_embeddings_connector.",
        ),
    }
    components = {}
    for name, (classification, component_id, native_prefix) in fixed.items():
        rows = component_rows[name]
        components[name] = {
            "classification": classification,
            "component_id": component_id,
            "native_prefix": native_prefix,
            "state_dict_shapes": {key: shape for key, (shape, _) in rows.items()},
        }
    payload = {
        "format": ltx25.BUILDER_ORACLE_FORMAT,
        "profile": "ltx25",
        "official_code_commit": ltx25.OFFICIAL_CODE_COMMIT,
        "config_bytes_sha256": ltx25._sha256_bytes(config.encode("utf-8")),
        "components": components,
    }
    payload["oracle_sha256"] = ltx25._sha256_bytes(ltx25._canonical_json_bytes(payload))
    path.write_bytes(ltx25._canonical_json_bytes(payload) + b"\n")


def _cli_config(lock) -> dict:
    return {
        "profiles": {
            "ltx25": {
                "source_repo_id": lock.repo_id,
                "artifact_revision": lock.artifact_revision,
                "source_filename": lock.filename,
                "source_expected_size": lock.expected_size,
                "source_sha256": lock.source_sha256,
                "lfs_sha256": lock.lfs_sha256,
                "git_blob_oid": lock.git_blob_oid,
            }
        }
    }


def _recompute_map(path, policy) -> None:
    policy["map_sha256"] = ltx25._sha256_bytes(
        ltx25._canonical_json_bytes({key: value for key, value in policy.items() if key != "map_sha256"})
    )
    path.write_bytes(ltx25._canonical_json_bytes(policy) + b"\n")


def _prepared_artifacts(tmp_path, *, tensor_count=1):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = '{"transformer":{"width":256}}'
    source = tmp_path / "prepared.safetensors"
    tensors = []
    shapes = {}
    for index in range(tensor_count):
        name = f"model.diffusion_model.transformer.{index}.weight"
        tensors.append((name, "BF16", _bf16(np.full(256, index + 1, np.float32)), [1, 256]))
        shapes[f"transformer.{index}.weight"] = [1, 256]
    _write_safetensors(source, tensors, {"config": config, "license": "fixture"})
    lock = _fixture_lock(source)
    inventory = tmp_path / "inventory.json"
    oracle = tmp_path / "oracle.json"
    policy = tmp_path / "map.json"
    output = tmp_path / "output.gguf"
    _write_oracle(oracle, config, shapes)
    ltx25.inspect_ltx25(source, source_lock=lock, audit_path=inventory, builder_oracle_path=oracle)
    approved = _approved_map(inventory, policy)
    return SimpleNamespace(
        source=source,
        config=config,
        lock=lock,
        inventory=inventory,
        oracle=oracle,
        policy_path=policy,
        policy=approved,
        output=output,
    )


def test_incomplete_source_lock_blocks_conversion_before_artifact_admission(tmp_path):
    source, _ = _source(tmp_path)
    incomplete_lock = replace(
        ltx25.DEFAULT_SOURCE_LOCK,
        source_sha256=ltx25.SOURCE_LOCK_UNCONFIRMED,
        lfs_sha256=ltx25.SOURCE_LOCK_UNCONFIRMED,
    )
    with pytest.raises(ltx25.SourceLockIncompleteError, match="source-lock-incomplete"):
        ltx25.inspect_ltx25(source, source_lock=incomplete_lock)


def test_cli_defaults_to_ltx23_and_gates_ltx25_without_traceback(capsys, monkeypatch):
    incomplete_lock = replace(
        ltx25.DEFAULT_SOURCE_LOCK,
        source_sha256=ltx25.SOURCE_LOCK_UNCONFIRMED,
        lfs_sha256=ltx25.SOURCE_LOCK_UNCONFIRMED,
    )
    monkeypatch.setattr(cli, "_load_config", lambda: _cli_config(incomplete_lock))
    parser = cli.build_parser()
    assert parser.parse_args(["convert"]).model == "ltx23"
    assert parser.parse_args(["build-policy", "--model", "ltx25"]).command == "build-policy"
    assert cli.main(["all", "--model", "ltx25"]) == 1
    assert "ltx23-only" in capsys.readouterr().err
    assert cli.main(["download", "--model", "ltx25"]) == 1
    err = capsys.readouterr().err
    assert "source-lock-incomplete" in err
    assert "Traceback" not in err
    for argv in (
        ["inspect", "--model", "ltx25", "--st-path", "does-not-exist"],
        ["build-map", "--model", "ltx25", "--inventory", "does-not-exist"],
        ["convert", "--model", "ltx25", "--st-path", "does-not-exist", "--map", "does-not-exist"],
        ["self-verify", "--model", "ltx25", "--out", "does-not-exist", "--map", "does-not-exist"],
    ):
        assert cli.main(argv) == 1
        assert "source-lock-incomplete" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parser.parse_args(["build-map", "--model", "ltx25", "--status", "approved"])
    with pytest.raises(SystemExit):
        parser.parse_args(["convert", "--quant-workers", "0"])


def test_cli_quant_worker_profile_defaults_are_frozen():
    assert cli._quant_worker_count(cli.argparse.Namespace(model="ltx23", quant_workers=None), {}) == 1
    assert cli._quant_worker_count(
        cli.argparse.Namespace(model="ltx25", quant_workers=None),
        {"profiles": {"ltx25": {"quant_workers": 4}}},
    ) == 4


def test_ltx23_omitted_and_explicit_cli_paths_and_legacy_stages_match(tmp_path, monkeypatch):
    source = tmp_path / "legacy.safetensors"
    typemap = tmp_path / "legacy.json"
    reference = tmp_path / "reference.gguf"
    source.write_bytes(b"fixture")
    typemap.write_text("{}", encoding="utf-8")
    reference.write_bytes(b"fixture")
    config = {
        "source": {"local_dir": str(tmp_path), "filename": source.name},
        "reference": {"gguf_path": str(reference), "typemap_path": str(typemap)},
        "output": {"dir": str(tmp_path), "filename": "legacy.gguf"},
    }
    parser = cli.build_parser()
    calls = []
    monkeypatch.setattr(
        cli.convert_mod,
        "convert",
        lambda st, tm, out, reference_expected=True, quant_workers=1: calls.append(
            (Path(st), Path(tm), Path(out), reference_expected, quant_workers)
        )
        or Path(out),
    )
    assert cli.cmd_convert(parser.parse_args(["convert"]), config) == 0
    assert cli.cmd_convert(parser.parse_args(["convert", "--model", "ltx23"]), config) == 0
    assert calls[0] == calls[1]
    assert calls[0][-1] == 1

    extracted = []
    monkeypatch.setattr(cli.typemap_mod, "extract_typemap", lambda path: extracted.append(Path(path)) or [])
    monkeypatch.setattr(cli.typemap_mod, "save_typemap", lambda *args, **kwargs: None)
    assert cli.cmd_extract_typemap(parser.parse_args(["extract-typemap"]), config) == 0
    assert cli.cmd_extract_typemap(parser.parse_args(["extract-typemap", "--model", "ltx23"]), config) == 0
    assert extracted == [reference, reference]

    stages = []
    monkeypatch.setattr(cli, "cmd_download", lambda *_args: stages.append("download") or 0)
    monkeypatch.setattr(cli, "cmd_extract_typemap", lambda *_args: stages.append("extract") or 0)
    monkeypatch.setattr(cli, "cmd_convert", lambda *_args: stages.append("convert") or 0)
    monkeypatch.setattr(cli, "cmd_verify", lambda *_args: stages.append("verify") or 0)
    assert cli.cmd_all(parser.parse_args(["all"]), config) == 0
    first = list(stages)
    stages.clear()
    assert cli.cmd_all(parser.parse_args(["all", "--model", "ltx23"]), config) == 0
    assert first == stages == ["download", "convert", "verify"]


def test_inspect_normalizes_once_and_records_diagnostic_without_rejecting(tmp_path):
    source, config = _source(tmp_path, raw_key="model.diffusion_model.quant_marker.weight")
    inventory_path = tmp_path / "inventory.json"
    oracle_path = tmp_path / "oracle.json"
    _write_oracle(oracle_path, config, {"quant_marker.weight": [1, 256]})
    inventory = ltx25.inspect_ltx25(
        source,
        source_lock=_fixture_lock(source),
        audit_path=inventory_path,
        builder_oracle_path=oracle_path,
    )

    assert inventory.tensors[0]["builder_state_dict_key"] == "quant_marker.weight"
    assert inventory.tensors[0]["raw_key"].startswith("model.diffusion_model.")
    assert any("quant" in marker for marker in inventory.diagnostics)
    assert inventory.builder_oracle_sha256 == ltx25.load_builder_oracle(oracle_path)["oracle_sha256"]
    loaded = ltx25.load_inventory(inventory_path)
    assert loaded["inventory_sha256"] == inventory.inventory_sha256


def test_cli_inspect_convert_and_self_verify_fixture_success(tmp_path):
    source, config = _source(tmp_path)
    lock = _fixture_lock(source)
    inventory_path = tmp_path / "inventory.json"
    oracle_path = tmp_path / "oracle.json"
    map_path = tmp_path / "map.json"
    output = tmp_path / "cli.gguf"
    _write_oracle(oracle_path, config, {"transformer.weight": [1, 256]})
    cfg = _cli_config(lock)

    inspect_args = cli.argparse.Namespace(
        model="ltx25", st_path=str(source), inventory=str(inventory_path), builder_oracle=str(oracle_path)
    )
    assert cli.cmd_inspect(inspect_args, cfg) == 0
    _approved_map(inventory_path, map_path)
    convert_args = cli.argparse.Namespace(
        model="ltx25",
        st_path=str(source),
        map=str(map_path),
        out=str(output),
        builder_oracle=str(oracle_path),
        force=False,
    )
    assert cli.cmd_convert(convert_args, cfg) == 0
    verify_args = cli.argparse.Namespace(
        model="ltx25",
        st_path=str(source),
        inventory=str(inventory_path),
        map=str(map_path),
        out=str(output),
        builder_oracle=str(oracle_path),
        manifest=None,
    )
    assert cli.cmd_verify(verify_args, cfg) == 0


def test_inspect_rejects_disallowed_dtype_unknown_and_duplicate_header_keys(tmp_path):
    int8_source, _ = _source(tmp_path, dtype="I8")
    with pytest.raises(ltx25.SourceRejectedError, match="E3 source-dtype allowlist"):
        ltx25.inspect_ltx25(
            int8_source, source_lock=_fixture_lock(int8_source), require_builder_oracle=False
        )

    unknown_source, _ = _source(tmp_path, raw_key="gemma.embed.weight")
    with pytest.raises(ltx25.SourceRejectedError, match="unknown component"):
        ltx25.inspect_ltx25(
            unknown_source, source_lock=_fixture_lock(unknown_source), require_builder_oracle=False
        )

    duplicate = tmp_path / "duplicate.safetensors"
    config = json.dumps({"transformer": {}})
    raw_header = (
        '{"model.diffusion_model.x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},'
        '"model.diffusion_model.x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},'
        f'"__metadata__":{{"config":{json.dumps(config)}}}}}'
    ).encode("utf-8")
    duplicate.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"\x00\x00")
    with pytest.raises(ltx25.SourceRejectedError, match="duplicate JSON key"):
        ltx25.inspect_ltx25(
            duplicate, source_lock=_fixture_lock(duplicate), require_builder_oracle=False
        )


def test_inspect_rejects_malformed_tensor_offset_payload(tmp_path):
    malformed = tmp_path / "bad-offset.safetensors"
    header = {
        "model.diffusion_model.weight": {
            "dtype": "BF16",
            "shape": [1, 256],
            "data_offsets": [0, 2],
        },
        "__metadata__": {"config": json.dumps({"transformer": {}})},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    malformed.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00\x00")
    with pytest.raises(ltx25.SourceRejectedError, match="payload is"):
        ltx25.inspect_ltx25(
            malformed, source_lock=_fixture_lock(malformed), require_builder_oracle=False
        )


@pytest.mark.parametrize(
    ("header_bytes", "payload", "message"),
    [
        (b"", b"", "shorter"),
        (b"\xff", b"", "not UTF-8"),
        (b"{", b"", "not valid JSON"),
    ],
    ids=["invalid-length", "non-utf8", "invalid-json"],
)
def test_inspect_rejects_malformed_header_encodings(tmp_path, header_bytes, payload, message):
    path = tmp_path / f"{message}.safetensors"
    if header_bytes:
        path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    else:
        path.write_bytes(b"\x00")
    with pytest.raises(ltx25.SourceRejectedError, match=message):
        ltx25.inspect_ltx25(path, source_lock=_fixture_lock(path), require_builder_oracle=False)


@pytest.mark.parametrize(
    ("entry", "payload", "message"),
    [
        ({"dtype": "BF16", "shape": [True], "data_offsets": [0, 2]}, b"\x00\x00", "invalid non-negative"),
        ({"dtype": "BF16", "shape": [1], "data_offsets": [0, 3]}, b"\x00\x00", "outside payload"),
        (
            {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
            b"\x00\x00\x00",
            "overlapping",
        ),
    ],
    ids=["invalid-shape", "offset-bounds", "overlap"],
)
def test_inspect_rejects_header_shape_and_range_failures(tmp_path, entry, payload, message):
    header = {
        "model.diffusion_model.a": entry,
        "__metadata__": {"config": json.dumps({"transformer": {}})},
    }
    if message == "overlapping":
        header["model.diffusion_model.b"] = {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [1, 3],
        }
    header_bytes = json.dumps(header).encode("utf-8")
    path = tmp_path / f"{message}.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    with pytest.raises(ltx25.SourceRejectedError, match=message):
        ltx25.inspect_ltx25(path, source_lock=_fixture_lock(path), require_builder_oracle=False)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "__metadata__.config"),
        ({"config": "not-json"}, "not JSON"),
    ],
    ids=["config-missing", "config-invalid"],
)
def test_inspect_rejects_missing_or_invalid_config(tmp_path, metadata, message):
    path = tmp_path / f"{message}.safetensors"
    _write_safetensors(
        path,
        [("model.diffusion_model.weight", "BF16", b"\x00\x00", [1])],
        metadata,
    )
    with pytest.raises(ltx25.SourceRejectedError, match=message):
        ltx25.inspect_ltx25(path, source_lock=_fixture_lock(path), require_builder_oracle=False)


@pytest.mark.parametrize(
    ("dtype", "raw", "shape"),
    [
        ("F16", b"\x00\x00", [1]),
        ("F8_E4M3FN", b"\x00", [1]),
        ("F4_E2M1FN", b"\x00", [2]),
    ],
    ids=["f16", "fp8", "nvfp4"],
)
def test_inspect_rejects_dtypes_outside_e3_allowlist(tmp_path, dtype, raw, shape):
    path = tmp_path / f"{dtype}.safetensors"
    _write_safetensors(
        path,
        [("model.diffusion_model.weight", dtype, raw, shape)],
        {"config": json.dumps({"transformer": {}})},
    )
    with pytest.raises(ltx25.SourceRejectedError, match="E3 source-dtype allowlist"):
        ltx25.inspect_ltx25(path, source_lock=_fixture_lock(path), require_builder_oracle=False)


def test_builder_oracle_must_match_exact_config_keys_and_shapes(tmp_path):
    source, config = _source(tmp_path)
    lock = _fixture_lock(source)
    oracle_path = tmp_path / "oracle.json"
    with pytest.raises(ltx25.InventoryMismatchError, match="builder-oracle-missing"):
        ltx25.inspect_ltx25(source, source_lock=lock)
    with pytest.raises(ltx25.InventoryMismatchError, match="builder-oracle-missing"):
        ltx25.inspect_ltx25(
            source,
            source_lock=lock,
            expected_builder_shapes={"transformer.weight": [1, 256]},
        )
    assert ltx25.inspect_ltx25(
        source,
        source_lock=lock,
        expected_builder_shapes={"transformer.weight": [1, 256]},
        require_builder_oracle=False,
    ).tensors
    _write_oracle(oracle_path, config, {"transformer.weight": [1, 256]})
    assert ltx25.inspect_ltx25(source, source_lock=lock, builder_oracle_path=oracle_path).tensors

    _write_oracle(oracle_path, config + " ", {"transformer.weight": [1, 256]})
    with pytest.raises(ltx25.InventoryMismatchError, match="config digest"):
        ltx25.inspect_ltx25(source, source_lock=lock, builder_oracle_path=oracle_path)

    _write_oracle(oracle_path, config, {"transformer.weight": [2, 128]})
    with pytest.raises(ltx25.InventoryMismatchError, match="shape_mismatch"):
        ltx25.inspect_ltx25(source, source_lock=lock, builder_oracle_path=oracle_path)


def test_three_component_oracle_emits_bare_bundle_connectors_and_preserves_official_f32(tmp_path):
    config = '{"transformer":{"width":256}}'
    source = tmp_path / "components.safetensors"
    connector_bits = np.resize(
        np.asarray([0x0000, 0x8000, 0x7FC1, 0x7FFF, 0x7F81, 0xFFC1, 0x3F80, 0xBF80], dtype="<u2"),
        256,
    )
    audio_raw = connector_bits.tobytes()
    video_raw = np.ascontiguousarray(connector_bits[::-1]).tobytes()
    _write_safetensors(
        source,
        [
            (
                "model.diffusion_model.transformer.f32.weight",
                "F32",
                np.ones(256, dtype="<f4").tobytes(),
                [1, 256],
            ),
            (
                "model.diffusion_model.audio_embeddings_connector.audio.weight",
                "BF16",
                audio_raw,
                [1, 256],
            ),
            (
                "model.diffusion_model.video_embeddings_connector.video.weight",
                "BF16",
                video_raw,
                [1, 256],
            ),
        ],
        {"config": config},
    )
    oracle = tmp_path / "components-oracle.json"
    _write_component_oracle(
        oracle,
        config,
        {
            "transformer": {"transformer.f32.weight": ([1, 256], "F32")},
            "audio_embeddings_connector": {"audio.weight": ([1, 256], "BF16")},
            "video_embeddings_connector": {"video.weight": ([1, 256], "BF16")},
        },
    )
    inventory_path = tmp_path / "components-inventory.json"
    map_path = tmp_path / "components-map.json"
    inventory = ltx25.inspect_ltx25(
        source,
        source_lock=_fixture_lock(source),
        builder_oracle_path=oracle,
        audit_path=inventory_path,
    )
    assert [row["classification"] for row in inventory.tensors] == ["emit", "emit", "emit"]
    emitted = {row["builder_state_dict_key"]: row for row in inventory.tensors}
    assert emitted["transformer.f32.weight"]["source_dtype"] == "F32"
    connector_names = {
        "audio_embeddings_connector.audio.weight",
        "video_embeddings_connector.video.weight",
    }
    assert connector_names <= set(emitted)
    assert {emitted[name]["component_id"] for name in connector_names} == {
        "gemma-audio-embeddings-connector",
        "gemma-video-embeddings-connector",
    }
    policy = ltx25.build_policy_map(inventory_path, map_path)
    assert policy["status"] == "draft"
    policy_by_name = {row["name"]: row for row in policy["tensors"]}
    assert policy_by_name["transformer.f32.weight"]["ggml_type"] == "F32"
    assert all(policy_by_name[name]["ggml_type"] == "BF16" for name in connector_names)
    policy_by_name["transformer.f32.weight"]["ggml_type"] = "BF16"
    _recompute_map(map_path, policy)
    with pytest.raises(ltx25.PolicyMapError, match="F32 source must remain F32"):
        ltx25.load_policy_map(map_path, require_approved=False)

    policy = ltx25.build_policy_map(inventory_path, map_path)
    connector_row = next(row for row in policy["tensors"] if row["name"] in connector_names)
    connector_row["ggml_type"] = "Q4_K"
    connector_row["nbytes"] = ltx25._map_nbytes(connector_row["shape_logical"], "Q4_K")
    policy["type_counts"] = ltx25._type_counts(policy["tensors"])
    _recompute_map(map_path, policy)
    with pytest.raises(ltx25.PolicyMapError, match="connector .* must remain BF16"):
        ltx25.load_policy_map(map_path, require_approved=False)

    policy = ltx25.build_policy_map(inventory_path, map_path)
    _approved_map(inventory_path, map_path)
    output = tmp_path / "bundle.gguf"
    ltx25.convert_ltx25(
        source,
        map_path,
        output,
        source_lock=_fixture_lock(source),
        builder_oracle_path=oracle,
    )
    gguf_reader = GGUFReader(str(output))
    gguf = {tensor.name: tensor.tensor_type.name for tensor in gguf_reader.tensors}
    assert {name: gguf[name] for name in connector_names} == {
        "audio_embeddings_connector.audio.weight": "BF16",
        "video_embeddings_connector.video.weight": "BF16",
    }
    gguf_payloads = {
        tensor.name: np.asarray(tensor.data).view(np.uint8).reshape(-1).tobytes()
        for tensor in gguf_reader.tensors
        if tensor.name in connector_names
    }
    assert gguf_payloads["audio_embeddings_connector.audio.weight"] == audio_raw
    assert gguf_payloads["video_embeddings_connector.video.weight"] == video_raw
    del gguf_reader

    manifest = ltx25.verify_ltx25(
        output,
        map_path,
        inventory_path=inventory_path,
        source_path=source,
        source_lock=_fixture_lock(source),
        builder_oracle_path=oracle,
    )
    assert manifest["tensor_count"] == 3
    assert manifest["type_counts"] == {"BF16": 2, "F32": 1}

    payload_offset = output.read_bytes().find(audio_raw)
    assert payload_offset >= 0
    with output.open("r+b") as fh:
        fh.seek(payload_offset)
        fh.write(b"\x01")
    broken_manifest = dict(manifest)
    broken_manifest["output_sha256"] = ltx25.sha256_of_file(output)
    output.with_name(output.name + ".manifest.json").write_bytes(
        ltx25._canonical_json_bytes(broken_manifest) + b"\n"
    )
    with pytest.raises(ltx25.Ltx25Error, match="raw payload differs from source"):
        ltx25.verify_ltx25(
            output,
            map_path,
            inventory_path=inventory_path,
            source_path=source,
            source_lock=_fixture_lock(source),
            builder_oracle_path=oracle,
        )


def test_connector_source_dtype_must_be_bf16(tmp_path):
    config = '{"transformer":{"width":256}}'
    source = tmp_path / "connector-f32.safetensors"
    _write_safetensors(
        source,
        [
            (
                "model.diffusion_model.transformer.weight",
                "BF16",
                _bf16(np.ones(256, np.float32)),
                [1, 256],
            ),
            (
                "model.diffusion_model.audio_embeddings_connector.audio.weight",
                "F32",
                np.ones(256, dtype="<f4").tobytes(),
                [1, 256],
            ),
            (
                "model.diffusion_model.video_embeddings_connector.video.weight",
                "BF16",
                _bf16(np.ones(256, np.float32)),
                [1, 256],
            ),
        ],
        {"config": config},
    )
    oracle = tmp_path / "connector-oracle.json"
    _write_component_oracle(
        oracle,
        config,
        {
            "transformer": {"transformer.weight": ([1, 256], "BF16")},
            "audio_embeddings_connector": {"audio.weight": ([1, 256], "BF16")},
            "video_embeddings_connector": {"video.weight": ([1, 256], "BF16")},
        },
    )
    with pytest.raises(ltx25.SourceRejectedError, match="expected official E3 BF16"):
        ltx25.inspect_ltx25(source, source_lock=_fixture_lock(source), builder_oracle_path=oracle)


def test_cli_build_map_defaults_to_draft_and_protects_approved_path(tmp_path, capsys):
    source, config = _source(tmp_path)
    lock = _fixture_lock(source)
    inventory_path = tmp_path / "inventory.json"
    oracle_path = tmp_path / "oracle.json"
    approved_path = tmp_path / "approved.json"
    draft_path = tmp_path / "approved.draft.json"
    _write_oracle(oracle_path, config, {"transformer.weight": [1, 256]})
    ltx25.inspect_ltx25(
        source, source_lock=lock, audit_path=inventory_path, builder_oracle_path=oracle_path
    )
    cfg = _cli_config(lock)
    cfg["profiles"]["ltx25"].update(
        {
            "inventory_path": str(inventory_path),
            "map_path": str(approved_path),
            "draft_map_path": str(draft_path),
        }
    )
    args = cli.argparse.Namespace(model="ltx25", inventory=None, map=None)
    assert cli.cmd_build_map(args, cfg) == 0
    assert draft_path.is_file()
    assert ltx25.load_policy_map(draft_path, require_approved=False)["status"] == "draft"

    approved_path.write_bytes(b"approved-map-must-survive")
    args.map = str(approved_path)
    assert cli.cmd_build_map(args, cfg) == 1
    assert "approved E4 path is protected" in capsys.readouterr().err
    assert approved_path.read_bytes() == b"approved-map-must-survive"


def test_checked_in_e2_e3_e4_static_contract():
    """Keep the small committed evidence coherent without reading .artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    oracle = ltx25.load_builder_oracle(project_root / "typemap" / "ltx25_builder_oracle.json")
    inventory = ltx25.load_inventory(project_root / "typemap" / "ltx25_inventory.json")
    draft_policy = ltx25.load_policy_map(
        project_root / "typemap" / "ltx25_conversion_map.draft.json",
        require_approved=False,
    )
    approved_policy = ltx25.load_policy_map(
        project_root / "typemap" / "ltx25_conversion_map.json",
        require_approved=True,
    )

    assert oracle["oracle_sha256"] == "fa3b8518b5b55cbf9c8f072282eddb334bc60ec85ce4f13ea056b52adeb5bfe5"
    assert {name: len(component["state_dict_shapes"]) for name, component in oracle["components"].items()} == {
        "transformer": 4091,
        "audio_embeddings_connector": 129,
        "video_embeddings_connector": 129,
    }
    assert inventory["inventory_sha256"] == "c0966ac37f55f318be16334c1f1e1c2db1f467dafde09c0c2bd1b853206bde97"
    assert inventory["builder_oracle_sha256"] == oracle["oracle_sha256"]
    emitted = [row for row in inventory["tensors"] if row["classification"] == "emit"]
    excluded = [row for row in inventory["tensors"] if row["classification"] == "exclude"]
    assert len(emitted) == 4349
    assert not excluded
    assert Counter(row["source_dtype"] for row in emitted) == {"BF16": 4059, "F32": 290}
    assert Counter(row["component_id"] for row in emitted) == {
        "transformer": 4091,
        "gemma-audio-embeddings-connector": 129,
        "gemma-video-embeddings-connector": 129,
    }
    connector_rows = [
        row for row in emitted if row["builder_state_dict_key"].startswith(("audio_embeddings_connector.", "video_embeddings_connector."))
    ]
    assert len(connector_rows) == 258
    assert all(row["source_dtype"] == "BF16" for row in connector_rows)

    assert draft_policy["status"] == "draft"
    assert draft_policy["map_sha256"] == "f9f3f83c43db9b06c90d700c3c2c560d52ec7edc8e114802e5f92735a9ae90c1"
    assert draft_policy["inventory_sha256"] == inventory["inventory_sha256"]
    assert draft_policy["type_counts"] == {"BF16": 2401, "F32": 290, "Q4_K": 1658}
    assert len(draft_policy["tensors"]) == 4349
    assert approved_policy["status"] == "approved"
    assert approved_policy["map_sha256"] == "6d41db41a1b6b8f485434a64be198abf3f9a8b66b0dcd7441624644c19317760"
    assert approved_policy["inventory_sha256"] == inventory["inventory_sha256"]
    assert approved_policy["type_counts"] == {"BF16": 2401, "F32": 290, "Q4_K": 1658}
    assert len(approved_policy["tensors"]) == 4349
    assert sum(1 for row in approved_policy["tensors"] if row["ggml_type"] == "Q4_K") == 1658
    assert all(
        row["shape_logical"][-1] % 256 == 0
        for row in approved_policy["tensors"]
        if row["ggml_type"] == "Q4_K"
    )


def test_pre_bundle_4091_policy_is_rejected_after_bundle_reinspection(tmp_path):
    config = '{"transformer":{"width":256}}'
    source = tmp_path / "bundle-source.safetensors"
    _write_safetensors(
        source,
        [
            ("model.diffusion_model.transformer.weight", "BF16", _bf16(np.ones(256, np.float32)), [1, 256]),
            (
                "model.diffusion_model.audio_embeddings_connector.audio.weight",
                "BF16",
                _bf16(np.ones(256, np.float32)),
                [1, 256],
            ),
            (
                "model.diffusion_model.video_embeddings_connector.video.weight",
                "BF16",
                _bf16(np.ones(256, np.float32)),
                [1, 256],
            ),
        ],
        {"config": config},
    )
    oracle = tmp_path / "bundle-oracle.json"
    _write_component_oracle(
        oracle,
        config,
        {
            "transformer": {"transformer.weight": ([1, 256], "BF16")},
            "audio_embeddings_connector": {"audio.weight": ([1, 256], "BF16")},
            "video_embeddings_connector": {"video.weight": ([1, 256], "BF16")},
        },
    )
    inventory_path = tmp_path / "bundle-inventory.json"
    map_path = tmp_path / "pre-bundle.json"
    ltx25.inspect_ltx25(source, source_lock=_fixture_lock(source), audit_path=inventory_path, builder_oracle_path=oracle)
    policy = ltx25.build_policy_map(inventory_path, map_path)
    policy["tensors"] = [row for row in policy["tensors"] if row["name"] == "transformer.weight"]
    policy["type_counts"] = ltx25._type_counts(policy["tensors"])
    policy["status"] = "approved"
    _recompute_map(map_path, policy)
    with pytest.raises(ltx25.PolicyMapError, match="missing=2"):
        ltx25.convert_ltx25(
            source,
            map_path,
            tmp_path / "must-not-write.gguf",
            source_lock=_fixture_lock(source),
            builder_oracle_path=oracle,
        )


def test_ltx25_q4_worker_gguf_is_byte_and_structure_identical(tmp_path):
    config = '{"transformer":{"width":256}}'
    source = tmp_path / "threaded.safetensors"
    values = np.linspace(-2.0, 2.0, 2051 * 256, dtype=np.float32).reshape(2051, 256)
    _write_safetensors(
        source,
        [("model.diffusion_model.transformer.weight", "BF16", _bf16(values), [2051, 256])],
        {"config": config},
    )
    lock = _fixture_lock(source)
    oracle = tmp_path / "oracle.json"
    inventory = tmp_path / "inventory.json"
    policy = tmp_path / "map.json"
    serial = tmp_path / "serial.gguf"
    threaded = tmp_path / "threaded.gguf"
    _write_oracle(oracle, config, {"transformer.weight": [2051, 256]})
    ltx25.inspect_ltx25(source, source_lock=lock, audit_path=inventory, builder_oracle_path=oracle)
    _approved_map(inventory, policy)

    ltx25.convert_ltx25(
        source, policy, serial, source_lock=lock, builder_oracle_path=oracle, quant_workers=1
    )
    ltx25.convert_ltx25(
        source, policy, threaded, source_lock=lock, builder_oracle_path=oracle, quant_workers=4
    )
    assert serial.read_bytes() == threaded.read_bytes()
    serial_reader = GGUFReader(str(serial))
    threaded_reader = GGUFReader(str(threaded))
    assert [(tensor.name, tensor.tensor_type.name, list(tensor.shape)) for tensor in serial_reader.tensors] == [
        (tensor.name, tensor.tensor_type.name, list(tensor.shape)) for tensor in threaded_reader.tensors
    ]


def test_ltx25_thread_worker_failure_closes_writer_and_cleans_temp(tmp_path, monkeypatch):
    config = '{"transformer":{"width":256}}'
    source = tmp_path / "threaded-failure.safetensors"
    values = np.ones((2051, 256), dtype=np.float32)
    _write_safetensors(
        source,
        [("model.diffusion_model.transformer.weight", "BF16", _bf16(values), [2051, 256])],
        {"config": config},
    )
    lock = _fixture_lock(source)
    oracle = tmp_path / "oracle.json"
    inventory = tmp_path / "inventory.json"
    policy = tmp_path / "map.json"
    output = tmp_path / "output.gguf"
    _write_oracle(oracle, config, {"transformer.weight": [2051, 256]})
    ltx25.inspect_ltx25(source, source_lock=lock, audit_path=inventory, builder_oracle_path=oracle)
    _approved_map(inventory, policy)
    monkeypatch.setattr(
        qk,
        "_quantize_q4_k_blocks",
        lambda _blocks: (_ for _ in ()).throw(RuntimeError("fixture threaded quant failure")),
    )
    with pytest.raises(RuntimeError, match="fixture threaded quant failure"):
        ltx25.convert_ltx25(
            source,
            policy,
            output,
            source_lock=lock,
            builder_oracle_path=oracle,
            quant_workers=4,
        )
    assert not list(tmp_path.glob(".output.gguf.*.tmp"))


def test_policy_map_roundtrip_temp_commit_force_and_manifest_verify(tmp_path):
    source, config = _source(tmp_path)
    lock = _fixture_lock(source)
    inventory_path = tmp_path / "inventory.json"
    map_path = tmp_path / "map.json"
    oracle_path = tmp_path / "oracle.json"
    output = tmp_path / "result.gguf"
    _write_oracle(oracle_path, config, {"transformer.weight": [1, 256]})
    inventory = ltx25.inspect_ltx25(
        source, source_lock=lock, audit_path=inventory_path, builder_oracle_path=oracle_path
    )
    draft = ltx25.build_policy_map(inventory_path, map_path)
    with pytest.raises(ltx25.PolicyMapError, match="approved"):
        ltx25.convert_ltx25(source, map_path, output, source_lock=lock, builder_oracle_path=oracle_path)
    policy = _approved_map(inventory_path, map_path)

    assert policy["tensors"][0]["ggml_type"] == "Q4_K"
    assert (
        ltx25.convert_ltx25(source, map_path, output, source_lock=lock, builder_oracle_path=oracle_path)
        == output
    )
    assert ltx25.estimate_gguf_size(policy["tensors"], {"config": config, "license": "fixture"}) == output.stat().st_size
    reader = GGUFReader(str(output))
    assert reader.tensors[0].name == "transformer.weight"
    assert reader.tensors[0].tensor_type.name == "Q4_K"
    assert reader.fields["config"].contents() == config
    del reader  # GGUFReader owns a Windows file mapping until it is collected.
    manifest = ltx25.verify_ltx25(
        output,
        map_path,
        inventory_path=inventory_path,
        source_path=source,
        source_lock=lock,
        builder_oracle_path=oracle_path,
    )
    assert manifest["inventory_sha256"] == inventory.inventory_sha256

    altered_manifest = dict(manifest)
    altered_manifest["source_sha256"] = "0" * 64
    manifest_path = output.with_name(output.name + ".manifest.json")
    manifest_path.write_bytes(ltx25._canonical_json_bytes(altered_manifest) + b"\n")
    with pytest.raises(ltx25.ManifestError, match="source SHA-256"):
        ltx25.verify_ltx25(
            output,
            map_path,
            inventory_path=inventory_path,
            source_path=source,
            source_lock=lock,
            builder_oracle_path=oracle_path,
        )
    altered_manifest["source_sha256"] = manifest["source_sha256"]
    altered_manifest["inventory_sha256"] = "0" * 64
    manifest_path.write_bytes(ltx25._canonical_json_bytes(altered_manifest) + b"\n")
    with pytest.raises(ltx25.ManifestError, match="inventory SHA-256"):
        ltx25.verify_ltx25(
            output,
            map_path,
            inventory_path=inventory_path,
            source_path=source,
            source_lock=lock,
            builder_oracle_path=oracle_path,
        )
    manifest_path.write_bytes(ltx25._canonical_json_bytes(manifest) + b"\n")

    before = output.read_bytes()
    with pytest.raises(ltx25.OutputExistsError, match="--force"):
        ltx25.convert_ltx25(source, map_path, output, source_lock=lock, builder_oracle_path=oracle_path)
    assert output.read_bytes() == before
    assert (
        ltx25.convert_ltx25(
            source, map_path, output, source_lock=lock, force=True, builder_oracle_path=oracle_path
        )
        == output
    )


@pytest.mark.parametrize("ggml_type", ["Q4_K", "Q5_K", "Q6_K"])
def test_k_policy_rows_exercise_shared_k_quant_and_map_checks(tmp_path, ggml_type):
    source, config = _source(tmp_path)
    lock = _fixture_lock(source)
    inventory_path = tmp_path / "inventory.json"
    map_path = tmp_path / "map.json"
    oracle_path = tmp_path / "oracle.json"
    output = tmp_path / f"{ggml_type}.gguf"
    _write_oracle(oracle_path, config, {"transformer.weight": [1, 256]})
    ltx25.inspect_ltx25(
        source, source_lock=lock, audit_path=inventory_path, builder_oracle_path=oracle_path
    )
    policy = _approved_map(inventory_path, map_path)
    row = policy["tensors"][0]
    row.update(
        ggml_type=ggml_type,
        nbytes=ltx25._map_nbytes(row["shape_logical"], ggml_type),
        rule_id=f"fixture-{ggml_type.lower()}",
        reason="Synthetic 256-element K-quant coverage.",
        evidence_ids=["E3", "E4", "E6"],
    )
    policy["type_counts"] = {ggml_type: 1}
    policy["map_sha256"] = ltx25._sha256_bytes(
        ltx25._canonical_json_bytes({key: value for key, value in policy.items() if key != "map_sha256"})
    )
    map_path.write_bytes(ltx25._canonical_json_bytes(policy) + b"\n")

    ltx25.convert_ltx25(source, map_path, output, source_lock=lock, builder_oracle_path=oracle_path)
    reader = GGUFReader(str(output))
    assert reader.tensors[0].tensor_type.name == ggml_type
    assert np.isfinite(ltx25.dequantize(np.asarray(reader.tensors[0].data), reader.tensors[0].tensor_type)).all()


def test_split_component_rule_must_match_exactly_one_classification(tmp_path):
    source, _ = _source(tmp_path, raw_key="vae.decoder.weight")
    lock = _fixture_lock(source)
    inventory = ltx25.inspect_ltx25(
        source,
        source_lock=lock,
        exclude_prefixes={"vae.": "vae"},
        require_builder_oracle=False,
    )
    assert inventory.tensors[0]["classification"] == "exclude"
    assert inventory.tensors[0]["component_id"] == "vae"
    with pytest.raises(ltx25.SourceRejectedError, match="multiple component rules"):
        ltx25.inspect_ltx25(
            source,
            source_lock=lock,
            exclude_prefixes={"vae.": "vae", "vae.decoder.": "other"},
            require_builder_oracle=False,
        )


def test_inventory_and_k_alignment_schema_reject_tampering(tmp_path):
    source, config = _source(tmp_path)
    lock = _fixture_lock(source)
    inventory_path = tmp_path / "inventory.json"
    map_path = tmp_path / "map.json"
    oracle_path = tmp_path / "oracle.json"
    _write_oracle(oracle_path, config, {"transformer.weight": [1, 256]})
    ltx25.inspect_ltx25(source, source_lock=lock, audit_path=inventory_path, builder_oracle_path=oracle_path)
    broken_inventory = ltx25._read_json(inventory_path)
    broken_inventory["source_size"] += 1
    inventory_path.write_bytes(ltx25._canonical_json_bytes(broken_inventory) + b"\n")
    with pytest.raises(ltx25.InventoryMismatchError, match="SHA-256"):
        ltx25.load_inventory(inventory_path)

    ltx25.inspect_ltx25(source, source_lock=lock, audit_path=inventory_path, builder_oracle_path=oracle_path)
    policy = _approved_map(inventory_path, map_path)
    row = policy["tensors"][0]
    row.update(
        ggml_type="Q4_K",
        shape_logical=[256, 1],
        shape_gguf=[1, 256],
        nbytes=144,
        evidence_ids=["E3", "E4", "E6"],
    )
    policy["type_counts"] = {"Q4_K": 1}
    policy["map_sha256"] = ltx25._sha256_bytes(
        ltx25._canonical_json_bytes({key: value for key, value in policy.items() if key != "map_sha256"})
    )
    map_path.write_bytes(ltx25._canonical_json_bytes(policy) + b"\n")
    with pytest.raises(ltx25.PolicyMapError, match="last dimension"):
        ltx25.load_policy_map(map_path)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (
            lambda policy: policy["tensors"].append(dict(policy["tensors"][0])),
            "duplicate",
        ),
        (
            lambda policy: policy["tensors"].__setitem__(0, {**policy["tensors"][0], "source_dtype": "F16"}),
            "source dtype",
        ),
        (
            lambda policy: policy["tensors"].__setitem__(0, {**policy["tensors"][0], "nbytes": 1}),
            "nbytes",
        ),
    ],
    ids=["duplicate", "dtype", "nbytes"],
)
def test_policy_schema_rejects_duplicate_dtype_and_nbytes(tmp_path, mutation, expected):
    artifacts = _prepared_artifacts(tmp_path)
    policy = json.loads(json.dumps(artifacts.policy))
    mutation(policy)
    policy["type_counts"] = ltx25._type_counts(policy["tensors"])
    _recompute_map(artifacts.policy_path, policy)
    with pytest.raises(ltx25.PolicyMapError, match=expected):
        ltx25.load_policy_map(artifacts.policy_path)


@pytest.mark.parametrize("mode", ["missing", "extra", "shape"], ids=["missing", "extra", "shape"])
def test_convert_rejects_map_rows_that_do_not_match_inventory(tmp_path, mode):
    artifacts = _prepared_artifacts(tmp_path, tensor_count=2)
    policy = json.loads(json.dumps(artifacts.policy))
    if mode == "missing":
        policy["tensors"].pop()
    elif mode == "extra":
        row = dict(policy["tensors"][0])
        row["name"] = "unexpected.weight"
        policy["tensors"].append(row)
    else:
        changed_shape = [2, 256]
        policy["tensors"][0].update(
            shape_logical=changed_shape,
            shape_gguf=list(reversed(changed_shape)),
            nbytes=ltx25._map_nbytes(changed_shape, policy["tensors"][0]["ggml_type"]),
        )
    policy["type_counts"] = ltx25._type_counts(policy["tensors"])
    _recompute_map(artifacts.policy_path, policy)
    with pytest.raises(ltx25.PolicyMapError, match=mode if mode != "shape" else "dtype or shape"):
        ltx25.convert_ltx25(
            artifacts.source,
            artifacts.policy_path,
            artifacts.output,
            source_lock=artifacts.lock,
            builder_oracle_path=artifacts.oracle,
        )


def test_convert_disk_preflight_happens_before_temp_output(tmp_path, monkeypatch):
    artifacts = _prepared_artifacts(tmp_path)
    monkeypatch.setattr(ltx25.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    with pytest.raises(ltx25.DiskPreflightError, match="disk-preflight-failed"):
        ltx25.convert_ltx25(
            artifacts.source,
            artifacts.policy_path,
            artifacts.output,
            source_lock=artifacts.lock,
            builder_oracle_path=artifacts.oracle,
        )
    assert not artifacts.output.exists()
    assert not list(tmp_path.glob(".output.gguf.*.tmp"))


def test_convert_verifies_temp_before_replacing_existing_output(tmp_path, monkeypatch):
    artifacts = _prepared_artifacts(tmp_path)
    artifacts.output.write_bytes(b"existing-output")
    monkeypatch.setattr(
        ltx25,
        "_verify_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ltx25.Ltx25Error("fixture verify failure")),
    )
    with pytest.raises(ltx25.Ltx25Error, match="fixture verify failure"):
        ltx25.convert_ltx25(
            artifacts.source,
            artifacts.policy_path,
            artifacts.output,
            source_lock=artifacts.lock,
            builder_oracle_path=artifacts.oracle,
            force=True,
        )
    assert artifacts.output.read_bytes() == b"existing-output"


def test_manifest_write_failure_is_visible_and_force_recovery_restores_sidecar(tmp_path, monkeypatch):
    artifacts = _prepared_artifacts(tmp_path)
    original_write_json = ltx25._write_json_atomic

    def fail_manifest(_path, _payload):
        raise ltx25.ManifestError("fixture manifest failure")

    monkeypatch.setattr(ltx25, "_write_json_atomic", fail_manifest)
    with pytest.raises(ltx25.ManifestError, match="committed GGUF"):
        ltx25.convert_ltx25(
            artifacts.source,
            artifacts.policy_path,
            artifacts.output,
            source_lock=artifacts.lock,
            builder_oracle_path=artifacts.oracle,
        )
    assert artifacts.output.is_file()
    assert not Path(f"{artifacts.output}.manifest.json").exists()
    monkeypatch.setattr(ltx25, "_write_json_atomic", original_write_json)
    ltx25.convert_ltx25(
        artifacts.source,
        artifacts.policy_path,
        artifacts.output,
        source_lock=artifacts.lock,
        builder_oracle_path=artifacts.oracle,
        force=True,
    )
    assert ltx25.verify_ltx25(
        artifacts.output,
        artifacts.policy_path,
        inventory_path=artifacts.inventory,
        source_path=artifacts.source,
        source_lock=artifacts.lock,
        builder_oracle_path=artifacts.oracle,
    )["profile"] == "ltx25"


def test_replace_permission_retry_and_retained_temp_report(tmp_path, monkeypatch):
    artifacts = _prepared_artifacts(tmp_path / "retained")
    temp = tmp_path / "temp.gguf"
    final = tmp_path / "final.gguf"
    temp.write_bytes(b"new")
    final.write_bytes(b"old")
    original_replace = ltx25.os.replace
    attempts = {"count": 0}

    def fail_once(source, destination):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError("locked")
        return original_replace(source, destination)

    monkeypatch.setattr(ltx25.os, "replace", fail_once)
    ltx25._replace_with_retry(temp, final)
    assert attempts["count"] == 2
    assert final.read_bytes() == b"new"

    monkeypatch.setattr(ltx25.os, "replace", lambda *_args: (_ for _ in ()).throw(PermissionError("locked")))
    monkeypatch.setattr(ltx25, "_cleanup_temp", lambda path: str(path))
    with pytest.raises(ltx25.Ltx25Error, match="temporary file retained"):
        ltx25.convert_ltx25(
            artifacts.source,
            artifacts.policy_path,
            artifacts.output,
            source_lock=artifacts.lock,
            builder_oracle_path=artifacts.oracle,
        )


def test_ltx25_transformer_profile_never_uses_i8():
    """I8 (added for the separate gemma4-ltx25 profile's U8 sidecars) must not
    change the ltx25 transformer profile's own allowed target types."""
    assert "I8" not in ltx25._ALLOWED_TARGET_TYPES
