"""Fixture coverage for the opt-in, fail-closed gemma4-ltx25 conversion profile.

Mirrors ``tests/test_ltx25.py``'s fixture style. Three groups of tests:

1. Pure derivation/classification logic (no file I/O): the E2 arithmetic
   derivation from a synthetic ``gemma_config.text_config``, and the ordered
   E3/E4 classification rule table's mutual exclusivity (including the three
   "trap" tensors that would multi-match a naive Q4_K predicate).
2. Real-header cross-checks, skipped when the ~26 GB pinned source file is not
   present locally (mirrors ``test_quant_roundtrip.py``'s ``REF_GGUF`` skip
   convention): the derivation and classification exactly reproduce the
   authenticated 686-tensor header and its {Q4_K:328, Q6_K:2, BF16:351, I8:5}
   type breakdown.
3. A small synthetic end-to-end fixture: inspect -> build-map -> approve ->
   convert -> self-verify, covering every classification rule at least once
   (including one I8 sidecar and one Q6_K aggregate row) on tiny tensors.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from converter import cli
from converter import ltx25
from converter import ltx25_gemma as gemma
from converter import quant_kernels as qk


GEMMA4_SOURCE = (
    Path(__file__).resolve().parents[1]
    / ".artifacts"
    / "official-ltx25"
    / "text_encoders"
    / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
)


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
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    if metadata is not None:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def _fixture_lock(path) -> ltx25.SourceLock:
    digest = ltx25.sha256_of_file(path)
    return ltx25.SourceLock(
        repo_id="fixture/gemma4-ltx25",
        artifact_revision="fixture-revision",
        filename="fixture-gemma.safetensors",
        expected_size=path.stat().st_size,
        source_sha256=digest,
        lfs_sha256=digest,
        git_blob_oid="",
    )


def _small_text_config() -> dict:
    return {
        "hidden_size": 256,
        "intermediate_size": 512,
        "head_dim": 256,
        "global_head_dim": 256,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "num_global_key_value_heads": 1,
        "attention_k_eq_v": True,
        "layer_types": ["sliding_attention", "full_attention"],
        "num_hidden_layers": 2,
        "vocab_size": 256,
    }


def _small_gemma_config_text() -> str:
    config = {
        "text_config": _small_text_config(),
        "vision_config": {"mm_embed_dim": 256, "mm_posemb_size": 2, "output_proj_dims": 256},
        "audio_config": {"audio_embed_dim": 256},
        "gemma_version": gemma.EXPECTED_GEMMA_VERSION,
    }
    return json.dumps(config)


def _write_gemma_oracle(path, config_text: str, tensors: dict) -> None:
    payload = {
        "format": gemma.ORACLE_FORMAT,
        "profile": gemma.PROFILE_ID,
        "official_code_commit": gemma.OFFICIAL_CODE_COMMIT,
        "config_bytes_sha256": ltx25._sha256_bytes(config_text.encode("utf-8")),
        "tensors": {name: {"dtype": dtype, "shape": list(shape)} for name, (dtype, shape) in tensors.items()},
    }
    payload["oracle_sha256"] = ltx25._sha256_bytes(ltx25._canonical_json_bytes(payload))
    path.write_bytes(ltx25._canonical_json_bytes(payload) + b"\n")


def _approved_gemma_map(inventory_path, map_path) -> dict:
    policy = gemma.build_policy_map(inventory_path, map_path)
    assert policy["status"] == "draft"
    policy["status"] = "approved"  # Simulates separate human review of checked-in E4.
    policy["map_sha256"] = ltx25._sha256_bytes(
        ltx25._canonical_json_bytes({key: value for key, value in policy.items() if key != "map_sha256"})
    )
    map_path.write_bytes(ltx25._canonical_json_bytes(policy) + b"\n")
    return policy


# A minimal fixture covering every classification rule at least once:
# Q4_K backbone, plain BF16 fallback (no-rmsnorm-folding), Q6_K aggregate,
# embed_tokens-bf16, audio-projector-bf16, multimodal-projector-bf16 (a Q4_K
# trap: shape [256, 256] is K-alignment-eligible), vision-tower-bf16 (also a
# trap: [256, 512]), and one I8 sidecar.
def _fixture_rows() -> list[tuple[str, str, bytes, list[int]]]:
    return [
        ("model.layers.0.self_attn.q_proj.weight", "BF16", _bf16(np.linspace(-1, 1, 256 * 256, dtype=np.float32)), [256, 256]),
        ("model.layers.0.input_layernorm.weight", "BF16", _bf16(np.ones(256, dtype=np.float32)), [256]),
        (
            "text_embedding_projection.video_aggregate_embed.weight",
            "BF16",
            _bf16(np.linspace(-1, 1, 4096 * 256, dtype=np.float32)),
            [4096, 256],
        ),
        (
            "text_embedding_projection.video_aggregate_embed.bias",
            "BF16",
            _bf16(np.zeros(4096, dtype=np.float32)),
            [4096],
        ),
        ("model.embed_tokens.weight", "BF16", _bf16(np.linspace(-1, 1, 256 * 256, dtype=np.float32)), [256, 256]),
        ("audio_projector.embedding_projection.weight", "BF16", _bf16(np.ones(256 * 100, dtype=np.float32)), [256, 100]),
        (
            "multi_modal_projector.embedding_projection.weight",
            "BF16",
            _bf16(np.ones(256 * 256, dtype=np.float32)),
            [256, 256],
        ),
        ("vision_model.patch_dense.weight", "BF16", _bf16(np.ones(256 * 512, dtype=np.float32)), [256, 512]),
        ("hf_asset__tokenizer_config.json", "U8", b'{"ok":true}', [11]),
    ]


def _prepared_gemma_artifacts(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_text = _small_gemma_config_text()
    rows = _fixture_rows()
    source = tmp_path / "fixture-gemma.safetensors"
    _write_safetensors(
        source,
        [(name, dtype, raw, shape) for name, dtype, raw, shape in rows],
        {"gemma_config": config_text, "format": "pt"},
    )
    lock = _fixture_lock(source)
    inventory = tmp_path / "inventory.json"
    oracle = tmp_path / "oracle.json"
    policy = tmp_path / "map.json"
    output = tmp_path / "output.gguf"
    _write_gemma_oracle(oracle, config_text, {name: (dtype, shape) for name, dtype, _raw, shape in rows})
    gemma.inspect_gemma4(source, source_lock=lock, audit_path=inventory, builder_oracle_path=oracle)
    approved = _approved_gemma_map(inventory, policy)
    return {
        "source": source,
        "config_text": config_text,
        "lock": lock,
        "inventory": inventory,
        "oracle": oracle,
        "policy_path": policy,
        "policy": approved,
        "output": output,
    }


# ---------------------------------------------------------------------------
# 1. pure derivation / classification logic
# ---------------------------------------------------------------------------
def test_derive_backbone_entries_formulas_on_small_config():
    entries = gemma.derive_backbone_entries(_small_text_config())
    # layer 0 (sliding): v_proj present; layer 1 (full): v_proj absent (k=v shared).
    assert entries["model.layers.0.self_attn.v_proj.weight"] == ("BF16", [256, 256])
    assert "model.layers.1.self_attn.v_proj.weight" not in entries
    assert entries["model.layers.0.self_attn.q_proj.weight"] == ("BF16", [256, 256])
    assert entries["model.layers.0.mlp.down_proj.weight"] == ("BF16", [256, 512])
    assert entries["model.layers.0.mlp.gate_proj.weight"] == ("BF16", [512, 256])
    assert entries["model.layers.0.layer_scalar"] == ("BF16", [1])
    # 2 layers: 14 rows (sliding, has v_proj) + 13 rows (full, no v_proj) = 27.
    assert len(entries) == 27


def test_derive_backbone_entries_requires_attention_k_eq_v():
    bad = dict(_small_text_config(), attention_k_eq_v=False)
    with pytest.raises(gemma.Gemma4InventoryDerivationError, match="attention_k_eq_v"):
        gemma.derive_backbone_entries(bad)


def test_derive_aggregate_entries_flat_dim_and_fixed_output_widths():
    entries = gemma.derive_aggregate_entries(_small_text_config())
    # flat_dim = hidden_size * (num_hidden_layers + 1) = 256 * 3 = 768.
    assert entries["text_embedding_projection.video_aggregate_embed.weight"] == ("BF16", [4096, 768])
    assert entries["text_embedding_projection.audio_aggregate_embed.weight"] == ("BF16", [2048, 768])
    assert entries["text_embedding_projection.video_aggregate_embed.bias"] == ("BF16", [4096])
    assert entries["text_embedding_projection.audio_aggregate_embed.bias"] == ("BF16", [2048])


def test_derive_fixed_entries_and_oracle_payload_roundtrip():
    text_config = _small_text_config()
    vision_config = {"mm_embed_dim": 256, "mm_posemb_size": 2}
    audio_config = {"audio_embed_dim": 256}
    entries = gemma.derive_fixed_entries(text_config, vision_config, audio_config)
    assert entries["model.embed_tokens.weight"] == ("BF16", [256, 256])
    assert entries["model.norm.weight"] == ("BF16", [256])
    assert entries["audio_projector.embedding_projection.weight"] == ("BF16", [256, 256])
    assert entries["vision_model.pos_embedding"] == ("BF16", [2, 2, 256])
    for name, size in gemma._SIDECAR_U8_SIZES.items():
        assert entries[name] == ("U8", [size])

    config_text = _small_gemma_config_text()
    payload = gemma.build_builder_oracle_payload(config_text)
    assert payload["format"] == gemma.ORACLE_FORMAT
    assert len(payload["tensors"]) == 27 + 4 + 13 + 5  # backbone + aggregate(w+b) + fixed(non-sidecar) + sidecar
    assert payload["oracle_sha256"] == ltx25._sha256_bytes(
        ltx25._canonical_json_bytes({k: v for k, v in payload.items() if k != "oracle_sha256"})
    )


def test_classify_ordered_rules_resolve_naive_q4k_traps():
    """The three rows a naive '2-D .weight, dim%256==0' predicate would wrongly Q4_K."""
    traps = {
        "vision_model.patch_dense.weight": [3840, 6912],
        "multi_modal_projector.embedding_projection.weight": [3840, 3840],
        "model.embed_tokens.weight": [262144, 3840],
    }
    for name, shape in traps.items():
        assert len(shape) == 2 and shape[-1] % 256 == 0  # the naive predicate DOES match
        ggml_type, _rule_id, _reason = gemma.classify_gemma4_tensor(name, shape)
        assert ggml_type != "Q4_K", f"{name} must be resolved by an earlier, more specific rule"
    ggml_type, rule_id, _ = gemma.classify_gemma4_tensor("model.layers.0.self_attn.q_proj.weight", [4096, 3840])
    assert (ggml_type, rule_id) == ("Q4_K", "q4-k-backbone-linear")


def test_classify_every_rule_id_reachable_and_dtype_consistent():
    cases = [
        ("hf_asset__tokenizer_config.json", [10], "I8", "i8-sidecar-passthrough"),
        ("vision_model.pos_norm.weight", [256], "BF16", "vision-tower-bf16"),
        ("multi_modal_projector.embedding_projection.weight", [256, 256], "BF16", "multimodal-projector-bf16"),
        ("audio_projector.embedding_projection.weight", [256, 100], "BF16", "audio-projector-bf16"),
        ("text_embedding_projection.audio_aggregate_embed.weight", [2048, 256], "Q6_K", "aggregate-embed-q6k"),
        ("model.embed_tokens.weight", [256, 256], "BF16", "embed-tokens-bf16-backend-expansion"),
        ("model.layers.3.mlp.down_proj.weight", [256, 512], "Q4_K", "q4-k-backbone-linear"),
        ("model.layers.3.post_attention_layernorm.weight", [256], "BF16", "no-rmsnorm-folding"),
    ]
    for name, shape, expected_type, expected_rule in cases:
        ggml_type, rule_id, reason = gemma.classify_gemma4_tensor(name, shape)
        assert (ggml_type, rule_id) == (expected_type, expected_rule)
        assert reason


# ---------------------------------------------------------------------------
# 2. real-header cross-checks (skipped without the pinned local source file)
# ---------------------------------------------------------------------------
_requires_real_source = pytest.mark.skipif(
    not GEMMA4_SOURCE.is_file(), reason="pinned gemma4-ltx25 source safetensors not present locally"
)


def _read_real_header() -> tuple[dict, str]:
    with GEMMA4_SOURCE.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
    metadata = header.pop("__metadata__")
    return header, metadata["gemma_config"]


@_requires_real_source
def test_real_header_derivation_matches_exactly_686_tensors():
    header, config_text = _read_real_header()
    table = gemma.derive_expected_tensor_table(json.loads(config_text))
    assert len(table) == gemma.EXPECTED_TOTAL == 686
    real_names = set(header)
    derived_names = set(table)
    assert real_names == derived_names
    for name in real_names:
        dtype, shape = table[name]
        assert dtype == header[name]["dtype"]
        assert list(shape) == list(header[name]["shape"])


@_requires_real_source
def test_real_header_classification_type_counts_match_expected():
    header, _config_text = _read_real_header()
    counts: dict[str, int] = {}
    for name, info in header.items():
        ggml_type, _rule_id, _reason = gemma.classify_gemma4_tensor(name, list(info["shape"]))
        counts[ggml_type] = counts.get(ggml_type, 0) + 1
        expected_source_dtype = "U8" if ggml_type == "I8" else "BF16"
        assert info["dtype"] == expected_source_dtype
    assert counts == gemma.EXPECTED_TYPE_COUNTS


# ---------------------------------------------------------------------------
# 3. small synthetic end-to-end fixture
# ---------------------------------------------------------------------------
def test_inspect_rejects_source_lock_incomplete_before_admission(tmp_path):
    source = tmp_path / "x.safetensors"
    _write_safetensors(
        source,
        [("a", "BF16", _bf16(np.ones(1, dtype=np.float32)), [1])],
        {"gemma_config": _small_gemma_config_text()},
    )
    incomplete_lock = ltx25.SourceLock(
        repo_id="x", artifact_revision="x", filename="x", expected_size=None,
        source_sha256=ltx25.SOURCE_LOCK_UNCONFIRMED, lfs_sha256=ltx25.SOURCE_LOCK_UNCONFIRMED, git_blob_oid="",
    )
    with pytest.raises(gemma.Gemma4SourceLockIncompleteError, match="source-lock-incomplete"):
        gemma.inspect_gemma4(source, source_lock=incomplete_lock)


def test_inspect_rejects_wrong_gemma_version(tmp_path):
    bad_config = json.loads(_small_gemma_config_text())
    bad_config["gemma_version"] = "gemma3-something-else"
    source = tmp_path / "x.safetensors"
    _write_safetensors(
        source,
        [("a", "BF16", _bf16(np.ones(1, dtype=np.float32)), [1])],
        {"gemma_config": json.dumps(bad_config)},
    )
    with pytest.raises(gemma.Gemma4SourceRejectedError, match="gemma_version"):
        gemma.inspect_gemma4(source, source_lock=_fixture_lock(source), require_builder_oracle=False)


def test_inspect_rejects_dtype_role_mismatch(tmp_path):
    """A sidecar name with BF16 dtype (or vice versa) is rejected, not silently reclassified."""
    source = tmp_path / "x.safetensors"
    _write_safetensors(
        source,
        [("hf_asset__tokenizer_config.json", "BF16", _bf16(np.ones(1, dtype=np.float32)), [1])],
        {"gemma_config": _small_gemma_config_text()},
    )
    with pytest.raises(gemma.Gemma4SourceRejectedError, match="expected 'U8'"):
        gemma.inspect_gemma4(source, source_lock=_fixture_lock(source), require_builder_oracle=False)


def test_inspect_build_map_convert_self_verify_small_fixture(tmp_path):
    artifacts = _prepared_gemma_artifacts(tmp_path)
    policy = artifacts["policy"]
    assert policy["type_counts"] == {
        "BF16": 6,
        "I8": 1,
        "Q4_K": 1,
        "Q6_K": 1,
    }
    result = gemma.convert_gemma4(
        artifacts["source"],
        artifacts["policy_path"],
        artifacts["output"],
        source_lock=artifacts["lock"],
        builder_oracle_path=artifacts["oracle"],
        quant_workers=2,
    )
    assert result == artifacts["output"]
    manifest_path = Path(f"{artifacts['output']}.manifest.json")
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["type_counts"] == policy["type_counts"]
    assert manifest["tensor_count"] == 9

    verify_manifest = gemma.verify_gemma4(
        artifacts["output"],
        artifacts["policy_path"],
        inventory_path=artifacts["inventory"],
        source_path=artifacts["source"],
        source_lock=artifacts["lock"],
        builder_oracle_path=artifacts["oracle"],
    )
    assert verify_manifest["output_sha256"] == manifest["output_sha256"]


def test_convert_rejects_draft_map(tmp_path):
    artifacts = _prepared_gemma_artifacts(tmp_path)
    draft_path = tmp_path / "draft.json"
    gemma.build_policy_map(artifacts["inventory"], draft_path)
    with pytest.raises(gemma.Gemma4PolicyMapError, match="approved"):
        gemma.convert_gemma4(
            artifacts["source"],
            draft_path,
            artifacts["output"],
            source_lock=artifacts["lock"],
            builder_oracle_path=artifacts["oracle"],
        )


def test_build_policy_map_rejects_inventory_oracle_dtype_role_mismatch(tmp_path):
    """A hand-tampered inventory row whose dtype contradicts its rule is rejected."""
    artifacts = _prepared_gemma_artifacts(tmp_path)
    inventory_payload = json.loads(Path(artifacts["inventory"]).read_text(encoding="utf-8"))
    for record in inventory_payload["tensors"]:
        if record["raw_key"] == "hf_asset__tokenizer_config.json":
            record["source_dtype"] = "BF16"  # would classify I8 but claims BF16 source
    inventory_payload["inventory_sha256"] = ltx25._sha256_bytes(
        ltx25._canonical_json_bytes({k: v for k, v in inventory_payload.items() if k != "inventory_sha256"})
    )
    tampered = tmp_path / "tampered_inventory.json"
    tampered.write_bytes(ltx25._canonical_json_bytes(inventory_payload) + b"\n")
    with pytest.raises(gemma.Gemma4InventoryMismatchError, match="classified I8"):
        gemma.build_policy_map(tampered, tmp_path / "tampered_map.json")


# ---------------------------------------------------------------------------
# quant_kernels: Q6_K bounded split (byte-identical + worker-range coverage
# live in tests/test_quant_roundtrip.py next to the Q4_K equivalents).
# ---------------------------------------------------------------------------
def test_q6_k_bounded_split_peak_memory_does_not_scale_linearly():
    """Splitting must bound peak memory: an 8x larger input should not need ~8x peak RAM."""
    import tracemalloc

    rng = np.random.default_rng(20260821)
    small = rng.normal(size=(8 * qk.Q6_K_BLOCKS_PER_TASK, 256)).astype(np.float32)
    large = rng.normal(size=(64 * qk.Q6_K_BLOCKS_PER_TASK, 256)).astype(np.float32)

    def peak_bytes(arr: np.ndarray) -> int:
        tracemalloc.start()
        try:
            qk.quantize_q6_k(arr, workers=1)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak

    peak_small = peak_bytes(small)
    peak_large = peak_bytes(large)
    # A linear (unsplit) implementation would show ~8x growth; bounded batching
    # keeps peak close to one Q6_K_BLOCKS_PER_TASK-sized working set regardless
    # of total tensor size. Assert well under half of the naive-linear bound.
    assert peak_large < peak_small * 4, (peak_small, peak_large)


def test_quantize_dispatch_routes_q6_k_through_worker_pool(monkeypatch):
    calls = []
    real = qk.quantize_q6_k

    def spy(arr, *, workers=1, executor=None):
        calls.append((workers, executor))
        return real(arr, workers=workers, executor=executor)

    monkeypatch.setattr(qk, "quantize_q6_k", spy)
    values = np.zeros((256,), dtype=np.float32)
    qk.quantize(values, "Q6_K", q4_workers=4, q4_executor=None)
    assert calls == [(4, None)]


# ---------------------------------------------------------------------------
# cli wiring
# ---------------------------------------------------------------------------
def test_cli_quant_worker_gemma_profile_defaults_are_frozen():
    assert (
        cli._quant_worker_count(
            cli.argparse.Namespace(model="gemma4-ltx25", quant_workers=None),
            {"profiles": {"gemma4-ltx25": {"quant_workers": 4}}},
        )
        == 4
    )
    # Missing profile table must still default to 4, never silently fall back to 1
    # (the exact regression the plan calls out for _quant_worker_count).
    assert cli._quant_worker_count(cli.argparse.Namespace(model="gemma4-ltx25", quant_workers=None), {}) == 4
    assert cli._quant_worker_count(cli.argparse.Namespace(model="ltx23", quant_workers=None), {}) == 1


def test_cli_model_choices_include_gemma4_ltx25():
    parser = cli.build_parser()
    assert parser.parse_args(["inspect", "--model", "gemma4-ltx25"]).model == "gemma4-ltx25"
    with pytest.raises(SystemExit):
        parser.parse_args(["inspect", "--model", "not-a-real-profile"])


def test_cli_extract_typemap_and_all_reject_gemma4_ltx25(capsys):
    config = {"source": {}, "reference": {}, "output": {}}
    parser = cli.build_parser()
    assert cli.cmd_extract_typemap(parser.parse_args(["extract-typemap", "--model", "gemma4-ltx25"]), config) == 1
    assert "ltx23-only" in capsys.readouterr().err
    assert cli.cmd_all(parser.parse_args(["all", "--model", "gemma4-ltx25"]), config) == 1
    assert "ltx23-only" in capsys.readouterr().err


def test_cli_inspect_and_build_map_gate_on_allowed_models(capsys):
    parser = cli.build_parser()
    assert cli.cmd_inspect(parser.parse_args(["inspect", "--model", "ltx23"]), {}) == 1
    assert "ltx25/gemma4-ltx25" in capsys.readouterr().err
    assert cli.cmd_build_map(parser.parse_args(["build-map", "--model", "ltx23"]), {}) == 1
    assert "ltx25/gemma4-ltx25" in capsys.readouterr().err


def test_cli_gemma_source_lock_incomplete_gates_before_traceback(capsys, monkeypatch):
    config = {"profiles": {"gemma4-ltx25": {"source_sha256": ltx25.SOURCE_LOCK_UNCONFIRMED}}}
    monkeypatch.setattr(cli, "_load_config", lambda: config)
    for argv in (
        ["inspect", "--model", "gemma4-ltx25", "--st-path", "does-not-exist"],
        ["build-map", "--model", "gemma4-ltx25", "--inventory", "does-not-exist"],
        ["convert", "--model", "gemma4-ltx25", "--st-path", "does-not-exist", "--map", "does-not-exist"],
        ["self-verify", "--model", "gemma4-ltx25", "--out", "does-not-exist", "--map", "does-not-exist"],
    ):
        assert cli.main(argv) == 1
        err = capsys.readouterr().err
        assert "source-lock-incomplete" in err
        assert "Traceback" not in err
