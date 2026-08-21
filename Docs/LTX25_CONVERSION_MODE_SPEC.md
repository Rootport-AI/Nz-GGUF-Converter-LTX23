# LTX 2.5 conversion mode specification

## Status, compatibility, and enablement gate

This document is the implementation contract for exactly three frozen profiles: legacy ltx23, opt-in ltx25 (the LTX 2.5 transformer), and opt-in gemma4-ltx25 (the LTX-fine-tuned Gemma 4 text encoder bundled in a separate source file). The LTX 2.3 default CLI, config, reference-typemap workflow, output naming, and artifact behavior remain backward compatible. gemma4-ltx25 is documented in its own section below; every clause elsewhere in this document that refers to "the ltx25 profile" or "the LTX 2.5 transformer" describes ltx25 specifically and does not extend to gemma4-ltx25 unless the gemma4-ltx25 section says so.

LTX 2.5 conversion accepts only an independently approved E4 map. E1-E3 are complete: a selected ltx25 command admits only the authenticated pinned source, uses the compatible official builder key/shape/component oracle, and writes/verifies the canonical 4,349-row bundle inventory; it refuses a draft map for conversion. The replacement 4,349-row E4 is independently approved. The earlier 4,091-row map and its E5/E6 output are intentionally stale after the bundle migration and cannot be reused. Fresh E5/E6 are complete for both opt-in profiles as of 2026-08-21; the recorded output digests are in the Open gates section. Passing them is necessary but not sufficient for publication or backend use, which remain separately gated.

The sole initial LTX 2.5 source is:

    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors

Its 4,349 emitted tensors must match the authenticated E3 safetensors-header per-key source-dtype table: 3,801 transformer BF16, 290 transformer F32, and 258 connector BF16. F16, INT8 ConvRot, FP8, NVFP4, or any other dtype that differs from the exact E3 row is rejected. The converter must not dequantize, reinterpret, or silently accept a prequantized source.

## Evidence register and primary sources

The six evidence IDs avoid duplicate reports. Each artifact is canonical JSON with deterministic field ordering and a SHA-256.

| ID | Artifact and purpose | Gate |
| --- | --- | --- |
| E1 | Source lock: official repository, artifact revision/path/size, license, and verified source content SHA-256. | Complete: authenticated Hub LFS SHA-256 and local full-file SHA-256 agree. |
| E2 | Official-code oracle: pinned code revision, direct meta-builder recipe, and fixed component key/shape contract. It has no module-role table and is not a dtype oracle. | Complete: 4,091 transformer, 129 audio connector, 129 video connector rows exactly match official meta builders. |
| E3 | Canonical safetensors header inventory: metadata bytes/digest, raw keys, normalized output keys, component roles, dtypes, shapes, and offsets. | Complete: 4,349 emitted rows; SHA-256 `c0966ac37f55f318be16334c1f1e1c2db1f467dafde09c0c2bd1b853206bde97`. |
| E4 | Concrete checked-in conversion map/policy, generated from E2 plus E3. | Independently approved: SHA-256 `6d41db41a1b6b8f485434a64be198abf3f9a8b66b0dcd7441624644c19317760`. The former 4,091-row map is stale and rejected by E3 hash and 258 missing rows. |
| E5 | Converted GGUF static verification and output digest. | Pending a fresh 4,349-row conversion. |
| E6 | K-quant numerical evidence for the concrete E4 types. | Pending the fresh E5 artifact. |

Primary sources, all immutable where a revision is supplied:

- [Hugging Face model](https://huggingface.co/Lightricks/LTX-2.5), [artifact tree at release dd53cc2cd45bbeaa3563dfb575cba3f49cf44761](https://huggingface.co/Lightricks/LTX-2.5/tree/dd53cc2cd45bbeaa3563dfb575cba3f49cf44761), [model API at that revision](https://huggingface.co/api/models/Lightricks/LTX-2.5/revision/dd53cc2cd45bbeaa3563dfb575cba3f49cf44761), and [official LICENSE](https://huggingface.co/Lightricks/LTX-2.5/blob/dd53cc2cd45bbeaa3563dfb575cba3f49cf44761/LICENSE).
- [Official LTX code oracle root at 400fd31054597515f47125691032c04b1c3ee24e](https://github.com/Lightricks/LTX-2/tree/400fd31054597515f47125691032c04b1c3ee24e). A local pinned checkout of this exact commit is also available offline at `.artifacts/official-ltx25/official-code-400fd310/` (repository root), so every file-level link below can be read locally instead of fetched from GitHub.
- [model_loader.py](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-core/src/ltx_core/loader/model_loader.py), [single_gpu_model_builder.py](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py), [sft_loader.py](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-core/src/ltx_core/loader/sft_loader.py), [helpers.py](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-core/src/ltx_core/loader/helpers.py).
- [transformer model_configurator.py](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py), [transformer model.py](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-core/src/ltx_core/model/transformer/model.py), and [sd_ops.py](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-core/src/ltx_core/loader/sd_ops.py).

The LTX code pin is E2, not the artifact pin. Current repository SHA 6c7e5e573ac1667efc83407806fe9b0b93730e60 is reference information only and is not a substitute. The implementation must confirm that E2 is appropriate for E1 before it creates E3/E4.

## E1 source lock

The following values are known and must be copied verbatim into the source-lock record.

| Field | Locked value |
| --- | --- |
| repository | Lightricks/LTX-2.5 |
| artifact release revision | dd53cc2cd45bbeaa3563dfb575cba3f49cf44761 |
| current repository SHA, reference only | 6c7e5e573ac1667efc83407806fe9b0b93730e60 |
| relative artifact path | diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors |
| exact byte size | 42,018,190,584 |
| Git blob OID | 3ba48d13c75fd5df2e868aa8c07b0c5119a6116f |
| gating | auto; authenticated access and accepted terms required |
| license | ltx-2-community-license-agreement; official LICENSE link above |
| LFS SHA-256/Xet content hash | 31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4 |
| canonical header inventory | `typemap/ltx25_inventory.json`, SHA-256 `c0966ac37f55f318be16334c1f1e1c2db1f467dafde09c0c2bd1b853206bde97` |

The Git blob OID is not a SHA-256 content digest. VirusTotal URL values are not a source hash and must never be copied to the lock as one. E1 was completed with authenticated official download, byte-size check, full local SHA-256, and matching Hub LFS/Xet content hash. The approved replacement E4 permits a fresh conversion; E5/E6 remain incomplete.

## Audited current repository and backend constraints

| Area | Existing contract | Evidence |
| --- | --- | --- |
| Streaming reader | The safetensors reader holds a small header and reads one payload at a time. | src/converter/convert.py:82-97, 137-164 |
| Source reader | BF16/F16/F32 only today; unsupported dtype raises. | src/converter/convert.py:143-164, 166-186 |
| LTX 2.3 selection | model.diffusion_model emits; VAE/audio VAE/vocoder/text projection skip; others fail. | src/converter/convert.py:65-70, 211-266 |
| Exact set check | Duplicate typemap names, missing, extra, unexpected keys fail. | src/converter/convert.py:234-266 |
| Writer | F32, BF16, Q4_K, Q5_K, Q6_K are the implemented target types. gemma4-ltx25 (below) added I8 as a sixth: a block-size-1 byte-identical passthrough type, used only for its five non-tensor U8 sidecar payloads (gguf-py has no U8 writer type but does have I8, which is bit-for-bit the same storage). | src/converter/convert.py:270-335 |
| Metadata | Config JSON is required; current writer emits project GGUF metadata. | src/converter/metadata.py:32-48, 65-133 |
| LTX 2.3 map | It is reference-GGUF-derived: 4,444 tensors and a 48-block-specific policy. | src/converter/typemap.py:15-18, 31-38, 121-161 |
| Existing CLI | Global config defaults drive download/convert/verify/all. | src/converter/cli.py:139-170, 265-290, 380-527 |
| Existing tests | Small hand-authored BF16 fixtures cover conversion, metadata, key safety, and K numerics. | tests/test_convert.py:1-13, 147-259; tests/test_metadata.py:54-154; tests/test_quant_roundtrip.py:180-328 |

The adjacent backend is a static consumer handoff only. It is not the LTX 2.5 oracle and no backend dependency belongs in converter CI.

| Backend observation | Evidence |
| --- | --- |
| Per-layer mode stores quantized raw bytes and dequantizes each Linear weight during forward. | ../Nz-LTX23-backend/engine/gguf/quant_service.py:729-734 |
| It reverses GGUF shape; F32/F16/BF16 are floats and other supported types remain packed bytes. | ../Nz-LTX23-backend/engine/gguf/quant_service.py:582-615 |
| Active per-layer load applies no remap, so emitted raw names must be exact. | ../Nz-LTX23-backend/engine/gguf/quant_service.py:617-621 |
| It reads embedded config but its current construction capability is not evidence of LTX 2.5 support. | ../Nz-LTX23-backend/engine/gguf/quant_service.py:536-561 |

## Profile, CLI, and scope boundary

The profile implementation is a frozen two-entry data table plus explicit functions or if statements. Do not use a class hierarchy, generic DSL/registry, plugin system, dynamic import, or automatic profile detection.

All profile-aware commands default to ltx23. Omission of model must preserve the current legacy source/reference/output resolution exactly. An unknown profile is an argparse error.

The legacy all and extract-typemap commands are ltx23-only and keep their present behavior. LTX 2.5 has no all shortcut. Once a separate review has checked in an approved E4 map, its normal user sequence is:

    inspect --model ltx25
    convert --model ltx25
    self-verify --model ltx25

The ltx25 sequence is blocked at inspect until E1 is complete. The ltx25 map is never extracted from a reference or third-party GGUF. `build-map`/`build-policy` is a reviewer-only evidence regeneration command: it writes only `typemap/ltx25_conversion_map.draft.json` and rejects the configured approved path.

`build-map` writes draft-only data and has no `--status approved` escape hatch. A separately reviewed checked-in E4 artifact is the only input accepted by `convert`.

The only profile-owned behavior is source admission, component classification, one normalization function, official inventory oracle, concrete map lookup, metadata validation, output/manifest settings, and static verification. Reader/writer/streaming/K kernels remain shared.

For E5, Q4_K divides each tensor into input-ordered 1,024-block tasks. The
unchanged float32/reduction/packing kernel runs once per independent block
batch, then batches are concatenated in source order. LTX 2.3 defaults to one
bounded serial worker; the ltx25 profile defaults to four and accepts at most
eight through `--quant-workers`. One executor spans a conversion, avoiding
nested pools and unbounded per-tensor workspaces. Output must be byte-identical
for one, two, four, or eight workers.

Gemma language-model weights, text projection, VAE, audio VAE, vocoder, upscalers, and LoRAs are outside **ltx25 transformer** GGUF scope and must not be emitted by the ltx25 profile. The two source-bundled embedding-connector components are the explicit exception: they are emitted in BF16 under their bare connector keys to preserve the existing LTX bundle-loader contract. This boundary is unchanged by gemma4-ltx25: that is a separate opt-in profile converting a separate source file (the standalone Gemma 4 text-encoder safetensors), not an extension of the ltx25 transformer's scope. The two profiles never read each other's source file and are independently source-locked.

## Official oracle, metadata, and component classification

E2 uses the official metadata to construct the three direct meta builders and obtain their expected state-dict keys, logical shapes, and fixed component membership. This is equivalent to the official one-strip builder handoff for this artifact; it contains neither a module-role table nor a source-dtype table. E3 authenticated safetensors-header rows are the only per-key source-dtype authority. Neither the current backend nor a third-party GGUF may supply either oracle. The converter consumes the resulting hashed `typemap/ltx25_builder_oracle.json` static artifact (SHA-256 `fa3b8518b5b55cbf9c8f072282eddb334bc60ec85ce4f13ea056b52adeb5bfe5`) and therefore does not introduce Torch or backend dependencies into converter CI.

For the split LTX 2.5 artifact, config.transformer exists and the artifact does not bundle vae, audio_vae, or vocoder. It does contain two Gemma embedding-connector components. The fixed exactly-one classification for every raw header key is:

- emit(transformer): after the one `model.diffusion_model.` strip, a non-connector key is a transformer `builder_state_dict_key` (4,091 rows).
- emit(bundle connector): `audio_embeddings_connector.` (129 rows) and `video_embeddings_connector.` (129 rows) are verified against their official meta-builder state dicts after their component-prefix strip, retain their explicit Gemma component IDs in E3, and emit their bare stripped names to GGUF as BF16.
- error: unknown key, no classification, or more than one classification.

There is no other category and no silent drop. Each of the three component key/shape sets must match E2 exactly before E3 is written; E3 then records the exact per-key source dtype from the authenticated safetensors header. All 4,349 rows emit; only the 258 connector rows use component-key shape comparison rather than the transformer builder key set.

Metadata source is __metadata__["config"]. It must be valid JSON and contain top-level config.transformer. Preserve its UTF-8 JSON bytes exactly in the output config KV; parse a copy only for validation. Do not reserialize or invent fields.

`__metadata__["gemma_source_checkpoint"]` is the second required metadata entry and follows the same verbatim rule. The official artifact carries `{"ltx_version": "2.5.0", "gemma_version": "gemma4-12b-ltx-v1"}`, and the backend's `_check_gemma_version` reads only its `gemma_version` field to decide which Gemma text encoder pairs with this transformer; without the KV a loader would have to guess or synthesize it, so ltx25 admits no source that lacks a non-empty JSON object with a non-empty `gemma_version` string. Admission happens before the multi-hour tensor write, not after. The output KV is the source string byte for byte -- `metadata.apply_kv` copies it through as a profile-specific passthrough key (absence is not warned about, because LTX 2.3 sources have never carried it and their output is unchanged), and E5 self-verification requires the KV to be present and byte-equal to the source. This is the sole GGUF KV contract change of the 2026-08-21 re-output; `general.architecture`, `general.quantization_version`, `general.file_type`, `config`, `license`, and `model_version` are unchanged, and no tensor payload byte changes (only the KV block grows, shifting the padded data offsets and therefore the file digest).

The following is the official-code-confirmed field vocabulary and defaults, not a claim of concrete checkpoint values. In model.py at E2, LTXModel accepts the configurable transformer fields model_type, num_attention_heads, attention_head_dim, in_channels, out_channels, num_layers, cross_attention_dim, norm_eps, positional_embedding_theta, positional_embedding_max_pos, timestep_scale_multiplier, use_middle_indices_grid, audio_num_attention_heads, audio_attention_head_dim, audio_in_channels, audio_out_channels, audio_cross_attention_dim, audio_positional_embedding_max_pos, av_ca_timestep_scale_multiplier, rope_type, double_precision_rope, apply_gated_attention, cross_attention_adaln, use_prompt_adaln_single, ff_bias, audio_ff_bias, and use_keyframes_abs_pos_embedding. The linked [model.py](https://github.com/Lightricks/LTX-2/blob/400fd31054597515f47125691032c04b1c3ee24e/packages/ltx-core/src/ltx_core/model/transformer/model.py#L43-L76) supplies the following fixed code defaults and conditional audio/video construction.

| Official-code field group | E2 constructor defaults | E3 artifact status |
| --- | --- | --- |
| Required metadata structure | top-level config.transformer | Must be present; its concrete values are not known before authenticated E3. |
| Video dimensions | model_type AudioVideo; heads 32; head dim 128; in/out channels 128; cross-attention dim 4096; layers 48 | Values are configurable input to the builder; record actual values and do not assume defaults. |
| Video numerics/position | norm epsilon 1e-6; theta 10000.0; timestep multiplier 1000; middle-grid true; default max position [20, 2048, 2048] | Record supplied values, including explicit null/omission resolved by the builder. |
| Audio dimensions/position | heads 32; head dim 64; in/out channels 128; cross-attention dim 2048; default max position [20] | Record only when transformer config/model type enables audio. |
| Feature flags | AV timestep multiplier 1; split RoPE; double precision false; gated attention false; cross-attention AdaLN false; prompt AdaLN true; FF biases true; keyframe absolute position false | All are configurable; E3 records the effective supplied value versus this code default. |

E3 must record actual config.transformer values, identify which supplied fields differ from code defaults, and validate only the required builder fields proven by E2. Do not hard-code defaults as checkpoint facts. general.architecture is a project adapter discriminator for GGUF writer/loader selection, not an official LTX 2.5 architecture contract.

## E3 admission and header extraction

The authenticated official artifact is inspected before any tensor payload conversion:

1. Require E1 repository, artifact revision/path, exact byte size, and local full SHA-256 to match the completed lock. Use explicit exceptions and raise ValueError or a domain error; never rely on assert for security/correctness.
2. Read the safetensors header only. Validate length, JSON type, duplicate JSON keys, metadata type, tensor name uniqueness, non-negative integer shapes, data offset structure/bounds/non-overlap, and exact dtype-times-shape payload byte count.
3. Preserve raw config bytes, validate JSON and top-level transformer, and build E2 meta model from the parsed copy.
4. Classify every raw tensor exactly once; normalize all emit keys with exactly one model.diffusion_model strip; compare transformer and both connector key/shape sets exactly with E2. Missing, extra, duplicate, shape mismatch, unknown, and multi-match errors show count plus bounded sorted examples.
5. Require every emitted row to match E2 key/shape exactly, and require its source dtype to match the authenticated E3 per-key header allowlist (transformer BF16/F32; both connector components BF16 only). This E3 allowlist and E1 source pin are the admission authority. Strings such as int8, fp8, quant, or convrot are diagnostics only; they may improve error messages but cannot alone reject a source.
6. Write E3 inventory only after all header/admission checks pass. Its minimal records are raw_key, classification, component_id, component_state_dict_key for connectors, builder_state_dict_key/output name, source_dtype, shape_logical, data_offsets, config_bytes_sha256, and inventory SHA-256.

This rejects prequantized inputs by observed selected dtype, not unreliable substring guesses. It excludes unrelated Gemma/VAE components while preserving the source-bundled connector payload required by the LTX bundle loader.

## E4 concrete conversion map and policy

E4 is a small checked-in JSON or YAML map created only after E1-E3. It is data, not a runtime policy DSL. Each entry is concrete:

    name, source_dtype, shape_logical, shape_gguf, ggml_type, nbytes, rule_id, reason, evidence_ids

Runtime conversion uses lookup only. Every E3 emitted builder key must map to exactly one E4 entry, with exact dtype and shape equality. Unknown source key, duplicate map name, missing map record, multiple map record, unsupported GGML type, invalid byte count, or K block-alignment failure is an error.

E3 confirms 4,349 emitted rows: transformer 3,801 BF16 and 290 F32, plus 258 BF16 connectors. The approved replacement E4 has 1,658 Q4_K, 2,401 BF16, and 290 F32 concrete rows. The independent E4 reviewer evidence, rather than E2, records the official `named_modules` role audit for the transformer: it preserves the 290 E3 F32 AdaLN/control-table rows as F32; 1,564 Linear biases, 576 RMSNorm weights, the direct keyframe-position parameter, and the two 128-wide patchify boundary weights as BF16; and assigns Q4_K only to 1,658 concrete BF16 rank-2 `torch.nn.Linear` weights whose last dimension is divisible by 256. The 258 connector rows have the direct `connector-bf16-bundle-preservation` rule: their source must be BF16 and their target must remain BF16, regardless of shape. Q5_K and Q6_K have no initial exceptions. Thus every row is concrete and exactly one type is selected; no pattern rule runs at conversion time. E4 contains only types supported by the existing writer until new writer/kernel support is separately designed: for ltx25 that remains {F32, BF16, Q4_K, Q5_K, Q6_K}; the gemma4-ltx25 profile (below) is the separately-designed extension that adds I8, and only its five sidecar rows may use it. Neither profile may select a writer-unsupported type by falling back to it silently -- an unsupported `ggml_type` is a map-validation error in both.

## Output, manifest, logging, and rollback

Before writing, calculate exact output bytes from E4 and GGUF framing, compare available free disk, and fail with required/available byte figures if insufficient. Write a unique temporary GGUF in the destination directory.

Verify the temporary GGUF completely for E5, close it, then make the sole commit point:

1. os.replace(temp_gguf, final_gguf)
2. write final_gguf.manifest.json

The pair is not claimed to be fully atomic. If manifest writing fails or later disagrees with final GGUF, verify must fail; do not silently recreate/guess a manifest. Existing final GGUF is preserved on failures before replace. On a current-run temporary cleanup failure, make one short bounded PermissionError retry, report the surviving path, and do not automatically delete that remnant later. No quarantine, journal, or database is needed.

The manifest is the minimal set: profile identifier, E1 source SHA-256, E3 inventory SHA-256, E4 policy/map SHA-256, E5 output SHA-256, tensor/type counts, and tool version. It does not duplicate E1-E6 reports, include paths/credentials/tensor values, or claim a backend result.

Logs include profile, E1/E3/E4 identifiers, estimated/free disk, temp/final paths at appropriate detail, counts, digests, and commit status. Expected failures have stable categories such as source-lock-incomplete, source-rejected, inventory-mismatch, map-mismatch, disk-preflight-failed, or manifest-missing, and say whether final replacement occurred.

Rollback is ltx25 disablement only. Never move, delete, or relabel a user output.

## gemma4-ltx25 profile (LTX-fine-tuned Gemma 4 text encoder)

This is a second, independent opt-in profile implemented in `src/converter/ltx25_gemma.py`. It shares the reader/writer/streaming/K-quant kernels with ltx23/ltx25 but has its own source lock, its own E2/E3/E4 artifacts, and its own quantization policy. It does not use `ltx25.py`'s emit/exclude component routing: the source file bundles nothing outside scope, so every tensor in the header is emitted verbatim under its raw safetensors key (no prefix strip, no renaming).

The sole source is `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (repo `Lightricks/LTX-2.5`, revision `dd53cc2cd45bbeaa3563dfb575cba3f49cf44761`, 26,263,860,594 bytes, SHA-256 `1c647a94c0e902fb87f9a403cbca36a8b6d8e5867094442df1b41ae557cfd1c6`, per `download-5files-report.json`). It has 686 tensors: 664 backbone (40 sliding-attention layers x 14 rows + 8 full-attention layers x 13 rows, full layers sharing k=v so they omit `v_proj`), `embed_tokens`/`norm`, two `aggregate_embed` weights plus their biases, two projector rows, nine vision-tower rows, and five U8 sidecar payloads (tokenizer/config JSON and the chat template, stored as raw byte blobs, not numeric tensors). Its `__metadata__` key is `gemma_config` (not `config`); its UTF-8 bytes (3,338 B) are written verbatim to the output `config` KV, matching the ltx25 transformer's own verbatim-config convention.

Quantization policy (concrete, checked-in E4, no pattern rule at conversion time): 328 backbone `nn.Linear` weights (the seven per-layer `{mlp.down_proj,mlp.gate_proj,mlp.up_proj,self_attn.q_proj,self_attn.k_proj,self_attn.o_proj,self_attn.v_proj}.weight` rows, all with an input width divisible by 256) are Q4_K. The two `aggregate_embed` weights are Q6_K (measured rel-RMSE 0.0173 vs. 0.0656 for Q4_K; both are `nn.Linear` weights so this is real VRAM savings). `embed_tokens` is BF16 despite being K-alignment-eligible, because the current backend always expands embeddings to BF16 at load and quantizing it would save nothing while adding risk. All remaining rows -- norms, `layer_scalar`, `q_norm`/`k_norm`, biases, the two projectors, and the nine-row vision tower -- are BF16 verbatim; none of these is llama.cpp-style RMSNorm, so the `(1+w)` fold llama.cpp applies to some RMSNorm weights must never be applied here (rule_id `no-rmsnorm-folding`). The five U8 sidecars are written as I8 (a byte-identical passthrough type; gguf-py has no U8 writer type). Total: Q4_K 328, Q6_K 2, BF16 351, I8 5 = 686.

Classification is an ordered, mutually-exclusive rule table evaluated most-specific-first, because at least three rows would otherwise multi-match a naive "2-D `.weight` with a 256-divisible last dim" Q4_K predicate: `vision_model.patch_dense.weight` ([3840, 6912], 6912 = 27x256), `multi_modal_projector.embedding_projection.weight` ([3840, 3840]), and `model.embed_tokens.weight` ([262144, 3840]). The table checks, in order: (1) the five named U8 sidecars -> I8; (2) `vision_model.*` -> BF16; (3) `multi_modal_projector.*` -> BF16; (4) `audio_projector.*` -> BF16; (5) the two named `aggregate_embed` weights -> Q6_K; (6) `model.embed_tokens.weight` -> BF16; (7) the seven-name backbone Linear allowlist (rank-2, `.weight` suffix, last dim % 256 == 0) -> Q4_K; (8) everything else -> BF16 (`no-rmsnorm-folding`). An unrecognized tensor name is not silently defaulted; it fails closed.

E2 for this profile is a deterministic, torch-free derivation from the source file's own `gemma_config.text_config` (layer count, hidden size, per-layer-type head/kv-head dimensions, `layer_types`) plus two named constants carried over from the already-audited ltx25 transformer config (video/audio aggregate output width 4096/2048, from `num_attention_heads*attention_head_dim` and `audio_num_attention_heads*audio_attention_head_dim` at `encoder_configurator.py`'s `_create_feature_extractor`, matching this document's own recorded transformer config values). It reconstructs all 664 backbone key/shape pairs from formula and pins the remaining 22 non-backbone rows (embed/norm/projectors/vision/sidecars) as fixed known contract rows keyed to the authenticated header. The checked-in artifact is `typemap/gemma4_ltx25_builder_oracle.json`.

GGUF KV metadata: `general.architecture` = `"ltxv"` (set via the shared writer's `arch=` constructor argument, matching ltx25 transformer for family consistency; the LTX 2.3 text encoder instead identifies as `gemma3` -- a future `check_kv` extension consuming both families needs to handle both), `config` = the verbatim `gemma_config` UTF-8 bytes, `ltx.component` = `"text_encoder"`, `ltx.text_encoder.gemma_version` = `"gemma4-12b-ltx-v1"` (the config's own `gemma_version` field, restated). No `model_version` KV is written; `metadata.apply_kv` is not used (its `config`-required design does not fit a `gemma_config`-keyed source) -- `ltx25_gemma.py` writes these KVs directly, and `metadata.py` is unmodified.

Errors subclass `Gemma4Error(ltx25.Ltx25Error)`, so `cli.py`'s existing `except ltx25_mod.Ltx25Error` handler catches them without modification; low-level OS-mechanics helpers (temp-file handling, atomic JSON write, disk preflight) are reused as-is from `ltx25.py` and may still surface as `ltx25.*` exception types on rare OS-level failure, which is an accepted, disclosed simplification (both ultimately subclass `Ltx25Error`, so the CLI's user-facing behavior is unaffected).

Consumer handoff (for the future backend loader, recorded here so a later implementation does not repeat a known mistake): (1) raw keys must never be remapped -- they are written to GGUF exactly as they appear in the source header; (2) the RMSNorm `(1+w)` correction some existing backend code applies to Gemma 3 norms must not be copied onto these BF16 norm rows -- see `no-rmsnorm-folding` above; (3) `layer_scalar` must be registered as a buffer, not a learnable parameter; (4) the 262,144-entry vocabulary declared in `gemma_config.text_config.vocab_size` has no separate padding entry recorded for Gemma 4 -- do not assume the 262,208-padded-vocabulary convention documented for Gemma 3 without re-checking; (5) the five I8 sidecar rows are the original UTF-8 byte payloads stored as int8 for byte-identical passthrough, not numeric tensors -- the loader must reinterpret their raw bytes as uint8 before treating them as JSON/text (e.g. viewing the tensor data as `np.uint8` and taking its bytes), never consume them as signed int8 values.

## Tests and acceptance

Converter CI remains converter-only: no backend, Torch, or full-artifact dependency. Required tests are:

- Existing LTX 2.3 regression and omitted-model versus explicit-ltx23 equivalence.
- Header-only small fixtures for malformed header/JSON, duplicate fields/names, shapes, offsets, bounds, byte-size mismatch, config absence/invalid JSON, and exact-one classification.
- E3-selected BF16/F32 acceptance and F16/INT8/FP8/NVFP4 or any E3 dtype-mismatch rejection. Substring markers are diagnostic-only tests, never an independent reject test.
- E2 exact raw-to-builder/component key-shape comparison plus E3 per-key dtype comparison, one-strip normalization, 4,091+129+129 bundle emission, bare connector GGUF names, BF16 connector preservation, and no unrelated VAE/Gemma leakage.
- E4 map lookup, duplicate/missing/unknown/multi-match behavior, shape/byte/K alignment, and static raw-key/shape/GGML handoff.
- Synthetic F32/BF16/Q4_K/Q5_K/Q6_K writer coverage and existing K numerical tests. E6 covers only concrete types actually selected by E4.
- Config byte preservation, GGUF metadata/static self-verification, disk-preflight calculation, temporary-file behavior, manifest absence/mismatch, and output commit ordering.
- `gemma_source_checkpoint` passthrough: verbatim bytes in the output KV, admission rejection for a missing/empty/non-JSON/non-object/`gemma_version`-less value before any tensor write, self-verify failure when a writer drops the KV, and no such KV or warning for an LTX 2.3 source.

E1-E4 are complete for the corrected bundle contract, and E5/E6 are complete for the fresh 4,349-row artifact (see Open gates). Backend integration remains a separate repository/stage. After it implements LTX 2.5 construction, run one representative fixed-seed end-to-end case; no broad benchmark campaign is part of this plan.

## Open gates

1. ltx25 transformer E5/E6 evidence for approved E4 `typemap/ltx25_conversion_map.json` (SHA-256 `6d41db41a1b6b8f485434a64be198abf3f9a8b66b0dcd7441624644c19317760`): **complete as of 2026-08-21.** Output `LTX-2.5-22B-distilled-transformer.gguf`, 14,738,670,464 bytes, SHA-256 `c26473d2d5c3eb61012f60769ac4dccdb8bb77935678206c0148fff351ba060d`, 4,349 tensors (Q4_K 1,658, BF16 2,401, F32 290), inventory SHA-256 `c0966ac37f55f318be16334c1f1e1c2db1f467dafde09c0c2bd1b853206bde97`. Self-verify passed both in-run and as a separate `self-verify --model ltx25` invocation, and the E6 K-quant numerical check passed.
   This is the re-output that adds the `gemma_source_checkpoint` KV. It supersedes the first 2026-08-21 build (SHA-256 `4ead2a7dbae374639794b717517a3d1938bb8a16629957467e617ee99f22cf98`, 14,738,670,368 bytes), which lacked that KV and which the official `_check_gemma_version` would have rejected outright: with `model_version` `"2.5.0"` at or above its `(2, 4, 0)` floor and a Gemma config that does declare `gemma_version`, the absent record raises `ValueError`, not a warning. Against that earlier build the two were compared tensor by tensor: all 4,349 payloads are byte-identical, every shared KV value is unchanged, and the only difference is the one added key (the +96 bytes are its KV entry plus data-offset padding). The earlier build's E6 numbers therefore carry over unchanged.
2. gemma4-ltx25 E5/E6 evidence: **complete as of 2026-08-21.** Output `LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf`, 9,231,374,624 bytes, SHA-256 `4e69e4a33065c856e039fb4c1b78adffae582c4da14b633224136a7e7e30fb90`, 686 tensors (Q4_K 328, Q6_K 2, BF16 351, I8 5). Both self-verify and the E6 K-quant numerical check passed.
3. Separate backend implementation/acceptance after converter E1-E6 pass, including one fixed-seed representative end-to-end case.
4. Any publication/release decision remains outside this converter evidence and must not be inferred from a staging output.
