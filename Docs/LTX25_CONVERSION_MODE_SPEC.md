# LTX 2.5 conversion mode specification

## Status, compatibility, and enablement gate

This document is the implementation contract for exactly two frozen profiles: legacy ltx23 and opt-in ltx25. The LTX 2.3 default CLI, config, reference-typemap workflow, output naming, and artifact behavior remain backward compatible.

LTX 2.5 conversion accepts only the independently approved E4 map. E1-E3 are complete: a selected ltx25 command admits only the authenticated pinned source, uses the compatible official builder key/shape/component oracle, and writes/verifies the canonical inventory; it refuses a draft map for conversion. E4 is independently approved, and the staged E5/E6 run below has completed. This converter evidence does not authorize publication or backend use.

The sole initial LTX 2.5 source is:

    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors

Its selected transformer tensors must match the authenticated E3 safetensors-header per-key source-dtype table: 3,801 BF16 and 290 F32. F16, INT8 ConvRot, FP8, NVFP4, or any other dtype that differs from the exact E3 row is rejected. The converter must not dequantize, reinterpret, or silently accept a prequantized source.

## Evidence register and primary sources

The six evidence IDs avoid duplicate reports. Each artifact is canonical JSON with deterministic field ordering and a SHA-256.

| ID | Artifact and purpose | Gate |
| --- | --- | --- |
| E1 | Source lock: official repository, artifact revision/path/size, license, and verified source content SHA-256. | Complete: authenticated Hub LFS SHA-256 and local full-file SHA-256 agree. |
| E2 | Official-code oracle: pinned code revision, direct meta-builder recipe, and fixed component key/shape contract. It has no module-role table and is not a dtype oracle. | Complete: 4,091 transformer, 129 audio connector, 129 video connector rows exactly match official meta builders. |
| E3 | Canonical safetensors header inventory: metadata bytes/digest, selected/excluded raw keys, normalized builder keys, dtypes, shapes, and offsets. | Complete: 4,349 independent header rows agree with converter inventory. |
| E4 | Concrete checked-in conversion map/policy, generated from E2 plus E3. | Independently approved: SHA-256 `3225111400b7405a0ddff9c6a3f0623f5f6f82ff17c67f6e093bea6873d4282e`. |
| E5 | Converted GGUF static verification and output digest. | Complete for the staged output: 10,706,310,592 bytes; SHA-256 `36a372cd5b8a6078b9b7422791b8cb702b9026f804c7dd5e44c5fb252d34033e`; converter and independent CLI verification both checked E1/E3/E4 linkage, 4,091 tensors, types/shapes, and config bytes. |
| E6 | K-quant numerical evidence for the concrete E4 types. | Complete: bounded dequantization found all 18,976,866,304 Q4_K values finite. A deterministic first/middle/last-block sample of all 1,658 Q4_K tensors (4,974 blocks / 1,273,344 values) yielded MAE 0.0025929245, RMSE 0.0038327283, L2 NRMSE 0.0782986655, and max absolute error 0.0468978882 against the admitted BF16 source. |

Primary sources, all immutable where a revision is supplied:

- [Hugging Face model](https://huggingface.co/Lightricks/LTX-2.5), [artifact tree at release dd53cc2cd45bbeaa3563dfb575cba3f49cf44761](https://huggingface.co/Lightricks/LTX-2.5/tree/dd53cc2cd45bbeaa3563dfb575cba3f49cf44761), [model API at that revision](https://huggingface.co/api/models/Lightricks/LTX-2.5/revision/dd53cc2cd45bbeaa3563dfb575cba3f49cf44761), and [official LICENSE](https://huggingface.co/Lightricks/LTX-2.5/blob/dd53cc2cd45bbeaa3563dfb575cba3f49cf44761/LICENSE).
- [Official LTX code oracle root at 400fd31054597515f47125691032c04b1c3ee24e](https://github.com/Lightricks/LTX-2/tree/400fd31054597515f47125691032c04b1c3ee24e).
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
| canonical header inventory | `typemap/ltx25_inventory.json`, SHA-256 `c6f17d849ced4743a70e3733bef678460ff0a1840fbb1d206e8578ecb08c3250` |

The Git blob OID is not a SHA-256 content digest. VirusTotal URL values are not a source hash and must never be copied to the lock as one. E1 was completed with authenticated official download, byte-size check, full local SHA-256, and matching Hub LFS/Xet content hash. The reviewed E4 map permits conversion; E5/E6 remain incomplete.

## Audited current repository and backend constraints

| Area | Existing contract | Evidence |
| --- | --- | --- |
| Streaming reader | The safetensors reader holds a small header and reads one payload at a time. | src/converter/convert.py:82-97, 137-164 |
| Source reader | BF16/F16/F32 only today; unsupported dtype raises. | src/converter/convert.py:143-164, 166-186 |
| LTX 2.3 selection | model.diffusion_model emits; VAE/audio VAE/vocoder/text projection skip; others fail. | src/converter/convert.py:65-70, 211-266 |
| Exact set check | Duplicate typemap names, missing, extra, unexpected keys fail. | src/converter/convert.py:234-266 |
| Writer | F32, BF16, Q4_K, Q5_K, Q6_K are the implemented target types. | src/converter/convert.py:269-309 |
| Metadata | Config JSON is required; current writer emits project GGUF metadata. | src/converter/metadata.py:32-42, 58-125 |
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

Gemma, text encoder/projection, VAE, audio VAE, vocoder, upscalers, and LoRAs are outside ltx25 transformer GGUF scope and must not be emitted.

## Official oracle, metadata, and component classification

E2 uses the official metadata to construct the three direct meta builders and obtain their expected state-dict keys, logical shapes, and fixed component membership. This is equivalent to the official one-strip builder handoff for this artifact; it contains neither a module-role table nor a source-dtype table. E3 authenticated safetensors-header rows are the only per-key source-dtype authority. Neither the current backend nor a third-party GGUF may supply either oracle. The converter consumes the resulting hashed `typemap/ltx25_builder_oracle.json` static artifact (SHA-256 `fa3b8518b5b55cbf9c8f072282eddb334bc60ec85ce4f13ea056b52adeb5bfe5`) and therefore does not introduce Torch or backend dependencies into converter CI.

For the split LTX 2.5 artifact, config.transformer exists and the artifact does not bundle vae, audio_vae, or vocoder. It does contain two Gemma embedding-connector components. The fixed exactly-one classification for every raw header key is:

- emit: after the one `model.diffusion_model.` strip, a key not in either connector prefix is a transformer `builder_state_dict_key` (4,091 rows).
- exclude(component-id): `audio_embeddings_connector.` (129 rows) and `video_embeddings_connector.` (129 rows) are verified against their official meta-builder state dicts after their component-prefix strip, and are excluded as the two explicit Gemma component IDs.
- error: unknown key, no classification, or more than one classification.

There is no other category and no silent drop. Each of the three component key/shape sets must match E2 exactly before E3 is written; E3 then records the exact per-key source dtype from the authenticated safetensors header. Gemma connector weights never enter transformer GGUF.

Metadata source is __metadata__["config"]. It must be valid JSON and contain top-level config.transformer. Preserve its UTF-8 JSON bytes exactly in the output config KV; parse a copy only for validation. Do not reserialize or invent fields.

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
4. Classify every raw tensor exactly once; normalize emit keys with exactly one model.diffusion_model strip; compare key and logical shape sets exactly with E2. Missing, extra, duplicate, shape mismatch, unknown, and multi-match errors show count plus bounded sorted examples.
5. Require every emitted and explicitly excluded row to match E2 key/shape exactly, and require its source dtype to match the authenticated E3 per-key header allowlist (BF16 or F32 for emitted rows; BF16 for both connector components). This E3 allowlist and E1 source pin are the admission authority. Strings such as int8, fp8, quant, or convrot are diagnostics only; they may improve error messages but cannot alone reject a source.
6. Write E3 inventory only after all header/admission checks pass. Its minimal records are raw_key, classification, component_id when excluded, builder_state_dict_key when emitted, source_dtype, shape_logical, data_offsets, config_bytes_sha256, and inventory SHA-256.

This rejects prequantized inputs by observed selected dtype, not unreliable substring guesses. It also guarantees no Gemma/VAE component is mixed into transformer GGUF.

## E4 concrete conversion map and policy

E4 is a small checked-in JSON or YAML map created only after E1-E3. It is data, not a runtime policy DSL. Each entry is concrete:

    name, source_dtype, shape_logical, shape_gguf, ggml_type, nbytes, rule_id, reason, evidence_ids

Runtime conversion uses lookup only. Every E3 emitted builder key must map to exactly one E4 entry, with exact dtype and shape equality. Unknown source key, duplicate map name, missing map record, multiple map record, unsupported GGML type, invalid byte count, or K block-alignment failure is an error.

E3 confirms 4,091 emitted transformer rows: 3,801 BF16 and 290 F32. The approved E4 has 1,658 Q4_K, 2,143 BF16, and 290 F32 concrete rows. The independent E4 reviewer evidence, rather than E2, records the official `named_modules` role audit: it preserves the 290 E3 F32 AdaLN/control-table rows as F32; 1,564 Linear biases, 576 RMSNorm weights, the direct keyframe-position parameter, and the two 128-wide patchify boundary weights as BF16; and assigns Q4_K only to 1,658 concrete BF16 rank-2 `torch.nn.Linear` weights whose last dimension is divisible by 256. Q5_K and Q6_K have no initial exceptions. Thus every row is concrete and exactly one type is selected; no pattern rule runs at conversion time. E4 contains only types supported by the existing writer until new writer/kernel support is separately designed.

## Output, manifest, logging, and rollback

Before writing, calculate exact output bytes from E4 and GGUF framing, compare available free disk, and fail with required/available byte figures if insufficient. Write a unique temporary GGUF in the destination directory.

Verify the temporary GGUF completely for E5, close it, then make the sole commit point:

1. os.replace(temp_gguf, final_gguf)
2. write final_gguf.manifest.json

The pair is not claimed to be fully atomic. If manifest writing fails or later disagrees with final GGUF, verify must fail; do not silently recreate/guess a manifest. Existing final GGUF is preserved on failures before replace. On a current-run temporary cleanup failure, make one short bounded PermissionError retry, report the surviving path, and do not automatically delete that remnant later. No quarantine, journal, or database is needed.

The manifest is the minimal set: profile identifier, E1 source SHA-256, E3 inventory SHA-256, E4 policy/map SHA-256, E5 output SHA-256, tensor/type counts, and tool version. It does not duplicate E1-E6 reports, include paths/credentials/tensor values, or claim a backend result.

Logs include profile, E1/E3/E4 identifiers, estimated/free disk, temp/final paths at appropriate detail, counts, digests, and commit status. Expected failures have stable categories such as source-lock-incomplete, source-rejected, inventory-mismatch, map-mismatch, disk-preflight-failed, or manifest-missing, and say whether final replacement occurred.

Rollback is ltx25 disablement only. Never move, delete, or relabel a user output.

## Tests and acceptance

Converter CI remains converter-only: no backend, Torch, or full-artifact dependency. Required tests are:

- Existing LTX 2.3 regression and omitted-model versus explicit-ltx23 equivalence.
- Header-only small fixtures for malformed header/JSON, duplicate fields/names, shapes, offsets, bounds, byte-size mismatch, config absence/invalid JSON, and exact-one classification.
- E3-selected BF16/F32 acceptance and F16/INT8/FP8/NVFP4 or any E3 dtype-mismatch rejection. Substring markers are diagnostic-only tests, never an independent reject test.
- E2 exact raw-to-builder key/shape component comparison plus E3 per-key dtype comparison, one-strip normalization, three-component exclude behavior, and no VAE/Gemma leakage into transformer GGUF.
- E4 map lookup, duplicate/missing/unknown/multi-match behavior, shape/byte/K alignment, and static raw-key/shape/GGML handoff.
- Synthetic F32/BF16/Q4_K/Q5_K/Q6_K writer coverage and existing K numerical tests. E6 covers only concrete types actually selected by E4.
- Config byte preservation, GGUF metadata/static self-verification, disk-preflight calculation, temporary-file behavior, manifest absence/mismatch, and output commit ordering.

The staged artifact has completed the converter gates: authenticated E1 download; E3 header extraction; E2 meta-builder exact oracle; reviewer-approved E4; full conversion/E5; and E6 numerical checks. Backend integration is a separate repository/stage. After it implements LTX 2.5 construction, run one representative fixed-seed end-to-end case; no broad benchmark campaign is part of this plan.

## Open gates

1. Separate backend implementation/acceptance after converter E1-E6 pass, including one fixed-seed representative end-to-end case.
2. Any publication/release decision remains outside this converter evidence and must not be inferred from the staging output.
