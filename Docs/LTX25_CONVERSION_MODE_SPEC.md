# LTX 2.5 conversion mode specification

## Status, compatibility, and enablement gate

This document is the implementation contract for exactly four frozen profiles: legacy ltx23, opt-in ltx25 (the LTX 2.5 transformer), opt-in gemma4-ltx25 (the LTX-fine-tuned Gemma 4 text encoder bundled in a separate source file), and opt-in ltx25-comfyquant (a community redistribution of the LTX 2.5 transformer whose Linear weights arrive pre-quantized in a ComfyUI format). The LTX 2.3 default CLI, config, reference-typemap workflow, output naming, and artifact behavior remain backward compatible. gemma4-ltx25 and ltx25-comfyquant are each documented in their own section below; every clause elsewhere in this document that refers to "the ltx25 profile" or "the LTX 2.5 transformer" describes ltx25 specifically and does not extend to gemma4-ltx25 or ltx25-comfyquant unless that profile's own section says so.

LTX 2.5 conversion accepts only an independently approved E4 map. E1-E3 are complete: a selected ltx25 command admits only the authenticated pinned source, uses the compatible official builder key/shape/component oracle, and writes/verifies the canonical 4,349-row bundle inventory; it refuses a draft map for conversion. The replacement 4,349-row E4 is independently approved. The earlier 4,091-row map and its E5/E6 output are intentionally stale after the bundle migration and cannot be reused. Fresh E5/E6 are complete for the ltx25 and gemma4-ltx25 profiles as of 2026-08-21 (ltx25-comfyquant has no E-numbered evidence of its own; its contract and gates are in its own section below); the recorded output digests are in the Open gates section. Passing them is necessary but not sufficient for publication or backend use, which remain separately gated.

The sole initial LTX 2.5 source is:

    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors

Its 4,349 emitted tensors must match the authenticated E3 safetensors-header per-key source-dtype table: 3,801 transformer BF16, 290 transformer F32, and 258 connector BF16. F16, INT8 ConvRot, FP8, NVFP4, or any other dtype that differs from the exact E3 row is rejected. Within ltx25 the converter must not dequantize, reinterpret, or silently accept a prequantized source.

This dtype clause governs the ltx25 profile only. ltx25-comfyquant is a separate opt-in profile with its own source, its own admission contract, and its own section below: it deliberately admits a pre-quantized community source and dequantizes it explicitly. It neither relaxes nor reinterprets anything above -- ltx25 keeps refusing every non-BF16/F32 source dtype, and neither profile reads the other's source file.

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
| Per-layer mode stores quantized raw bytes and dequantizes each Linear weight during forward. | ../Nz-Videomni/engine/gguf/quant_service.py:729-734 |
| It reverses GGUF shape; F32/F16/BF16 are floats and other supported types remain packed bytes. | ../Nz-Videomni/engine/gguf/quant_service.py:582-615 |
| Active per-layer load applies no remap, so emitted raw names must be exact. | ../Nz-Videomni/engine/gguf/quant_service.py:617-621 |
| It reads embedded config but its current construction capability is not evidence of LTX 2.5 support. | ../Nz-Videomni/engine/gguf/quant_service.py:536-561 |

## Profile, CLI, and scope boundary

The profile implementation is a frozen four-entry data table plus explicit functions or if statements. Do not use a class hierarchy, generic DSL/registry, plugin system, dynamic import, or automatic profile detection.

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

## ltx25-comfyquant profile (community pre-quantized LTX 2.5 transformer)

### Purpose and boundary

This is a fourth, independent opt-in profile implemented in `src/converter/ltx25_comfyquant.py`, with the numeric inverse quantization isolated in `src/converter/comfy_dequant.py`. It admits a *community* redistribution of the same LTX 2.5 transformer architecture whose `torch.nn.Linear` weights were pre-quantized by ComfyUI-style tooling, dequantizes them to float32, and re-quantizes to the GGUF types the backend's fused kernels implement. Its purpose is the case where no bf16 original of a fine-tune exists anywhere, so the pre-quantized file is the only obtainable source.

The admitted source quantization formats are exactly `int8_tensorwise` (8-bit weights with a per-output-row scale, optionally rotated by ConvRot) and `asym_w4a8_int8` (4-bit codebook, per-group FP8 scale, per-row F32 scale, ConvRot unconditional). A tensor that carries no quantization sidecars passes through as plain BF16 or F32. Every other stored dtype and every other marker format -- FP8, NVFP4, or any name not on the allowlist -- is rejected, as is an unknown marker key, a ConvRot group size other than 256, or a w4a8 group size other than 16.

The scope boundary of the ltx25 section applies here unchanged: only the 4,091 transformer rows and the two 129-row embedding connectors are emitted, the connectors keep their bare stripped names and BF16 type, and Gemma language-model weights, text projection, VAE, audio VAE, vocoder, upscalers, and LoRAs remain out of scope. ltx25-comfyquant and ltx25 never read each other's source file. This profile shares the reader/writer/streaming/K kernels and reuses `ltx25.py`'s header, digest, disk-preflight and atomic-write helpers -- the same "separate module, reuse the private helpers" arrangement `ltx25_gemma.py` uses. `convert.py`, `quant_kernels.py`, `metadata.py`, `ltx25_gemma.py` and every checked-in `typemap/` artifact are unmodified by it.

### Structural admission: no file-identity lock

A community build has no authenticated repository, revision or published digest, so this profile has no E1-style source lock. Identity is replaced by five structural checks. All of them are fail-closed and all of them run before the multi-hour tensor write:

1. **Config digest.** The UTF-8 bytes of `__metadata__["config"]` must hash to the pinned builder oracle's `config_bytes_sha256`. A file whose architecture definition differs by one byte is not this architecture and is refused.
2. **Key set.** Every raw header key must carry the official `model.diffusion_model.` prefix; exactly one strip is applied; the folded logical key set must equal the official three-component oracle (4,091 transformer + 129 audio connector + 129 video connector = 4,349) with no missing, extra or duplicate row.
3. **Shape.** Every logical row's shape must equal the oracle's shape exactly.
4. **Format allowlist.** Every quantized layer must carry a `comfy_quant` marker naming an admitted format, and its sidecar set, dtypes and shapes must match that format's geometry exactly.
5. **`gemma_source_checkpoint`.** Required, as for ltx25: a non-empty JSON object with a non-empty `gemma_version` string, copied verbatim to the output KV.

`--expect-sha256` is an *optional* identity pin: when supplied, the source's full SHA-256 must equal it. It is deliberately not required and is not stored in `config.toml`, because there is no authority to pin it to.

Unknown `__metadata__` keys are recorded, never rejected: the community tooling writes its own entries (`quant_format`, `quant_mixed_hi_layers`), which are preserved in the inventory and the manifest and never written to the GGUF.

### Header completeness

After the shared header validation and before anything else, the largest declared `data_offsets[1]` must equal `file size - payload base`. A mismatch is rejected with a message that names the declared payload size, the size the file actually provides, the difference, and how many tensors end beyond the end of the file. The shared tensor-entry validation would also stop such a file, but its message names one arbitrary out-of-range tensor; this front-loaded check exists so the operator is told the real cause, which is almost always a truncated download.

### Logical tensors and sidecar geometry

A quantized weight is stored as `<layer>.weight` plus sidecars. The suffixes `weight_scale`, `weight_codebook`, `weight_s_channel`, `weight_s_rel` and `comfy_quant` are folded into the parent `<layer>.weight` row; a sidecar with no parent weight is an error, and no sidecar ever reaches the output. The stripped logical name is what the type policy and the GGUF use; the raw key is retained beside it because the shared component classifier requires the `model.diffusion_model.` prefix and rejects an already-stripped key.

The `comfy_quant` marker is a 1-D `U8` UTF-8 JSON byte string. Per format the sidecar set must be exactly:

| Format | Stored `weight` | Required sidecars | Logical width |
| --- | --- | --- | --- |
| `int8_tensorwise` | I8 `[out, in]` | `weight_scale` F32 `[out, 1]` or `[out]`; `comfy_quant` | `in` |
| `asym_w4a8_int8` | I8 `[out, in/2]`, two 4-bit codes per byte | `weight_s_rel` F8_E4M3 `[out, in/group_size]`; `weight_s_channel` F32 `[out]`; `weight_codebook` F32 `[16]`; `comfy_quant` | twice the stored width |

The logical width must be divisible by `group_size` and, where ConvRot is enabled, by `convrot_groupsize`. A tensor with no sidecars is admitted only as BF16 or F32.

`ltx25._DTYPE_BITS` gains a single `F8_E4M3` entry so the shared header reader can size the w4a8 scale sidecar. That entry is a size table, not an admission list: the ltx25 profile's own source-dtype allowlist is unchanged and still rejects an F8_E4M3 source with its existing BF16/F32 message.

### Type policy

The approved ltx25 E4 map is the type policy. It is loaded with the same approved-status requirement as ltx25 and read for `name`, `shape_logical` and `ggml_type` only. Its `inventory_sha256` binds it to the *official* bf16 inventory and its `source_dtype` column describes the *official* file, so neither applies to a community source and both are deliberately ignored; the map file is never written and its SHA-256 is recorded in the manifest.

The join requires the policy's name set and the folded inventory's name set to be equal and every shape to match. Source dtypes are intentionally not compared, because a community file legitimately stores some rows at a different width than the official one. Target selection is then fixed:

- a `Q4_K` map row is emitted as `--quant-type` (default `Q6_K`; `Q4_K` reproduces the official type layout exactly),
- a `BF16` or `F32` map row is emitted unchanged,
- any other map type is an error,
- narrowing an F32 source row to a BF16 target is an error (no such row exists in the current data, but the guard is explicit rather than assumed).

The post-assertion is derived from the policy itself -- the policy's own type histogram with `Q4_K` rewritten to the selected type -- so a miniature fixture states its own expected counts without a second code path. On the approved 4,349-row map that derivation is `{BF16: 2401, F32: 290, <quant_type>: 1658}`.

Connector rows that the community file stores quantized are dequantized and emitted BF16, preserving the bundle-loader contract. Raw-byte containment (`_verify_connector_raw_bytes`) cannot hold for a payload that was dequantized, so `_verify_output` is called with `source_path=None` and the numerical comparison against the official artifact (gates G-A and G-B below) is the replacement evidence.

### Dequantization

`comfy_dequant.py` implements the inverse of both formats in float32 with NumPy only. It knows nothing about safetensors, GGUF, type maps or the CLI: it takes plain arrays and returns a plain array. The formats themselves are documented by the upstream comfy-kitchen project (Apache-2.0), which was read as a *specification*; no code was copied from it, and nothing was ported from ComfyUI itself (GPL-3.0).

- **ConvRot** uses the normalized regular Hadamard matrix: the Kronecker power of the 4x4 seed `[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]` divided by the square root of its size. That matrix is symmetric, orthogonal and involutory, so the inverse rotation is the same operation as the forward one. Only size 256 is accepted.
- **`int8_tensorwise`**: multiply the int8 codes by the per-output-row F32 scale, then, when the marker sets `convrot`, apply the rotation to contiguous 256-channel groups.
- **`asym_w4a8_int8`**: split each byte low nibble first (byte *i* holds element *2i* in bits 0-3 and element *2i+1* in bits 4-7); look the 4-bit codes up in the per-tensor 16-entry F32 codebook; multiply by the group's FP8 scale; round to the int8 grid and clip to [-127, 127], as the format defines the intermediate; multiply by the per-row F32 channel scale; apply the inverse rotation.
- **FP8 E4M3** is decoded through a 256-entry table built from the format definition: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias 7, subnormals at `2^-6 * mantissa/8`, no infinities, NaN only at `0x7F`/`0xFF`, largest finite value 448. NumPy has no FP8 dtype, so the sidecar is read as raw `uint8`.
- The rotation is performed in **float32**. ComfyUI casts to bf16 before rotating; for a GGUF target the extra precision is kept.
- **Non-finite values are an immediate error**, checked on every row chunk of the result and on the decoded FP8 scales, with the layer named. This is load-bearing: the connector rows emitted as BF16 are outside `_verify_output`'s NaN/Inf sweep, which covers Q types only.
- Memory is bounded. Nibble expansion, codebook lookup, scale multiplication and the rotation run in one row-chunk loop writing straight into a pre-allocated float32 output array, so the transient working set stays in the tens of MiB instead of exceeding a gigabyte on the largest Linear.
- bf16 emission rounds half-to-even, bit-identical to the shared reader's F32 branch, so a dequantized row emitted as BF16 rounds the way the rest of the converter rounds.

### Output contract

The output GGUF carries exactly seven KV keys: `general.architecture`, `general.quantization_version`, `general.file_type`, `config`, `license`, `model_version`, `gemma_source_checkpoint`. `metadata.apply_kv` writes only keys it knows, so the community file's own quantization metadata is dropped by construction; the check turns that into a contract in both directions and compares *sets*, not counts, so the failure names the offending key. An eighth key that `apply_kv` does know (for example `encrypted_wandb_properties`) fails just as loudly as a missing one.

`general.file_type` stays at the fixed value 15 (Q4_K_M) that `metadata.py` writes for every profile. With `--quant-type Q6_K` that value does not describe the file. `metadata.py` is deliberately unmodified: the backend never reads `general.*`, and the authoritative type record is the manifest's `quant_type`, restated in the manifest's `general_file_type_note` for the benefit of third-party tools that do read `general.file_type`.

The written file must be exactly the size calculated from the records and the writer framing before the write began. `_verify_output` runs on the temporary file -- tensor count, names, order, types and shapes against the records, `config` KV bytes equal to the source, `gemma_source_checkpoint` KV bytes equal to the source, and a bounded finiteness sweep over every Q tensor -- followed by the KV-key-set check and the size assertion. The commit order of the ltx25 section is unchanged: verify the temporary file, `os.replace`, then write `<out>.inventory.json` and `<out>.manifest.json`. A failure after the replace is a manifest error that names the committed output.

### Manifest

Format identifier `nz-ltx25-comfyquant-manifest-v1`, written to `<out>.manifest.json`. It is the only record of provenance; none of it is added to the GGUF KV. Fields:

`format`, `profile`, `source_path`, `source_size`, `source_sha256`, `source_quant_format`, `source_quant_mixed_hi_layers`, `quant_kind_counts`, `official_map_path`, `official_map_sha256`, `builder_oracle_sha256`, `inventory_sha256`, `quant_type`, `dequant_layout`, `output_sha256`, `output_size`, `tensor_count`, `type_counts`, `tool_version`, `general_file_type_note`.

`dequant_layout` records the conventions this build implements, each one a decision that would silently change the weights if it were flipped: `nibble_order`, `s_rel_op`, `int8_grid_rounding`, `hadamard`, `rotation_dtype`.

### Inventory

Format identifier `nz-ltx25-comfyquant-inventory-v1`, written to `<out>.inventory.json` and never into `typemap/`. Fields: `format`, `profile`, `source_size`, `source_sha256`, `config_text`, `config_bytes_sha256`, `builder_oracle_sha256`, `logical_tensors` (per row: `name`, `raw_key`, `kind`, `shape_logical`, `source_dtype`, `sidecars`, `marker`), `raw_tensors` (per row: `raw_key`, `source_dtype`, `shape`, `data_offsets`), `quant_summary` (logical and raw counts, kind and dtype histograms, and the distinct marker payloads with their occurrence counts), `source_metadata_extra`, `diagnostics`, and `inventory_sha256`, the canonical-JSON digest of everything above.

### CLI

    inspect     --model ltx25-comfyquant --st-path <src> [--inventory <path>] [--quant-type Q6_K|Q4_K] [--expect-sha256 <hex>]
    convert     --model ltx25-comfyquant --st-path <src> [--out <path>] [--quant-type Q6_K|Q4_K] [--quant-workers N] [--force] [--expect-sha256 <hex>]
    self-verify --model ltx25-comfyquant --st-path <src> [--out <path>] [--quant-type Q6_K|Q4_K] [--expect-sha256 <hex>]

`--st-path` is mandatory: the source is community-built, so `config.toml` carries no source path and no source lock for this profile. `inspect` is the dry run; `convert` performs the same admission itself, so it does not depend on a prior `inspect`. The default output is `<output_dir>/<source stem>-<quant_type>.gguf`. On `self-verify`, `--quant-type` only selects that default file name; the type policy actually verified is the manifest's.

`download`, `build-map`/`build-policy`, `extract-typemap` and `all` are not available for this profile. Each reports why on stderr and exits 1 rather than falling through to the LTX 2.3 tables; `download` in particular must not silently fetch the unrelated pinned LTX 2.3 artifact.

The `[profiles."ltx25-comfyquant"]` table holds `official_map_path`, `builder_oracle_path`, `output_dir`, `quant_workers` (default 4, CLI range 1-8) and `quant_type` (default `Q6_K`). It deliberately holds no `source_*` keys. `convert` and `self-verify` additionally accept `--map` and `--builder-oracle` to override those two paths; `inspect` accepts only `--builder-oracle` and takes the map from the profile table.

### Verification gates

Converter CI remains converter-only: no backend, Torch, or full-artifact dependency. The required evidence is:

- **Unit and fixture tests.** Hadamard symmetry/orthogonality/involution and its rejection of non-powers-of-four; the FP8 E4M3 table against an independent bit-assembly implementation for all 256 bytes, its anchor values, and the absence of infinities; round trips for both formats through an independently written forward quantizer; the proof that a deliberately swapped nibble order destroys the signal, so the round trip is a detector rather than a tautology; marker parsing and every rejection it owes; bf16 rounding equal to the shared reader; structural rejections (truncated payload, missing or orphaned sidecar, tampered sidecar shape, oracle key surplus/deficit, shape disagreement, config-digest disagreement, missing `gemma_source_checkpoint`, foreign prefix, unknown format, non-allowlisted plain dtype); a full miniature pipeline through `inspect` -> `convert` -> `self-verify` for both quant types; the absence of the source's quantization metadata from the output KV; the CLI refusals; and a SHA-256 assertion that every checked-in `typemap/` artifact is byte-unchanged.
- **Numerical comparison with the official artifact** (`scripts/compare_comfyquant_vs_official.py`, CPU-only, one tensor at a time): **G-0** coverage -- every selected quantized layer was compared (no unreadable layer, no dequantization error) and every format marker was read from the file rather than inferred from the sidecar set (`markers_assumed == 0`); a truncated source fails G-0 by design, and the script's exit code includes it; **G-A** the connector's plain BF16 rows are byte-identical to the official file; **G-B** the connector's quantized rows reach cosine similarity >= 0.99 against the official weights; **G-C** every quantized layer reaches cosine >= 0.50; **G-D** the transformer's quantized layers reach a cosine median >= 0.90 and a first percentile >= 0.60; **G-E** every layer's standard-deviation ratio lies in [0.50, 2.00]; **G-F** every layer's per-output-row norm correlation is >= 0.80. G-A establishes that the connector was carried over unmodified, which is what makes G-B a measurement of dequantization error rather than of fine-tuning.
- **Output verification**: `_verify_output`, the seven-key KV contract, the exact size assertion, the manifest, and a separate `self-verify` invocation.
- **Backend acceptance** (separate repository and stage, GPU): placing the output under the backend's LTX 2.5 weights directory, seeing it enumerated by `GET /models`, loading it through the pipeline with the Gemma-version check passing, the transformer self-test's dispose/rebuild round trip, one short fixed-seed generation compared against the official GGUF's output at the same seed, and optionally the same comparison against a `--quant-type Q4_K` build.

Measured results, and which of these gates are still outstanding, are recorded in `Docs/VERIFICATION.md`, section 8.

### Design approach

Unchanged from the rest of this document: a frozen data table plus explicit functions or if statements. No class hierarchy, generic DSL/registry, plugin system, dynamic import, or automatic profile detection. Admission is fail-closed throughout -- an unknown key, an unknown format, a missing sidecar or an unexpected count is an error, never a silently ignored input. Errors subclass `ltx25.Ltx25Error`, so the existing CLI handler prints them without a traceback.

### Terms of use

The source is a third-party redistribution of a derivative of the LTX 2.5 weights, obtained by the operator, and remains subject to the LTX-2.x Community License and to the distribution terms of the site that hosts it; observing both is the operator's responsibility. **Output produced by this profile is for the operator's own use and must not be redistributed.** comfy-kitchen (Apache-2.0) was read as a format specification and not copied; ComfyUI (GPL-3.0) was not ported.

## Open gates

1. ltx25 transformer E5/E6 evidence for approved E4 `typemap/ltx25_conversion_map.json` (SHA-256 `6d41db41a1b6b8f485434a64be198abf3f9a8b66b0dcd7441624644c19317760`): **complete as of 2026-08-21.** Output `LTX-2.5-22B-distilled-transformer.gguf`, 14,738,670,464 bytes, SHA-256 `c26473d2d5c3eb61012f60769ac4dccdb8bb77935678206c0148fff351ba060d`, 4,349 tensors (Q4_K 1,658, BF16 2,401, F32 290), inventory SHA-256 `c0966ac37f55f318be16334c1f1e1c2db1f467dafde09c0c2bd1b853206bde97`. Self-verify passed both in-run and as a separate `self-verify --model ltx25` invocation, and the E6 K-quant numerical check passed.
   This is the re-output that adds the `gemma_source_checkpoint` KV. It supersedes the first 2026-08-21 build (SHA-256 `4ead2a7dbae374639794b717517a3d1938bb8a16629957467e617ee99f22cf98`, 14,738,670,368 bytes), which lacked that KV and which the official `_check_gemma_version` would have rejected outright: with `model_version` `"2.5.0"` at or above its `(2, 4, 0)` floor and a Gemma config that does declare `gemma_version`, the absent record raises `ValueError`, not a warning. Against that earlier build the two were compared tensor by tensor: all 4,349 payloads are byte-identical, every shared KV value is unchanged, and the only difference is the one added key (the +96 bytes are its KV entry plus data-offset padding). The earlier build's E6 numbers therefore carry over unchanged.
2. gemma4-ltx25 E5/E6 evidence: **complete as of 2026-08-21.** Output `LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf`, 9,231,374,624 bytes, SHA-256 `4e69e4a33065c856e039fb4c1b78adffae582c4da14b633224136a7e7e30fb90`, 686 tensors (Q4_K 328, Q6_K 2, BF16 351, I8 5). Both self-verify and the E6 K-quant numerical check passed.
3. Separate backend implementation/acceptance after converter E1-E6 pass, including one fixed-seed representative end-to-end case.
4. Any publication/release decision remains outside this converter evidence and must not be inferred from a staging output.
