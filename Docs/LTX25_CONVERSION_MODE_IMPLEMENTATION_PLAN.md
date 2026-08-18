# LTX 2.5 conversion mode implementation plan

## Scope and non-negotiable guardrails

Implement only [LTX25_CONVERSION_MODE_SPEC.md](LTX25_CONVERSION_MODE_SPEC.md). The converter phases in this plan have completed through staged E1-E6 evidence; the plan still does not authorize backend, Torch, model-registry, VAE, Gemma, audio, LoRA, UI, or generic framework work.

The design is two frozen entries, ltx23 and ltx25, plus explicit functions/if statements. No class hierarchy, profile registry/DSL, plugin framework, dynamic import, auto-detection, quarantine, journal, database, or large benchmark suite.

At each future implementation phase check git status --short and preserve unrelated work/user artifacts. Legacy ltx23 remains the default. Its all and extract-typemap commands are unchanged and ltx23-only.

## Phase 0: freeze ltx23 behavior

1. Capture converter help, legacy config, LTX 2.3 typemap digest/type counts, and test baseline.
2. Characterize omitted model and explicit ltx23 as identical in resolved settings and output behavior.
3. Keep legacy source/reference/output config and reference-GGUF typemap path intact.

This is necessary because LTX 2.3 currently has global prefix/skip logic at src/converter/convert.py:65-70 and exact global config resolution at src/converter/cli.py:139-170 and 265-290.

Exit criterion: no existing default command changes semantics or output name.

## Phase 1: complete E1 source lock before enabling ltx25

1. Create ltx25 disabled profile data using the known E1 values: repository Lightricks/LTX-2.5; artifact revision dd53cc2cd45bbeaa3563dfb575cba3f49cf44761; path diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors; size 42,018,190,584; blob OID 3ba48d13c75fd5df2e868aa8c07b0c5119a6116f; auto gating; ltx-2-community-license-agreement.
2. Do not use the blob OID, a URL, or VirusTotal as SHA-256.
3. Completed: with accepted official terms/authenticated access, the artifact was verified at the exact local size and full SHA-256 `31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4`, matching its Hub LFS/Xet content identity.
4. Store this as E1 and only then allow ltx25 inspect. Existing downloader has size/digest mechanics at src/converter/download.py:71-86 and 127-169; LTX 2.5 must require a digest on both download and explicit local-file paths.
5. Add clear source-lock-incomplete error tests. No assert guards source identity.

Exit criterion: E1 is complete, reproducible, and distinguishes artifact revision, Git blob OID, and content SHA-256. Status: complete.

## Phase 2: pin E2 official builder oracle

1. Pin official source to 400fd31054597515f47125691032c04b1c3ee24e. Treat current repository SHA 6c7e5e573ac1667efc83407806fe9b0b93730e60 as reference-only.
2. Before generating E3, confirm the artifact release and E2 code are compatible.
3. Use official model_loader, single_gpu_model_builder, sft_loader, helpers, transformer model_configurator/model, and sd_ops sources listed in the specification.
4. Parse metadata config and construct the official direct meta builders. Record the equivalent one-strip state-dict handoff as fixed component membership, builder key, and logical shape. Do not claim that this small E2 artifact is a module-role or dtype oracle.
5. Emit canonical E2 oracle to `typemap/ltx25_builder_oracle.json`: pinned official-code commit, config-byte SHA-256, component/key/logical-shape table, and canonical oracle SHA-256. The converter validates this static artifact without importing Torch. E3 safetensors headers own the per-key source-dtype allowlist. Do not use the existing backend or an external GGUF as either oracle.
6. Preserve config bytes; parse only a copy. Confirm top-level config.transformer. Record actual supplied fields versus official-code defaults rather than baking defaults into the checkpoint profile.

Exit criterion: E2 contains the exact official builder key/shape set for the E1 artifact config.

## Phase 3: header-only E3 extraction and admission

Add an ltx25 inspect command, blocked unless E1 is complete. It reads safetensors header only and writes E3; it does not convert tensors.

1. Validate header length, JSON/duplicate JSON keys, metadata type, names, shapes, offsets, overlap/bounds, and dtype-times-shape byte sizes.
2. Require metadata config JSON with config.transformer. Retain its original UTF-8 bytes.
3. Classify each raw key exactly once: after the one model.diffusion_model strip, emit transformer keys and exclude the two E2-confirmed Gemma connector prefixes with explicit component IDs; otherwise error. The artifact has config.transformer and no vae/audio_vae/vocoder.
4. Compare emitted key and logical shape set exactly with E2. Error on missing, extra, duplicate, unknown, multi-match, collision, or shape mismatch.
5. Require every row to match its E2 key/shape and its authenticated E3 per-key dtype (emitted: 3,801 BF16 + 290 F32; both connectors: BF16). The E3 dtype profile and E1 pin are authority. Names/metadata containing int8, convrot, fp8, or quant are diagnostic hints only.
6. Emit compact E3 records and canonical hashes. Do not create detailed duplicate histograms or redundant report families.

Exit criterion: authenticated official header proves source identity, three-component handling, the E3 BF16/F32 profile, and exact E2 builder inventory. Status: complete (4,091 emit; 258 explicit Gemma connector excludes).

## Phase 4: create E4 concrete map

1. Generate a draft JSON/YAML concrete map from E2 plus E3 at a dedicated draft path; never call legacy extract-typemap and never import a third-party map. CLI generation must reject the configured approved path and never self-approves it: promote only a separately reviewed checked-in E4 artifact.
2. Each row is name, source_dtype, shape_logical, shape_gguf, ggml_type, nbytes, rule_id, reason, evidence_ids.
3. Runtime uses direct map lookup. Each emitted tensor has exactly one exact dtype/shape matching row.
4. Reject unknown/missing/duplicate/multiple rows, unsupported type, incorrect bytes, or invalid K alignment with domain errors, not asserts.
5. E3 established 3,801 BF16 and 290 F32 emitted transformer rows. Approved E4 preserves F32 tables, biases, RMSNorm weights, the keyframe position parameter, and the two 128-wide patchify boundary weights; it assigns Q4_K to 1,658 concrete, K-row-aligned `torch.nn.Linear` weights. The supporting `named_modules` role audit is E4 reviewer evidence, not E2 data. Keep no Q5_K/Q6_K exceptions.

Exit criterion: E4 is complete, deterministic, independently reviewable, and has evidence IDs for every row.

## Phase 5: conversion and output protocol

1. Reuse existing two-pass streaming writer and K kernels. Current sequence registers infos, writes header/KV/tensor infos, then streams one payload at a time at src/converter/convert.py:369-398.
2. Validate config copy against E2 builder requirements while preserving original config bytes in GGUF config. General architecture metadata is a project adapter discriminator, not an official LTX 2.5 statement.
3. Before output, compute exact E4 payload plus GGUF framing estimate and preflight available disk. Fail with required/available numbers before opening temporary output when insufficient.
4. Write a unique temporary GGUF in final directory. Run E5 static verification entirely on that temporary, then close it.
5. Commit only by os.replace(temp_gguf, final_gguf), then create final_gguf.manifest.json. The GGUF and manifest are not a two-file atomic transaction.
6. If manifest creation fails/mismatches, ltx25 verify fails. Preserve existing final on failure before replace. Attempt current-run temp cleanup; use one short bounded PermissionError retry, report surviving path, and do not later auto-delete it.
7. Manifest is minimal: profile, E1 source SHA-256, E3 inventory SHA-256, E4 map SHA-256, E5 output SHA-256, counts, tool version.

Exit criterion: final GGUF only replaces after temp E5 success; manifest failure is visible and never repaired silently.

## Phase 6: CLI and verification

1. Add model option to profile-aware commands with ltx23 default. Never infer profile.
2. Keep all/extract-typemap ltx23-only. After E4 review, LTX 2.5 user flow is inspect, convert, self-verify. Treat build-map as a maintainer-only draft-evidence operation. Reject ltx25 all/extract-typemap with a concise guided error.
3. Verify ltx25 from E1/E3/E4/E5/manifest static data: source/map/inventory/output hashes; count; exact keys/shapes/types; config byte preservation; allowed GGML types; K finite/dequant checks.
4. Preserve current LTX 2.3 reference-GGUF verifier, whose structural checks are at src/converter/verify.py:115-255 and 294-327.
5. Use stable errors: source-lock-incomplete, source-rejected, inventory-mismatch, map-mismatch, disk-preflight-failed, manifest-missing. Expected validation errors are concise and do not emit tracebacks.

## Phase 7: converter-only tests and gated acceptance

| Area | Required coverage |
| --- | --- |
| LTX 2.3 regression | Existing suite; omitted model equals explicit ltx23; legacy all/extract behavior unchanged. |
| E1 | Source-lock-incomplete; exact size/digest success and mismatch; no assert dependency. |
| Header/E3 | Header/JSON/duplicate/name/shape/offset/bounds/byte/config failures; exact-one classification and one-strip normalization. |
| Admission | E3-matched BF16/F32 success; F16/INT8/FP8/NVFP4 or any per-key dtype mismatch rejection. Marker strings remain diagnostic-only. |
| E2 handoff | Exact component key/shape compare against a small static oracle fixture; E3 owns dtype comparison; no backend/Torch dependency in normal converter CI. |
| E4 | Direct map lookup; duplicate/missing/unknown/multiple; exact dtype/shape/bytes/K alignment. |
| Writer/E5/E6 | Synthetic F32/BF16/Q4_K/Q5_K/Q6_K as selected; current numerical tests; finite dequant; config byte preservation. |
| Output | Disk calculation; temporary/final ordering; PermissionError retry/report; missing/mismatched manifest makes verify fail. |

Large authenticated conversion runs only in its gated job: E1 then E2/E3 then E4 then conversion/E5/E6. It does not run in normal CI.

## Separate backend stage and rollback

Only after converter E1-E6 acceptance, a separate backend repository plan may implement LTX 2.5 construction and run one representative fixed-seed end-to-end case. That stage checks 16 GB per-layer behavior; no broad benchmark campaign is required. The existing backend raw-key/no-remap behavior at ../Nz-LTX23-backend/engine/gguf/quant_service.py:617-621 is a static handoff constraint, not proof of LTX 2.5 compatibility.

Rollback means disable ltx25. Do not move, remove, quarantine, or relabel user outputs. Re-pin, regenerate E2-E6, and re-gate any corrected artifact.
