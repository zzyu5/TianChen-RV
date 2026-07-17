# Stage2 RVV typed conversion and SEW-policy route foundation

## Goal

Make the selected-body RVV conversion/dtype-policy route boundary more
explicit and fail-closed for the existing widening conversion proof class.
The production owner is the path from selected `tcrv.exec` RVV variant with
typed low-level `tcrv_rvv` widening conversion body/config/runtime facts,
through RVV plugin-owned legality and provider-derived route facts, into
target artifact validation. Common EmitC/export remains neutral and must not
choose conversion semantics, dtype, SEW, LMUL, policy, ABI roles, or intrinsic
families.

## What I Already Know

- The repository starts on `main` at `567bbcf1 rvv: close runtime-strided
  memory executable abi`.
- The worktree has one unrelated untracked note:
  `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
- `.trellis/spec/index.md` requires the RVV authority chain to flow through
  selected typed `tcrv_rvv` body, RVV plugin legality/realization/provider,
  common `TCRVEmitCLowerableRoute`, neutral EmitC materialization, target
  artifact, and `ssh rvv` evidence only when runtime correctness/performance
  is claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` places dtype/config,
  conversion, memory form, VL/AVL, and runtime ABI consumption in the
  selected typed RVV body plus RVV plugin. Route ids, artifact names, ABI
  strings, intrinsic spellings, status fields, and common EmitC/export are not
  authority.
- `.trellis/spec/lowering-runtime/emitc-route.md` already defines a widening
  conversion fact surface and a conversion dtype-policy route validation
  contract for `widen_i32_to_i64` and `widen_i16_to_i32`, with embedded
  runtime AVL/VL selected-boundary validation and mirror-only metadata checks.
- Existing IR has `tcrv_rvv.typed_widening_conversion_pre_realized_body` and
  realized `tcrv_rvv.widening_convert`.
- Existing verifiers already reject unsupported conversion kinds, stale
  source/destination SEW/LMUL relation, unsupported memory form, non-agnostic
  policy, mismatched runtime ABI C types, missing runtime count role, and
  authority metadata such as `route_id`.
- Existing fixtures include positive and negative dialect/lowering tests plus
  generated-bundle dry-run tests for `widen_i32_to_i64` and
  `widen_i16_to_i32`.
- The immediately previous archived task was runtime-strided memory executable
  evidence closure. This task must not continue that evidence-only direction
  as the primary deliverable.

## Requirements

- Use one coherent proof conversion class: the existing signed widening
  conversion surface, covering `widen_i32_to_i64` and preserving the already
  adjacent `widen_i16_to_i32` contract where touched.
- Keep conversion authority in typed body/config/runtime facts and provider
  facts. Retained code must be generic over conversion dtype-policy facts, not
  a new dtype/LMUL clone batch.
- RVV provider-side route facts must be the source for route id, source/result
  element type, source/result SEW/LMUL, conversion kind/relation, memory form,
  tail/mask policy, runtime ABI order, operand binding, headers, C type
  mapping, statement plan, and intrinsic leaves.
- Target artifact validation must consume the provider conversion
  dtype-policy validation contract and reject missing, stale, ambiguous, or
  cross-family mirror facts before accepting a bundle.
- Common EmitC/export must only carry provider payloads and mirrors; it must
  not infer conversion semantics from strings, artifact names, ABI strings, or
  intrinsic spellings.
- Add or repair focused positive/negative tests for provider/target contract
  behavior and generated conversion route metadata.
- Preserve the existing runtime-strided memory work; do not claim new runtime
  correctness unless new `ssh rvv` evidence is collected.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` truthfully describe this
      bounded conversion/dtype-policy route task and its spec basis.
- [x] Production C++ changes strengthen the existing conversion dtype-policy
      route foundation. A report-only, prompt-only, helper-only, or metadata-
      only change is not sufficient.
- [x] Positive provider/target tests prove the conversion contract is
      accessible for selected widening conversion routes and absent for
      non-conversion routes.
- [x] Negative tests prove fail-closed behavior for stale source/destination
      type or SEW/LMUL facts, unsupported/ambiguous conversion kind or
      relation, stale runtime AVL/VL mirrors, stale header/type summaries,
      stale operand binding summaries, stale provider support/target profile
      mirrors, and stale non-conversion route-family mirrors.
- [x] Focused generated artifact or lit/manual pipeline checks show the route
      emits provider-derived conversion metadata such as
      `provider_supported_mirror`, `source_sew`, `dest_sew`,
      `conversion_kind`, `conversion_relation`, `required_header_declarations`,
      and `c_type_mapping`.
- [x] Focused checks include `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test` when provider/target code is
      changed, plus the relevant lit/script fixtures if test tools are
      available.
- [x] Bounded old-authority/q-name scan over touched files shows no new
      positive legacy i32 route authority, q8/q4 route authority, descriptor,
      source-front-door, route-id, artifact-name, ABI-string, intrinsic, or
      C-string semantics.
- [x] `git diff --check`, `git diff --cached --check`, and final status are
      recorded. Commit only after the task is complete.

## Out Of Scope

- q8/q4/llama benchmark routes.
- New runtime-strided memory executable evidence as the primary deliverable.
- High-level Linalg/Vector/StableHLO frontend work.
- Broad dtype/LMUL matrices or one-intrinsic wrapper dialects.
- Compatibility wrappers preserving legacy i32m1 authority.
- Source-front-door positive routes.
- Conversion semantics selected by common EmitC/export, target-local tables,
  route strings, artifact names, ABI strings, or intrinsic spellings.
- Dashboards, autotuning databases, broad smoke matrices, and report-only
  tasks.

## Technical Approach

1. Inspect the existing conversion fact accessors, conversion validation
   contract, target route-family validator, C++ tests, and generated-bundle
   fixtures.
2. Identify the smallest production gap that keeps target validation consuming
   provider-owned conversion facts instead of target-local or mirror strings.
3. Implement the gap in C++/MLIR/TableGen/lit as appropriate.
4. Add focused positive and negative tests for the changed contract behavior.
5. Run focused C++/lit/script checks, repair failures, and record the exact
   commands.
6. Update task notes/status, archive only if complete, and create one coherent
   commit.

## Completion Evidence

- Production owner:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` now formats
  conversion dtype-policy runtime ABI count diagnostics from the provider
  contract instead of hard-coding the widening-only `lhs,out,n` list. This
  keeps dequantization's `lhs,scale,out,n` ABI boundary provider-derived.
- Target contract coverage:
  `test/Target/TargetArtifactExportTest.cpp` now builds a direct selected-body
  `dequantize_i32_to_f32` typed `tcrv_rvv` fixture, validates it through the
  target artifact exporter, rebuilds provider route validation inputs, and
  proves the same conversion dtype-policy target consumer accepts provider
  facts and candidate mirrors.
- Positive facts checked:
  the C++ test asserts dequantization facts from
  `getRVVDequantizationRouteFacts(DequantizeI32ToF32)`, including route-family
  plan, typed compute op `tcrv_rvv.dequantize`, conversion kind
  `i32_to_f32_scaled`, relation `signed-i32m1-to-f32m1-scale-f32`, convert and
  scale intrinsics, scale role/type/name, runtime ABI order, runtime ABI
  parameters, and five loop-body statement steps.
- Negative diagnostics checked:
  missing dequant scale ABI parameter, stale dequant scale role, stale
  widening conversion facts on a dequantization route, stale route scale
  operand, stale candidate scale-role mirror, and stale widening candidate
  mirror all fail closed before target artifact acceptance.
- Spec update:
  `.trellis/spec/lowering-runtime/emitc-route.md` now records that
  `dequantize_i32_to_f32` is covered by the same conversion dtype-policy target
  consumer, with runtime ABI order `lhs,scale,out,n`, scale facts, and required
  dequant stale-mirror tests.
- Self-repair:
  the first missing-runtime-ABI negative test deleted runtime `n`, which
  correctly triggered the earlier runtime AVL/VL contract instead of the new
  ABI-list diagnostic. The test was corrected to delete `scale` while
  preserving `n`, proving the intended provider-contract ABI list.
- Checks run:
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tcrv-opt test/Dialect/RVV/generic-widening-conversion-dataflow.mlir --split-input-file --verify-diagnostics`;
  `build/bin/tcrv-opt test/Transforms/LoweringBoundary/rvv-pre-realized-widening-conversion-negative.mlir --split-input-file --verify-diagnostics --tcrv-materialize-selected-lowering-boundaries`;
  `build/bin/tcrv-opt test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact`;
  `rg` over the generated dequant header artifact for relation, convert/scale
  intrinsic, provider mirror, operand binding, required headers, and C type
  mapping; bounded added-diff old-authority/q-name scan; `git diff --check`.
- Runtime evidence:
  no `ssh rvv` run was needed or claimed; this task is route-supported target
  validation coverage, not executable correctness or performance evidence.
