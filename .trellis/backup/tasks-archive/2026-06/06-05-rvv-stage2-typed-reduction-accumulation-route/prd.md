# Stage2 RVV typed reduction/accumulation route foundation

## Goal

Add one route-supported typed RVV reduction/accumulation foundation on the
corrected `tcrv_rvv` surface. The proof route is a simple sum-style reduction
that flows from a selected `tcrv.exec` RVV variant, through an explicit typed
`tcrv_rvv` reduction body, into RVV plugin-owned legality and route planning,
then through `TCRVEmitCLowerableRoute` and target artifact validation.

This is a Stage 2 compiler capability-class task. It is not a Gearbox,
benchmark, metadata-only, source-front-door, or common EmitC semantics task.

## What I already know

- No `.trellis/.current-task` existed when this round began, so this task was
  created from the Hermes Direction Brief.
- Commit `77d54e81` closed the Gearbox dequant executable ABI path; that module
  should not be expanded as the main work in this round.
- The stable RVV authority chain is `tcrv.exec` selected variant -> typed
  low-level `tcrv_rvv` body -> RVV plugin legality/realization/route provider
  -> `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles only. Reduction compute, dtype, SEW/LMUL, policy, accumulator layout,
  and route facts must come from typed `tcrv_rvv` structure and RVV provider
  validation.
- Current Stage 2 reduction work must not introduce dtype-prefixed helper
  families such as `tcrv_rvv.i32_reduction_*` or new `!tcrv_rvv.i32m*`
  authority.
- If executable closure is too large, the accepted completion boundary is
  route-supported plus target-validation closure, with `ssh rvv` correctness
  left as the explicit next continuation point.
- Live repo inspection showed that a typed standalone reduction surface,
  selected-body realization, provider route planning, target artifact
  validation, and several positive/negative reduction fixtures already existed.
  This round therefore closed the missing representative plain standalone
  reduction-kind mirror contract and stale-mirror proof instead of duplicating
  the route family.

## Requirements

- Add production dialect/config/construction/provider/target behavior for one
  coherent sum-style typed RVV reduction/accumulation route.
- The selected `tcrv_rvv` body must structurally carry:
  - reduction kind;
  - source element type and result/accumulator element type;
  - SEW and LMUL;
  - accumulator/result layout;
  - VL/tail policy;
  - runtime AVL use;
  - memory/value role bindings.
- The RVV provider must derive legality, route type, statement plan, required
  headers, C/RVV type facts, operand binding summary, and route metadata from
  typed body/config/capability/runtime facts.
- Common EmitC may only materialize provider-supplied payloads. It must not
  infer reduction semantics, dtype, policy, accumulator layout, intrinsic
  spelling, or runtime ABI semantics.
- Target artifact validation must compare provider-derived reduction facts
  against mirrored candidate metadata before accepting the bundle.
- Unsupported reduction kind, accumulator/result layout, dtype/config,
  VL/tail policy, missing runtime AVL, invalid memory/result binding, stale
  mirrors, and route-string/artifact-name/intrinsic-spelling authority must
  fail closed with focused diagnostics.
- Keep the route narrow: one proof reduction kind and one accumulator/result
  form are enough for this round.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      reduction/accumulation task and its spec basis.
- [x] Production code changes exist in provider route planning and target
      validation/export surfaces. Existing dialect/config/construction code
      already provided the typed standalone reduction body and verifier
      foundation used by this round.
- [x] A selected-body positive test proves that route support comes from the
      typed reduction body structure, not route ids, artifact names, test names,
      ABI strings, descriptors, or intrinsic spelling.
- [x] Provider tests or FileCheck coverage prove the derived reduction route
      facts: reduction kind, source/result type facts, SEW/LMUL, tail policy,
      accumulator/result layout, runtime AVL, required headers, type mapping,
      operand binding, and statement plan.
- [x] Negative tests prove fail-closed diagnostics for unsupported reduction
      kind, stale accumulator/type/config mirrors, missing runtime AVL, invalid
      memory/result role binding, and stale target mirrors.
- [x] Target artifact validation rejects stale reduction metadata mirrors and
      accepts the provider-derived route-supported proof case.
- [x] Focused `tcrv-opt` / `tcrv-translate` / RVV plugin / target artifact tests
      for the changed behavior pass.
- [x] Bounded scans over touched files show no new active legacy i32 route
      authority, q-name authority, descriptor authority, source-front-door
      positive route, or common EmitC RVV semantic branch.
- [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are recorded.
- [x] If `ssh rvv` runtime correctness is not claimed, the next continuation
      point names the missing executable closure explicitly.

## Out of Scope

- High-level Linalg/Vector/StableHLO frontend lowering.
- Per-Linalg-op route authority.
- Broad dtype, LMUL, reduction-kind, or accumulator-layout matrices.
- One-intrinsic wrapper dialects or dtype-prefixed helper op families.
- q-name, benchmark-name, artifact-name, ABI-string, route-string, descriptor,
  source-front-door, or intrinsic-spelling authority.
- Common EmitC reduction semantics.
- Gearbox/cache/benchmark expansion as the main task.
- Runtime correctness or performance claims unless real `ssh rvv` evidence is
  collected in this same round.

## Technical Notes

- Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`.
- Previous task context:
  `.trellis/tasks/archive/2026-06/06-05-rvv-gearbox-dequant-executable-abi-closure/`.
- Main files to inspect and likely change:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/`,
  `lib/Plugin/RVV/Construction/`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  and nearby route-supported tests for dequant, memory, select, and conversion.

## Decision

Use one sum-style reduction proof route with one accumulator/result layout as
the MVP. Finish route-supported provider planning plus target artifact
validation in this round. Treat generated artifact and `ssh rvv` executable
closure as optional follow-up evidence unless the route materialization path is
already cheap to close after the focused tests pass.

## Completion Evidence

- Production change: standalone reduction route facts now derive and carry
  `tcrv_rvv.reduction_kind` (`add`, `min`, `max`, or
  `signed_widening_reduce_add`) from provider-owned typed body/operation facts.
- Production change: `RVVSelectedBodyEmitCRouteDescription`,
  `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`,
  `RVVStandaloneReductionRouteFacts`, and
  `RVVStandaloneReductionRouteValidationContract` carry the provider-derived
  reduction kind.
- Production change: emission-plan metadata and target header export now mirror
  `tcrv_rvv.reduction_kind`; target artifact validation rejects stale
  candidate mirrors before bundle acceptance.
- Production change: `RVVTargetSupportBundle.cpp` allows the new
  `reduction_kind` header metadata key as an optional provider mirror.
- Test change:
  `test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-add.mlir`
  checks positive emission-plan/header `reduction_kind = add` and stale target
  mirror rejection for reduction kind, accumulator layout, route binding, and
  type mapping.
- Spec update:
  `.trellis/spec/lowering-runtime/emitc-route.md` now records the executable
  RVV standalone reduction mirror contract.
- Focused checks run:
  `cmake --build build --target tcrv-opt tcrv-translate`;
  manual FileCheck for PLAN, HEADER, STALE-REDUCTION-KIND,
  STALE-REDUCTION-ACC, STALE-REDUCTION-BINDING, and STALE-REDUCTION-TYPE;
  `build/bin/tcrv-opt test/Dialect/RVV/standalone-reduction-dataflow.mlir
  --split-input-file --verify-diagnostics`;
  `cmake --build build --target tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `git diff --check`; `git diff --cached --check`.
- `ctest --test-dir build -N` reported `Total Tests: 0`, so focused manual
  FileCheck and the target artifact test binary were used instead of CTest.
- `ssh rvv` was not run because this round claims route-supported plus target
  validation closure only, not executable correctness or performance.
- Bounded old-authority scan over touched files found no new positive legacy
  authority in added lines; the only added match was the negative
  `artifact-name-derived-vector` stale-mirror test string.
- Next continuation point if desired: generated artifact and real `ssh rvv`
  correctness closure for the standalone reduction route.
