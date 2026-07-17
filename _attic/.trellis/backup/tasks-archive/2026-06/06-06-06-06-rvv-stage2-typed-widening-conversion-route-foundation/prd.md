# Stage2 RVV typed widening conversion route foundation

## Goal

Establish one bounded Stage 2 route-supported typed RVV widening conversion
proof route. A selected `tcrv.exec` RVV variant must carry either an explicit
typed `tcrv_rvv` body or a pre-realized selected body where one source vector
is loaded, signed-widened, and stored to the result buffer:

```text
active lanes: out[i] = sign_extend(lhs[i])
```

The route must flow through RVV plugin-owned selected-body realization and
route planning, provider-built `TCRVEmitCLowerableRoute`, Common EmitC neutral
materialization, and target artifact validation. Executable `ssh rvv` evidence
is in scope only if the generated-bundle path is already reachable without
turning this task into a broad dtype/LMUL matrix.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV typed widening conversion route foundation`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `469862d0 rvv: close computed-mask select route foundation`.
- No `.trellis/.current-task` existed at the start of the round, so this task
  was created before source edits.
- The previous computed-mask select task proved selected-body realization,
  provider facts, target artifact validation, generated bundle output, and real
  `ssh rvv` correctness for explicit and pre-realized paths.
- Specs require widening conversion support to be derived from selected typed
  `tcrv_rvv` body/config/runtime facts, widening conversion family plans,
  materialization facts, math operand binding, route-control facts, and an
  RVV-owned statement plan before provider route construction.
- Current source inspection shows production surfaces for widening conversion:
  `typed_widening_conversion_pre_realized_body`, `tcrv_rvv.widening_convert`,
  `RVVWideningConversionSelectedBodyRealizationOwner`, widening conversion
  route-family plan facts, provider preflight,
  conversion dtype-policy target validation, explicit/pre-realized target
  fixtures, C++ target artifact negative tests, and generated-bundle dry-run
  script tests.
- The bounded positive route shapes already present are `widen_i32_to_i64`
  and `sign_extend_widen_vf2` (`widen_i16_to_i32`).
- There is no separate explicit `widen_i16_to_i32` MLIR target fixture file;
  C++ route/target validation fixtures cover both i16->i32 and i32->i64, while
  target MLIR fixtures cover explicit i32->i64 and pre-realized i16->i32 /
  i32->i64.

## Requirements

- Keep scope to one typed widening conversion route foundation:
  `widen_i32_to_i64` / `widen_i16_to_i32` as the existing bounded route family.
- Preserve the authority chain:
  selected typed `tcrv_rvv` body/config/runtime facts -> RVV selected-body
  realization/validation -> RVV route-family provider facts -> lowerable route
  -> Common EmitC/export -> target artifact mirrors -> optional generated
  bundle runtime evidence.
- The provider/target route path must structurally carry or derive:
  - ABI roles and order `lhs,out,n`;
  - conversion kind and signed widening relation;
  - source/result element type, source/result SEW, source/result LMUL, vector
    type, C type, source-load leaf, conversion leaf, store leaf, result name,
    source memory form, and destination memory form;
  - tail/mask policy, config contract, runtime AVL/VL control plan, selected
    boundary, and route-control plan;
  - route operand-binding plan/summary with exported header/prototype
    participation markers;
  - required headers, C type mapping, target leaf profile, and
    `provider_supported_mirror`.
- Common EmitC/export may only carry provider-built route payloads and metadata
  mirrors. It must not infer conversion kind, signedness, dtype, SEW/LMUL,
  runtime ABI roles, intrinsic spelling, source/result layout, or route
  support.
- Stale or missing facts must fail closed before provider route construction or
  target artifact acceptance: conversion kind, conversion relation, source and
  result dtype/config, source/result memory form, operand binding, runtime ABI
  order, runtime AVL/VL facts, target leaf profile, header/type mapping,
  statement leaves, `provider_supported_mirror`, and cross-family stale
  mirrors.
- If production code already supports the route, finish by proving focused
  positive and negative evidence instead of adding helper-only changes.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      widening conversion route foundation and cite the relevant specs.
- [x] Focused code inspection identifies whether current production
      provider/route/target/generated-bundle support for widening conversion is
      complete or where it fails.
- [x] Positive route/artifact/header evidence proves provider-derived
      conversion kind, signedness/relation, source/result dtype, SEW/LMUL,
      vector C types, runtime ABI order, operand binding, runtime AVL/VL,
      source/result memory form, header/type/intrinsic mapping, target leaf
      profile, and `provider_supported_mirror`.
- [x] Negative evidence proves stale or missing conversion kind, relation,
      source/result dtype, source/result SEW/LMUL, operand binding, runtime
      ABI/order, runtime AVL/VL, source/result memory form, header/type/
      statement leaves, provider support mirror, and cross-family residue fail
      closed before route/artifact acceptance.
- [x] Common EmitC/export remains neutral and does not choose widening
      conversion semantics.
- [x] Focused `tcrv-opt` / `tcrv-translate` / RVV provider / target artifact
      checks for changed behavior pass.
- [x] If executable behavior is claimed, generated artifact plus real
      `ssh rvv` correctness passes over counts including `0`, `1`, a VL
      boundary, a tail case, and a multi-chunk case, with scalar oracle coverage
      for converted values and tail sentinel preservation.
- [x] If executable closure is too large, finish route-supported plus
      target-validation closure and record generated-bundle `ssh rvv` as the
      exact next continuation point. Not applicable: executable closure passed
      for supported generated-bundle modes.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor, source-front-door,
      source-artifact, route-string/artifact-name/ABI-string/test-name,
      exact-intrinsic-spelling, or Common EmitC semantic authority.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      final clean worktree, archive, and one coherent commit complete the round
      if the module behavior is complete.

## Evidence Plan

- Run focused positive target artifact checks for explicit `widen_i32_to_i64`
  and pre-realized `widen_i32_to_i64` / `widen_i16_to_i32`.
- Inspect provider preflight, route-family facts, conversion dtype-policy target
  validation, and generated-bundle script checks for production support.
- Add or repair production provider/target validation only where live evidence
  shows missing route-supported behavior or stale fact rejection.
- If production support is already real, add focused target fixture evidence for
  missing mirror fail-closed gaps and rely on existing C++ target artifact tests
  for non-textual route/statement mutation coverage.
- Extend generated-bundle dry-run and `ssh rvv` evidence only if the route
  reaches generated artifact form without broad harness work.

## Definition Of Done

- Current HEAD either has route-supported typed widening conversion provider and
  target artifact validation closure, or records the precise production blocker
  and next continuation point without false executable claims.
- Any implementation changes are production-path changes or focused evidence for
  already-existing production support; helper/test/spec-only work is not a
  completion unless production support was already real.
- Specs are updated only if this round discovers a durable rule not already
  captured in `.trellis/spec/`.
- The task is finished/archived and one coherent commit is created when the
  module behavior is complete.

## Out Of Scope

- Broad dtype/LMUL/source clone batches beyond the existing bounded i32->i64
  and i16->i32 widening conversion shapes.
- High-level Linalg, Vector, StableHLO, or source-front-door lowering.
- Reduction, MAcc, compare/select, memory movement, segment2, Gearbox
  dequantization, TensorExt, IME, Offload, performance benchmarking,
  autotuning, or dashboards.
- One-intrinsic wrapper routes.
- Descriptor-driven computation or Common EmitC semantic inference.
- Treating route ids, artifact names, ABI strings, test names, intrinsic
  spellings, status fields, manifests, or metadata mirrors as route authority.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Relevant archived task read:

- `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-computed-mask-select-merge-route-foundation/prd.md`

Primary inspection surfaces from the direction brief:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.cpp`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/*widen*i*to*i*.mlir`
- `test/Target/TargetArtifactExportTest.cpp`
- `test/Scripts/rvv-generated-bundle-abi-e2e-*widen-i*-dry-run.test`

## Completion Evidence

Completed as a bounded Stage 2 typed widening conversion route foundation.
Production support was already real in the inspected provider and target path:
`RVVEmitCRouteProvider.cpp` calls
`verifyRVVSelectedBodyWideningConversionRouteProviderFacts(...)` before
constructing the lowerable route; `RVVEmitCRoutePlanning.cpp` validates the
widening conversion family plan, materialization facts, route-control plan,
math operand binding, runtime ABI order, source/result dtype policy,
conversion relation, statement leaves, and stale cross-family facts; target
artifact validation rebuilds the conversion dtype-policy contract before
accepting provider facts and candidate mirrors.

Focused fixture change:

- `test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir`
  now positively checks `tcrv_rvv.conversion_kind =
  sign_extend_widen_vf2`.
- Added target artifact fail-closed checks for stale
  `tcrv_rvv.conversion_kind`, `tcrv_rvv.source_sew`,
  `tcrv_rvv.dest_lmul`, `tcrv_rvv.runtime_abi_order`,
  `tcrv_rvv.route_operand_binding_operands`,
  `tcrv_rvv.required_header_declarations`, and stale
  `tcrv_rvv.dequantization_route_family_plan` residue.

Provider and target route facts proved:

- Operations: `widen_i32_to_i64` and `widen_i16_to_i32`.
- Typed compute op: `tcrv_rvv.widening_convert`.
- Runtime ABI order: `lhs,out,n`.
- `widen_i32_to_i64` source/result policy:
  `i32/m1/sew32 -> i64/m2/sew64`, relation
  `signed-i32m1-to-i64m2`.
- `widen_i16_to_i32` source/result policy:
  `i16/mf2/sew16 -> i32/m1/sew32`, relation
  `signed-i16mf2-to-i32m1`.
- Target mirrors: `rvv-widening-conversion-route-family-plan.v1`,
  route operand binding plans/summaries with exported `hdr` participation,
  `stddef.h,stdint.h,riscv_vector.h`, conversion dtype-policy C type mapping,
  route-local runtime AVL/VL plan, target leaf profiles, and
  `provider_supported_mirror`.

Generated artifact evidence:

- Explicit i32->i64 dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-explicit-widen-i32-to-i64-dryrun`
- Pre-realized i32->i64 dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-widen-i32-to-i64-dryrun`
- Pre-realized i16->i32 dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-widen-i16-to-i32-dryrun`
- Direct pre-realized route-entry fail-closed:
  `stage2-direct-pre-realized-widen-i32-to-i64-fail-closed` and
  `stage2-direct-pre-realized-widen-i16-to-i32-fail-closed`

Real RVV executable evidence:

```text
PASS op=widen_i32_to_i64 counts=0,1,16,23,257
PASS op=widen_i16_to_i32 counts=0,1,16,23,257
PASS op=widen_i32_to_i64 counts=0,1,16,23,257
```

The first `widen_i32_to_i64` run was pre-realized selected-body input; the
second `widen_i32_to_i64` run was explicit selected-body input. Both i32->i64
runs checked sign extension, wide-magnitude values, two input patterns, and
tail preservation. The i16->i32 run checked sign extension, two input
patterns, and tail preservation. Explicit `widen_i16_to_i32` generated-bundle
mode is not supported by the current script/fixture set and was not expanded
in this bounded task.

Self-repair:

- Initial generated-bundle dry-run failed because `llvm-readobj` is not
  installed on this host. Re-ran dry-run and runtime evidence with
  `--llvm-readobj ''`, which keeps tcrv-opt/tcrv-translate bundle generation
  and skips only the optional local object header/symbol reader.

Focused checks:

- `build/bin/tianchenrv-target-artifact-export-test` passed.
- Positive target header export for explicit `widen_i32_to_i64` passed.
- Positive target header export for pre-realized `widen_i32_to_i64` passed.
- Positive target header export for pre-realized `widen_i16_to_i32` passed.
- New stale target mirror commands in
  `pre-realized-selected-body-artifact-widen-i16-to-i32.mlir` were manually
  checked and fail closed in `tcrv-translate --tcrv-export-target-header-artifact`.
- Configured lit filter for the changed fixture passed:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter pre-realized-selected-body-artifact-widen-i16-to-i32 .`
- `ninja -C build check-tianchenrv` completed 489/494 passing and failed on
  five unrelated existing tests:
  `Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir`,
  `Plugin/construction-protocol-common.test`,
  `Plugin/template-extension-plugin.test`,
  `Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir`,
  and `Target/RVV/pre-realized-selected-body-artifact-unit-load-strided-store.mlir`.

Old-authority scan:

- Added-line scan over the touched target fixture produced no matches for
  legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact, descriptor,
  direct-C/source-export, exact intrinsic spelling, or common EmitC semantic
  authority patterns.
