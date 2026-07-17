# Stage2 RVV low-precision widening-product route foundation

## Goal

Implement one production route-supported RVV low-precision contraction
primitive foundation: selected typed `tcrv_rvv` signed i8 vector load/config
facts plus i8 x i8 widening product to i16 route support. The route must be
owned by the RVV plugin/provider and validated by target artifact logic. This
is not a q8/q4 benchmark route and not an evidence-recording-only task.

## What I Already Know

- The repository has no active Trellis task at session start; this task was
  created from the Hermes Direction Brief.
- Current mainline is RVV-first: `tcrv.exec` envelope -> selected typed
  `tcrv_rvv` body -> RVV plugin legality/realization/provider route ->
  common EmitC -> target artifact.
- Specs require dtype/config/operation facts to be structural in typed
  `tcrv_rvv` body/config/runtime use, not inferred from q8 names, route ids,
  artifact names, test names, descriptor residue, or metadata mirrors.
- Existing nearby surfaces include RVV config contracts, contraction route
  family planning/owners, route planning/provider code, target artifact route
  validation, and focused widening/contraction tests.

## Scope

- Add or repair the production RVV route-supported slice for signed i8 vector
  load/config plus i8 x i8 widening product to i16.
- Keep the route at low-level RVV execution-body level; do not add high-level
  Linalg/frontend lowering or benchmark-specific q8/q4/llama authority.
- Derive provider route facts from typed body/config/runtime facts:
  element type, signedness, SEW=8, LMUL, tail/mask policy, runtime AVL/VL,
  load memory form, operation kind, and widening i16 result type.
- Validate target artifact mirrors only after provider route construction and
  fail closed on stale or mismatched low-precision mirrors.

## Requirements

- Production code changes are required before closure.
- Typed body/config validation accepts the selected signed i8 source/vector
  facts only when element type, signedness, SEW, LMUL, policy, runtime AVL/VL,
  load memory form, and widening product result type are structurally present.
- RVV provider route planning derives load/product C types, vector types,
  headers, and `vwmul`-style intrinsic facts from typed body/config facts.
- Common EmitC/export may carry provider-built payloads but must not choose
  RVV dtype, vector type, intrinsic spelling, or operation semantics.
- Target validation rejects stale or mismatched low-precision provider
  descriptions and candidate metadata mirrors fail-closed.
- Focused positive and negative tests cover route-supported behavior and
  unsupported signedness/SEW/result-type or stale metadata paths.

## Acceptance Criteria

- [x] A production RVV dialect/plugin/target route or validation surface is
  changed to support the bounded signed i8 widening-product route foundation.
- [x] Positive focused test proves provider route-supported behavior for signed
  i8 unit-load plus i8 x i8 widening product to i16.
- [x] Negative focused tests prove mismatched signedness, SEW, result type, or
  stale low-precision metadata fail closed where those paths exist.
- [x] `tianchenrv-rvv-extension-plugin-test` passes if provider/plugin code
  changes.
- [x] `tianchenrv-target-artifact-export-test` passes if target validation code
  changes.
- [x] Focused lit/unit tests for the changed route pass.
- [x] `git diff --check` passes.
- [x] Bounded scan over touched files finds no new q8/q4/name-authority or
  legacy i32m1 positive route authority.
- [x] Worktree is clean after the final commit.

## Out Of Scope

- q8/q4/llama benchmark-specific route names, authority, or harnesses.
- i16-to-i32 widening reduction, dequantization, or full dot-product closure;
  those remain explicit continuation work unless the bounded slice proves small.
- Runtime/correctness/performance claims without real `ssh rvv` evidence.
- High-level Linalg/frontend generalization.
- Descriptor-driven computation, source-front-door positive routes, or
  compatibility wrappers preserving legacy i32m1 authority.
- Broad smoke matrices, dashboards, report-only tasks, or Trellis archive-only
  updates.

## Technical Notes

- Required specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- First code inspection targets:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  and nearby widening/contraction tests.

## Completion Evidence

- Added bounded signed i8 widening-product surface:
  `tcrv_rvv.widening_product` consumes two `!tcrv_rvv.vector<i8, "mf4">`
  values and selected VL, validates `signed_widening_product` plus
  `signed-i8mf4xi8mf4-to-i16mf2`, and produces
  `!tcrv_rvv.vector<i16, "mf2">` under selected SEW16/LMUL mf2 config.
- Added RVV config/runtime ABI contracts for SEW8 source facts, SEW16/MF2
  result facts, and `lhs,rhs,out,n` ABI with `const int8_t *` inputs and
  `int16_t *` output.
- Added RVV plugin-owned route planning, contraction family facts, route
  control, operand binding, statement planning, and construction protocol
  branches for the low-precision widening-product route. Provider-derived
  facts carry `__riscv_vle8_v_i8mf4`, `__riscv_vwmul_vv_i16mf2`,
  `__riscv_vse16_v_i16mf2`, `__riscv_vsetvl_e16mf2`, `vint8mf4_t`, and
  `vint16mf2_t`.
- Added target artifact validation for the `low-precision-widening-product`
  route family, including fail-closed checks for stale provider facts, stale
  route payloads, stale candidate mirrors, and masked/dot/macc residue.
- Added focused dialect verifier lit coverage and C++ target export tests for
  positive product route support plus negative kind, relation, source type,
  result type, result config, provider fact, route payload, and metadata mirror
  mismatches.

## Checks Run

- `rtk build/bin/tianchenrv-target-artifact-export-test`
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- `rtk build/bin/tianchenrv-construction-protocol-common-test`
- `rtk bash -lc 'build/bin/tcrv-opt test/Dialect/RVV/generic-widening-product-dataflow.mlir --split-input-file --verify-diagnostics | /usr/lib/llvm-20/bin/FileCheck test/Dialect/RVV/generic-widening-product-dataflow.mlir'`
- `rtk bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter generic-widening-product-dataflow'`
- `rtk git diff --check`
- `rtk bash -lc 'git diff -U0 | rg -n "q8|q4|llama|RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|__riscv_.*_i32m1"'`

## Continuation

This round stops at route-supported signed i8 unit-load plus i8 x i8 widening
product to i16 and target artifact validation. It does not claim runtime
correctness/performance and does not add i16-to-i32 widening reduction,
dequantization, or full q8/q4 contraction closure.
