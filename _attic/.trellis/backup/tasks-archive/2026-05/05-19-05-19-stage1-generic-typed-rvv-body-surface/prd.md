# Stage1 legacy RVV route-authority retirement on the generic typed surface

## Goal

Continue the in-progress Stage 1 task after commit `58190e2`: keep the generic
typed `tcrv_rvv` body skeleton as the only positive arithmetic route authority
and retire the remaining legacy arithmetic paths that still treat finite
`i32m1` helper ops, helper names, source-front-door construction, or old
positive fixtures as executable RVV route authority.

```text
tcrv.exec selected RVV variant
  -> explicit tcrv_rvv.runtime_abi_value imports
  -> tcrv_rvv.setvl / tcrv_rvv.with_vl config and VL control
  -> generic typed vector load / binary{kind} / store body
  -> RVV provider-derived TCRVEmitCLowerableRoute
  -> neutral common EmitC / target artifact mechanics
```

This round is Stage 1 route-authority replacement. It is not Stage 2 coverage
growth. The retained i32 add/sub/mul case may stay executable only as an
ordinary specialization of the generic typed surface.

## Current Facts

- HEAD `58190e2` added `!tcrv_rvv.vector<i32, "m1"|"m2">`, generic
  `tcrv_rvv.load`, `tcrv_rvv.binary`, and `tcrv_rvv.store`, plus focused
  positive generic materialization and target artifact fixtures.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` still has an explicit
  compatibility path that recognizes `tcrv_rvv.i32_add`,
  `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul` classes as positive arithmetic
  routes.
- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` still contains a legacy
  source-front-door materializer that constructs `tcrv_rvv.i32_*` selected
  bodies instead of failing closed during Stage 1.
- Old positive EmitC/target fixtures still exercise `tcrv_rvv.i32_*` bodies
  for add/sub/mul object/header generation. They must be migrated to the
  generic typed surface or converted into negative/deprecated inventory.
- Legacy `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `rvv-i32m1-*`, and exact
  intrinsic spellings may remain only as parse/verify coverage, fail-closed
  diagnostics, or mirror strings after provider construction. They must not
  select the positive production arithmetic route.

## Requirements

1. Keep retained add/sub/mul executable support on the generic
   `tcrv_rvv.load` -> `tcrv_rvv.binary {kind}` -> `tcrv_rvv.store` surface.
2. Remove or fail-close provider arithmetic recognition based on
   `I32AddOp`, `I32SubOp`, `I32MulOp`, `tcrv_rvv.i32_*` helper names, or
   `rvv-i32m1-*` identifiers.
3. Make source-front-door RVV Stage 1 behavior fail closed with a targeted
   diagnostic instead of constructing a positive `tcrv_rvv.i32_*` body.
4. Migrate old positive add/sub/mul EmitC and target fixtures to generic typed
   bodies, or convert them into explicit negative/deprecated tests.
5. Do not add broadcast, compare/select, reductions, conversions, new dtype
   families, new LMUL batches, high-level frontends, or source-front-door
   positive routes in this round.
6. Common EmitC/materializer/target export must remain neutral. RVV-specific
   semantic decisions stay in RVV dialect/provider/target-owned support leaves.
7. Unsupported or stale legacy combinations must fail closed before target
   artifact construction.

## Acceptance Criteria

- Positive lit/FileCheck coverage proves selected RVV add/sub/mul routes using
  the generic typed `tcrv_rvv` load/binary/store surface reach provider-built
  route metadata and target header or artifact output.
- The provider no longer has a positive arithmetic route selected by finite
  `tcrv_rvv.i32_add/sub/mul` operation classes, helper op names, or
  `rvv-i32m1-*` identifiers.
- Route description validation checks generic body-derived operation kind,
  vector type/config facts, memory form, runtime ABI values, and intrinsic/type
  leaves.
- Negative coverage proves stale legacy `tcrv_rvv.i32_*` arithmetic bodies,
  source-front-door inputs, and mismatched generic body/config facts fail closed
  before artifact construction.
- An active-authority scan over `include lib test .trellis/spec` classifies
  remaining `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, `__riscv_*_i32m1`, source-front-door/source-artifact, and
  emission-plan matches as deprecated/fail-closed/negative-test inventory or
  mirrors, not the production route authority completed here.
- Focused build/lit/C++ checks for touched RVV dialect/provider/target behavior
  pass.

## Non-Goals

- No broadcast, compare/select, reductions, conversions, dtype/LMUL clone
  batches, high-level Linalg/Vector/StableHLO frontend work, Scalar/IME/Offload
  work, source-front-door positive routes, dashboards, or report-only progress.
- No Python compiler-core implementation.
- No descriptor-driven computation, direct-C/source-export route restoration,
  or compatibility wrapper that preserves old i32 authority as the positive
  production route.
- No runtime/correctness/performance claim unless real `ssh rvv` evidence is
  collected in this round.

## Validation Plan

- Build focused targets touched by RVV dialect/provider/target changes:
  `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Run focused RVV dialect/provider C++ tests if touched.
- Run focused lit tests for new generic selected-body fixtures and changed
  negative diagnostics.
- Run `git diff --check`.
- Run Trellis task validation.
- Run the active-authority scan from the Direction Brief and classify
  remaining matches in the final report.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- focused RVV target/conversion tests under `test/Target/RVV` and
  `test/Conversion/EmitC`

## Definition Of Done

- One coherent generic selected-body add/sub/mul surface is wired through the
  production route provider and focused tests.
- Old i32 helper names are not expanded or preserved as positive arithmetic
  route authority.
- The explicit RVV source-front-door materializer no longer produces positive
  Stage 1 RVV artifacts.
- The Trellis task context, PRD, and final notes truthfully reflect the bounded
  module completed.
- The task is finished/archived and one coherent commit is created if the
  module is complete; otherwise the task remains open with an exact
  continuation point.
