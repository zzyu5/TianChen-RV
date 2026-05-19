# Stage1 generic typed RVV body-surface replacement

## Goal

Replace the remaining positive selected-body route surface for bounded RVV
arithmetic with a minimal generic typed `tcrv_rvv` body skeleton:

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

- Current HEAD is already past several selected-body cleanup tasks, including
  route-profile authority replacement and LMUL/broadcast/compare selected-body
  proofs.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` derives a route profile from
  selected-body/config/memory facts, but its body slice still matches finite
  `tcrv_rvv.i32_load`, `tcrv_rvv.i32_add/sub/mul`, and
  `tcrv_rvv.i32_store` operation classes.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` still exposes positive
  dtype-prefixed vector and mask types such as `!tcrv_rvv.i32m1` and
  `!tcrv_rvv.i32m2`, plus dtype-prefixed ops.
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp` still serializes
  selected-body role realization and mirror metadata with `i32_*` op names and
  `rvv-i32m1-*` mirror labels.
- The active prompt and specs require the next owner to introduce the generic
  typed low-level RVV surface before any further Stage 2 coverage expansion.

## Requirements

1. Add a minimal generic typed RVV vector dataflow type or equivalent surface
   that structurally carries element dtype and LMUL, with SEW/policy kept on
   `setvl` / `with_vl`.
2. Add generic selected-body dataflow ops for the retained arithmetic slice:
   load, binary with `kind = add|sub|mul`, and store. These ops must consume
   explicit runtime ABI values and the active VL token.
3. Rewire RVV provider route analysis so positive add/sub/mul selected-body
   routes are accepted through the generic surface, deriving operation kind,
   dtype, SEW, LMUL, policy, memory form, runtime ABI order, type mappings, and
   intrinsic leaves from body/config/runtime facts.
4. Do not add broadcast, compare/select, reductions, conversions, new dtype
   families, or source-front-door positive routes in this round.
5. Legacy `tcrv_rvv.i32_*` ops and `!tcrv_rvv.i32m*` types may remain only as
   deprecated/fail-closed inventory or untouched legacy coverage. They must not
   be the positive route authority for the retained add/sub/mul production path
   completed by this task.
6. Common EmitC/materializer/target export must remain neutral. RVV-specific
   semantic decisions stay in RVV dialect/provider/target-owned support leaves.
7. Unsupported generic combinations must fail closed with targeted diagnostics
   before target artifact construction.

## Acceptance Criteria

- Positive lit/FileCheck coverage proves selected RVV add/sub/mul routes using
  the generic typed `tcrv_rvv` load/binary/store surface reach provider-built
  route metadata and target header or artifact output.
- The provider no longer needs finite `tcrv_rvv.i32_add/sub/mul` operation
  classes to recognize the retained add/sub/mul route.
- Route description validation checks generic body-derived operation kind,
  vector type/config facts, memory form, runtime ABI values, and intrinsic/type
  leaves.
- Negative coverage proves stale or mismatched generic body/config facts fail
  closed before artifact construction.
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
- Old i32 helper names are not expanded as architecture.
- The Trellis task context, PRD, and final notes truthfully reflect the bounded
  module completed.
- The task is finished/archived and one coherent commit is created if the
  module is complete; otherwise the task remains open with an exact
  continuation point.
