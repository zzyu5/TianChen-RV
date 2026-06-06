# Stage2 RVV Segment2 Unit-Memory Executable Artifact ABI Boundary

## Goal

Prove the existing plain segment2 unit-memory selected-body routes are real
generated RVV artifact boundaries, not dry-run-only or metadata-authorized
claims. The bounded owner is the path from selected/pre-realized typed
`tcrv_rvv` segment2 interleave/deinterleave bodies through RVV provider-owned
plain segment2 facts, `TCRVEmitCLowerableRoute`, common EmitC materialization,
target artifact validation, generated bundle ABI, and `ssh rvv` correctness
evidence.

## What I Already Know

* The live repository is on `main`, with a clean status at session start and
  recent commit `e8836b62 rvv: prove segment2 update artifact abi boundary`.
* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes/user Direction Brief.
* The relevant specs require plain segment2 interleave/deinterleave target
  validation to consume a provider-owned fact surface, not route ids, artifact
  names, fixture names, exact intrinsics, or candidate metadata mirrors.
* The plain ABI orders are fixed by provider facts:
  `segment2_interleave_unit_load` uses `src0,src1,dst,n`; and
  `segment2_deinterleave_unit_store` uses `src,out0,out1,n`.
* Common EmitC only carries provider-built route payloads; it must not infer
  segment direction, field roles, ABI/header/type facts, or RVV support.
* Initial audit found production target validation already has
  `validateRVVSegment2MemoryRouteHeaders`,
  `validateRVVSegment2MemoryRouteTypeMappings`, and
  `validateRVVSegment2MemoryRouteABIMappings`, but the focused C++ negative
  evidence for plain segment2 rebuilt route header/type/ABI payloads was not
  present at the same level as the previous computed-mask segment2 update task.

## Requirements

* Keep the task scoped to pre-realized/selected plain segment2 interleave
  unit-load and deinterleave unit-store executable artifact ABI boundaries.
* Preserve the production ownership chain:
  typed `tcrv_rvv` body -> RVV segment2 planning owner -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact
  export validator.
* Add focused fail-closed C++ coverage for stale or missing rebuilt route
  payload facts on the plain unit-memory routes.
* Positive evidence must cover both pre-realized plain segment2 routes through
  materialized selected boundary, emission plan, target artifact export,
  generated bundle compile, and `ssh rvv` correctness when runtime behavior is
  claimed.
* Do not broaden into a segment2 matrix, dtype/LMUL clones, source-front-door
  routes, computed-mask segment2 update rework, or unrelated memory/reduction/
  contraction families.

## Acceptance Criteria

* [x] PRD records the audit finding and selected contract for this round.
* [x] Target C++ coverage proves plain segment2 deinterleave rejects a missing
      rebuilt `riscv_vector.h` route header, stale vector type mapping, and
      stale `src` ABI value mapping before artifact acceptance.
* [x] Target C++ coverage proves plain segment2 interleave rejects a missing
      rebuilt `riscv_vector.h` route header, stale vector type mapping, and
      stale `src0` ABI value mapping before artifact acceptance.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run tests for pre-realized segment2
      interleave/deinterleave and direct pre-realized fail-closed shortcuts
      pass.
* [x] Non-dry-run generated-bundle evidence passes on `ssh rvv` for
      `segment2_interleave_unit_load`.
* [x] Non-dry-run generated-bundle evidence passes on `ssh rvv` for
      `segment2_deinterleave_unit_store`.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new legacy `i32m1`, descriptor, source-front-door, artifact-name, or
      exact-intrinsic-as-authority route drift.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean after commit.

## Out of Scope

* No broad segment2 matrix.
* No dtype/LMUL clone batch.
* No computed-mask segment2 update rework except as reference.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No performance tuning database, dashboard, or report-only closure.
* No source-front-door positive route.
* No common EmitC invention of RVV semantics.
* No mass rewrite of unit-stride masked memory, product/dequant, contraction,
  standalone reduction, compare/select, conversion, or unrelated route families.

## Technical Notes

* Direction source: Hermes/user Direction Brief on 2026-06-06.
* Relevant specs:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
* Prior reference task:
  * `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-computed-masked-segment2-update-artifact-abi/`
* Primary production files audited:
  * `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`
  * `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  * `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  * `scripts/rvv_generated_bundle_abi_e2e.py`
  * relevant Script and Target/RVV segment2 fixtures

## Audit Finding And Selected Contract

The initial audit found that the current production path already routes plain
segment2 interleave/deinterleave through the required owner chain:

```text
typed/pre-realized tcrv_rvv body
  -> plain segment2 route-family planning owner
  -> segment2 statement-plan boundary
  -> provider-built TCRVEmitCLowerableRoute
  -> segment2 target artifact validator
  -> RVVSegment2MemoryRouteValidationContract
```

The target validator already checks provider-owned route id, runtime AVL/VL
contract, plain segment2 route-family plan, runtime ABI order, route operand
binding summary, required headers, type mappings, ABI mappings, source/
destination memory forms, field roles/names, segment layout, and statement
shape. The bounded missing evidence selected for this round was direct rebuilt
route-payload fail-closed coverage for the two plain unit-memory routes, plus
non-dry-run `ssh rvv` generated-bundle correctness evidence.

## Current Phase

Finish.

## Evidence

* Added focused target C++ route-payload negative checks in
  `test/Target/TargetArtifactExportTest.cpp` for both plain segment2 routes:
  missing rebuilt route header `riscv_vector.h`, stale vector type mapping, and
  stale first ABI value mapping now fail through the production
  `RVVSegment2MemoryRouteValidationContract` target validator before artifact
  acceptance.
* `cmake --build build --target tianchenrv-target-artifact-export-test`
  completed successfully.
* `build/bin/tianchenrv-target-artifact-export-test` passed.
* `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
  completed successfully.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* From `build/test`,
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'segment2-(interleave-unit-load|deinterleave-unit-store)'` passed 8 selected
  tests.
* Non-dry-run generated-bundle evidence for `segment2_interleave_unit_load`
  passed on `ssh rvv` with `dry_run: false`, `ssh_evidence: true`, `status:
  success`, ABI
  `rvv-generic-segment2-interleave-unit-load-callable-c-abi.v1`, selected
  variant `pre_realized_body_rvv_segment2_interleave_unit_load`, counts
  `0,1,7,16,23,257`, field-order distinguishing lanes, and tail preservation.
* Non-dry-run generated-bundle evidence for
  `segment2_deinterleave_unit_store` passed on `ssh rvv` with `dry_run: false`,
  `ssh_evidence: true`, `status: success`, ABI
  `rvv-generic-segment2-deinterleave-unit-store-callable-c-abi.v1`, selected
  variant `pre_realized_body_rvv_segment2_deinterleave_unit_store`, counts
  `0,1,7,16,17,23,257`, field-order distinguishing lanes, and tail
  preservation.
* Bounded staged added-line scan over `test/Target/TargetArtifactExportTest.cpp`
  found no new legacy `i32m1`, descriptor, source-front-door, source-export,
  direct-C, or exact-intrinsic-as-authority route drift in this round's C++
  additions.
* `git diff --check` and `git diff --cached --check` passed.

## Spec Update Decision

No `.trellis/spec/` update is needed. This round implements and verifies the
already documented plain segment2 unit-memory target export contract: target
validation consumes provider-owned route validation contracts and treats
metadata as mirrors only. The new checks are focused regression evidence for an
existing contract, not a new architectural rule.
