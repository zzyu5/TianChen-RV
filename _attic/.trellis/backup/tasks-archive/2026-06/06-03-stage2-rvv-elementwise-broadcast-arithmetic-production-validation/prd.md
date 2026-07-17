# Stage2 RVV elementwise/broadcast arithmetic production validation boundary

## Goal

Make the production provider-to-target validation boundary for existing RVV
unit-stride elementwise arithmetic, masked elementwise arithmetic, and
scalar-broadcast elementwise arithmetic routes fail closed on provider-owned
facts instead of accepting stale route metadata or operation-mismatched binding
summaries.

The bounded route set is:

* `add`, `sub`, `mul`
* `masked_add`, `masked_sub`, `masked_mul`
* `scalar_broadcast_add`, `scalar_broadcast_sub`, `scalar_broadcast_mul`

## Direction Source

Hermes Direction Brief:

`Stage2 RVV elementwise/broadcast arithmetic production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned no dirty file list through RTK.
* Initial `git log --oneline -8` started at
  `295def8f trellis: close duplicate widening macc validation brief`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* The previous archived closeout was a duplicate widening MAcc task and does
  not cover this elementwise/broadcast owner.
* Current target validation already has an elementwise arithmetic family
  consumer for plain, masked, scalar-broadcast, and strided elementwise routes.
* Current provider/planning code already emits per-operation binding plans such
  as `rvv-route-operand-binding:add.v1`, `masked_sub.v1`, and
  `scalar_broadcast_mul.v1`.
* Live inspection found a bounded validation gap: target artifact provider
  validation only has a dedicated scalar-broadcast binding-summary check for
  `scalar_broadcast_add`, while plain add/sub/mul, masked add/sub/mul, and
  scalar-broadcast sub/mul are not all checked against an
  operation-specific canonical binding plan/summary at the provider-fact
  boundary.

## Requirements

* Keep common EmitC/export neutral. The target validator may consume provider
  facts and candidate mirrors, but it must not infer RVV computation or
  intrinsic semantics from route ids, artifact names, C strings, or test names.
* Validate operation-specific provider-owned route operand binding facts for
  the in-scope plain, masked, and scalar-broadcast elementwise routes.
* Preserve existing provider/planning ownership of operation kind, typed compute
  op, source/destination memory forms, RHS vector or scalar-broadcast source,
  scalar RHS role and C type, mask facts, ABI order, SEW/LMUL/policy,
  vector/mask/scalar C types, intrinsic leaves, header/type summaries, target
  leaf profile, `provider_supported_mirror`, and route-family mirror labels.
* Target artifact validation must fail closed when the route description carries
  stale or operation-mismatched route operand binding plans/summaries for
  plain, masked, or scalar-broadcast elementwise arithmetic.
* Add focused C++ tests that mutate provider facts and candidate mirrors for
  representative in-scope add/sub/mul families without broadening the route
  surface.

## Acceptance Criteria

* [x] `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` validates
      canonical operation-specific binding plan IDs and summaries for
      `add/sub/mul`, `masked_add/sub/mul`, and
      `scalar_broadcast_add/sub/mul`.
* [x] The validator rejects stale scalar-splat-only, MAcc, widening,
      reduction, compare/select, conversion, memory, vector-RHS vs
      scalar-broadcast, plain vs masked, operation-kind, RHS source, mask, ABI,
      header/type, target-profile, provider mirror, route-family mirror, and
      route operand binding facts within the in-scope elementwise family.
* [x] `test/Target/TargetArtifactExportTest.cpp` includes direct fail-closed
      C++ checks proving operation-mismatched binding plans/summaries and
      stale candidate mirrors do not pass for representative plain, masked, and
      scalar-broadcast arithmetic routes.
* [x] Existing lit/script fixture checks for explicit and pre-realized
      add/sub/mul, masked add/sub/mul, and scalar-broadcast add/sub/mul still
      pass under focused filters.
* [x] No new runtime behavior, generated arithmetic semantics, runtime ABI
      order, mask inactive-lane behavior, tail behavior, or destination
      preservation behavior changes are introduced; therefore no new `ssh rvv`
      correctness run is required.
* [x] Bounded old-authority scans over touched files find no new positive
      dependency on legacy `i32m1`, descriptor, source-front-door, route-id,
      artifact-name, exact intrinsic spelling, or mirror-only authority.

## Out Of Scope

* Unit-stride MAcc, scalar-broadcast MAcc, computed-mask MAcc, widening MAcc,
  widening dot-reduce, standalone reductions, compare/select expansion,
  conversion, indexed/strided/segment2/base memory movement redo, runtime
  scalar splat-store, source-front-door routes, and high-level frontend
  lowering.
* New dtype/LMUL clone batches, global tuning, dashboards, broad smoke
  matrices, or evidence-only packaging.
* Changing generated C/C++ runtime arithmetic, runtime ABI order, mask
  inactive-lane behavior, tail behavior, or destination preservation semantics.

## Technical Approach

* Keep the existing elementwise target artifact family validator as the
  consumer boundary.
* Replace the add-only scalar-broadcast binding check with a shared
  operation-specific binding summary validator for in-scope elementwise routes.
* Derive the expected binding plan ID from the selected operation kind and the
  expected logical operands from the family shape:
  plain vector RHS: `lhs,rhs,out,n`;
  masked vector RHS: `lhs,rhs,out,n`;
  scalar broadcast RHS: `lhs,rhs_scalar,out,n`.
* Reuse existing provider-owned ABI parameters and binding-summary entry
  validation helpers so the check stays a consumer of provider facts rather
  than a new source of RVV semantics.
* Add targeted C++ mutations for operation-mismatched provider facts and stale
  candidate mirrors.

## Definition of Done

* [x] Focused implementation and tests are ready for one coherent commit.
* [x] `tianchenrv-target-artifact-export-test` builds and passes.
* [x] `tcrv-opt` and `tcrv-translate` build for lit/script checks.
* [x] Focused lit/script filters for in-scope elementwise/broadcast routes pass.
* [x] Direct generated-bundle dry-runs for script-supported in-scope explicit
      and pre-realized forms pass for runtime counts `0,1,16,17,257`.
* [x] `rtk git diff --check` passes.
* [x] Task status and journal record the completed module behavior and
      evidence.

## Completion Notes

Implemented the provider-to-target validation boundary by exposing
provider-owned elementwise operand binding logical operands and materialized
uses from `RVVEmitCElementwiseRouteFamilyPlanOwners`, then making the target
artifact route-family validator consume those facts for the in-scope
plain/masked/scalar-broadcast arithmetic routes. The new validator checks the
operation-specific binding plan ID, logical operand order, runtime ABI order,
ABI role/C name, and exact materialized-use tokens before artifact export.

The production path remains provider-owned:

```text
selected typed tcrv_rvv body
  -> RVV provider-owned route facts
  -> TCRVEmitCLowerableRoute
  -> target artifact validation consuming provider facts
```

No generated arithmetic semantics, runtime ABI order, intrinsic spelling,
mask inactive-lane behavior, tail behavior, or destination preservation logic
was changed. Runtime `ssh rvv` correctness was therefore not rerun; the
evidence here is provider/target validation, local bundle dry-run, and existing
runtime behavior reuse.

Checks completed:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-(add|sub|mul|masked-add|masked-sub|masked-mul|scalar-broadcast-add|scalar-broadcast-sub|scalar-broadcast-mul)'`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(selected-body|pre-realized|masked-add|pre-realized-masked-add|scalar-broadcast-add|pre-realized-scalar-broadcast-add|pre-realized-scalar-broadcast-sub)-dry-run'`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-direct-pre-realized-scalar-broadcast-add-fail-closed'`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root build/trellis-validation/stage2-elementwise-broadcast --run-id explicit-all --overwrite --op-kind add --op-kind sub --op-kind mul --op-kind masked_add --op-kind masked_sub --op-kind masked_mul --op-kind scalar_broadcast_add --op-kind scalar_broadcast_sub --op-kind scalar_broadcast_mul --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root build/trellis-validation/stage2-elementwise-broadcast --run-id pre-realized-all --overwrite --op-kind add --op-kind sub --op-kind mul --op-kind masked_add --op-kind masked_sub --op-kind masked_mul --op-kind scalar_broadcast_add --op-kind scalar_broadcast_sub --op-kind scalar_broadcast_mul --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* `rtk git diff --check`
* Bounded touched-file legacy authority scan and diff-only legacy authority
  scan. The full-file scan found only pre-existing legacy/fail-closed test
  strings in `test/Target/TargetArtifactExportTest.cpp`; the diff-only scan
  found no new legacy authority strings.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived task read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-widening-macc-production-validation-duplicate-closeout/prd.md`

Relevant live files inspected:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/*broadcast*.mlir`
* `test/Target/RVV/*masked-add*.mlir`
* `test/Target/RVV/*masked-sub*.mlir`
* `test/Target/RVV/*masked-mul*.mlir`
* `test/Target/TargetArtifactExportTest.cpp`
