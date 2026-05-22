# Stage2 RVV Closure-Gated Runtime Scalar-Vector Add Boundary

## Goal

Complete the bounded Stage 2 RVV runtime scalar-vector add/store boundary:

```text
out_i32[i] = lhs_i32[i] + rhs_scalar_i32
```

for runtime `n` / AVL on the corrected typed `tcrv_rvv` surface. Current HEAD
already has the core `scalar_broadcast_add` production route and pre-realized
generated-bundle evidence; this round must not duplicate it. The bounded
missing consumer is the explicit selected-body artifact/runtime path plus a
RouteOperandBindingPlan closure repair that records and requires header mirrors
for the scalar-vector add operands that appear in the generated C ABI header.

## Direction Source

- Direction title: `Stage2 RVV closure-gated runtime scalar-vector add
  boundary`.
- Module owner: RVV plugin-owned route-supported runtime i32 scalar plus vector
  add/store on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `27851162 rvv: add closure-gated runtime scalar splat store`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle owners and archived
runtime scalar broadcast/add tasks.

- `runtime_i32_splat_store` is complete from the previous task and covers
  runtime scalar splat to store without lhs load or binary compute.
- `scalar_broadcast_add` already exists as the runtime scalar-vector add
  operation:
  - `rhs-scalar-value` ABI role;
  - generic `tcrv_rvv.splat` body node;
  - pre-realized `typed_binary_pre_realized_body` with
    `memory_form = "rhs-scalar-broadcast"` and `op_kind = "add"`;
  - realization to `setvl/with_vl/load/splat/binary/store`;
  - route family plan `RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan`;
  - provider materialization through `RouteOperandBindingPlan`;
  - pre-realized target/header artifact coverage;
  - pre-realized generated-bundle dry-run and real `ssh rvv` evidence for
    runtime counts `7,16,23` and scalar values `-37,91`.
- The current explicit selected-body generated-bundle table does not expose
  `scalar_broadcast_add`; explicit mode supports `scalar_broadcast_sub/mul`
  and pre-realized mode supports `scalar_broadcast_add`.
- The current scalar-broadcast elementwise binding summary records
  materialized lhs load and RHS scalar splat use, plus output/runtime header
  mirrors, but does not record `header-mirror` for the `lhs` and
  `rhs_scalar` C ABI header operands. The generated header nevertheless
  declares both operands.

## Scope

Repair one coherent add boundary:

- `scalar_broadcast_add`, explicit selected-body mode:
  - `lhs`: `lhs-input-buffer`, `const int32_t *`, unit-stride vector load,
    materialized lhs call operand, header mirror.
  - `rhs_scalar`: `rhs-scalar-value`, `int32_t`, runtime scalar value,
    materialized `tcrv_rvv.splat` operand, header mirror.
  - `out`: `output-buffer`, `int32_t *`, unit-stride vector store,
    materialized output memory, header mirror.
  - `n`: `runtime-element-count`, `size_t`, setvl AVL, loop control,
    header mirror.
  - typed config: signed i32, SEW32, LMUL m1, tail agnostic / mask agnostic.
  - body structure: `runtime_abi_value`, `setvl`, `with_vl`, `load`,
    `splat`, `binary {kind = "add"}`, `store`.

## Requirements

1. Do not add a duplicate scalar-broadcast route or revive legacy `RVVI32M1`,
   `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor, direct-C, or
   source-front-door authority.
2. Keep route authority in typed `tcrv_rvv` body/config/runtime facts and RVV
   plugin-owned planning/provider code. Common EmitC/export stays neutral.
3. Add explicit selected-body target artifact and generated-bundle support for
   `scalar_broadcast_add`, using the existing production route rather than a
   helper-only or report-only path.
4. Tighten `RouteOperandBindingPlan` closure for scalar-vector add so
   materialized operands and header mirrors agree for `lhs`, `rhs_scalar`,
   `out`, and `n`.
5. Preserve pre-realized scalar-broadcast add support and update its binding
   mirror expectations if the shared scalar-broadcast binding summary changes.
6. Runtime/correctness evidence must use value-distinguishing lhs vectors,
   positive and negative `rhs_scalar` values, counts `7`, `16`, and `23`,
   runtime `n`/AVL variation, active lanes equal `lhs + rhs_scalar`, and tail
   sentinel preservation.

## Acceptance Criteria

- [x] Current scalar-broadcast inventory is recorded with exact already
      supported and missing surfaces.
- [x] Explicit selected-body `scalar_broadcast_add` target/header artifact
      coverage proves `lhs`, `rhs_scalar`, `out`, `n`, SEW32/LMUL m1/policy,
      runtime control plan, route operand binding plan, provider-supported
      mirror, required headers, and generated C prototype.
- [x] RouteOperandBindingPlan summary for scalar-vector add includes header
      mirrors for every generated C ABI operand and provider construction
      requires those materialized uses before route construction.
- [x] Pre-realized scalar-broadcast add still realizes into
      `setvl/with_vl/load/splat/binary/store` before route construction.
- [x] Generated-bundle dry-runs pass for explicit and pre-realized
      `scalar_broadcast_add`, counts `7,16,23`, and scalar values `-37,91`.
- [x] Real `ssh rvv` generated-bundle runs pass for explicit and pre-realized
      `scalar_broadcast_add`, proving active lanes equal `lhs + rhs_scalar`
      and tails remain sentinel-preserved.
- [x] Negative fail-closed coverage remains present for missing/wrong scalar
      role, missing lhs/output/runtime roles, bad dtype/config, wrong op kind,
      mixed RHS buffer broadcast with scalar splat, stale route metadata,
      descriptor/direct-C/source-front-door authority, and common/export
      semantic inference.
- [x] Active-authority scan over touched RVV include/lib/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite positive
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, generated-bundle evidence, `git diff --check`,
      `check-tianchenrv`, truthful task finish/archive state, clean final git
      status, and one coherent commit complete if this task finishes.

## Completion Evidence

- Added explicit selected-body `scalar_broadcast_add` target/header artifact
  coverage. The fixture structurally carries `lhs`, runtime `rhs_scalar`,
  `out`, runtime `n`/AVL, SEW32, LMUL m1, agnostic policy, `tcrv_rvv.splat`,
  `tcrv_rvv.binary {kind = "add"}`, and `tcrv_rvv.store`.
- Repaired scalar-broadcast elementwise RouteOperandBindingPlan closure so
  `lhs`, `rhs_scalar`, `out`, and `n` all carry header mirrors in the provider
  binding summary. Provider construction now requires the header-mirror uses
  for the scalar-broadcast operands before materializing the route.
- Added C++ plugin coverage for the scalar-broadcast route operand binding
  summary and for fail-closed lookup when the RHS scalar header mirror is
  missing.
- Added explicit selected-body generated-bundle support and a script dry-run
  test for `scalar_broadcast_add`.
- Preserved pre-realized scalar-broadcast add and updated its evidence checks
  to assert the closure-gated binding summary.
- The shared scalar-broadcast summary update mechanically changed sub/mul
  FileCheck expectations because the existing route-family plan is shared.
  This did not add new sub/mul coverage or migrate those routes.

## Validation Performed

- `pwd`
- `git status --short`
- `git log --oneline -8`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-22-05-22-stage2-rvv-runtime-scalar-vector-add-boundary`
- `git diff --check`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused target/header checks:
  - explicit scalar-broadcast add/sub/mul `PLAN` and `HEADER`;
  - pre-realized scalar-broadcast add/sub/mul `REALIZED`, `PLAN`, and
    `HEADER`.
- Focused conversion and negative checks:
  - `test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-add-materialization.mlir`
  - `test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-negative.mlir`
  - `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`
- Generated-bundle dry-runs:
  - explicit:
    `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id scalar-vector-add-explicit-dryrun --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  - pre-realized:
    `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id scalar-vector-add-pre-realized-dryrun --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- Manual FileCheck equivalents for the explicit and pre-realized script dry-run
  tests: `ROOT`, `SCALAR` with negative authority guards, and `HARNESS`.
- Real RVV evidence:
  - explicit:
    `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id scalar-vector-add-explicit-ssh --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --connect-timeout 8 --timeout 120`
  - pre-realized:
    `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id scalar-vector-add-pre-realized-ssh --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --connect-timeout 8 --timeout 120`
- Active-authority scan over added diff lines found no positive
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, or exact
  intrinsic route authority, and no positive descriptor/direct-C/source-export
  or source-front-door authority.
- `cmake --build build --target check-tianchenrv -j2`

`check-tianchenrv` result:

```text
Total Discovered Tests: 331
  Passed: 331 (100.00%)
```

Fresh `ssh rvv` output for both explicit and pre-realized routes:

```text
rvv_generated_bundle_abi_e2e: success
[scalar_broadcast_add] scalar_broadcast_add case n=7 ok rhs_scalar=-37
scalar_broadcast_add case n=16 ok rhs_scalar=-37
scalar_broadcast_add case n=23 ok rhs_scalar=-37
scalar_broadcast_add case n=7 ok rhs_scalar=91
scalar_broadcast_add case n=16 ok rhs_scalar=91
scalar_broadcast_add case n=23 ok rhs_scalar=91
tcrv_rvv_generated_bundle_abi_scalar_broadcast_add_ok counts=7,16,23 rhs_scalars=-37,91
PASS op=scalar_broadcast_add counts=7,16,23 rhs_scalars=-37,91
```

Evidence artifacts:

- `artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-vector-add-explicit-dryrun`
- `artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-vector-add-pre-realized-dryrun`
- `artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-vector-add-explicit-ssh`
- `artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-vector-add-pre-realized-ssh`

Self-repair performed:

- Initial authority scan included an existing exact intrinsic string from
  context in a C++ test hunk. The scan was rerun over added diff lines only,
  confirming that this round did not add exact intrinsic route authority.

## Final Status

Completed. The existing production scalar-broadcast add route remains the
route authority. This round repaired its RouteOperandBindingPlan header-mirror
closure and added the missing explicit selected-body artifact/runtime consumer
with focused dry-run, real `ssh rvv`, and full `check-tianchenrv` validation.

## Spec Update Judgment

No `.trellis/spec/**` update is needed. The long-term specs already require
typed `tcrv_rvv` body authority, RVV plugin-owned route/provider semantics,
RouteOperandBindingPlan closure, common EmitC neutrality, and real `ssh rvv`
evidence for runtime/correctness claims. This round applies those rules to one
bounded scalar-vector add boundary without changing the durable architecture.

## Non-Goals

- No migration or clone batch for `scalar_broadcast_sub/mul`. If a shared
  scalar-broadcast binding summary changes mechanically, it is only to keep
  shared closure semantics coherent, not to expand those routes.
- No redo of `runtime_i32_splat_store`, widening conversion, reduction,
  masked macc, memory movement, compare/select, or general elementwise
  matrices except as regression anchors.
- No dtype/LMUL clone batch, masked scalar-binary family, high-level
  Linalg/Vector/StableHLO frontend lowering, source-front-door positive route,
  dashboard, report-only work, or helper-only cleanup.
- No route through legacy i32 authority, descriptor-driven direct C, or common
  EmitC/export semantic inference.

## Validation Plan

1. Validate this task.
2. Build focused binaries as needed:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused target/header checks for explicit and pre-realized
   `scalar_broadcast_add`.
5. Run focused conversion/materialization and existing negative scalar
   broadcast checks.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
7. Run generated-bundle dry-runs for explicit and pre-realized
   `scalar_broadcast_add`, counts `7,16,23`, scalar values `-37,91`.
8. Run real `ssh rvv` for the same explicit and pre-realized cases.
9. Run active-authority scans over touched RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- scalar-broadcast target, conversion, transform, script, and C++ plugin tests.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant prior task context read:

- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-broadcast-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-runtime-scalar-broadcast-add-production/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-scalar-broadcast-add-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-contract-closure-gate/prd.md`
