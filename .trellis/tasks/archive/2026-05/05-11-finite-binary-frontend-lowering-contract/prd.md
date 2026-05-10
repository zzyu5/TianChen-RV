# Finite binary frontend lowering contract

## Goal

Make the bounded linalg-to-exec frontend boundary derive, validate, and
materialize one structured finite binary lowering descriptor for the existing
i32/i64 add/sub/mul routes. The frontend contract must be owned in C++ and must
feed the existing RVV/scalar planning, runtime ABI, and target artifact export
surfaces without treating i64 routes as incidental behavior of an i32-named
path.

## Requirements

- Introduce a neutral C++ finite binary frontend lowering descriptor for the
  existing bounded families: `i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`,
  `i64-vsub`, and `i64-vmul`.
- Use that descriptor in the marked `linalg.generic` lowering pass to validate
  frontend marker, integer element width, arithmetic op, kernel symbol, target
  profile, capability-provider imports, and runtime ABI role materialization
  before plugin variant planning or target artifact export.
- Stop letting the active implementation look like an i32-only lowering owner.
  Keep deprecated i32 pass options as compatibility aliases, but make the
  implementation and CMake ownership use a finite/RVV binary name.
- Preserve `tcrv.exec` as the execution/capability/variant/dispatch/runtime ABI
  structure only. The pass must not create compute ops, RVV/scalar semantics,
  target-specific lowering, artifacts, or runtime claims.
- Ensure RVV and scalar planning/export continue consuming the same finite
  family marker/descriptor naming contract for all existing bounded families.
- Preserve the existing `frontend_i64_vadd` dispatch route and existing i32
  routes; do not add arbitrary dtypes, reductions, tensors, or generic vector
  lowering.

## Acceptance Criteria

- Positive lit/FileCheck coverage proves `i64-vadd` and at least one non-vadd
  i64 family, `i64-vsub`, lower through the frontend pass with correct kernel
  marker, `int64_t` ABI mem-window roles, runtime `n`, RVV/scalar descriptors,
  and downstream planning/export metadata.
- Existing i32 frontend regression coverage still passes.
- Negative lit/FileCheck coverage fails closed for unknown frontend marker,
  unsupported element type, and descriptor/operator mismatch before creating a
  `tcrv.exec.kernel`.
- Focused RVV/scalar planning and dispatch bundle checks continue to prove
  `frontend_i64_vadd`, `rvv_first_slice`, `i64m1`,
  `__riscv_vadd_vv_i64m1`, scalar fallback, and dispatch wrapper metadata.
- `git diff --check` passes, and `check-tianchenrv` is run if practical after
  focused checks.
- The Trellis task validates and archives only after active C++/MLIR/lit
  behavior is complete.

## Technical Approach

Add a shared C++ descriptor API under `include/TianChenRV/Support/` for the
finite binary frontend contract. The descriptor exposes only neutral frontend
facts: family id, frontend marker, dtype/element width, arithmetic kind, C ABI
buffer types, runtime element-count ABI role, and selected arithmetic marker.

Then update the linalg frontend pass implementation to use this descriptor
instead of a local transform-only spec derived directly from RVV family data.
Rename the implementation file to the RVV binary frontend owner while keeping
the deprecated i32 pass factories/CLI aliases delegating to the same code.

RVV/scalar plugin and target paths remain plugin/target-owned. They may retain
their richer family descriptors, but those descriptors are linked back to the
neutral frontend contract so marker/family naming stays coherent across the
frontend, plugin planning, runtime ABI, and artifact export paths.

## Decision (ADR-lite)

Context: The previous task completed the i64-vadd front-door dispatch runtime
ABI boundary. The remaining upstream problem is that the frontend lowering
surface still exposes an i32-named implementation and local transform metadata
while the repo now carries finite i32/i64 add/sub/mul routes.

Decision: Introduce a neutral finite binary frontend descriptor and make the
existing bounded linalg pass consume it directly. Keep RVV/scalar lowering and
artifact semantics in plugin/target code.

Consequences: This keeps the frontend contract explicit without promoting
TianChen-RV into a high-level tensor IR. Deprecated pass aliases remain for
compatibility, but new implementation ownership and tests use the finite/RVV
binary name.

## Out of Scope

- Generic linalg lowering, tensor/tile IR, reductions, matmul, or arbitrary
  vector lowering.
- New runtime correctness, performance, or throughput claims.
- New hardware evidence unless a new runtime correctness claim is made.
- Python compiler internals.
- Docs-only, helper-only, smoke-only, or report-only closeout.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived context read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-frontdoor-dispatch-runtime-abi/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-finite-binary-descriptor-planning-contract/prd.md`.
- Current active implementation starts at
  `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`, with the public pass already
  named `--tcrv-lower-linalg-rvv-binary-to-exec` and old i32 aliases retained.
- Current downstream family data lives in
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h` and
  `include/TianChenRV/Target/RVVScalarBinaryFamily.h`; this task must not move
  RVV/scalar emission semantics into the frontend pass.

## Definition of Done

The task is done when the neutral frontend descriptor is active in the bounded
linalg-to-exec pass, the implementation ownership no longer presents as i32
only, focused positive and negative lit/script/C++ checks pass, Trellis
validates and archives the task, and one coherent commit records the completed
module. If unfinished, leave the task open with the exact failing command,
source file/function, and blocker category.

## Completion Notes

Completed in this round:

- Added `support::FiniteBinaryFrontendLoweringDescriptor` as a neutral C++
  frontend contract for the existing bounded i32/i64 add/sub/mul families. It
  owns frontend marker, family id, descriptor name, arithmetic kind, element
  width, pointer C ABI spellings, and runtime ABI mem-window/runtime-param
  specs; it does not own RVV/scalar lowering, selected vector shape, route ids,
  artifacts, runtime execution, correctness, or performance evidence.
- Renamed the transform implementation file to
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp` and updated CMake ownership.
  The public `--tcrv-lower-linalg-rvv-binary-to-exec` pass remains the active
  surface, while the old i32 pass factories/options remain compatibility
  aliases into the same implementation.
- Updated the linalg frontend pass to derive its bounded body, element-type,
  arithmetic-op, and runtime ABI materialization checks from the neutral
  frontend descriptor instead of a local transform-only spec derived from RVV
  family data.
- Linked RVV finite family descriptors back to the same neutral frontend
  contract and tightened RVV binary planning registration validation so a
  registered RVV family must agree with the frontend contract on marker,
  descriptor, element width, and callable pointer C types.
- Added negative lit coverage for unknown frontend marker, unsupported element
  type, and i64 descriptor/operator mismatch. Existing i64-vadd and i64-vsub
  positive frontend tests continue to prove correct i64 marker, ABI, selected
  RVV descriptor, scalar descriptor where applicable, and target/export
  metadata.

Validation:

- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-scalar-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`:
  passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`:
  passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`:
  passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`:
  passed.
- Focused lit from `artifacts/tmp/tianchenrv-build/test` with filter
  `linalg-finite-binary-frontend-invalid|linalg-i64-vadd-to-rvv-artifact|linalg-i64-vsub-to-rvv-artifact|linalg-i32-vadd-to-exec|rvv-scalar-dispatch-bundle-e2e`:
  6/6 selected tests passed.
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run
  --use-target-artifact-bundle --use-plan-and-export-bundle-front-door
  --arithmetic-family=i64-vadd --input
  test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir
  --expect-selected-kernel=frontend_i64_vadd --run-id
  codex-finite-frontend-contract-i64-vadd-dry --overwrite --timeout 120`:
  passed.
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run
  --use-target-artifact-bundle --use-plan-and-export-bundle-front-door
  --arithmetic-family=i64-vsub --run-id
  codex-finite-frontend-contract-i64-vsub-bundle-dry --overwrite --timeout
  120`: passed.
- `python3 scripts/rvv_microkernel_e2e.py --dry-run
  --use-target-artifact-bundle --use-plan-and-export-bundle-front-door
  --arithmetic-family=i64-vsub --input
  test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir
  --expect-selected-kernel=frontend_i64_vsub --run-id
  codex-finite-frontend-contract-i64-vsub-direct-dry --overwrite --timeout
  120`: passed.
- `git diff --check`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2`: 200/200 lit tests passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-finite-binary-frontend-lowering-contract`: passed.

No new `ssh rvv` evidence was collected and no new RVV runtime correctness or
performance claim is made. Runtime claims remain limited to prior archived
evidence.
