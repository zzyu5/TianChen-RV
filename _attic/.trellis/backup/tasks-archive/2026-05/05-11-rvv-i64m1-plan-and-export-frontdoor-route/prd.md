# RVV i64m1 plan-and-export front-door route

## Goal

Repair the bounded RVV `i64-vadd` / `i64m1` route so the normal plan-and-export
front door can propose, select, materialize, verify, and export the selected RVV
artifact bundle. The existing direct selected fixture must no longer be the only
successful path for the finite i64m1 route.

## Background

The previous task `rvv-verified-i64m1-microkernel-ssh-evidence` proved that the
already-selected direct `i64-vadd` fixture can pass
`validateRVVBinaryMicrokernelBody`, export artifacts, compile/link/run on
`ssh rvv`, and validate deterministic numeric output. That task also exposed the
next blocker: the plan-and-export bundle front door re-runs proposal planning and
still rejects the i64 direct route because the planning/capability path expects
finite i32 config capability ids.

## Module Boundary

Owned by this task:

- RVV plugin/planning/capability behavior for the finite `i64-vadd` / `i64m1`
  route.
- Selected vector capability metadata for SEW=64, LMUL=m1, tail policy, and mask
  policy.
- Selected lowering/emission boundary materialization used by target artifact
  bundle export.
- Focused C++ and lit/FileCheck coverage for the route.

Out of scope:

- Generic RVV backend work.
- Broad RVV family expansion or benchmark sweeps.
- New high-level tensor/linalg i64 frontend lowering unless proven to be the
  single minimal blocker for this module.
- Hand-written source/runtime artifact as truth.
- Performance, throughput, latency, or generalized correctness claims.
- RVV semantic branches in generic core passes or generic artifact routing.
- Compute semantics in `tcrv.exec`.
- Compiler internals implemented in Python.

## Requirements

- Identify the exact planning/capability mismatch that causes the front-door
  route to expect i32 selected config capability ids for the i64 route.
- Repair the mismatch in RVV plugin/planning/capability-owned C++ code.
- Preserve existing i32/i32m2 RVV planning, legality, export, and dispatch
  behavior.
- Keep metadata layered:
  - hardware facts and capability ids;
  - compile-time selected vector config;
  - runtime AVL/VL control;
  - descriptor-local element count.
- Keep the route finite to already-supported `i64-vadd` / `i64m1`.
- Document any remaining boundary if high-level i64 frontend lowering is not in
  scope.

## Acceptance Criteria

- Focused C++ tests prove RVV planning/legality can select the i64m1 route with
  correct i64 selected capability ids.
- Existing i32/i32m2 planning and legality tests continue to pass.
- A focused plan-and-export bundle dry-run test exercises the normal front-door
  route for `i64-vadd`; it must not only exercise the already-selected direct
  fixture route.
- Exported artifact inspection verifies the expected i64 RVV intrinsic route
  such as `__riscv_vadd_vv_i64m1`.
- The route must not silently fall back to i32, sub, mul, or scalar paths.
- `git diff --check` passes.
- Focused RVV plugin/planning/export builds and tests pass.
- `check-tianchenrv` runs if practical after focused checks pass.
- Runtime correctness is claimed only if a changed runtime artifact is validated
  through `ssh rvv` with deterministic numeric output comparison.

## Validation Plan

- Run focused C++ tests around:
  - `RVVBinaryPlanningTest`
  - `RVVBinaryVariantLegalityTest`
  - `RVVExtensionPluginTest`
- Run the focused bundle/export lit test covering the plan-and-export dry-run
  route.
- Inspect generated/exported content for `__riscv_vadd_vv_i64m1` and absence of
  unintended fallback routes.
- Run `git diff --check`.
- Run `check-tianchenrv` if available and practical.
- Run `ssh rvv` evidence only if this round claims runtime correctness for a new
  or changed runtime artifact.

## Prior Context To Read

- `.trellis/tasks/archive/2026-05/05-11-rvv-verified-i64m1-microkernel-ssh-evidence/prd.md`
- `.trellis/tasks/archive/2026-05/05-11-rvv-microkernel-body-selected-config-verifier/prd.md`
- Relevant RVV plugin, planning, legality, emission, target descriptor,
  microkernel, verifier, script, and tests listed in the task brief.

## Completion State

Completed. The exact mismatch was the RVV proposal path for a direct selected
fixture with no `tcrv_frontend_lowering`: `RVVExtensionPlugin` called
`buildRVVBinaryProposalPlan` with an empty frontend family marker, so the
proposal planner used the default `i32-vadd` family and required finite i32
config ids even when the direct fixture already carried an i64 lowering
descriptor and i64m1 selected vector metadata.

The repair is plugin-local. `RVVExtensionPlugin` now infers the proposal family
from exactly one direct RVV finite `tcrv_rvv.lowering_descriptor` in the same
kernel when the frontend marker is absent. This keeps the normal marked-linalg
front door unchanged, preserves existing i32/i32m2 default selection, and keeps
the fallback bounded to registered finite RVV binary descriptors. Generic core
passes and target artifact routing were not changed.

Focused coverage added:

- `RVVExtensionPluginTest` now proves a direct `i64-vadd` descriptor with i64m1
  capability facts produces a proposal requiring the i64m1 selected capability
  ids and selected vector metadata instead of i32 ids.
- `rvv-microkernel-bundle-e2e.test` now runs the previously failing
  `--use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd`
  route and checks the bundle evidence for `__riscv_vadd_vv_i64m1`, int64 ABI
  parameters, direct i64 route ids, and no i32/sub/mul fallback.

Validation results:

- Baseline failure reproduced before the patch:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd --expect-selected-kernel=export_i64_vadd --run-id codex-i64-vadd-frontdoor-baseline --overwrite --timeout 120`
  failed because proposal planning required finite i32 config ids.
- Post-fix direct fixture front-door dry-run passed:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd --expect-selected-kernel=export_i64_vadd --run-id codex-i64-vadd-frontdoor-after-cpp --overwrite --timeout 120`.
- Post-fix linalg front-door dry-run passed:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd --input test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i64_vadd --run-id codex-i64-vadd-frontdoor-linalg-after-cpp --overwrite --timeout 120`.
- Focused build:
  `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-translate -j2`.
- Focused C++ tests:
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-binary-variant-legality-test`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test` passed.
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `rvv-microkernel-bundle-e2e|rvv-extension-plugin` passed 2/2.
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `plan-linalg-i64-vadd|rvv-binary-planning|rvv-binary-variant-legality`
  passed 2/2.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 199/199 lit tests.
- `git diff --check` passed.
- Trellis task validation passed before archive.

No new `ssh rvv` runtime correctness claim is made in this task. The changed
behavior is proposal-family inference and dry-run bundle front-door export
selection; runtime correctness remains limited to prior ssh evidence unless a
new ssh run is explicitly collected.
