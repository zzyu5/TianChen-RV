# RVV finite binary descriptor planning contract

## Goal

Centralize the RVV plugin-owned finite binary descriptor/family planning
contract for the existing bounded i32/i32m2/i64m1 add/sub/mul routes. Proposal
planning must resolve the selected family and vector-shape capability ids from
either a bounded frontend lowering marker or exactly one direct
`tcrv_rvv.lowering_descriptor` / RVV microkernel descriptor in the kernel,
without keeping descriptor-specific inference logic in `RVVExtensionPlugin.cpp`.

## Background

The archived `rvv-i64m1-plan-and-export-frontdoor-route` task repaired the
immediate i64m1 front-door blocker by adding direct descriptor-family inference
inside `RVVExtensionPlugin.cpp`. The archived
`rvv-i64m1-frontdoor-linalg-ssh-evidence` task then proved the linalg i64-vadd
front-door route through generated bundle export and real `ssh rvv` evidence.

That fix is now becoming an architectural bottleneck: finite RVV family
knowledge already participates in proposal planning, selected capability
metadata, variant legality, selected lowering-boundary materialization,
emission planning, and export preflight. This task moves the active planning
decision into a reusable RVV-owned descriptor/family API.

## Module Boundary

Owned by this task:

- RVV plugin/target-owned C++ descriptor-family resolution for finite RVV
  binary add/sub/mul families.
- Proposal planning consumption of that resolution for frontend-marker and
  direct-descriptor cases.
- Selected vector shape and selected capability id derivation for existing
  i32m1, i32m2, and i64m1 routes.
- Focused C++ and lit/script coverage for positive and negative descriptor
  planning behavior.
- Trellis task status, validation, archive, and one coherent commit if the
  module completes.

Out of scope:

- Generic RVV backend, arbitrary RVV lowering, or new family matrix expansion.
- New runtime correctness or performance claims without fresh `ssh rvv`
  evidence.
- Moving RVV semantics into generic core passes or `tcrv.exec`.
- Python implementation of compiler internals.
- Docs-only, helper-only, metadata-only, smoke-only, or evidence-only closeout.

## Requirements

- Factor direct descriptor-family inference out of `RVVExtensionPlugin.cpp`
  into RVV-owned descriptor/family planning support near the existing
  `RVVBinaryPlanning`, `RVVBinaryDescriptor`, `RVVBinaryFamilyRegistry`, and
  `RVVVectorShape` surfaces.
- The contract must cover only already-supported finite routes:
  `i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`,
  with i32 selecting finite i32m1/i32m2 capability facts and i64 selecting the
  finite i64m1 capability facts.
- Proposal planning must use the centralized contract for both
  `tcrv_frontend_lowering` markers and direct descriptor fallback.
- Unknown descriptors, multiple conflicting direct descriptors in one kernel,
  and descriptor/selected-shape mismatches must fail closed with bounded
  plugin-local diagnostics before stale metadata reaches export.
- Hardware capability facts, compile-time selected vector config, runtime
  AVL/VL/ABI control, and descriptor-local element count must remain separate in
  APIs, metadata, and diagnostics.
- Existing i32/i32m2 add/sub/mul behavior and i64-vadd/i64m1
  plan-and-export behavior must remain stable.

## Acceptance Criteria

- Focused C++ tests prove direct `i64-vadd-microkernel.v1` descriptor
  resolution derives family `i64-vadd`, shape `i64m1`, and selected capability
  ids:
  `rvv.i64_m1.sew64`, `rvv.i64_m1.lmul_m1`,
  `rvv.i64_m1.tail_policy.agnostic`, and
  `rvv.i64_m1.mask_policy.agnostic`.
- Focused tests prove existing i32/i32m2 proposal plans and selected capability
  ids remain unchanged.
- Negative coverage proves unknown direct descriptor and ambiguous multiple
  direct descriptors fail closed. Descriptor/shape mismatch must be covered if
  expressible through the current IR/test surface.
- `RVVExtensionPlugin.cpp` delegates descriptor-family resolution to the shared
  RVV planning contract instead of scanning and interpreting descriptor values
  inline.
- Focused RVV C++ tests and the existing i64-vadd plan-and-export dry-run
  lit/script route pass.
- `git diff --check` passes.
- `check-tianchenrv` runs if practical after focused checks pass.
- The task is validated and archived only if active RVV C++ code is moved to
  the shared contract and checks pass.

## Validation Plan

- Build focused targets:
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-binary-variant-legality-test`,
  `tianchenrv-rvv-extension-plugin-test`, `tcrv-translate`.
- Run focused C++ tests:
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`,
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`,
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`.
- Run focused lit/script path from `artifacts/tmp/tianchenrv-build/test` with a
  filter covering `rvv-microkernel-bundle-e2e` and the i64-vadd
  plan-and-export route. It must still select `frontend_i64_vadd`,
  `rvv_first_slice`, `i64m1`, and `__riscv_vadd_vv_i64m1`.
- Run `git diff --check`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if practical.
- Do not run `ssh rvv` unless emitted runtime artifact semantics change or a
  new runtime correctness claim is intentionally made.

## Initial Technical Notes

- `RVVBinaryFamilyRegistry.h` already owns finite family descriptors and
  lookup by frontend lowering and lowering descriptor.
- `RVVVectorShape.h` already owns finite shape configs and selected vector
  capability ids for i32m1, i32m2, and i64m1.
- `RVVBinaryPlanning.cpp` already builds `RVVBinaryProposalPlan` and derives
  selected capability ids from the selected plan. It is the natural owner for
  the reusable proposal-family resolution API.
- `RVVExtensionPlugin.cpp` currently contains the direct descriptor scan in
  `buildRVVFirstSliceProposal`; this task should replace that local inference
  with an RVV planning API call.
- Existing tests already cover i64 frontend marker planning and one direct i64
  descriptor inference case. This task should add contract-level tests and
  negative coverage rather than broad runtime matrices.

## Definition of Done

The task is done when the centralized RVV finite descriptor/family planning
contract is implemented in C++, consumed by proposal planning, covered by
focused positive and negative tests, validated through focused build/lit checks,
archived through Trellis, and committed once. If any required behavior is still
missing, leave the task open and record the exact failing command plus the
source file/function to continue from.

## Completion Notes

Completed. `RVVBinaryPlanning` now owns a finite descriptor/family planning
resolution contract:

- `resolveRVVBinaryFamilyForProposal` resolves a family from
  `tcrv_frontend_lowering`, exactly one direct
  `tcrv_rvv.lowering_descriptor`, or a direct RVV binary microkernel op.
- `RVVBinaryFamilyPlanningResolution` records the resolved family, resolution
  source, optional direct selected vector shape, and selected-shape capability
  ids.
- The kernel-aware `buildRVVBinaryProposalPlan` overload consumes that
  resolution, derives the selected vector shape and selected capability ids,
  and preserves direct i32m2/i64m1 selected-shape constraints when present.
- Unknown direct descriptors, ambiguous direct descriptor families, and
  descriptor/selected-shape mismatches fail closed with RVV plugin-local
  diagnostics.

`RVVExtensionPlugin.cpp` no longer scans direct variants or interprets
descriptor strings inline. It delegates proposal-family resolution to the
shared RVV planning API and only materializes the resulting proposal metadata.

Focused coverage added in `RVVBinaryPlanningTest`:

- direct `i64-vadd-microkernel.v1` resolves to family `i64-vadd`, shape
  `i64m1`, and selected capability ids `rvv.i64_m1.sew64`,
  `rvv.i64_m1.lmul_m1`, `rvv.i64_m1.tail_policy.agnostic`, and
  `rvv.i64_m1.mask_policy.agnostic`;
- direct i32 descriptor resolution preserves i32m2 selected shape and
  capability ids when the descriptor metadata requires it;
- unknown direct descriptor, ambiguous direct descriptors, and descriptor/shape
  mismatch fail closed.

Validation results:

- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-binary-variant-legality-test
  tianchenrv-rvv-extension-plugin-test tcrv-translate -j2`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`:
  passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`:
  passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`:
  passed.
- `python3 scripts/rvv_microkernel_e2e.py --dry-run
  --use-target-artifact-bundle --use-plan-and-export-bundle-front-door
  --arithmetic-family=i64-vadd --input
  test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir
  --expect-selected-kernel=frontend_i64_vadd --run-id
  codex-finite-descriptor-contract-i64-vadd-dry --overwrite --timeout 120`:
  passed. The dry-run evidence records `frontend_i64_vadd`, `rvv_first_slice`,
  `i64m1`, and `__riscv_vadd_vv_i64m1`.
- Focused lit from `artifacts/tmp/tianchenrv-build/test` with filter
  `rvv-microkernel-bundle-e2e|plan-linalg-i64-vadd|rvv-extension-plugin|rvv-binary-planning|rvv-binary-variant-legality`:
  4/4 selected tests passed.
- `git diff --check`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  199/199 lit tests passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-rvv-finite-binary-descriptor-planning-contract`: passed.

No `ssh rvv` run was collected and no new RVV runtime correctness or
performance claim is made. This task changed planning contract ownership and
fail-closed metadata validation, not emitted runtime artifact semantics.
