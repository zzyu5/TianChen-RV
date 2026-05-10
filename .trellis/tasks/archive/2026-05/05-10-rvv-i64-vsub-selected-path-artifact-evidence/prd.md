# RVV i64-vsub selected path from plugin planning to artifact evidence

## Goal

Prove and, if needed, repair the bounded finite-family RVV `i64-vsub`
selected path from marked linalg input through plugin proposal/planning,
plugin-local variant legality, selected lowering-boundary materialization,
selected emission planning, and target artifact export. If this round makes a
runtime correctness claim, it must also produce fresh real `ssh rvv` evidence
for the compiler-generated artifact path.

This task is a composition and evidence task for an existing finite family. It
is not a generic RVV backend, not an MLIR vector/scalable-vector lowering
route, and not a new dtype/family/op expansion.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting state was clean on `main` at commit
  `a10658e refactor(rvv): extract variant legality`.
- There was no active `.trellis/.current-task`; this task was created from the
  Hermes module brief.
- Recent archived work extracted finite RVV binary emission identity planning,
  proposal planning, selected emission planning, selected lowering-boundary
  materialization, and variant legality into plugin-local modules.
- The latest extraction proves ownership decomposition, but this task must
  prove those modules compose through the selected `i64-vsub` compiler path
  into durable artifact evidence.
- Existing finite RVV binary family scope is exactly `i32-vadd`, `i32-vsub`,
  `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`.
- Prior `i64-vsub`/`i64-vmul` work added finite family support and live
  evidence, but this round must check the current post-extraction selected
  path on current HEAD instead of relying on stale pre-extraction evidence.

## Requirements

- Inventory the current `i64-vsub` path from marked linalg frontend input to
  `tcrv.exec` kernel/ABI boundary, RVV plugin proposal planning, variant
  legality, selected lowering boundary, selected emission plan, and target
  artifact export.
- Record the inventory in this task context as supporting evidence, not as the
  main deliverable.
- If the extraction left a missing or stale link, make the smallest
  C++/MLIR/CMake/lit repair in the owning module.
- Prefer plugin-local RVV modules and target/export modules for repairs. Do
  not add RVV semantic branches to shared core transforms.
- Add or repair focused durable coverage proving the extracted modules are
  consumed together for `i64-vsub`: selected RVV binary path, legality
  metadata, selected lowering-boundary metadata, selected emission metadata,
  and artifact route metadata must agree.
- Run the existing RVV C-intrinsics artifact route for `i64-vsub` through the
  front door. Dry-run evidence is required; fresh `ssh rvv` evidence is
  required only if this round claims runtime correctness.
- Preserve parameter layering: VLEN/vlenb/ELEN stay target capability facts;
  SEW/LMUL/tail/mask stay selected RVV config or variant facts; AVL/VL remain
  runtime control values; descriptor-local `element_count` remains descriptor
  local and must not become fake high-level shape semantics.

## Acceptance Criteria

- [x] PRD and task context truthfully describe the exact `i64-vsub` selected
      path, non-goals, and validation plan.
- [x] Current `i64-vsub` inventory names the actual frontend input, pipeline
      command, plugin-local modules, target/export route, and generated
      artifact evidence path.
- [x] Focused C++ and/or lit coverage proves current HEAD consumes the
      extracted proposal planning, variant legality, selected lowering-boundary
      materialization, selected emission planning, and target artifact route
      together for `i64-vsub`.
- [x] The front-door artifact runner succeeds in dry-run mode for
      `i64-vsub` with `--lower-linalg-frontend` and writes sanitized evidence
      under `artifacts/tmp`.
- [x] If runtime correctness is claimed, the same bounded `i64-vsub` route
      succeeds on real `ssh rvv`, with sanitized evidence JSON under
      `artifacts/tmp` and no committed generated artifacts or raw logs.
- [x] Focused build/tests and relevant lit/FileCheck checks pass, or any
      blocker is recorded with exact failing command and next continuation
      point.
- [x] Task validation passes before finish/archive.

## Non-goals

- No generic RVV backend claim.
- No MLIR vector/scalable-vector lowering route.
- No new dtype, family, operation, mask/tail policy, or vector shape beyond
  existing `i64-vsub`.
- No compiler internals implemented in Python. Python may only orchestrate
  evidence, run tools, and parse artifacts.
- No compute semantics added to `tcrv.exec`.
- No RVV-specific branches added to generic core orchestration passes when the
  owner is a plugin or target/export module.
- No helper-only, report-only, manifest-only, broad smoke-matrix-only, or
  one-more-negative-test closeout as the main result.
- No runtime correctness or performance claim without fresh real `ssh rvv`
  evidence.

## Minimal Validation Plan

- `git diff --check`
- Build focused touched targets plus `tcrv-opt` and `tcrv-translate`.
- Run focused C++ tests for any touched plugin/target modules, especially RVV
  binary planning, variant legality, selected emission, selected lowering
  boundary, target artifact export, and extension plugin tests as applicable.
- Run focused lit/FileCheck for the `i64-vsub` linalg-to-exec,
  execution-planning, lowering-boundary, emission-readiness, and
  target-artifact path touched by this task.
- Run dry-run evidence:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i64-vsub --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir --run-id codex-i64-vsub-selected-path-dry --overwrite`
- If runtime correctness is claimed, run:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i64-vsub --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir --run-id codex-i64-vsub-selected-path-live --overwrite --ssh-target rvv --timeout 120`
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if practical after focused checks pass.
- Validate this Trellis task path before finish/archive.

## Inventory

- Frontend input:
  `test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir`.
  The marked `linalg.generic` carries `tcrv_frontend_lowering = "i64-vsub"`,
  `tcrv_frontend_kernel = "frontend_i64_vsub"`, and target profile
  `@frontend_rvv_i64_profile`.
- Frontend lowering command:
  `tcrv-opt <input> --tcrv-lower-linalg-rvv-binary-to-exec` creates
  `tcrv.exec.kernel @frontend_i64_vsub` with `target = @frontend_rvv_i64_profile`,
  lhs/rhs/out `tcrv.exec.mem_window` boundaries, and direct
  `tcrv.exec.runtime_param @abi_runtime_element_count`.
- Selected pipeline command:
  `tcrv-opt <input> --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline`.
  The pipeline materializes `tcrv.exec.variant @rvv_first_slice` with
  `origin = "rvv-plugin"`, `requires = [@frontend_rvv_i64_profile]`,
  `tcrv_rvv.lowering_descriptor = "i64-vsub-microkernel.v1"`,
  `tcrv_rvv.required_march = "rv64gcv"`, and selected `i64m1` metadata.
- Proposal planning owner:
  `RVVExtensionPlugin::buildRVVFirstSliceProposal` delegates finite-family
  proposal facts to `buildRVVBinaryProposalPlan`.
- Variant legality owner:
  `RVVExtensionPlugin::verifyVariantLegality` delegates to
  `verifyRVVBinaryVariantLegality`; this round added direct `i64-vsub`
  legality coverage in `RVVBinaryVariantLegalityTest.cpp`.
- Selected lowering-boundary owner:
  `RVVExtensionPlugin::materializeSelectedLoweringBoundary` delegates to
  `materializeRVVBinarySelectedLoweringBoundary`; this round parameterized the
  direct module test and now covers both `i64-vsub` and `i64-vmul`.
- Selected emission owner:
  `RVVExtensionPlugin::buildVariantEmissionPlan` delegates the selected
  callable path to `buildRVVBinarySelectedVariantEmissionPlan`. The
  `i64-vsub` frontend lit now checks descriptor-owned runtime ABI parameters
  and selected-plan metadata on the materialized emission-plan diagnostic.
- Target/export route:
  target-owned RVV family descriptor
  `getI64VSubFamilyDescriptor()` owns route
  `tcrv-export-rvv-i64-vsub-microkernel-c`, runtime ABI
  `rvv-i64-vsub-runtime-callable-c-abi.v1`, function ABI
  `rvv-i64-vsub-runtime-callable-c-function.v1`, and intrinsic prefix
  `__riscv_vsub_vv_`; source export combines that prefix with selected
  suffix `i64m1`.
- Generated artifact evidence:
  dry-run evidence lives at
  `artifacts/tmp/rvv_microkernel_e2e/codex-i64-vsub-selected-path-dry/evidence.json`.
  Live `ssh rvv` evidence lives at
  `artifacts/tmp/rvv_microkernel_e2e/codex-i64-vsub-selected-path-live/evidence.json`.

## Completion Evidence

- Code/test changes:
  - `test/Plugin/RVVBinaryVariantLegalityTest.cpp` now validates a direct
    `i64-vsub` RVV selected variant through the extracted legality module.
  - `test/Plugin/RVVBinarySelectedLoweringBoundaryTest.cpp` now runs the
    extracted selected lowering-boundary module for `i64-vsub` and `i64-vmul`.
  - `test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir` now
    asserts descriptor-owned runtime ABI parameters and selected-plan metadata
    on the `i64-vsub` emission-plan diagnostic.
- Dry-run front-door evidence:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i64-vsub --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir --run-id codex-i64-vsub-selected-path-dry --overwrite`
  passed with `status = "success"`, `mode = "dry-run"`,
  `manifest_record.kernel = "frontend_i64_vsub"`, and
  `source_export_route = "tcrv-export-rvv-i64-vsub-microkernel-c"`.
- Live real hardware evidence:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i64-vsub --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir --run-id codex-i64-vsub-selected-path-live --overwrite --ssh-target rvv --timeout 120`
  passed with `status = "success"`, `mode = "ssh"`,
  `ssh_evidence.success = true`, remote compile/link/run success, and output
  marker `tcrv_rvv_i64_vsub_microkernel_external_abi_ok`.
- Validation performed:
  - `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-variant-legality-test tianchenrv-rvv-selected-lowering-boundary-test tcrv-opt tcrv-translate -j2`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
  - manual PIPE/SOURCE FileCheck for
    `test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir`
    using `artifacts/tmp/tianchenrv-build/bin` and `/usr/lib/llvm-20/bin`
    in `PATH`
  - `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
  - `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
  - `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'linalg-i64-vsub-to-rvv-artifact|rvv-microkernel-e2e'`
    from `artifacts/tmp/tianchenrv-build/test` passed 2/2 selected tests.
  - `git diff --check`
  - `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-i64-vsub-selected-path-artifact-evidence`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
    passed 194/194 lit tests.
- Self-repair:
  an initial manual FileCheck invocation failed because `FileCheck` was not in
  `PATH`; rerun with `/usr/lib/llvm-20/bin` succeeded. An initial lit command
  from the repo root failed because `lit.site.cfg.py` uses a relative config
  path; rerun from `artifacts/tmp/tianchenrv-build/test` succeeded.
- Claim scope:
  the runtime correctness claim is bounded to the generated finite-family
  `i64-vsub` RVV C-intrinsics direct helper artifact and external caller on
  real `ssh rvv`. This is not a generic RVV backend, not an MLIR vector
  lowering route, and not a performance claim.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/capability-model/capability-contract.md`.
- Recent archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-planning-extraction/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-proposal-legality-planning/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-selected-emission-planning/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-selected-lowering-boundary-materialization/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-variant-legality-validation/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-i64-binary-sub-mul-frontend-artifact-ssh-evidence/prd.md`.
