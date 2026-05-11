# Descriptor exit from direct RVV variant planning and export

## Goal

Move the bounded direct hand-authored RVV `i32-vadd` selected/export path away
from descriptor-owned compute identity. When a selected RVV path has a typed
`tcrv_rvv.i32_vadd_microkernel` body, planning and export must derive or verify
the finite family from the typed microkernel op, its structured
`setvl`/`with_vl` control layer, finite load/add/store dataflow, selected
vector-shape metadata, and IR-backed callable ABI roles. A
`tcrv_rvv.lowering_descriptor` may remain only as an optional legacy route
marker or cross-check, not the production/default source of compute semantics.

This is a structural migration task. It is not a helper-only task, a
coverage-only task, a report-only task, a benchmark task, or a new-family task.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Branch is `main`; the worktree was clean before this task was created.
* Current HEAD before this task is
  `7e690f0 feat: derive finite binary frontend from source linalg`.
* No `.trellis/.current-task` existed at session start; this task was created
  as `.trellis/tasks/05-12-descriptor-exit-direct-rvv-variant-planning/`.
* The archived predecessor
  `.trellis/tasks/archive/2026-05/05-12-descriptor-exit-linalg-finite-binary-frontend/`
  moved the linalg frontend boundary so source linalg body/types authority
  finite binary identity and legacy descriptor metadata is rejected there.
* Current specs say descriptor-driven computation is bounded implementation
  debt; the main route is typed extension family ops -> common EmitC ->
  intrinsic/runtime C/C++.
* Current source still has direct RVV descriptor authority surfaces:
  `resolveRVVBinaryFamilyForProposal`, `direct-lowering-descriptor` source
  kinds, direct descriptor planning tests, selected emission planning that can
  return a descriptor-selected plan, and export diagnostics around
  `tcrv_rvv.lowering_descriptor`.
* Existing body/export verification already checks many typed-body details
  before source output: arithmetic op mismatch, selected vector shape mismatch,
  `setvl`/`with_vl` config mismatch, dataflow SSA mismatch, and missing/swapped
  load/store ABI roles. This task should wire those facts into direct planning
  authority rather than leaving descriptor lookup first.

## Requirements

* Add or strengthen shared C++ helpers that can inspect a direct typed
  `tcrv_rvv.*_microkernel` body and derive the finite RVV binary family from
  typed extension-family ops rather than from
  `tcrv_rvv.lowering_descriptor`.
* Rewire the direct RVV proposal/planning path so typed body evidence is
  considered before descriptor-family lookup for the bounded `i32-vadd` direct
  selected path.
* When both typed body and `tcrv_rvv.lowering_descriptor` are present, require
  descriptor family, dtype, arithmetic, and microkernel identity to agree with
  typed body evidence. Mismatches must fail closed before artifact emission.
* Descriptor-only direct i32 compute must not remain a production/default
  direct-compute authority path. It may remain only as explicit legacy
  compatibility for non-migrated surfaces, or it must fail closed for this
  bounded direct i32 route.
* A typed direct `i32-vadd` body without descriptor compute authority must still
  select/export through the existing `i32-vadd` RVV route and common
  EmitC-backed generated artifact behavior.
* The selected shape/config contract must agree with body-local
  `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, and typed vector token dataflow before
  source/header/object artifact output.
* Direct RVV load/store role metadata must remain tied to the IR-backed
  callable ABI roles: lhs input, rhs input, output, and runtime element count.
  Missing, duplicate, or swapped roles must fail before artifact output.
* Keep `tcrv.exec` compute-free; RVV semantics stay plugin/target-owned through
  typed RVV family ops and common EmitC/lowerable route interfaces.

## Acceptance Criteria

* [x] Direct hand-authored `i32-vadd` selected/export succeeds when the typed
      body contains `tcrv_rvv.i32_vadd_microkernel` with `setvl`,
      `with_vl`, lhs/rhs `i32_load`, `i32_add`, and `i32_store`, and no
      descriptor compute authority is needed.
* [x] Direct planning resolves typed direct `i32-vadd` family identity from the
      typed microkernel body before descriptor-family fallback.
* [x] A direct path with descriptor `i32-vadd-microkernel.v1` but typed body
      arithmetic `tcrv_rvv.i32_sub` or `tcrv_rvv.i32_mul` fails before
      artifact emission.
* [x] A direct descriptor-only i32 selected path without typed
      `tcrv_rvv.*_microkernel` body is rejected or explicitly quarantined as
      legacy, not accepted as the production/default direct compute source.
* [x] Selected shape/config mismatch with body `setvl`/`with_vl` or typed
      vector token dataflow fails before artifact emission.
* [x] Missing or swapped direct RVV load/store ABI roles fail before artifact
      emission.
* [x] C++ tests prove typed-body planning/export success and the required
      negative cases for descriptor/body mismatch, missing typed body,
      selected config/body mismatch, and missing/swapped ABI roles.
* [x] A focused lit/FileCheck path proves direct `i32-vadd` still reaches the
      generated EmitC-backed RVV artifact/export behavior through typed-body
      authority.
* [x] Python remains tooling-only; no compiler core, dialect, pass, plugin
      registry, capability model, lowering, or emission logic is implemented
      in Python.

## Out Of Scope

* Adding new RVV dtypes, arithmetic families, LMULs, benchmarks, performance
  claims, runtime correctness claims, or generic RVV backend claims.
* Adding an MLIR vector/scalable-vector lowering route.
* Adding descriptor-to-C exporters or descriptor-owned compute semantics.
* Moving compute semantics into `tcrv.exec`.
* IME, AME, Sophgo/offload, Template, Toy, scalar migration, or unrelated
  descriptor-exit work.
* Preserving obsolete descriptor-driven tests as default-path authority just to
  minimize diffs.
* Running `ssh rvv`; this task makes no fresh runtime/correctness/performance
  claim.

## Minimal Validation

* Run `git diff --check`.
* Build focused targets:
  `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
* Run focused C++ tests affected by RVV planning, selected lowering boundary,
  selected emission, and target artifact export.
* Run focused lit/FileCheck tests under `test/Target/RVVMicrokernel/`,
  `test/Transforms/EmissionReadiness/`, `test/Transforms/LoweringBoundary/`,
  and closest direct RVV artifact/export routes.
* Run full `cmake --build artifacts/tmp/tianchenrv-build --target
  check-tianchenrv -j2` if focused checks pass and local time/tooling allows.
* Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-12-descriptor-exit-direct-rvv-variant-planning` before
  finishing, and validate the archived task path after archive if completed.

## Technical Notes

* `.trellis/spec/index.md` defines descriptor-driven computation as bounded
  debt and forbids Python compiler implementation.
* `.trellis/spec/architecture/unified-riscv-mlir.md` keeps `tcrv.exec`
  compute-free and puts RVV semantics in the extension family.
* `.trellis/spec/lowering-runtime/emitc-route.md` says the correct route is
  typed extension op -> generated lowerable interface -> target/plugin mapping
  -> common EmitC route provenance.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires
  selected vector shape to remain compile-time config and runtime AVL/VL to
  remain runtime/control/ABI layered.
* `.trellis/spec/extension-plugins/rvv-plugin.md` defines the typed
  `tcrv_rvv.i32_vadd_microkernel` body shape and the descriptor boundary.
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md` requires
  plugin-proposed variants and typed extension-family ops rather than generic
  core compute.
* `.trellis/spec/testing/mlir-testing-contract.md` requires focused C++ and
  lit/FileCheck coverage for typed RVV body/export behavior and fail-closed
  mismatch diagnostics.

## Round Result

Completed in this round:

* Added a typed direct RVV binary body resolver for the bounded `i32-vadd`
  route. It derives family, selected vector shape, SEW/LMUL/policy,
  `setvl`/`with_vl` control layering, load/add/store dataflow, ABI buffer
  roles, and runtime element-count layering from typed RVV extension-family
  ops and selected config metadata.
* Rewired direct planning so typed microkernel body evidence is resolved before
  descriptor-family lookup. When descriptor metadata is also present, it is
  merged only as a cross-check and mismatches fail closed.
* Rewired selected emission planning so explicit typed microkernel evidence
  builds the selected plan after checking any descriptor attachment, rather
  than returning the descriptor-selected plan as compute authority.
* Rewired selected lowering-boundary materialization so typed selected
  attachments are preferred, and descriptor-only direct `i32-vadd` boundary
  materialization is quarantined as legacy instead of emitted by default.
* Replaced the old descriptor-only missing-march lit test with an explicit
  descriptor-only `i32-vadd` quarantine test.

Validation completed:

* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-planning-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-selected-lowering-boundary-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* Focused lit/FileCheck filter for direct RVV microkernel/export and lowering
  boundary tests: 8 selected tests passed.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed with 206/206 lit tests.

No `ssh rvv` run was needed or used because this round makes no fresh RVV
runtime correctness or performance claim.
