# Core Semantic Branch Erasure: delete RVV source-to-exec pass family

## Goal

Delete the core RVV source-to-exec semantic pass family as a Wrong Logic
Deletion Campaign owner. This round removes the production/default path where
core transforms and public tool front doors understand finite RVV
linalg/vector source semantics, look up RVV binary family records, and
materialize `tcrv.exec` directly.

## Why Now

Commit `16a977c` completed direct-C semantic exporter erasure and left the
worktree clean. The next deletion surface is still in core transform/tool code:
`lib/Transforms/LowerSourceRVVBinaryToExec.cpp` imports
`Target/RVV/RVVBinaryFamily.h`, recognizes finite RVV add/sub/mul
linalg/vector shapes, looks up RVV family records, and constructs
`tcrv.exec`. `Passes.h`, `Passes.td`, `tcrv-opt`, and `tcrv-translate` still
publish or invoke those source front doors, including compatibility aliases.

## Scope

- Delete or fail-close the `LowerSourceRVVBinaryToExec` transform family in
  core transforms.
- Remove public pass declarations, TableGen definitions, CMake membership, and
  `tcrv-opt` registrations for:
  `tcrv-lower-source-rvv-binary-to-exec`,
  `tcrv-lower-linalg-rvv-binary-to-exec`,
  `tcrv-lower-linalg-i32-binary-to-exec`,
  `tcrv-lower-linalg-i32-vadd-to-exec`,
  `tcrv-lower-vector-rvv-i32-vadd-to-exec`,
  `tcrv-lower-vector-rvv-i32-vsub-to-exec`, and
  `tcrv-lower-vector-rvv-i32-vmul-to-exec`.
- Remove `tcrv-translate` front-door invocation of bounded source RVV binary
  lowering before plan-and-export bundle export.
- Delete or rewrite lit tests and fixtures that only protect old core-owned
  linalg/vector source lowering behavior.
- Rewrite directly stale spec wording that currently documents these source
  passes as production behavior or required test coverage.
- Keep `RVVSmokeProbe.cpp` only as hardware/toolchain smoke harness support if
  it remains unrelated to source-to-exec semantic lowering.

## Acceptance Criteria

- [x] Core transform/tool code no longer imports
      `TianChenRV/Target/RVV/RVVBinaryFamily.h` for source frontend lowering.
- [x] Core transform/tool code no longer looks up
      `RVVBinaryFamilyRecord` / `getRVVBinaryFamilyRegistrationRecords()` to
      recognize add/sub/mul linalg/vector source semantics.
- [x] The removed source-to-exec pass names are no longer registered through
      `tcrv-opt` and no compatibility aliases remain.
- [x] `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` no
      longer runs source RVV binary lowering before planning/export.
- [x] Tests that only asserted old core source-lowering success or old
      semantic validation diagnostics are deleted or rewritten to assert
      absence/fail-closed behavior.
- [x] If deletion exposes missing future extension-family lowering
      architecture, report the exact build/test gaps and do not rebuild that
      architecture in this round.
- [x] No Common Extension Interface Foundation, Common Lower-To-EmitC pass,
      plugin construction template, replacement frontend lowering, wrapper,
      compatibility alias, descriptor route, direct-C route, Python compiler
      core logic, or new extension-family implementation is added.
- [x] Bounded ref-scan over touched transform/tool/test/spec areas reports
      remaining hits for the deleted pass names and RVV source-family hooks.
- [x] Focused build/lit checks, full `check-tianchenrv` when coherent,
      `git diff --check`, `git diff --cached --check`, Trellis validation, and
      final clean worktree are run before finish/archive/commit.

## Non-Goals

- No Common Lower-To-EmitC implementation.
- No executable plugin construction template.
- No new RVV extension-family rebuild.
- No selected-boundary materialization replacement.
- No replacement frontend lowering for linalg, vector, stablehlo, tosa, or
  custom high-level dialects.
- No target artifact route rebuild.
- No direct C exporter restoration.
- No descriptor restoration.
- No compatibility wrapper, quarantine, or legacy alias for removed passes.
- No finite RVV family expansion.
- No `ssh rvv` evidence campaign.
- No broad repository audit beyond bounded deletion evidence.

## Minimal Evidence

- Inventory declarations, definitions, CMake membership, `tcrv-opt`
  registration, `tcrv-translate` front-door invocation, and lit/spec references
  for the deleted pass family.
- Bounded ref-scan over touched transform/tool/test/spec areas for
  `LowerSourceRVVBinaryToExec`, `LowerLinalgRVVBinaryToExec`,
  `LowerLinalgI32BinaryToExec`, `LowerLinalgI32VAddToExec`,
  `LowerVectorRVVI32VAddToExec`, `LowerVectorRVVI32VSubToExec`,
  `LowerVectorRVVI32VMulToExec`,
  `tcrv-lower-source-rvv-binary-to-exec`,
  `tcrv-lower-linalg-rvv-binary-to-exec`,
  `tcrv-lower-vector-rvv`,
  `RVVBinaryFamilyRecord`,
  `getRVVBinaryFamilyRegistrationRecords`, and
  `Target/RVV/RVVBinaryFamily.h`.
- Focused build targets for `TianChenRVTransforms`, `tcrv-opt`, and
  `tcrv-translate`.
- Affected lit subsets first, then full `cmake --build build --target
  check-tianchenrv -j2` when the deletion is coherent.
- `git diff --check`, `git diff --cached --check`, Trellis validation before
  finish and after archive, and final clean `git status --short`.

## Completion Notes

- Deleted `lib/Transforms/LowerSourceRVVBinaryToExec.cpp` and removed it from
  `TianChenRVTransforms`.
- Removed pass declarations/TableGen definitions and `tcrv-opt` registrations
  for the source RVV binary pass family and all compatibility aliases.
- Removed `tcrv-translate`'s plan-and-export pre-planning call to the deleted
  source frontend pass. The front door now runs execution planning directly on
  already materialized TianChen-RV execution surfaces.
- Deleted old LinalgToExec/VectorToExec source-lowering lit files and the stale
  target-bundle marker-mismatch source-frontdoor test. Added deleted-route lit
  coverage proving all removed pass names fail as unregistered options without
  creating `tcrv.exec.kernel`.
- Rewrote stale spec/README/support-comment wording so the source-to-exec pass
  family is documented as deleted and future high-level frontend work is routed
  through plugin/interface-owned rebuild work.
- Bounded core transform/tool ref-scan found no remaining
  `Target/RVV/RVVBinaryFamily.h`, `RVVBinaryFamilyRecord`,
  `getRVVBinaryFamilyRegistrationRecords`, or deleted pass-name registrations
  in `lib/Transforms`, `include/TianChenRV/Transforms`, or `tools`.
- No replacement lowering, wrapper, compatibility alias, descriptor route,
  direct-C route, Common EmitC rebuild, Python compiler semantics, or new RVV
  extension-family implementation was added.
- Checks run: focused `TianChenRVTransforms`, `tcrv-opt`, and
  `tcrv-translate` build; focused deleted-route lit; focused plan-and-export
  no-viable lit; full `cmake --build build --target check-tianchenrv -j2`
  passed 127/127; `git diff --check` passed; Trellis validation passed.

## Technical Notes

- `.trellis/spec/index.md` defines TianChen-RV as unified RISC-V MLIR and
  rejects core passes with hard-coded RVV/IME/Sophgo extension branches.
- `.trellis/spec/architecture/design-boundaries.md` rejects a collection of
  hard-coded backend branches in core passes.
- `.trellis/spec/lowering-runtime/emitc-route.md` keeps the current rebuild
  direction as extension family ops -> EmitC -> intrinsic/vendor/runtime C/C++.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md` contain stale
  production/test wording for the now-deleted core source front doors.
- `.trellis/tasks/archive/2026-05/05-15-direct-c-semantic-exporter-erasure/prd.md`
  is the immediately preceding deletion-campaign owner and confirms direct C
  semantic source routes are already deleted/fail-closed.
