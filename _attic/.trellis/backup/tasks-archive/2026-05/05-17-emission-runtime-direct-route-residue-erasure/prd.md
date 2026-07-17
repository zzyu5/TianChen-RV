# Emission-Runtime Direct-Route Residue Erasure

## Goal

Delete or rewrite stale emission-runtime contract wording that can still be
read as authorizing direct RVV/RVV+scalar microkernel routes, descriptor body
policy, scalar direct source routes, direct source/header/object helpers,
bundle helpers, or self-check object routes outside the explicit
extension-family IR -> materialized EmitC/runtime route.

## What I Already Know

- Current HEAD is `311c198` (`test(rvv): prove runtime-callable C ABI
  linkage`), and the worktree was clean before this task was created.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- `.trellis/spec/index.md` defines the current main route as extension-family
  ops -> EmitC ops -> intrinsic/vendor builtin/runtime C/C++ -> native
  compiler, with descriptor/direct-C paths treated as historical residue,
  deletion targets, or fail-closed debt.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` already has
  many deletion/fail-closed statements, but targeted inspection found mixed
  wording in the Target Artifact Export Route and related scalar/dispatch
  sections that can still read as positive authority for direct RVV/scalar
  source/header/object helpers.
- `.trellis/spec/extension-plugins/rvv-plugin.md` allows only the bounded RVV
  i32m1 add/sub/mul executable slice through explicit typed `tcrv_rvv` IR,
  selected lowering boundary, RVV-owned EmitC lowerable route, MLIR EmitC
  C/C++ emitter, and clang RISC-V object packaging.
- The prior archived RVV C ABI link proof changed evidence tooling and task
  archive state. It did not authorize a production direct-C route or a new
  descriptor/source-export architecture.
- `scripts/rvv_generated_bundle_abi_e2e.py` is evidence tooling only; its
  header/object/bundle checks are bounded to the existing materialized EmitC
  path and must not become compiler route authority.

## Requirements

- Rewrite stale contract wording so current target artifacts may be produced
  only from selected extension-family IR plus a materialized EmitC/runtime
  route.
- Preserve negative fail-closed statements when they explicitly state that
  historical descriptor/direct-C/source-export/microkernel/self-check routes
  are absent, deleted, unsupported, or unusable as compiler authority.
- Preserve the bounded RVV materialized EmitC object/header/bundle path as a
  positive route only when the text names explicit typed RVV IR, the selected
  materialized EmitC route, MLIR EmitC C/C++ emission, and target artifact
  packaging.
- Delete or rewrite wording that positively authorizes direct RVV microkernel
  object/header routes, scalar direct source/header/object routes, RVV+scalar
  dispatch direct source/header/object routes, descriptor-backed body policy,
  source-export as a current production path, or self-check object route
  authority.
- If targeted scans expose directly related test, script, or `lib/Target`
  residue that protects the old direct route, delete or rewrite it to protect
  the materialized EmitC boundary instead.
- If removal exposes a missing new-architecture gap, report that gap; do not
  restore the old route or add compatibility wrappers.

## Acceptance Criteria

- [x] Touched specs/docs/tests contain no positive authorizing wording for
      descriptor body policy, direct RVV microkernel routes, RVV+scalar direct
      dispatch source/header/object routes, scalar direct source routes,
      self-check object route authority, or source-export as a current
      production path.
- [x] Remaining mentions of descriptor/direct-C/source-export/microkernel/
      scalar/dispatch/self-check residue are clearly historical, deleted,
      unsupported, fail-closed, or negative checks.
- [x] The bounded RVV object/header/bundle path remains described only as
      explicit typed extension-family IR -> materialized EmitC route -> MLIR
      EmitC C/C++ emitter -> target object/header/bundle packaging.
- [x] Focused `rg` scans over the touched spec plus directly related
      `lib/Target`, `test/Target`, and `scripts/rvv_generated_bundle_abi_e2e.py`
      surfaces show no restored descriptor/direct-C/source-export route
      authority.
- [x] `git diff --check` passes.
- [x] If the change remains spec/task-doc only, no build or full compiler check
      is required; if code/tests/scripts are touched, run the narrow affected
      tests and escalate to `check-tianchenrv` only if compiler route surfaces
      changed.

## Definition Of Done

- Trellis task status and context are truthful.
- The task is finished and archived according to the repo's Trellis convention
  if the acceptance criteria pass.
- One coherent commit records the PRD, spec cleanup, validation, and
  finish/archive state.

## Out Of Scope

- New lowering, new RVV family coverage, new runtime ABI features, new bundle
  routes, new plugin templates, new evidence scripts, or new `ssh rvv` runs as
  the main result.
- Compatibility wrappers, legacy modes, descriptor adapters, direct C semantic
  exporters, Python compiler-core behavior, or common/core RVV/scalar semantic
  branches.
- Treating prompt edits, reports, helper-only changes, guardrails, or broad
  smoke tests as the main achievement.
- Broad repo audits or full test matrices unless code/test changes make them
  necessary.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Prior task reference:
  `.trellis/tasks/archive/2026-05/05-17-rvv-runtime-callable-c-abi-link-proof/prd.md`.
- Evidence-tool boundary read:
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Primary expected edit surface:
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.

## Completion Summary

- Created and started Trellis task
  `05-17-emission-runtime-direct-route-residue-erasure` from the Direction
  Brief, with PRD scope locked to deletion/refactor-only contract cleanup.
- Rewrote the Target Artifact Export Route so target-owned artifact generation
  requires a selected materialized EmitC/runtime route before route-specific
  target facts can become usable.
- Replaced stale positive wording for direct RVV microkernel object/header
  routes, descriptor body policy, scalar fallback source/header/object helper
  routes, RVV+scalar dispatch source/header/object helpers, dispatch bundle
  source/header/object records, and self-check object helpers with explicit
  deleted/fail-closed or future-rebuild-only language.
- Preserved the allowed RVV path only as explicit typed RVV IR -> selected
  materialized EmitC route -> MLIR EmitC C/C++ emission -> target
  object/header/bundle packaging.
- Confirmed directly related `lib/Target`, `test/Target`, and
  `scripts/rvv_generated_bundle_abi_e2e.py` matches are negative checks,
  forbidden-token guards, or materialized EmitC route checks. No code, lit test,
  or evidence script change was needed.
- Added no compatibility layer, legacy mode, descriptor adapter, helper wrapper,
  or new route.

## Validation

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-emission-runtime-direct-route-residue-erasure`
- `rg -n "may (also )?select|remains an explicit|source candidate|callable source|direct helper|descriptor body policy|descriptor bodies|scalar .*helper|dispatch .*helper|direct C evidence|source-export exception|materialized source route|generated source, generated header|supported dispatch fallback|selected dispatch fallback|direct-printer generated C" .trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `rg -n "tcrv-export-scalar-|rvv-direct|direct-c|direct_C|source-export|source_export|descriptor-driven|descriptor body|direct-printer|self-check object|RVV\\+scalar.*direct|direct .*route authority|source/header/object routes? must not|must not select any historical" .trellis/spec/lowering-runtime/emission-runtime-contract.md lib/Target test/Target scripts/rvv_generated_bundle_abi_e2e.py`
- `rg -n "exportRVVMicrokernel|exportRVVScalar|RVVScalar.*SelfCheck|tcrv-export-rvv|tcrv-export-scalar|direct-microkernel|source_authority|TCRVEmitCSourceAuthority|lowerTCRVEmitCLowerableToEmitCSource" lib/Target test/Target scripts/rvv_generated_bundle_abi_e2e.py`
- `rg -n "rvv-runtime-callable-direct-c-source-exporter-deleted|scalar-runtime-callable-direct-c-source-exporter-deleted|unsupported-deleted-direct-c-route|unsupported missing-materialized-EmitC|materialized-emitc-cpp-rvv-intrinsic-object|rvv-i32m1-arithmetic-emitc-route-family|runtime-callable-c-header|riscv-elf-relocatable-object" .trellis/spec/lowering-runtime/emission-runtime-contract.md scripts/rvv_generated_bundle_abi_e2e.py`
- `git diff --check`
- No build, lit, `ssh rvv`, or `check-tianchenrv` run: this round changed only
  Trellis task docs and the lowering-runtime contract spec, not compiler code,
  tests, scripts, or generated artifacts.
