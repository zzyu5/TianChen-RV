# Journal - codex (Part 20)

> Continuation from `journal-19.md` (archived at ~2000 lines)
> Started: 2026-06-01

---



## Session 365: Stage2 RVV widening conversion selected-body realization boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV widening conversion selected-body realization boundary
**Branch**: `main`

### Summary

Added focused RVV plugin C++ evidence that pre-realized widening conversion bodies fail closed before route construction, realize through the public selected lowering-boundary producer into explicit setvl/with_vl/load/widening_convert/store structure for both supported widening cases, and feed provider/statement-plan route construction.

### Main Changes

- Created and archived Trellis task
  `06-01-stage2-rvv-widening-conversion-realization-boundary`.
- Strengthened `runWideningConversionSelectedBodyRealizationOwnerTest` to
  cover both pre-realized `widen_i16_to_i32` and `widen_i32_to_i64` selected
  bodies.
- Added focused evidence that direct route description and
  `TCRVEmitCLowerableRoute` construction fail closed before selected-boundary
  materialization, then succeed after the public selected lowering-boundary
  producer realizes explicit `setvl` / `with_vl` / `load` /
  `widening_convert` / `store` structure.
- Verified realized widening conversion bodies feed route-family provider
  checks, materialization facts, math operand bindings, route-control provider
  plan, statement-plan preflight, and provider-built route construction.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-widening-conversion-realization-boundary`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused widening conversion lit filter: 8/8 passed.
- [OK] Bounded old-authority scan classified remaining hits as spec text,
  negative/stale tests, provider-derived leaves, mirror checks, or legacy
  fail-closed inventory.
- [OK] `rtk git diff --check`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 465/465.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 368: Stage2 RVV computed-mask MAcc selected-body realization boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV computed-mask MAcc selected-body realization boundary
**Branch**: `main`

### Summary

Added focused computed-mask MAcc selected-body owner-boundary evidence:
direct pre-realized route facts/build fail closed, owner-local negative
validation, realization to setvl/with_vl/compare-mask/masked_macc/store, and
provider facts/statement-plan/route checks.

### Main Changes

- Created and archived Trellis task
  `06-01-stage2-rvv-computed-mask-macc-realization-boundary`.
- Added `runComputedMaskMAccSelectedBodyRealizationOwnerTest` to prove vector
  and runtime-scalar computed-mask MAcc selected-body realization before route
  construction.
- Verified direct pre-realized route description, direct selected-body route
  construction, stale route metadata bypass, and retired direct route-entry
  attempts all fail closed before provider route construction.
- Added owner-local negative coverage for invalid predicate/mask facts,
  invalid LMUL/config, non-agnostic policy, wrong compare/payload/acc/out/n
  roles, and wrong runtime-scalar RHS role.
- Verified realized route description, route-family provider plans,
  materialization facts, math operand-binding facts, route-control plan,
  computed-mask accumulation statement plan, and provider-built route consume
  realized facts only.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j 8`
- [OK] `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk git diff --check`
- [INFO] `rtk ctest --test-dir build -R tianchenrv-rvv-extension-plugin-test --output-on-failure` found no registered tests in the current build tree.
- [OK] Bounded old-authority scan classified new exact intrinsic hits as
  provider-derived leaf evidence; existing hits remain spec/fail-closed
  inventory or provider-derived route evidence.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 366: Stage2 RVV reduction-accumulation selected-body realization boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV reduction-accumulation selected-body realization boundary
**Branch**: `main`

### Summary

Hardened the focused RVV ordinary reduction selected-body realization boundary evidence before route construction.

### Main Changes

- Extended `runReductionSelectedBodyRealizationOwnerTest` with direct pre-realized route-description and route-construction fail-closed checks.
- Added negative owner checks for unsupported reduction op kind, invalid runtime n/AVL ABI role, and invalid accumulator layout.
- Verified realized reduction facts flow through route description, route-family checks, materialization facts, math operand binding, ordinary reduction's route-control non-consumer result, statement plan, and provider-built route construction.
- Validation: task context validate passed; RVV extension plugin test target built; `tianchenrv-rvv-extension-plugin-test` passed; bounded old-authority scan classified hits; `rtk git diff --check` passed; `check-tianchenrv` passed 465/465.
- Commit: included in final task commit for this session.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | Stage2 RVV reduction-accumulation selected-body realization boundary |

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-reduction-accumulation-realization-boundary`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] bounded old-authority scan classified remaining hits
- [OK] `rtk git diff --check`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` (465/465)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 367: Stage2 RVV standalone reduction selected-body realization boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV standalone reduction selected-body realization boundary
**Branch**: `main`

### Summary

Added focused standalone reduction selected-body owner-boundary evidence: direct pre-realized route facts/build fail closed, owner-local negative validation, realization to setvl/with_vl/load/standalone_reduce/store, and provider facts/statement-plan/route checks.

### Main Changes

- Created and archived Trellis task
  `06-01-06-01-stage2-rvv-standalone-reduction-realization-boundary`.
- Added `runStandaloneReductionSelectedBodyRealizationOwnerTest` to prove the
  standalone reduction selected-body owner boundary before route construction.
- Verified direct pre-realized route description and provider route
  construction fail closed before selected lowering-boundary materialization.
- Added owner-local negative coverage for unsupported op kind, unsupported
  LMUL/config, non-agnostic policy, wrong runtime `n` role, wrong scalar output
  role, wrong accumulator seed role/layout, and wrong scalar result layout.
- Verified public selected-boundary materialization erases the pre-realized op
  and creates explicit `setvl` / `with_vl` / `load` /
  `standalone_reduce` / `store` structure.
- Verified realized route description, route-family provider plans,
  materialization facts, math operand-binding facts, route-control plan,
  standalone statement plan, and provider-built route consume realized facts.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-06-01-stage2-rvv-standalone-reduction-realization-boundary`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Bounded old-authority scan classified remaining hits as spec text,
  fail-closed legacy inventory, stale-negative tests, or provider-derived leaf
  evidence.
- [OK] `rtk git diff --check`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 465/465.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 368: Stage2 RVV contraction selected-body realization boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV contraction selected-body realization boundary
**Branch**: `main`

### Summary

Added focused contraction owner negative evidence, archived the Trellis task, and verified RVV plugin checks.

### Main Changes

- Created and archived Trellis task `06-01-stage2-rvv-contraction-realization-boundary` from the Hermes direction brief.
- Added focused owner-local negative coverage to `runPreRealizedContractionRouteEntryOwnerTest` for computed-mask strided contraction rejecting non-agnostic policy and wrong ABI roles on compare lhs/rhs, dot lhs/rhs, accumulator seed, output, runtime `n`/AVL, lhs stride, and rhs stride.
- No production contraction owner/provider changes were required; existing code already materializes the five bounded contraction pre-realized families before route analysis and provider construction.
- Bounded old-authority scan found no requested legacy-authority strings in the new diff hunk; existing hits remain fail-closed guards, negative tests, provider-derived exact-intrinsic leaf evidence, selected-route diagnostics, or spec guardrails.
- Checks passed: task context validation, RVV plugin test target build, direct RVV plugin test binary, `git diff --check`, and `check-tianchenrv` 465/465.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 369: Stage2 RVV selected-body artifact runtime ABI boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV selected-body artifact runtime ABI boundary
**Branch**: `main`

### Summary

Closed the widening_macc_add pre-realized selected-body to generated bundle runtime ABI boundary with selected-boundary dry-run evidence, direct route-entry fail-closed regression, ssh rvv correctness for counts 0,1,16,17,257, and check-tianchenrv 465/465.

### Main Changes

- Added a `widening_macc_add` self-test regression proving
  `--direct-pre-realized-route-entry` remains retired/fail-closed for the
  selected pre-realized contraction body.
- Updated the testing contract so the current positive pre-realized generated
  bundle path is selected-boundary materialization before provider route facts;
  direct pre-realized route-entry is documented as a negative mode.
- Archived the completed runtime ABI evidence task with the final dry-run,
  direct fail-closed, `ssh rvv`, focused binary, `git diff --check`, and
  `check-tianchenrv` evidence recorded.

### Git Commits

| Hash | Message |
|------|---------|
| `none` | No commit created in this session; committed by Session 370 closeout. |

### Testing

- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Selected-boundary `widening_macc_add` generated-bundle dry-run
- [OK] Direct route-entry negative check exited 1 with the retired shortcut diagnostic
- [OK] Real `ssh rvv` runtime ABI correctness for counts `0,1,16,17,257`
- [OK] Bounded old-authority scan
- [OK] `rtk git diff --check`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 465/465.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 370: Stage2 RVV selected-body artifact ABI closeout

**Date**: 2026-06-01
**Task**: Stage2 RVV selected-body artifact ABI closeout
**Branch**: `main`

### Summary

Closed the dirty repository state from Session 369 by reviewing the
uncommitted runtime ABI boundary diff, preserving and rerunning focused
`widening_macc_add` selected-body-to-artifact evidence, cleaning archived task
context placeholders, and preparing one coherent commit.

### Main Changes

- Created closeout task `06-01-stage2-rvv-selected-body-artifact-abi-closeout`
  with PRD/context scoped to repository coherence, validation, archive, commit,
  and clean status.
- Removed stale `_example` context rows from the archived Session 369 task's
  `implement.jsonl` and `check.jsonl`.
- Revalidated the previous runtime ABI task and reran focused closeout checks:
  script self-test, selected-boundary dry-run, direct route-entry fail-closed
  negative check, real `ssh rvv` runtime ABI correctness, focused build target,
  focused test binaries, bounded old-authority scan, task validation, and
  `git diff --check`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log after commit) |

### Testing

- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind widening_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-abi-closeout/final-dry-run`
- [OK] Direct route-entry negative command exited 1 with the expected retired direct route-entry diagnostic.
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-abi-closeout/final-ssh-rvv`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Bounded old-authority scan over touched tracked diff and task directories.
- [OK] `rtk git diff --check`
- [OK] Trellis context validation for both the closeout task and the previous archived runtime ABI task.

### Status

[OK] **Completed**

### Next Steps

- None - task complete after commit and clean status verification


## Session 371: Stage2 RVV standalone-reduction selected-body artifact ABI

**Date**: 2026-06-01
**Task**: Stage2 RVV standalone-reduction selected-body artifact ABI
**Branch**: `main`

### Summary

Proved the existing `standalone_reduce_add` selected-body-to-generated-bundle
scalar-result ABI boundary with selected-boundary dry-run evidence, direct
route-entry fail-closed regression, and real `ssh rvv` correctness for runtime
counts `0,1,16,17,257`.

### Main Changes

- Created task `06-01-stage2-rvv-standalone-reduction-artifact-abi` with PRD
  and context scoped to exactly one supported standalone reduction selected
  body.
- Verified the production path already carries `standalone_reduce_add` through
  RVV selected-body realization, provider route facts, common EmitC, RVV target
  artifact bundle export, and external scalar-result ABI execution.
- Added a `rvv_generated_bundle_abi_e2e.py --self-test` regression for the
  retired direct pre-realized `standalone_reduce_add` route-entry mode.
- Recorded final dry-run, direct fail-closed, `ssh rvv`, focused binary,
  old-authority scan, and `git diff --check` evidence in the PRD.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log after commit) |

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-01-stage2-rvv-standalone-reduction-artifact-abi`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-standalone-reduction-artifact-abi/final-dry-run`
- [OK] Direct route-entry negative command exited 1 with the expected retired direct route-entry diagnostic.
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-standalone-reduction-artifact-abi/final-ssh-rvv`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Bounded old-authority scan over touched script/task files and relevant owner/provider/materializer/target/test files.
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete after archive, commit, and clean status verification

## Session 377: Stage2 RVV widening dot-reduction accumulator artifact ABI boundary

**Date**: 2026-06-02
**Task**: Stage2 RVV widening dot-reduction accumulator artifact ABI boundary
**Branch**: `main`

### Summary

Closed the bounded pre-realized `widening_dot_reduce_add` selected-body
artifact/runtime ABI boundary by pinning target-validator source/result ABI
role fail-closed coverage, strengthening generated-bundle boundary self-tests
and FileCheck evidence, and proving real `ssh rvv` correctness for counts
`0,1,16,23,257`.

### Main Changes

- Created Trellis task
  `06-02-stage2-rvv-widening-dot-reduce-artifact-abi` from the Hermes brief.
- Verified the existing production path carries `widening_dot_reduce_add`
  through RVV selected-body realization, contraction route-family validation,
  math operand-binding facts, direct contraction provider plan, common EmitC,
  RVV target artifact bundle export, and external scalar-result ABI execution.
- Added target artifact export negative coverage for stale plain widening-dot
  `lhs` and `out` runtime ABI roles.
- Strengthened `rvv_generated_bundle_abi_e2e.py --self-test` so the
  widening-dot boundary summary must preserve selected source ABI roles,
  provider route facts, statement-plan seed/carry/store facts, direct
  route-entry unsupported status, and runtime counts.
- Updated the focused pre-realized widening-dot dry-run FileCheck to use
  `0,1,16,23,257` and pin `selected_source_abi` / `statement_plan` fields.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] selected-boundary generated-bundle dry-run for
  `widening_dot_reduce_add`, counts `0,1,16,23,257`
- [OK] direct pre-realized `widening_dot_reduce_add` route-entry failed closed
  with the expected retired shortcut diagnostic.
- [OK] FileCheck equivalent for focused `ROOT`, `WDOT`, and `HARNESS` prefixes
  because `llvm-lit` is not installed in this environment.
- [OK] `REALIZED`, `PLAN`, and `HEADER` FileCheck prefixes for
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir`
- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] real `ssh rvv` generated-bundle correctness for
  `widening_dot_reduce_add`, counts `0,1,16,23,257`
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-02-stage2-rvv-widening-dot-reduce-artifact-abi`
- [OK] `rtk git diff --check`
- [OK] Bounded added-line old-authority scan; only provider-derived exact
  intrinsic evidence strings were added.

### Status

[OK] **Completed**

### Next Steps

- None - task complete after archive, commit, and clean status verification.


## Session 376: Stage2 RVV widening conversion artifact ABI boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV widening conversion artifact ABI boundary
**Branch**: `main`

### Summary

Proved the pre-realized `widen_i16_to_i32` / `sign_extend_widen_vf2`
selected-body-to-generated-bundle conversion ABI with stricter evidence JSON,
focused FileCheck, direct route-entry fail-closed coverage, focused C++ tests,
and real `ssh rvv` correctness for runtime counts `0,1,16,23,257`.

### Main Changes

- Created task
  `06-01-stage2-rvv-widening-conversion-artifact-abi` with PRD and context
  scoped to exactly one widening conversion selected body.
- Verified the existing production path carries `widen_i16_to_i32` through RVV
  selected-body realization, widening-conversion provider facts, common EmitC,
  RVV target artifact bundle export, and external ABI execution.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so
  `conversion_sew_policy_boundary` explicitly records selected ABI roles,
  source-load/result-store statement-plan facts, provider route facts,
  tail/mask policy mirrors, and retired direct route-entry status.
- Updated the focused dry-run FileCheck test to pin those boundary fields.
- Recorded final dry-run, direct fail-closed, `ssh rvv`, focused C++/FileCheck,
  old-authority scan, and `git diff --check` evidence in the task PRD.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-widening-conversion-artifact-abi`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind widen_i16_to_i32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-widening-conversion-artifact-abi/focused-dry-run-v2`
- [OK] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `widen_i16_to_i32`.
- [OK] Focused FileCheck checks for script `ROOT`, `WIDEN`, `HARNESS` prefixes
  and target fixture `REALIZED`, `PLAN`, `HEADER` prefixes.
- [OK] `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widen_i16_to_i32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-widening-conversion-artifact-abi/final-ssh-rvv-v2`
- [OK] Bounded old-authority scan over touched script/test/task files.
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete after archive, commit, and clean status verification


## Session 375: Stage2 RVV dual runtime-scalar mask-and-select artifact ABI boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV dual runtime-scalar mask-and-select artifact ABI boundary
**Branch**: `main`

### Summary

Proved the base `runtime_scalar_dual_cmp_mask_and_select` selected-body to
generated-bundle ABI path with stricter threshold-pair evidence, focused
fail-closed/direct-route self-test coverage, and real `ssh rvv` correctness for
counts `0,1,16,23,257` across four runtime scalar threshold pairs.

### Main Changes

- Created and archived task
  `06-01-06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi` with a
  PRD scoped to exactly one base dual runtime-scalar selected body.
- Verified the existing production C++ path carries the selected body through
  RVV selected-body realization, computed-mask select route facts,
  operand-binding facts, statement-plan facts, provider preflight, common EmitC,
  and RVV target artifact bundle export.
- Added explicit `*_threshold_pairs_required_minimum = 2` evidence fields for
  the dual runtime-scalar compare/select path.
- Added `--self-test` coverage for the dual runtime-scalar RHS threshold
  minimum, retired direct pre-realized route-entry diagnostic, and generated
  harness aggregate mask/mask-and/select-payload checks.
- Recorded final dry-run, direct fail-closed, `ssh rvv`, focused FileCheck,
  C++ target/plugin, old-authority scan, and `git diff --check` evidence in the
  task PRD.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind runtime_scalar_dual_cmp_mask_and_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi/focused-dry-run-v1 --overwrite`
- [OK] Manual FileCheck equivalents for
  `rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run.test`
  prefixes `ROOT`, `RSD`, and `HARNESS`.
- [OK] Direct route-entry negative FileCheck for
  `rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-dual-cmp-mask-and-select-fail-closed.test`.
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind runtime_scalar_dual_cmp_mask_and_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi/final-ssh-rvv-v1 --overwrite`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused `REALIZED`, `PLAN`, and `HEADER` FileCheck commands for
  `pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`.
- [OK] Bounded old-authority scan over added tracked diff lines and touched task
  PRD/script/test files.
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete after archive, commit, and clean status verification


## Session 373: Stage2 RVV computed-mask compare-select artifact ABI boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV computed-mask compare-select artifact ABI boundary
**Branch**: `main`

### Summary

Proved `computed_mask_select` selected-body-to-generated-bundle artifact ABI
with provider-derived compare/mask/select evidence, two generated compare-data
runtime patterns, direct route-entry fail-closed regression, focused C++ and
FileCheck verification, and real `ssh rvv` correctness for runtime counts
`0,1,16,17,257`.

### Main Changes

- Created and archived task
  `06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi` with PRD and
  context scoped to exactly one computed-mask compare/select selected body.
- Verified the existing production C++ path carries `computed_mask_select`
  through RVV selected-body realization, computed-mask select route facts,
  operand-binding facts, compare/select statement-plan facts, provider
  preflight, common EmitC, and RVV target artifact bundle export.
- Tightened `scripts/rvv_generated_bundle_abi_e2e.py` so generated
  `computed_mask_select` harnesses execute two compare-data patterns per
  runtime count and expose that requirement in
  `compare_select_predicate_boundary`.
- Added a script `--self-test` regression for the retired direct pre-realized
  `computed_mask_select` route-entry mode.
- Updated the focused generated-bundle dry-run test to check the new
  compare-data pattern evidence and harness fields.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_mask_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi/final-dry-run-v3 --overwrite`
- [OK] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `computed_mask_select`.
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_mask_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-computed-mask-cmp-select-artifact-abi/final-ssh-rvv-v1 --overwrite`
- [OK] Manual FileCheck equivalents for the focused generated-bundle dry-run
  and direct pre-realized fail-closed script tests.
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused REALIZED/PLAN/HEADER FileCheck commands for
  `pre-realized-selected-body-artifact-computed-mask-select.mlir`.
- [OK] Bounded old-authority scan over added lines in touched script/test/task
  files.
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete after archive, commit, and clean status verification


## Session 374: Stage2 RVV runtime-scalar compare-select artifact ABI boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV runtime-scalar compare-select artifact ABI boundary
**Branch**: `main`

### Summary

Proved `runtime_scalar_cmp_select` selected-body-to-generated-bundle ABI with
pre-realized selected-boundary dry-run evidence, direct route-entry
fail-closed regression, focused C++ and FileCheck coverage, and real `ssh rvv`
correctness for runtime counts `0,1,16,17,257` with RHS scalar thresholds
`-500,-37,91`.

### Main Changes

- Created task `06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi` with
  PRD and context scoped to exactly one runtime-scalar compare/select selected
  body.
- Verified the existing production C++ path carries
  `runtime_scalar_cmp_select` through RVV selected-body realization, provider
  route facts, compare/select statement planning, common EmitC, RVV target
  artifact bundle export, and external ABI execution.
- Added a `rvv_generated_bundle_abi_e2e.py --self-test` regression for the
  retired direct pre-realized `runtime_scalar_cmp_select` route-entry mode.
- Repaired the generated runtime-scalar compare/select harness so all-false
  threshold cases are accepted as valid evidence while aggregate coverage still
  requires true lanes, false lanes, and at least one mixed case.

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind runtime_scalar_cmp_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -500 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/final-dry-run-v2`
- [OK] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `runtime_scalar_cmp_select`.
- [OK] Initial real `ssh rvv` run exposed the all-false threshold harness
  overconstraint and was self-repaired.
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind runtime_scalar_cmp_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -500 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/final-ssh-rvv-v2`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused REALIZED/PLAN/HEADER FileCheck checks for
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir`.
- [OK] Bounded old-authority scan over touched script/task files and relevant
  owner/provider/materializer/target/test files.
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete after archive, commit, and clean status verification


## Session 373: Stage2 RVV runtime-scalar masked memory artifact ABI boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV runtime-scalar masked memory artifact ABI boundary
**Branch**: `main`

### Summary

Proved `runtime_scalar_cmp_masked_load_store` selected-body-to-generated-bundle
memory ABI evidence with dry-run evidence, direct route-entry fail-closed
regression, focused C++ and lit/FileCheck tests, and real `ssh rvv`
correctness for runtime counts `0,1,16,17,257` with RHS scalar thresholds
`-500,-37,91`.

### Main Changes

- Created task
  `06-01-06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi`
  with PRD and context scoped to exactly one runtime-scalar computed-mask
  load-store selected body.
- Verified the existing production path carries
  `runtime_scalar_cmp_masked_load_store` through RVV selected-body
  realization, provider route facts, common EmitC, RVV target artifact bundle
  export, and external memory ABI execution.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so runtime-scalar
  computed-mask memory evidence requires at least two RHS scalar thresholds,
  self-test covers the retired direct pre-realized load-store route-entry
  diagnostic, and the generated harness accepts all-inactive threshold cases
  while aggregating mixed-mask and payload-distinguishing evidence.
- Recorded final dry-run, direct fail-closed, `ssh rvv`, focused C++/lit,
  old-authority scan, and `git diff --check` evidence in the task PRD.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -500 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/final-dry-run-v3`
- [OK] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `runtime_scalar_cmp_masked_load_store`.
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -500 --rhs-scalar -37 --rhs-scalar 91 --artifact-root artifacts/tmp/06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/final-ssh-rvv-v2`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused lit/FileCheck checks for
  `pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir`
  and `runtime-scalar-computed-mask-load-store-dataflow.mlir`.
- [OK] Bounded old-authority scan over touched script/task files and relevant
  owner/provider/materializer/target/test files.
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete after archive, commit, and clean status verification


## Session 372: Stage2 RVV computed-mask standalone-reduction artifact ABI boundary

**Date**: 2026-06-01
**Task**: Stage2 RVV computed-mask standalone-reduction artifact ABI boundary
**Branch**: `main`

### Summary

Proved `computed_mask_standalone_reduce_add` selected-body-to-generated-bundle
scalar-result ABI with dry-run evidence, direct route-entry fail-closed
regression, focused C++ tests, and real `ssh rvv` correctness for runtime
counts `0,1,16,17,257` with seeds `-11` and `17`.

### Main Changes

- Created and archived task
  `06-01-06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi`
  with PRD and context scoped to exactly one supported computed-mask
  standalone reduction selected body.
- Verified the existing production path carries
  `computed_mask_standalone_reduce_add` through RVV selected-body realization,
  provider route facts, common EmitC, RVV target artifact bundle export, and
  external scalar-result ABI execution.
- Added a `rvv_generated_bundle_abi_e2e.py --self-test` regression for the
  retired direct pre-realized `computed_mask_standalone_reduce_add`
  route-entry mode.
- Recorded final dry-run, direct fail-closed, `ssh rvv`, focused binary,
  old-authority scan, and `git diff --check` evidence in the archived PRD.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_mask_standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi/final-dry-run`
- [OK] Direct route-entry negative command exited 1 with the expected retired
  direct route-entry diagnostic for `computed_mask_standalone_reduce_add`.
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_mask_standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-computed-mask-standalone-reduction-artifact-abi/final-ssh-rvv`
- [OK] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Bounded old-authority scan over touched script/task files and relevant
  owner/provider/materializer/target/test files.
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete after archive, commit, and clean status verification
