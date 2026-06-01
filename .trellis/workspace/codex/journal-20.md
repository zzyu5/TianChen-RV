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
