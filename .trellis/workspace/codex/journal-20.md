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
