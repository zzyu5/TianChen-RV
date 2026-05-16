# Journal - codex (Part 8)

> Continuation from `journal-7.md` (archived at ~2000 lines)
> Started: 2026-05-16

---



## Session 93: RVV construction-backed hardware artifact proof

**Date**: 2026-05-16
**Task**: RVV construction-backed hardware artifact proof
**Branch**: `main`

### Summary

Refreshed current-HEAD proof that RVV i32m1 add/sub/mul selected paths cross construction-checked EmitC, target object/header export, bundle metadata, and real ssh rvv correctness evidence without requiring source rewiring.

### Main Changes

- Created and archived Trellis task `05-16-rvv-construction-backed-hardware-artifact-proof` from the Hermes direction brief.
- Wrote the PRD around current-HEAD proof of the existing RVV i32m1 add/sub/mul route instead of expanding RVV coverage.
- Confirmed no source-code repair was required: RVV construction mapping, route provider validation, selected `tcrv_rvv.with_vl`, common selected EmitC artifact materialization, object/header export, and translate routes are already wired.
- Generated current add/sub/mul default and exact object/header artifacts plus bundle indexes under `artifacts/tmp/rvv_construction_backed_hardware_artifact_proof/20260516T125646Z`.
- Bundle metadata records route ids, callable component groups, `plugin-owned-runtime-abi`, and ordered `lhs`, `rhs`, `out`, `n` target-export ABI parameters.
- Linked and ran the generated add/sub/mul objects on `ssh rvv` through `rvv_i32m1_arithmetic_harness.c`; remote output was `tcrv_rvv_i32m1_arithmetic_current_head status=PASS n=4 add=[12,6,16,12] sub=[2,-12,24,0] mul=[35,-27,-80,36]`.
- Focused build, RVV plugin/unit tests, target artifact export test, construction/dialect tests, focused RVV lit set, full `check-tianchenrv`, `git diff --check`, and legacy/common-core scans passed.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | task(rvv): record construction-backed hardware proof |

### Testing

- [OK] Focused build for RVV construction protocol, RVV EmitC route provider,
  RVV plugin, RVV target support, target artifact export, `tcrv-opt`, and
  `tcrv-translate`.
- [OK] RVV plugin/unit tests, target artifact export test, construction
  protocol test, and RVV dialect test.
- [OK] Focused lit filter for RVV EmitC materialization, Target/RVV i32m1, and
  selected with_vl boundary tests: 17/17 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 102/102 passed.
- [OK] `ssh rvv` add/sub/mul external ABI harness passed.
- [OK] `git diff --check`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete
