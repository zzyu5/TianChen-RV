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


## Session 95: Plugin-owned source seed registration interface

**Date**: 2026-05-16
**Task**: Plugin-owned source seed registration interface
**Branch**: `main`

### Summary

Moved the RVV bounded selected-boundary source seed pass behind a common
extension-plugin source-seed registration surface. `tcrv-opt` now registers the
seed by collecting enabled plugin registrations instead of directly including
or invoking the RVV pass factory.

### Main Changes

- Created and archived Trellis task
  `05-16-plugin-owned-source-seed-registration-interface` from the supplied
  Direction Brief.
- Added `SourceSeedPassRegistration` and
  `ExtensionPluginRegistry::collectSourceSeedPasses` as common pass
  descriptor/factory plumbing.
- Added `ExtensionPlugin::registerSourceSeedPasses` with default no-op
  behavior, and moved the existing RVV i32m1 selected-boundary seed factory
  behind `RVVExtensionPlugin`.
- Removed direct `RVVSelectedBoundarySeed.h` include and direct RVV seed pass
  factory registration from `tools/tcrv-opt/tcrv-opt.cpp`.
- Removed the direct `TianChenRVRVVPlugin` link from `tcrv-opt`; the tool now
  reaches built-in RVV behavior through built-in plugin/target registration.
- Added registry and RVV plugin C++ coverage for source-seed pass registration,
  plus lit coverage proving `--tcrv-disable-builtin-plugins` leaves the RVV
  seed pass unregistered.
- Promoted the source-seed pass provider interface rules into
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | plugin(registry): route source seeds through plugins |

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVPlugin TianChenRVRVVPlugin tcrv-opt
  tcrv-translate tianchenrv-plugin-registry-test
  tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] C++ tests:
  `tianchenrv-plugin-registry-test`,
  `tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit from `build/test`:
  `rvv-i32m1-selected-boundary-seed|rvv-first-slice-materialization|Target/RVV/i32m1|rvv-with-vl-selected-boundary`,
  19/19 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 104/104
  passed.
- [OK] `git diff --check`.
- [OK] Public-tool direct RVV seed wiring scan returned no matches for
  `RVVSelectedBoundarySeed` or
  `createMaterializeRVVI32M1SelectedBoundarySeedPass` under `tools/`.

### Hardware Evidence

Not rerun. This round changed pass registration ownership only; selected
boundary output, emission-plan/EmitC route semantics, target artifact semantics,
and runtime ABI shape are unchanged and covered by focused lit. The previous
seed task already produced `ssh rvv` correctness evidence for the same route.

### Self-Repair

- First focused build exposed an incomplete `mlir::Pass` type in
  `RVVExtensionPlugin.cpp`; added `mlir/Pass/Pass.h` and reran successfully.
- First lit attempt used `lit.py build/test` from repo root and hit this repo's
  generated relative `lit.site.cfg.py` path. Reran from `build/test`, which is
  the project convention, and the focused set passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 94: Bounded MLIR-to-RVV selected-boundary seed

**Date**: 2026-05-16
**Task**: Bounded MLIR-to-RVV selected-boundary seed
**Branch**: `main`

### Summary

Added an RVV-owned bounded vector i32 add source seed pass that materializes selected RVV i32m1 boundary IR, covered positive/negative lit, generated target artifacts, and passed ssh rvv correctness plus full check-tianchenrv.

### Main Changes

- Created and archived Trellis task
  `05-16-bounded-mlir-to-rvv-selected-boundary-seed` from the supplied
  Direction Brief.
- Added RVV plugin-owned pass
  `--tcrv-rvv-materialize-i32m1-selected-boundary-seed`, registered through
  `tcrv-opt`, for one explicitly marked `func`/`scf`/`vector`/`arith` i32 add
  source shape.
- The pass materializes `tcrv.exec.kernel`, a selected `origin = "rvv-plugin"`
  variant, explicit `lhs`/`rhs`/`out`/`n` runtime ABI bindings,
  `tcrv_rvv.setvl`, selected `tcrv_rvv.with_vl`, and RVV
  `i32_load` / `i32_add` / `i32_store`.
- Added positive and fail-closed lit coverage for selected-boundary output,
  emission-plan/EmitC route consumption, missing ABI operands, unsupported
  dtype/rank/vector shape, malformed source body, and stale
  `tcrv.exec`/`tcrv_rvv` residue.
- Promoted the bounded seed command/input/output/error contract into
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Generated seed object/header artifacts under
  `artifacts/tmp/rvv_selected_boundary_seed/20260516T132508Z` and ran a real
  `ssh rvv` link/run harness.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Focused build for RVV construction protocol, RVV EmitC route provider,
  RVV plugin, transforms, `tcrv-opt`, `tcrv-translate`, and focused C++ tests.
- [OK] C++ tests:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-rvv-dialect-test`.
- [OK] Focused lit filter:
  `rvv-i32m1-selected-boundary-seed|rvv-first-slice-materialization|Target/RVV/i32m1|rvv-with-vl-selected-boundary`,
  19/19 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 104/104
  passed.
- [OK] `ssh rvv` seed harness printed
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.
- [OK] `git diff --check`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete
