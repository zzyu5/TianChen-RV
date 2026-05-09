# RVV capability-driven vector-shape proposal and selection

## Goal

Move the bounded RVV i32 binary `i32m1` / `i32m2` choice one step further upstream in the C++ compiler path. The RVV plugin must select a finite vector-shape config from structured target capabilities, validate it against the target/RVV descriptor-owned shape registry, carry the selected config capability refs through proposal/materialization/legality, and preserve the selected shape through lowering-boundary and target/export consumption.

## What I Already Know

- The previous descriptor registry task is complete and archived at commit `3e9e2af`; this task must not reopen it.
- `RVVVectorShape.h` already owns finite `i32m1` / `i32m2` shape metadata and selected-plan metadata descriptors.
- `RVVI32BinaryDescriptor.h` already composes i32 add/sub/mul family descriptors with selected RVV vector-shape descriptors for target/export use.
- `RVVExtensionPlugin.cpp` already emits selected-shape variant metadata, selected lowering-boundary metadata, and i32 add/sub/mul microkernel ops.
- The current weak point is selection: when no explicit required config is supplied, `selectAvailableFirstSliceConfigCapabilities` chooses by availability order (`i32m1` before `i32m2`). That does not let a structured capability say "select i32m2" when both finite config shapes are available.
- Core transforms such as `VariantSelection.cpp`, `PluginVariantLegality.cpp`, `ExecutionPlanningPipeline.cpp`, and `LoweringBoundary.cpp` already route through generic plugin/capability interfaces and must remain RVV-shape agnostic.

## Boundaries

- Compiler implementation remains C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck.
- Python remains out of compiler decisions.
- Keep family and vector-shape semantics RVV plugin/target-local.
- Do not add new dtype families, floating point, reductions, broad RVV operation coverage, performance tuning, or a new runtime ABI.
- Do not add RVV-specific branches to generic transforms.
- Do not commit build outputs, `artifacts/tmp`, raw ssh logs, secrets, or transient generated files.

## Requirements

1. Add a bounded RVV plugin/target-owned capability contract for selecting the finite i32 binary vector shape when target capabilities expose more than one valid shape.
2. RVV plugin proposal selection must read that structured capability fact, validate the selected shape key against the finite descriptor registry, and then require the selected shape's SEW/LMUL/tail/mask capability ids.
3. If a selector capability requests `i32m2` while the target lacks any required `i32m2` config fact, proposal must decline/fail closed instead of silently falling back to `i32m1`.
4. If no selector capability is present, preserve the existing bounded default behavior: choose a single available finite config shape, with existing `i32m1` fixtures remaining valid.
5. Materialized variant legality, selected lowering-boundary materialization, emission plans, and target/export consumption must keep using selected variant metadata and required capability refs rather than recomputing from helper flags.
6. Use the existing descriptor APIs where reasonable so proposal requirements and selected-shape metadata stay consistent with the target/export descriptor path.

## Acceptance Criteria

- [ ] A positive selected `i32m2` case proves both `i32m1` and `i32m2` capability facts may be present, while a structured selector capability drives the proposal to `i32m2`.
- [ ] A positive/default `i32m1` path still works without a selector capability.
- [ ] A missing-capability case proves selector-requested `i32m2` fails closed when the `i32m2` capability set is incomplete, even if `i32m1` facts are available.
- [ ] Planned IR / lowering-boundary checks show selected variant metadata, selected vector-shape metadata, required capability refs, and generated RVV microkernel agree.
- [ ] Existing `i32-vsub` + `i32m2` and `i32-vmul` + `i32m2` artifact/export paths are not regressed.
- [ ] Focused C++ and lit checks pass, plus `git diff --check` and affected tool build.

## Minimal Validation

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused lit/FileCheck tests for RVV plugin proposal/materialization, missing selected-shape capability rejection, selected lowering-boundary metadata, and one descriptor/export regression path.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` if practical after focused checks.

## Out Of Scope

- No performance benchmark or new `ssh rvv` correctness claim unless runtime-visible artifact behavior changes materially.
- No broad matrix run unless focused checks expose systemic breakage.
- No new core `tcrv.exec` compute semantics.
- No descriptor-only or evidence-only closeout.

## Technical Notes

- Relevant specs are listed in `implement.jsonl` and `check.jsonl`.
- Relevant source surfaces:
  - `include/TianChenRV/Target/RVV/RVVVectorShape.h`
  - `include/TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h`
  - `include/TianChenRV/Plugin/RVV/RVVCapabilityProfile.h`
  - `lib/Plugin/RVV/RVVCapabilityProfile.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - focused tests under `test/Plugin/`, `test/Transforms/`, and `test/Target/`
