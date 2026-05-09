# rvv add sub artifact metadata coherence

## Goal

Make the generic target artifact front doors and bundle or manifest metadata family-aware for the two existing RVV i32 microkernel family members, i32-vadd and i32-vsub, without moving computation into `tcrv.exec` or adding Python compiler internals.

## Requirements

* Preserve the project spine: high-level or selected-boundary TianChen-RV MLIR flows through capability, plugin variant proposal, legality, selection/dispatch, plugin-owned lowering/emission/runtime glue, then executable RVV/IME/offload/fallback paths.
* Support selected i32-vsub RVV boundary export through the same generic target source, header, object, and bundle/manifest surfaces as i32-vadd.
* Ensure source export emits the selected RVV family C intrinsic: vsub must emit `__riscv_vsub_vv_i32m1`, not add.
* Ensure header and bundle/manifest metadata expose selected-family route, component group, external ABI name, runtime ABI kind/name, and runtime ABI parameters.
* Ensure selected vsub records do not inherit vadd external ABI name or vadd component group.
* Keep existing i32-vadd generic and direct route behavior working, preserving old route IDs or providing backward-compatible aliases with tests.
* Keep generic target artifact code target-neutral. RVV-specific family details must come from target/plugin-owned registration, descriptors, or callbacks.
* If object export needs compiler discovery to be testable, keep the improvement bounded to the target-owned object exporter path and diagnostics precise.

## Acceptance Criteria

* [ ] `tcrv-opt --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans` followed by `tcrv-translate --tcrv-export-target-source-artifact` on selected i32-vsub produces source containing `__riscv_vsub_vv_i32m1` and not an add intrinsic for the arithmetic step.
* [ ] The same selected i32-vsub surface flows through `tcrv-translate --tcrv-export-target-header-artifact` with vsub runtime ABI name and vsub external ABI/component metadata.
* [ ] The same selected i32-vsub surface flows through `tcrv-translate --tcrv-export-target-artifact` to a non-empty object when a local compiler is available, or fails with a precise missing compiler/tool diagnostic that preserves the selected vsub family in source/header/manifest evidence.
* [ ] Existing i32-vadd generic front-door behavior still passes.
* [ ] Bundle or manifest coverage shows vsub route/component group/external ABI name/runtime ABI name are vsub-specific.
* [ ] One focused mismatch test rejects selected vsub paired with add ABI/component metadata or route before producing a mislabeled bundle.
* [ ] `git diff --check` passes.
* [ ] Configure and build check pass with LLVM/MLIR 20: `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir` and `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.

## Definition of Done

* Compiler/exporter changes are implemented in C++/MLIR/TableGen/CMake/lit/FileCheck only, except Trellis/task support files.
* Focused tests prove compiler behavior through generic target artifact front doors and manifest or bundle records.
* No ssh RVV runtime correctness or performance claim is made unless fresh `ssh rvv` evidence is intentionally collected and saved under `artifacts/tmp`.
* Trellis task state is validated and archived if the workflow supports it.
* One coherent commit is created and the worktree is clean.

## Out of Scope

* Adding a third arithmetic operation, i64/e64, masks, tail policy expansion, dynamic-shape lowering, or a new frontend.
* Requiring high-level linalg/stablehlo/tosa lowering before this selected-boundary plugin/export path works.
* Moving computation into `tcrv.exec`.
* Adding Python implementations of compiler internals, target artifact selection, RVV semantics, plugin registry, lowering, emission, or object export.
* Claiming RVV runtime correctness, performance, or generic backend capability without fresh ssh RVV evidence.

## Technical Notes

* User-provided expected starting point: HEAD includes `5daebe3 feat(rvv): generalize microkernel family export`.
* Main implementation area: target/exporter and focused tests.
* Likely files include `lib/Target/RVV/RVVMicrokernel.cpp`, `include/TianChenRV/Target/RVV/RVVMicrokernel.h`, `include/TianChenRV/Target/TargetArtifactExport.h`, `lib/Target/TargetArtifactExport.cpp`, `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`, `tools/tcrv-translate/tcrv-translate.cpp`, and focused tests under `test/Target` and `test/Transforms/EmissionReadiness`.
* Required first inspection list comes from the user request and must be completed before code edits.
