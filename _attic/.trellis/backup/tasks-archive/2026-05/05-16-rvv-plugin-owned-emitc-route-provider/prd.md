# RVV plugin-owned EmitC route provider

## Goal

Move the bounded RVV i32m1 add EmitC route ownership out of RVV target artifact
packaging and into an RVV plugin-owned emission-route provider consumed by the
common selected EmitC artifact front door. Target/RVV remains responsible for
artifact exporter registration, clang/object/header packaging, and export
policy; it must not remain the owner of RVV compute-route semantics.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD is `5ec61f9 target: add common selected emitc artifact front door`.
- The previous completed path made common selected emission-plan to EmitC/C++
  artifact export real and had RVV consume that front door.
- `.trellis/spec/index.md` defines the main lowering route as extension family
  ops -> EmitC ops -> intrinsic/vendor builtin/runtime C/C++ -> native compiler.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires selected artifact
  exporters to consume selected supported emission-plan candidates, not infer a
  target by scanning direct variants.
- `.trellis/spec/extension-plugins/rvv-plugin.md` defines the bounded route as
  selected RVV path -> explicit typed `tcrv_rvv` i32m1 add body -> RVV-owned
  EmitC lowerable route -> MLIR EmitC C/C++ emitter -> clang object/header
  bundle.
- `.trellis/spec/testing/mlir-testing-contract.md` requires local lit/FileCheck
  coverage for selected RVV explicit-op EmitC materialization and fail-closed
  unsupported shapes; RVV runtime/correctness claims require separate `ssh rvv`
  evidence only when ABI/runtime behavior changes.

## Requirements

- Add or expose an RVV plugin-owned provider/interface for the bounded i32m1 add
  EmitC lowerable route.
- The provider owns RVV typed-body shape validation, source-op provenance,
  RVV intrinsic/header names, ABI value mapping, and route construction for the
  current bounded i32m1 add slice.
- The common selected EmitC artifact front door calls the provider through a
  generic/plugin-facing boundary and remains RVV-agnostic.
- RVV target support visibly thins down to exporter registration, selected
  candidate consumption, artifact kind registration, clang/object/header
  packaging, and target export policy.
- Unsupported RVV shapes fail closed from the RVV plugin-owned provider before
  object/header/bundle output.
- Existing selected dispatch and selected-path sibling RVV artifact tests must
  still pass through the common front door.
- No descriptor-driven computation, direct C semantic exporter, binary-family
  registry, GCC-default path, Python compiler-core logic, compatibility wrapper,
  or extension-specific semantic branch in common/core orchestration may be
  restored.

## Acceptance Criteria

- [ ] `test/Target/RVV/i32m1-selected-dispatch-artifact.mlir` still passes.
- [ ] `test/Target/RVV/i32m1-selected-path-sibling-artifact.mlir` still passes.
- [ ] Ambiguous/unselected selected-artifact fail-closed coverage still passes.
- [ ] Unsupported RVV shape coverage fails closed through the plugin-owned route
  provider.
- [ ] `TargetArtifactExport` common code contains no RVV intrinsic/header names
  such as `riscv_vector.h` or `__riscv_`.
- [ ] RVV target support no longer embeds the bounded body validation and route
  semantic mapping directly; it delegates to the RVV plugin-owned provider.
- [ ] Focused C++ target artifact export coverage passes.
- [ ] `check-tianchenrv` runs if practical; otherwise the final report explains
  the blocker and gives the focused evidence that did run.
- [ ] If generated RVV C/header/object ABI changes, rerun the selected-dispatch
  `ssh rvv` link/run harness. If ABI bytes/signatures do not change, state why
  previous ssh evidence remains applicable and do not claim new runtime
  evidence.

## Out of Scope

- New RVV dtype, LMUL, op families, or general RVV lowering.
- MLIR vector lowering, TensorExt/IME/offload implementation, Toy executable
  routes, or performance matrices.
- Descriptor or binary-family registries, direct C compute printers, Python
  compiler-core logic, GCC-default routes, or compatibility layers.
- Extension-specific semantic branches in common/core orchestration.
- Treating prompt/report/helper-only changes or broad smoke tests as the main
  achievement.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/plugin-protocol/index.md`
- Initial files to inspect:
  - `include/TianChenRV/Conversion/EmitC/`
  - `lib/Conversion/EmitC/`
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `include/TianChenRV/Target/TargetArtifactExport.h`
  - `lib/Target/TargetArtifactExport.cpp`
  - `test/Target/RVV/i32m1-selected-dispatch-artifact.mlir`
  - `test/Target/RVV/i32m1-selected-path-sibling-artifact.mlir`

## Definition of Done

- Code implements the provider ownership boundary in C++/MLIR stack.
- Focused lit and C++ checks pass or failures are explained as real blockers.
- Changed-surface scan proves descriptor/direct-C/source-export legacy terms
  were not restored and common target code does not contain RVV route spellings.
- Trellis task status/context is truthful and the completed round is archived.
- One coherent git commit records the implementation.
