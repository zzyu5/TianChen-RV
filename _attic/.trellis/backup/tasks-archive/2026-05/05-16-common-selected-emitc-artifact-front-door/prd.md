# Common selected-emission-plan EmitC artifact front door

## Goal

Extract a common target artifact front door for the selected emission-plan to
EmitC artifact workflow, with the existing RVV bounded i32m1 add
object/header bundle as the first consumer.

The common layer should own generic selected `TargetArtifactCandidate`
validation/selection handoff and the reusable materialized EmitC/C++ emission
orchestration. RVV target support should remain responsible for RVV slice
legality, intrinsic names, ABI details, clang flags, route registration, and
object/header packaging details.

## What I Already Know

- Current HEAD is `50108a8`, which made RVV artifact export consume the
  selected execution-plan/emission-plan boundary.
- The existing RVV i32m1 add object/header bundle already works through the
  selected dispatch and selected-path sibling artifact tests.
- The next bottleneck is code shape: selected candidate resolution, route
  validation, EmitC materialization, C/C++ emission, object compile, and
  header packaging still sit together in the RVV target-support slice.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires target artifact
  exporters to select supported emission-plan candidates from the current
  selected dispatch/path surface, and forbids selecting a variant by scanning
  for exactly one direct variant.
- `.trellis/spec/extension-plugins/rvv-plugin.md` keeps RVV as a plugin-owned
  first executable slice, not an independent backend or generic source printer.
- `.trellis/spec/testing/mlir-testing-contract.md` requires lit/FileCheck and
  C++ coverage for MLIR/compiler behavior, and `ssh rvv` only for runtime,
  correctness, or performance claims.

## Requirements

- Add a common C++ API/submodule for selected emission-plan artifact export.
- Move generic selected `TargetArtifactCandidate` filtering/validation and
  materialized EmitC/C++ emitter orchestration out of RVV-specific target
  support.
- Keep all RVV-specific semantics local to RVV target/plugin code:
  intrinsic names, i32m1 add shape legality, external ABI spelling, headers,
  clang/RISC-V flags, object/header route metadata, and packaging details.
- Preserve the existing RVV selected dispatch and selected-path sibling
  object/header bundle metadata and callable ABI surface.
- Preserve fail-closed behavior for ambiguous supported candidates,
  unselected/missing selected paths, unknown/unsupported routes, unsupported
  RVV shapes, stale runtime ABI metadata, and malformed runtime ABI parameters.
- Ensure common code has no RVV intrinsic names and no family-specific semantic
  branches.

## Acceptance Criteria

- [x] RVV selected dispatch artifact export still produces the same
      object/header bundle metadata and callable ABI surface through the common
      front door.
- [x] RVV selected-path sibling artifact export still ignores non-selected
      sibling variants.
- [x] Ambiguous supported candidates, unselected inputs, unsupported selected
      RVV shapes, and malformed runtime ABI metadata still fail closed before
      artifact output.
- [x] Common code owns reusable selected-candidate validation/handoff and
      EmitC/C++ emission orchestration.
- [x] RVV target support is smaller or clearer because it provides only
      plugin-owned mapping, validation, route registration, and target package
      details.
- [x] A changed-surface scan confirms descriptor/direct-C/source-export legacy
      terms were not restored as compiler authority.

## Out Of Scope

- New RVV dtype, LMUL, or op families.
- General RVV lowering, MLIR vector lowering, TensorExt, IME, Offload, or new
  frontend lowering.
- Descriptor or binary-family registries.
- Descriptor-driven computation or direct C compute printers.
- Python compiler-core logic.
- GCC-default routes or compatibility wrappers.
- Extension-specific semantic branches in common/core orchestration.
- New performance matrices or broad runtime claims.

## Validation Plan

- Build focused compiler/export targets.
- Run focused lit/FileCheck coverage for:
  - RVV selected dispatch artifact export;
  - RVV selected-path sibling artifact export;
  - ambiguous supported candidate fail-closed behavior;
  - unselected/missing selected-path fail-closed behavior;
  - unsupported selected RVV shape fail-closed behavior.
- Run target artifact export C++ unit coverage.
- Run `check-tianchenrv` if practical.
- If artifact bytes, ABI, or header output change, rerun the selected-dispatch
  `ssh rvv` link/run harness. If they do not change, report that prior ssh
  evidence remains applicable because this round only moved common
  orchestration.
- Run a changed-surface scan for descriptor/direct-C/source-export residue.

## Validation Results

- `ninja -C build tianchenrv-target-artifact-export-test tcrv-translate tcrv-opt`
  passed after one self-repair for a missing test include.
- `build/bin/tianchenrv-target-artifact-export-test` passed after one
  self-repair to make the mock compute EmitC step produce a result value.
- `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'Target/RVV/(i32m1-selected-dispatch-artifact|i32m1-selected-path-sibling-artifact|i32m1-artifact-ambiguous-selected|i32m1-artifact-unselected-multivariant|i32m1-object-unsupported-shape|emitc-to-cpp)'`
  passed 7/7 focused tests.
- `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'Target/ArtifactExport/target-artifact-export-registry'` passed 1/1 focused
  registry test.
- `ninja -C build check-tianchenrv` passed 82/82 tests.
- `git diff --check` passed.
- `git diff -- include lib test | rg -n
  "descriptor|direct-C|direct C|source-export|source export|microkernel" ||
  true` returned no changed-source matches.
- `rg -n "__riscv|riscv_vector|rvv|RVV"
  include/TianChenRV/Target/TargetArtifactExport.h
  lib/Target/TargetArtifactExport.cpp || true` returned no matches, confirming
  the common front door did not gain RVV intrinsic names or RVV semantic
  branches.
- No new `ssh rvv` run was required in this round: the RVV intrinsic route
  builder, clang flags, runtime ABI metadata, header text contract, and object
  packaging path remain RVV-owned and behaviorally unchanged; local focused
  tests regenerated and verified the selected dispatch/sibling object/header
  bundle surface through the common orchestration.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/plugin-protocol/index.md`
- Relevant code:
  - `include/TianChenRV/Conversion/EmitC/`
  - `lib/Conversion/EmitC/`
  - `include/TianChenRV/Target/TargetArtifactExport.h`
  - `lib/Target/TargetArtifactExport.cpp`
  - `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- Relevant tests:
  - `test/Target/RVV/i32m1-selected-dispatch-artifact.mlir`
  - `test/Target/RVV/i32m1-selected-path-sibling-artifact.mlir`
