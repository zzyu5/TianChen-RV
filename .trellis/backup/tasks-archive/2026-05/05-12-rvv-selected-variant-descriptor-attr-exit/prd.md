# RVV selected variant descriptor attribute exit

## Goal

Remove `tcrv_rvv.lowering_descriptor` and selected descriptor mirror metadata
as default-path selected-family authority for bounded RVV binary selected
variants. The production/default route must be driven by typed RVV family
metadata, selected vector config metadata, typed RVV microkernel ops, and the
common EmitC/body materialization authority. Any descriptor metadata that
remains in this round is explicit legacy/quarantine mirror data and cannot
choose family, dtype, route, ABI, materialized body, or exported source.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean at task creation.
- Task start HEAD is `86a6c45 feat(rvv): exit descriptor compute api`.
- `.trellis/.current-task` was absent at session start, so this task was
  created as `.trellis/tasks/05-12-rvv-selected-variant-descriptor-attr-exit/`.
- The archived predecessor
  `.trellis/tasks/archive/2026-05/05-12-rvv-descriptor-compute-api-exit/`
  removed descriptor compute helper APIs and default descriptor compute
  provenance.
- Current specs already state that legacy `tcrv_rvv.lowering_descriptor`, when
  present beside typed RVV binary authority, is non-authoritative mirror
  metadata only.
- Live inventory shows default route progress already exists: proposals for
  typed i32/i64 RVV binary families no longer attach
  `tcrv_rvv.lowering_descriptor` by default, and frontend tests already carry
  `CHECK-NOT` coverage for default artifacts.
- Remaining live descriptor surfaces include default-path validation and
  metadata surfaces in RVV planning, variant legality, selected-boundary
  materialization, selected emission planning, target export validation, tests,
  and local e2e scripts.

## Initial Inventory Classification

- Default-path / migration target:
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp` still has selected
    plan metadata code that can append
    `tcrv_rvv.selected_lowering_descriptor` when the selected variant carries a
    legacy descriptor mirror.
  - `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h` exposes
    `appendRVVBinaryLegacyDescriptorMirrorMetadata`, including
    `tcrv_rvv.selected_lowering_descriptor`; this must remain mirror-only and
    must not be required by typed routes.
  - `scripts/rvv_microkernel_e2e.py` still validates
    `tcrv_rvv.selected_lowering_descriptor` as selected-family evidence; if
    touched, it must move to typed family metadata.
- Legacy mirror / quarantine:
  - `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
    `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp`,
    `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
    `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`, and
    `lib/Target/RVV/RVVMicrokernel.cpp` validate
    `tcrv_rvv.lowering_descriptor` only after typed family/body authority or
    reject descriptor-only paths.
  - Existing negative tests under `test/Plugin/` and
    `test/Target/RVVMicrokernel/` use `lowering_descriptor` to prove stale
    mirrors or descriptor-only paths fail closed.
- Test-only default absence coverage:
  - Default frontend and artifact tests under `test/Transforms/LinalgToExec/`,
    `test/Target/RVVMicrokernel/`, and `test/Target/RVVScalarDispatch/`
    already contain `CHECK-NOT` / `IR-NOT` coverage for default descriptor
    absence.
- Out of immediate scope unless directly required:
  - `RVVBinaryDescriptor` / `RVVBinaryFamilyDescriptor` registration records
    still contain finite route strings for route registration, intrinsic
    spelling, ABI names, and explicit legacy mirror validation. They are not
    removed wholesale in this round.

## Requirements

- Keep default RVV i32 add/sub/mul selected variants descriptorless:
  `tcrv_rvv.lowering_descriptor` must not be emitted or required as default
  selected-family identity when typed family/source metadata is present.
- Ensure selected emission planning, target artifact route preflight, and
  default exported artifact metadata consume typed family/source metadata:
  `tcrv_rvv.selected_binary_dtype`,
  `tcrv_rvv.selected_binary_family`,
  `tcrv_rvv.selected_binary_operator`,
  `tcrv_rvv.emitc_source_op`,
  `tcrv_rvv.emitc_lowerable_op_interface`, selected vector config metadata,
  and IR-backed runtime ABI metadata.
- Do not require `tcrv_rvv.selected_lowering_descriptor` for typed default RVV
  binary source/header/object routes.
- Keep fail-closed behavior for legacy mirrors:
  matching descriptor mirrors may be accepted only as explicit compatibility
  mirrors, stale mirrors must fail before export, and descriptor-only selected
  variants must not become legal or exportable.
- Update tests that still present descriptor mirror metadata as default
  selected-family identity. Remaining descriptor tests must be explicitly
  legacy/quarantine/fail-closed in name, comments, or diagnostics.
- Keep `tcrv.exec` compute-free and keep all compiler implementation in the
  C++/MLIR/LLVM/TableGen/CMake stack. Python is tooling/evidence only.

## Acceptance Criteria

- [x] Focused inventory over `include/`, `lib/`, `test/`, and `scripts/`
      classifies remaining `lowering_descriptor`,
      `selected_lowering_descriptor`, `RVVBinaryDescriptor`, and
      `getLoweringDescriptor` hits as legacy/quarantine, registration, or
      test-only.
- [x] Default RVV i32 add/sub/mul proposals, selected boundaries, emission
      plans, materialized typed microkernel bodies, and source/header/object
      export candidates do not need `tcrv_rvv.lowering_descriptor`.
- [x] Default typed RVV selected-plan metadata does not require
      `tcrv_rvv.selected_lowering_descriptor`; typed family/source metadata and
      selected vector config metadata are sufficient.
- [x] A stale or unsupported legacy descriptor mirror beside typed RVV family
      authority still fails closed before materialization/export.
- [x] Descriptor-only RVV binary variants remain unsupported/quarantined and do
      not manufacture legal selected-family, materialized body, emission plan,
      or target artifact authority.
- [x] Focused C++ and lit tests cover default descriptor absence and legacy
      descriptor mirror quarantine for the touched surfaces.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Trellis validation passes before finish/archive.
- [x] Focused build/test targets pass first; `check-tianchenrv -j2` is run if
      focused checks pass and the existing build tree is usable.
- [x] One coherent commit records the completed module, or the task remains
      open with an exact continuation point.

## Non-Goals

- No new RVV family, dtype, LMUL, broad matrix, benchmark, performance claim,
  runtime claim, or generic RVV backend claim.
- No descriptor-to-C exporter and no computation semantics through descriptors.
- No compute semantics in `tcrv.exec`.
- No IME, AME, Sophgo/offload, TensorExt, Template, Toy, or unrelated scalar
  redesign.
- No Python compiler internals. Python may run existing probes or artifact
  parsers only when needed.
- No ssh-only, helper-only, report-only, metadata-only, wrapper-only, or broad
  smoke-test closeout.

## Minimal Validation

- Run focused source inventory for:
  `lowering_descriptor`, `selected_lowering_descriptor`,
  `RVVBinaryDescriptor`, `getLoweringDescriptor`, and descriptor vocabulary in
  `include`, `lib`, `test`, and `scripts`.
- Build focused targets touched by the module, at minimum available RVV plugin
  and target/export targets such as `TianChenRVRVVPlugin`,
  `TianChenRVRVVTarget`, `TianChenRVBuiltinTargetArtifactExporters`,
  `tcrv-opt`, `tcrv-translate`, and focused C++ tests.
- Run focused C++ tests for RVV binary planning, RVV binary variant legality,
  RVV selected lowering boundary, RVV extension plugin behavior, target
  artifact export, and RVV/scalar dispatch if touched.
- Run focused lit/FileCheck for default RVV microkernel artifacts,
  RVV+scalar dispatch artifacts, and frontend-to-artifact routes touched by
  the attribute exit. Preserve or add `CHECK-NOT` coverage for default
  absence of `lowering_descriptor`.
- Run `git diff --check` and `git diff --cached --check`.
- Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-12-rvv-selected-variant-descriptor-attr-exit`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target
  check-tianchenrv -j2` if focused checks pass and the build tree is usable.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Predecessor PRDs read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-descriptor-compute-api-exit/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-rvv-self-check-typed-authority/prd.md`
- Initial source focus:
  - `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h`
  - `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVBinarySelectedLoweringBoundary.h`
  - `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
  - `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp`
  - `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `test/Plugin/RVVBinaryPlanningTest.cpp`
  - `test/Plugin/RVVBinaryVariantLegalityTest.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - `test/Target/RVVMicrokernel/`
  - `test/Target/RVVScalarDispatch/`
  - `test/Transforms/LinalgToExec/`

## Completion Notes

- Removed the implicit RVV legality fallback that treated a descriptorless
  selected variant with only `tcrv_rvv.element_count` as default i32-vadd
  authority.
- Default descriptorless i32-vadd legality now requires typed
  selected-source metadata, kernel frontend authority, or an actual typed RVV
  microkernel body.
- Updated the selected-boundary default fixture to carry typed
  selected-source metadata.
- Updated RVV direct and RVV+scalar e2e tooling samples to use typed RVV/scalar
  selected-family metadata roles instead of selected descriptor role names.
- Remaining `selected_lowering_descriptor` hits are legacy mirror helpers,
  scalar legacy tests, RVV negative/fallback tests, frontend rejection tests, or
  stale artifact fixture data.
- No `ssh rvv` runtime, correctness, or performance claim was made.

## Definition Of Done

The task is complete only when default RVV binary selected-family semantics no
longer depend on `tcrv_rvv.lowering_descriptor` or
`tcrv_rvv.selected_lowering_descriptor`, typed family/source metadata plus
typed RVV body and common EmitC/body authority drive the production route,
remaining descriptor hits are explicitly legacy/quarantine/registration/test
surfaces, focused and available broad checks pass, Trellis state is truthful,
and one coherent commit records the module. If unfinished, leave the task open
and record the exact continuation point.
