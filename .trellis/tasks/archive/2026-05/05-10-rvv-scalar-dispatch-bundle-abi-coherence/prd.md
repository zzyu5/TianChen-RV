# RVV scalar dispatch bundle ABI coherence

## Goal

Make the composite RVV+scalar dispatch export path validate and carry one IR-backed runtime ABI contract from selected dispatch metadata through RVV/scalar child callable plans, target artifact records, generated dispatch source/header/object helpers, and bundle index records for the currently supported finite RVV binary families.

This task follows commit `46f378b`, where direct selected RVV binary microkernel export started validating candidate runtime ABI metadata against `tcrv.exec.mem_window` and `tcrv.exec.runtime_param`. The next module boundary is the composite dispatch path: top-level dispatch ABI, RVV child ABI, scalar fallback ABI, manifest route records, and exported source/header/object records must agree before artifact or bundle output is accepted.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state was clean on `main`, with HEAD `46f378b feat(rvv): validate binary ABI IR coherence`.
- No `.trellis/.current-task` existed, so this task was created for the Hermes brief.
- The implementation must remain in the C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck stack.
- Python may only remain runner/tooling support; it must not define compiler ABI semantics, route selection, lowering, emission, or bundle contracts.
- Runtime ABI truth for callable/buffer/scalar boundaries must come from direct `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` IR and validated emission-plan mirrors.
- The dispatch availability guard must be resolved through typed `tcrv.exec.case runtime_guard_required = true` plus `runtime_guard = @...` linkage to a direct same-kernel `tcrv.exec.runtime_param` with role `dispatch-availability-guard`.
- RVV/scalar child callable candidates must be preflighted through target artifact exporter candidate validation before composite source/header/object or bundle output.
- Generic execution-plan/export coherence must stay target-neutral; RVV/scalar descriptor policy and family-specific ABI names/types belong in target/plugin-owned code.
- The finite dispatch family surface is bounded to existing add/sub/mul i32/i64 binary routes; this task must not add new families, vector shapes, masks, VL policy, or performance tuning.

## Requirements

- Inspect existing dispatch source/header/object and bundle routes before changing code.
- Reuse existing diagnostic/runtime ABI/mem-window attributes instead of inventing a parallel metadata model.
- Validate the top-level RVV+scalar dispatch candidate against:
  - selected binary family descriptor;
  - selected RVV dispatch-case callable plan;
  - selected scalar dispatch-fallback callable plan;
  - IR-backed lhs/rhs/out `mem_window` roles;
  - IR-backed runtime element-count `runtime_param`;
  - IR-backed dispatch availability guard `runtime_param`;
  - parent/child runtime ABI kind/name and callable ABI identity.
- Ensure emission manifest and target artifact bundle records for composite dispatch are produced from the validated ABI metadata, not stale i32-vadd or family-agnostic defaults.
- Add positive coverage for at least one i64 RVV+scalar dispatch family and one existing i32 family, proving stable:
  - runtime ABI parameter roles/types/order;
  - mem-window roles;
  - dispatch ABI name;
  - child ABI names;
  - source/header/object bundle records.
- Add focused negative coverage for stale or mismatched composite metadata, including one or more of:
  - i64 dispatch carrying i32 pointer ABI metadata;
  - missing runtime element-count linkage;
  - missing child ABI parameters;
  - duplicate buffer role;
  - parent/child ABI name mismatch.
- Keep diagnostics route-specific and stable enough for FileCheck.
- Update specs only if implementation clarifies a durable lowering/runtime or plugin-locality contract.

## Acceptance Criteria

- [x] Composite RVV+scalar dispatch export validates one shared IR-backed ABI boundary before emitting source, header, object, or bundle records.
- [x] RVV child callable and scalar fallback callable ABI metadata are checked as exact mirrors of `tcrv.exec.mem_window` / `tcrv.exec.runtime_param` IR.
- [x] Dispatch guard ABI comes from selected dispatch-case `runtime_guard` linkage, not printable guard strings or detached metadata.
- [x] i64 dispatch source/header/object/bundle path uses i64 family pointer types, runtime ABI names, external ABI name, component group, and child ABI names without inheriting stale i32 defaults.
- [x] At least one existing i32 dispatch family remains covered and stable.
- [x] Negative tests fail before artifact/bundle output for stale or mismatched composite ABI metadata.
- [x] Focused changed targets and tests pass.
- [x] Full `check-tianchenrv` passes before archive.

## Completion Notes

- Added target-local RVV+scalar dispatch candidate shape preflight for selected dispatch-case and dispatch-fallback child routes. Known composite child route IDs now fail closed when origin, role, emission kind, artifact kind, runtime ABI, runtime ABI kind/name, or runtime glue role are stale.
- Kept standalone RVV source/header/object route behavior intact by applying stale child-route preflight only when the selected path role is already `dispatch case` or `dispatch fallback`.
- Extended i64-vsub dispatch coverage with child ABI name, mem-window, runtime-param, dispatch ABI parameter, stale scalar child ABI, and stale i64 pointer metadata checks.
- Extended bundle e2e coverage to assert i64-vsub source/header/object bundle records carry the i64 dispatch ABI name and runtime ABI parameter roles/types.
- No new `ssh rvv` evidence was required: this round changed metadata validation/front-door failure behavior and bundle index checks, not generated runtime-callable compute behavior or a new runtime correctness/performance claim.

## Minimal Validation

- `git diff --check`
- Build focused changed targets, especially `tcrv-translate` and affected target/plugin test binaries.
- Focused lit/C++ tests for:
  - `test/Target/RVVScalarDispatch/`
  - `test/Scripts/rvv-scalar-dispatch-e2e.test`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`
  - `test/Target/ArtifactExport/`
  - affected RVV/scalar plugin tests.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate <archived-task-path>` after finish/archive.
- Real `ssh rvv` evidence is required only if this round changes generated runtime-callable dispatch behavior and makes a new runtime/correctness claim. Otherwise report that no new RVV runtime evidence was required.

## Out Of Scope

- New RVV families, new vector shapes, new SEW/LMUL coverage, masks, dynamic VL policy, or performance tuning.
- Broad `ssh rvv` matrices or evidence-packaging-only work.
- Python compiler internals, Python ABI semantics, or Python route selection.
- New compute semantics in `tcrv.exec`.
- RVV/scalar semantic branches in core orchestration passes.
- IME, AME, Sophgo/offload, provider29, prompt/report/status-only changes, or helper-only cleanup.

## Technical Notes

- Relevant specs read this round:
  - `.trellis/spec/index.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant implementation surfaces to inspect next:
  - `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`
  - `include/TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/EmissionManifest.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `lib/Transforms/EmissionReadiness.cpp`
  - `lib/Transforms/ExecutionPlanCoherence.cpp`
  - `test/Target/RVVScalarDispatch/`
  - `test/Scripts/rvv-scalar-dispatch-e2e.test`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`
  - `test/Target/ArtifactExport/`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - `test/Plugin/ScalarExtensionPluginTest.cpp`
