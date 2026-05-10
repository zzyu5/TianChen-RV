# Bounded linalg RVV binary frontend pass naming

## Goal

Repair the public bounded linalg frontend pass/API/spec contract after finite
i32 and i64 RVV binary family support. The compiler should expose a bounded
RVV binary frontend pass name, use that name in current tool-owned
plan-and-export paths and evidence metadata, and keep the older i32-named pass
options as compatibility aliases.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial task state was clean on `main` at `04a28f9 feat(target): own
  translate route registration`.
- `.trellis/.current-task` was absent. The previous task,
  `target-owned-artifact-route-translation-registration`, was finished and
  archived.
- Latest supervisor audit/review inputs were:
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0046-20260510T083843Z/repo_audit.md`
  - `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0046-20260510T083843Z/review_input.md`
- Current `lib/Transforms/LowerLinalgI32BinaryToExec.cpp` already routes
  frontend markers through `RVVBinaryFamilyDescriptor` and accepts finite
  i32/i64 add/sub/mul families.
- The public pass/API names, TableGen pass description, specs, translate
  diagnostics, and script evidence strings still present the current pass as
  `i32`-only.

## Requirements

1. Add a public bounded RVV binary frontend pass surface:
   `--tcrv-lower-linalg-rvv-binary-to-exec`.
2. Add `createLowerLinalgRVVBinaryToExecPass()` as the current C++ factory and
   keep `createLowerLinalgI32BinaryToExecPass()` plus
   `createLowerLinalgI32VAddToExecPass()` as deprecated compatibility aliases
   delegating to the same implementation.
3. Preserve all existing lowering behavior:
   explicitly marked bounded `linalg.generic` wrappers still lower to
   `tcrv.exec.kernel`, selected target profile references, ABI
   `mem_window`/`runtime_param` boundaries, and existing frontend family
   markers.
4. Do not introduce generic linalg lowering, new compute semantics in
   `tcrv.exec`, new RVV families, new vector shapes, runtime execution claims,
   or performance claims.
5. Update `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` to
   call the RVV-named factory and emit RVV binary frontend diagnostics rather
   than i32-only wording.
6. Update script/evidence metadata that describes the planned frontend
   pipeline so current i32/i64 frontend evidence names the RVV binary pass.
7. Update durable specs for the bounded linalg frontend pass contract,
   including explicit compatibility alias wording.
8. Add focused lit/FileCheck coverage proving:
   - the new RVV-named pass lowers an i32 frontend fixture;
   - the new RVV-named pass lowers an i64 frontend fixture;
   - the old i32 binary pass option still delegates to the same implementation;
   - the old i32-vadd alias remains compatibility-only.

## Acceptance Criteria

- [x] `tcrv-opt --tcrv-lower-linalg-rvv-binary-to-exec` is registered and works
      on bounded i32 and i64 frontend fixtures.
- [x] `tcrv-opt --tcrv-lower-linalg-i32-binary-to-exec` remains accepted as a
      deprecated alias and preserves existing behavior.
- [x] `tcrv-opt --tcrv-lower-linalg-i32-vadd-to-exec` remains accepted as the
      older compatibility alias.
- [x] `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` uses the
      RVV binary pass factory and no longer reports i32-only frontend wording.
- [x] Focused lit/FileCheck tests cover the new pass name and both aliases.
- [x] Relevant script self-tests and focused script lit checks pass after
      planned-pipeline metadata updates.
- [x] `git diff --check` passes.
- [x] Focused build and full `check-tianchenrv` pass if feasible.
- [x] Task context validates, task is finished/archived, and one coherent
      commit records the work.

## Out Of Scope

- New high-level linalg families beyond the already finite RVV binary
  add/sub/mul descriptors.
- New runtime ABI shape, target artifact route behavior, object generation,
  linkage, RVV runtime correctness, or performance evidence.
- Python implementation of compiler passes, legality, selection, lowering, or
  emission semantics.
- Broad evidence matrices or new `ssh rvv` claims. No fresh RVV runtime
  evidence is required unless generated runtime-callable behavior changes.
- Moving RVV-specific semantic branches into core orchestration passes.

## Minimal Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Focused lit/FileCheck on:
  - `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`
  - `test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir`
  - `test/Transforms/LinalgToExec/linalg-i32-binary-compat-vadd-alias.mlir`
  - representative plan-and-export target artifact bundle fixture
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-bounded-linalg-rvv-binary-frontend-pass`

## Technical Notes

- Specs and guides read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/variant-pipeline/index.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/implementation-stack/index.md`
  - `.trellis/spec/implementation-stack/compiler-stack-contract.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/guides/index.md`
  - `.trellis/spec/guides/compute-boundary-review-guide.md`
  - `.trellis/spec/guides/capability-first-design-guide.md`
- Current implementation surfaces inspected:
  - `include/TianChenRV/Transforms/Passes.h`
  - `include/TianChenRV/Transforms/Passes.td`
  - `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`
  - `tools/tcrv-translate/tcrv-translate.cpp`
  - `scripts/rvv_microkernel_e2e.py`
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - representative `test/Transforms/LinalgToExec/` and bundle fixtures.
