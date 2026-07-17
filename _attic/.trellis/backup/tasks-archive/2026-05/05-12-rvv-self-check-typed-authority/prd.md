# RVV self-check typed authority descriptor exit

## Goal

Rewire RVV direct microkernel and RVV+scalar dispatch self-check artifact
generation so expected values, scalar C types, and computation provenance come
from typed RVV/scalar family-op bodies, validated microkernel bodies,
IR-backed callable ABI boundaries, and the common EmitC/dataflow route.
Legacy RVV/scalar descriptors may remain only as bounded intrinsic/config or
mirror metadata and must not define self-check computation semantics.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean before this task was created.
- Task start HEAD is
  `2068a11 feat(rvv): require typed variant legality authority`.
- `.trellis/.current-task` was absent at session start, so this task was
  created as `.trellis/tasks/05-12-rvv-self-check-typed-authority/`.
- The predecessor
  `.trellis/tasks/archive/2026-05/05-12-rvv-variant-legality-descriptor-exit/`
  completed typed-authority-first RVV variant legality. Descriptor-only RVV
  variants no longer pass plugin legality, and descriptors are mirror-only for
  legality.
- The next structural leak is target self-check artifact generation:
  `lib/Target/RVV/RVVMicrokernel.cpp` and
  `lib/Target/Builtin/RVVScalarDispatch.cpp` still derive harness expected
  values and scalar C types from descriptor helpers such as
  `getScalarCType()` and `getCArithmeticCheckExpression()`.
- Specs define the current main route as extension family ops -> EmitC ->
  intrinsic/runtime C/C++ and keep descriptor-driven computation as bounded
  migration debt, not architecture.
- `tcrv.exec` must remain compute-free; family computation belongs in typed
  extension-family ops and target/plugin-owned exporters.

## Requirements

- Add or reuse a C++ typed self-check expectation/provenance boundary that is
  derived from the validated typed microkernel body, EmitC/dataflow route, and
  IR-backed ABI, not descriptor-owned compute helpers.
- Rewire direct RVV self-check source generation in
  `lib/Target/RVV/RVVMicrokernel.cpp` so the harness expected expression and
  scalar element type come from typed body/dataflow/ABI authority. A stale
  legacy descriptor mirror must not be able to change generated expected
  arithmetic.
- Rewire RVV+scalar dispatch self-check source generation in
  `lib/Target/Builtin/RVVScalarDispatch.cpp` so dispatcher harness expected
  expression/type comes from validated RVV and scalar callable component
  authorities plus shared runtime ABI, not from
  `getRVVBinaryIntrinsicDescriptor(...).getCArithmeticCheckExpression(...)` or
  descriptor-owned family computation.
- Preserve parameter layering:
  hardware facts, selected vector config, and compile flags may constrain
  intrinsic spelling and toolchain routes; runtime element count remains an
  ABI/runtime parameter; descriptor-local `element_count` and lowering
  descriptor fields remain bounded mirror/quarantine metadata only.
- Update or remove tests that encode descriptor-owned self-check semantics as
  production behavior.
- Add focused negative coverage proving a stale descriptor mirror cannot change
  self-check computation and that typed body/source mismatch fails before
  artifact export.
- Keep implementation bounded to one coherent self-check authority module,
  covering one default direct RVV slice and one RVV+scalar dispatch slice.

## Acceptance Criteria

- [x] Direct RVV self-check expected arithmetic is selected from validated typed
      RVV family body/dataflow authority, not descriptor helper text.
- [x] Direct RVV self-check scalar element type is selected from typed
      microkernel/ABI authority, not descriptor helper text.
- [x] RVV+scalar dispatch self-check expected arithmetic and scalar element type
      are selected from validated RVV/scalar component callable authorities and
      the shared runtime ABI.
- [x] A stale legacy descriptor mirror beside a typed direct RVV body does not
      alter self-check arithmetic; stale or mismatched descriptor/body metadata
      fails before artifact output where the existing contract requires it.
- [x] A stale legacy descriptor mirror beside a typed RVV+scalar dispatch pair
      does not alter dispatcher self-check arithmetic; selected component
      family mismatch fails before artifact output.
- [x] Focused lit/C++ tests cover the direct RVV and RVV+scalar dispatch
      self-check surfaces touched by this task.
- [x] `git diff --check` passes.
- [x] Focused changed targets and lit slices pass before broad validation.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
      -j2` passes before finish/archive if the existing build tree is usable.
- [x] Trellis task validation passes before finish/archive.
- [x] One coherent commit records the completed module.

## Non-Goals

- No generic RVV backend, new extension family, new high-level tensor/tile IR,
  broad family matrix, scheduler, tuning, cost model, or performance work.
- No descriptor-to-C computation path, descriptor-owned expected-result logic,
  or descriptor-driven tests preserved as the default behavior.
- No compiler core, dialects, passes, plugin registry, capability model,
  lowering, emission, self-check semantics, route selection, or runtime ABI
  implemented as Python data structures.
- No runtime, correctness, or performance claim without fresh bounded
  `ssh rvv` evidence for the exact touched self-check route.
- No helper-only, metadata-only, report-only, broad smoke, or ssh-evidence-only
  closeout.
- Do not remove descriptors wholesale unless a small deletion is necessary to
  remove compute authority from the default self-check path.

## Minimal Validation

- Build focused changed targets first, including RVV microkernel, scalar
  microkernel if touched, RVV+scalar dispatch, and target artifact export tests
  relevant to changed code.
- Run focused lit under `test/Target/RVVMicrokernel/` and
  `test/Target/RVVScalarDispatch/`, including a stale descriptor/mirror
  negative case.
- Run the relevant frontend-to-artifact lit slice, such as
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir` or the exact
  dispatch fixture touched by this task.
- Run `git diff --check`.
- Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-12-rvv-self-check-typed-authority`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` before finish/archive.
- If final reporting claims RVV runtime correctness for the changed harness,
  collect fresh bounded `ssh rvv` evidence through the existing runner/probe
  path. Otherwise explicitly report that no runtime/correctness/performance
  claim is made.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/validation/experiment-reference.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Predecessor PRD read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-variant-legality-descriptor-exit/prd.md`
- Initial source focus:
  - `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h`
  - `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h`
  - `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `test/Target/RVVMicrokernel/`
  - `test/Target/RVVScalarDispatch/`
  - `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`
  - relevant RVV/scalar/dispatch C++ tests under `test/Plugin/` and
    `test/Target/`

## Definition Of Done

The task is complete only when descriptor compute authority is removed from
the touched direct RVV and RVV+scalar dispatch self-check default paths,
focused and available broad checks pass, Trellis state is truthful, and one
coherent commit records the completed module. If unfinished, leave the task
open and record the exact remaining source function/test continuation point in
`implement.jsonl` or the workspace journal.
