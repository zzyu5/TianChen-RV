# RVV Runtime-Callable C ABI Link Proof

## Goal

Prove that the existing RVV selected vector-source add path exposes a real
runtime-callable C ABI across the materialized EmitC object/header/bundle
route. The proof must cover the selected RVV extension-family route from
TianChen-RV MLIR input through materialized EmitC, MLIR EmitC C/C++ emission,
RISC-V relocatable object packaging, declaration-only runtime-callable C
header output, coherent object+header bundle metadata, and an external C
harness that compiles, links, and runs on `ssh rvv`.

## What I Already Know

- Current HEAD is `5910292` (`toy: package materialized EmitC object bundle`),
  and the worktree was clean before task creation.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief before source edits.
- `.trellis/spec/lowering-runtime/emitc-route.md` now has a
  "Runtime-Callable C ABI Linkage" scenario requiring generated headers with
  C++ `extern "C"` guards, objects with unmangled C symbols, bundle/header/
  object ABI coherence, and real `ssh rvv` external C harness evidence before
  claiming RVV runtime/correctness.
- `TCRVEmitCMaterializationOptions` already has `emitExternC`, and
  `materializeSelectedEmitCArtifactModule` currently sets it for selected
  materialized target artifact routes.
- `TargetArtifactExport.cpp` already renders declaration-only headers through
  the common materialized EmitC header helper, including C++ `extern "C"`
  guards and ordered runtime ABI parameter declarations.
- `RVVTargetSupportBundle.cpp` already consumes the common materialized
  object/header bundle helper for the RVV i32m1 arithmetic family and packages
  generated EmitC C/C++ source into RISC-V relocatable objects with clang.
- Existing lit coverage under `test/Target/RVV` checks object/header/bundle
  shape for add/sub/mul, including header guards, runtime ABI metadata, and
  unmangled object symbol names.
- `scripts/rvv_generated_bundle_abi_e2e.py` is an evidence script that
  generates the RVV target artifact bundle, verifies local bundle metadata,
  writes an external C harness, and can compile/link/run the generated
  header/object on `ssh rvv`.
- The immediately completed Toy task proved common object/header/bundle helper
  consumption for Toy; this RVV round must not broaden into Toy, Template, or
  TensorExtLite work.

## Requirements

- Keep the owner on the bounded RVV vector-source add artifact path and shared
  materialized EmitC target artifact helpers needed for C ABI linkage.
- Use the current route:
  TianChen-RV MLIR vector-source add -> selected RVV extension-family route ->
  materialized EmitC module -> MLIR EmitC C/C++ emitter -> RISC-V relocatable
  object + runtime-callable C header + coherent object/header bundle ->
  external C harness on `ssh rvv`.
- Runtime-callable public functions must be emitted with C linkage in the
  materialized EmitC function boundary and must expose unmangled object symbols.
- Generated runtime-callable C headers must be usable from both C and C++
  callers by wrapping the public declaration in a C++ `extern "C"` guard.
- Header declaration name, object global symbol, external ABI name, runtime ABI
  kind/name, ordered ABI parameters, bundle records, and harness call order
  must all describe the same selected callable boundary.
- The external C harness must include only the generated header, link only the
  generated object, execute on `ssh rvv`, and print a bounded PASS marker for
  the selected add path.
- Python changes, if any, are limited to evidence tooling. Compiler behavior
  remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- If an implementation gap is exposed, fix only the C ABI linkage gap inside
  the existing materialized EmitC/common target path.

## Acceptance Criteria

- [x] The task context points implement/check agents or serial worker context
      at the relevant long-term specs and this PRD, with no stale example JSONL.
- [x] The RVV generated runtime-callable header test asserts the C++ `extern
      "C"` guard around the public declaration for the selected add route.
- [x] The RVV generated object test asserts the selected add object contains
      the unmangled symbol
      `tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add` and does not
      rely on the corresponding C++-mangled symbol.
- [x] The generated bundle index checks preserve object/header component
      coherence for the selected add route: same selected variant, route
      authority, component group, external ABI name, runtime ABI kind/name, and
      ordered `lhs, rhs, out, n` runtime ABI parameters.
- [x] The RVV generated bundle ABI evidence tool validates the header guard,
      unmangled selected symbol, object/header/bundle ABI coherence, harness
      source, and fail-closed self-test cases for stale or unsafe evidence.
- [x] A bounded `ssh rvv` run generates or copies the add bundle, compiles an
      external C harness with clang, links the generated object, runs it, and
      records PASS evidence under `artifacts/tmp/rvv_generated_bundle_abi_e2e`.
- [x] Focused build/checks for `tcrv-opt`, `tcrv-translate`, RVV target
      artifact tests, the evidence script self-test, and `git diff --check`
      pass.
- [x] `check-tianchenrv` is run if common EmitC or target artifact C++ code
      changes; this round did not change common EmitC or target artifact C++,
      so the trigger did not apply.

## Definition Of Done

- Trellis task status and journal entries are truthful.
- The task is finished and archived according to this repo's Trellis
  convention if all acceptance criteria pass.
- One coherent commit records PRD, implementation/tests/tooling changes,
  validation evidence notes, and task finish/archive state.

## Out Of Scope

- New RVV ops, SEW/LMUL families, performance matrices, descriptor routes,
  direct C semantic exporters, source-export bodies, compatibility or legacy
  modes, GCC-default paths, or a new general RVV lowering.
- Broadening the task to Toy, TensorExtLite, Template, IME, Offload, scalar, or
  documentation-only cleanup unless a common C ABI helper is the single
  failing blocker.
- Claiming RVV performance or general RVV correctness beyond the bounded
  external C ABI add-path harness evidence.
- Treating generated artifacts, reports, prompt edits, or broad smoke tests as
  the main achievement without an actual link/run proof.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Previous task reference:
  `.trellis/tasks/archive/2026-05/05-17-05-17-toy-emitc-object-bundle/prd.md`.
- Primary code surfaces:
  `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Primary tests:
  `test/Target/RVV/vector-source-target-artifact-object.mlir`,
  `test/Target/RVV/vector-source-target-artifact-header.mlir`,
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`, and
  `test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test`.

## Completion Summary

- Created and started Trellis task
  `05-17-rvv-runtime-callable-c-abi-link-proof` from the Hermes direction
  brief, with PRD scope locked to the selected RVV vector-source add C ABI link
  proof.
- Kept the compiler route unchanged: existing RVV materialized EmitC object,
  declaration header, and bundle production remains in the C++/MLIR target
  artifact path.
- Tightened `scripts/rvv_generated_bundle_abi_e2e.py` so the evidence tool now
  verifies that the generated header wraps the public declaration with the C++
  `extern "C"` guard required by the runtime-callable C ABI.
- Tightened the same evidence tool so `llvm-readobj --symbols` must show the
  unmangled selected add function symbol and must not show a C++-mangled
  selected function symbol.
- Extended the evidence script self-test fake bundle to include the required
  `extern "C"` guard and added a negative self-test that fails when that guard
  is missing.
- Ran an add-only `ssh rvv` proof that generated the RVV bundle, staged the
  generated header/object plus an external C harness, compiled and linked with
  remote clang, executed on the RVV machine, and printed:
  `PASS op=add counts=1,7,16,17,257`.
- No `.trellis/spec/` update was needed: the durable runtime-callable C ABI
  contract already exists in `lowering-runtime/emitc-route.md`; this task only
  brought the evidence tool up to that contract.

## Validation

- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind add --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-script-symbol-add --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- From `build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='Scripts/rvv-generated-bundle-abi-e2e-self-test|Target/RVV/vector-source-target-artifact-(object|header)'`
- `ssh -o BatchMode=yes -o ConnectTimeout=10 rvv 'hostname; uname -m; command -v clang; clang --version | head -n 1'`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind add --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-rvv-add-abi-link --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 120 --connect-timeout 10`
- `git diff --check`

## Evidence

- Local dry-run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-script-symbol-add/evidence.json`
- Real `ssh rvv` evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-add-abi-link/evidence.json`
- Add-path remote compile facts:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Add-path remote run PASS marker:
  `tcrv_rvv_generated_bundle_abi_add_ok counts=1,7,16,17,257` and
  `PASS op=add counts=1,7,16,17,257`.
