# RVV Generated Bundle ABI Execution Proof On ssh rvv

## Goal

Prove that the existing RVV i32m1 vector-source add route exports a coherent
materialized EmitC object/header bundle whose generated declaration-only
header and generated relocatable object can be consumed together by a small
external C harness on the real `ssh rvv` target.

This task is an ABI-consumption proof for the current rebuilt route, not a new
RVV operation, not a new artifact format, and not a replacement compiler path.

## Current Repository Facts

- Session start repo root was `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean before task creation.
- Starting HEAD was `fd53088 target: add common materialized emitc bundle construction`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief as `05-17-rvv-generated-bundle-abi-execution-proof`.
- Current local bundle export already produces:
  - `artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o`;
  - `artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h`;
  - `tianchenrv-target-artifact-bundle.index`.
- Current bundle index ties both artifacts to selected variant
  `@vector_source_rvv_i32_add`, role `dispatch case`, route
  `rvv-i32m1-arithmetic-emitc-route-family`, component group
  `rvv-i32m1-arithmetic-materialized-emitc-bundle.v1`, external ABI
  `rvv-i32m1-add-callable-c-abi.v1`, ordered ABI parameters
  `lhs`, `rhs`, `out`, `n`, and RVV multi-VL metadata.
- Current header is declaration-only and has existing FileCheck coverage
  rejecting RVV intrinsic bodies, `main`, descriptors, direct-C, source-export,
  and historical route residue.

## Requirements

- Add a bounded ABI consumer runner or harness surface that:
  - runs the current vector-source front-door pipeline through `tcrv-opt`;
  - exports the current target artifact bundle through `tcrv-translate`;
  - verifies the bundle index, generated header, generated object, selected
    variant, component group, runtime ABI identity, ordered ABI parameters,
    materialized EmitC provenance, and RVV runtime AVL/VL metadata before
    remote execution;
  - stages only the generated header, generated object, and a small external C
    harness to `ssh rvv`;
  - compiles and links with clang/LLVM on the remote RVV target;
  - runs correctness cases for several `n` values including tail and
    multi-VL-sized cases;
  - records sanitized structured evidence under `artifacts/tmp`.
- The harness must be an ABI consumer only. It may initialize inputs and check
  outputs, but it must not become a compiler semantic path, direct C compute
  exporter, descriptor adapter, artifact ledger protocol, or fallback runtime.
- Keep the compiler route as extension-family ops -> materialized EmitC module
  -> MLIR EmitC C/C++ emitter -> generated object/header bundle.
- Negative/local coverage must fail closed for missing or mismatched generated
  header/object, stale ABI order, missing runtime AVL/VL metadata, fallback-only
  or ambiguous selected candidates, historical descriptor/direct-C/source-export
  route residue, and header content that embeds RVV intrinsic compute bodies.
- If `ssh rvv` is unavailable for credentials, connectivity, toolchain, or
  hardware reasons, record the concrete blocker and do not claim runtime or
  correctness evidence.

## Acceptance Criteria

- [x] A positive local path runs:
      `tcrv-opt test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir --tcrv-source-artifact-front-door-pipeline`
      piped to
      `tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=<dir>`.
- [x] The generated bundle directory contains exactly one RVV object artifact,
      exactly one declaration-only header artifact, and one bundle index.
- [x] Bundle verification proves object and header agree on selected variant,
      role, component group, external ABI name, route family, runtime ABI
      kind/name, ordered ABI parameters `lhs`, `rhs`, `out`, `n`, materialized
      EmitC route provenance, and RVV multi-VL metadata.
- [x] Header verification proves the generated header declares the selected ABI
      and does not embed RVV intrinsic bodies, `main`, descriptor/direct-C/
      source-export text, or historical route IDs.
- [x] Real `ssh rvv` evidence compiles and links the generated object plus the
      generated header with a C harness using clang/LLVM, then runs correctness
      for multiple sizes including tail and multi-VL cases.
- [x] Focused lit or script self-test coverage exercises the ABI consumer
      verifier without contacting `ssh rvv`.
- [x] Focused C++/lit checks for touched target/RVV/tooling surfaces pass.
- [x] Targeted scans over touched RVV target/plugin/translate/tests confirm no
      descriptor route authority, no direct C semantic exporter, no source-export
      route, and no restored legacy header/object/bundle route IDs.
- [x] `git diff --check` passes.

## Definition Of Done

- Runtime ABI consumer evidence is written under `artifacts/tmp` and includes
  the exact local bundle command, generated bundle layout, header/object/index
  checks, remote compile/link/run command, input sizes, and correctness result.
- Trellis task status, journal, archive state, and final report state the exact
  proven boundary and any remaining rebuild gaps.
- One coherent commit records the completed proof if the task is finished; if
  remote execution is blocked, the task remains open with the exact blocker and
  next continuation point.

## Out Of Scope

- New RVV operations, sub/mul expansion, source-shape expansion, broader
  SEW/LMUL coverage, a new EmitC route, a new artifact format, a state machine,
  a bundle ledger, descriptor compatibility, direct C semantic exporters,
  Python compiler-core logic, scalar fallback compute, generic performance
  matrices, broad smoke-only tests, and evidence reports detached from the
  generated bundle ABI consumer.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  and `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-17-common-executable-plugin-construction-template/prd.md`.
- Current code/test surfaces inspected:
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`,
  `test/Target/RVV/vector-source-target-artifact-object.mlir`,
  and `test/Target/RVV/vector-source-target-artifact-header.mlir`.

## Implementation Summary

- Added `scripts/rvv_generated_bundle_abi_e2e.py` as evidence tooling for the
  current RVV generated bundle ABI path. The runner:
  - invokes the existing `tcrv-opt` source artifact front-door pipeline;
  - invokes `tcrv-translate --tcrv-export-target-artifact-bundle`;
  - verifies the generated bundle index, header, and object before execution;
  - generates a small external C harness that only includes the generated
    header and calls the declared ABI symbol from the generated object;
  - stages the generated header, generated object, and harness to `ssh rvv`;
  - compiles/links/runs the harness with remote clang;
  - writes sanitized evidence JSON under `artifacts/tmp`.
- Added `test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test` for local
  parser/verifier negative coverage without contacting `ssh rvv`.
- Updated `.trellis/spec/testing/mlir-testing-contract.md` with the durable
  rule that live RVV generated-bundle ABI correctness requires a bounded
  external ABI consumer on `ssh rvv`, and that Python runners/harnesses remain
  evidence tooling rather than compiler semantic paths.

## Completion Evidence

- Live evidence directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-generated-bundle-abi-ssh`.
- Exact positive bundle command recorded in evidence:
  `build/bin/tcrv-opt test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir --tcrv-source-artifact-front-door-pipeline | build/bin/tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-generated-bundle-abi-ssh/generated_bundle`.
- Generated bundle artifacts:
  - `generated_bundle/tianchenrv-target-artifact-bundle.index`;
  - `generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o`;
  - `generated_bundle/artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h`;
  - `rvv_generated_bundle_abi_harness.c`;
  - `remote_compile_stdout.txt`;
  - `remote_run_stdout.txt`;
  - `evidence.json`.
- Bundle verifier proved both generated components agree on selected variant
  `@vector_source_rvv_i32_add`, role `dispatch case`, component group
  `rvv-i32m1-arithmetic-materialized-emitc-bundle.v1`, external ABI
  `rvv-i32m1-add-callable-c-abi.v1`, runtime ABI kind/name, ordered parameters
  `lhs`, `rhs`, `out`, `n`, materialized EmitC provenance, and RVV runtime
  AVL/VL metadata including `multi_vl = supported`.
- Header verifier proved declaration-only ABI content and rejected RVV
  intrinsic body tokens, `vint32m1_t`, `main`, descriptor, direct-C,
  source-export, and historical microkernel route residue.
- Local object verifier ran `llvm-readobj -h` and checked
  `elf64-littleriscv`, `riscv64`, and `Relocatable`.
- Real `ssh rvv` result:
  remote compile/link succeeded, remote run succeeded, and output included:
  `case n=1 ok`, `case n=7 ok`, `case n=16 ok`, `case n=17 ok`,
  `case n=257 ok`, and
  `tcrv_rvv_generated_bundle_abi_ok counts=1,7,16,17,257`.
- Checks passed:
  - `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`;
  - `./build/bin/tianchenrv-target-artifact-export-test`;
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --run-id codex-rvv-generated-bundle-abi-dry --overwrite --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257`;
  - `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-generated-bundle-abi-e2e-self-test.test Target/RVV/vector-source-target-artifact-object.mlir Target/RVV/vector-source-target-artifact-header.mlir`
    from `build/test`, 3/3 passed;
  - `cmake --build build --target check-tianchenrv -j2`, 113/113 lit tests
    passed;
  - `git diff --check`.
- Targeted scans over touched RVV target/plugin/translate/tests and the new
  runner found descriptor/direct-C/source-export strings only in fail-closed
  validation, forbidden-token lists, comments explaining non-authority, or
  `CHECK-NOT` assertions. The external harness contains `main` only as the
  ABI consumer executable and the generated header remains declaration-only.

## Status

Complete. No compatibility layer was added. No descriptor-driven computation,
direct C semantic exporter, source-export route, scalar fallback compute, new
RVV operation, or legacy route alias was introduced.
