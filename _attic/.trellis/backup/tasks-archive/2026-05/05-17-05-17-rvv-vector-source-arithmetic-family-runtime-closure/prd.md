# RVV Vector-Source Arithmetic Family ssh-rvv Runtime Closure

## Goal

Produce truthful hardware runtime closure for the current bounded
source-derived RVV i32m1 add/sub/mul family. The proof must generate object,
header, and bundle artifacts through the existing compiler path, then consume
those generated artifacts on the real `ssh rvv` target with distinct
correctness cases for add, sub, and mul.

The route under proof is:

```text
source MLIR vector arithmetic
  -> RVV selected boundary
  -> explicit tcrv_rvv add/sub/mul ops
  -> materialized EmitC module
  -> generated object/header/bundle
  -> external ABI consumer on ssh rvv
```

## Current Repository Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean at task creation.
- Starting HEAD was `ab5949b rvv: materialize vector source arithmetic family`.
- No `.trellis/.current-task` existed, so this task was created from the
  direction brief as
  `05-17-05-17-rvv-vector-source-arithmetic-family-runtime-closure`.
- The previous completed materializer task generalized the production
  vector-source front door to source-derived `arith.addi`, `arith.subi`, and
  `arith.muli`, producing selected variants
  `*_rvv_i32_add/sub/mul` and typed `tcrv_rvv.i32_add/sub/mul` bodies.
- Current focused lit coverage already proves add/sub/mul source-front-door
  materialization and object/header/bundle packaging through the materialized
  EmitC route family.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently proves the generated
  bundle ABI only for the add fixture. It hardcodes the add selected variant,
  add runtime ABI name, add function symbol, add expected arithmetic, and one
  add-only artifact directory layout.
- The existing evidence runner is an acceptable boundary to adapt because it
  invokes `tcrv-opt`, `tcrv-translate`, generated bundle files, remote clang,
  and remote execution; it must remain evidence tooling and must not become
  compiler semantics, route selection, lowering, codegen, or artifact truth.

## Requirements

- Generalize the ABI e2e evidence runner so it can execute the existing
  source-derived arithmetic fixtures:
  - `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir` for add;
  - `test/Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir` for sub;
  - `test/Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir` for mul.
- For each op kind, the runner must generate artifacts through:

  ```text
  tcrv-opt <fixture> --tcrv-source-artifact-front-door-pipeline
    | tcrv-translate --tcrv-export-target-artifact-bundle
  ```

- For each op kind, the runner must verify generated bundle facts before
  remote execution:
  - selected variant matches the fixture-derived selected RVV variant;
  - component group is
    `rvv-i32m1-arithmetic-materialized-emitc-bundle.v1`;
  - object route is `rvv-i32m1-arithmetic-emitc-route-family`;
  - header route is `rvv-i32m1-arithmetic-emitc-route-family.header`;
  - runtime ABI kind is `plugin-owned-runtime-abi`;
  - runtime ABI name is `rvv-i32m1-{add,sub,mul}-callable-c-abi.v1`;
  - runtime ABI parameters are ordered as `lhs`, `rhs`, `out`, `n`;
  - metadata includes `rvv_emitc_lowerable_route`,
    `rvv_arithmetic_op`, and the RVV runtime AVL/VL multi-VL contract;
  - generated header is declaration-only and contains no intrinsic body,
    `main`, descriptor, direct-C, source-export, or historical route residue.
- For each op kind, generate an external C ABI harness that includes the
  generated header and calls only the generated function symbol from the
  generated object. The harness may initialize inputs and compute expected
  values for validation, but it must not duplicate RVV intrinsic bodies or act
  as compiler/runtime fallback.
- The add, sub, and mul harness cases must use distinct input patterns and
  expected arithmetic, and must exercise tail and multi-VL counts.
- The artifact directory must record, for each op kind:
  - the source MLIR fixture copy or path/hash;
  - generated bundle index;
  - generated header and object;
  - generated harness;
  - local command logs;
  - remote compile log;
  - remote run log with PASS output.
- The runner must fail closed for stale add-only assumptions, mismatched
  selected variant, mismatched runtime ABI name, wrong function prototype,
  missing arithmetic metadata, descriptor/source-export/direct-C residue, or
  hand-written generated-header compute fallback.

## Acceptance Criteria

- [x] The e2e runner defaults to covering add, sub, and mul in one coherent
      run, while retaining a focused way to run a single op kind.
- [x] Dry-run mode generates and verifies add, sub, and mul bundles locally
      without contacting `ssh rvv`.
- [x] Real `ssh rvv` mode compiles, links, and runs each generated
      header/object bundle with a generated external ABI harness, and reports
      PASS output for add, sub, and mul.
- [x] Evidence JSON records per-op source fixture, generated bundle files,
      header/object/index checks, harness file, remote commands, remote
      compile output, and remote run output.
- [x] Script self-test covers add/sub/mul parser/verifier/harness generation
      locally and includes negative cases for stale selected variant, stale
      arithmetic metadata, stale ABI order, and forbidden header body tokens.
- [x] Focused build for `tcrv-opt`, `tcrv-translate`, and relevant RVV target
      tests passes.
- [x] Focused lit for vector-source add/sub/mul and target artifact
      object/header/bundle surfaces passes.
- [x] `check-tianchenrv` passes if practical; if not, the exact bounded
      substitute and blocker are recorded.
- [x] Targeted scans over touched script/test/RVV plugin/target surfaces show
      no descriptor route authority, no direct C semantic exporter, no
      source-export route, and no restored compatibility route.
- [x] `git diff --check` passes.

## Definition Of Done

- The completed proof is hardware runtime evidence for the existing bounded
  source-derived add/sub/mul family only.
- Trellis task state, journal, and archive record what was proven, which
  artifacts were generated, and the exact `ssh rvv` result.
- One coherent commit records the runner/test/PRD work if complete. If `ssh rvv`
  is blocked, leave the task open with the exact remote blocker and next
  continuation point.

## Out Of Scope

- New RVV ops, new SEW/LMUL families, new source patterns, new route families,
  new artifact formats, performance benchmarking, broad smoke matrices,
  descriptor adapters, source-export/direct-C semantic exporters, scalar
  fallback compute, compatibility wrappers, Python compiler-core logic,
  generated runtime fallback libraries, or generic RVV lowering.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior PRDs read:
  `.trellis/tasks/archive/2026-05/05-17-rvv-vector-source-arithmetic-family-materializer/prd.md`,
  `.trellis/tasks/archive/2026-05/05-17-rvv-generated-bundle-abi-execution-proof/prd.md`,
  `.trellis/tasks/archive/2026-05/05-17-rvv-runtime-avl-vl-abi-artifact-closure/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-17-05-17-rvv-runtime-avl-multivl-emitc-loop/prd.md`.
- Primary implementation surface:
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Primary evidence/test surfaces:
  `test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test`,
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door*.mlir`, and
  `test/Target/RVV/vector-source-target-artifact-{object,header}.mlir`.

## Implementation Summary

- Generalized `scripts/rvv_generated_bundle_abi_e2e.py` from an add-only
  verifier into an op-kind-driven generated-bundle ABI evidence runner for
  add, sub, and mul.
- Added per-op expectations for source fixture, selected variant, runtime ABI
  name, generated function symbol, EmitC route metadata, input initialization,
  expected arithmetic, and PASS marker.
- The default run now covers `add`, `sub`, and `mul`; `--op-kind` can be used
  repeatedly for focused runs, and `--input` may override the source fixture
  only when exactly one op kind is selected.
- Each op kind gets its own artifact subdirectory containing `source.mlir`,
  generated bundle index, generated object, generated header, generated
  external ABI harness, per-op `evidence.json`, and remote compile/run logs.
- The generated harness remains an external ABI consumer: it includes the
  generated header and calls the generated object symbol, while computing only
  host-side expected values for correctness checking.
- The script self-test now verifies all three op kinds and covers negative
  cases for missing header/object, stale ABI order, stale multi-VL metadata,
  stale arithmetic metadata, forbidden header intrinsic body residue, and
  mismatched selected variant.

## Validation

- Python syntax:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- Script self-test:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Local dry-run generated and verified add/sub/mul bundles:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --run-id codex-rvv-vector-source-family-dry --overwrite --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257`.
- Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Focused C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test` and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit from `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-generated-bundle-abi-e2e-self-test.test Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir Target/RVV/vector-source-target-artifact-object.mlir Target/RVV/vector-source-target-artifact-header.mlir`,
  6/6 passed.
- Real `ssh rvv` evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --run-id codex-rvv-vector-source-family-ssh --overwrite --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257 --timeout 180`.
- Live evidence directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rvv-vector-source-family-ssh`.
- Remote PASS output:
  - add: `PASS op=add counts=1,7,16,17,257`;
  - sub: `PASS op=sub counts=1,7,16,17,257`;
  - mul: `PASS op=mul counts=1,7,16,17,257`.
- Full check:
  `cmake --build build --target check-tianchenrv -j2`, 115/115 lit tests
  passed.
- `git diff --check` passed.
- Targeted scans over touched script/test surfaces and RVV plugin/target
  surfaces found descriptor/direct-C/source-export strings only in
  fail-closed validation, forbidden-token lists, non-authority docstrings,
  existing target preflight rejection, or `CHECK-NOT` assertions.

## Spec Update Review

No `.trellis/spec/` update was needed. The existing testing and lowering specs
already require live RVV generated-bundle ABI correctness to use a bounded
external ABI consumer on `ssh rvv` and already state that Python runners remain
evidence tooling rather than compiler semantic paths. This task instantiated
that existing contract for the now-supported add/sub/mul family; it did not
introduce a new compiler contract.

## Status

Complete. The bounded source-derived add/sub/mul RVV family now has fresh
generated object/header/bundle ABI runtime correctness evidence on real
`ssh rvv`. No compatibility layer, descriptor-driven computation, direct C
semantic exporter, source-export route, scalar fallback compute, new artifact
route, or legacy route alias was introduced.
