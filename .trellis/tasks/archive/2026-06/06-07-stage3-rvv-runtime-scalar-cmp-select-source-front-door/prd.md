# Stage3 RVV runtime-scalar-cmp select source-front-door materializer and artifact bridge

## Goal

Add or complete one production RVV Vector-like source-front-door family for
runtime-scalar compare/select. The workflow must start from bounded source-only
MLIR with vector input, runtime scalar RHS compare binding, passthrough/select
semantics, and runtime `n`; run a registered RVV plugin-owned materializer;
produce a selected typed `tcrv_rvv` `runtime_scalar_cmp_select` body; reuse the
existing RVV provider route and common EmitC materializer; export a generated
artifact bundle with truthful ABI/runtime mirrors; and prove executable behavior
on `ssh rvv` when runtime correctness is claimed.

## What I Already Know

* No current Trellis task existed at session start; this task was created from
  the Hermes Direction Brief.
* Commit `b4ca6a0e` completed registered source-front-door artifact bridge
  hardening for the existing binary and compare/select families.
* Current `RVVVectorSourceFrontDoor.cpp` registers exactly two active families:
  `bounded-vector-binary-source-front-door` and
  `bounded-vector-compare-select-source-front-door`.
* The public header only exposes binary and compare/select materializer pass
  factories, so runtime-scalar compare/select source-front-door registration is
  not yet a production C++ entry.
* Existing provider/script infrastructure already has mature
  `runtime_scalar_cmp_select` selected-body expectations, route/provider
  validation, generated bundle ABI checks, runtime scalar RHS coverage, and
  explicit/pre-realized dry-run tests.
* The new family must not make source marker strings, artifact names, helper
  names, route ids, or Common EmitC code choose RVV semantics. The materializer
  may only translate a bounded source shape into explicit typed `tcrv_rvv`
  structure; the provider remains the route authority.

## Requirements

* Register one new RVV vector source-front-door family for bounded
  runtime-scalar compare/select.
* Accept only source-only MLIR with one bounded function carrying:
  `cmp_lhs` memref, `rhs_scalar` scalar i32 runtime value, `passthrough`
  memref, `out` memref, and `n` index.
* Match exactly one rank-1 i32 vector `transfer_read` for `cmp_lhs`, one rank-1
  i32 vector `transfer_read` for `passthrough`, one scalar-to-vector RHS splat
  path, one supported `arith.cmpi` predicate (`eq`, `slt`, or `sle`) comparing
  vector lhs to splatted scalar RHS, one `arith.select` selecting lhs when true
  and passthrough when false, and one rank-1 unit-stride `transfer_write` to
  `out`.
* Materialize a selected `tcrv.exec` RVV variant whose typed body uses runtime
  ABI roles in the provider-expected order, including explicit runtime scalar
  RHS binding and `tcrv_rvv.splat`.
* Preserve the existing selected-body route/provider contract for
  `runtime_scalar_cmp_select`; do not add Common EmitC semantic branches or
  descriptor/source-artifact routes.
* Extend generated-bundle source-front-door expectations so
  `--vector-source-front-door --op-kind runtime_scalar_cmp_select` uses the new
  family contract, runs the new materializer, validates selected variant,
  runtime purpose, dispatch policy, provider mirror, generated function,
  prototype/ABI order, scalar RHS binding, AVL/VL behavior, source/tail
  preservation, and compare/select predicate coverage.
* Add focused positive transform/FileCheck coverage and fail-closed coverage for
  unsupported source shape, stale family marker/unsupported source kind, missing
  runtime scalar binding, stale ABI order or runtime purpose, stale dispatch
  policy, missing provider mirror, and disabled artifact-front-door eligibility
  where those checks are locally owned by this module.

## Acceptance Criteria

* [x] New family registration appears in RVV plugin source-front-door pass
  registration and default artifact-front-door eligibility is explicit.
* [x] Source-only transform test proves bounded runtime-scalar compare/select
  source MLIR materializes into selected `tcrv.exec` plus typed
  `tcrv_rvv.runtime_abi_value`, `setvl`, `with_vl`, `load`, `splat`,
  `compare`, `select`, and `store` body facts.
* [x] Negative transform/script evidence fails closed for stale markers,
  unsupported source shape/kind, stale selected variant/runtime purpose/dispatch
  policy, missing runtime scalar binding, stale ABI order, missing provider
  mirror, or disabled artifact-front-door eligibility as applicable.
* [x] Generated-bundle dry-run evidence proves selected variant, runtime
  purpose, dispatch policy mirror, provider-supported mirror, generated
  function/prototype, ABI order, runtime scalar value binding, AVL/VL behavior,
  source/tail preservation, and route metadata agreement.
* [x] Non-dry-run generated-bundle evidence runs on `ssh rvv` for bounded
  counts and records PASS output before runtime correctness is claimed.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit/script tests pass.
* [x] `python3 -m py_compile` and `--self-test` pass if the e2e script is
  touched.
* [x] Bounded old-authority scan over touched files and added diff lines shows
  no new legacy i32m1, descriptor/direct-C/source-export, route-id,
  artifact-name, or Common EmitC authority.
* [x] `git diff --check`, `git diff --cached --check`, clean final worktree,
  Trellis finish/archive, and one coherent commit.

## Non-Goals

* No broad source-front-door registry rewrite.
* No additional source-front-door families beyond runtime-scalar compare/select.
* No i64 or LMUL m2 clone batch unless it falls out of the same coherent base
  module without widening scope.
* No MAcc, reduction, memory, segment2, Linalg/Vector/StableHLO frontend
  generalization, performance database, dashboard, or report-only completion.
* No source-front-door or artifact metadata as route authority.
* No Common EmitC invention of RVV semantics.
* No revival of legacy `RVVI32M1*`, `rvv-i32m1`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` route authority.

## Technical Notes

Read and follow:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-07-06-07-stage3-rvv-registered-vector-source-front-door-artifact-bridge/`
* `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`
* `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* Existing runtime-scalar selected-body fixtures and generated-bundle tests
  under `test/Target/RVV/` and `test/Scripts/`.

## Completion Evidence

### Production Changes

* Added the registered
  `bounded-vector-runtime-scalar-cmp-select-source-front-door` RVV source
  family with default artifact-front-door eligibility.
* Added the public pass factory
  `createMaterializeRVVVectorRuntimeScalarCompareSelectSourceFrontDoorPass`.
* Added a bounded source matcher for one source-only MLIR function with
  `lhs`, runtime `rhs_scalar`, `true_value`, `false_value`, `out`, and `n`
  ABI roles.
* The materializer constructs a selected `tcrv.exec` RVV dispatch case and an
  explicit typed `tcrv_rvv` body containing runtime ABI values, `setvl`,
  `with_vl`, unit-stride loads, `tcrv_rvv.splat`, `tcrv_rvv.compare`,
  `tcrv_rvv.select`, and `tcrv_rvv.store`.
* Extended generated-bundle source-front-door evidence contracts so
  `--vector-source-front-door --op-kind runtime_scalar_cmp_select` runs the new
  registered materializer and validates selected variant, runtime purposes,
  dispatch mirrors, generated function/prototype, ABI order, provider mirror,
  runtime scalar RHS values, and selected-dispatch bundle mirrors.

### Evidence Commands

* `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "rvv-vector-runtime-scalar-cmp-select-source-front-door|rvv-vector-source-front-door-family-registry-negative"` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "rvv-generated-bundle-abi-e2e-vector-runtime-scalar-cmp-select-source-front-door-dry-run|rvv-generated-bundle-abi-e2e-vector-source-front-door-fail-closed|rvv-generated-bundle-abi-e2e-self-test"` from `build/test`
* Non-dry-run `ssh rvv` generated-bundle evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --vector-source-front-door --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-runtime-scalar-cmp-select-source-front-door-ssh --overwrite --op-kind runtime_scalar_cmp_select --runtime-count 0 --runtime-count 1 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`

### Runtime Result

`ssh rvv` output reported:

```text
PASS op=runtime_scalar_cmp_select counts=0,1,17,257 rhs_scalars=-37,91 true_lanes=332 false_lanes=216 mixed_cases=4 all_true_cases=0 all_false_cases=0
```

The per-case output also reported `tail_preserved` for counts `0`, `1`, `17`,
and `257` for both runtime scalar RHS values.

### Final Hygiene Before Commit

* `git diff --check` passed.
* Added-line old-authority scan for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, exact `__riscv_*_i32m1`,
  descriptor/direct-C/source-export/source-artifact terms found no added-line
  matches.
