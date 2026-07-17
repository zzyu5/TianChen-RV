# RVV source-front-door demotion to explicit selected-body input

## Goal

Make the active RVV artifact/evidence path start from an explicit selected
`tcrv.exec` RVV variant containing a typed `tcrv_rvv` body. The retained vector
source front door is only a bounded, explicit seed that may construct typed
selected-body IR; it must not be the default artifact authority and must not
authorize route/export/runtime semantics through source patterns, test names,
route ids, metadata, descriptors, or `i32m1` helper names.

## Current Facts

- Current HEAD is `32aeca9`, and the worktree was clean when this task was
  created.
- Previous task
  `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-executable-artifact-gate`
  proved one selected-body add route can generate an object/header bundle and
  pass an external `ssh rvv` harness.
- The previous evidence runner still defaulted to
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door*.mlir` and invoked
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door` before target export.
- Focused target coverage already contains hand-authored explicit selected RVV
  IR under `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`,
  but the names and runner path still make the source seed look like the
  primary executable artifact entry.
- Specs require the authority chain:
  explicit selected `tcrv.exec.variant` with typed `tcrv_rvv` body -> RVV
  provider description -> common EmitC materialization -> target artifact
  export. Common EmitC/export remains neutral.

## Requirements

1. Add or repair the active evidence/export path so a hand-authored explicit
   selected `tcrv.exec`/`tcrv_rvv` fixture can drive RVV provider route
   description, common EmitC materialization, target artifact object/header/
   bundle export, and the generated-bundle evidence runner without first
   invoking the legacy vector source front-door pass.
2. Keep the source-front-door pass explicit and bounded. It may materialize the
   accepted tail-safe vector source seed into typed selected-body IR, but any
   remaining use must be documented/tested as seed construction only.
3. Prove source-front-door pattern labels, source-family fixture names,
   `tcrv_rvv.lowering_seed`, route ids, artifact names, metadata, descriptors,
   direct-C/source-export residue, and `i32m1` helper names cannot independently
   authorize route/export/runtime claims.
4. Do not expand RVV feature coverage. Retained add/sub/mul support remains the
   bounded specialization already present on the selected typed-body route
   surface.
5. Keep common EmitC and target/export code neutral. RVV semantics stay in the
   RVV plugin/provider/construction boundary.
6. If executable status is claimed for the explicit fixture, collect real
   `ssh rvv` evidence for that explicit selected-body fixture. Otherwise leave
   runtime status unclaimed and record the exact blocker.

## Acceptance Criteria

- [x] The generated-bundle evidence runner can use an explicit selected-body
      MLIR fixture by default and no longer requires the legacy vector
      source-front-door pass for the main artifact path.
- [x] Focused lit coverage proves explicit selected-body IR reaches selected
      emission-plan metadata and target artifact object/header/bundle export.
- [x] Focused negative coverage proves stale source seed metadata, source-only
      default front-door input, or stale selected-body route metadata cannot
      bypass selected typed-body validation.
- [x] Any retained source-front-door coverage is clearly explicit seed coverage:
      source input -> typed selected-body IR -> provider route construction.
- [x] Bounded residue scans over touched files show no descriptor/direct-C/
      source-export authority and no new common-code RVV semantic branch.
- [x] Focused build/tests and runner dry-run pass; `ssh rvv` run is collected
      only if this task claims executable explicit-fixture status.

## Non-Goals

- No broadcast, compare/select, reduction, conversion, dtype, LMUL,
  source-shape, intrinsic-table, Linalg/Vector/StableHLO, Scalar, IME, Offload,
  TensorExt, autotuning, dashboard, or broad smoke-matrix work.
- No descriptor-driven computation, direct-C semantic exporter restoration, or
  compatibility wrapper that preserves old source/i32 route authority.
- No report-only, helper-only, or artifact-index-only completion.

## Validation Plan

- Build focused tools/tests: `tcrv-opt`, `tcrv-translate`, RVV plugin tests,
  and target artifact export tests as needed.
- Run focused lit files under `test/Transforms/RVV`, `test/Conversion/EmitC`,
  `test/Target/RVV`, and `test/Scripts` touched by this task.
- Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
  the runner self-test.
- Run the runner dry-run with explicit selected-body fixture input.
- Run `ssh rvv` evidence only if local explicit selected-body artifact path is
  green and the task claims executable status.
- Run bounded scans for `source-front-door`, `lowering_seed`, `descriptor`,
  `direct-C`, `source-export`, `route id`, and `i32m1` residue over touched
  files.
- Run `git diff --check` and Trellis task validation before finish.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-executable-artifact-gate/prd.md`

## Initial Code Surface

- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door*.mlir`
- Focused selected-body EmitC/target tests under `test/Conversion/EmitC` and
  `test/Target/RVV`.

## Implementation Notes

- Added explicit selected-body target fixtures for add/sub/mul:
  `test/Target/RVV/explicit-selected-body-artifact-{add,sub,mul}.mlir`.
  These fixtures begin with a selected `tcrv.exec` RVV variant containing
  typed `tcrv_rvv` runtime ABI values, `setvl`, `with_vl`, load/compute/store
  body, and dispatch case. They do not invoke the legacy vector source
  materializer.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so default evidence input
  is explicit selected-body IR. The old source materializer remains available
  only behind `--source-seed`, records `input_mode =
  "legacy-rvv-source-seed"`, and documents that it may only construct typed
  selected-body IR before provider route construction.
- Replaced the old source-family script dry-run lit file with
  `test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test`,
  proving add/sub/mul dry-run evidence uses explicit selected-body fixtures,
  emits `front_door = "explicit-selected-tcrv-exec-rvv-body"`, records
  `source_seed = false`, and does not contain source-front-door, descriptor,
  direct-C/source-export, or `rvv_arithmetic_op` authority.
- No `.trellis/spec/` update was needed. This task implements the existing
  RVV plugin and lowering-runtime contracts rather than adding a new durable
  architecture rule.

## Evidence

- Local explicit selected-body dry-run artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-selected-body-front-door-demotion-dry`.
- Real `ssh rvv` explicit selected-body add artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-selected-body-front-door-demotion-add`.
- Source-seed sanity dry-run artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-selected-body-front-door-demotion-source-seed-dry`.
- Remote `ssh rvv` output for explicit selected-body add:

```text
add case n=7 ok
add case n=16 ok
add case n=23 ok
tcrv_rvv_generated_bundle_abi_add_ok counts=7,16,23
PASS op=add counts=7,16,23
```

## Checks Run

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-selected-body-front-door-demotion-dry --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --source-seed --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-selected-body-front-door-demotion-source-seed-dry --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-selected-body-front-door-demotion-add --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test ../test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test ../test/Target/RVV/explicit-selected-body-artifact-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-mul.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir`
  passed 7/7.
- Bounded residue scans over the explicit selected-body dry-run and `ssh rvv`
  artifacts:
  `rg -n "tcrv-rvv-materialize-i32m1-vector-source-front-door|legacy-rvv-source-front-door-seed|explicit-rvv-source-seed-to-selected-typed-body|source-family" ...`
  and
  `rg -n "descriptor|direct-C|direct_c|source-export|source_export|rvv_arithmetic_op" ...`
  found no matches.
- `git diff --check`
