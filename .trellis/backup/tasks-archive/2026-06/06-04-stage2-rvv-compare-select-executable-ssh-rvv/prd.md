# Stage2 RVV compare/select executable artifact closure on ssh rvv

## Goal

Close one representative Stage2 RVV compare/select executable-artifact evidence
gap. The selected path is the existing pre-realized selected-body plain
compare/select route, preferably `cmp_select_sle` or `cmp_select` depending on
live evidence. The path must start from a selected `tcrv.exec` RVV variant with
an explicit typed `tcrv_rvv` compare/select body, flow through RVV
plugin-owned selected-body realization and compare/select provider route
construction, materialize through neutral common EmitC, generate a target
artifact/header plus external ABI harness, and compile/run on real `ssh rvv`
with correctness evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select executable artifact closure on ssh rvv`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: no short status entries were reported.
* Initial `rtk git log --oneline -8` starts at
  `43357577 rvv: reject compare select stale route families`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* This round is run by one serial Codex worker; no subagents or multi-agent
  workflow are used.

## What I Already Know

* `.trellis/spec/index.md` defines the current RVV authority chain:
  selected `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC -> target artifact -> `ssh rvv` evidence when runtime,
  correctness, or performance is claimed.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires
  elementwise/compare-select selected-body realization to be an RVV
  plugin-owned boundary before route construction. Provider/common EmitC must
  not synthesize compare, select, mask, dtype, policy, or runtime facts from
  route ids, artifact names, status fields, C ABI names, or intrinsic strings.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires compare/select
  target validation to consume `RVVCompareSelectRouteValidationContract` and
  its embedded `RVVRuntimeAVLVLSelectedBoundaryContract` before accepting
  route payloads, header/type mappings, runtime ABI mappings,
  predicate/select facts, statement plans, or mirrors.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  execution for any runtime/correctness claim. For compare/select, the harness
  must prove both predicate-true and predicate-false lanes; dry-run evidence
  cannot substitute for runtime correctness.
* Archived task `06-04-stage2-rvv-compare-select-artifact` strengthened
  dry-run generated-bundle evidence for `cmp_select` and `cmp_select_sle`,
  but intentionally did not run `ssh rvv` or change runtime behavior.
* Archived task `06-04-06-04-stage2-rvv-compare-select-production` hardened
  compare/select target/provider fail-closed stale fact rejection, but also
  intentionally did not run `ssh rvv` or change generated C/C++ behavior.
* Live `scripts/rvv_generated_bundle_abi_e2e.py` already has a non-dry-run
  mode that stages generated object/header/harness to `ssh rvv`, compiles with
  `clang -O2 -march=rv64gcv -mabi=lp64d`, runs the harness, records per-op
  remote compile/run stdout, and summarizes `ssh_execution_summary` in
  evidence.
* Live dry-run test
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run.test`
  already uses the public selected lowering-boundary path and checks
  `cmp_select` plus `cmp_select_sle` predicate, runtime AVL/VL, generated C,
  and harness fields.

## Requirements

* Keep the owner bounded to one representative compare/select executable path.
* Prefer the existing pre-realized selected-body `cmp_select_sle` or
  `cmp_select` path, because it is the narrowest existing path with typed body,
  selected-boundary realization, provider route, artifact, and correctness
  harness coverage.
* Generate the artifact through production compiler/tooling, not by hand-writing
  C, descriptors, source-front-door routes, or route-id-derived shims.
* The generated artifact evidence must show predicate, select true/false
  operands, mask/policy, runtime `n`, setvl/VL, ABI prototype, generated
  object/header, and external C ABI harness facts.
* Non-dry-run evidence must compile and run on real `ssh rvv`; if the remote
  compile/run fails, record the exact blocker and do not claim runtime
  correctness.
* If route construction, ABI binding, EmitC emission, target validation, or
  harness integration blocks real execution, repair the production/tooling
  boundary that owns the blocker. Do not hide blockers with test-only expected
  output.
* If existing production code already works, keep the final diff honest as
  executable evidence/test/task plumbing only and record the no-production-code
  rationale.

## Acceptance Criteria

* [x] A representative pre-realized typed compare/select selected-body case is
      chosen and documented.
* [x] The selected path generates a target artifact/header and external ABI
      harness through `rvv_generated_bundle_abi_e2e.py` without
      `--dry-run`.
* [x] The same evidence records selected-body materialization through
      `--tcrv-materialize-selected-lowering-boundaries`, provider route
      metadata, runtime AVL/VL boundary, compare/select predicate boundary,
      generated object/header paths, and harness path.
* [x] The generated harness compiles and runs on `ssh rvv`; the remote output
      includes the expected pass marker and `PASS op=<chosen-op>`.
* [x] The correctness oracle covers runtime counts including zero, one, at
      least one multi-lane/multi-VL count, and both predicate-true and
      predicate-false lanes.
* [x] If production/tooling code changes, focused checks cover the changed
      owner: generated-bundle lit/export checks plus
      `tianchenrv-target-artifact-export-test` and/or
      `tianchenrv-rvv-extension-plugin-test` as applicable.
* [x] If no production code changes, focused dry-run lit and the non-dry-run
      `ssh rvv` evidence are sufficient, and the no-production-change
      rationale is recorded.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      route-id/artifact-name authority, common EmitC semantic inference,
      exact `__riscv_*_i32m1`, or mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task context is valid and task status is truthful.
* [ ] A coherent commit is created if the task completes.

## Technical Approach

1. Run a focused non-dry-run generated-bundle command for one representative
   pre-realized compare/select case, starting with `cmp_select_sle` because it
   exercises a non-equality predicate while preserving the same minimal route
   family.
2. Inspect generated `evidence.json`, per-op `evidence.json`,
   `materialized_rvv_emitc.cpp`, generated bundle header/object, generated
   harness, `remote_compile_stdout.txt`, and `remote_run_stdout.txt`.
3. If the run fails, classify the blocker as remote environment, generated C/C++
   emission, target validation/bundle, ABI/harness integration, or selected
   realization/provider route, then repair the owning production/tooling code.
4. Add or update focused lit/test coverage only if the fix changes observable
   behavior or if the repository lacks a stable way to preserve the executable
   evidence command.
5. Run focused checks, bounded old-authority scan, Trellis validation,
   finish/archive, and commit.

## Evidence Plan

Primary executable evidence command:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv --run-id cmp-select-sle-ssh-rvv --overwrite --op-kind cmp_select_sle --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj --ssh-target rvv
```

Focused dry-run/lit regression if no production code changes:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-cmp-select-dry-run'
```

Production checks if provider/target/plugin code changes:

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 8
rtk ./build/bin/tianchenrv-target-artifact-export-test
rtk ./build/bin/tianchenrv-rvv-extension-plugin-test
```

Always:

```text
rtk git diff --check
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-compare-select-executable-ssh-rvv
```

## Out Of Scope

* New high-level Linalg/Vector/StableHLO frontend lowering.
* Source-front-door positive routes.
* One-intrinsic wrapper dialects.
* New compare/select predicates, dtype/LMUL clone batches, or unrelated
  reduction/conversion/memory/contraction expansion.
* Broad smoke matrices, dashboards, or report-only artifacts.
* Inferring compare/select semantics from route ids, artifact names, test
  names, manifests, descriptors, C strings, or mirror metadata.
* Moving RVV semantics into common EmitC/export.

## Definition Of Done

One representative pre-realized typed compare/select selected-body path has a
generated artifact, generated harness, real `ssh rvv` compile/run transcript,
and correctness result. Any production blocker found on that path is fixed in
the owning code; otherwise the task records that no production code change was
needed because the existing route/emission/harness path already executed on
real RVV hardware.

## Implementation Results

Chosen representative submodule:

* `cmp_select_sle`
* input fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-cmp-select-sle.mlir`
* selected variant: `pre_realized_body_rvv_cmp_select_sle`
* external ABI function:
  `tcrv_emitc_pre_realized_body_cmp_select_sle_kernel_pre_realized_body_rvv_cmp_select_sle`

No production code changes were required. Live execution showed the existing
production path already performs:

```text
pre-realized selected tcrv.exec RVV variant
  -> public selected lowering-boundary materialization
  -> realized typed tcrv_rvv setvl/with_vl/load/compare/select/store body
  -> provider-built compare/select TCRVEmitCLowerableRoute
  -> target artifact bundle export
  -> generated external C ABI harness
  -> ssh rvv compile/run
```

The task changes are limited to Trellis task context and PRD evidence records.
The generated evidence under `artifacts/tmp/` is intentionally not tracked by
git; it remains in the workspace as the runtime transcript/artifact output for
this round.

## Evidence Results

### Executable ssh rvv Evidence

Command:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv --run-id cmp-select-sle-ssh-rvv --overwrite --op-kind cmp_select_sle --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
artifact_dir: artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv
[cmp_select_sle] cmp_select_sle case n=0 ok predicate_true_lanes=0 predicate_false_lanes=0
cmp_select_sle case n=1 ok predicate_true_lanes=1 predicate_false_lanes=0
cmp_select_sle case n=7 ok predicate_true_lanes=5 predicate_false_lanes=2
cmp_select_sle case n=16 ok predicate_true_lanes=10 predicate_false_lanes=6
cmp_select_sle case n=23 ok predicate_true_lanes=14 predicate_false_lanes=9
cmp_select_sle case n=257 ok predicate_true_lanes=155 predicate_false_lanes=102
tcrv_rvv_generated_bundle_abi_cmp_select_sle_ok counts=0,1,7,16,23,257
PASS op=cmp_select_sle counts=0,1,7,16,23,257
```

Primary evidence paths:

* Root evidence:
  `artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv/evidence.json`
* Per-op evidence:
  `artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv/cmp_select_sle/evidence.json`
* Materialized selected body:
  `artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv/cmp_select_sle/materialized_selected_body.mlir`
* Generated EmitC C++:
  `artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv/cmp_select_sle/materialized_rvv_emitc.cpp`
* Generated bundle object:
  `artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv/cmp_select_sle/generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o`
* Generated bundle header:
  `artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv/cmp_select_sle/generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h`
* External ABI harness:
  `artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv/cmp_select_sle/rvv_generated_bundle_abi_cmp_select_sle_harness.c`
* Remote compile stdout:
  `artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv/cmp_select_sle/remote_compile_stdout.txt`
* Remote run stdout:
  `artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv/cmp-select-sle-ssh-rvv/cmp_select_sle/remote_run_stdout.txt`

Remote compile summary:

```text
remote_arch=riscv64
clang_path=/usr/bin/clang
clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)
```

Evidence highlights:

* `ssh_evidence`: `true`
* remote compile: succeeded
* remote run: succeeded
* predicate: `sle`
* selected value operands: true=`lhs`, false=`rhs`, output=`out`
* generated intrinsics:
  `__riscv_vsetvl_e32m1`, `__riscv_vle32_v_i32m1`,
  `__riscv_vmsle_vv_i32m1_b32`, `__riscv_vmerge_vvm_i32m1`,
  `__riscv_vse32_v_i32m1`
* runtime counts: `0, 1, 7, 16, 23, 257`
* multi-lane predicate coverage includes both true and false lanes for
  `n=7`, `n=16`, `n=23`, and `n=257`.

### Focused Lit

Run from `build/test`:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-cmp-select-dry-run'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 476 (99.79%)
  Passed  :   1 (0.21%)
```

### Provider/Target C++ Tests

`tianchenrv-target-artifact-export-test` and
`tianchenrv-rvv-extension-plugin-test` were not rerun. This round did not
change provider, target validation, plugin realization, route construction,
common EmitC, generated C/C++ emission, runtime ABI order, statement ordering,
or harness-generation code. The existing focused dry-run lit test plus the
non-dry-run `ssh rvv` generated-bundle evidence cover the changed task claim.

### Static Checks

```text
rtk git diff --check
```

Result: passed.

Bounded old-authority scan judgment:

* Touched files are Trellis task files only.
* Matches for legacy/source/export guardrail terms, if any, are in PRD
  red-line, out-of-scope, or scan-criteria text.
* No touched compiler, test, or tooling file adds a positive dependency on old
  route authority, source-front-door, descriptor/direct-C/source-export, route
  id or artifact-name semantics, common EmitC semantic inference, exact
  intrinsic spelling authority, or mirror-only authority.

### Trellis Validation

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-compare-select-executable-ssh-rvv
```

Result: passed; `implement.jsonl` and `check.jsonl` each contain four valid
spec entries.

## Spec Update Judgment

No `.trellis/spec/` update is needed. The durable rule already exists in
`.trellis/spec/testing/mlir-testing-contract.md`: runtime/correctness claims
require real `ssh rvv` evidence, and compare/select generated-bundle evidence
must include predicate true/false lane correctness. This round applies that
existing rule to the representative `cmp_select_sle` path without introducing
a new architecture contract.
