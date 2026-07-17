# Stage2 RVV vector-reduction executable artifact closeout

## Goal

Carry the existing provider-contract-backed vector reduction path from selected
typed RVV body to generated artifact and real RVV correctness evidence. The
bounded owner is the current `reduce_add` route with `VectorRHSLoad` /
`vector-rhs-load` memory form:

```text
selected tcrv.exec RVV variant
  -> typed/pre-realized tcrv_rvv reduce_add body
  -> RVV plugin-owned selected-body realization and provider route facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> generated object/header bundle and external C ABI harness
  -> ssh rvv compile/run correctness evidence against the scalar oracle
```

This is an executable closeout for an already validated provider route
contract. It must not expand reduction coverage or move route authority into
metadata, target validation, common EmitC, scripts, or generated C strings.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV vector-reduction executable artifact closeout`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `4b8f60a6 rvv: extract vector reduction route validation contract`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md` require the RVV authority
  chain to remain selected `tcrv.exec` envelope -> typed low-level `tcrv_rvv`
  body -> RVV plugin-owned realization/provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> generated artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
* Common EmitC may materialize provider-built route payloads, but must not
  choose reduction semantics, dtype, SEW/LMUL, policy, ABI roles, intrinsic
  names, or statement structure.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-vector-reduction-route-family-provider-contract-extraction/`
  added the vector-reduction provider-owned route validation contract and
  passed focused local C++ checks plus the dry-run generated-bundle filter:
  `explicit-selected-body-artifact-reduce-add`,
  `pre-realized-selected-body-artifact-reduce-add`, and
  `rvv-generated-bundle-abi-e2e-pre-realized-reduce-add-dry-run`.
* The ordinary reduction realization archive already proved that
  `tcrv_rvv.typed_reduce_pre_realized_body` is consumed by the RVV reduction
  owner into explicit `setvl`, `with_vl`, source `load`, accumulator `load`,
  `reduce`, and `store` structure before provider route construction.
* The current evidence tool
  `scripts/rvv_generated_bundle_abi_e2e.py` already supports non-dry-run
  `ssh rvv` execution. It stages the generated object, header, and harness to
  the RVV target, compiles with
  `clang -O2 -march=rv64gcv -mabi=lp64d -I.`, runs the harness, and requires
  `PASS op=<kind>` in remote stdout.
* The focused pre-realized vector-reduction dry-run fixture is
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-reduce-add-dry-run.test`,
  which uses
  `test/Target/RVV/pre-realized-selected-body-artifact-reduce-add.mlir` and
  runtime counts `7,16,23`.

## Requirements

* Keep scope to one concrete `reduce_add` vector-reduction fixture or generated
  bundle using the existing typed/pre-realized selected-body path and existing
  `VectorRHSLoad` / `vector-rhs-load` route family.
* Generate the concrete vector-reduction artifact from the selected RVV IR via
  the production tool path, not by hand-writing C or bypassing
  `tcrv-opt` / `tcrv-translate`.
* Compile and run the generated object/header bundle on `ssh rvv` with the
  external C ABI harness and scalar-reference oracle.
* Runtime evidence must capture the generated artifact path, selected input,
  materialized selected body, generated RVV C/C++, object/header bundle,
  harness path, runtime counts, runtime ABI/order, `n`/AVL/VL behavior,
  accumulator/result layout, dtype/config, intrinsic/header/type mapping,
  provider-supported mirrors, and remote compile/run result.
* If runtime evidence exposes a real ABI, emitted C, header/type, layout,
  statement-plan, or runtime mismatch, fix the production owner that created
  the mismatch: RVV selected-body realization, provider route facts/statement
  plan, `TCRVEmitCLowerableRoute`, common materialization, target artifact
  export, or evidence harness generation as appropriate.
* If the existing production path is already executable, keep source changes to
  focused task context/evidence recording only. Do not invent helper-only
  compiler changes.
* Preserve provider route validation contracts. Do not weaken target
  validation or treat successful runtime execution as a replacement for the
  provider contract.
* Do not use route ids, artifact names, descriptor residue, exact intrinsic
  spelling, generated C strings, test names, metadata mirrors, or common EmitC
  as route authority.

## Acceptance Criteria

* [ ] The current task PRD, `implement.jsonl`, and `check.jsonl` record the
      bounded executable closeout context and relevant specs/tasks.
* [ ] Focused vector-reduction local generated-bundle checks still pass for
      the explicit selected-body fixture, pre-realized selected-body fixture,
      and pre-realized reduce-add dry-run script filter.
* [ ] `tianchenrv-target-artifact-export-test` passes.
* [ ] `tianchenrv-rvv-extension-plugin-test` passes if provider/plugin code is
      touched; otherwise it remains a focused confidence check because the
      route owner was just changed in the immediately preceding commit.
* [ ] A concrete pre-realized `reduce_add` generated bundle is produced under
      `artifacts/tmp/...` from
      `test/Target/RVV/pre-realized-selected-body-artifact-reduce-add.mlir`.
* [ ] The generated bundle is compiled and run on `ssh rvv` with scalar oracle
      correctness and reports `PASS op=reduce_add` for the selected runtime
      counts.
* [ ] Evidence records `ssh_evidence: true`, remote compile success, remote
      run success, artifact paths, runtime counts, runtime ABI/order,
      `n`/AVL/VL boundary, accumulator/result layout, dtype/config closure,
      header/type/intrinsic mirrors, and provider-supported mirror fields.
* [ ] Any production fix required by the executable run has focused local
      coverage and does not add new route-family coverage.
* [ ] Bounded old-authority scans over touched source/test/task files find no
      new positive dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m*`, descriptor/source-front-door/source-artifact,
      direct-C/source-export, exact-intrinsic, route-id, artifact-name,
      common-EmitC, or mirror-only route authority.
* [ ] `git diff --check` passes.
* [ ] Trellis task status, journal, archive state, final report, and commit
      truthfully record the executable evidence or the exact blocker.

## Technical Approach

1. Validate the existing focused vector-reduction dry-run path and identify the
   concrete artifact paths generated by
   `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind reduce_add`.
2. Run the same pre-realized `reduce_add` path without `--dry-run` against
   `ssh rvv`, using the production generated object/header bundle and harness.
3. Inspect the resulting `evidence.json`, `remote_compile_stdout.txt`, and
   `remote_run_stdout.txt` for ABI/order, `n`/AVL/VL, accumulator/result,
   dtype/config, header/type/intrinsic, and provider mirror closure.
4. If the remote run fails, repair only the production owner responsible for
   the failing route/emission/runtime behavior, then rerun the focused checks.
5. Record completed evidence in `check.jsonl` and task notes, run the bounded
   old-authority scan and diff check, then finish/archive and commit.

## Evidence Plan

* `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-reduce-add|pre-realized-selected-body-artifact-reduce-add|rvv-generated-bundle-abi-e2e-pre-realized-reduce-add-dry-run'`
* Dry-run artifact generation:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-04-vector-reduction-executable-closeout --run-id pre-realized-reduce-add-dry --overwrite --op-kind reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* Real RVV run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-04-vector-reduction-executable-closeout --run-id pre-realized-reduce-add-ssh-rvv --overwrite --op-kind reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* Bounded old-authority scan over changed files.
* `git diff --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-vector-reduction-executable-closeout`

## Out Of Scope

* No new reduction operations, standalone reduction expansion, accumulator
  layouts, dtype/LMUL clone batches, source-front-door routes, high-level
  Linalg/Vector frontend lowering, broad benchmark matrices, dashboards,
  artifact indexes, or unrelated route-family work.
* No weakening of provider contracts or movement of semantics into target
  validation, common EmitC, scripts, descriptors, metadata, route ids,
  artifact names, generated C strings, or exact intrinsic spellings.
* No performance claim. This task only claims executable correctness for the
  generated vector-reduction artifact if the `ssh rvv` run passes.

## Definition Of Done

* The existing provider-contract-backed vector-reduction path is either proven
  executable on `ssh rvv` with a scalar-reference harness, or the task remains
  open with the exact failing production boundary and artifact path recorded.
* Focused local checks, real RVV evidence, bounded old-authority scan, and
  `git diff --check` are recorded.
* Trellis status and archive state are truthful, and one coherent commit is
  created if the task is complete.

## Completion Notes

* No production source change was required. The existing provider-contract
  backed `reduce_add` / `vector-rhs-load` path already generated an executable
  object/header bundle and passed the scalar-reference harness on real RVV
  hardware.
* Focused local C++ checks passed:
  `tianchenrv-target-artifact-export-test` and
  `tianchenrv-rvv-extension-plugin-test`.
* Focused lit/generated-bundle checks passed from `build/test` for
  `explicit-selected-body-artifact-reduce-add`,
  `pre-realized-selected-body-artifact-reduce-add`, and
  `rvv-generated-bundle-abi-e2e-pre-realized-reduce-add-dry-run`.
* Dry-run generated artifact evidence passed at
  `artifacts/tmp/06-04-vector-reduction-executable-closeout/pre-realized-reduce-add-dry`.
* Real `ssh rvv` executable correctness passed at
  `artifacts/tmp/06-04-vector-reduction-executable-closeout/pre-realized-reduce-add-ssh-rvv`.
  The generated bundle used input
  `test/Target/RVV/pre-realized-selected-body-artifact-reduce-add.mlir`,
  generated object
  `reduce_add/generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o`,
  generated header
  `reduce_add/generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h`,
  and harness `reduce_add/rvv_generated_bundle_abi_reduce_add_harness.c`.
* Remote compile succeeded on `riscv64` with `/usr/bin/clang`
  `Ubuntu clang version 18.1.3 (1ubuntu1)` using
  `clang -O2 -march=rv64gcv -mabi=lp64d -I.`.
* Remote run output:

  ```text
  reduce_add case n=7 ok
  reduce_add case n=16 ok
  reduce_add case n=23 ok
  tcrv_rvv_generated_bundle_abi_reduce_add_ok counts=7,16,23
  PASS op=reduce_add counts=7,16,23
  ```

* Evidence records `ssh_evidence: true`, runtime counts `7,16,23`, ABI order
  `lhs,rhs,out,n`, header prototype
  `void tcrv_emitc_pre_realized_body_reduce_add_kernel_pre_realized_body_rvv_reduce_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);`,
  provider-derived vector reduction boundary, runtime AVL/VL boundary,
  accumulator layout `rhs-vector-seed-lane0-per-vl-chunk`, result layout
  `store-reduction-lane0-to-output-chunk-base`, store VL `1`, typed config
  `i32/sew32/lmul=m1`, and required header/type/intrinsic mirrors.
* Bounded old-authority scan over touched files found only task-file negative
  guardrails and common EmitC boundary descriptions; no production source/test
  old-authority drift was introduced.
* `git diff --check` passed.
