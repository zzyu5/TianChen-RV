# Stage2 RVV pre-realized selected-body executable closure

## Goal

Turn the existing direct pre-realized selected-body route-entry generated-bundle
ABI proof into a real executable closure on the RVV target for one bounded
statement-plan-backed cluster. The selected path must start from a pre-realized
typed `tcrv_rvv` body in a selected `tcrv.exec` RVV variant, realize through the
RVV plugin-owned route-entry bridge before route facts are collected, build a
provider-owned `TCRVEmitCLowerableRoute`, materialize through common EmitC,
export a generated target artifact bundle, compile/link it on `ssh rvv`, and run
a deterministic correctness harness.

This round is an execution-closure task for existing direct route-entry
behavior. It is not new RVV operation coverage and not another dry-run matrix.

## Direction Source

- Direction title: `Stage2 RVV pre-realized selected-body executable closure`.
- Module owner: RVV pre-realized selected-body generated artifact compiled and
  run through the real `ssh rvv` executable path for one bounded
  statement-plan-backed kernel cluster.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `13f762d4 rvv: prove pre-realized route-entry artifact abi`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` requires the RVV-first authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence when runtime/correctness is claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires the route-entry
  bridge to realize supported pre-realized selected bodies before route facts
  are collected. Unsupported or malformed pre-realized route-entry families
  must fail closed before provider/common route construction.
- `.trellis/spec/lowering-runtime/emitc-route.md` keeps RVV dtype/config,
  operation, ABI mapping, intrinsic spelling, and route support in the RVV
  plugin/provider. Common EmitC and target artifact export are neutral
  consumers of provider-built routes.
- `.trellis/spec/testing/mlir-testing-contract.md` says direct pre-realized
  route-entry generated-bundle evidence proves:
  selected pre-realized `tcrv_rvv` body -> route-entry bridge ->
  provider-built route -> target artifact bundle -> external C ABI harness.
  Runtime/correctness claims require non-dry-run `ssh rvv` compile/run
  evidence.
- The archived previous task
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-pre-realized-selected-body-artifact-abi-integration/prd.md`
  added direct route-entry dry-run script/test/fixture evidence for
  `cmp_select` and `strided_load_unit_store`.
- Current code evidence:
  - `scripts/rvv_generated_bundle_abi_e2e.py` already has
    `--pre-realized-selected-body --direct-pre-realized-route-entry`, bounded
    to `cmp_select`, `cmp_select_sle`, and `strided_load_unit_store`.
  - The same script has a non-dry-run remote path that stages the generated
    object/header/harness, compiles them on `ssh rvv` with
    `clang -O2 -march=rv64gcv -mabi=lp64d`, runs the harness, and requires the
    op-specific `PASS` marker.
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-route-entry-dry-run.test`
    proves the direct route-entry generated-bundle dry-run path only; it does
    not itself reach `ssh rvv`.
  - `test/Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir` and
    `test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir`
    already prove direct `--tcrv-materialize-emission-plans` artifact/header
    export without the explicit selected-boundary materializer.

## Requirements

1. Use the existing bounded direct route-entry cluster:
   - `cmp_select`, proving compare/select predicate true/false lane coverage;
   - `strided_load_unit_store`, proving byte-strided load, contiguous output,
     runtime `stride_bytes`, runtime `n`/AVL, and tail preservation.
2. The main evidence must be a non-dry-run generated-bundle ABI/e2e command
   that reaches `ssh rvv` compile/link/run for the generated artifact bundle.
3. The generated path must preserve direct route-entry semantics:
   - input mode is `pre-realized-selected-body`;
   - `--direct-pre-realized-route-entry` is set;
   - local pipeline includes `--tcrv-materialize-emission-plans` and target
     artifact bundle export;
   - local pipeline does not insert
     `--tcrv-materialize-selected-lowering-boundaries`;
   - `materializer` is `rvv-route-entry-selected-body-realization`;
   - `route_entry_realization` and `pre_realized_body_consumed` are true.
4. The correctness oracle must be deterministic and external to compiler
   metadata:
   - `cmp_select`: harness checks `lhs[index] == rhs[index] ? lhs : rhs`,
     requires both true and false predicate lanes, and reports
     `PASS op=cmp_select counts=7,16,23`.
   - `strided_load_unit_store`: harness checks loads from
     `src_raw + index * stride_bytes`, writes contiguous output, preserves
     output tail sentinels, and reports
     `PASS op=strided_load_unit_store counts=7,16,23 stride_bytes=4,8,12`.
5. Unsupported or incomplete direct route-entry cases must still fail closed
   before execution. Existing negative CLI checks for missing
   `--pre-realized-selected-body` and unsupported direct op kinds must remain.
6. Do not move RVV semantics into common EmitC, target artifact plumbing, test
   names, route ids, descriptor/source-front-door metadata, or Python compiler
   data structures.

## Acceptance Criteria

- [x] Trellis PRD, `implement.jsonl`, and `check.jsonl` reference the relevant
      RVV plugin, EmitC route, testing specs, and the previous direct
      route-entry artifact ABI PRD.
- [x] A focused non-dry-run command reaches `ssh rvv` compile/link/run for
      `cmp_select` and `strided_load_unit_store` using the direct pre-realized
      route-entry generated-bundle path.
- [x] Remote compile evidence records RISC-V architecture/toolchain output and
      successful `clang -O2 -march=rv64gcv -mabi=lp64d` compile/link.
- [x] Remote run evidence records `PASS op=cmp_select counts=7,16,23` and
      `PASS op=strided_load_unit_store counts=7,16,23 stride_bytes=4,8,12`.
- [x] Evidence JSON for both ops records `ssh_evidence: true`,
      `route_entry_realization: true`, `pre_realized_body_consumed: true`, and
      the direct route-entry materializer label.
- [x] Existing dry-run direct route-entry lit/FileCheck coverage remains
      passing.
- [x] Existing fail-closed direct-route-entry CLI checks remain passing.
- [x] If source changes are needed, they are bounded to the executable closure
      blocker and do not add new RVV route coverage or common EmitC semantics.
- [x] Bounded active-authority scans over touched files show no new legacy
      `RVVI32M1`, `rvv-i32m1`, positive finite `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      or mirror-only route authority drift.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is documented.
- [x] Final task status and journal truthfully record whether the executable
      closure is complete, blocked, or still dry-run-only in any part.

## Out Of Scope

- No new route coverage, operation families, reductions, contractions,
  dtype/LMUL clone batches, high-level frontend lowering, source-front-door
  positive routes, descriptor/direct-C/source-export paths, dashboards, broad
  smoke matrices, or common EmitC semantic logic.
- No performance claim; correctness/runtime execution evidence is enough.
- No migration to Scalar, IME, Offload, TensorExt, Template/Toy, or future
  plugin work.
- No reliance on status fields, result labels, route ids, artifact names,
  manifests, or test names as execution authority.

## Technical Approach

1. Validate the current non-dry-run generated-bundle path for the bounded direct
   route-entry cluster.
2. If it already compiles/runs on `ssh rvv`, keep source changes minimal and
   record the executable closure through task evidence, focused checks, and
   journal/task status.
3. If the remote path fails, fix only the single blocker that prevents the
   generated artifact bundle from compiling/linking/running on RVV, preserving
   the existing RVV plugin/provider/common ownership boundaries.
4. Retain existing dry-run lit coverage as local fallback/regression coverage,
   but do not treat dry-run success as the main result.

## Validation Plan

1. Validate/start this Trellis task.
2. Run the focused non-dry-run direct route-entry generated-bundle command:

   ```bash
   python3 scripts/rvv_generated_bundle_abi_e2e.py \
     --pre-realized-selected-body \
     --direct-pre-realized-route-entry \
     --op-kind cmp_select \
     --op-kind strided_load_unit_store \
     --runtime-count 7 \
     --runtime-count 16 \
     --runtime-count 23 \
     --stride-bytes 4 \
     --stride-bytes 8 \
     --stride-bytes 12
   ```

3. Inspect generated evidence JSON and remote stdout files for the direct
   route-entry materializer and `PASS` markers.
4. Run the existing direct dry-run lit/FileCheck test.
5. Run direct-route-entry negative CLI checks for missing
   `--pre-realized-selected-body` and unsupported op kind.
6. Run bounded active-authority scans over touched files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Implementation Result

- No compiler source, dialect, provider, common EmitC, target artifact, fixture,
  or script code change was required in this round.
- The existing direct pre-realized route-entry generated-bundle path was
  executed in non-dry-run mode on real `ssh rvv` for the bounded cluster:
  `cmp_select` and `strided_load_unit_store`.
- The generated evidence is under:

  ```text
  artifacts/tmp/stage2-pre-realized-executable-closure/direct-pre-realized-route-entry-ssh-rvv
  ```

- Root evidence records:
  - `dry_run: false`;
  - `input_mode: pre-realized-selected-body`;
  - `pre_realized_route_entry_mode: direct`;
  - `ssh_evidence: true`;
  - `status: success`;
  - `op_kinds: cmp_select, strided_load_unit_store`.
- Per-op evidence records:
  - `materializer: rvv-route-entry-selected-body-realization`;
  - `route_entry_realization: true`;
  - `pre_realized_body_consumed: true`;
  - local pipeline uses `--tcrv-materialize-emission-plans` and
    `--tcrv-export-target-artifact-bundle`;
  - local pipeline does not use
    `--tcrv-materialize-selected-lowering-boundaries`;
  - `remote_compile_succeeded: true`;
  - `remote_run_succeeded: true`;
  - `ssh_evidence: true`.
- Remote compile evidence for both ops records:
  - `remote_arch=riscv64`;
  - `clang_path=/usr/bin/clang`;
  - `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`;
  - compile command includes `clang -O2 -march=rv64gcv -mabi=lp64d`.
- Correctness oracle:
  - `cmp_select` harness checks true and false predicate lanes for
    `n=7,16,23`, then reports
    `PASS op=cmp_select counts=7,16,23`.
  - `strided_load_unit_store` harness checks `stride_bytes=4,8,12` for
    `n=7,16,23`, validates byte-strided load, contiguous output, and tail
    preservation, then reports
    `PASS op=strided_load_unit_store counts=7,16,23 stride_bytes=4,8,12`.

## Validation Result

- Trellis context validation passed:

  ```bash
  rtk python3 ./.trellis/scripts/task.py validate \
    .trellis/tasks/05-25-stage2-rvv-pre-realized-selected-body-executable-closure
  ```

- Focused non-dry-run direct route-entry `ssh rvv` evidence passed:

  ```bash
  rtk python3 scripts/rvv_generated_bundle_abi_e2e.py \
    --pre-realized-selected-body \
    --direct-pre-realized-route-entry \
    --artifact-root artifacts/tmp/stage2-pre-realized-executable-closure \
    --run-id direct-pre-realized-route-entry-ssh-rvv \
    --overwrite \
    --op-kind cmp_select \
    --op-kind strided_load_unit_store \
    --runtime-count 7 \
    --runtime-count 16 \
    --runtime-count 23 \
    --stride-bytes 4 \
    --stride-bytes 8 \
    --stride-bytes 12 \
    --tcrv-opt build/bin/tcrv-opt \
    --tcrv-translate build/bin/tcrv-translate \
    --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
  ```

- Remote run output included:
  - `PASS op=cmp_select counts=7,16,23`;
  - `PASS op=strided_load_unit_store counts=7,16,23 stride_bytes=4,8,12`.
- Focused dry-run lit regression passed after rerunning from `build/test` so
  `lit.site.cfg.py` resolves its relative config path:

  ```bash
  rtk /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv \
    /home/kingdom/phdworks/TianchenRV/build/test \
    --filter rvv-generated-bundle-abi-e2e-direct-pre-realized-route-entry-dry-run
  ```

  Result: 365 discovered, 364 excluded, 1 passed.
- Direct route-entry fail-closed CLI checks passed by rejecting invalid input:
  - missing `--pre-realized-selected-body` rejected with
    `--direct-pre-realized-route-entry requires --pre-realized-selected-body`;
  - unsupported direct `reduce_add` rejected with
    `--direct-pre-realized-route-entry is bounded to pre-realized cmp_select/cmp_select_sle and strided_load_unit_store fixtures`.
- Full quality gate passed:

  ```bash
  rtk cmake --build build --target check-tianchenrv -j2
  ```

  Result: 365/365 tests passed.
- First focused lit invocation from the repository root failed because
  `build/test/lit.site.cfg.py` loads `../../test/lit.cfg.py` relative to the
  `build/test` cwd used by the generated CMake command. This was a command
  invocation issue, not a compiler or test failure; rerunning from `build/test`
  passed.
