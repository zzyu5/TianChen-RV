# Stage2 RVV computed-mask standalone reduction executable artifact

## Goal

Close one representative Stage2 RVV computed-mask standalone reduction
executable-artifact gap. The selected path is
`computed_mask_standalone_reduce_add` unless live inspection exposes a tighter
blocker. The path must start from a selected `tcrv.exec` RVV variant with an
explicit typed `tcrv_rvv` computed-mask standalone reduction body, flow through
RVV plugin-owned selected-body realization and provider route construction,
materialize through neutral common EmitC, generate a target artifact/header plus
external ABI harness, and compile/run on real `ssh rvv` with correctness
evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask standalone reduction route to executable artifact`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: no short status entries were reported.
* Initial `rtk git log --oneline -8` starts at
  `07c6c79b rvv: record compare select ssh evidence`.
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
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires standalone
  reduction, computed-mask standalone reduction, and runtime-scalar
  computed-mask standalone reduction facts to be owned by RVV provider/route
  contracts. Common EmitC/export must not infer reduction kind, mask
  production, accumulator seed/layout, dtype, policy, or runtime facts from
  route ids, artifact names, C ABI names, or mirror fields.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires standalone reduction
  target validation to consume provider-built route facts, runtime AVL/VL
  boundary facts, operand binding facts, and route-family validation contracts
  before accepting payloads, header/type mappings, statement plans, or mirrors.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  execution for any runtime/correctness claim. Generated-bundle dry-run
  evidence cannot substitute for hardware execution.
* Archived compare/select task
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-compare-select-executable-ssh-rvv/`
  shows the expected pattern for this round: if production route/emission/ABI
  already works, record focused executable evidence honestly instead of making
  unrelated code changes.
* Live repository search shows existing pre-realized computed-mask standalone
  reduction fixtures and dry-run script tests for `computed_mask_standalone_reduce_add`
  and LMUL/min/max variants. This task must convert one representative add path
  into executable compiler-path evidence, not broaden the family.

## Requirements

* Keep the owner bounded to one representative computed-mask standalone
  reduction executable path.
* Prefer existing `computed_mask_standalone_reduce_add`, because it directly
  exercises compare-produced mask consumption, inactive-lane neutralization,
  accumulator seed, scalar result storage, runtime `n`, and setvl/VL.
* Generate the artifact through production compiler/tooling, not by hand-writing
  C, descriptors, source-front-door routes, or route-id-derived shims.
* The generated artifact evidence must preserve compare-produced mask facts,
  reduction kind, accumulator seed/layout, scalar output, inactive-lane policy,
  runtime `n`, setvl/VL, ABI prototype, generated object/header, and external C
  ABI harness facts.
* Unsupported or stale non-reduction provider/candidate facts must fail closed
  before they become artifact authority.
* Non-dry-run evidence must compile and run on real `ssh rvv`; if remote
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

* [x] A representative pre-realized typed computed-mask standalone reduction
      selected-body case is chosen and documented.
* [x] The selected path generates a target artifact/header and external ABI
      harness through `rvv_generated_bundle_abi_e2e.py` without `--dry-run`.
* [x] The same evidence records selected-body materialization through
      provider-owned realization/route metadata, runtime AVL/VL boundary,
      compare-produced mask facts, inactive-lane neutralization, accumulator
      seed/layout, scalar result storage, generated object/header paths, and
      harness path.
* [x] The generated harness compiles and runs on `ssh rvv`; the remote output
      includes the expected pass marker and
      `PASS op=computed_mask_standalone_reduce_add`.
* [x] The correctness oracle covers runtime counts including zero, one, at
      least one multi-lane/multi-VL count, mask-selected lanes, mask-inactive
      lanes, accumulator seed use, and scalar reduction output.
* [x] If production/tooling code changes, focused checks cover the changed
      owner: generated-bundle lit/export checks plus
      `tianchenrv-target-artifact-export-test` and/or
      `tianchenrv-rvv-extension-plugin-test` as applicable.
* [x] If no production code changes, focused dry-run lit and the non-dry-run
      `ssh rvv` evidence are sufficient, and the no-production-change rationale
      is recorded.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      route-id/artifact-name authority, common EmitC semantic inference, exact
      `__riscv_*_i32m1`, or mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task context is valid and task status is truthful.
* [x] A coherent commit is created if the task completes.

## Technical Approach

1. Inspect the standalone reduction specs, existing pre-realized fixtures,
   route provider/planning/validation code, and generated-bundle harness support
   for `computed_mask_standalone_reduce_add`.
2. Run the existing dry-run/generated-bundle path to confirm the route emits the
   expected artifact, statement plan, harness, and evidence facts.
3. Run a focused non-dry-run generated-bundle command for
   `computed_mask_standalone_reduce_add` on `ssh rvv` with counts
   `0,1,7,16,23,257`.
4. Inspect generated `evidence.json`, per-op `evidence.json`,
   `materialized_selected_body.mlir`, `materialized_rvv_emitc.cpp`, generated
   bundle header/object, generated harness, `remote_compile_stdout.txt`, and
   `remote_run_stdout.txt`.
5. If the run fails, classify the blocker as remote environment, generated C/C++
   emission, target validation/bundle, ABI/harness integration, or selected
   realization/provider route, then repair the owning production/tooling code.
6. Add or update focused lit/test coverage only if the fix changes observable
   behavior or if the repository lacks a stable way to preserve the executable
   evidence command.
7. Run focused checks, bounded old-authority scan, Trellis validation,
   finish/archive, and commit.

## Evidence Plan

Primary executable evidence command:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact --run-id computed-mask-standalone-reduce-add-ssh-rvv --overwrite --op-kind computed_mask_standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Focused dry-run/lit regression if no production code changes:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-mask-standalone-reduce-add'
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
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-computed-mask-standalone-reduction-executable-artifact
```

## Out Of Scope

* All reductions, min/max clone batches, dtype/LMUL expansion beyond what the
  chosen existing path already requires.
* High-level Linalg/Vector/StableHLO frontend lowering.
* Source-front-door positive routes.
* One-intrinsic wrapper dialects.
* Unrelated memory/conversion/contraction work.
* Broad smoke matrices, dashboards, or report-only artifacts.
* Inferring reduction semantics from route ids, artifact names, test names,
  manifests, descriptors, C strings, or mirror metadata.
* Moving RVV semantics into common EmitC/export.

## Definition Of Done

One representative pre-realized typed computed-mask standalone reduction
selected-body path has a generated artifact, generated harness, real `ssh rvv`
compile/run transcript, and correctness result. Any production blocker found on
that path is fixed in the owning code; otherwise the task records that no
production code change was needed because the existing route/emission/harness
path already executed on real RVV hardware.

## Implementation Results

Chosen representative submodule:

* `computed_mask_standalone_reduce_add`
* input fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-standalone-reduce-add.mlir`
* selected variant: `rvv_pre_cm_standalone_reduce`
* external ABI function:
  `tcrv_emitc_pre_cm_standalone_reduce_kernel_rvv_pre_cm_standalone_reduce`

No production code changes were required. Live execution showed the existing
production path already performs:

```text
pre-realized selected tcrv.exec RVV variant
  -> RVV plugin selected-body materialization
  -> realized typed tcrv_rvv setvl/with_vl/load/compare/masked_standalone_reduce/store body
  -> provider-built computed-mask standalone reduction TCRVEmitCLowerableRoute
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
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact --run-id computed-mask-standalone-reduce-add-ssh-rvv --overwrite --op-kind computed_mask_standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
artifact_dir: artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv
tcrv_rvv_generated_bundle_abi_computed_mask_standalone_reduce_add_ok counts=0,1,7,16,23,257 seeds=-11,17 patterns=0,1
PASS op=computed_mask_standalone_reduce_add counts=0,1,7,16,23,257 seeds=-11,17 patterns=0,1
```

The remote compile summary recorded:

```text
remote_arch=riscv64
clang_path=/usr/bin/clang
clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)
```

Primary evidence paths:

* Root evidence:
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv/evidence.json`
* Per-op evidence:
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv/computed_mask_standalone_reduce_add/evidence.json`
* Materialized selected body:
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv/computed_mask_standalone_reduce_add/materialized_selected_body.mlir`
* Generated EmitC C++:
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv/computed_mask_standalone_reduce_add/materialized_rvv_emitc.cpp`
* Generated bundle object:
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv/computed_mask_standalone_reduce_add/generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o`
* Generated bundle header:
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv/computed_mask_standalone_reduce_add/generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h`
* Generated external ABI harness:
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv/computed_mask_standalone_reduce_add/rvv_generated_bundle_abi_computed_mask_standalone_reduce_add_harness.c`
* Remote compile transcript:
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv/computed_mask_standalone_reduce_add/remote_compile_stdout.txt`
* Remote run transcript:
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv/computed_mask_standalone_reduce_add/remote_run_stdout.txt`

### Structural Evidence

The materialized selected body contains explicit structural facts:

```text
tcrv_rvv.setvl %n
tcrv_rvv.with_vl
tcrv_rvv.load cmp_lhs
tcrv_rvv.load cmp_rhs
tcrv_rvv.load src
tcrv_rvv.compare kind = "sle"
tcrv_rvv.masked_standalone_reduce kind = "add"
tcrv_rvv.store out
```

The generated C++ materializes the provider-built route using RVV intrinsics:

```text
__riscv_vsetvl_e32m1
__riscv_vle32_v_i32m1
__riscv_vmsle_vv_i32m1_b32
__riscv_vmv_v_x_i32m1(0, ...)
__riscv_vmerge_vvm_i32m1
__riscv_vredsum_vs_i32m1_i32m1
__riscv_vse32_v_i32m1
```

Per-op evidence records:

* runtime ABI order: `cmp_lhs,cmp_rhs,src,acc,out,n`
* compare-produced mask role/source/form:
  `predicate-mask-produced-by-compare`,
  `compare-produced-mask-same-vl-scope`, `compare-produced-mask`
* inactive-lane contract:
  `masked-standalone-reduction-zero-inactive-lanes-before-reduction`
* inactive neutral literal: `0`
* scalar result runtime boundary:
  `scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1`
* standalone source/result channel types:
  `!tcrv_rvv.vector<i32, "m1">`, `vint32m1_t`
* provider mirror:
  `provider_supported_mirror:rvv-computed-mask-standalone-reduction-plan-validated`

The generated harness checked:

* counts `0,1,7,16,23,257`
* seeds `-11,17`
* patterns `0,1`
* active and inactive compare mask lanes
* all-inactive mask cases preserving the seed
* scalar output in `out[0]`
* source preservation
* tail sentinel preservation

## Check Results

Checks run:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact --run-id computed-mask-standalone-reduce-add-dry-run --overwrite --op-kind computed_mask_standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact --run-id computed-mask-standalone-reduce-add-ssh-rvv --overwrite --op-kind computed_mask_standalone_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Additional focused checks and final status are recorded during task finish.

Focused lit:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-mask-standalone-reduce-add'
```

The first invocation from the repository root failed because `.` is not a lit
suite root in this checkout:

```text
error: did not discover any tests for provided path(s)
```

Self-repair reran the same filter from `build/test`, where the configured lit
site file can load `../../test/lit.cfg.py` correctly:

```text
workdir: /home/kingdom/phdworks/TianchenRV/build/test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-mask-standalone-reduce-add'
Total Discovered Tests: 477
  Excluded: 471 (98.74%)
  Passed  :   6 (1.26%)
```

Trellis context validation:

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-computed-mask-standalone-reduction-executable-artifact
implement.jsonl: ok (5 entries)
check.jsonl: ok (4 entries)
```

Archive validation:

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-04-stage2-rvv-computed-mask-standalone-reduction-executable-artifact
implement.jsonl: ok (5 entries)
check.jsonl: ok (4 entries)
```

Whitespace check:

```text
rtk git diff --check
rtk git diff --cached --check
```

Result: pass, no output.

Bounded old-authority scan:

```text
rtk git diff --cached -U0 -- .trellis/tasks/archive/2026-06/06-04-stage2-rvv-computed-mask-standalone-reduction-executable-artifact .trellis/workspace/codex/index.md .trellis/workspace/codex/journal-22.md | rtk rg -n '^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|provider_supported_mirror|route id|artifact name|common EmitC)'
```

Result: no production/source dependency was introduced. Matches are limited to
negative guardrails in the PRD, task status JSON, and evidence strings that
record provider-derived generated C++/metadata after route construction. The
task did not add a route-id, artifact-name, descriptor, source-front-door,
common EmitC, or exact-intrinsic authority path.

## Spec Update Judgment

No `.trellis/spec/` update is needed. This round did not introduce a new
route contract, command signature, payload field, validation rule, or error
behavior. It exercised an already documented Stage2 computed-mask standalone
reduction path on real RVV hardware and recorded executable evidence.
