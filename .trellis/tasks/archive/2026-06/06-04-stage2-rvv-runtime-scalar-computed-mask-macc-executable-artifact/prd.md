# Stage2 RVV runtime-scalar computed-mask MAcc executable artifact

## Goal

Close one representative Stage2 RVV runtime-scalar computed-mask
multiply-accumulate executable-artifact gap. The selected path is
`runtime_scalar_cmp_masked_macc_add` unless live inspection exposes a tighter
blocker. The path must start from a selected `tcrv.exec` RVV variant with an
explicit typed `tcrv_rvv` runtime-scalar computed-mask MAcc body, flow through
RVV plugin-owned selected-body realization and provider route construction,
materialize through neutral common EmitC, generate a target artifact/header
plus external ABI harness, and compile/run on real `ssh rvv` with correctness
evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar computed-mask MAcc route to executable artifact`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: no short status entries were reported.
* Initial `rtk git log --oneline -8` starts at
  `a7a77419 rvv: record computed mask reduction ssh evidence`.
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
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires computed-mask MAcc
  and runtime-scalar computed-mask MAcc selected-body routes to remain RVV
  plugin-owned. Provider plans, statement plans, operand bindings, scalar
  splat/compare facts, accumulator facts, and route-family validation must not
  be inferred from route ids, artifact names, ABI strings, descriptors, common
  EmitC, or legacy i32 helper residue.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC/export
  to materialize provider-built routes neutrally. Provider operand binding,
  header/type summaries, runtime AVL/VL facts, and route-family metadata
  mirrors must match provider facts before target artifact acceptance.
* `.trellis/spec/testing/mlir-testing-contract.md` has a dedicated
  `Computed-Mask MAcc Generated-Bundle Evidence` contract. Runtime-scalar
  computed-mask MAcc evidence must check the `rhs_scalar` ABI role as both
  scalar splat source and compare RHS, exercise at least two runtime scalar
  values and two data patterns, distinguish `acc + lhs * rhs` from add-only or
  mul-only behavior, verify inactive accumulator/pass-through and tail sentinel
  preservation, and use real `ssh rvv` for runtime correctness claims.
* Archived reduction executable task
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-computed-mask-standalone-reduction-executable-artifact/`
  shows the expected pattern for this round: if production route/emission/ABI
  already works, record focused executable evidence honestly instead of making
  unrelated production changes.
* Live repository inspection shows an existing pre-realized fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`.
  It realizes into setvl/with_vl/load/splat/load/load/load/compare/
  masked_macc/store and already has stale-fact negative checks for provider
  mirror, binding plan, binding summary, runtime ABI order, predicate, header,
  type mapping, scalar producer, mask source/form, inactive-lane contract,
  passthrough layout, accumulator layout, result layout, and source memory.
* `scripts/rvv_generated_bundle_abi_e2e.py` already exposes
  `--op-kind runtime_scalar_cmp_masked_macc_add`,
  `--pre-realized-selected-body`, repeated `--runtime-count`, and repeated
  `--rhs-scalar` arguments.

## Requirements

* Keep the owner bounded to one representative runtime-scalar computed-mask
  MAcc executable path.
* Prefer existing `runtime_scalar_cmp_masked_macc_add`, because it exercises
  runtime `n`, runtime scalar threshold/splat, compare-produced mask, payload
  `lhs`/`rhs`, accumulator input, inactive-lane passthrough, masked MAcc,
  output store, setvl/VL, and generated external ABI harness.
* Generate the artifact through production compiler/tooling, not by
  hand-writing C, descriptors, source-front-door routes, route-id-derived
  shims, or script-side compiler-core substitutes.
* The generated artifact evidence must preserve compare-produced mask facts,
  runtime scalar threshold/splat facts, MAcc kind, operand binding order,
  accumulator layout, inactive-lane passthrough, result layout, setvl/VL,
  runtime `n`, and route-local mirror boundaries.
* Unsupported or stale non-MAcc provider/candidate facts must fail closed
  before they become artifact authority.
* Non-dry-run evidence must compile and run on real `ssh rvv`; if remote
  compile/run fails, record the exact blocker and do not claim runtime
  correctness.
* If route construction, ABI binding, EmitC emission, target validation,
  harness generation, or runtime oracle blocks real execution, repair the
  production/tooling boundary that owns the blocker.
* If existing production code already works, keep the final diff honest as
  executable evidence/test/task plumbing only and record the
  no-production-change rationale.

## Acceptance Criteria

* [x] A representative pre-realized typed runtime-scalar computed-mask MAcc
      selected-body case is chosen and documented.
* [x] The selected path generates a target artifact/header and external ABI
      harness through `rvv_generated_bundle_abi_e2e.py` without `--dry-run`.
* [x] Evidence records selected-body materialization through provider-owned
      realization/route metadata, runtime AVL/VL boundary, runtime scalar
      threshold/splat facts, compare-produced mask facts, payload loads,
      accumulator passthrough, masked MAcc result, generated object/header
      paths, and harness path.
* [x] The generated harness compiles and runs on `ssh rvv`; the remote output
      includes the expected pass marker and
      `PASS op=runtime_scalar_cmp_masked_macc_add`.
* [x] The correctness oracle covers runtime counts including zero, one, at
      least one multi-lane/multi-VL count, mask-selected lanes, mask-inactive
      lanes, accumulator input use, scalar RHS values, and tail preservation.
* [x] Runtime-scalar evidence exercises at least two `rhs_scalar` values and at
      least two data patterns.
* [x] If production/tooling code changes, focused checks cover the changed
      owner: generated-bundle lit/export checks plus
      `tianchenrv-target-artifact-export-test` and/or
      `tianchenrv-rvv-extension-plugin-test` as applicable.
* [x] If no production code changes, focused dry-run/lit and the non-dry-run
      `ssh rvv` evidence are sufficient, and the no-production-change
      rationale is recorded.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      route-id/artifact-name authority, common EmitC semantic inference, exact
      `__riscv_*_i32m1`, or mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task context is valid and task status is truthful.
* [x] A coherent commit is created if the task completes.

## Technical Approach

1. Inspect the runtime-scalar computed-mask MAcc specs, existing pre-realized
   fixture, route provider/planning/validation code, and generated-bundle
   harness support for `runtime_scalar_cmp_masked_macc_add`.
2. Run the existing dry-run/generated-bundle path to confirm the route emits
   the expected artifact, statement plan, harness, scalar-loop/data-pattern
   coverage, and evidence facts.
3. Run a focused non-dry-run generated-bundle command for
   `runtime_scalar_cmp_masked_macc_add` on `ssh rvv` with counts
   `0,1,7,16,23,257` and scalar values `-3,5`.
4. Inspect generated `evidence.json`, per-op `evidence.json`,
   `materialized_selected_body.mlir`, `materialized_rvv_emitc.cpp`, generated
   bundle header/object, generated harness, `remote_compile_stdout.txt`, and
   `remote_run_stdout.txt`.
5. If the run fails, classify the blocker as remote environment, generated
   C/C++ emission, target validation/bundle, ABI/harness integration,
   correctness oracle, or selected realization/provider route, then repair the
   owning production/tooling code.
6. Add or update focused lit/test coverage only if the fix changes observable
   behavior or if the repository lacks a stable way to preserve the executable
   evidence command.
7. Run focused checks, bounded old-authority scan, Trellis validation,
   finish/archive, and commit.

## Evidence Plan

Primary executable evidence command:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact --run-id runtime-scalar-cmp-masked-macc-add-ssh-rvv --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -3 --rhs-scalar 5 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Focused dry-run/lit regression if no production code changes:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-dry-run --run-id runtime-scalar-cmp-masked-macc-add-dry-run --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -3 --rhs-scalar 5 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir
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
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact
```

## Out Of Scope

* All MAcc variants, widening dot-product/contraction batches, dtype/LMUL
  expansion beyond what the chosen existing path already requires.
* High-level Linalg/Vector/StableHLO frontend lowering.
* Source-front-door positive routes.
* One-intrinsic wrapper dialects.
* Unrelated memory/conversion/reduction work.
* Broad smoke matrices, dashboards, or report-only artifacts.
* Inferring MAcc semantics from route ids, artifact names, test names,
  manifests, descriptors, C strings, ABI names, script constants, or mirror
  metadata.
* Moving RVV semantics into common EmitC/export.

## Definition Of Done

One representative pre-realized typed runtime-scalar computed-mask MAcc
selected-body path has a generated artifact, generated harness, real `ssh rvv`
compile/run transcript, and correctness result. Any production blocker found on
that path is fixed in the owning code; otherwise the task records that no
production code change was needed because the existing route/emission/harness
path already executed on real RVV hardware.

## Implementation Results

Chosen representative submodule:

* `runtime_scalar_cmp_masked_macc_add`
* input fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
* selected variant: `rvv_pr_rt_scalar_masked_macc`
* external ABI function:
  `tcrv_emitc_pr_rt_scalar_masked_macc_kernel_rvv_pr_rt_scalar_masked_macc`

No production code changes were required. Live execution showed the existing
production path already performs:

```text
selected tcrv.exec RVV variant
  -> typed runtime-scalar computed-mask MAcc pre-realized tcrv_rvv body
  -> RVV plugin selected-body realization
  -> realized typed tcrv_rvv setvl/with_vl/load/splat/load/load/load/compare/masked_macc/store body
  -> RVV-owned computed-mask accumulation MAcc provider plan and operand binding
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact bundle export
  -> generated external C ABI harness
  -> ssh rvv compile/run
```

The inspected MAcc owner/provider code confirms the route is guarded by
RVV-owned facts and fail-closed checks:

* `RVVEmitCMAccRouteFamilyPlanOwners.h` exposes the MAcc route-family owner
  boundary, runtime ABI order, route operand binding, provider-plan verifier,
  and pre-realized runtime-scalar computed-mask MAcc validator.
* `RVVEmitCMAccRouteFamilyPlanOwners.cpp` validates
  `runtime_scalar_cmp_masked_macc_add`, `sle`,
  `runtime-scalar-computed-mask-unit-stride-macc`, mask source/form,
  accumulator role/layout, result layout, SEW32 LMUL m1/m2, policy, and runtime
  ABI roles before realization/provider facts.
* The same owner derives the `rhs_scalar` binding as
  `abi|splat|cmp-rhs|hdr`, and verifies the runtime ABI order
  `cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n`.
* `RVVEmitCRouteProvider.h` has a dedicated
  `RVVRuntimeScalarComputedMaskMAccRouteFacts` and MAcc validation contract.
* `RVVTargetArtifactRouteFamilyValidation.cpp` validates the MAcc description
  against provider-owned contract fields, including runtime AVL/VL contract,
  route operand binding, runtime ABI parameters, predicate, mask role/source,
  inactive-lane contract, passthrough layout, source/destination memory, and
  accumulator/result layout.

The task changes are limited to Trellis task context and PRD evidence records.
Generated evidence under `artifacts/tmp/` is intentionally not tracked by git;
it remains in the workspace as the runtime transcript/artifact output for this
round.

## Evidence Results

### Dry-Run Generated Bundle

Command:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-dry-run --run-id runtime-scalar-cmp-masked-macc-add-dry-run --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -3 --rhs-scalar 5 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: dry_run_success
artifact_dir: artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-dry-run/runtime-scalar-cmp-masked-macc-add-dry-run
```

Dry-run evidence confirmed:

* `rhs_scalar` is recorded as `rhs-scalar-value`, realized through
  `tcrv_rvv.splat`, and emitted through `__riscv_vmv_v_x_i32m1`.
* compare uses `__riscv_vmsle_vv_i32m1_b32`.
* active MAcc uses `__riscv_vmacc_vv_i32m1`.
* inactive-lane passthrough uses `__riscv_vmerge_vvm_i32m1`.
* store uses the merged result and runtime loop VL.
* harness loops over counts `0,1,7,16,23,257`, rhs scalars `-3,5`, and
  patterns `0,1`.
* harness checks active/inactive lanes, inactive accumulator preservation,
  add-only/mul-only distinguishability, signed product lanes, and tail
  sentinel preservation.

Primary dry-run paths:

* Root evidence:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-dry-run/runtime-scalar-cmp-masked-macc-add-dry-run/evidence.json`
* Per-op evidence:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-dry-run/runtime-scalar-cmp-masked-macc-add-dry-run/runtime_scalar_cmp_masked_macc_add/evidence.json`
* Materialized selected body:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-dry-run/runtime-scalar-cmp-masked-macc-add-dry-run/runtime_scalar_cmp_masked_macc_add/materialized_selected_body.mlir`
* Generated EmitC C++:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-dry-run/runtime-scalar-cmp-masked-macc-add-dry-run/runtime_scalar_cmp_masked_macc_add/materialized_rvv_emitc.cpp`
* Generated harness:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-dry-run/runtime-scalar-cmp-masked-macc-add-dry-run/runtime_scalar_cmp_masked_macc_add/rvv_generated_bundle_abi_runtime_scalar_cmp_masked_macc_add_harness.c`

### Executable ssh rvv Evidence

Command:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact --run-id runtime-scalar-cmp-masked-macc-add-ssh-rvv --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -3 --rhs-scalar 5 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
artifact_dir: artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact/runtime-scalar-cmp-masked-macc-add-ssh-rvv
tcrv_rvv_generated_bundle_abi_runtime_scalar_cmp_masked_macc_add_ok counts=0,1,7,16,23,257 rhs_scalars=-3,5 patterns=0,1
PASS op=runtime_scalar_cmp_masked_macc_add counts=0,1,7,16,23,257 rhs_scalars=-3,5 patterns=0,1
```

The remote compile summary recorded:

```text
remote_arch=riscv64
clang_path=/usr/bin/clang
clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)
```

The remote runtime transcript covers 24 cases: six runtime counts, two
`rhs_scalar` values, and two data patterns. Representative multi-lane results:

```text
runtime_scalar_cmp_masked_macc_add case n=7 rhs_scalar=-3 pattern=0 ok runtime_scalar_computed_mask_macc active_lanes=4 inactive_lanes=3 inactive_acc_preserved=3 add_only_distinguishing=4 mul_only_distinguishing=4 tail_preserved
runtime_scalar_cmp_masked_macc_add case n=23 rhs_scalar=5 pattern=0 ok runtime_scalar_computed_mask_macc active_lanes=15 inactive_lanes=8 inactive_acc_preserved=8 add_only_distinguishing=15 mul_only_distinguishing=15 tail_preserved
runtime_scalar_cmp_masked_macc_add case n=257 rhs_scalar=5 pattern=1 ok runtime_scalar_computed_mask_macc active_lanes=129 inactive_lanes=128 inactive_acc_preserved=128 add_only_distinguishing=129 mul_only_distinguishing=129 tail_preserved
```

Primary executable evidence paths:

* Root evidence:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact/runtime-scalar-cmp-masked-macc-add-ssh-rvv/evidence.json`
* Per-op evidence:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact/runtime-scalar-cmp-masked-macc-add-ssh-rvv/runtime_scalar_cmp_masked_macc_add/evidence.json`
* Materialized selected body:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact/runtime-scalar-cmp-masked-macc-add-ssh-rvv/runtime_scalar_cmp_masked_macc_add/materialized_selected_body.mlir`
* Generated EmitC C++:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact/runtime-scalar-cmp-masked-macc-add-ssh-rvv/runtime_scalar_cmp_masked_macc_add/materialized_rvv_emitc.cpp`
* Generated harness:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact/runtime-scalar-cmp-masked-macc-add-ssh-rvv/runtime_scalar_cmp_masked_macc_add/rvv_generated_bundle_abi_runtime_scalar_cmp_masked_macc_add_harness.c`
* Remote compile stdout:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact/runtime-scalar-cmp-masked-macc-add-ssh-rvv/runtime_scalar_cmp_masked_macc_add/remote_compile_stdout.txt`
* Remote run stdout:
  `artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact/runtime-scalar-cmp-masked-macc-add-ssh-rvv/runtime_scalar_cmp_masked_macc_add/remote_run_stdout.txt`

## Checks And Self-Repair

Checks run:

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-dry-run --run-id runtime-scalar-cmp-masked-macc-add-dry-run --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -3 --rhs-scalar 5 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv /home/kingdom/phdworks/TianchenRV/build/test --filter pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact --run-id runtime-scalar-cmp-masked-macc-add-ssh-rvv --overwrite --op-kind runtime_scalar_cmp_masked_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -3 --rhs-scalar 5 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Focused lit result:

```text
Total Discovered Tests: 477
  Excluded: 475 (99.58%)
  Passed  :   2 (0.42%)
```

Final diff check:

```text
rtk git diff --check
```

Result: passed with no output.

Bounded old-authority scan:

```text
rtk rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|__riscv_.*_i32m1|descriptor|direct-C|source-export|source-front-door|route-id/artifact-name|common EmitC semantic inference|mirror-only authority" .trellis/tasks/06-04-stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact
rtk rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|__riscv_.*_i32m1" include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir scripts/rvv_generated_bundle_abi_e2e.py
```

Result:

* Touched task files contain the old-authority strings only in negative
  constraints or evidence descriptions, not as executable route authority.
* No production files changed in this round.
* The bounded production scan over the inspected owner/provider/target/script
  surface showed existing exact intrinsic strings in evidence tooling and an
  existing fail-closed `tcrv_rvv.i32_` rejection check in
  `RVVEmitCRoutePlanning.cpp`; this round did not add or promote any of those
  as route authority.

Self-repair:

* Initial direct system-lit invocation against the source test path failed
  because `test/lit.cfg.py` expects `tianchenrv_obj_root` from the generated
  site config.
* A second invocation against `build/test` from the repository root failed
  because `build/test/lit.site.cfg.py` loads `../../test/lit.cfg.py` relative
  to its working directory.
* The corrected invocation ran from `build/test`, matching the Ninja command
  shape, and passed the two focused tests.

Because no provider, target, plugin, or tooling code changed,
`tianchenrv-target-artifact-export-test` and
`tianchenrv-rvv-extension-plugin-test` were not required for this round.

## Spec Update Decision

No `.trellis/spec/` update was made. The relevant executable contract already
exists in `.trellis/spec/testing/mlir-testing-contract.md` under
`Computed-Mask MAcc Generated-Bundle Evidence`, and the RVV plugin/provider and
EmitC route ownership rules already cover the checked runtime-scalar
computed-mask MAcc path. This round added executable evidence for an existing
contract rather than changing a command/API, provider payload shape, validator
contract, or project convention.
