# Stage2 RVV computed-mask widening dot-reduction executable artifact

## Goal

Close one representative Stage2 RVV computed-mask widening dot-reduction
executable-artifact gap. The selected path is
`computed_masked_widening_dot_reduce_add` from the existing pre-realized
selected-body fixture unless live execution exposes a tighter blocker. The path
must start from a selected `tcrv.exec` RVV variant with an explicit typed
`tcrv_rvv` computed-mask widening dot-reduction pre-realized body, flow through
RVV plugin-owned realization and provider route construction, materialize
through neutral common EmitC, generate a target artifact/header plus external
ABI harness, and compile/run on real `ssh rvv` with correctness evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask widening dot-reduction executable artifact`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: no short status entries were reported.
* Initial `rtk git log --oneline -8` starts at
  `ff1148b6 rvv: record runtime scalar masked macc ssh evidence`.
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
* `.trellis/spec/lowering-runtime/emitc-route.md` requires widening
  dot-reduce provider facts and target validation contracts to be provider
  owned for plain, strided-input, computed-mask, and computed-mask-strided
  variants. Target validation must consume provider facts and the embedded
  runtime AVL/VL contract rather than reconstructing truth from target-local
  constants, route names, artifact metadata, fixture names, descriptors,
  scripts, C strings, or intrinsic spellings.
* `.trellis/spec/extension-plugins/rvv-plugin.md` defines the math
  operand-binding facts boundary and the direct contraction route-provider
  owner boundary. The active direct-provider contraction routes include
  `computed_masked_widening_dot_reduce_add`; they must be selected exactly once
  by an RVV plugin-owned direct contraction owner after family-plan,
  materialization, math operand-binding, and route-control facts are validated.
* `.trellis/spec/testing/mlir-testing-contract.md` has a dedicated
  `computed_masked_widening_dot_reduce_boundary` generated-bundle evidence
  contract. Runtime claims must include real `ssh rvv` output across runtime
  counts including zero, one, exact-VL, tail, and stress cases. The harness must
  distinguish signed computed-mask widening horizontal dot-reduction from
  add-only, mul-only, unmasked, non-widening, missing-seed, wrong
  sign-extension, scalar-output overwrite, inactive-lane contribution, and tail
  overwrite behavior.
* The archived MAcc executable task
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-runtime-scalar-computed-mask-macc-executable-artifact/`
  shows the expected pattern for this round: if production
  route/emission/ABI already works, record executable compiler-path evidence
  honestly instead of making unrelated production changes.
* Live repository inspection shows the preferred pre-realized input fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`.
  It starts from runtime ABI values `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n` and a
  `tcrv_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body` with
  `source_sew=16`, `source_lmul=mf2`, accumulator/result SEW32 LMUL m1,
  `predicate_kind=slt`, `mask_source=compare-produced-mask-same-vl-scope`,
  `mask_memory_form=compare-produced-mask`, scalar i32 accumulator seed/result
  layout, and the signed i16-by-i16 to i32 dot relation.
* The same fixture FileChecks realized `setvl`, `with_vl`, compare-vector
  loads, i16 dot source loads, `tcrv_rvv.compare`, and
  `tcrv_rvv.masked_widening_dot_reduce`, then checks emission-plan/header
  mirrors for runtime ABI order, binding plan, source/accumulator/result
  SEW/LMUL, mask facts, widening product intrinsics, reduction store VL, and
  external C ABI prototype.
* `scripts/rvv_generated_bundle_abi_e2e.py` already exposes
  `--op-kind computed_masked_widening_dot_reduce_add`,
  `--pre-realized-selected-body`, repeated `--runtime-count`, and a
  `computed_masked_widening_dot_reduce_boundary` summary. The script evidence
  records provider route facts, target-validator consumed facts, materialized
  body checks, emitted C++ checks, route metadata mirrors, generated header and
  object paths, and generated harness path.
* The generated unit-stride computed-mask widening dot harness currently checks
  `out[0] == acc[0] + sum_i(cmp_lhs[i] < cmp_rhs[i] ? lhs[i] * rhs[i] : 0)`,
  active and inactive lanes, positive and negative active products, nonzero
  inactive products, nonzero seed, scalar-only output, and tail/non-scalar
  sentinel preservation.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes
  `RVVWideningDotReduceRouteFacts` and
  `RVVWideningDotReduceRouteValidationContract` with provider-owned route
  facts, runtime ABI parameters, mask facts, source/result type facts,
  statement-plan names, and an embedded `RVVRuntimeAVLVLSelectedBoundaryContract`.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` already requires
  same-analysis contraction family plan, route-control provider plan,
  materialization facts, math operand-binding facts, compare/dot/acc/out/n ABI
  operands, setvl/load/compare/masked-product/merge/seed/reduction/store
  leaves before direct contraction route construction.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` already consumes
  `getRVVWideningDotReduceRouteValidationContract(description)` before route
  payload acceptance and validates candidate mirrors for binding plan, provider
  support mirror, contraction family plan, memory form, runtime control, ABI
  order, headers, C type mappings, target profile, source/accumulator/result
  SEW/LMUL, dot relation/layouts, scalar store VL, mask role/source/form,
  inactive-lane zeroing, predicate, and masked widening product facts.

## Requirements

* Keep the owner bounded to one representative unit-stride computed-mask
  widening dot-reduction executable path.
* Prefer existing `computed_masked_widening_dot_reduce_add`, because it
  exercises runtime `n`, compare-produced mask, i16 dot source vectors, i32
  accumulator seed/result, signed widening dot relation, inactive-lane zeroing
  before reduction, scalar result layout, setvl/VL, generated target artifact,
  and external C ABI harness.
* Generate the artifact through production compiler/tooling, not by
  hand-writing C, descriptors, source-front-door routes, route-id-derived
  shims, script-side compiler-core substitutes, or common EmitC semantic
  inference.
* The generated artifact evidence must preserve compare mask provenance,
  source SEW/LMUL, accumulator/result SEW/LMUL, dot-product relation,
  accumulator seed layout, scalar result layout, setvl/VL, runtime `n`, and
  route-local mirror boundaries.
* Unsupported or stale non-widening-dot provider/candidate facts must fail
  closed before they become artifact authority.
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

* [x] A representative pre-realized typed computed-mask widening
      dot-reduction selected-body case is chosen and documented.
* [x] The selected path generates a target artifact/header and external ABI
      harness through `rvv_generated_bundle_abi_e2e.py` without `--dry-run`.
* [x] Evidence records selected-body realization, provider route facts, target
      validator consumption, runtime AVL/VL boundary, compare-produced mask
      facts, i16 source vector facts, i32 accumulator/result facts, widening
      dot relation, scalar seed/result layout, generated object/header paths,
      and harness path.
* [x] The generated harness compiles and runs on `ssh rvv`; remote output
      includes the expected pass marker and
      `PASS op=computed_masked_widening_dot_reduce_add`.
* [x] Correctness evidence covers runtime counts including zero, one, at least
      one exact-VL or multi-lane count, tail count, and stress count; it checks
      active mask lanes, inactive mask lanes, positive products, negative
      products, nonzero inactive products, nonzero accumulator seed, scalar
      output, and tail/non-scalar sentinel preservation.
* [x] If production/tooling code changes, focused checks cover the changed
      owner: generated-bundle lit/export checks plus
      `tianchenrv-target-artifact-export-test` and/or
      `tianchenrv-rvv-extension-plugin-test` as applicable.
* [x] If no production code changes are needed, focused dry-run/lit and the
      non-dry-run `ssh rvv` evidence are sufficient, and the
      no-production-change rationale is recorded.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      route-id/artifact-name authority, common EmitC semantic inference, exact
      `__riscv_*_i32m1`, or mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task context is valid and task status is truthful.
* [x] A coherent commit is created if the task completes.

## Technical Approach

1. Inspect the computed-mask widening dot-reduction specs, existing
   pre-realized fixture, route planning/provider/target validation code, and
   generated-bundle harness support for
   `computed_masked_widening_dot_reduce_add`.
2. Run the existing dry-run/generated-bundle path to confirm the route emits
   the expected artifact facts, materialized body, emitted C++ checks, evidence
   boundary, statement plan, harness, and runtime count contract.
3. Run a focused non-dry-run generated-bundle command for
   `computed_masked_widening_dot_reduce_add` on `ssh rvv` with counts
   `0,1,7,16,23,257`.
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
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact --run-id computed-masked-widening-dot-reduce-add-ssh-rvv --overwrite --op-kind computed_masked_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Focused dry-run/lit regression if no production code changes:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-dry-run --run-id computed-masked-widening-dot-reduce-add-dry-run --overwrite --op-kind computed_masked_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir
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
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact
```

## Out Of Scope

* Broadening into all widening dot variants.
* Strided-input widening dot batches.
* Dtype/LMUL clone batches.
* High-level Linalg/Vector/StableHLO frontend lowering.
* Source-front-door positive routes.
* One-intrinsic wrapper dialects.
* Unrelated memory, conversion, reduction, or MAcc work.
* Dashboards, reports, or broad smoke matrices.
* Inferring widening-dot semantics from route ids, artifact names, test names,
  manifests, descriptors, C strings, ABI names, script constants, intrinsic
  spellings, or mirror metadata.
* Moving RVV semantics into common EmitC/export.

## Definition Of Done

One representative pre-realized typed computed-mask widening dot-reduction
selected-body path has a generated artifact, generated harness, real `ssh rvv`
compile/run transcript, and correctness result. Any production blocker found on
that path is fixed in the owning code; otherwise the task records that no
production code change was needed because the existing route/emission/harness
path already executed on real RVV hardware.

## Implementation Results

Chosen representative submodule:

* `computed_masked_widening_dot_reduce_add`
* input fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`
* selected variant:
  `pre_realized_body_rvv_masked_widening_dot_reduce_add`
* external ABI function:
  `tcrv_emitc_pre_realized_body_masked_widening_dot_reduce_add_kernel_pre_realized_body_rvv_masked_widening_dot_reduce_add`

No production code changes were required. Live execution showed the existing
production path already performs:

```text
selected tcrv.exec RVV variant
  -> typed computed-mask widening dot-reduction pre-realized tcrv_rvv body
  -> RVV plugin selected-body realization
  -> realized typed tcrv_rvv setvl/with_vl/load/compare/masked_widening_dot_reduce/store body
  -> RVV-owned contraction family plan, math operand binding, route-control plan, and direct contraction owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact bundle export
  -> generated external C ABI harness
  -> ssh rvv compile/run
```

The inspected owner/provider/target surfaces confirm that route support is
guarded by RVV-owned facts and fail-closed checks:

* `RVVWideningDotReduceRouteFacts` and
  `RVVWideningDotReduceRouteValidationContract` carry source i16mf2,
  accumulator/result i32m1, runtime ABI order, mask role/source/form,
  binding plan, target leaf profile, header/type summaries, statement-plan
  facts, and the embedded runtime AVL/VL boundary.
* `getRVVSelectedBodyDirectContractionRouteProviderPlan` requires the same
  analysis contraction family plan, route-control provider plan,
  materialization facts, math operand-binding facts, compare/dot/acc/out/n ABI
  operands, and setvl/load/compare/masked-product/merge/seed/reduction/store
  leaves before route construction.
* `RVVTargetArtifactRouteFamilyValidation.cpp` consumes
  `getRVVWideningDotReduceRouteValidationContract(description)` before target
  artifact acceptance and checks route payloads, ABI mappings, statement plan,
  headers/type mappings, provider mirrors, mask facts, source/result facts, and
  stale non-family residues against that contract.

## Evidence Artifacts

Primary non-dry-run artifact root:

```text
artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv
```

Generated per-op evidence:

```text
artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv/computed_masked_widening_dot_reduce_add/evidence.json
```

Generated materialized body and emitted C++:

```text
artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv/computed_masked_widening_dot_reduce_add/materialized_selected_body.mlir
artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv/computed_masked_widening_dot_reduce_add/materialized_rvv_emitc.cpp
```

Generated bundle and harness:

```text
artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv/computed_masked_widening_dot_reduce_add/generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o
artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv/computed_masked_widening_dot_reduce_add/generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h
artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv/computed_masked_widening_dot_reduce_add/rvv_generated_bundle_abi_computed_masked_widening_dot_reduce_add_harness.c
```

Remote compile and run transcript:

```text
artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv/computed_masked_widening_dot_reduce_add/remote_compile_stdout.txt
artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv/computed_masked_widening_dot_reduce_add/remote_run_stdout.txt
```

Remote compile summary:

```text
remote_arch=riscv64
clang_path=/usr/bin/clang
clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)
```

Remote correctness output:

```text
computed_masked_widening_dot_reduce_add case n=0 ok compare_masked_signed_horizontal_dot seed_added inactive_lanes_skipped scalar_output tail_preserved
computed_masked_widening_dot_reduce_add case n=1 ok compare_masked_signed_horizontal_dot seed_added inactive_lanes_skipped scalar_output tail_preserved
computed_masked_widening_dot_reduce_add case n=7 ok compare_masked_signed_horizontal_dot seed_added inactive_lanes_skipped scalar_output tail_preserved
computed_masked_widening_dot_reduce_add case n=16 ok compare_masked_signed_horizontal_dot seed_added inactive_lanes_skipped scalar_output tail_preserved
computed_masked_widening_dot_reduce_add case n=23 ok compare_masked_signed_horizontal_dot seed_added inactive_lanes_skipped scalar_output tail_preserved
computed_masked_widening_dot_reduce_add case n=257 ok compare_masked_signed_horizontal_dot seed_added inactive_lanes_skipped scalar_output tail_preserved
tcrv_rvv_generated_bundle_abi_computed_masked_widening_dot_reduce_add_ok counts=0,1,7,16,23,257
PASS op=computed_masked_widening_dot_reduce_add counts=0,1,7,16,23,257
```

Evidence JSON summary:

* `status`: `success`
* `dry_run`: `false`
* `computed_masked_widening_dot_reduce_boundary.authority`:
  `provider-derived typed tcrv_rvv computed-mask widening dot-reduce body/config/runtime facts`
* `computed_masked_widening_dot_reduce_boundary.direct_pre_realized_route_entry_supported`:
  `false`
* `selected_source_abi`: `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, `acc`, `out`,
  `n`
* provider facts include `compare_predicate_kind=slt`,
  `route_operand_binding_plan=rvv-route-operand-binding:masked_widening_dot_reduce.v1`,
  source `i16/mf2`, accumulator/result `i32/m1`, compare-produced mask facts,
  inactive-lane zeroing, signed widening dot relation, scalar seed layout, and
  scalar result layout.

## Checks And Self-Repair

Checks run:

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-dry-run --run-id computed-masked-widening-dot-reduce-add-dry-run --overwrite --op-kind computed_masked_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact --run-id computed-masked-widening-dot-reduce-add-ssh-rvv --overwrite --op-kind computed_masked_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
rtk bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add'
rtk git diff --check
```

Focused lit result:

```text
Total Discovered Tests: 477
  Excluded: 476 (99.79%)
  Passed  :   1 (0.21%)
```

Initial direct lit attempt:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir
```

Result: failed before running the target test because `test/lit.cfg.py` expects
`tianchenrv_obj_root` from the generated site config. Self-repair used the
`build/test` working directory and the generated `lit.site.cfg.py` entry,
matching the Ninja command shape.

Bounded old-authority scan:

```text
rtk rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|__riscv_.*_i32m1|descriptor|direct-C|source-export|source-front-door|route-id/artifact-name|common EmitC semantic inference|mirror-only authority" .trellis/tasks/06-04-06-04-stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact
rtk rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|__riscv_.*_i32m1" include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir scripts/rvv_generated_bundle_abi_e2e.py
```

Result:

* Touched task files contain old-authority strings only in negative
  constraints and evidence descriptions.
* No provider, target, plugin, fixture, or script source file changed in this
  round.
* The bounded production scan showed existing exact intrinsic strings in the
  fixture/script evidence tooling and an existing fail-closed `tcrv_rvv.i32_`
  rejection check in `RVVEmitCRoutePlanning.cpp`; this round did not add or
  promote any of those as route authority.

Because no provider, target, plugin, or tooling code changed,
`tianchenrv-target-artifact-export-test` and
`tianchenrv-rvv-extension-plugin-test` were not rerun. The already-passing
dry-run, target fixture lit check, non-dry-run `ssh rvv` compile/run, task
validation, old-authority scan, and diff check cover the changed behavior for
this evidence-only closure.

## Spec Update Review

No `.trellis/spec/` update was made. This task did not introduce a new
compiler contract, API signature, validation matrix, or cross-layer behavior.
The computed-mask widening dot-reduction provider facts, target validation
contract, generated-bundle evidence boundary, and `ssh rvv` evidence
requirements were already present in the relevant specs before this round.
