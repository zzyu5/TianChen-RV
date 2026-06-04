# Stage2 RVV strided-input widening dot-reduction executable artifact

## Goal

Close one representative Stage2 RVV computed-mask strided-input widening
dot-reduction executable-artifact gap. The selected path is
`computed_masked_strided_input_widening_dot_reduce_add`, because live
inspection confirms the repository has a pre-realized selected-body fixture,
RVV route/provider support, target artifact validation surfaces, generated
bundle tooling, and a direct continuation from the previously closed
unit-stride computed-mask widening-dot evidence.

The path must start from a selected `tcrv.exec` RVV variant with explicit typed
`tcrv_rvv` computed-mask strided-input widening-dot body facts, flow through
RVV plugin-owned selected-body realization and provider route construction,
materialize through neutral common EmitC, generate a target artifact/header
plus external C ABI harness, and compile/run on real `ssh rvv` with focused
correctness evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV strided-input widening dot-reduction executable artifact`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: no short status entries were reported.
* Initial `rtk git log --oneline -8` starts at
  `f3b16012 rvv: record computed mask widening dot ssh evidence`.
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
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md` requires
  Stage2 RVV selected-body realization to consume code-affecting body/config/
  runtime facts before route construction; selected-path metadata is diagnostic
  mirror only.
* `.trellis/spec/extension-plugins/rvv-plugin.md` defines the active direct
  contraction provider routes as `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`. They must be
  selected exactly once by the RVV plugin-owned direct contraction owner after
  family-plan, materialization, math operand-binding, and route-control facts
  are validated.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires widening dot-reduce
  target artifact validation to consume
  `getRVVWideningDotReduceRouteValidationContract(description)` for plain,
  strided-input, computed-mask, and computed-mask-strided variants. Target
  validation must not reconstruct route facts from route names, artifact
  metadata, fixture names, descriptors, scripts, C strings, or intrinsic
  spellings.
* `.trellis/spec/testing/mlir-testing-contract.md` requires computed-mask
  widening-dot generated-bundle evidence for
  `computed_masked_strided_input_widening_dot_reduce_add`: selected ABI must
  include `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, `acc`, `out`, `n`,
  `lhs_stride`, and `rhs_stride`; strided evidence must mirror strided memory
  form/layout, stride ABI sources, and the strided load intrinsic; runtime
  evidence must include multiple runtime counts and either multiple stride
  patterns or an equivalent oracle proving runtime stride use.
* The archived computed-mask unit-stride executable task
  `.trellis/tasks/archive/2026-06/06-04-06-04-stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/`
  shows the expected closure pattern: if the production provider/target/export
  path already works, keep this task honest as executable compiler-path
  evidence instead of making unrelated code changes.
* The relevant workspace journal entry records that
  `computed_masked_widening_dot_reduce_add` already generated an artifact and
  passed real `ssh rvv` correctness for counts `0,1,7,16,23,257` without
  production code changes.
* Live repository inspection confirms the preferred input fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`.
  It starts from runtime ABI values `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`,
  `acc`, `out`, `n`, `lhs_stride`, and `rhs_stride`, and contains
  `tcrv_rvv.typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body`
  with source SEW16 LMUL mf2, accumulator/result SEW32 LMUL m1,
  predicate `slt`, compare-produced mask provenance, element stride unit,
  scalar i32 seed/result layouts, and signed i16-by-i16 to i32 dot relation.
* The same fixture FileChecks selected-body realization into
  `tcrv_rvv.setvl`, `with_vl`, compare vector loads, `tcrv_rvv.strided_load`
  for both i16 dot sources, `tcrv_rvv.compare`,
  `tcrv_rvv.masked_widening_dot_reduce`, and store; it explicitly rejects an
  i16 unit `tcrv_rvv.load` on the dot sources.
* The fixture also checks emission-plan/header mirrors for runtime ABI order,
  binding plan, source/accumulator/result SEW/LMUL, compare mask facts,
  strided memory layout, `lhs_stride`/`rhs_stride` sources, source/destination
  memory forms, widening dot relation, strided load intrinsic, reduction store
  VL, and external C ABI prototype.
* `scripts/rvv_generated_bundle_abi_e2e.py` exposes
  `--op-kind computed_masked_strided_input_widening_dot_reduce_add` and the
  corresponding pre-realized selected-body expectation. It has generated
  bundle, evidence summary, harness, and runtime-check branches for the
  computed-mask strided widening-dot route.
* Live source inspection shows provider/target surfaces already reference
  `ComputedMaskStridedInputWideningDotReduceAdd`, strided input booleans,
  strided source load leaves, `RVVWideningDotReduceRouteValidationContract`,
  target validator consumption, and stale strided/unit/computed-mask residue
  rejection logic.

## Requirements

* Keep the owner bounded to one representative computed-mask strided-input
  widening dot-reduction executable path:
  `computed_masked_strided_input_widening_dot_reduce_add`.
* Generate the artifact through the production compiler/tooling path, not by
  hand-writing C, descriptors, source-front-door routes, route-id-derived
  shims, script-side compiler-core substitutes, or common EmitC semantic
  inference.
* Preserve selected `tcrv.exec` runtime `n` plus `lhs_stride`/`rhs_stride` ABI
  boundaries through typed body realization, provider route facts, target
  artifact validation, generated header, generated C/C++, and harness calls.
* Preserve compare mask provenance, i16 strided source addressing, source
  SEW/LMUL, accumulator/result SEW/LMUL, signed widening dot relation,
  accumulator seed layout, scalar result layout, setvl/VL, tail handling, and
  route-local mirror boundaries.
* Unsupported or stale non-strided/non-computed/non-widening-dot provider or
  candidate facts must fail closed before becoming artifact authority.
* Non-dry-run evidence must compile and run on real `ssh rvv`; if remote
  compile/run fails, record the exact blocker and do not claim runtime
  correctness.
* If route construction, ABI binding, EmitC emission, target validation,
  harness generation, or runtime oracle blocks real execution, repair the
  owning production/tooling code.
* If existing production code already works, keep the final diff honest as
  executable evidence/task plumbing only and record the no-production-change
  rationale.

## Acceptance Criteria

* [x] A representative pre-realized typed computed-mask strided-input widening
      dot-reduction selected-body case is chosen and documented.
* [x] The selected path generates a target artifact/header and external ABI
      harness through `rvv_generated_bundle_abi_e2e.py` without `--dry-run`.
* [x] Evidence records selected-body realization, provider route facts, target
      validator consumption, runtime AVL/VL boundary, compare-produced mask
      facts, i16 strided source vector facts, i32 accumulator/result facts,
      widening dot relation, scalar seed/result layout, generated object/header
      paths, harness path, `lhs_stride`, and `rhs_stride`.
* [x] Generated C/C++ and harness evidence prove strided source addressing via
      runtime stride ABI values, not unit-stride dot-source loads.
* [x] The generated harness compiles and runs on `ssh rvv`; remote output
      includes the expected pass marker and
      `PASS op=computed_masked_strided_input_widening_dot_reduce_add`.
* [x] Correctness evidence covers runtime counts including zero, one, at least
      one exact-VL or multi-lane count, a tail count, and a stress count; it
      checks active mask lanes, inactive mask lanes, positive products,
      negative products, nonzero inactive products, nonzero accumulator seed,
      scalar output, strided skipped-source behavior, and tail/non-scalar
      sentinel preservation.
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

1. Inspect the computed-mask strided-input widening dot-reduction specs,
   existing pre-realized fixture, route planning/provider/target validation
   code, and generated-bundle harness support for
   `computed_masked_strided_input_widening_dot_reduce_add`.
2. Run the existing dry-run/generated-bundle path to confirm the route emits
   expected artifact facts, materialized body, emitted C++ checks, evidence
   boundary, statement plan, harness, stride facts, and runtime count contract.
3. Run a focused non-dry-run generated-bundle command for
   `computed_masked_strided_input_widening_dot_reduce_add` on `ssh rvv` with
   counts `0,1,7,16,23,257`.
4. Inspect generated `evidence.json`, per-op `evidence.json`,
   `materialized_selected_body.mlir`, `materialized_rvv_emitc.cpp`, generated
   bundle header/object, generated harness, `remote_compile_stdout.txt`, and
   `remote_run_stdout.txt`.
5. If the run fails, classify the blocker as remote environment, generated
   C/C++ emission, target validation/bundle, ABI/harness integration,
   correctness oracle, or selected realization/provider route, then repair the
   owning production/tooling code.
6. Add or update focused lit/test coverage only if the fix changes observable
   behavior or if the repository lacks a stable way to preserve executable
   evidence.
7. Run focused checks, bounded old-authority scan, Trellis validation,
   finish/archive, and commit.

## Evidence Plan

Primary executable evidence command:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact --run-id computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv --overwrite --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Focused dry-run/lit regression if no production code changes:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-dry-run --run-id computed-masked-strided-input-widening-dot-reduce-add-dry-run --overwrite --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add'
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
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-strided-input-widening-dot-reduction-executable-artifact
```

## Out Of Scope

* Broadening into all widening dot variants.
* Dtype/LMUL clone batches.
* High-level Linalg/Vector/StableHLO frontend lowering.
* Source-front-door positive routes.
* One-intrinsic wrapper dialects.
* Unrelated memory, conversion, reduction, or MAcc work.
* Dashboards, reports, or broad smoke matrices.
* Inferring strided-input, computed-mask, or widening-dot semantics from route
  ids, artifact names, test names, manifests, descriptors, C strings, ABI
  names, script constants, intrinsic spellings, or mirror metadata.
* Moving RVV semantics into common EmitC/export.

## Definition Of Done

One representative pre-realized typed computed-mask strided-input widening
dot-reduction selected-body path has a generated artifact, generated harness,
real `ssh rvv` compile/run transcript, and correctness result. Any production
blocker found on that path is fixed in the owning code; otherwise the task
records that no production code change was needed because the existing
route/emission/harness path already executed on real RVV hardware.

## Implementation Results

Chosen representative submodule:

* `computed_masked_strided_input_widening_dot_reduce_add`
* input fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`
* selected variant:
  `rvv_computed_mask_strided_input_dot`
* external ABI function:
  `tcrv_emitc_pre_realized_masked_strided_dot_kernel_rvv_computed_mask_strided_input_dot`

No production code changes were required. Live execution showed the existing
production path already performs:

```text
selected tcrv.exec RVV variant
  -> typed computed-mask strided-input widening dot-reduction pre-realized tcrv_rvv body
  -> RVV plugin selected-body realization
  -> realized typed tcrv_rvv setvl/with_vl/load/strided_load/compare/masked_widening_dot_reduce/store body
  -> RVV-owned contraction family plan, math operand binding, route-control plan, and direct contraction owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact bundle export
  -> generated external C ABI harness
  -> ssh rvv compile/run
```

The generated materialized body contains `tcrv_rvv.strided_load` for both i16
dot sources with runtime `%lhs_stride` and `%rhs_stride`, then
`tcrv_rvv.masked_widening_dot_reduce` over the compare-produced mask. The
generated C++ uses `__riscv_vlse16_v_i16mf2` for both dot source loads,
`__riscv_vwmul_vv_i32m1_m` for the masked widening product,
`__riscv_vmerge_vvm_i32m1` to zero inactive lanes before reduction, and
`__riscv_vredsum_vs_i32m1_i32m1` for the horizontal reduction.

The inspected evidence confirms that route support is guarded by RVV-owned
facts and fail-closed target validation:

* `selected_source_abi` includes `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, `acc`,
  `out`, `n`, `lhs_stride`, and `rhs_stride`.
* `materialized_body.source_load_op` is `tcrv_rvv.strided_load` and
  `uses_strided_inputs` is `true`.
* `provider_route_facts.runtime_abi_order` is
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`.
* `strided_input_facts` carry `source_memory_form=strided-load`,
  `destination_memory_form=unit-stride-store`,
  `lhs_stride_source=runtime_abi:lhs_stride`,
  `rhs_stride_source=runtime_abi:rhs_stride`,
  `strided_load_intrinsic=__riscv_vlse16_v_i16mf2`, and
  `strided_memory_layout=unit-stride-compare-element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi`.
* `target_validator_consumed_facts` is populated in the generated evidence,
  and the target artifact metadata remains mirror-only after provider route
  construction.

## Evidence Artifacts

Primary non-dry-run artifact root:

```text
artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv
```

Generated per-op evidence:

```text
artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv/computed_masked_strided_input_widening_dot_reduce_add/evidence.json
```

Generated materialized body and emitted C++:

```text
artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv/computed_masked_strided_input_widening_dot_reduce_add/materialized_selected_body.mlir
artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv/computed_masked_strided_input_widening_dot_reduce_add/materialized_rvv_emitc.cpp
```

Generated bundle and harness:

```text
artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv/computed_masked_strided_input_widening_dot_reduce_add/generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o
artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv/computed_masked_strided_input_widening_dot_reduce_add/generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h
artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv/computed_masked_strided_input_widening_dot_reduce_add/rvv_generated_bundle_abi_computed_masked_strided_input_widening_dot_reduce_add_harness.c
```

Remote compile and run transcript:

```text
artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv/computed_masked_strided_input_widening_dot_reduce_add/remote_compile_stdout.txt
artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv/computed_masked_strided_input_widening_dot_reduce_add/remote_run_stdout.txt
```

Remote compile summary:

```text
remote_arch=riscv64
clang_path=/usr/bin/clang
clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)
```

Remote correctness output:

```text
computed_masked_strided_input_widening_dot_reduce_add case n=0 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=2,3 mask_pattern=0 input_pattern=0 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=1 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=2,3 mask_pattern=0 input_pattern=0 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=7 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=2,3 mask_pattern=0 input_pattern=0 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=16 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=2,3 mask_pattern=0 input_pattern=0 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=23 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=2,3 mask_pattern=0 input_pattern=0 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=257 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=2,3 mask_pattern=0 input_pattern=0 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=0 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=3,2 mask_pattern=1 input_pattern=1 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=1 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=3,2 mask_pattern=1 input_pattern=1 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=7 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=3,2 mask_pattern=1 input_pattern=1 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=16 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=3,2 mask_pattern=1 input_pattern=1 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=23 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=3,2 mask_pattern=1 input_pattern=1 skipped_source_elements_ignored scalar_output tail_preserved
computed_masked_strided_input_widening_dot_reduce_add case n=257 ok compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=3,2 mask_pattern=1 input_pattern=1 skipped_source_elements_ignored scalar_output tail_preserved
tcrv_rvv_generated_bundle_abi_computed_masked_strided_input_widening_dot_reduce_add_ok counts=0,1,7,16,23,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2
PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,7,16,23,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2
```

## Checks And Self-Repair

Checks run:

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-strided-input-widening-dot-reduction-executable-artifact
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-dry-run --run-id computed-masked-strided-input-widening-dot-reduce-add-dry-run --overwrite --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact --run-id computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv --overwrite --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
rtk bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add'
rtk git diff --check
```

Focused lit result:

```text
Total Discovered Tests: 477
  Excluded: 476 (99.79%)
  Passed  :   1 (0.21%)
```

Bounded old-authority scan:

```text
rtk rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|__riscv_.*_i32m1|descriptor|direct-C|source-export|source-front-door|route-id/artifact-name|common EmitC semantic inference|mirror-only authority" .trellis/tasks/06-04-stage2-rvv-strided-input-widening-dot-reduction-executable-artifact
rtk rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|__riscv_.*_i32m1" include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir scripts/rvv_generated_bundle_abi_e2e.py
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
dry-run, focused target fixture lit check, non-dry-run `ssh rvv` compile/run,
task validation, old-authority scan, and diff check cover the changed behavior
for this evidence-only closure.

## Spec Update Review

No `.trellis/spec/` update was made. This task did not introduce a new
compiler contract, API signature, validation matrix, or cross-layer behavior.
The computed-mask strided-input widening dot-reduction provider facts, target
validation contract, generated-bundle evidence boundary, and `ssh rvv`
requirements were already present in the relevant specs before this round.
