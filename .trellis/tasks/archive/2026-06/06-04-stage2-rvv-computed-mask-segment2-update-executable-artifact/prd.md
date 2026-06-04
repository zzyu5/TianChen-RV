# Stage2 RVV computed-mask segment2 update executable artifact

## Goal

Close one representative Stage2 RVV computed-mask segment2 update executable
artifact path. The selected submodule is
`computed_masked_segment2_update_unit_load`, because the repository already has
a pre-realized selected-body fixture, provider/target validation surfaces,
generated-bundle tooling, and this is the next memory-movement/update class
after the recently closed widening-dot contraction evidence.

The path must start from a selected `tcrv.exec` RVV variant with explicit typed
`tcrv_rvv` computed-mask segment2 update body facts, flow through RVV
plugin-owned selected-body realization and provider route construction,
materialize through neutral common EmitC, generate a target artifact/header
plus external C ABI harness, and compile/run on real `ssh rvv` with focused
correctness evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask segment2 update executable artifact`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: `ok` from RTK before task creation, with
  no injected dirty-state warning.
* Initial `rtk git log --oneline -8` starts at
  `214c595e rvv: record strided widening dot ssh evidence`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* After task creation, `rtk proxy git status --short` reports the new untracked
  task directory only.
* This round is run by one serial Codex worker; no subagents, spawned agents,
  parallel agents, or multi-agent workflow are used.

## What I Already Know

* `.trellis/spec/index.md` defines the current RVV authority chain: selected
  `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV plugin-owned
  legality/realization/provider -> `TCRVEmitCLowerableRoute` -> common EmitC
  -> target artifact -> `ssh rvv` evidence when runtime, correctness, or
  performance is claimed.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires selected
  pre-realized RVV bodies to be realized by the RVV plugin before provider
  route construction. `tcrv.exec` owns ABI/runtime role declarations, while
  `tcrv_rvv` body/config/runtime facts own dtype, memory form, operation, mask,
  VL, and compute authority.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the computed-mask
  segment2 memory fact surface and the segment2 memory route validation
  contract. For `computed_masked_segment2_update_unit_load`, provider/target
  facts must include runtime ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`, typed
  compute op `tcrv_rvv.binary`, arithmetic kind `add`, compare predicate
  `slt`, compare-produced mask facts, segment count `2`, masked segment-store
  facts, field roles/names, header/type summaries, binding summary, provider
  mirror, target leaf profile, runtime control, and statement-plan facts.
* `.trellis/spec/lowering-runtime/emitc-route.md` also requires target artifact
  validation to consume the provider-owned
  `RVVSegment2MemoryRouteValidationContract` and provider-owned metadata mirror
  contracts. Target validation must not reconstruct segment2/update facts from
  route ids, artifact names, fixture names, descriptors, scripts, common EmitC,
  candidate metadata mirrors, or exact intrinsic spellings.
* `.trellis/spec/testing/mlir-testing-contract.md` requires current
  pre-realized selected-body generated-bundle evidence to use
  `--pre-realized-selected-body`, the selected lowering-boundary
  materialization path, provider-built route facts, target bundle generation,
  and external C ABI harness. Runtime/correctness claims require non-dry-run
  `ssh rvv` compile/run evidence.
* The archived task
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-strided-input-widening-dot-reduction-executable-artifact/`
  shows the expected closure pattern: if production provider/target/export
  already works, keep this task honest as executable compiler-path evidence
  rather than making unrelated code changes.
* The relevant workspace journal entry records the immediately prior closure
  for `computed_masked_strided_input_widening_dot_reduce_add` with generated
  artifact, external ABI harness, and real `ssh rvv` correctness over runtime
  counts `0,1,7,16,23,257`.
* Live repository inspection confirms the preferred fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`.
  It declares runtime ABI values `cmp_lhs`, `cmp_rhs`, `src0`, `src1`, `dst`,
  and `n`; the selected body uses
  `tcrv_rvv.typed_computed_mask_segment2_store_pre_realized_body` with
  `op_kind = "computed_masked_segment2_update_unit_load"`,
  `arithmetic_kind = "add"`, SEW32 LMUL m1, compare predicate `slt`,
  compare-produced mask facts, field0/field1 unit-stride payload loads,
  destination memory form `segment2-interleaved-unit-stride-store`,
  inactive-lane policy `preserve-output-on-false-lanes`, and segment count 2.
* The fixture FileChecks realization into `setvl`, `with_vl`, compare loads,
  field payload loads, `tcrv_rvv.compare`, `tcrv_rvv.binary {kind = "add"}`,
  and `tcrv_rvv.masked_segment2_store`, while rejecting indexed/strided/plain
  segment2 and generic mask-load residue.
* The fixture FileChecks emission-plan/header mirrors for runtime ABI order,
  binding plan and operands, computed-mask family plan, target leaf profile,
  provider-supported mirror, headers, C type mapping, mask facts, segment
  layout, segment count, masked segment-store intrinsic, tuple-create
  intrinsic, update arithmetic kind/intrinsic, and external C ABI prototype.
* `scripts/rvv_generated_bundle_abi_e2e.py` exposes
  `--op-kind computed_masked_segment2_update_unit_load` and contains
  expectation/harness branches for computed-mask segment2 load/store/update
  operations.
* Live source inspection confirms segment2 provider and target surfaces in
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.

## Requirements

* Keep the owner bounded to one representative computed-mask segment2 update
  executable path: `computed_masked_segment2_update_unit_load`.
* Generate the artifact through the production compiler/tooling path, not by
  hand-writing C, descriptors, source-front-door routes, route-id-derived
  shims, script-side compiler-core substitutes, or common EmitC semantic
  inference.
* Preserve selected `tcrv.exec` runtime `n` plus `cmp_lhs`, `cmp_rhs`, `src0`,
  `src1`, and `dst` ABI boundaries through typed body realization, provider
  route facts, target artifact validation, generated header, generated C/C++,
  and harness calls.
* Preserve compare mask provenance, predicate `slt`, field0/field1 payload
  load roles, binary update kind `add`, segment count 2, segment tuple/store
  facts, destination memory form, inactive-lane preservation contract,
  setvl/VL, tail handling, runtime AVL source, and route-local mirror
  boundaries.
* Unsupported or stale non-segment2, non-update, non-computed-mask, non-provider
  facts must fail closed before becoming artifact authority.
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

* [x] A representative pre-realized typed computed-mask segment2 update
      selected-body case is chosen and documented.
* [x] The selected path generates a target artifact/header and external ABI
      harness through `rvv_generated_bundle_abi_e2e.py` without `--dry-run`.
* [x] Evidence records selected-body realization, provider route facts, target
      validator consumption, runtime AVL/VL boundary, compare-produced mask
      facts, field0/field1 payload load roles, update arithmetic kind,
      segment2 tuple/store facts, destination memory form, inactive-lane
      preservation, generated object/header paths, harness path, and runtime
      `n`.
* [x] Generated C/C++ and harness evidence prove masked segment2 update
      behavior: compare-produced mask, field0/field1 payload loads, `add`
      update into field0, field1 passthrough, masked segment2 store,
      inactive-lane destination preservation, and tail sentinel preservation.
* [x] The generated harness compiles and runs on `ssh rvv`; remote output
      includes the expected pass marker and
      `PASS op=computed_masked_segment2_update_unit_load`.
* [x] Correctness evidence covers runtime counts including zero, one, at least
      one exact-VL or multi-lane count, a tail count, and a stress count; it
      checks active mask lanes, inactive mask lanes, field0 update semantics,
      field1 preservation, destination inactive-lane preservation, and tail
      preservation.
* [x] If production/tooling code changes, focused checks cover the changed
      owner: generated-bundle lit/export checks plus
      `tianchenrv-target-artifact-export-test` and/or
      `tianchenrv-rvv-extension-plugin-test` as applicable.
* [x] If no production code changes are needed, focused dry-run/lit and the
      non-dry-run `ssh rvv` evidence are sufficient, and the
      no-production-change rationale is recorded.
* [x] Bounded old-authority scan over touched and relevant files finds no new
      positive dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      route-id/artifact-name authority, common EmitC semantic inference, exact
      `__riscv_*_i32m1`, or mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task context is valid and task status is truthful.
* [x] A coherent commit is created if the task completes.

## Technical Approach

1. Inspect the computed-mask segment2 update specs, fixture, route
   planning/provider/target validation code, and generated-bundle harness
   support for `computed_masked_segment2_update_unit_load`.
2. Run the existing dry-run/generated-bundle path to confirm the route emits
   expected artifact facts, materialized body, emitted C++ checks, evidence
   boundary, statement plan, harness, and runtime count contract.
3. Run a focused non-dry-run generated-bundle command for
   `computed_masked_segment2_update_unit_load` on `ssh rvv` with counts
   `0,1,7,16,23,257`.
4. Inspect generated `evidence.json`, per-op evidence, materialized selected
   body, materialized RVV EmitC/C++, generated bundle header/object, generated
   harness, `remote_compile_stdout.txt`, and `remote_run_stdout.txt`.
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
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-segment2-update-executable-artifact --run-id computed-masked-segment2-update-ssh-rvv --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv
```

Focused dry-run/lit regression if no production code changes:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-computed-mask-segment2-update-dry-run --run-id computed-masked-segment2-update-dry-run --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk proxy bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-computed-masked-segment2-update'
```

Production checks if provider/target/plugin code changes:

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 8
```

## Evidence Results

Chosen submodule:

```text
computed_masked_segment2_update_unit_load
```

Selected fixture:

```text
test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir
```

Dry-run generated artifact evidence:

```text
artifacts/tmp/stage2-rvv-computed-mask-segment2-update-dry-run/computed-masked-segment2-update-dry-run
```

Dry-run command completed with:

```text
rvv_generated_bundle_abi_e2e: dry_run_success
```

Non-dry-run `ssh rvv` generated artifact evidence:

```text
artifacts/tmp/stage2-rvv-computed-mask-segment2-update-executable-artifact/computed-masked-segment2-update-ssh-rvv
```

The non-dry-run evidence records:

* `input_mode = pre-realized-selected-body`.
* `ssh_evidence = true`.
* Runtime ABI prototype:
  `void tcrv_emitc_pre_realized_body_cmseg_update_kernel_pre_realized_body_rvv_cmseg_update(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);`
* Runtime AVL source: selected runtime ABI `n` -> `tcrv_rvv.setvl` ->
  `tcrv_rvv.with_vl` -> emitted loop setvl -> artifact ABI.
* Binding summary:
  `rvv-route-operand-binding:cmseg2_update_unit_load.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|add-lhs|tuple0|f0-role|src0-mem|hdr;src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|add-rhs|tuple1|f1-role|src1-mem|hdr;dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr`.
* Generated object:
  `computed_masked_segment2_update_unit_load/generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o`.
* Generated header:
  `computed_masked_segment2_update_unit_load/generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h`.
* Harness:
  `computed_masked_segment2_update_unit_load/rvv_generated_bundle_abi_computed_masked_segment2_update_unit_load_harness.c`.

Materialized selected-body inspection confirmed:

* `tcrv_rvv.setvl` consumes runtime ABI `n`.
* `tcrv_rvv.with_vl` scopes the selected RVV variant.
* Compare inputs, `src0`, and `src1` are loaded from explicit runtime ABI
  values.
* `tcrv_rvv.compare {kind = "slt"}` produces the mask.
* `tcrv_rvv.binary {kind = "add"}` updates field0 payload.
* `tcrv_rvv.masked_segment2_store` stores updated field0 plus field1 using
  segment count 2, destination memory form
  `segment2-interleaved-unit-stride-store`, and inactive-lane policy
  `preserve-output-on-false-lanes`.

Generated C++ inspection confirmed:

* Runtime loop uses `__riscv_vsetvl_e32m1`.
* Compare/source payloads use unit `__riscv_vle32_v_i32m1` loads derived from
  the selected typed body.
* Compare mask uses `__riscv_vmslt_vv_i32m1_b32`.
* Update arithmetic uses `__riscv_vadd_vv_i32m1`.
* Segment tuple/store uses `__riscv_vcreate_v_i32m1x2` and
  `__riscv_vsseg2e32_v_i32m1x2_m`.

The generated harness checks:

* active compare-mask lanes;
* inactive compare-mask lanes;
* field0 update as `src0[index] + src1[index]`;
* field1 passthrough from `src1[index]` on active lanes;
* inactive-lane preservation of the old interleaved destination;
* source buffer preservation;
* tail sentinel preservation beyond `2 * n`;
* two mask/input patterns for counts `0,1,7,16,23,257`.

Remote compile/run evidence:

```text
remote_arch=riscv64
clang_path=/usr/bin/clang
clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)
PASS op=computed_masked_segment2_update_unit_load counts=0,1,7,16,23,257 patterns=0,1
```

Focused lit evidence:

```text
rtk proxy bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-computed-masked-segment2-update'
Total Discovered Tests: 477
Excluded: 476
Passed: 1
```

No production code changes were needed. The existing selected-body realization,
RVV segment2 provider, common EmitC materialization, target artifact
validation, generated-bundle tooling, and harness path already executed the
representative segment2 update path end to end.

Bounded old-authority scan:

* Hard Stage1 legacy markers
  `RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m` appeared only in this
  PRD's negative acceptance wording.
* The selected fixture contains provider-derived current RVV intrinsic mirrors
  for segment2 update (`__riscv_vsseg2e32_v_i32m1x2_m`,
  `__riscv_vcreate_v_i32m1x2`, `__riscv_vadd_vv_i32m1`). These are mirrored
  after typed body/provider route construction, not used as route authority.
* Existing script-wide intrinsic constants and descriptor/source-front-door
  negative checks were not modified in this round.

## Out of Scope

* All segment2 variants beyond the representative update case.
* Indexed gather/scatter, dtype/LMUL clone batches, high-level
  Linalg/Vector/StableHLO frontend lowering, source-front-door positive
  routes, one-intrinsic wrapper dialects, unrelated contraction/reduction/MAcc
  work, dashboards, reports, or broad smoke matrices.
* Inferring segment2 semantics from route ids, artifact names, test names,
  manifests, C strings, descriptors, intrinsic spellings, or mirror metadata.
* Moving RVV semantics into common EmitC/export.
