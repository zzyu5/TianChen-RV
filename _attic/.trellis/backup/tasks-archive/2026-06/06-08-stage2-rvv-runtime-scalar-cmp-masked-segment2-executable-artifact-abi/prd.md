# Stage2 RVV runtime-scalar-cmp masked segment2 load/store executable artifact ABI boundary

## Goal

Make the existing runtime-scalar compare + computed-mask + masked segment2
load/store selected-body route family a truthful generated RVV artifact ABI
workflow, or fail it closed at the RVV plugin / target artifact boundary when
any executable artifact fact is missing or stale.

The focused production path is:

```text
selected or pre-realized tcrv_rvv body
  -> RVV provider-owned runtime-scalar compare, mask, segment2 field, dtype,
     config, ABI, header, runtime AVL/VL, and statement facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target object/header bundle
  -> generated C ABI consumer
  -> ssh rvv correctness evidence when runtime executability is claimed
```

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at
  `9b1c2b89 rvv: record indexed gather macc scatter ssh evidence`.
* The immediately previous archived task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-executable-artifact-abi/`
  proved explicit and pre-realized generated object/header bundles for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` on `ssh rvv`
  without compiler source changes.
* The RVV plugin and EmitC route specs already require segment2 route support
  to derive authority from typed selected-body facts, provider-owned segment2
  route-family plans, memory operand-binding facts, statement plans, and target
  artifact validation contracts.
* Runtime-scalar computed-mask segment2 load uses runtime ABI order
  `lhs,rhs_scalar,src,out0,out1,n`.
* Runtime-scalar computed-mask segment2 store uses runtime ABI order
  `lhs,rhs_scalar,src0,src1,dst,n`.
* `rhs_scalar` is an ABI scalar threshold that the RVV plugin must realize as
  a scalar splat before compare. Common EmitC and target export may only carry
  provider-built route payloads and mirrors.
* Current brief names existing dry-run generated-bundle tests for explicit and
  pre-realized runtime-scalar segment2 load/store fixtures. Those tests are
  not by themselves executable runtime evidence.

## Requirements

* Stay on the runtime-scalar-cmp masked segment2 memory family. Do not choose a
  different RVV route family.
* Treat segment2 load/store as one memory-family owner. If both sides cannot be
  completed coherently, finish one bounded load/store submodule and record the
  exact continuation point inside this owner.
* The production seam must remain RVV provider-owned and target-consumed.
  Common EmitC may only materialize provider-built route payloads.
* Explicit selected-body and pre-realized selected-body paths are both in
  scope for the completed submodule.
* Runtime-scalar segment2 load must preserve ABI/header binding for `lhs`,
  `rhs_scalar`, `src`, `out0`, `out1`, and `n`.
* Runtime-scalar segment2 store must preserve ABI/header binding for `lhs`,
  `rhs_scalar`, `src0`, `src1`, `dst`, and `n`.
* Typed or realized bodies must be authority for compare predicate,
  runtime-scalar splat, mask provenance, segment2 load/store direction,
  field0/field1 tuple roles, active/inactive lane behavior, dtype, SEW, LMUL,
  policy, runtime AVL/VL, source/destination memory forms, header/prototype
  bindings, and statement facts.
* The executable artifact proof must use the generated target object/header
  bundle and generated external C ABI consumer. It must not use a handwritten
  direct runtime wrapper as the route authority.
* Positive runtime evidence must be real `ssh rvv` evidence, not dry-run or
  local host execution. The evidence must show `dry_run: false` and
  `ssh_evidence: true`.
* Runtime evidence must cover active lanes, inactive lane preservation or
  no-write behavior, field0/field1 distinction, runtime scalar threshold
  variation, runtime counts including tail cases, and source/output tail
  preservation.
* If executable evidence exposes a production seam defect, fix the production
  seam or add a fail-closed guardrail at the RVV plugin/target artifact
  boundary before accepting any generated bundle.
* If production source does not need changes, the no-source-change conclusion
  must be justified by fresh materialized selected boundary, emission plan,
  target artifact export, generated bundle compile, `ssh rvv` runtime evidence,
  and existing focused fail-closed checks.

## Acceptance Criteria

* [x] Positive generated-bundle ABI e2e for the explicit selected-body
  runtime-scalar segment2 load path compiles and runs on `ssh rvv`, reports
  `dry_run: false`, `ssh_evidence: true`, and passes oracle checks; or the PRD
  records the precise executable blocker.
* [x] Positive generated-bundle ABI e2e for the pre-realized selected-body
  runtime-scalar segment2 load path compiles and runs on `ssh rvv` with the
  same correctness scope; or the PRD records the precise executable blocker.
* [x] Positive generated-bundle ABI e2e for the explicit selected-body
  runtime-scalar segment2 store path compiles and runs on `ssh rvv`, reports
  `dry_run: false`, `ssh_evidence: true`, and passes oracle checks; or the PRD
  records the precise executable blocker.
* [x] Positive generated-bundle ABI e2e for the pre-realized selected-body
  runtime-scalar segment2 store path compiles and runs on `ssh rvv` with the
  same correctness scope; or the PRD records the precise executable blocker.
* [x] Existing generated-bundle dry-run lit checks for explicit and
  pre-realized selected bodies still pass and still expose provider route,
  runtime ABI order, route operand binding summary, runtime-scalar mask
  producer, segment2 field roles, header prototype, and generated harness
  checks.
* [x] Existing target artifact fixtures still pass and still fail closed for
  stale provider facts, stale ABI/header/prototype binding, stale mask producer,
  stale field ordering, stale operand binding, and stale segment2 statement
  facts.
* [x] At least one focused fail-closed check covers a stale or missing
  executable-boundary fact if this round changes production source. No
  production source changed in this round; existing target artifact fixture
  fail-closed coverage was revalidated by focused lit.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passes.
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if
  the evidence script is touched or if the generated-bundle harness path needs
  validation before `ssh rvv`.
* [x] Bounded old-authority scan over touched files and added diff lines finds
  no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C, source-front-door, or mirror-only
  route authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean or explicitly reported.
* [x] Trellis task context, workspace journal, finish/archive status, and one
  coherent commit are completed if the task finishes.

## Definition of Done

* Runtime-scalar segment2 executable claims are backed by generated bundle
  evidence on `ssh rvv`, or blocked by a named production ABI/artifact defect.
* Provider-owned route/statement/operand-binding facts remain the authority;
  target artifact export consumes those facts and metadata remains a mirror.
* The task is archived with truthful PRD/check notes and a coherent commit.

## Out Of Scope

* No broad memory matrix.
* No dtype/LMUL clone batch.
* No indexed gather-MAcc-scatter rework except as reference.
* No MAcc expansion.
* No standalone report/dashboard/status work.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No performance tuning database.
* No source-front-door positive route.
* No common EmitC invention of RVV semantics.
* No mass rewrite of unrelated indexed, reduction, compare/select, widening
  conversion, non-segment2 memory, or future plugin routes.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Prior task read:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-executable-artifact-abi/`.
* Primary files from the brief for this round:
  `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-segment2-load-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-segment2-load-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-segment2-store-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-segment2-store-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-segment2-load.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-segment2-load.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store.mlir`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store.mlir`.

## Results

No compiler source change was required. The existing production
provider/target contract already generated executable object/header bundles for
both runtime-scalar segment2 load and store paths, in explicit and
pre-realized selected-body modes. This round refreshed real `ssh rvv`
generated-bundle evidence for the load/store executable artifact ABI boundary.

Positive `ssh rvv` evidence:

* Explicit selected-body segment2 load:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-explicit-runtime-scalar-cmp-segment2-load-ssh/runtime_scalar_cmp_masked_segment2_load_unit_store/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Pre-realized selected-body segment2 load:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-prerealized-runtime-scalar-cmp-segment2-load-ssh/runtime_scalar_cmp_masked_segment2_load_unit_store/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Explicit selected-body segment2 store:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-explicit-runtime-scalar-cmp-segment2-store-ssh/runtime_scalar_cmp_masked_segment2_store_unit_load/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Pre-realized selected-body segment2 store:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-prerealized-runtime-scalar-cmp-segment2-store-ssh/runtime_scalar_cmp_masked_segment2_store_unit_load/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.

Remote PASS summaries:

```text
PASS op=runtime_scalar_cmp_masked_segment2_load_unit_store counts=0,1,16,17,257 patterns=0,1
PASS op=runtime_scalar_cmp_masked_segment2_load_unit_store counts=0,1,16,17,257 patterns=0,1
PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,16,17,257 patterns=0,1
PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,7,16,23,257 patterns=0,1
```

The remote harness evidence covers runtime scalar thresholds `-37` and `91`,
active and inactive lanes, inactive destination field preservation, field0 /
field1 distinction, source preservation, and tail preservation. Load evidence
uses ABI order `lhs,rhs_scalar,src,out0,out1,n`; store evidence uses ABI order
`lhs,rhs_scalar,src0,src1,dst,n`; both record
`runtime-scalar-splat-compare-rhs` as the computed-mask producer source.

Negative/fail-closed evidence was revalidated by focused lit for
`runtime-scalar-cmp-masked-segment2`, which passed 8/8 selected tests and
covers the four generated-bundle dry-run fixtures plus the four target artifact
fixtures.

Additional checks:

* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-segment2'`
  from `build/test` passed 8/8 selected tests.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `build/bin/tianchenrv-target-artifact-export-test` passed.

Spec update judgment: no `.trellis/spec/` change is needed in this round. The
durable segment2 provider/target/export contracts already exist in the RVV
plugin and EmitC route specs; this task refreshed executable artifact evidence
and did not discover a new contract, signature, validation matrix, or reusable
convention.
