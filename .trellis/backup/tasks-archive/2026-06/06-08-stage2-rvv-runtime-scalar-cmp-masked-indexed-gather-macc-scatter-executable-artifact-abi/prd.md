# Stage2 RVV runtime-scalar-cmp masked indexed gather-MAcc-scatter executable artifact ABI boundary

## Goal

Make the existing
`runtime_scalar_cmp_masked_indexed_gather_macc_scatter` selected-body route
family a truthful generated RVV artifact ABI workflow, or fail it closed at the
RVV plugin / target artifact boundary if any required executable artifact fact
is missing or stale.

The focused production path is:

```text
selected or pre-realized tcrv_rvv body
  -> RVV provider-owned runtime-scalar compare, mask, indexed gather, MAcc,
     indexed scatter, dtype/config, ABI, and header facts
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
  `ed179e8d rvv: record masked strided dot ssh evidence`.
* The immediately previous archived task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-computed-masked-strided-dot-executable-artifact-abi/`
  closed the computed-masked strided dot route by recording fresh explicit and
  pre-realized `ssh rvv` generated-bundle evidence without changing compiler
  source.
* The RVV plugin spec already contains a dedicated
  "Runtime-Scalar Indexed Gather-MAcc-Scatter Route Contract" for this route
  family.
* The spec requires runtime ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`.
* The selected explicit body must realize one VL scope containing runtime
  scalar `sle` compare, compare-produced mask, indexed gather from
  `gather_src[index[i]]`, masked MAcc using gathered value, payload, and
  accumulator, and masked indexed scatter of the MAcc result to `dst[index[i]]`.
* Common EmitC is neutral. It may only materialize provider-built route payloads
  and must not infer mask, index, gather, MAcc, scatter, dtype, policy, ABI, or
  header semantics.
* `scripts/rvv_generated_bundle_abi_e2e.py` already has explicit and
  pre-realized expectations for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`, including prototype,
  runtime ABI order, route operand binding summary, oracle expression, and
  generated harness checks.
* The existing generated-bundle dry-run lit test
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`
  covers both explicit and pre-realized selected bodies for counts
  `0,1,16,17,257` and RHS scalar values `-37,91`, but records
  `ssh_evidence: false`.
* The explicit target artifact fixture already checks stale provider mirror,
  stale ABI order, stale exec binding, stale operand binding, missing exec
  mirror, missing exec binding, missing dispatch case/fallback, stale dispatch
  mirrors, stale composite resource/plan, stale write-side contract, missing
  composite resource fact, and unsupported fallback export.
* The pre-realized target artifact fixture already checks composite realization
  into the explicit body shape, provider facts, target header metadata, stale
  provider mirror, stale ABI order, stale exec binding, stale operand binding,
  stale composite resource, and missing exec binding.

## Requirements

* Stay on the existing
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` route family. Do not
  choose another route family.
* The production seam must remain RVV provider-owned and target-consumed.
  Common EmitC may only materialize provider-built route payloads.
* Explicit selected-body and pre-realized selected-body paths are both in
  scope. The pre-realized path must be consumed by the RVV plugin-local
  composite realization owner before route construction.
* Runtime ABI order must be
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`.
* Runtime ABI/header binding must include compare lhs, scalar threshold, gather
  source, payload, accumulator, index, destination, and runtime AVL/count with
  exported header/prototype participation markers.
* The selected typed body or realized body must be the authority for compare
  predicate, mask provenance, indexed gather source, MAcc operands and
  accumulator role, indexed scatter destination, inactive-lane behavior,
  passthrough layout, dtype, SEW, LMUL, policy, runtime AVL/VL, and index
  memory facts.
* The executable artifact proof must use the generated target object/header
  bundle and generated external C ABI consumer. It must not use a handwritten
  direct runtime wrapper as the route authority.
* Positive runtime evidence must be real `ssh rvv` evidence, not dry-run or
  local host execution. The evidence must show `dry_run: false` and
  `ssh_evidence: true`.
* Runtime evidence must compare against an oracle over varied counts, runtime
  scalar thresholds, active and inactive lanes, noncontiguous unique indices,
  signed products, source preservation, payload/accumulator preservation, and
  output tail preservation.
* If executable evidence exposes a production seam defect, fix the production
  seam or add a fail-closed guardrail at the RVV plugin/target artifact
  boundary before accepting any generated bundle.
* If production source does not need changes, the no-source-change conclusion
  must be justified by fresh materialized selected boundary, emission plan,
  target artifact export, generated bundle compile, `ssh rvv` runtime evidence,
  and existing focused fail-closed checks.

## Acceptance Criteria

* [x] Positive generated-bundle ABI e2e for the explicit selected-body
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` path compiles and
  runs on `ssh rvv`, reports `dry_run: false`, `ssh_evidence: true`, and passes
  oracle checks for counts `0,1,16,17,257`, RHS scalars `-37,91`, active and
  inactive lanes, noncontiguous unique indices, signed products, source
  preservation, payload/accumulator preservation, inactive destination
  preservation, and output tail preservation; or the PRD records the precise
  executable blocker.
* [x] Positive generated-bundle ABI e2e for the pre-realized selected-body path
  compiles and runs on `ssh rvv` with the same correctness scope, or the PRD
  records the precise executable blocker.
* [x] Existing dry-run generated-bundle checks for explicit and pre-realized
  selected bodies still pass and still show provider route, runtime ABI order,
  route operand binding summary, composite resource selection, indexed
  gather/scatter contract, MAcc oracle, header prototype, and generated harness
  checks.
* [x] Existing target artifact fixtures still pass and still fail closed for
  stale provider facts, stale ABI/header/prototype binding, stale exec binding,
  stale operand binding, stale dispatch mirrors, stale composite resource/plan,
  stale indexed write-side contract, missing composite resource fact, and
  unsupported executable export.
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

## Out Of Scope

* No broad indexed-memory matrix.
* No dtype/LMUL clone batch.
* No new high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No source-front-door positive route.
* No performance tuning database, dashboard, or report-only closeout.
* No common EmitC invention of RVV semantics.
* No mass rewrite of unrelated memory, segment2, compare/select, reduction, or
  dequant routes outside this indexed gather-MAcc-scatter artifact seam.
* No rework of completed dot/reduction/MAcc evidence except as reference.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Archived task read:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-computed-masked-strided-dot-executable-artifact-abi/prd.md`.
* Primary files from the brief and repo inspection:
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Plugin/ConstructionProtocolCommonTest.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Results

No compiler source change was required. The existing production
provider/target contract already generated executable object/header bundles for
both bounded selected-body paths, and this round refreshed explicit
`ssh rvv` runtime evidence for the more kernel-like indexed
gather-MAcc-scatter artifact ABI boundary.

Positive `ssh rvv` evidence:

* Explicit selected-body:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-explicit-runtime-scalar-cmp-masked-indexed-gms-ssh/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Pre-realized selected-body:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-prerealized-runtime-scalar-cmp-masked-indexed-gms-ssh/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.

Both remote runs printed:

```text
PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1
```

The evidence boundary records provider-derived authority for runtime ABI order
`cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, route operand binding
plan `rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1`, composite
resource selection
`rvv-composite-gather-macc-scatter-resource-candidate.v1[rt-scmp-indexed-gather-macc-scatter,e32m1,u1]`,
runtime scalar compare predicate `sle`, compare-produced mask, indexed gather
from `gather_src[index[i]]`, MAcc over gathered value/payload/accumulator,
indexed scatter to `dst[index[i]]`, inactive destination preservation,
noncontiguous unique index coverage, signed product coverage, source
preservation, payload/accumulator preservation, and tail preservation.

Negative/fail-closed evidence was revalidated by:

* filtered lit for
  `runtime-scalar-cmp-masked-indexed-gather-macc-scatter`, which passed 3/3
  tests and covers the generated-bundle dry-run fixture plus explicit and
  pre-realized target artifact fixtures;
* `build/bin/tianchenrv-rvv-extension-plugin-test`;
* `build/bin/tianchenrv-target-artifact-export-test`.

Spec update judgment: no `.trellis/spec/` change is needed in this round. The
durable route contract already exists in the RVV plugin and EmitC route specs;
this task refreshed runtime artifact evidence and did not discover a new
contract, signature, validation matrix, or reusable convention.
