# Stage2 RVV segment2 unit interleave/deinterleave executable artifact ABI boundary

## Goal

Make the existing plain RVV segment2 unit interleave/deinterleave selected-body
route family a truthful generated RVV artifact ABI workflow, or fail it closed
at the RVV plugin / target artifact boundary when any executable artifact fact
is missing or stale.

The focused production path is:

```text
selected or pre-realized tcrv_rvv segment2 body
  -> RVV provider-owned plain segment2 field, direction, dtype, config,
     ABI, header, runtime AVL/VL, and statement facts
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
  `e3e6179d rvv: demote source front-door artifact authority`.
* The immediately previous archived task
  `.trellis/tasks/archive/2026-06/06-08-rvv-source-front-door-positive-route-demotion/`
  demoted RVV source-front-door generated-bundle authority and left explicit /
  pre-realized typed selected-body modes as the positive artifact path.
* The prior runtime-scalar segment2 task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-executable-artifact-abi/`
  refreshed real `ssh rvv` evidence for runtime-scalar masked segment2
  load/store without production source changes.
* `.trellis/spec/extension-plugins/rvv-plugin.md` and
  `.trellis/spec/lowering-runtime/emitc-route.md` already define plain
  segment2 deinterleave/interleave provider facts, statement-plan ownership,
  route validation contracts, runtime ABI order, binding summaries, and
  source-front-door/direct-C prohibitions.
* Plain segment2 deinterleave uses runtime ABI order `src,out0,out1,n`,
  typed compute op `tcrv_rvv.move`, memory form `segment2-load-unit-store`,
  one segment2 interleaved source, field0/field1 output stores, segment-load
  and field-extract facts, and no tuple-create/segment-store residue.
* Plain segment2 interleave uses runtime ABI order `src0,src1,dst,n`, typed
  compute op `tcrv_rvv.segment2_store`, memory form
  `unit-load-segment2-store`, two unit-stride field sources, one interleaved
  segment2 destination, tuple-create and segment-store facts, and no
  segment-load/field-extract residue.
* Current production code already has:
  `RVVSegment2MemorySelectedBodyRealizationOwner` pre-realized realization for
  plain deinterleave/interleave; plain segment2 route-family provider plans;
  migrated segment2 memory statement-plan construction; provider-side
  `TCRVEmitCLowerableRoute` construction through the owner-built plan; and
  target artifact validation through a provider-owned segment2 contract.
* `scripts/rvv_generated_bundle_abi_e2e.py` already defines explicit and
  pre-realized expectations plus generated harnesses for
  `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`.
  Existing script dry-run tests cover the pre-realized modes only; this task
  must refresh real runtime evidence before claiming executability.

## Requirements

* Stay on the plain segment2 unit memory movement family. Do not choose a
  different RVV route family.
* Treat deinterleave and interleave as one coherent segment2 unit memory
  owner. If both sides cannot be completed coherently, finish one side and
  record the exact continuation point inside this owner.
* The production seam must remain RVV provider-owned and target-consumed.
  Common EmitC may only materialize provider-built route payloads.
* Explicit selected-body and pre-realized selected-body paths are both in
  scope for deinterleave and interleave.
* Deinterleave must preserve ABI/header binding for `src`, `out0`, `out1`,
  and `n`.
* Interleave must preserve ABI/header binding for `src0`, `src1`, `dst`, and
  `n`.
* Typed or realized bodies must be authority for segment2 direction,
  field0/field1 tuple roles, source/destination memory roles, dtype, SEW,
  LMUL, policy, runtime AVL/VL, header/prototype bindings, route-family facts,
  and statement facts.
* The executable artifact proof must use the generated target object/header
  bundle and generated external C ABI consumer. It must not use a handwritten
  direct runtime wrapper as route authority.
* Positive runtime evidence must be real `ssh rvv` evidence, not dry-run or
  local host execution. The evidence must show `dry_run: false` and
  `ssh_evidence: true`.
* Runtime evidence must cover active element movement, field0/field1 order
  distinction, runtime counts including tail cases, destination tail
  preservation, and no source-front-door/source-artifact authority.
* If executable evidence exposes a production seam defect, fix the production
  seam or add a fail-closed guardrail at the RVV plugin/target artifact
  boundary before accepting any generated bundle.
* If production source does not need changes, the no-source-change conclusion
  must be justified by fresh materialized selected boundary, emission plan,
  target artifact export, generated bundle compile, `ssh rvv` runtime
  evidence, and existing focused fail-closed checks.

## Acceptance Criteria

* [x] Positive generated-bundle ABI e2e for the explicit selected-body
  `segment2_deinterleave_unit_store` path compiles and runs on `ssh rvv`,
  reports `dry_run: false`, `ssh_evidence: true`, and passes oracle checks; or
  this PRD records the precise executable blocker.
* [x] Positive generated-bundle ABI e2e for the pre-realized selected-body
  `segment2_deinterleave_unit_store` path compiles and runs on `ssh rvv` with
  the same correctness scope; or this PRD records the precise executable
  blocker.
* [x] Positive generated-bundle ABI e2e for the explicit selected-body
  `segment2_interleave_unit_load` path compiles and runs on `ssh rvv`, reports
  `dry_run: false`, `ssh_evidence: true`, and passes oracle checks; or this
  PRD records the precise executable blocker.
* [x] Positive generated-bundle ABI e2e for the pre-realized selected-body
  `segment2_interleave_unit_load` path compiles and runs on `ssh rvv` with the
  same correctness scope; or this PRD records the precise executable blocker.
* [x] Existing generated-bundle dry-run lit checks for pre-realized
  deinterleave/interleave still pass and still expose provider route, runtime
  ABI order, route operand binding summary, segment2 field roles,
  header/prototype, harness oracle, and no descriptor/direct-C/source-export
  residue.
* [x] Existing target artifact fixtures for explicit and pre-realized
  deinterleave/interleave still pass and still expose materialized selected
  boundary, emission plan, object/header bundle metadata, provider mirror,
  header prototype, and binding summaries.
* [x] Existing C++ plugin and target artifact export tests still fail closed
  for stale provider facts, stale ABI/header/prototype binding, stale
  interleave/deinterleave direction, stale field ordering, stale operand
  binding, and stale segment2 statement facts.
* [x] At least one focused fail-closed check covers a stale or missing
  executable-boundary fact if this round changes production source. If no
  production source changes, existing focused fail-closed coverage is
  revalidated and named.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passes.
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if
  the evidence script is touched or if the generated-bundle harness path needs
  validation before `ssh rvv`.
* [x] Bounded old-authority scan over touched files and added diff lines finds
  no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C, source-front-door, source-artifact,
  or mirror-only route authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean or explicitly reported.
* [x] Trellis task context, workspace journal, finish/archive status, and one
  coherent commit are completed if the task finishes.

## Definition Of Done

Plain segment2 interleave/deinterleave executable claims are backed by
generated bundle evidence on `ssh rvv`, or blocked by a named production
ABI/artifact defect. Provider-owned route/statement/operand-binding facts remain
the authority; target artifact export consumes those facts and metadata remains
a mirror.

## Out Of Scope

* No source-front-door positive route or vector-source-front-door generated
  bundle mode.
* No direct pre-realized route-entry positive path.
* No broad segment2 matrix.
* No dtype/LMUL clone batch.
* No runtime-scalar masked segment2 expansion.
* No indexed gather/scatter, MAcc, reduction, compare/select, conversion, or
  unrelated memory route rework.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No performance tuning database.
* No common EmitC invention of RVV semantics.
* No mass rewrite of unrelated RVV route families.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`, and
  `.trellis/spec/guides/capability-first-design-guide.md`.
* Prior tasks read:
  `.trellis/tasks/archive/2026-06/06-08-rvv-source-front-door-positive-route-demotion/`
  and
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-executable-artifact-abi/`.
* Primary production files from the brief inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
* Primary tests from the brief inspected:
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-segment2-deinterleave-unit-store-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-segment2-interleave-unit-load-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-segment2-deinterleave-unit-store-fail-closed.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-segment2-interleave-unit-load-fail-closed.test`,
  `test/Target/RVV/explicit-selected-body-artifact-segment2-deinterleave-unit-store.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-segment2-deinterleave-unit-store.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-segment2-interleave-unit-load.mlir`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`.

## Results

No compiler source change was required. The existing production
provider/target/script contract already generated executable object/header
bundles for both plain segment2 deinterleave and interleave paths, in explicit
and pre-realized selected-body modes. This round refreshed real `ssh rvv`
generated-bundle evidence for the segment2 unit executable artifact ABI
boundary.

Positive `ssh rvv` evidence:

* Explicit selected-body segment2 deinterleave:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-explicit-segment2-deinterleave-ssh/segment2_deinterleave_unit_store/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Pre-realized selected-body segment2 deinterleave:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-prerealized-segment2-deinterleave-ssh/segment2_deinterleave_unit_store/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Explicit selected-body segment2 interleave:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-explicit-segment2-interleave-ssh/segment2_interleave_unit_load/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Pre-realized selected-body segment2 interleave:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-prerealized-segment2-interleave-ssh/segment2_interleave_unit_load/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.

Remote PASS summaries:

```text
PASS op=segment2_deinterleave_unit_store counts=0,1,16,17,257
PASS op=segment2_deinterleave_unit_store counts=0,1,16,17,257
PASS op=segment2_interleave_unit_load counts=0,1,7,16,23,257
PASS op=segment2_interleave_unit_load counts=0,1,7,16,23,257
```

The remote harness evidence covers two field-distinguishing source patterns,
runtime count zero, singleton, vector-sized, tail, and large cases. Deinterleave
evidence validates even/odd interleaved source fields into field0/field1
outputs and output tail sentinel preservation. Interleave evidence validates
field0/field1 source fields into even/odd interleaved destination lanes and
destination tail sentinel preservation past runtime `2*n`.

Negative/fail-closed evidence was revalidated by focused lit. The selected
filter passed 13/13 tests covering pre-realized deinterleave/interleave
generated-bundle dry-run, direct-pre-realized route-entry fail-closed behavior,
and explicit/pre-realized target artifact fixtures.

Additional checks:

* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-(deinterleave|interleave)|selected-body-artifact-segment2-(deinterleave|interleave)'`
  from `build/test` passed 13/13 selected tests.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `build/bin/tianchenrv-target-artifact-export-test` passed.

Spec update judgment: no `.trellis/spec/` change is needed in this round. The
durable plain segment2 provider/target/export contracts already exist in the
RVV plugin and EmitC route specs; this task refreshed executable artifact
evidence and did not discover a new contract, signature, validation matrix, or
reusable convention.
