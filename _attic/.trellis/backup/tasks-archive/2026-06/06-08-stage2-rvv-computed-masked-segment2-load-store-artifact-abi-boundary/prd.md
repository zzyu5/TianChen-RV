# Stage2 RVV computed-masked segment2 load/store executable artifact ABI boundary

## Goal

Make the existing RVV computed-masked segment2 load/store selected-body route
family truthful at the executable artifact ABI boundary. The workflow must line
up typed `tcrv_rvv` segment2 memory body facts, computed predicate facts,
inactive-lane policy, field0/field1 roles, dtype/config/policy, runtime AVL/VL,
per-operand ABI/header binding, RVV plugin-owned route validation, common EmitC
materialization, target artifact export, generated bundle ABI, and `ssh rvv`
correctness evidence. If any required executable-boundary fact is stale or
missing, the route must fail closed with a targeted diagnostic instead of
claiming support.

## What I Already Know

* The previous plain segment2 task closed unit interleave/deinterleave evidence
  for explicit and pre-realized selected bodies without production source
  changes.
* This task continues the segment2 memory-movement direction specifically for
  computed-mask segment2 load/store, because that exercises predicate and
  inactive-lane semantics on the same generated artifact ABI path.
* The starting scope is the load/store pair:
  `computed_masked_segment2_deinterleave_unit_store` and
  `computed_masked_segment2_interleave_unit_load`, including explicit selected
  bodies and pre-realized selected bodies.
* Computed-masked segment2 update and runtime-scalar-cmp masked segment2 routes
  are reference material only unless the base load/store boundary is already
  complete and the same coherent module can be closed in this round.

## Requirements

* Inspect the current production route, realization, statement, target artifact,
  and generated bundle harness code before changing source.
* Verify whether computed-masked segment2 load/store is merely dry-run covered
  or already has a complete production executable path.
* If the executable path is under-validated, harden the owner-local production
  boundary where facts are consumed: computed mask binding, inactive-lane
  policy, segment field mapping, dtype/config, runtime AVL/VL, ABI/header order,
  generated C type, and target artifact mirror contract.
* If production code is already sufficient, record the exact no-source-change
  justification and provide executable `ssh rvv` evidence through the generated
  bundle path.
* Preserve architecture boundaries: RVV semantics stay in RVV plugin-owned
  selected-body realization, route planning, statement planning, and target
  validation; common EmitC/export remains neutral.
* Keep old-authority residue out of touched production and test lines: no route
  authority from `i32m1` helper names, route ids, artifact names,
  source-front-door artifacts, descriptor residue, common EmitC semantic
  branches, or exact intrinsic spelling.

## Acceptance Criteria

* [x] Explicit computed-masked segment2 load generated bundle dry-run shows
      materialized selected boundary, emission plan, header/prototype, harness,
      computed mask, field0/field1 roles, runtime count, inactive-lane policy,
      and tail/source preservation checks.
* [x] Pre-realized computed-masked segment2 load generated bundle dry-run covers
      the same executable boundary after selected-body realization.
* [x] Explicit computed-masked segment2 store generated bundle dry-run covers
      computed predicate, field mapping, destination preservation, inactive
      lane semantics, ABI order, runtime count, and tail preservation checks.
* [x] Pre-realized computed-masked segment2 store generated bundle dry-run
      covers the same executable boundary after selected-body realization.
* [x] Non-dry-run generated bundle evidence on `ssh rvv` proves runtime
      correctness for computed-masked segment2 load and store when executable
      behavior is claimed.
* [x] At least one focused fail-closed case proves a stale or missing executable
      boundary fact is rejected, unless existing focused tests already cover the
      same load/store boundary and are cited in this PRD.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run lit tests pass.
* [x] Bounded old-authority scan over touched files and added diff lines is
      clean.
* [x] `git diff --check`, `git diff --cached --check`, and final `git status`
      are clean after commit.

## Definition of Done

* Production behavior is implemented or a precise no-source-change production
  justification is recorded.
* Focused tests and generated bundle evidence cover the changed or validated
  executable boundary.
* Trellis task metadata, PRD notes, and workspace journal are truthful.
* Specs are reviewed for new durable lessons; update `.trellis/spec/` only if
  this round discovers a reusable rule not already covered.
* The task is finished/archived and one coherent commit is created.

## Out of Scope

* Repeating plain segment2 evidence refresh as the main goal.
* Broad segment2 matrix expansion.
* Dtype/LMUL clone batches.
* Computed-masked segment2 update unless the base load/store boundary is already
  complete and fits the same coherent round.
* Runtime-scalar-cmp masked segment2 expansion.
* Unrelated memory, MAcc, reduction, compare-select, Linalg/Vector/StableHLO
  frontend, performance tuning database, dashboard, report-only, or
  source-front-door positive route work.

## Technical Notes

* Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-segment2-unit-artifact-abi-boundary/`.
* Primary code paths to inspect:
  `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.
* Primary test fixtures:
  `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-segment2-load-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-load-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-segment2-store-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-store-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-load.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-load.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-store.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-store.mlir`.

## Results

No production source change was required. The current RVV provider and target
artifact path already consumes the computed-mask segment2 selected-body facts
through the expected owner-local boundaries:

* `RVVSegment2MemorySelectedBodyRealizationOwner` routes pre-realized
  computed-mask segment2 load/store bodies through the public selected
  lowering-boundary materialization owner before provider facts are collected.
* `RVVEmitCRouteProvider` requires the matched segment2 route-family planning
  owner to build a provider plan before constructing
  `TCRVEmitCLowerableRoute`.
* Computed-mask segment2 route construction requires the verified
  computed-mask memory family plan, mask producer/source/form, inactive-lane
  contract, segment direction, segment count, runtime ABI parameters,
  operand-binding summary, header/type mapping, typed config, route-control
  plan, and migrated segment2 statement-plan leaves before route export.
* `RVVTargetArtifactRouteFamilyValidation` selects explicit
  computed-mask-segment2 load/store validators and revalidates provider route
  payload, runtime AVL/VL, ABI/header/type mapping, segment/mask/field facts,
  statement steps, and candidate mirrors after rebuilding the provider route.
* `scripts/rvv_generated_bundle_abi_e2e.py` already generates harnesses that
  snapshot source/destination inputs, require active and inactive mask lanes
  for multi-lane cases, check field0/field1 distinction, verify false-lane
  preservation, and check source/tail preservation before printing PASS.

Positive dry-run evidence:

* Explicit load:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-load-dry-run/explicit-load/computed_masked_segment2_load_unit_store/evidence.json`
* Pre-realized load:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-load-prerealized-dry-run/pre-load/computed_masked_segment2_load_unit_store/evidence.json`
* Explicit store:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-store-dry-run/explicit-store/computed_masked_segment2_store_unit_load/evidence.json`
* Pre-realized store:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-store-prerealized-dry-run/pre-store/computed_masked_segment2_store_unit_load/evidence.json`

Positive `ssh rvv` evidence:

* Explicit computed-mask segment2 load:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-load-ssh/explicit-load/computed_masked_segment2_load_unit_store/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Pre-realized computed-mask segment2 load:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-load-prerealized-ssh/pre-load/computed_masked_segment2_load_unit_store/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Explicit computed-mask segment2 store:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-store-ssh/explicit-store/computed_masked_segment2_store_unit_load/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Pre-realized computed-mask segment2 store:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-store-prerealized-ssh/pre-store/computed_masked_segment2_store_unit_load/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.

Remote PASS summaries:

```text
PASS op=computed_masked_segment2_load_unit_store counts=0,1,16,17,257 patterns=0,1
PASS op=computed_masked_segment2_load_unit_store counts=0,1,7,16,23,257 patterns=0,1
PASS op=computed_masked_segment2_store_unit_load counts=0,1,16,17,257 patterns=0,1
PASS op=computed_masked_segment2_store_unit_load counts=0,1,7,16,23,257 patterns=0,1
```

The remote harness output covers runtime count zero, singleton, exact vector
chunk, tail, and large cases; two data patterns; compare-produced active and
inactive mask lanes; field0/field1 distinction; source preservation; inactive
lane preservation; and destination/source tail sentinel preservation.

Focused fail-closed evidence was revalidated by
`build/bin/tianchenrv-rvv-extension-plugin-test`. Existing C++ coverage includes
computed-mask segment2 stale route-id, stale provider-supported mirror, stale
ABI order, stale operand-binding, stale inactive policy, stale statement-plan,
stale direction, stale memory-form, stale policy, stale analysis, and
route-control failures around the provider and materialization boundary.

Checks run:

* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* Four generated-bundle dry-runs for explicit/pre-realized load/store using
  `--llvm-readobj /usr/bin/llvm-readobj-20`.
* Four non-dry-run generated-bundle runs on `ssh rvv` for explicit/pre-realized
  load/store.
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* Four focused MLIR emission-plan commands for explicit/pre-realized
  computed-masked segment2 load/store fixtures.

The local machine does not currently expose `llvm-lit` or Python `lit`; the
same generated-bundle dry-run behavior was executed directly through the
evidence script with repository-built `tcrv-opt`/`tcrv-translate`.

Final hygiene:

* Specs were reviewed and did not need an update: the durable computed-mask
  segment2 ABI, route-validation, runtime AVL/VL, and fail-closed ownership
  rules are already captured in `emitc-route.md` and `rvv-plugin.md`.
* Bounded old-authority scanning over this Trellis closure found no new
  production or test-line route authority from legacy i32m1 helpers, route ids,
  descriptors, artifact names, or common EmitC semantic branches.
* `git diff --check` and `git diff --cached --check` passed before commit; the
  post-commit worktree status is reported in the final handoff.
