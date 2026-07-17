# Stage2 RVV computed-masked segment2 update executable artifact ABI boundary

## Goal

Make the existing RVV computed-masked segment2 update selected-body route
family truthful at the executable artifact ABI boundary. The workflow must line
up typed `tcrv_rvv` segment2 body facts, computed mask/predicate facts,
masked active/inactive lane policy, field0/field1 update roles,
source/destination preservation, dtype/SEW/LMUL/config/policy, runtime AVL/VL,
per-operand ABI/header bindings, RVV plugin-owned route validation, common
EmitC materialization, target artifact export, generated bundle ABI, and
`ssh rvv` correctness evidence. If any required executable-boundary fact is
missing or stale, the route must fail closed with a targeted diagnostic instead
of claiming support.

## What I Already Know

* The previous task closed computed-masked segment2 load/store executable
  evidence for explicit and pre-realized selected bodies without production
  source changes.
* This task is not another load/store evidence refresh. The bounded owner is
  the computed-masked segment2 update path, because it exercises read/update/
  write behavior rather than only standalone segment2 load or store.
* The route family is `computed_masked_segment2_update_unit_load`.
* The spec requires update to use runtime ABI order
  `cmp_lhs,cmp_rhs,src0,src1,dst,n`, typed op `tcrv_rvv.binary`, update
  arithmetic kind `add`, provider-derived update arithmetic callee, field
  payload loads, tuple creation, and masked segment-store.
* Common EmitC/export may carry provider-built payload and mirrors unchanged,
  but must not infer segment2, computed-mask, update arithmetic, dtype/config,
  ABI, header, or inactive-lane semantics.
* Memory notes warn against evidence-only drift in TianChenRV RVV Stage 2; this
  task must either harden a real production seam or provide exact no-source
  justification plus missing executable evidence.

## Requirements

* Inspect the current production route, realization, statement-plan, target
  artifact validation, generated bundle script, and explicit/pre-realized
  update fixtures before changing source.
* Verify whether computed-masked segment2 update is dry-run-only,
  under-validated, stale, or already complete at the executable artifact ABI
  boundary.
* If the executable boundary is under-validated, harden the owner-local
  production seam where facts are consumed: update field roles, computed mask
  binding, inactive-lane policy, source/destination preservation, dtype/config,
  runtime AVL/VL, ABI/header order, generated C type, route-family validation,
  statement facts, and target artifact mirrors.
* If production code is already sufficient, record the exact no-source-change
  justification and close the missing executable evidence with focused
  generated-bundle and `ssh rvv` runs.
* Preserve architecture boundaries: update semantics stay in RVV
  plugin-owned selected-body realization, route-family planning, statement
  planning, and target validation; common EmitC/export remains neutral.
* Keep old-authority residue out of touched production and test lines: no route
  authority from legacy i32 helper names, route ids, artifact names,
  source-front-door artifacts, descriptor residue, common EmitC semantic
  branches, or exact intrinsic spelling.

## Acceptance Criteria

* [x] Explicit computed-masked segment2 update generated-bundle dry-run shows
      materialized selected boundary, emission plan, header/prototype, harness,
      computed mask, field0/field1 update roles, runtime count, inactive-lane
      policy, source/destination preservation, update arithmetic, tuple
      creation, masked segment-store, and tail checks.
* [x] Pre-realized computed-masked segment2 update generated-bundle dry-run
      covers the same executable boundary after selected-body realization.
* [x] If executable runtime behavior is claimed, explicit and pre-realized
      generated bundles compile and pass on `ssh rvv`.
* [x] At least one focused fail-closed case proves stale or missing update
      boundary facts are rejected, such as stale update arithmetic, stale
      field role, stale mask binding, stale inactive-lane policy, stale
      header/prototype binding, stale ABI order, wrong generated C type, or
      stale statement fact.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run lit tests or direct script equivalents
      pass if lit is unavailable.
* [x] Bounded old-authority scan over touched files and added diff lines is
      clean.
* [x] `git diff --check`, `git diff --cached --check`, and final `git status`
      are clean after commit.

## Definition of Done

* Production behavior is implemented or a precise no-source-change production
  justification is recorded.
* Focused tests and generated-bundle evidence cover the changed or validated
  executable boundary.
* Trellis task metadata, PRD notes, and workspace journal are truthful.
* Specs are reviewed for durable lessons; update `.trellis/spec/` only if this
  round discovers a reusable rule not already covered.
* The task is finished/archived and one coherent commit is created.

## Out of Scope

* Broad segment2 matrix expansion.
* Dtype/LMUL clone batches.
* Runtime-scalar-cmp segment2 expansion in this round.
* Additional computed-masked load/store rework except as reference.
* MAcc, reduction, dequant, clamp, high-level Linalg/Vector/StableHLO frontend,
  per-Linalg route authority, performance tuning database, dashboard, report-
  only work, source-front-door positive routes, or mass rewrite of unrelated
  RVV route families.

## Technical Notes

* Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-computed-masked-segment2-load-store-artifact-abi-boundary/`.
* Primary code paths to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.
* Primary fixtures/tests:
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-segment2-update-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-update-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-segment2-update-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-update.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`.

## Results

No production source change was required. The current RVV provider and target
artifact path already consumes the computed-masked segment2 update selected-body
facts through the expected owner-local boundaries:

* `RVVSegment2MemorySelectedBodyRealizationOwner` realizes the pre-realized
  update body through the public selected lowering-boundary materialization
  owner before provider facts are collected.
* `RVVEmitCSegment2RouteFamilyPlanOwners` selects the exact
  `computed-mask segment2 update` planning owner and requires the verified
  computed-mask memory plan, `usesSegment2Update`, segment count 2, structural
  add arithmetic facts, runtime ABI order, operand-binding summary, typed
  config, mask facts, field roles, header/type facts, and runtime AVL/VL
  facts before provider route construction.
* `verifyRVVSelectedBodySegment2MemoryRouteProviderFacts(...)` requires the
  owner-built segment2 provider plan, route-control provider plan, matching
  memory operand-binding facts, selected typed body/config facts, and migrated
  segment2 statement-plan leaves before `TCRVEmitCLowerableRoute` is accepted.
* `RVVTargetArtifactRouteFamilyValidation` rebuilds the provider route and
  rejects stale candidate mirrors for route id, provider support, ABI order,
  operand binding, header/type mapping, mask facts, field facts, update
  arithmetic, runtime AVL/VL, and statement facts before target artifact
  acceptance.
* `scripts/rvv_generated_bundle_abi_e2e.py` already generates update harnesses
  that snapshot source and destination inputs, require active and inactive mask
  lanes for multi-lane cases, check field0/field1 distinction, verify
  compare-false destination preservation, verify active lane
  `add(field0, field1)` into field0 and field1 payload propagation into field1,
  and check source/tail preservation before printing PASS.

The exact missing blocker was executable evidence for the update read/update/
write boundary. This round closed it with focused generated-bundle dry-run and
`ssh rvv` evidence rather than changing production code.

Positive dry-run evidence:

* Explicit update:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-update-explicit-dry-run/explicit-update/computed_masked_segment2_update_unit_load/evidence.json`
  reports `status: dry_run_success`, `dry_run: true`,
  `input_mode: explicit-selected-body`, and carries the update field,
  inactive-lane, and computed-mask coverage contracts.
* Pre-realized update:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-update-prerealized-dry-run/pre-update/computed_masked_segment2_update_unit_load/evidence.json`
  reports `status: dry_run_success`, `dry_run: true`,
  `input_mode: pre-realized-selected-body`, and carries the same update field,
  inactive-lane, and computed-mask coverage contracts after selected-body
  realization.

Positive `ssh rvv` evidence:

* Explicit update:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-update-explicit-ssh/explicit-update/computed_masked_segment2_update_unit_load/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Pre-realized update:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cmseg-update-prerealized-ssh/pre-update/computed_masked_segment2_update_unit_load/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.

Remote PASS summaries:

```text
PASS op=computed_masked_segment2_update_unit_load counts=0,1,7,16,23,257 patterns=0,1
PASS op=computed_masked_segment2_update_unit_load counts=0,1,7,16,23,257 patterns=0,1
```

The remote harness output covers runtime count zero, singleton, tail, exact
vector chunk, and large cases; two data patterns; compare-produced active and
inactive mask lanes; inactive destination preservation; field0/field1
distinction; source preservation; active update arithmetic; and tail sentinel
preservation.

Focused fail-closed evidence was revalidated by
`build/bin/tianchenrv-target-artifact-export-test` and the existing update
lit fixtures. Existing C++ coverage mutates stale update operation facts, route
id, mask binding, field role, source/destination memory form, operand binding
summary, ABI/header markers, inactive-lane contract, runtime ABI order,
provider mirror, target leaf, headers, C type mapping, rebuilt route header,
rebuilt route type mapping, ABI value mapping, runtime AVL/VL facts,
computed-mask plan, mask role/form, runtime ABI role, segment count, segment
layout, stale segment-load/field-extract facts, update arithmetic callee,
pre-loop/loop setvl operands, field payload load, compare/mask operand, update
arithmetic operand/result, masked segment-store pointer, loop source
provenance, and candidate metadata mirrors. The existing target fixtures also
reject stale route/provider/binding/ABI/header/type/mask/inactive/field/update
metadata mirrors before export.

Additional fail-closed script evidence:

```text
--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): computed_masked_segment2_update_unit_load
```

Checks run:

* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* Explicit computed-masked segment2 update generated-bundle dry-run using
  `build/bin/tcrv-opt`, `build/bin/tcrv-translate`, and
  `/usr/bin/llvm-readobj-20`.
* Pre-realized computed-masked segment2 update generated-bundle dry-run using
  the same tools.
* Explicit computed-masked segment2 update generated-bundle non-dry-run on
  `ssh rvv`.
* Pre-realized computed-masked segment2 update generated-bundle non-dry-run on
  `ssh rvv`.
* Deprecated direct pre-realized route-entry negative script check for update.
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`

The local machine does not expose `tcrv-opt`, `tcrv-translate`, or
`llvm-readobj` on `PATH`; direct generated-bundle script equivalents were run
with repository-built `build/bin/tcrv-opt`, `build/bin/tcrv-translate`, and
`/usr/bin/llvm-readobj-20`.

Specs were reviewed and did not need updates: the durable update ABI,
route-validation, selected-body realization, route-family planning, statement
plan, runtime AVL/VL, stale mirror, and fail-closed ownership rules are already
captured in `emitc-route.md` and `rvv-plugin.md`.
