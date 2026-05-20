# Journal - codex (Part 11)

> Continuation from `journal-10.md` (archived at ~2000 lines)
> Started: 2026-05-19

---

## Session 143: Stage2 RVV route-provider planning extraction

**Date**: 2026-05-20
**Task**: `stage2-rvv-route-provider-planning-extraction`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and extracted the
active realized typed `tcrv_rvv` body -> route-plan boundary out of the
monolithic RVV EmitC route provider. The RVV provider still owns public route
authority and `TCRVEmitCLowerableRoute` construction, but route slice
collection, typed fact validation, fail-closed planning diagnostics, route
description/header/type/intrinsic planning, config artifact metadata, and route
description verification now live in `RVVEmitCRoutePlanning`.

### Main Changes

- Added `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` and
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
- Moved the current supported realized-route planning for generic add/sub/mul,
  compare/select, RHS broadcast arithmetic, strided add, masked add, macc add,
  and reduce add out of `RVVEmitCRouteProvider.cpp`.
- Kept `RVVEmitCRouteProvider.cpp` as the plan consumer that materializes
  provider-owned route payloads into `TCRVEmitCLowerableRoute`.
- Left common EmitC/export unchanged and did not add operation coverage,
  dtype/LMUL expansion, source-front-door positive routing, descriptor/direct-C
  behavior, runtime behavior, correctness claims, or performance claims.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV dialect/provider,
  construction protocol, and target artifact export tests.
- [OK] Direct C++ checks:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-target-artifact-export-test`, and
  `tianchenrv-rvv-dialect-test`.
- [OK] Focused provider/materialization/selected-body lit: 21/21 passed for
  unit-stride binary, strided, masked, macc, reduce, pre-realized
  selected-body positive fixtures, and negative fail-closed fixtures.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Focused generated-bundle dry-run lit: 6/6 passed for explicit
  selected-body and pre-realized unit-stride/strided/masked/macc/reduce paths.
- [OK] `git diff --check`
- [OK] Active-authority scan: only provider-derived intrinsic leaves, an
  existing fail-closed `tcrv_rvv.i32_` rejection branch, and PRD guardrail text
  matched; no positive legacy/source-front-door/descriptor/common-export route
  authority was introduced.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 176/176 lit tests
  passed.

### Status

[OK] **Completed and ready to archive**. This round makes no new runtime,
correctness, or performance claim, so no fresh `ssh rvv` run was required.

### Next Steps

- Future continuation, if requested: keep any new route-family expansion inside
  the extracted planning owner plus provider materialization consumer, and do
  not put body-specific route-planning tables back into
  `RVVEmitCRouteProvider.cpp`.


## Session 142: Stage2 RVV selected-body realization for reduce_add

**Date**: 2026-05-19
**Task**: `stage2-rvv-selected-reduce-add-realization`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented one
bounded pre-realized selected-body realization path for `reduce_add`. The RVV
plugin now consumes explicit reduction operation, accumulator seed role/layout,
result layout, lhs/rhs/out ABI values, runtime `n`/AVL, dtype/config, memory
form, and policy facts into realized typed `tcrv_rvv` structure before the
existing reduce-add provider route constructs the lowerable EmitC route.

### Main Changes

- Added `tcrv_rvv.typed_reduce_pre_realized_body` with verifier checks for
  bounded `reduce_add`, RHS input-buffer accumulator role, reduction
  accumulator/result layout, SEW32/LMUL m1, agnostic policy, runtime ABI roles,
  and stale authority metadata.
- Extended RVV plugin selected-body realization to materialize the
  pre-realized reduce body into `setvl/with_vl/load/load/reduce/store`, then
  reuse the existing provider/materializer/target artifact path.
- Extended generated-bundle evidence tooling with
  `--pre-realized-selected-body --op-kind reduce_add`.
- Added positive pre-realized reduce target/script fixtures and reduce-specific
  fail-closed negative coverage.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV dialect/plugin,
  construction protocol, and target artifact export tests.
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Focused lit set covering new pre-realized reduce-add
  positive/negative/script evidence, explicit reduce-add, existing
  pre-realized add/sub/mul/strided/masked/macc fixtures, and existing
  pre-realized negatives: 11/11 passed.
- [OK] Local pre-realized reduce-add generated-bundle dry-run at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-pre-realized-reduce-add-dry-run`.
- [OK] Real `ssh rvv` reduce-add correctness evidence at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-pre-realized-reduce-add-evidence`:
  `PASS op=reduce_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 176/176 passed.
- [OK] `git diff --check`
- [OK] diff active-authority scan introduced no positive legacy RVV route
  authority; new matches were anti-authority text and negative FileCheck
  guardrails.

### Self-Repair

- Re-ran lit from `build/test` after an initial root invocation failed to
  resolve the generated lit site config relative path.
- Repaired the new script dry-run FileCheck expectation to match the generated
  reduce-add harness loop.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for bounded pre-realized `reduce_add` counts
`7,16,23`.

### Next Steps

- Future continuation, if requested: broaden reduction Stage2 coverage beyond
  this single selected-boundary reduce_add handoff under a separate bounded
  task.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 145: Stage2 RVV strided memory movement executable slice

**Date**: 2026-05-20
**Task**: `stage2-rvv-strided-memory-slice`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented one
bounded Stage2 RVV i32 / SEW32 / LMUL m1 strided memory-movement executable
slice. The production route now carries `src`, `out`, `n`, and `src_stride`
through the selected `tcrv.exec` RVV boundary, typed/pre-realized `tcrv_rvv`
body, RVV selected-body realization, route planning/provider metadata,
generated bundle emission, and real `ssh rvv` correctness evidence.

### Main Changes

- Added `tcrv_rvv.typed_strided_memory_pre_realized_body` and `tcrv_rvv.move`
  for the bounded `strided_load_unit_store` slice.
- Added verifier coverage for source/output/runtime-count/runtime-stride roles,
  memory form, stride unit, SEW32, LMUL m1, policy, selected-variant nesting,
  and generic vector type consistency.
- Extended RVV selected-body realization to materialize the pre-realized body
  into `setvl`, `with_vl`, `strided_load`, `move`, and unit-stride `store`.
- Extended RVV route planning/provider/construction facts to derive ABI order
  `src,out,n,src_stride`, memory-form metadata, vector C types, and strided-load
  materialization from typed facts.
- Extended the generated-bundle script and added explicit/pre-realized artifact
  tests and negative fail-closed MLIR tests.

### Testing

- [OK] Trellis task context validation.
- [OK] focused build for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] focused C++ RVV dialect/plugin/construction/export tests.
- [OK] focused lit for generic dataflow, pre-realized strided memory negative,
  EmitC strided memory negative, selected-body materialization regressions, and
  explicit/pre-realized strided target artifacts: 12/12 passed.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] generated-bundle dry-run for explicit and pre-realized
  `strided_load_unit_store` counts `7,16,23`.
- [OK] real `ssh rvv` generated-bundle correctness for explicit and
  pre-realized `strided_load_unit_store`: `PASS op=strided_load_unit_store
  counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`
- [OK] `git diff --check`
- [OK] active-authority scan found no positive RVVI32M1, rvv-i32m1,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, or exact i32m1 intrinsic authority in
  added diff hunks. The only broad-scan matches are guardrail text saying those
  surfaces are not route authority.

### Self-Repair

- Fixed construction protocol role realization after the new structural
  `tcrv_rvv.move` op initially made the protocol invalid.
- Fixed route typed-config validation so the memory-movement route does not
  read a nonexistent RHS vector.
- Fixed generated-bundle selected-body checks to distinguish real op
  invocations from construction metadata that lists unsupported capability
  names.

### Status

[OK] **Completed and ready to archive**. This round claims route-supported,
generated-bundle dry-run, and executable correctness evidence for this bounded
memory-movement slice only. No performance claim is made.

### Next Steps

- Future continuation, if requested: open a separate Trellis task for the next
  Stage2 memory class. Do not generalize this slice into gather/scatter,
  segmented memory, high-level transfer lowering, or dtype/LMUL clone batches
  without a new PRD.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 145: Stage2 RVV macc executable slice

**Date**: 2026-05-20
**Task**: `05-20-stage2-rvv-macc-executable-slice`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and completed one
bounded Stage2 signed i32 / SEW32 / LMUL m1 / unit-stride RVV
multiply-accumulate executable slice. The production gap was that
`typed_macc_pre_realized_body` already carried accumulator/result layout facts,
but realized `tcrv_rvv.macc` did not; route planning filled macc layout mirrors
from constants. This round makes those layout facts structural in the realized
typed body and provider-derived route metadata.

### Main Changes

- Added structural `accumulator_layout` and `result_layout` attrs to generic
  `tcrv_rvv.macc`.
- Updated `MAccOp::verify()` to reject missing or unsupported macc layout
  facts with targeted diagnostics.
- Updated RVV selected-body realization so pre-realized macc copies its
  accumulator/result layout facts into realized `tcrv_rvv.macc`.
- Updated RVVEmitCRoutePlanning so macc metadata mirrors are derived from the
  realized typed body instead of unconditional constants.
- Updated generated-bundle evidence checks, explicit/pre-realized macc MLIR
  fixtures, EmitC materialization coverage, and target artifact C++ fixtures.
- Added negative dialect coverage for missing/unsupported macc accumulator and
  result layout facts.

### Testing

- [OK] Trellis task context validation.
- [OK] focused build for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] focused C++ RVV dialect/plugin/construction/export tests.
- [OK] focused lit: 8/8 passed for macc dataflow, EmitC materialization,
  target artifacts, pre-realized negatives, and dry-run fixtures.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] local explicit and pre-realized `macc_add` generated-bundle dry-runs for
  counts `7,16,23`.
- [OK] real `ssh rvv` explicit macc correctness:
  `PASS op=macc_add counts=7,16,23`.
- [OK] real `ssh rvv` pre-realized macc correctness:
  `PASS op=macc_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 193/193 passed.
- [OK] `git diff --check`
- [OK] added-authority scan found no forbidden legacy/source/descriptor/common
  RVV authority in diff-added hunks.

### Self-Repair

- The first broad text replacement for macc layout attrs also touched unrelated
  `binary`/`masked_binary` add fixtures. Those accidental attrs were removed
  before validation; focused lit and full `check-tianchenrv` passed afterward.

### Status

[OK] **Completed and ready to archive**. This round claims executable
correctness only for bounded `macc_add` counts `7,16,23`; no performance claim
is made.

### Next Steps

- No continuation is required for this bounded slice. Future Stage2 work should
  open a separate PRD and stay at low-level RVV typed-body capability classes.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 145: Stage2 RVV reduce-add production executable slice

**Date**: 2026-05-20
**Task**: `stage2-rvv-reduce-add-production-executable-slice`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes redirect brief and moved the
reduce-add work out of metadata-only closeout mode. The production gap in
current HEAD was that pre-realized reduce-add carried accumulator/result layout
facts, but realization erased them into a `tcrv_rvv.reduce` with only
`kind = "add"`; route planning then filled layout mirrors from bounded
operation constants. This round made accumulator/result layout structural on
the realized typed `tcrv_rvv.reduce` body consumed by the RVV provider.

### Main Changes

- Added structural `accumulator_layout` and `result_layout` attrs to generic
  `tcrv_rvv.reduce`.
- Tightened `ReduceOp::verify()` to fail closed for missing or unsupported
  reduce accumulator/result layout facts.
- Updated RVV selected-body realization so
  `tcrv_rvv.typed_reduce_pre_realized_body` consumes its layout facts into the
  realized `tcrv_rvv.reduce` before provider route construction.
- Updated RVV EmitC route planning to validate and read layout facts from the
  typed reduce op, then populate provider-derived reduction metadata mirrors
  from those facts.
- Updated generated-bundle evidence tooling to assert materialized reduce-add
  MLIR contains the structural layout attrs.
- Updated positive explicit/pre-realized reduce-add tests and C++ target
  artifact fixture generation; added negative dialect coverage for missing and
  unsupported layout facts.

### Testing

- [OK] Trellis task context validation.
- [OK] focused build for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] focused C++ RVV dialect/plugin/construction/export tests.
- [OK] focused lit 6/6 for generic dataflow, reduce-add materialization,
  reduce-add negatives, explicit/pre-realized target artifact tests, and
  pre-realized generated-bundle dry-run.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260520T-typed-reduce-layout-dry`.
- [OK] real `ssh rvv` generated-bundle correctness:
  `PASS op=reduce_add counts=7,16,23` with evidence root
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260520T-typed-reduce-layout-ssh`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 193/193 passed.
- [OK] `git diff --check`
- [OK] diff-only active-authority scan found no positive legacy/source/
  descriptor/common-export authority reintroduced in active RVV code/tests.

### Self-Repair

- The first lit invocation from the repository root could not resolve the
  generated site config relative path. Re-ran focused lit from `build/test`,
  which is the repository's lit working directory convention.

### Spec Update Decision

No `.trellis/spec/` update was needed. Existing specs already require Stage2
RVV reduction/accumulator/result facts to be structural in typed or realized
`tcrv_rvv` body structure and require common EmitC/export neutrality.

### Status

[OK] **Completed and ready to archive**. This round claims executable
correctness only for bounded pre-realized `reduce_add` counts `7,16,23`; no
performance claim is made.

### Next Steps

- Future continuation, if requested: broaden reduction classes under a new
  PRD only after preserving this typed-body layout authority, without adding
  dtype-prefixed helper ops or route-id/artifact-name authority.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 146: Stage2 RVV reduce-add executable slice closure

**Date**: 2026-05-20
**Task**: `05-20-stage2-rvv-reduce-add-executable-slice`
**Branch**: `main`

### Summary

Created a Trellis task from the Hermes Direction Brief and checked it against
current HEAD. The brief was stale with respect to current code: the bounded
Stage2 pre-realized `reduce_add` executable path already exists in the active
RVV plugin route. This round did not change compiler source; it wrote the PRD,
curated context, validated the existing production path, collected fresh
`ssh rvv` correctness evidence for counts `7,16,23`, and recorded the closure.

### Main Findings

- Current HEAD already has `tcrv_rvv.typed_reduce_pre_realized_body`,
  `tcrv_rvv.reduce`, RVV selected-body realization, RVVEmitCRoutePlanning,
  target artifact metadata, generated-bundle dry-run coverage, and
  `scripts/rvv_generated_bundle_abi_e2e.py --op-kind reduce_add`.
- The pre-realized selected body carries explicit input, accumulator seed,
  result/output, SEW32, LMUL m1, agnostic policy, unit-stride memory form,
  runtime `n`/AVL, accumulator role/layout, result layout, and ABI roles.
- RVV selected-body realization consumes the pre-realized boundary into
  `setvl`, `with_vl`, input load, accumulator load,
  `tcrv_rvv.reduce {kind = "add"}`, and store before provider planning.
- RVVEmitCRoutePlanning/provider derive the bounded reduction layout mirrors:
  `rhs-vector-seed-lane0-per-vl-chunk`,
  `store-reduction-lane0-to-output-chunk-base`, and store VL `1`.

### Testing

- [OK] Trellis task context validation.
- [OK] focused build for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] focused C++ RVV dialect/plugin/construction/export tests.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] generated-bundle dry-run for pre-realized `reduce_add` counts
  `7,16,23`.
- [OK] focused lit from `build/test`: 7/7 passed for reduce-add realization,
  explicit/pre-realized target artifacts, reduction negatives, materialization,
  generic dataflow, and generated-bundle dry-run.
- [OK] real `ssh rvv` generated-bundle correctness:
  `PASS op=reduce_add counts=7,16,23`.
- [OK] `git diff --check`
- [OK] active-authority scan found no source/code reintroduction in this task;
  broader matches remain pre-existing deprecated parse-only inventory,
  negative fail-closed tests, source-front-door rejection code, harness
  guardrails, or provider-derived intrinsic leaves.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 193/193 passed.

### Self-Repair

- None needed. Focused and full validation passed without source edits.

### Status

[OK] **Completed and ready to archive**. This round makes a fresh executable
correctness claim only for bounded pre-realized `reduce_add` counts `7,16,23`;
it makes no performance claim.

### Next Steps

- No continuation is needed for this bounded reduce-add executable slice.
- Future reduction work, if requested, should be a separate bounded Stage2
  coverage task such as another reduction kind/layout, not a replay of this
  reduce_add closure.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 145: Stage2 RVV widening conversion executable slice

**Date**: 2026-05-20
**Task**: `stage2-rvv-widening-conversion`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented one
bounded Stage2 signed widening conversion executable slice. The new path
imports `lhs: const int32_t *`, `out: int64_t *`, and runtime `n`, realizes a
typed source i32/m1 to destination i64/m2 `tcrv_rvv.widening_convert` body, and
routes it through RVV plugin-owned route planning/provider code into the common
EmitC materializer.

### Main Changes

- Added a bounded `typed_widening_conversion_pre_realized_body` hook and
  `tcrv_rvv.widening_convert` generic typed op with source/destination config,
  policy, memory form, conversion relation, and runtime role verification.
- Extended selected-body realization to materialize i32/m1 load,
  i64/m2 setvl/with_vl, widening conversion, and i64/m2 store only for the
  signed i32-to-i64 unit-stride tuple.
- Extended RVVEmitCRoutePlanning/provider to derive source/destination ABI
  types, vector C types, load/store leaves, conversion intrinsic leaf,
  metadata, runtime ABI order `lhs,out,n`, and artifact/header evidence from
  typed config facts.
- Kept non-conversion SEW64/m2 binary routes fail-closed after validation
  exposed an accidental i64m2 add materialization path.
- Added generated-bundle dry-run and real RVV harness support for
  `widen_i32_to_i64`.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused `tcrv-opt`/`FileCheck` checks for the new widening dataflow,
  pre-realized realization, emission plan, header artifact, fail-closed
  diagnostics, and script dry-run evidence.
- [OK] generated-bundle dry-run for `widen_i32_to_i64` counts `7,16,23`.
- [OK] real `ssh rvv` generated-bundle correctness:
  `PASS op=widen_i32_to_i64 counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2` (`193/193`).
- [OK] `git diff --check`
- [OK] active-authority scan found no positive RVVI32M1/rvv-i32m1/tcrv_rvv.i32_
  route authority added; new legacy hits are a negative rejection fixture or
  provider-derived i32/m1 source-load leaves for the typed conversion source.

### Self-Repair

- Fixed a conversion route-planning crash caused by assuming an RHS ABI value
  exists in role ordering.
- Added `route_id` to forbidden pre-realized authority metadata so stale route
  authority fails with a targeted diagnostic.
- Extended header artifact metadata handling with optional conversion-specific
  provenance instead of forcing all non-conversion routes to carry conversion
  facts.
- Restored fail-closed behavior for non-conversion SEW64/m2 binary routes after
  check-tianchenrv exposed an accidental i64m2 add materialization.

### Status

[OK] **Completed and ready to archive**. This round claims executable
correctness only for the bounded signed `widen_i32_to_i64` counts `7,16,23`;
no performance claim is made.

### Next Steps

- Future continuation, if requested: add the next bounded Stage2 conversion
  class under a separate PRD, without turning this slice into a dtype/LMUL
  matrix or one-intrinsic wrapper surface.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 145: Stage2 RVV tail/mask policy executable slice

**Date**: 2026-05-20
**Task**: `stage2-rvv-tail-mask-policy-executable-slice`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and completed one
bounded Stage2 policy-authority slice for `masked_add`: i32 / SEW32 / LMUL m1 /
unit-stride, with explicit mask role, mask source, tail/mask policy,
passthrough inactive-lane contract, typed body/config facts, runtime `n`/AVL,
provider route planning, generated artifact metadata, and real `ssh rvv`
correctness evidence.

### Main Changes

- Added provider-derived `maskRole` and `inactiveLaneContract` to
  `RVVSelectedBodyEmitCRouteDescription`.
- Made `RVVEmitCRoutePlanning` derive and verify the bounded masked-add policy
  facts:
  `predicate-mask-produced-by-compare`,
  `compare-produced-mask-same-vl-scope`,
  `masked-off-lanes-preserve-passthrough-vector`, and
  `passthrough-vector-preserves-inactive-lanes`.
- Mirrored the new policy facts into target artifact metadata as explicit RVV
  mirrors, without moving semantic authority into common EmitC/export.
- Strengthened generated-bundle `masked_add` evidence so the external C ABI
  harness requires true and false mask lanes, checks masked-off lanes preserve
  the passthrough vector, and records the inactive-lane and coverage contract.
- Added provider-level C++ coverage for the masked-add route description, the
  compare/add/merge lowerable route steps, and stale `maskRole` /
  `inactiveLaneContract` fail-closed verification.
- Updated masked-add artifact and generated-bundle dry-run lit checks for the
  new policy metadata and harness coverage fields.

### Testing

- [OK] Trellis task context validation.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused build for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] focused C++ RVV dialect/plugin/construction/export tests.
- [OK] focused lit for masked-add generated-bundle dry-runs, selected-body
  artifact metadata, and pre-realized negative coverage.
- [OK] generated-bundle dry-run for `masked_add` counts `7,16,23`.
- [OK] real `ssh rvv` generated-bundle correctness:
  `PASS op=masked_add counts=7,16,23`.
- [OK] remote lane coverage:
  `n=7 true=2 false=5 passthrough=5`,
  `n=16 true=4 false=12 passthrough=12`,
  `n=23 true=6 false=17 passthrough=17`.
- [OK] `cmake --build build --target check-tianchenrv -j2`
  passed `189/189`.
- [OK] `git diff --check`
- [OK] diff-added active-authority scan found no positive reintroduction of
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor/direct-C/
  source-export, or common/export RVV semantic authority.

### Self-Repair

No self-repair was needed after the final implementation pass. `clang-format`
was unavailable in this environment, so compile checks and `git diff --check`
were used for local formatting/whitespace validation.

### Status

[OK] **Completed and ready to archive**. This round claims executable
correctness only for the bounded `masked_add` counts `7,16,23`; no performance
claim is made and no policy matrix expansion was attempted.

### Next Steps

- Future continuation, if requested: add another bounded Stage2 tail/mask policy
  slice or widen policy support under a separate PRD, preserving RVV
  plugin-local typed body authority and fail-closed unsupported combinations.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 143: Stage2 RVV i64 SEW64 executable route slice

**Date**: 2026-05-20
**Task**: `stage2-rvv-non-i32-dtype-sew-route-policy`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented one
bounded non-i32 RVV typed route instance: pre-realized selected-body
`i64`/SEW64/LMUL m1 unit-stride add now realizes to generic RVV body structure,
routes through RVVEmitCRoutePlanning, emits through common EmitC, exports a
generated target artifact, and executes on real RVV hardware.

### Main Changes

- Added SEW64/LMUL m1 selected-body config contract and i64 runtime ABI
  parameter contract.
- Extended generic RVV verification and selected-body realization so i64 add is
  an ordinary typed route instance, not a dtype-prefixed op family.
- Extended RVV route planning to derive int64 ABI/header types, SEW64 metadata,
  `vint64m1_t`, and e64/i64 intrinsic leaves from typed route facts.
- Added positive target/materialization/generated-bundle coverage plus
  fail-closed i64 route-policy negative cases.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Focused lit for i64 materialization, negative route-policy cases,
  target/header export, script dry-run, setvl, and with-vl: 6/6 passed.
- [OK] Focused C++ tests:
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-target-artifact-export-test`.
- [OK] Generated-bundle dry-run for `--op-kind i64_add`.
- [OK] Real `ssh rvv` evidence for counts `7,16,23`:
  `PASS op=i64_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 187/187 passed.
- [OK] Active-authority scan found no positive added/new-file hits for legacy
  i32/source/descriptor/common-export route authority.
- [OK] `git diff --check`.

### Status

[OK] **Completed and ready to archive**. This round makes a correctness/runtime
claim only for bounded `i64_add` counts `7,16,23`; no performance claim is made.

### Next Steps

- Future continuation, if requested: broaden non-i32 RVV coverage under a new
  bounded PRD, still deriving dtype/SEW/LMUL/policy from typed body/config
  facts rather than route labels or artifact names.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 141: Stage2 RVV vector-scalar broadcast executable path

**Date**: 2026-05-20
**Task**: `stage2-rvv-vector-scalar-broadcast-executable-path`
**Branch**: `main`

### Summary

Implemented one bounded Stage2 RVV vector-scalar executable path:
`lhs` vector load plus explicit runtime `rhs_scalar` ABI value, realized through
generic `tcrv_rvv.splat`, generic binary add, provider-built EmitC route,
target artifact bundle, and `ssh rvv` correctness evidence.

### Main Changes

- Added `rhs-scalar-value` as an explicit runtime ABI role with scalar i32
  type validation and the ordered ABI contract `lhs,rhs_scalar,out,n`.
- Added generic `tcrv_rvv.splat` as the typed RVV scalar-to-vector dataflow
  op; it is verifier-guarded to consume an explicit RHS scalar runtime ABI
  value inside the matching `with_vl` scope.
- Extended pre-realized selected-body validation and realization so
  `memory_form = "rhs-scalar-broadcast"` realizes to
  `setvl/with_vl/load/splat/binary/store`.
- Extended RVV route planning, construction metadata, and EmitC materialization
  for one `scalar_broadcast_add` route while preserving common EmitC/export
  neutrality.
- Added target, conversion, dialect, lowering-boundary, script dry-run, and
  generated-bundle evidence coverage for the new path and its fail-closed
  cases.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused `tcrv-opt`/`FileCheck` checks for the new pre-realized
  scalar-broadcast target fixture, materialization fixture, dialect dataflow,
  conversion negatives, and lowering-boundary negatives.
- [OK] generated-bundle dry-run for `scalar_broadcast_add` counts `7,16,23`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-scalar-broadcast-dryrun`.
- [OK] real `ssh rvv` generated-bundle correctness for `scalar_broadcast_add`
  counts `7,16,23`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-scalar-broadcast-rvv`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 183/183 lit
  tests passed after updating stale construction/plugin test expectations.
- [OK] `git diff --check`
- [OK] added-line authority scan found no new `RVVI32M1`, `rvv-i32m1`,
  finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor, or legacy route-id authority.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for bounded `scalar_broadcast_add` counts `7,16,23`.

### Next Steps

- Future Stage2 coverage should stay in low-level RVV capability classes and
  use a separate bounded PRD; this scalar-broadcast route must not be expanded
  by adding dtype-prefixed helper ops or route-id/metadata authority.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 141: Stage2 RVV compare/select executable ABI closure

**Date**: 2026-05-20
**Task**: `stage2-rvv-cmp-select-executable-abi`
**Branch**: `main`

### Summary

Closed executable ABI evidence for the already route-supported pre-realized
RVV `cmp_select` selected-body path. The existing generated bundle compiled
and ran on `ssh rvv`; the required code change was limited to the evidence
harness so compare/select input now covers both predicate-true and
predicate-false lanes for multi-lane runtime counts.

### Main Changes

- Created and archived the bounded Trellis task from the Hermes Direction
  Brief.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so `cmp_select` generated
  inputs include equality lanes and non-equality lanes.
- Added `cmp_select` harness checks for `predicate_true_lanes` and
  `predicate_false_lanes`, requiring both for multi-lane cases while keeping
  expected output tied to the explicit `eq` predicate and
  `select-lhs-when-mask-else-rhs` semantics.
- Added dry-run lit assertions for the predicate coverage contract and
  generated harness checks.
- No C++/MLIR dialect, selected-body realization, route provider, common
  EmitC/export, or target artifact exporter code changed.

### Testing

- [OK] Trellis task context validation.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] pre-realized `cmp_select` dry-run counts `7,16,23`.
- [OK] focused dry-run lit: 1/1 passed.
- [OK] real `ssh rvv` evidence counts `7,16,23`:
  `PASS op=cmp_select counts=7,16,23`.
- [OK] remote predicate coverage:
  `n=7 true=2 false=5`, `n=16 true=4 false=12`,
  `n=23 true=6 false=17`.
- [OK] `git diff --check`
- [OK] diff-only active-authority scan found no added legacy/source/descriptor
  or common/export RVV semantic authority.

### Status

[OK] **Completed and archived**. This round makes a real RVV correctness claim
only for bounded pre-realized `cmp_select` counts `7,16,23`.

### Next Steps

- No continuation is required for compare/select executable ABI closure. Future
  Stage2 work should pick a separate low-level RVV capability class or
  coverage slice under a new bounded task.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 142: Stage2 RVV compare/select selected-body route path

**Date**: 2026-05-20
**Task**: `stage2-rvv-compare-select-route`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented one
bounded Stage2 compare/select selected-body realization path on the corrected
generic typed RVV surface. The new pre-realized body carries explicit
`cmp_select` / `eq` / mask-source / select-layout / runtime ABI facts, then
the RVV plugin realizes it into generic `setvl`, `with_vl`, `load`, `compare`,
`select`, and `store` structure before provider route construction.

### Main Changes

- Added `tcrv_rvv.typed_compare_select_pre_realized_body` with verifier
  checks for op kind, predicate kind, memory form, mask source, select layout,
  SEW/LMUL/policy, and lhs/rhs/out/n runtime ABI roles.
- Extended `RVVSelectedBodyRealization` to consume that pre-realized body and
  feed the existing generic compare/select route-planning/provider path.
- Added positive target artifact lit coverage and negative fail-closed
  coverage for predicate/layout/mask/memory/config/policy/runtime-role/mixed
  body failures.
- Extended `rvv_generated_bundle_abi_e2e.py` plus a dry-run lit fixture for
  `--pre-realized-selected-body --op-kind cmp_select`.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV dialect/plugin,
  construction-protocol, and target artifact test targets.
- [OK] RVV dialect/plugin/construction-protocol/target-artifact C++ tests.
- [OK] Focused lit for new compare/select positive, negative, and script
  dry-run tests: 3/3 passed.
- [OK] Focused lit regression for explicit compare/select, existing
  pre-realized add/sub/mul/strided/masked/reduce/macc, and provider negatives:
  14/14 passed.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] local generated-bundle dry-run in
  `/tmp/tianchenrv-pre-realized-cmp-select/dryrun`.
- [OK] `git diff --check`
- [OK] active-authority scan found no new positive legacy/source/descriptor or
  common/export RVV authority.
- [OK] `check-tianchenrv`: 179/179 passed.

### Status

[OK] **Completed and ready to archive**. This round makes no new runtime,
correctness, or performance claim, so no fresh `ssh rvv` evidence was required.

### Next Steps

- Future continuation, if requested: broader predicate classes or compare
  kinds should be a separate Stage2 PRD and must remain on the generic typed
  RVV body surface.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 141: Stage2 RVV selected-body realization module extraction

**Date**: 2026-05-20
**Task**: `stage2-rvv-selected-body-realization-extraction`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and extracted the
active RVV selected-body realization owner out of the monolithic
`RVVExtensionPlugin.cpp` implementation. The RVV plugin still owns the selected
pre-realized body -> realized typed `tcrv_rvv` body boundary, but the body
discovery, fact validation, fail-closed diagnostics, and realized-body
construction now live in `RVVSelectedBodyRealization`.

### Main Changes

- Added a focused Trellis PRD and curated implement/check context for the
  selected-body realization extraction.
- Added `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h` and
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
- Moved realization decisions for generic unit-stride, strided, masked, macc,
  and reduce pre-realized selected bodies into the new RVV plugin-local module.
- Kept `RVVExtensionPlugin::materializeSelectedLoweringBoundary` as the
  orchestration consumer: legality, realized-boundary lookup, extracted
  realization call, validation, and materialized-boundary result.
- Left provider route authority and common EmitC/export behavior unchanged.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV dialect/plugin,
  construction protocol, and target artifact export tests.
- [OK] Direct C++ checks:
  `tianchenrv-rvv-dialect-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] Focused lit: 12/12 passed for pre-realized add/strided/masked/macc/reduce
  positive fixtures, selected-body negatives, and representative
  provider/materializer fixtures.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Focused generated-bundle lit dry-runs: 5/5 passed for pre-realized
  unit-stride, strided, masked, macc, and reduce paths.
- [OK] `git diff --check`
- [OK] Diff-added active-authority scan found no newly added positive
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, or common/export RVV semantic authority.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 176/176 passed.

### Status

[OK] **Completed and ready to archive**. This round makes no new runtime,
correctness, or performance claim, so no fresh `ssh rvv` run was required.

### Next Steps

- Future continuation, if requested: keep Stage2 expansion on the corrected
  typed RVV body surface. Do not grow selected-body realization by putting new
  body-specific dispatcher logic back into `RVVExtensionPlugin.cpp`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 141: Stage1 legacy finite RVV i32 parse-only residue cleanup

**Date**: 2026-05-20
**Task**: `stage1-legacy-rvv-i32-residue-cleanup`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and cleaned the
remaining active-looking legacy finite RVV i32 parse-only residue. The retained
`tcrv_rvv.i32_*` parser/verifier surface no longer implements
`TCRVEmitCLowerableOpInterface`; positive EmitC-lowerable assertions now target
the corrected generic typed RVV dataflow surface.

### Main Changes

- Removed EmitC lowerable interface declarations from retained legacy
  `tcrv_rvv.i32_*` ODS ops.
- Removed obsolete `I32LoadOp`, `I32BroadcastLoadOp`, and `I32StoreOp`
  lowerable source-role method implementations.
- Reworked the RVV dialect C++ test so positive lowerable roundtrip uses
  `tcrv_rvv.load`, `tcrv_rvv.binary`, and `tcrv_rvv.store`.
- Added a C++ assertion that retained legacy `i32_load`, `i32_add`, and
  `i32_store` remain parse-only and non-lowerable.
- Updated RVV dialect descriptions so `with_vl`, `status`, and
  `rvv_emitc_route_mapping` read as mirror/handoff facts, not route or
  acceptance authority.

### Testing

- [OK] Trellis task context validation.
- [OK] `cmake --build build --target tcrv-opt tianchenrv-rvv-dialect-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] Focused lit: 5/5 passed for RVV dialect dataflow, generic binary
  materialization, legacy fail-closed materialization, and target artifact
  smoke.
- [OK] Focused lit: 13/13 passed for legacy selected-body fail-closed,
  legacy source-front-door fail-closed, and generic typed Stage2
  materialization paths.
- [OK] After-scan found no legacy i32 EmitC-lowerable authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 176/176 passed.

### Status

[OK] **Completed and ready to archive**. This round makes no runtime,
correctness, or performance claim, so no fresh `ssh rvv` evidence was required.

### Next Steps

- Future continuation, if requested: further Stage1 cleanup can delete the
  deprecated parse-only i32 ODS surface entirely once no parser/verifier
  inventory tests need it.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |


## Session 133: Stage2 generic RVV reduction accumulation route

**Date**: 2026-05-19
**Task**: Stage2 generic RVV reduction accumulation route
**Branch**: `main`

### Summary

Added a bounded generic typed RVV reduce(add) selected-body route skeleton with verifier, construction protocol, provider materialization, artifact/header dry-run, fail-closed negatives, and no ssh-rvv runtime claim.

### Main Changes

- Added generic `tcrv_rvv.reduce` for a typed vector input plus typed accumulator/result under matching `tcrv_rvv.with_vl`; the bounded supported reduction kind is `add`.
- Extended RVV construction protocol route recognition with `reduce_add`, generic `tcrv_rvv.reduce`, and provider-owned `rvv-generic-reduce-add-*` route/runtime ABI identity.
- Extended the RVV EmitC route provider to derive a reduction leaf from typed body/config/runtime facts and to reject RHS-broadcast reduction in this slice.
- Added positive EmitC materialization and target header/artifact dry-run fixtures plus negative verifier/provider coverage.

### Testing

- [OK] Trellis task context validation.
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused reduction lit set: 5/5 passed.
- [OK] focused existing first-slice EmitC regression lit set: 5/5 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 154/154.
- [OK] `git diff --check`
- [OK] active-authority scan: diff-only legacy-shaped addition is provider-derived `__riscv_vredsum_vs_i32m1_i32m1`; no new `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`, or `tcrv_rvv.i32_macc` helper surface.

### Status

[OK] **Completed** as route-supported and artifact/header dry-run evidence. No `ssh rvv` correctness or performance claim was made in this round.

### Next Steps

- Future continuation, if requested: turn the reduction skeleton into executable runtime correctness evidence on `ssh rvv` and refine accumulator/result layout beyond this bounded vector accumulator form.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |


## Session 141: Stage2 RVV selected-body realization for macc add

**Date**: 2026-05-19
**Task**: `05-19-stage2-rvv-selected-body-macc-add`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented one
bounded pre-realized selected-body realization path for `macc_add`. The RVV
plugin now consumes explicit accumulator role/layout, lhs/rhs/out ABI values,
runtime `n`/AVL, dtype/config, memory form, and policy facts into realized
typed `tcrv_rvv` structure before the existing macc provider route constructs
the lowerable EmitC route.

### Main Changes

- Added `tcrv_rvv.typed_macc_pre_realized_body` with verifier checks for
  bounded `macc_add`, output-buffer accumulator role, accumulator/result
  layout, SEW32/LMUL m1, agnostic policy, runtime ABI roles, and stale
  authority metadata.
- Extended RVV plugin selected-body realization to materialize the pre-realized
  macc body into `setvl/with_vl/load/load/load/macc/store`, then reuse the
  existing provider/materializer/target artifact path.
- Extended generated-bundle evidence tooling with
  `--pre-realized-selected-body --op-kind macc_add`.
- Added positive pre-realized macc target/script fixtures and macc-specific
  fail-closed negative coverage.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV dialect/plugin,
  construction protocol, and target artifact export tests.
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Focused lit set covering new pre-realized macc positive/negative/script
  evidence, explicit macc, existing pre-realized add/sub/mul/strided/masked
  fixtures, existing pre-realized negatives, and macc provider negatives:
  16/16 passed.
- [OK] Local pre-realized macc generated-bundle dry-run at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-pre-realized-macc-add-dry-run`.
- [OK] Real `ssh rvv` macc correctness evidence at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-pre-realized-macc-add-evidence`:
  `PASS op=macc_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 173/173 passed.
- [OK] `git diff --check`
- [OK] diff-only active-authority scan introduced no positive legacy RVV route
  authority; new matches were anti-authority text and negative guardrails.

### Self-Repair

- Re-ran lit from `build/test` after an initial run from `build/` failed to
  resolve the relative lit config path.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for bounded pre-realized `macc_add` counts `7,16,23`.

### Next Steps

- Future continuation, if requested: broaden contraction-supporting Stage2
  coverage beyond this single selected-boundary macc_add handoff under a
  separate bounded task.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 141: Stage2 RVV selected-body realization for masked add policy

**Date**: 2026-05-19
**Task**: `05-19-stage2-rvv-selected-body-masked-add-policy`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented the
bounded pre-realized selected-body masked add handoff. The new
`tcrv_rvv.typed_masked_binary_pre_realized_body` carries explicit masked-add
operation/config/mask-source/passthrough/runtime ABI facts, and the RVV plugin
realizes it into `setvl/with_vl/load/compare/masked_binary/store` before the
existing masked-add provider route runs.

### Main Changes

- Added the generic pre-realized masked selected-body op and verifier checks.
- Extended RVV plugin selected-body realization for one masked add path.
- Reused the existing masked-add provider/common EmitC/target artifact route.
- Added positive target/script fixtures and fail-closed negative coverage.
- Extended generated-bundle evidence tooling for
  `--pre-realized-selected-body --op-kind masked_add`.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV dialect/plugin,
  construction protocol, and target artifact export tests.
- [OK] Direct realization/emission/header FileCheck for pre-realized masked add.
- [OK] Direct negative verifier run for pre-realized selected-body failures.
- [OK] Focused lit filter: 11/11 passed.
- [OK] RVV dialect/plugin/construction/target C++ smoke tests.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Local generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-pre-realized-masked-add-dry-run`.
- [OK] Real `ssh rvv` correctness evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-pre-realized-masked-add-evidence`,
  `PASS op=masked_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 170/170 passed.
- [OK] `git diff --check`
- [OK] Diff active-authority scan introduced no positive legacy RVV/source-front-door/descriptor authority.

### Status

[OK] **Completed and archived**. This round makes a real RVV
correctness claim only for bounded pre-realized `masked_add` counts `7,16,23`.

### Next Steps

- Future continuation, if requested: broaden selected-body realization beyond
  this single masked add policy path under a separate bounded PRD.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 141: Stage2 RVV selected-body realization for typed strided add

**Date**: 2026-05-19
**Task**: `stage2-rvv-selected-body-strided-add`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented the
bounded pre-realized selected-body realization path for typed strided add. The
RVV plugin now consumes explicit lhs/rhs/out stride ABI operands from
`tcrv_rvv.typed_binary_pre_realized_body` and realizes them into
`setvl/with_vl/strided_load/binary/strided_store` structure before the existing
`strided_add` provider route is constructed.

### Main Changes

- Extended `tcrv_rvv.typed_binary_pre_realized_body` with optional stride
  operands while preserving unit-stride add/sub/mul pre-realized bodies.
- Hardened verifier checks for strided selected-body realization: exact
  lhs/rhs/out stride count, index type, runtime ABI source, stride roles,
  `op_kind = "add"`, SEW32, LMUL m1, agnostic policy, and correct base roles.
- Extended RVV plugin-local realization to produce typed strided memory ops
  and reuse the existing strided_add provider/materializer/target path.
- Added pre-realized strided add target artifact and generated-bundle dry-run
  fixtures plus negative fail-closed cases for missing/wrong stride facts.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with
  `--pre-realized-selected-body --op-kind strided_add`.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV dialect/plugin,
  construction protocol, and target artifact export tests.
- [OK] Focused lit: 11/11 passed for pre-realized strided add, existing
  pre-realized add/sub/mul, pre-realized negatives, explicit strided add, and
  generated-bundle dry-run coverage.
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] local dry-run at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-pre-realized-strided-add-dry-run`.
- [OK] real `ssh rvv` correctness evidence at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-pre-realized-strided-add-evidence`:
  `PASS op=strided_add counts=7,16,23`.
- [OK] `git diff --check`
- [OK] diff-only active-authority scan: no newly added positive legacy RVV
  route authority, source-front-door/source-seed authority, descriptor/direct-C
  source-export authority, or common/export RVV semantic authority.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 168/168 lit tests
  passed.

### Spec Update

No `.trellis/spec/` update was needed. This round implemented the existing
selected-body realization and plugin-local route-authority contract for one
bounded strided add case; it did not add a new long-term architecture rule.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for the pre-realized selected-body `strided_add` path at
counts `7,16,23`.

### Next Steps

- Future continuation, if requested: broaden selected-body realization to
  another bounded Stage2 memory/control class under a separate PRD; do not
  infer semantics from route ids, artifact names, ABI names, or source-front
  door metadata.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 139: Stage2 RVV typed strided memory-form route semantics

**Date**: 2026-05-19
**Task**: `stage2-rvv-typed-strided-memory-route`
**Branch**: `main`

### Summary

Added one bounded generic typed RVV strided load/store memory-form route through
an explicit selected-body add path. The route binds lhs/rhs/out stride runtime
ABI values, verifies typed `tcrv_rvv.strided_load/store` structure, derives RVV
provider route/header/intrinsic/artifact facts from typed body/config/runtime
facts, and reaches real `ssh rvv` correctness evidence.

### Main Changes

- Added `tcrv_rvv.strided_load` and `tcrv_rvv.strided_store` plus verifier
  checks for stride runtime ABI roles, base pointer roles, VL, vector
  type/config, and selected-body placement.
- Added explicit runtime ABI stride roles and the bounded seven-parameter
  selected-body ABI contract for strided add.
- Extended RVV construction protocol and RVV EmitC route provider with the
  `strided_add` route, strided layout mirrors, provider-derived intrinsic
  leaves, and generated target metadata.
- Kept common EmitC/export neutral by adding generic expression materialization
  for provider-built scaled pointer expressions.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with `--op-kind
  strided_add`, dry-run checks, and real RVV correctness harness support.

### Testing

- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV dialect/plugin,
  construction protocol, and target artifact export tests.
- [OK] Focused lit filter for strided dialect, EmitC, target artifact, and
  script dry-run fixtures: 5/5 passed.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Focused C++ test binaries for RVV dialect, RVV plugin, construction
  protocol, and target artifact export.
- [OK] Local strided add generated-bundle dry-run at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-strided-add-dry-run`.
- [OK] Real `ssh rvv` strided add correctness evidence at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-strided-add`:
  `PASS op=strided_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 166/166 passed.
- [OK] `git diff --check`
- [OK] active-authority scan found no new positive legacy RVV route authority.

### Self-Repair

- Repaired stride runtime ABI verifier handling for `index`-typed
  `runtime_abi_value` results.
- Replaced literal C pointer payloads with neutral EmitC SSA expression
  materialization.
- Repaired target artifact runtime ABI preflight for provider-derived dynamic
  route signatures.
- Updated stale FileCheck diagnostics after the selected-body op list gained
  strided load/store.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for bounded `strided_add` counts `7,16,23`.

### Next Steps

- Future continuation, if requested: broaden Stage2 memory movement beyond this
  single strided add memory-form submodule under a separate bounded task.


## Session 134: Stage1 Gate A RVV route identity cleanup

**Date**: 2026-05-19
**Task**: `stage1-gate-a-rvv-route-identity-cleanup`
**Branch**: `main`

### Summary

Closed the bounded Stage1 Gate A route-identity residues for the active RVV selected-body route path. Route ids, runtime ABI names, artifact route/header/bundle names, config-contract APIs, source-front-door pass naming, and pre-realized selected-body entry points now use generic selected-body / typed-body authority rather than `rvv-i32m1`, `RVVI32M1`, or `i32_binary_pre_realized_body` as positive route identity.

### Main Changes

- Renamed active RVV construction identities to `rvv-generic-typed-body-*` and per-operation provider-derived `rvv-generic-binary-*`, `rvv-generic-cmp-select-*`, and `rvv-generic-reduce-add-*` route/runtime ABI labels.
- Renamed active config-contract APIs and metadata to selected-body generic names while preserving SEW32/LMUL/policy facts as typed config facts.
- Replaced positive `tcrv_rvv.i32_binary_pre_realized_body` fixtures with `tcrv_rvv.typed_binary_pre_realized_body`.
- Kept retained i32 add/sub/mul as ordinary generic typed `tcrv_rvv.binary {kind = ...}` instances, and kept legacy `tcrv_rvv.i32_*` selected-body snippets only as negative fail-closed fixtures.
- Renamed the RVV source-front-door pass to a fail-closed legacy vector-source front-door identity.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] focused lit filter for `Target/RVV`, RVV lowering/source-front-door transforms, and RVV script dry-runs: 40/40 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 154/154 passed.
- [OK] `git diff --check`
- [OK] exact Stage1 Gate A brief scan over `include/TianChenRV lib/Plugin/RVV lib/Dialect/RVV test/Target/RVV`: no `rvv-i32m1`, `RVVI32M1`, or `i32_binary_pre_realized_body` matches.
- [OK] remaining extended legacy matches are classified as negative fail-closed Target/RVV fixtures, dialect parse/verifier debt, and provider-derived intrinsic leaf spellings.

### Status

[OK] **Completed and ready to archive**. No `ssh rvv` runtime, correctness, or performance claim was made.

### Next Steps

- Future continuation, if requested: delete or further fail-close the remaining parseable legacy dialect debt under a separate Stage1 deletion task; do not treat it as Stage2 coverage work.


## Session 135: Stage2 executable closure for generic RVV reduction

**Date**: 2026-05-19
**Task**: `stage2-generic-rvv-reduction-executable-closure`
**Branch**: `main`

### Summary

Closed the existing generic `tcrv_rvv.reduce {kind = "add"}` selected-body
path as executable RVV evidence. The provider now makes the bounded reduction
layout explicit: RHS is the vector seed, lane 0 carries the per-VL chunk result,
and generated code stores only lane 0 to `out[offset]`.

### Main Changes

- Added provider-derived reduction accumulator/result layout fields to
  `RVVSelectedBodyEmitCRouteDescription`.
- Updated the RVV EmitC route provider so reduce-add stores with VL `1`, making
  non-result lanes untouched and giving the external ABI harness a precise
  correctness contract.
- Added reduce-add metadata mirrors to target artifact bundle metadata.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with `--op-kind
  reduce_add` dry-run and `ssh rvv` correctness support.
- Updated focused reduce-add materialization and artifact fixtures.

### Testing

- [OK] Trellis context validation.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused C++ build for `tcrv-opt`, `tcrv-translate`, RVV plugin,
  construction protocol, and target artifact export tests.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] reduce-add generated bundle dry-run.
- [OK] existing add/sub/mul/cmp_select generated bundle dry-run.
- [OK] real `ssh rvv` run for reduce-add counts `1,7,16,17,257`:
  `tcrv_rvv_generated_bundle_abi_reduce_add_ok counts=1,7,16,17,257`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 154/154 passed.
- [OK] `git diff --check`
- [OK] diff-only active-authority scan introduced no legacy RVV route authority.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for the bounded reduce-add chunk layout described above.

### Next Steps

- Future continuation, if requested: broaden reduction/accumulation semantics
  beyond this bounded per-VL chunk layout under a separate Stage2 coverage task.


## Session 136: Stage2 generic RVV masked add route semantics

**Date**: 2026-05-19
**Task**: `stage2-rvv-masked-add-route-semantics`
**Branch**: `main`

### Summary

Added one bounded generic `tcrv_rvv.masked_binary {kind = "add"}` selected-body
route. The route carries compare-produced mask flow, passthrough/result vector
role, VL, typed vector/mask shape, and policy facts through RVV dialect
verification into RVV provider route construction and target artifact evidence.

### Main Changes

- Added `tcrv_rvv.masked_binary` and verifier checks for compare mask source,
  same `with_vl` scope, VL token consistency, vector/mask compatibility,
  passthrough/lhs/rhs/result agreement, and bounded `kind = "add"`.
- Added RVV construction protocol support for the `masked_add` selected-body
  route, typed op identity, role sequence, ABI mapping, and artifact metadata.
- Added provider-side masked-add route validation and materialization:
  compare mask, active add, passthrough merge, and store. Common EmitC/export
  remains neutral.
- Extended target artifact export tests and `rvv_generated_bundle_abi_e2e.py`
  with `--op-kind masked_add` dry-run and executable harness support.
- Added positive/negative dialect, materialization, artifact, and script tests;
  updated stale generic-op diagnostic fixtures to include `tcrv_rvv.masked_binary`.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV plugin,
  construction protocol, and target artifact export tests.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused masked-add dialect/materialization/negative/target artifact
  FileCheck commands.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] masked-add generated-bundle dry-run.
- [OK] real `ssh rvv` masked-add correctness evidence for counts
  `1,7,16,17,257`:
  `tcrv_rvv_generated_bundle_abi_masked_add_ok counts=1,7,16,17,257`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 157/157 passed.
- [OK] `git diff --check`
- [OK] diff active-authority scan introduced no legacy RVV route authority;
  remaining exact i32m1 intrinsic matches are provider-derived leaf checks.

### Self-Repair

- Replaced an initially attempted five-argument masked add intrinsic shape after
  current RVV clang rejected it during generated object packaging. The final
  bounded route uses active add plus provider-derived merge with passthrough.
- Repaired stale broad lit diagnostics that omitted the newly allowed
  `tcrv_rvv.masked_binary` op from generic selected-body op lists.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for bounded masked add counts `1,7,16,17,257`.

### Next Steps

- Future continuation, if requested: expand Stage2 masked execution classes
  beyond this single masked add submodule under a separate bounded task.


## Session 137: Stage2 RVV typed SEW/LMUL config derivation for generic arithmetic

**Date**: 2026-05-19
**Task**: `stage2-rvv-typed-config-arithmetic`
**Branch**: `main`

### Summary

Closed the bounded Stage2 generic arithmetic typed-config task by repairing the
RVV provider-local resolver to cross-check typed vector/mask element width and
LMUL against selected `setvl`/`with_vl` config before route/profile/intrinsic
selection. Existing m2 arithmetic support was confirmed as a real generic typed
selected-body route, not stale artifact metadata, and was revalidated through
local artifact dry-run plus real `ssh rvv` correctness evidence.

### Main Changes

- Added provider-side typed-config validation for generic
  `!tcrv_rvv.vector` and `!tcrv_rvv.mask` values before deriving route
  profiles and intrinsic/header payloads.
- Added C++ plugin coverage that proves LMUL m2 selected-body add derives m2
  RVV intrinsics and then fails closed if the same typed m2 body is paired with
  stale LMUL m1 config metadata.
- Kept the change inside the RVV provider/test surface; common EmitC/export,
  source front doors, reduction/mask/broadcast coverage, and route-family
  breadth were not expanded.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV plugin,
  construction protocol, and target artifact export tests.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] focused m2 target artifact/script lit commands: 4/4 passed.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] m2 add/sub/mul generated-bundle dry-run at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-typed-config-m2-dry`.
- [OK] real `ssh rvv` m2 add/sub/mul correctness evidence at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-typed-config-m2-ssh`:
  `PASS op=add counts=7,16,23`, `PASS op=sub counts=7,16,23`,
  and `PASS op=mul counts=7,16,23`.
- [OK] `git diff --check`
- [OK] diff active-authority scan introduced no legacy RVV route authority.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 157/157 passed.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for bounded m2 generic arithmetic add/sub/mul counts
`7,16,23`.

### Next Steps

- Future continuation, if requested: broaden typed-config derivation beyond
  this bounded generic arithmetic m1/m2 surface under a separate Stage2 task.


## Session 138: Stage2 RVV contraction multiply-add route skeleton

**Date**: 2026-05-19
**Task**: `stage2-rvv-multiply-add-route-skeleton`
**Branch**: `main`

### Summary

Added one bounded generic typed `tcrv_rvv.macc {kind = "add"}` selected-body
route for `accumulator + lhs * rhs`. The route now carries lhs/rhs/output-buffer
accumulator/result roles through RVV dialect verification, construction protocol
recognition, RVV provider route derivation, common EmitC materialization, target
artifact export, generated-bundle dry-run, and real `ssh rvv` correctness
checks.

### Main Changes

- Added non-dtype-prefixed `tcrv_rvv.macc` and verifier checks for bounded add
  macc kind, same generic vector type across lhs/rhs/accumulator/result, same
  selected VL scope, and typed config agreement.
- Extended RVV construction protocol with `macc_add` selected-body route,
  provider-owned route/runtime ABI identity, role sequence, and artifact role
  metadata.
- Extended the RVV EmitC route provider to derive `rvv-generic-macc-add-*`
  facts after typed config validation, recognize output-buffer accumulator
  loads, materialize accumulator load + vmacc + store, and emit macc layout
  metadata mirrors.
- Added positive materialization, target artifact, generated-bundle dry-run, and
  negative fail-closed tests for unsupported macc forms.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with `--op-kind macc_add`
  dry-run and real RVV correctness harness support.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV plugin,
  construction protocol, and target artifact export tests.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused macc dialect/materialization/negative/target/script and
  existing first-slice regression lit set: 10/10 passed.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] macc generated-bundle dry-run at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-macc-add-dry-run`.
- [OK] real `ssh rvv` macc correctness evidence at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-macc-add-evidence`:
  `PASS op=macc_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 161/161 passed.
- [OK] `git diff --check`
- [OK] diff-only active-authority scan: only added legacy-shaped token is
  provider-derived `__riscv_vmacc_vv_i32m1`; no new positive legacy RVV route
  authority was introduced.

### Self-Repair

- Repaired the macc accumulator-type negative verifier test so the intended
  `tcrv_rvv.macc` diagnostic is reached.
- Repaired the macc generated-bundle FileCheck expectations to match the actual
  evidence schema and harness execution order.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for bounded `macc_add` counts `7,16,23`.

### Next Steps

- Future continuation, if requested: broaden contraction-supporting Stage2
  coverage beyond this single macc_add route under a separate bounded task.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 140: Stage2 RVV selected-body realization for generic add

**Date**: 2026-05-19
**Task**: `stage2-rvv-selected-body-generic-add`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and verified the
current HEAD already contains the bounded generic typed pre-realized selected
body realization path for add. The RVV plugin consumes
`tcrv_rvv.typed_binary_pre_realized_body` into realized
`setvl/with_vl/load/binary/store` structure before provider route construction.

### Main Changes

- Added the task PRD and curated implement/check context for the bounded
  generic-add selected-body realization round.
- Added one missing fail-closed lit negative case for a pre-realized selected
  body with no explicit `op_kind` attribute.
- Confirmed no production C++ behavior change was needed: the current
  plugin-local realization hook and provider handoff already satisfy the
  bounded generic add module behavior.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV plugin test, and
  target artifact export test.
- [OK] focused lit for pre-realized add, pre-realized negative cases, existing
  explicit selected-body add/sub/mul, and pre-realized generated-bundle dry-run:
  6/6 passed.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] add-only local dry-run at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-pre-realized-generic-add-dry`.
- [OK] `git diff --check`
- [OK] active-authority scan classified remaining high-risk RVV matches as
  provider-derived intrinsic leaves, deprecated parse-only inventory,
  fail-closed legacy tests, unsupported source-seed diagnostics, or residue
  rejection filters.

### Status

[OK] **Completed and archived**. This round makes no new runtime, correctness,
or performance claim, so no fresh `ssh rvv` run was required.

### Next Steps

- Future continuation, if requested: Stage2 selected-body realization beyond
  this bounded generic binary add handoff should use a separate PRD and must
  not infer semantics from route ids, artifact names, or source-front-door
  markers.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

## Session 144: Stage2 RVV LMUL route-policy executable slice

**Date**: 2026-05-20
**Task**: `stage2-rvv-lmul-route-policy-executable-slice`
**Branch**: `main`

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented one
bounded Stage2 LMUL m2 route-policy executable slice. The new positive path is
an i32 / SEW32 / LMUL m2 / unit-stride `add` pre-realized selected body that is
realized by the RVV plugin into generic typed `tcrv_rvv` structure and then
routed through the existing provider-owned EmitC route planning path.

### Main Changes

- Extended the typed binary pre-realized verifier to accept only the bounded
  SEW32 / LMUL m2 / unit-stride `add` tuple outside the existing m1 and
  i64/m1 slices.
- Extended RVV selected-body realization validation for the same bounded m2
  tuple, preserving plugin-local realization into `setvl`, `with_vl`, `load`,
  `binary`, and `store`.
- Added `lmul_m2_add` to the generated-bundle evidence harness as a script
  selector that maps to provider operation `add` and verifies the realized
  generic vector type `!tcrv_rvv.vector<i32, "m2">`.
- Added a pre-realized LMUL m2 positive MLIR artifact test and a generated
  bundle dry-run lit test with descriptor/direct-C/source-export/i32 helper
  guardrails.
- Updated negative coverage so m2/sub still fails closed and m2/add with the
  wrong policy fails closed.

### Testing

- [OK] Trellis task context validation.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused build for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] focused C++ RVV dialect/plugin/export tests.
- [OK] focused lit for the new LMUL m2 pre-realized artifact, generated-bundle
  dry-run, pre-realized negative suite, and existing i64/m2 controls.
- [OK] manual generated-bundle dry-run for `lmul_m2_add` counts `7,16,23`.
- [OK] real `ssh rvv` generated-bundle correctness:
  `PASS op=lmul_m2_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`
- [OK] `git diff --check`
- [OK] active-authority scan found no positive reintroduction in diff-added
  hunks; broader matches remain pre-existing deprecated inventory,
  fail-closed tests, harness guardrails, or provider-derived leaves.

### Self-Repair

- After adding the direct m2 wrong-policy negative, the first focused lit rerun
  failed because the expected diagnostic contained an extra `binary` word. The
  test expectation was corrected to the verifier's actual message and the
  focused rerun passed.

### Status

[OK] **Completed and ready to archive**. This round claims executable
correctness only for the bounded `lmul_m2_add` counts `7,16,23`; no performance
claim is made.

### Next Steps

- Future continuation, if requested: expand LMUL policy beyond this single
  bounded m2/add slice under a separate PRD, without creating a dtype/LMUL
  clone matrix or reviving old `!tcrv_rvv.i32m*` route authority.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |


## Session 139: Stage2 RVV indexed gather memory executable slice

**Date**: 2026-05-20
**Task**: Stage2 RVV indexed gather memory executable slice
**Branch**: `main`

### Summary

Implemented bounded i32 SEW32 LMUL m1 indexed gather load to unit-stride store through typed RVV body facts, plugin-owned realization, route planning/provider emission, generated-bundle dry-run, and ssh rvv correctness for counts 7,16,23.

### Main Changes

### Summary

Created the Trellis task from the Hermes Direction Brief and implemented one
bounded Stage2 indexed gather memory movement executable slice. The selected
boundary now carries `data`, `index`, `out`, and `n` ABI/runtime roles; the RVV
body carries explicit index vector, indexed data load, offset EEW, element offset
unit, i32/SEW32/LMUL m1 config, policy, and unit-stride destination facts.

### Main Changes

- Added the `index-input-buffer` runtime ABI role and bounded indexed gather ABI
  contract `data,index,out,n`.
- Added generic typed `!tcrv_rvv.index_vector`, `tcrv_rvv.index_load`,
  `tcrv_rvv.indexed_load`, and the bounded pre-realized indexed gather body.
- Extended RVV selected-body realization to materialize the pre-realized form to
  `setvl`, `with_vl`, `index_load`, `indexed_load`, `move`, and `store`.
- Extended RVV EmitC route planning/provider metadata, ABI order, index/vector C
  types, indexed-load leaf, byte-offset scale leaf, and unit-store handoff from
  typed facts.
- Extended construction-protocol summaries and generated-bundle evidence harness
  for explicit and pre-realized indexed gather artifacts.
- Added positive artifact tests, verifier/dataflow negative tests, and adjusted
  legacy fail-closed checks to include the new generic indexed ops.

### Testing

- [OK] Trellis task context validation.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] focused RVV dialect/config/plugin build targets and `tcrv-opt` /
  `tcrv-translate`.
- [OK] focused positive/negative `tcrv-opt` indexed gather verifier and artifact
  materialization checks.
- [OK] generated-bundle dry-run for explicit and pre-realized
  `indexed_gather_unit_store` counts `7,16,23`.
- [OK] real `ssh rvv` generated-bundle correctness for explicit and
  pre-realized paths: `PASS op=indexed_gather_unit_store counts=7,16,23`, with
  `non_monotonic_indexed_gather` cases for `n=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv`: 200/200 passed.
- [OK] `git diff --check`
- [OK] active-authority scan: no diff-added positive `RVVI32M1*`, `rvv-i32m1`
  route id, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door, descriptor, direct-C, or source-export authority. The one
  diff-added exact i32m1 intrinsic is the provider-derived indexed-load leaf,
  not route authority.

### Self-Repair

- Repaired the route planner runtime ABI validator so indexed and strided memory
  movement routes do not require finite-binary RHS ABI binding.
- Repaired `check-tianchenrv` expectation drift in legacy fail-closed tests after
  adding `index_load` / `indexed_load` to the generic Stage2 op list.
- Repaired the construction-protocol common test so indexed gather role-step and
  metadata checks use `move`, `indexed_load`, and `data,index,out,n` without
  dangling `ArrayRef` temporaries.

### Status

[OK] **Completed and ready to archive**. This round claims route-supported and
executable correctness only for the bounded i32 SEW32 LMUL m1 indexed gather
slice; no performance claim is made.

### Next Steps

- Future continuation, if requested: expand indirect memory movement only under a
  new PRD, for example scatter or masked/indexed variants, without turning this
  into a gather/scatter framework or dtype/LMUL clone matrix.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
