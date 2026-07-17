# Stage 3 换心 — BaseMemoryMovement statement-plan owner RETIRED

Date: 2026-06-13. Base HEAD: 9a0f5756 (BaseMemory family already 100% converted,
byte-identical, ssh rvv PASS). This change DELETES the now-dead string statement-plan
owner. NOT committed.

## Verdict

- BaseMemoryMovement string statement-plan owner: **RETIRED** (773 net lines deleted).
- lit: **clean — exactly 3 environmental reds** (the documented baseline: self-test
  + 2 computed-masked-strided-input-widening-DOT-REDUCE-ADD dry-runs; none base-memory,
  none ComputedMaskMemory/Segment2). Zero new reds.
- ComputedMaskMemory + Segment2 owners: **intact** (shared file portions untouched,
  ssh rvv PASS below).
- All 6 base-memory op-kinds: **PASS on real ssh rvv** (riscv64), owner GONE.

## What was BaseMemory-SPECIFIC (deleted) vs SHARED (kept)

### Deleted (BaseMemory-only; grep-gated to 0 non-comment refs after removal)
In `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`:
- `requireRVVBaseMemoryMovementStatementPlanLeaf`
- `requireRVVBaseMemoryMovementStatementPlanABI`
- `getRVVBaseMemoryMovementStatementPlanSourceProvenance`
- `makeRVVBaseMemoryMovementStatementPlanStep`
- `addRVVBaseMemoryMovementStatementPlanLoopStep`
- `buildBaseMemoryMovementMigratedRouteStatementPlan` (anon-ns orchestration)
- `buildRVVSelectedBodyBaseMemoryMovementMigratedRouteStatementPlan` (public entry)
- `getRVVSelectedBodyBaseMemoryMovementRouteStatementPlan` (~520-line builder)

In `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`:
- the `{"base memory movement", ...}` dispatch-table entry (owners[] array)

In `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`:
- decls for `getRVVSelectedBodyBaseMemoryMovementRouteStatementPlan`
  + `buildRVVSelectedBodyBaseMemoryMovementMigratedRouteStatementPlan`

In `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`:
- struct `RVVSelectedBodyBaseMemoryMovementRouteStatementPlan` (owner-only)

### Kept (SHARED — grep-proven live consumers elsewhere)
- `isRVVSelectedBodyBaseMemoryMovementStatementPlanConsumer` — consumer predicate.
  Live in `RVVEmitCRouteProvider.cpp:288` AND the description-source family-plan file
  `RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp:2736,2838` (do-not-touch). NOT owner-only.
- `getRVVSelectedBodyBaseMemoryMovementRouteProviderPlan` / `...RouteProviderPlan` struct
  / `...RouteFamilyPlan` struct / `verifyRVVSelectedBodyBaseMemoryMovementRouteProviderFacts`
  — all in the do-not-touch family-plan file + route provider. The description/provider
  source of truth; KEPT.
- `RVVSelectedBodyMigratedRouteStatementPlanFamily::BaseMemoryMovement` ENUM value —
  still referenced by the family-plan verify + the stringify switch. KEPT.
- `setRVVSelectedBodyMemoryMigratedRouteStatementPlan`,
  `stringifyRVVSelectedBodyMemoryMigratedRouteStatementPlanFamily`,
  the ComputedMaskMemory + Segment2 builders/predicates/dispatch entries,
  `kRVVMemoryStatementPlanOwnerEmitCLowerableOpInterfaceName` — all SHARED, KEPT.

## Why clean isolation was possible (empirical, not assumed)

Risk: the do-not-touch family-plan file's `verifyRVVSelectedBodyBaseMemoryMovementRouteProviderFacts`
(lines 3149-3221) hard-requires the owner selection == BaseMemoryMovement with the owner's
statements. If a base-memory body reached route construction, deleting the owner would error there.

Empirical probe (remove only the dispatch entry, rebuild, full lit): **0 new reds** — exactly
3 environmental. Reason: (a) every VALID base-memory body fully converts → the gate
`rvvSelectedBodyFullyConvertsToEmitC` decouples it → never reaches the string route; (b) every
MALFORMED base-memory body fails in `analyzeRVVSelectedBodyRoute` / the family-plan validators
BEFORE route construction → never reaches the owner verify. So the owner is genuinely dead for
both positives and negatives. Contrast: the elementwise owner had a SECOND live consumer
(emission-plan diagnostic, 57 reds on probe) — base-memory does NOT (its emission-plan/header
path goes through the conversion gate; all Target/RVV artifact PLAN/HEADER fixtures green).

## Rebuild recipe (header/struct changed → full clean rebuild)

    cd build && ninja -t clean && ninja && ninja bin/tcrv-opt bin/tcrv-translate

Timestamps verified: tcrv-opt + tcrv-translate newer than libTianChenRVRVVEmitCRouteProvider.a.

## Full lit (post-deletion)

    Total Discovered Tests: 562
    Failed: 3  (Scripts/rvv-generated-bundle-abi-e2e-*self-test +
                explicit/pre-realized computed-masked-strided-input-widening-dot-reduce-add-dry-run)
== documented baseline environmental reds, zero new.

## ssh rvv (real riscv64, NO --dry-run) — all 6 base-memory op-kinds PASS, owner GONE

    PASS op=strided_load_unit_store    counts=1,7,16,17,257 stride_bytes=4,8,12 source_preserved
    PASS op=unit_load_strided_store    counts=1,7,16,17,257 stride_bytes=4,8,12 source_preserved
    PASS op=indexed_gather_unit_store  counts=1,7,16,17,257 index_patterns=2 element_offset_indices unit_store_output tail_preserved runtime_n_avl_honored
    PASS op=indexed_scatter_unit_load  counts=1,7,16,17,257 index_patterns=2
    PASS op=masked_unit_load_store     counts=1,7,16,17,257
    PASS op=masked_unit_store          counts=1,7,16,17,257   (--pre-realized-selected-body mode)

(log: ssh-rvv-6-base-memory.log)

## ssh rvv — ComputedMaskMemory + Segment2 owners INTACT (still work)

    PASS op=computed_masked_unit_load_store   counts=1,7,16,17,257
    PASS op=computed_masked_strided_store     (exit 0, all cases ok)
    PASS op=segment2_deinterleave_unit_store  counts=1,7,16,17,257
    PASS op=segment2_interleave_unit_load     counts=1,7,16,17,257

(logs: ssh-rvv-computedmask-segment2-intact.log, ssh-rvv-computed-masked-strided-store.log)

## Adversarial / negative-fixture handling (I5-honest)

12 base-memory + computed-mask NEGATIVE fixtures all PASS post-deletion (FileCheck verdicts
unchanged). With the owner gone, the fallback error now comes from the SURVIVING conversion
guards + the route/profile family-plan validators in `analyzeRVVSelectedBodyRoute` /
`RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp` (which fire in analysis, before route
construction). No test weakened, no re-baselining needed — the surviving guards reproduce
every expected diagnostic (e.g. "requires source-byte-stride runtime ABI value",
"requires exactly one tcrv_rvv.load op for old destination", "runtime ABI contract invalid").
