# Stage 3 换心 — STEP 2 blocked: elementwise owner has a SECOND live consumer

Date: 2026-06-13. HEAD: 6f3ba3ad.

## Summary

STEP 1 (decouple the export seam from the string route) is DONE, build-green,
lit-baseline, and hardware-PASS on real ssh rvv for all 7 elementwise op-kinds.

STEP 2/STEP 3 (delete `RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp`)
is **blocked**: the task premise "every elementwise family converts and never
reaches the route builder, so the owner is dead" is **false**. There is a
SECOND, independent, lit-covered live consumer the task overlooked.

## The second live consumer

`tcrv-opt --tcrv-materialize-emission-plans` (the emission-plan diagnostic pass,
`createMaterializeEmissionPlansPass`, lib/Transforms/{ExecutionPlanningPipeline,
EmissionReadiness}.cpp) collects each variant's emission plan by querying the
RVV plugin's EmitC route provider:

  MaterializeEmissionPlans pass
    -> RVVExtensionPlugin emission-plan query
    -> RVVEmitCRouteProvider / RVVEmitCRoutePlanning
    -> getRVVSelectedBodyMigratedRouteStatementPlan (the dispatch table)
    -> elementwise statement-plan owner (the file we wanted to delete)

This path is NOT the export seam `materializeSelectedEmitCArtifactModule` that
STEP 1 decoupled. It builds the string route to emit the `tcrv.exec.diagnostic`
emission-plan PLAN metadata (rvv_selected_body_operation=add,
tcrv_rvv.route_operand_binding_plan, tcrv_rvv.elementwise_arithmetic_route_family_plan,
etc.). ~61 elementwise lit fixtures assert this PLAN diagnostic, and the HEADER
artifact export (`tcrv-translate --tcrv-export-target-header-artifact`, the I8
product path) chains off the emission-plan output.

## Empirical proof

Removing the elementwise dispatch entry + deleting the owner builds and links
clean (the link did NOT catch it — the route provider reaches the owner through
a function pointer in the dispatch table, so an undefined-reference never fires;
the table just silently loses the elementwise entry). But the full lit suite
went from 3 reds (baseline) to **57 reds**: every elementwise emission-plan +
artifact fixture failed with

  "selected RVV EmitC route construction requires an explicit migrated or
   direct-contraction statement-plan owner before provider-local route
   statement construction for operation 'add', memory_form 'vector-rhs-load'"

(`diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner`,
RVVEmitCStatementPlanOwners.cpp:2751). That is the owner being genuinely
reached and now missing — i.e. a live caller, not dead code.

Per the dead-mirror-removal guide rule 4 and the task's own guardrail ("if a
test breaks on a symbol, that symbol had a live caller you must understand
before proceeding — do NOT force"), STEP 2 was reverted. Only STEP 1 remains.

## What would unblock the deletion (separate, larger task)

The emission-plan diagnostic is an I4 metadata mirror, but it is consumed by the
I8 HEADER artifact export. To make the elementwise owner truly dead, the
emission-plan-materialization + header-export machinery (EmissionReadiness /
RVVEmitCRouteProvider / RVVEmitCRoutePlanning for elementwise families) must
first be decoupled from the string statement-plan owner the same way STEP 1
decoupled the export seam — or those ~61 PLAN fixtures must be migrated to the
converted-body source of truth. That is a substantially larger change than this
task's "delete the export-seam owner" and is out of scope here.

## STEP 1 evidence (this task's real deliverable)

- Build: green (full relink of tcrv-opt + tcrv-translate verified by timestamp).
- Full lit: 3 reds == baseline (the named environmental dequant/widening-dot
  self-test + 2 computed-masked-widening-dot dry-runs), zero new failures.
- Hardware (real ssh rvv, riscv64), export path now validates converted
  elementwise families WITHOUT the string route — all 7 PASS:
    PASS op=add counts=1,257
    PASS op=sub counts=1,257
    PASS op=mul counts=1,257
    PASS op=scalar_broadcast_add counts=1,257 rhs_scalars=-37,91
    PASS op=masked_add counts=1,257
    PASS op=strided_add counts=1,257 stride_triples=2:3:2,3:2:4
    PASS op=runtime_scalar_splat_store counts=1,257 rhs_scalars=-37,91
  (full log: step1-decouple-lamps.log)
