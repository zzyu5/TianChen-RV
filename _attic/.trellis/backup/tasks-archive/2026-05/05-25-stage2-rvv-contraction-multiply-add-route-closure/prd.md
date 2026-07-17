# Stage2 RVV contraction-supporting multiply-add route closure

## Goal

Close one bounded Stage 2 RVV contraction-supporting multiply-add or multiply-accumulate path on the corrected typed `tcrv_rvv` surface. The route must carry multiply-add kind, vector input types, accumulator/result type, runtime AVL/VL, and policy facts from a selected `tcrv.exec` RVV variant through RVV plugin-owned planning/provider construction, common EmitC materialization, target artifact mirrors, and focused `ssh rvv` correctness evidence.

## What I already know

* The previous completed task closed the standalone `reduce_add` route boundary with RVV-owned statement planning, fail-closed provider tests, generated-bundle evidence, and `ssh rvv` correctness for counts 0, 7, 16, and 23.
* This task must improve the next low-level Stage 2 contraction-supporting boundary, not high-level matmul/contraction lowering.
* Current repo inspection shows plain `macc_add` is the smallest bounded multiply-add route to close in this round. It already has typed `tcrv_rvv.macc` / `typed_macc_pre_realized_body`, selected-body realization, math operand-binding facts, target fixtures, and generated-bundle dry-run coverage.
* Current provider construction still locally assembles the plain `macc_add` `setvl/load/load/accumulator-load/macc/store` sequence after route-family/materialization/operand-binding facts are available. Adjacent routes now consume RVV-owned statement plans through `getRVVSelectedBodyMigratedRouteStatementPlan`.
* Multiply-add or multiply-accumulate semantics must come from typed `tcrv_rvv` body/config/runtime facts, not route names, high-level concepts, descriptors, artifact metadata, fixture names, ABI strings, or harness constants.
* The implementation must stay in the C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck compiler stack. Python may only support tooling, probes, runners, artifact parsing, or small scripts.
* `tcrv.exec` owns envelope, variant, ABI/runtime role binding, dispatch/fallback, and diagnostics. It does not own RVV compute semantics.
* The RVV plugin owns RVV legality, selected-body realization, route support, intrinsic/C type/ABI mapping, and fail-closed diagnostics. Common EmitC/export must remain neutral.

## Assumptions

* The selected route for this round is plain signed i32 / SEW32 / LMUL m1 / unit-stride `macc_add`, because it is the bounded route with existing typed surface and executable harness support.
* No new dialect op is expected. The likely code change is an RVV-owned plain MAcc statement-plan boundary plus migrated provider consumption and evidence tightening.
* Runtime executable evidence should use the existing generated bundle ABI/e2e path for `macc_add`; otherwise the exact blocker must be documented without overclaiming executable support.

## Requirements

* Validate `macc_add` kind, lhs/rhs vector input types, accumulator input type/layout, result type/layout, runtime VL/AVL, policy, and ABI operand bindings from typed `tcrv_rvv` body/config/runtime structure.
* Add a plain `macc_add` RVV-owned statement-plan boundary that builds provider-ready full-chunk `setvl`, loop `setvl`, lhs/rhs loads, accumulator load, multiply-accumulate compute, and store steps from materialization facts and math operand-binding facts.
* Route provider must consume this plan through the migrated statement-plan boundary before generic provider-local statement assembly.
* Fail closed before route/artifact authority for missing math operand-binding facts, missing lhs/rhs/acc/out/n ABI binding, missing `setvl`/load/macc/store leaves, inconsistent result or accumulator layout, stale route-id/name authority, descriptor residue, or metadata-derived authority.
* Keep artifact metadata mirror-only with explicit mirror labels.
* Keep common EmitC and target artifact mechanics neutral; do not move multiply-add or accumulation semantics into common plumbing.
* Add or update focused verifier/provider/FileCheck tests for the bounded route and negative cases.
* Provide focused generated-bundle ABI/e2e `ssh rvv` correctness evidence across at least three runtime counts if executable behavior is claimed.

## Acceptance Criteria

* [x] Plain `macc_add` is route-supported through selected body/config/runtime facts, RVV-owned math operand-binding facts, an RVV-owned statement plan, and provider-built `TCRVEmitCLowerableRoute`.
* [x] Provider route facts and the plain MAcc statement plan explicitly carry lhs/rhs/accumulator/result/runtime-VL semantics and reject missing or inconsistent facts before route construction.
* [x] FileCheck evidence shows emitted `__riscv_vmacc_vv_i32m1` operands derive from typed facts and statement-plan output, not route names or artifact metadata.
* [x] Generated artifact mirrors use explicit mirror labels and do not become route authority.
* [x] Negative tests cover missing/stale statement-plan dependencies before common EmitC, including missing math operand-binding facts and missing MAcc/vector-load leaves.
* [x] Focused `ssh rvv` correctness evidence covers counts 0, 7, 16, and 23 for pre-realized `macc_add`.
* [x] Bounded scans over touched RVV planning/provider/script/fixture files show no new name-, metadata-, descriptor-, high-level-matmul-, harness-, or legacy i32-derived multiply-add authority; the only production-code legacy i32 hit is the existing fail-closed guard in `RVVEmitCRoutePlanning.cpp`.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` runs successfully.

## Definition of Done

* PRD and Trellis task context are truthful.
* Relevant specs have been read before implementation.
* Code/tests implement only the bounded route closure described above.
* Focused checks have been run and any failures self-repaired.
* Task status is updated and archived if complete.
* One coherent commit is created if the task reaches completion.

## Out of Scope

* High-level matmul/contraction lowering, Linalg/Vector frontend lowering, global tiling, autotuning, dashboards, broad smoke matrices, or performance tuning.
* Broad dot/macc coverage, many dtype/LMUL variants, source-front-door positive routes, or compatibility wrappers preserving legacy i32 authority.
* New dtype-prefixed helper op families such as `tcrv_rvv.i32_macc`, `tcrv_rvv.i32_reduction_*`, or `tcrv_rvv.i32_accumulator_*`.
* Moving RVV multiply-add or accumulation semantics into common EmitC/export or target artifact plumbing.

## Technical Notes

* Task source: Hermes Direction Brief supplied on 2026-05-25.
* Required first reads: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, archived PRD for `05-25-stage2-rvv-reduction-accumulation-route-closure`, relevant RVV dialect/config/op definitions, route planning/provider, selected-body realization, target bundle generation, generated-bundle ABI/e2e script, and focused RVV tests.
* Focused implementation target after repo inspection: migrate plain `macc_add` provider statement construction into `RVVSelectedBodyPlainMAccRouteStatementPlan` or equivalent, wire it into `RVVSelectedBodyMigratedRouteStatementPlan`, tighten `macc_add` FileCheck/generated-bundle evidence, then run focused local and `ssh rvv` checks.
