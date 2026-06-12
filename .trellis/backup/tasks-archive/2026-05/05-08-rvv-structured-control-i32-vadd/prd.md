# RVV structured control-plane i32 vadd emission

## Goal

Implement one bounded compiler/runtime slice for the i32-vadd RVV path so plugin-owned lowering materializes compiler-visible RVV control-plane IR and target-owned emission validates or consumes that structured body. The slice must connect the selected RVV microkernel path to explicit `tcrv_rvv.setvl` and `tcrv_rvv.with_vl` IR without moving RVV semantics into generic `tcrv.exec` or generic target routing.

## What I Already Know

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Precondition audit passed on 2026-05-08: worktree was clean, supervisor-policy files were clean, and HEAD was `a38d971 feat: add capability relation provider lookup`.
* The previous run added relation-aware capability/proposal/materialization/export scaffolding and left no new RVV runtime/correctness/performance claim.
* This task must run as one serial full-access non-TUI worker with no subagents or multi-agent workflow.
* Core compiler implementation must remain C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck. Python is only allowed for runner/support utilities.

## Requirements

* Prefer wiring the existing RVV plugin-owned lowering-boundary/materialization path so the finite selected i32-vadd RVV microkernel surface contains a structured body with `tcrv_rvv.setvl` and `tcrv_rvv.with_vl`.
* If existing ODS cannot legally contain that body, add the smallest RVV dialect/plugin-owned companion op or nested region needed for the finite i32-vadd source/object export path.
* Keep `tcrv.exec` compute-free and focused on execution, capability, variant, dispatch, fallback, and related orchestration.
* Keep RVV-specific control-plane interpretation in RVV dialect/plugin/target-owned code.
* Generic shared code may validate generic structural facts or route through registries/interfaces, but must not branch on RVV/scalar/IME/offload/vendor/dtype/shape/runtime/toolchain/microarchitecture/intrinsic names.
* Preserve parameter layering:
  * VLEN, vlenb, and vector hardware facts stay target capability/profile/probe facts.
  * SEW, LMUL, tail policy, mask policy, and lowering strategy stay compile-time variant/lowering config.
  * AVL/vl/runtime length and dispatch availability stay runtime SSA/control-plane/ABI values when represented.
  * `element_count` remains bounded descriptor/fixture/export metadata unless real IR models it as runtime value.
  * `required_march` may configure target export but must not become broad generic string-comparison logic.
* Existing RVV+scalar dispatch source/object export should continue to work unless intentionally migrated with equivalent or stronger tests.

## Acceptance Criteria

* [x] Focused test proves selected i32-vadd RVV path materializes or requires structured `tcrv_rvv.setvl` and `tcrv_rvv.with_vl` control-plane body.
* [x] Focused test proves RVV exporter validates or consumes that body rather than silently ignoring stale or mismatched descriptor/control metadata.
* [x] Checks make parameter layering visible: runtime n/AVL/vl remains runtime/control-plane/ABI while `element_count` remains descriptor metadata.
* [x] Scalar fallback and RVV+scalar dispatch export path still has deterministic metadata and does not require generic core logic to interpret scalar/RVV semantics.
* [x] Malformed or ambiguous structured RVV body surfaces fail with bounded diagnostics before export.
* [x] `git diff --check` passes.
* [x] `cmake --build build --target check-tianchenrv -j2` passes after configure refresh if needed.

## Out of Scope

* No Python compiler/runtime behavior implementation.
* No supervisor-policy edit unless the exact policy files become dirty before compiler work.
* No docs-only, registry-only, guardrail-only, smoke-probe-only, or evidence-packaging-only closeout.
* No new generic compute ops in `tcrv.exec`.
* No hard-coded target-family/vendor/toolchain branches in generic core passes or generic target routing.
* No arbitrary RVV lowering, generic vector lowering, full runtime ABI integration, scheduler, broad profile lattice, full conflict solver, or performance tuning.
* No RVV runtime/correctness/performance claim unless backed by bounded `ssh rvv` compile/run evidence under `artifacts/tmp`.

## Technical Notes

* Files to inspect first are listed in the user prompt and include `AGENTS.md`, `README.md`, relevant `.trellis/spec/` layers, Exec/RVV/Scalar dialect ODS/C++, capability model, RVV/scalar plugins, lowering/materialization/emission transforms, target exporters, `tcrv-opt`, `tcrv-translate`, and focused tests.
* If generated C/object output changes, update only relevant FileCheck expectations.
* Final state must include exactly one coherent compiler/runtime commit for this slice, leave the worktree clean, and archive the Trellis task.
