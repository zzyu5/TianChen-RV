# IR-linked dispatch runtime guard

## Goal

Make bounded RVV+scalar dispatch C generation derive the selected RVV branch guard from an explicit `tcrv.exec.case` symbol reference to a same-kernel `tcrv.exec.runtime_param` with ABI role `dispatch-availability-guard`, instead of relying on detached role lookup or free-form guard metadata alone.

## Requirements

* Add one narrow optional symbol-ref attribute on `tcrv.exec.case`, preferred spelling `runtime_guard = @abi_dispatch_availability_guard`.
* Verify the symbol generically in the core exec dialect when present: same-kernel resolution, referenced op kind is `tcrv.exec.runtime_param`, and role is `dispatch-availability-guard`.
* Preserve existing free-form `guard` metadata as plugin-local descriptive text if still useful.
* Update bounded builtin RVV+scalar planning/synthesis so exactly one dispatch availability runtime param is materialized and attached to the selected RVV case through the new symbol reference.
* Update RVV+scalar dispatch C export so branch control reads the guard C name/type/ownership through the selected case symbol reference, and fails closed on missing, unresolved, wrong role/type/ownership, duplicate/ambiguous, or stale-detached metadata cases.
* Preserve existing IR-backed callable ABI behavior for RVV/scalar microkernel lhs/rhs/out/n parameters.
* Keep generic exec verification target-neutral and keep RVV/scalar details in plugin-local or target-owned code.

## Acceptance Criteria

* `tcrv.exec.case` round-trips with the new `runtime_guard` symbol attribute.
* Verifier accepts a case runtime guard that resolves to a same-kernel runtime param with role `dispatch-availability-guard`.
* Execution planning emits `tcrv.exec.runtime_param @abi_dispatch_availability_guard` and attaches it to the RVV case.
* RVV+scalar dispatch C export emits `if (<guard c_name>)` from the referenced runtime param and still emits RVV/scalar callable calls using `RuntimeABICallablePlan` derived lhs/rhs/out/n params.
* Changing the runtime param C name changes the generated dispatch condition through the symbol-linked param.
* Narrow negative tests cover missing runtime guard for the exporter, unknown/non-runtime-param symbols, wrong role/type/ownership, and stale detached dispatch metadata disagreement.
* `git diff --check`, `cmake --build build --target tcrv-opt tcrv-translate -j2`, and `cmake --build build --target check-tianchenrv -j2` pass.
* Because generated dispatch C behavior changes, run bounded `ssh rvv` evidence path or report the blocker without runtime/correctness/performance claims.

## Definition of Done

* Implementation remains C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck, with Python only for allowed orchestration if used.
* Tests added or updated for compiler behavior and diagnostics.
* Relevant spec/doc notes updated if the IR contract changes.
* Trellis task validates and is archived before commit.
* Final repository state is clean with a focused compiler commit.

## Out of Scope

* Broad runtime probing, generic dispatch predicate IR, arbitrary condition evaluation, variant cost/tuning, performance optimization, new backend support, offload/IME/AME work.
* Python implementation of IR, ABI modeling, target routing, lowering, or emission decisions.
* Broad negative matrix, docs-only closeout, helper-only closeout, or hand-authored C fixture as proof.

## Technical Notes

* Required preflight on 2026-05-08 confirmed repo root `/home/kingdom/phdworks/TianchenRV`, clean worktree, clean supervisor-policy files, and HEAD `43dc6eb feat: derive callable ABI from exec IR`.
* User specified exact files to inspect before editing and exact final report sections.
* No subagents or parallel agent workflow will be used in this task.
