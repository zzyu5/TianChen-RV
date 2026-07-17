# RVV scalar dispatch ABI role binding

## Goal

Make the bounded RVV+scalar i32-vadd dispatch exporter consume structured runtime ABI role bindings instead of exact default C parameter names, matching the existing single-callable RVV emission-plan-backed route.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial precondition confirmed HEAD `0a7d7b9 feat: bind RVV vadd ABI roles structurally`.
* Supervisor-policy files were clean, so this task proceeds as the compiler/runtime dispatch owner rather than a policy-only commit.
* The previous route bound RVV vadd ABI roles structurally for single-callable export while preserving direct no-plan default ABI behavior.
* The current live gap is `lib/Target/Builtin/RVVScalarDispatch.cpp`, where dispatch candidate validation and generated calls still depend on literal default C names.

## Requirements

* Validate RVV and scalar callable candidates by runtime ABI role, C type, and ownership for the four bounded i32-vadd callable roles.
* Require exactly one target-export-owned parameter for lhs input buffer, rhs input buffer, output buffer, and runtime element count on each callable candidate.
* Reject missing, duplicate, wrong-type, wrong-ownership, unsafe extra, or unknown role metadata before dispatch export with bounded diagnostics.
* Build deterministic dispatcher public ABI parameters from role-bound metadata, using the selected RVV candidate names/types for the four data roles and appending a dispatch availability guard role parameter.
* Generate dispatcher signature, guard branch, RVV call, and scalar call from role-resolved variable names and role order, not literal `lhs`, `rhs`, `out`, `n`, or `rvv_available`.
* Keep embedded RVV and scalar callable source target-owned and calls positional by role order.
* Preserve direct no-plan fallback behavior only where it is already explicitly target/export-owned.
* Keep `tcrv.exec` compute-free and keep concrete i32-vadd computation in RVV/scalar dialect and target-owned exporter surfaces.

## Acceptance Criteria

* [x] Dispatch C export accepts structurally compatible RVV/scalar callable candidates whose parameter names differ from default names.
* [x] Generated dispatcher C uses role-resolved public parameter names including a non-default dispatch availability guard name.
* [x] Generated RVV and scalar calls pass arguments in role order.
* [x] Negative tests fail closed for malformed runtime ABI role metadata relevant to this route.
* [x] Existing RVV microkernel, scalar fallback, dispatch source, and dispatch object/self-check routes still pass.
* [x] Local checks include `git diff --check` and `cmake --build build --target check-tianchenrv -j2`.

## Definition of Done

* Minimal C++/MLIR/TableGen/lit/FileCheck implementation and tests are committed as one coherent compiler/runtime commit.
* No Python compiler/runtime implementation is introduced.
* No supervisor-policy files, build directories, artifacts, caches, or unrelated Trellis scratch are committed.
* Trellis task state is finished and archived.

## Out of Scope

* No new RVV runtime/correctness/performance claim unless exactly one bounded `ssh rvv` self-check route is run and sanitized evidence is stored under `artifacts/tmp`.
* No broad smoke/probe matrix, generic ABI framework, conflict solver, profile lattice, provider ranking, runtime scheduler, arbitrary RVV lowering, or generic vector lowering.
* No generic target-family branching in core passes or shared target routing.

## Technical Notes

* Inspect first: `AGENTS.md`, `README.md`, relevant `.trellis/spec/` files, runtime ABI support headers, RVV/scalar ODS, RVV/scalar target exporters, plugin descriptors, dispatch exporter, artifact fixtures, and related lit tests.
* Parameter layering must remain explicit: target hardware facts, compile-time variant/lowering config, runtime SSA/control/ABI values, and descriptor-local fixture metadata stay distinct.
