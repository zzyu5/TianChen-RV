# plugin-local materialized variant cost ranking

## Goal

Add the next compiler-spine layer for plugin-local materialized-variant cost estimation and deterministic generic ranking, without implementing full selection, lowering, emission, runtime behavior, or target-specific cost rules.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD is expected to include `2d8c2b7 feat: preserve variant decision metadata`.
* Current live-code gap is a reusable C++ plugin protocol and registry orchestration slice for cost estimation/ranking over already materialized `tcrv.exec.variant` ops.
* Implementation must remain C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck or C++ tests.
* Python is only allowed for runner/supervisor/probe/artifact/support scripts, not core IR/protocol/pipeline logic.

## Requirements

* Add a generic `VariantCostRequest` carrying a materialized `tcrv.exec.variant`, enclosing `tcrv.exec.kernel`, and `TargetCapabilitySet`.
* Add a generic `VariantCostEstimate` result with finite non-negative score/rank key, plugin/origin context, variant symbol, and optional non-empty generic explanation/policy text.
* Add an `ExtensionPlugin` cost hook with deterministic safe default behavior.
* Add `ExtensionPluginRegistry` APIs to route one cost request to the plugin named by the variant origin and reject malformed/unknown/disabled/failing/invalid cases with contextual diagnostics.
* Add a kernel-level helper that visits direct variants in IR order, collects estimates, and returns deterministic stable ranking for equal scores.
* Keep proposal collection/materialization/legality/dispatch/full selection/tuning/lowering/emission/runtime ABI/concrete target rules out of scope.
* Update stable specs only where implementation contracts changed.
* Add C++ tests using real MLIR objects and lit/CMake wiring if a new executable is created.

## Acceptance Criteria

* [ ] Positive tests cover routed plugin request, request fields, capability-sensitive generic cost, IR-order collection, stable tie ranking, and explanation/policy preservation.
* [ ] Negative tests cover missing variant/kernel, wrong enclosing kernel, missing origin, unknown origin, disabled plugin, plugin-local failure context, non-finite/negative scores, and empty present explanation/policy fields.
* [ ] Existing plugin/proposal/legalization/materialization/dispatch/capability tests continue passing.
* [ ] `git diff --check` passes.
* [ ] CMake configure with LLVM/MLIR 20 passes.
* [ ] `check-tianchenrv` passes.
* [ ] Worktree is clean and one coherent commit is created.

## Definition of Done

* Tests added/updated.
* Lint/build/test checks pass or failures are explicitly reported with cause.
* Specs updated for durable behavior.
* Trellis task is validated if supported and archived.
* Final report follows the requested eight-part format and invariant checklist.

## Out of Scope

* Python core implementations.
* Concrete RVV/IME/offload/scalar/vendor/shape/layout/runtime-specific cost behavior.
* tcrv.exec compute operations.
* Full selection pass, variant erasure, dispatch rewriting, tuning engine, lowering, emission, runtime ABI, backend work, or hardware performance claims.

## Technical Notes

* Required inspection files and command outputs will be used as the source of truth before editing implementation files.
