# Generic Cost-Aware Variant Selection Planning

## Goal

Add a bounded C++/MLIR selection planning layer that consumes materialized `tcrv.exec.variant` operations, generic target capabilities, and plugin-local cost rankings to produce deterministic static or runtime dispatch plans. This round must not implement target-specific plugins, lowering, emission, runtime ABI, hardware probes, or performance claims.

## What I Already Know

- The repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current `HEAD` is expected to include `4a6e80e feat: add plugin variant cost ranking`.
- The existing pipeline already has proposal collection, capability validation, materialization, plugin-local legality routing, generic decision metadata, dispatch synthesis, and plugin-local cost ranking.
- The live-code gap is that selection is not represented as a reusable compiler decision, and existing dispatch synthesis remains capability/IR-order based rather than consuming plugin cost ranking.
- Implementation must remain in MLIR/C++/TableGen/CMake/lit or C++ tests.
- Python is allowed only for runner, supervisor, remote probe, artifact parsing, and small support scripts.

## Requirements

- Add a generic C++/MLIR selection planning API, preferably in `include/TianChenRV/Transforms/VariantSelection.h` and `lib/Transforms/VariantSelection.cpp`.
- Consume real materialized `tcrv.exec.variant` operations, a `tcrv.exec.kernel`, `TargetCapabilitySet`, and `ExtensionPluginRegistry::rankKernelVariantsByCost`.
- Define explicit, testable result types such as a plan, cases, and a selection kind covering static selection, runtime dispatch, fallback-only, or no viable variant.
- Collect only direct `tcrv.exec.variant` children of the kernel.
- Preserve deterministic ranking: plugin cost order first, stable original IR order for ties via the existing ranking API.
- Require selected/dispatched variants to have valid generic `requires` metadata and origin-owned cost information.
- Allow static selection when the lowest-ranked viable variant is generically available and no lower-cost runtime-guarded variant must be retained.
- Allow runtime dispatch when non-fallback variants have non-empty generic condition, guard, or policy metadata and a generically available fallback exists.
- Always require fallback to be generically available under the same generic target capabilities.
- Pick fallback by cost ranking rather than old IR order when the best fallback is not first in the IR.
- Do not parse or branch on the contents of condition, guard, or policy strings.
- Reject or diagnose unavailable variants without generic decision metadata according to a documented generic rule.
- Add a helper to materialize planned runtime dispatch into typed `tcrv.exec.dispatch`, `tcrv.exec.case`, and `tcrv.exec.fallback` ops, preserving planned case order and copying generic decision metadata.
- Reject kernels that already contain a direct dispatch during materialization unless the implementation deliberately documents and tests another behavior.
- Do not erase variants.
- Keep `tcrv-synthesize-variant-dispatch` working without requiring a runtime plugin registry.

## Acceptance Criteria

- [ ] C++ tests cover lowest-cost static/fallback choice independent of IR order.
- [ ] C++ tests cover equal-cost stable tie-breaking by original kernel IR order.
- [ ] C++ tests cover runtime-guarded lower-cost variants retained as dispatch cases when a fallback exists.
- [ ] C++ tests cover dispatch case order following cost ranking and stable ties.
- [ ] C++ or lit tests cover typed dispatch materialization and metadata copying.
- [ ] Negative tests cover no direct variants, missing available fallback, unavailable unguarded variants, existing direct dispatch rejection, ranking failure propagation, and missing kernel/body or out-of-kernel variants.
- [ ] Existing plugin, transform, exec dialect, capability model, and capability-requires tests continue to pass.
- [ ] `git diff --check` passes.
- [ ] CMake configure and `check-tianchenrv` pass locally.

## Out of Scope

- Python implementation of IR, dialects, passes, plugin interfaces, capability decisions, selection, dispatch planning, lowering, or emission.
- Concrete RVV, IME, offload, AME, Sophgo, scalar, vendor, or target-family plugin implementation.
- Target-specific branches in core selection planning or tests, except inert mock plugin names/data that do not drive core branching.
- Generic compute operations inside `tcrv.exec`.
- Variant erasure, extension lowering, emission, runtime ABI, tuning engine, profile database, hardware probes, or performance claims.
- Public `tcrv-opt` pass unless it performs real tested selection behavior.

## Technical Notes

- Required initial inspection includes repository status, recent history, core specs, plugin protocol specs, variant-pipeline specs, capability specs, testing specs, and the existing Exec/Plugin/Capability/Variant transform sources and tests named in the user request.
- If Trellis task state is updated, it must be finished and archived coherently before final commit.
