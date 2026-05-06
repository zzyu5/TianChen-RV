# worker: capability milestone from audit

## Goal

Close the next real compiler milestone after `dabdaa3`: connect the generic variant selection planner to an MLIR pass with an explicit extension plugin registry injection path, so selection can be run through pass infrastructure without hard-coding RVV/IME/offload/scalar behavior in core orchestration.

## What I already know

* Repo root is `/home/kingdom/phdworks/TianchenRV`; current HEAD is `dabdaa3 feat: add generic variant selection planning` and the worktree was clean before this task.
* Latest available supervisor audit with review input is `artifacts/tmp/hermes_codex_supervisor/runs/20260506T144732Z-r0015-20260506T191710Z/`; a newer `r0016` directory exists but has no `repo_audit.md` / `review_input.md`.
* The latest completed round added `VariantSelection.h/.cpp` C++ planner APIs and C++ tests, but explicitly left “no public `tcrv-opt` selection pass” and “Future integration still needs a real registry injection path before selection can become a public pass.”
* Specs require core selection to consume `ExtensionPluginRegistry::rankKernelVariantsByCost`, preserve plugin locality, avoid target-family branches, and materialize runtime dispatch only through typed `tcrv.exec.dispatch` / `tcrv.exec.case` / `tcrv.exec.fallback`.
* User explicitly required single full-access worker execution and no subagents; this task uses Trellis only for local task state and context files.

## Requirements

* Add a core MLIR module pass for generic variant selection that can be constructed with an injected `ExtensionPluginRegistry`.
* Register a default public `tcrv-opt` pass name that is honest when no plugin registry is installed: it may no-op on kernels without variants, but must diagnose unknown/unregistered origin plugins instead of inventing Python or attribute-only costs.
* Preserve the existing planner and materialization contract: static/fallback-only plans do not invent target-specific IR; runtime-dispatch plans materialize typed dispatch IR and reject competing existing dispatch.
* Keep selection target-neutral and plugin-local: no RVV, IME, Sophgo/offload, scalar, dtype, shape, vendor, ABI, or microarchitecture branches in core pass code.
* Add tests for injected-registry pass execution and default public pass diagnostics.
* Update durable specs for the pass/injection contract.

## Acceptance Criteria

* [ ] `createSelectVariantsPass(const ExtensionPluginRegistry &registry)` exists and is covered by a C++ pass-manager test with mock plugins.
* [ ] `--tcrv-select-variants` is registered in `tcrv-opt` and emits clear diagnostics if variants name unregistered origin plugins.
* [ ] Runtime dispatch materialized by the pass verifies and is accepted by the capability-requires pass.
* [ ] Specs document that public default pass registration does not replace plugin-owned cost models and that production tools must inject a populated registry.
* [ ] CMake/lit build checks pass, or exact missing tool diagnostics are recorded.

## Definition of Done

* Tests added/updated using lit/FileCheck and C++ where appropriate.
* CMake configure/build/check target passes locally if MLIR tools are available.
* Trellis task validates and is archived or clearly reported.
* One coherent commit is created unless validation blocks it.

## Out of Scope

* Concrete RVV/IME/offload/scalar plugin implementation.
* Lowering/emission/runtime ABI or real RVV correctness/performance claims.
* Python compiler representations or Python-only pass simulation.
* Hardware `ssh rvv` claims; no RVV runtime result is targeted in this round.

## Technical Notes

* Relevant specs: `.trellis/spec/implementation-stack/compiler-stack-contract.md`, `.trellis/spec/architecture/design-boundaries.md`, `.trellis/spec/capability-model/capability-contract.md`, `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
* Relevant implementation: `include/TianChenRV/Transforms/VariantSelection.h`, `lib/Transforms/VariantSelection.cpp`, `include/TianChenRV/Transforms/Passes.td`, `include/TianChenRV/Transforms/Passes.h`, `tools/tcrv-opt/tcrv-opt.cpp`, `test/Transforms/VariantSelectionTest.cpp`.
