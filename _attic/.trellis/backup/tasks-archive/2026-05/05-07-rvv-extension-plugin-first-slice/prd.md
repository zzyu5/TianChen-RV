# RVV extension plugin first slice

## Goal

Add a bounded, plugin-local C++ RVV extension plugin first slice for TianChen-RV MLIR. The slice should exercise the existing ExtensionPluginRegistry, capability participation, proposal materialization, legality, and selection planning paths without adding RVV lowering, runtime ABI, executable kernels, or performance claims.

## What I already know

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* HEAD is expected to include `c2dd412 feat: expose registry-injected variant selection pass`.
* The existing pipeline has proposal collection, capability validation, proposal materialization, plugin-local legality, plugin-local cost ranking, generic selection planning, and a public selection pass injection point.
* The current gap is the absence of a concrete extension plugin in the repository.
* The implementation must remain C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck or C++ tests.
* Python is allowed only for runner/supervisor/probe/artifact support, not core compiler internals.

## Requirements

* Add a bounded C++ RVV extension plugin implementation in a plugin-local layout.
* Provide safe lifetime registration APIs such as `registerRVVExtensionPlugin(ExtensionPluginRegistry &)` and, if useful, `registerBuiltinExtensionPlugins(ExtensionPluginRegistry &)`.
* Use the existing `TargetCapabilitySet` to decide whether RVV proposals are supported.
* Propose a deterministic minimal `tcrv.exec.variant` metadata proposal only when an explicit RVV capability is present and available.
* Keep RVV-specific behavior plugin-local behind `ExtensionPluginRegistry`; do not add target-family branches to core passes.
* Keep `tcrv.exec` compute-free and avoid adding RVV-specific fields or operations there.
* Do not add RVV lowering, vector codegen, runtime ABI, benchmarks, executable kernels, or fake hardware/performance claims.
* Add C++ tests and any needed lit wrappers/CMake wiring.
* Update durable spec files for capability names, ids, symbols, and scope boundaries.
* Archive this Trellis task before the final commit.

## Acceptance Criteria

* [ ] Registering the RVV plugin succeeds and duplicate registration is rejected by existing registry behavior.
* [ ] Registry capability collection exposes RVV plugin capability metadata.
* [ ] The plugin proposes no variant when the required RVV capability is missing or unavailable.
* [ ] With available RVV capability, the plugin proposes deterministic RVV-origin metadata with required capability ids/symbols.
* [ ] Generic materialization turns the proposal into typed `tcrv.exec.variant` IR.
* [ ] Existing legality/capability checks accept the materialized RVV variant when capability is available.
* [ ] Existing selection planning or injected pass consumes RVV-origin variants through `ExtensionPluginRegistry` without core target-specific branches.
* [ ] Malformed RVV-style plugin proposals are rejected by generic validation.
* [ ] Empty-registry selection remains honestly diagnosed and invents no costs or variants.
* [ ] Local checks pass: `git diff --check`, CMake configure, and `check-tianchenrv`.

## Out of Scope

* RVV lowering, emission, runtime ABI, executable path, correctness proof, benchmarking, or performance model.
* Minimal `tcrv.rvv` dialect skeleton unless repository inspection shows it is the cleanest bounded proof; otherwise plugin dialect registration may remain an explicit no-op.
* Any target-family branches in core passes or generic capability/variant code.
* Any Python implementation of compiler internals.

## Technical Notes

* Required inspection and spec-reading list is provided in the user request and will be followed before code edits.
* If any actual RVV hardware/toolchain claim is made, it must be backed by bounded non-destructive `ssh rvv` evidence stored only under `artifacts/tmp/...` and not committed.
