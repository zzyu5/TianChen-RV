# Canonical Execution Planning Pipeline Coherence

## Goal

Make `tcrv-opt --tcrv-execution-planning-pipeline` produce a selected execution plan that has already passed the existing execution-plan coherence gate against the active extension plugin registry and target artifact exporter registry, so downstream target artifact front doors consume a checked plan without requiring a separate manual `--tcrv-check-execution-plan-coherence` step.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial `HEAD` is expected and observed as `2487b52 chore: archive RVV evidence task`.
* Initial worktree is clean.
* This round returns to active MLIR/C++ engineering after an evidence/archive cleanup commit.
* Primary implementation stack is C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
* Python is allowed only for scripts, probes, artifact parsing, evidence orchestration, and small helpers, not compiler internals.
* The existing coherence implementation must be reused through `createCheckExecutionPlanCoherencePass(plugins, targetExporters)`.

## Requirements

* Extend execution-planning pipeline registration/building APIs so the canonical pipeline can receive both `ExtensionPluginRegistry` and `TargetArtifactExporterRegistry`.
* Preserve compatibility for no-argument and plugin-only pipeline builders/registrations by using an explicit empty target exporter registry where needed.
* Keep the existing pipeline sequence: plugin proposal materialization, plugin variant legality, capability-aware selection, capability requirement checking, selected lowering-boundary materialization, and emission-plan materialization.
* Add the existing execution-plan coherence pass as the final canonical pipeline gate after emission plans are materialized.
* Register `--tcrv-execution-planning-pipeline` in `tools/tcrv-opt/tcrv-opt.cpp` with populated built-in plugin and target artifact exporter registries.
* Do not add target artifact exporters, RVV microkernels, offload runtime implementation, evidence scripts, or runtime ABI contracts.
* Keep target artifact route validation exporter-owned and registry-mediated.
* Keep capability, runtime ABI, hardware facts, selected variant config, descriptor metadata, bundle metadata, and evidence metadata as separate layers.

## Acceptance Criteria

* Positive lit/FileCheck coverage proves a selected builtin path passes with `--tcrv-execution-planning-pipeline` alone where it previously needed pipeline plus manual coherence check.
* Negative lit/FileCheck coverage proves a stale, unknown, ambiguous, or artifact-kind-mismatched selected target artifact route fails with `--tcrv-execution-planning-pipeline` alone.
* The tests prove canonical pipeline ownership of the coherence gate, not only label churn.
* `git diff --check` passes.
* `cmake --build build --target tcrv-opt tcrv-translate -j2` passes if the existing build directory is valid.
* `cmake --build build --target check-tianchenrv -j2` or equivalent passes if configured locally.

## Out of Scope

* No new RVV runtime/correctness/performance evidence.
* No `ssh rvv` run unless a new RVV runtime/correctness/performance claim is made.
* No target-specific branches in generic pipeline orchestration beyond registry plumbing.
* No generic compute ops in `tcrv.exec`.
* No performance-looking metadata or claims.
* No unrelated cleanup, broad reformatting, or build/artifact/cached output commits.

## Technical Notes

* Required initial inspection commands were run before source implementation.
* Required files to read before implementation are listed in the user request and will be consulted before source edits.
