# Plugin-local RVV binary proposal legality planning

## Goal

Extract the next coherent RVV binary planning responsibility from
`RVVExtensionPlugin.cpp` into the plugin-local C++ `RVVBinaryPlanning` module:
finite binary proposal planning and required capability metadata. The RVV
extension plugin should consume a structured planner result instead of directly
owning finite-family proposal shape, selected vector-shape requirement ids,
descriptor element count, and capacity metadata decisions.

## Why Now

The archived `rvv-binary-planning-extraction` task completed selected emission
identity planning. The remaining concentration in `RVVExtensionPlugin.cpp` is
earlier in the pipeline: frontend family marker interpretation, finite i32/i64
shape requirements, descriptor-local element count derivation, and proposal
metadata assembly are still decided inline in the monolithic plugin file.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree for this round is clean at
  `e29ab7a refactor(rvv): extract binary emission identity planning`.
- There was no active `.trellis/.current-task`; this task was created from the
  Hermes brief and started as the current task.
- The previous task is finished and archived at
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-planning-extraction`; it
  must remain closed.
- The prior extraction added `RVVBinaryEmissionIdentity`; this round should not
  redo that work or turn into an evidence/script task.
- The finite RVV binary family scope remains exactly `i32-vadd`, `i32-vsub`,
  `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`.
- Route ids, artifact kinds, runtime ABI names, and dispatch route facts remain
  target-owned finite family / manifest facts.
- Compiler behavior must remain implemented in C++ / MLIR / LLVM / TableGen /
  CMake / lit / FileCheck; Python is not an implementation language for
  plugin planning, legality, lowering, emission, or route selection.

## Requirements

- Inventory the RVV plugin paths that build `VariantProposal`, attach required
  capability metadata, interpret finite binary family descriptors, validate
  selected vector-shape requirements, and feed later selected-plan validation.
- Extract proposal planning and required capability metadata into
  `RVVBinaryPlanning` or an equivalently plugin-local C++ module.
- The extracted API must return structured planning data derived from the
  target-owned finite RVV binary family descriptors plus generic
  `TargetCapabilitySet` facts.
- The structured plan must cover at least one i32 and one i64 family, including
  `i64-vmul` as the representative path already proven in dispatch evidence.
- `RVVExtensionPlugin.cpp` must call the new API and stop directly owning the
  extracted finite-family proposal and requirement decision logic.
- Preserve parameter layering: hardware/profile facts, compile-time selected
  vector-shape metadata, runtime ABI descriptors, runtime SSA/control values,
  and descriptor-local element count must remain distinct.
- Preserve manifest-backed route ownership and avoid introducing a second
  plugin-local route table.
- Add focused C++ coverage for the new planner API and keep RVV extension
  plugin behavior tests passing.

## Acceptance Criteria

- [x] `RVVBinaryPlanning` exposes a named structured proposal-planning API for
      finite RVV binary families.
- [x] The planner derives selected finite family, selected vector shape,
      required capability ids, descriptor-local element count, required march,
      and optional capacity metadata from descriptors plus capabilities.
- [x] `RVVExtensionPlugin.cpp` consumes that planner result when building
      proposals and no longer duplicates the finite-family proposal/requirement
      decision path.
- [x] Focused C++ tests cover an i32 proposal plan, an i64 proposal plan, and
      the `i64-vmul` dispatch representative's selected RVV component facts.
- [x] Existing RVV extension plugin public behavior remains unchanged.
- [x] No new RVV runtime/correctness/performance claim is made unless backed by
      fresh real `ssh rvv` evidence.

## Completion Result

This round extracted finite RVV binary proposal planning into
`RVVBinaryPlanning`. The new `RVVBinaryProposalPlan` API derives the selected
finite family, selected vector shape, required capability ids, descriptor-local
element count, required march, optional capacity facts, and generic
condition/guard/policy metadata from target-owned family descriptors plus
`TargetCapabilitySet` facts.

`RVVExtensionPlugin.cpp` now consumes that structured plan while constructing
`VariantProposal`; it no longer directly owns frontend family lookup,
selected-shape requirement id construction, descriptor element-count derivation,
or capacity metadata decisions for proposal generation. The same module also
now owns reusable capability/property view and materialized variant finite
shape requirement helpers used by legality and selected-plan paths.

The external diagnostics were preserved after self-repair: the first full
`check-tianchenrv` run exposed five lit failures caused only by changed
diagnostic wording after moving the helpers. The planner error text was adjusted
back to the existing public wording, and the full suite then passed.

No runtime artifacts changed, no Python runner changed, and this round makes no
new RVV runtime/correctness/performance claim; no fresh `ssh rvv` run was
required.

## Validation Performed

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-planning-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-rvv-binary-planning-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-binary-planning|rvv-extension-plugin|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e'`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  first run failed 5 lit diagnostics after helper extraction; after restoring
  public diagnostic wording, rerun passed `194/194`.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-binary-proposal-legality-planning`

## Non-goals

- No new RVV family, dtype, vector shape, mask/tail policy, generic vector
  dialect route, or generic RVV backend claim.
- No performance benchmarking or speedup claim.
- No Python implementation of compiler IR, dialects, passes, plugin registry,
  capability model, lowering, emission, route selection, proposal planning, or
  runtime ABI decisions.
- No compute semantics in `tcrv.exec`.
- No RVV-specific branches in generic core passes.
- No broad smoke matrix, dashboard/status/report-only work, route-count-only
  cleanup, or evidence-schema-only change as the main result.
- No credentials, tokens, passwords, or connection strings in source, tasks,
  artifacts, or reports.

## Minimal Validation Plan

- `git diff --check`
- Build affected tools/tests at minimum: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, and the finite family registry
  test if touched.
- Run focused C++ binaries for RVV binary planning, RVV extension plugin
  behavior, target artifact export, and finite family registry if relevant.
- Run focused lit from `artifacts/tmp/tianchenrv-build/test` covering
  `rvv-binary-planning`, `rvv-extension-plugin`,
  `rvv-scalar-dispatch-e2e`, and `rvv-scalar-dispatch-bundle-e2e`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  because this round changes public plugin planning behavior.
- Do not run Python script checks unless script files change.
- Do not run `ssh rvv` unless generated runtime artifacts change or a fresh
  runtime/correctness claim is made.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-planning-extraction/prd.md`
  and `.trellis/workspace/codex/journal-2.md`.
- Primary code surfaces:
  `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `test/Plugin/RVVBinaryPlanningTest.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`,
  `include/TianChenRV/Target/RVVScalarDispatch.h`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`.
