# Plugin-local RVV binary planning extraction

## Goal

Extract one coherent RVV binary planning responsibility from
`RVVExtensionPlugin.cpp` into a plugin-local C++/MLIR planning component that is
consumed by the RVV extension plugin and covered by focused tests. The target
module is maintainability and ownership reduction in the RVV plugin surface, not
new runtime evidence.

## Why Now

The archived `rvv-frontend-pipeline-dispatch-runtime-evidence` task completed
the fixture-free marked frontend `i64-vmul` path through the C++/MLIR compiler
pipeline, manifest-backed target export, and real `ssh rvv`
runtime/correctness evidence at `12751f0`. That evidence module is closed and
must not be reopened. The next bottleneck is plugin maturity: RVV binary
planning, finite-family route metadata, lowering-boundary parameters, and
selected artifact metadata remain concentrated in `RVVExtensionPlugin.cpp`.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree for this round is clean at
  `12751f0 feat(rvv): prove fixture-free dispatch runtime evidence`.
- There was no active `.trellis/.current-task`; this task was created from the
  Hermes brief.
- The previous task is archived at
  `.trellis/tasks/archive/2026-05/05-10-rvv-frontend-pipeline-dispatch-runtime-evidence`
  and remains closed.
- The finite binary-family scope remains exactly `i32-vadd`, `i32-vsub`,
  `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`.
- RVV route identity and artifact kinds are target-owned manifest facts. The
  plugin-local planner may consume manifest-backed facts but must not create a
  second hard-coded route table.
- Compiler implementation must remain in C++ / MLIR / LLVM / TableGen / CMake
  / lit / FileCheck. Python is not an implementation language for compiler IR,
  plugin registry, lowering, emission, route selection, or runtime ABI logic.

## Requirements

- Inventory current RVV plugin paths that propose variants,
  validate/interpret finite family metadata, choose direct vs RVV+scalar
  dispatch lowering intent, attach selected vector-shape/runtime ABI/route
  metadata, and feed emission-plan/exporter metadata.
- Move one real RVV binary planning responsibility out of
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp` into an explicit plugin-local C++
  component, preferably via the existing `RVVBinaryPlanning` surface unless the
  code proves a narrower name is more accurate.
- Make the RVV extension plugin consume the new component directly and remove
  or demote the old inlined ownership. Do not leave duplicate logic or a dead
  wrapper that only forwards without reducing concentration.
- Preserve route identity ownership: direct RVV and RVV+scalar dispatch
  manifest route ids/artifact kinds remain target-owned facts.
- Preserve parameter layering: hardware capabilities, compile-time selected
  vector shape, runtime SSA/control values, runtime ABI descriptors, and
  descriptor-local artifact properties must stay distinct.
- Preserve finite family scope exactly.
- Add focused unit-style C++ coverage for the extracted planner API across at
  least one `i32` and one `i64` family, including the already-proven
  `i64-vmul` dispatch path as a representative case.
- Keep existing fixture-free and selected evidence paths passing without making
  new runtime evidence the main deliverable.

## Acceptance Criteria

- [x] A named plugin-local C++ planning API owns a coherent selected RVV binary
      planning responsibility previously embedded in `RVVExtensionPlugin.cpp`.
- [x] `RVVExtensionPlugin.cpp` consumes the extracted API and no longer owns the
      duplicated inlined logic for that responsibility.
- [x] The extraction preserves finite-family scope and rejects unsupported
      families rather than widening behavior.
- [x] The extraction preserves manifest-backed route ownership and does not add
      a second plugin-local route table.
- [x] Focused C++ tests cover at least one `i32` family, one `i64` family, and
      the `i64-vmul` dispatch representative.
- [x] Focused C++/lit validation proves existing planning/plugin/export paths
      still pass.
- [x] If no runtime artifacts or runtime claims change, the final report states
      that no new RVV runtime/correctness claim was made and no `ssh rvv` run
      was required.

## Completion Result

This round extracted the selected RVV binary emission identity responsibility
from `RVVExtensionPlugin.cpp` into the plugin-local `RVVBinaryPlanning` module.
The new `RVVBinaryEmissionIdentity` API derives readiness path, supported
source artifact kind, bounded support explanation, route id, emission kind,
runtime ABI kind/name, and runtime glue role from the target-owned finite RVV
binary family descriptor. It validates that the input descriptor is one of the
registered finite RVV binary families rather than accepting an ad hoc route
record.

`RVVExtensionPlugin.cpp` now consumes that planning identity for both emission
readiness and emission-plan construction. The previous plugin-local
`RVVI32MicrokernelFamilySpec` table and its three duplicated i32
route/message records were removed. Direct RVV route ids and runtime ABI facts
still come from the target-owned family descriptors; RVV+scalar dispatch route
metadata remains target-owned in the dispatch family/manifest surfaces.

The focused planner test now covers:

- direct `i32-vadd` and `i32-vsub` selected emission identity;
- direct `i64` selected-plan metadata, including `i64-vmul`;
- the `i64-vmul` RVV+scalar dispatch representative by checking that the
  dispatch family reuses the selected RVV route/emission/ABI facts while its
  dispatch object route and success marker remain target-owned dispatch facts.

## Inventory

- Proposal path: `RVVExtensionPlugin.cpp` still resolves frontend family
  markers, capability property evidence, selected finite vector shape,
  descriptor-local element count, and selected variant attributes.
- Selected-plan path: `RVVBinaryPlanning` now owns finite family selected-plan
  validation and selected RVV binary emission identity construction.
- Direct emission path: `RVVExtensionPlugin.cpp` finds/validates matching
  selected direct RVV microkernel attachments, then consumes
  `RVVBinaryEmissionIdentity` to populate readiness and emission-plan metadata.
- Dispatch path: selected dispatch composition and dispatch source/header/object
  route manifests remain in target-owned RVV+scalar dispatch code; the plugin
  does not create or duplicate a dispatch route table.
- Parameter layering is unchanged: capability/profile facts, selected
  vector-shape metadata, runtime ABI parameters, runtime guard/control values,
  and descriptor-local element counts remain separate.

## Validation Performed

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-rvv-binary-planning-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-binary-planning|rvv-extension-plugin|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e'`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  `194/194` passed.

No Python runner changed, so no Python py_compile/self-test was needed. No
runtime artifact generation changed and this round makes no new RVV
runtime/correctness claim, so no fresh `ssh rvv` run was required.

## Non-goals

- No new RVV family, dtype, vector shape, mask/tail policy, generic vector
  dialect route, or generic RVV backend claim.
- No performance benchmarking or speedup claim.
- No Python implementation of compiler IR, dialects, passes, plugin registry,
  capability model, lowering, emission, route selection, planning, or runtime
  ABI decisions.
- No compute semantics in `tcrv.exec`.
- No RVV-specific branches in generic core passes.
- No broad smoke matrix, dashboard/status/report-only work,
  route-count-only cleanup, or evidence-schema-only change as the main result.
- Do not weaken existing fixture-free runtime evidence, selected-fixture
  evidence, manifest lookup tests, or bundle tests.
- Do not store credentials, tokens, passwords, or connection strings.

## Minimal Validation Plan

- `git diff --check`
- Build affected plugin/target/tools/tests, at minimum `tcrv-opt`,
  `tcrv-translate`, `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-target-artifact-export-test`, and any RVV extension plugin test
  target touched by the change.
- Run focused C++ binaries covering RVV binary planning, RVV extension plugin
  behavior if touched, target artifact export, and finite family registry
  behavior if touched.
- Run focused lit from `artifacts/tmp/tianchenrv-build/test` covering RVV
  binary planning/plugin tests, fixture-free `rvv-scalar-dispatch-e2e`, and
  bundle dispatch evidence enough to prove behavior did not regress.
- Run `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py` and runner
  self-test only if the runner changes.
- Run real `ssh rvv` only if implementation changes generated runtime
  artifacts or makes a fresh runtime/correctness claim.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if public tool, plugin, target exporter, pass pipeline, route registration,
  script, or lit surfaces change.
- Validate the active Trellis task before finish and the archived task after
  finish if completed.

## Technical Notes

- Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Required prior context:
  `.trellis/tasks/archive/2026-05/05-10-rvv-frontend-pipeline-dispatch-runtime-evidence/prd.md`.
- Primary source/test surfaces:
  `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `test/Plugin/RVVBinaryPlanningTest.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h`,
  `include/TianChenRV/Target/RVVScalarDispatch.h`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `test/Scripts/rvv-scalar-dispatch-e2e.test`,
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`.
