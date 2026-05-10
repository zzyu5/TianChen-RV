# Fixture-free RVV dispatch object path from frontend pipeline

## Goal

Prove one bounded compiler-pipeline path for the finite RVV binary family
`i64-vmul`: marked frontend binary IR reaches `tcrv.exec` planning, RVV plugin
materialization/selection, selected lowering or emission route metadata, and a
manifest-backed RVV+scalar artifact export path without using a hand-authored
selected fixture as the primary boundary.

## Why Now

The previous task `rvv-selected-dispatch-object-evidence` completed selected
`i64-vmul` RVV+scalar dispatch object and self-check object evidence at commit
`dc4146e`. That proves the target/export side for selected metadata. This task
must move the ownership boundary upstream: selected route metadata should be
produced by the compiler pipeline from fixture-free frontend input and then
validated by target-owned manifests.

## Scope

- Representative finite family: `i64-vmul`.
- Preserve the finite binary family set exactly:
  `i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, `i64-vmul`.
- Compiler implementation remains in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
- Python may only orchestrate tools, collect evidence, and parse emitted
  artifacts.
- RVV semantics and route ownership remain plugin/target-local; generic core
  passes must not gain RVV-specific branches.
- `tcrv.exec` remains compute-free and only models execution planning objects,
  not tensor/tile/vector compute semantics.

## Requirements

- Inventory whether existing marked linalg/RVV binary frontend lowering and the
  execution planning pipeline already produce selected RVV+scalar dispatch
  emission metadata for `i64-vmul`.
- Add or repair the smallest C++/MLIR pipeline wiring needed for fixture-free
  `i64-vmul` frontend input to produce selected dispatch artifact route
  metadata consumed by the manifest-backed exporter.
- If the existing pipeline already produces that metadata, add focused lit/C++
  coverage proving the path instead of duplicating pipeline code.
- Ensure evidence records:
  selected pipeline route id, manifest route id, artifact kind, artifact
  path/hash, and whether export came from the fixture-free pipeline path.
- Preserve direct RVV microkernel and RVV+scalar dispatch as distinct
  target-owned surfaces.
- Do not make a runtime/correctness claim unless a real `ssh rvv` run is
  performed for the fixture-free path.

## Acceptance Criteria

- [x] A focused fixture-free frontend pipeline test for `i64-vmul` proves the
      compiler produces selected RVV+scalar dispatch route metadata.
- [x] Manifest-backed export evidence validates the selected route id against
      target-owned manifests and records artifact identity.
- [x] No selected hand-authored fixture is the primary source of route metadata
      in the new proof.
- [x] Relevant affected tools/tests build.
- [x] Focused lit and C++ checks for the changed surfaces pass.
- [x] Trellis task status and journal truthfully record the completed edge and
      any remaining continuation point.

## Completion Result

The inventory found that the existing marked linalg RVV binary frontend
lowering plus `tcrv-execution-planning-pipeline` already reaches selected
RVV+scalar dispatch metadata for the representative `i64-vmul` path. This
round therefore did not add duplicate pipeline logic. Instead, it added focused
evidence surfaces proving the fixture-free path:

- marked frontend `i64-vmul` linalg input lowers through
  `--tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline`;
- the pipeline-produced selected route metadata is consumed by manifest-backed
  RVV+scalar dispatch object and self-check object export;
- the runner records selected route id, manifest route id, artifact kind,
  artifact path/hash, and `fixture_free_frontend_pipeline`;
- bundle front-door dry-run evidence records the same fixture-free provenance
  for source/header/object route records.

No new runtime/correctness claim is made in this round, and no new `ssh rvv`
run was performed. The completed edge is fixture-free frontend pipeline to
manifest-backed source/header/object and direct dispatch/self-check object
export evidence for `i64-vmul`.

## Validation Performed

- `git diff --check`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Fixture-free direct dry run:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i64-vmul --lower-linalg-frontend --run-id lit-rvv-scalar-i64-vmul-frontend-dispatch-e2e --overwrite --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir`
- Fixture-free bundle dry run:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --run-id lit-rvv-scalar-i64-vmul-dispatch-bundle-plan-and-export --overwrite`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-scalar-i64-vmul-dispatch-object|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e'`
- Broader focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'plan-linalg-i64-vmul-and-export-target-artifact-bundle|rvv-scalar-i64-vmul-dispatch-object|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e|target-artifact-bundle'`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  `194/194` passed

## Non-goals

- No new RVV family, dtype, vector shape, mask/tail policy, generic vector
  dialect route, or generic RVV backend claim.
- No performance claim or benchmark target.
- No Python implementation of compiler IR, passes, plugin registry, lowering,
  emission, route selection, or runtime ABI decisions.
- No compute semantics in `tcrv.exec`.
- No RVV-specific branches in generic core passes.
- No broad smoke matrix, dashboard/status/report-only work, or route-count-only
  cleanup as the main result.
- Do not weaken the selected fixture/object/self-check evidence from the
  archived previous task.
- Do not store credentials, tokens, passwords, or connection strings in source,
  tasks, artifacts, or reports.

## Minimal Validation Plan

- `git diff --check`
- Build at minimum: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-target-artifact-export-test`, and focused RVV
  planning/registry/export tests touched by the change.
- Run focused C++ tests covering target artifact export and RVV
  planning/manifest behavior when touched.
- Run focused lit from `artifacts/tmp/tianchenrv-build/test` covering the new
  fixture-free frontend pipeline test, selected dispatch object/self-check
  export, target artifact export, and relevant script e2e filter.
- Run `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py` and script
  self-test if the runner changes.
- Run a bounded fixture-free `i64-vmul` dry run if the runner supports it.
- Run real `ssh rvv` only if this round claims runtime/correctness evidence.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if public tool, pass pipeline, target exporter, route registration,
  script, or lit surfaces change.

## Technical Notes

- Archived prerequisite task:
  `.trellis/tasks/archive/2026-05/05-10-rvv-selected-dispatch-object-evidence`.
- Primary code surfaces to inspect:
  `LowerLinalgI32BinaryToExec`, `ExecutionPlanningPipeline`,
  `VariantMaterialization`, `VariantSelection`, `LoweringBoundary`,
  `EmissionReadiness`, `ExecutionPlanCoherence`, RVV plugin planning, RVV
  binary family registry, microkernel/dispatch target exporters, and
  `scripts/rvv_scalar_dispatch_e2e.py`.
