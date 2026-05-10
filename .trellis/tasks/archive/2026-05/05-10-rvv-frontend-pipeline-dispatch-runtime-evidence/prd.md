# Fixture-free RVV dispatch runtime evidence from frontend pipeline

## Goal

Complete one bounded live evidence path for fixture-free marked frontend
`i64-vmul`: marked frontend IR lowers through the compiler pipeline to selected
RVV+scalar dispatch route metadata, target-owned manifests validate and export
the dispatch object/self-check object artifacts, and the exported self-check
path runs successfully on the real `ssh rvv` target.

## Why Now

The archived `rvv-selected-dispatch-object-evidence` task proved selected
`i64-vmul` dispatch object and self-check object export through target-owned
manifests, including real `ssh rvv` runtime evidence. The archived
`rvv-frontend-pipeline-dispatch-object-evidence` task then moved the route
metadata ownership boundary upstream to the fixture-free marked frontend
pipeline, but intentionally stopped at dry-run artifact evidence and made no
fresh runtime claim. This task connects those two completed edges without
reopening either archived task.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree for this round is clean at
  `6cdf5eb feat(rvv): prove fixture-free dispatch route evidence`.
- There was no `.trellis/.current-task`; this task was created from the Hermes
  brief and started as the current Trellis task.
- Prior selected runtime evidence is archived at
  `.trellis/tasks/archive/2026-05/05-10-rvv-selected-dispatch-object-evidence`.
- Prior fixture-free dry-run route evidence is archived at
  `.trellis/tasks/archive/2026-05/05-10-rvv-frontend-pipeline-dispatch-object-evidence`.
- The current representative family is exactly `i64-vmul`; the finite family
  scope remains exactly `i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`,
  `i64-vsub`, and `i64-vmul`.
- Python may orchestrate compiler invocations, artifact parsing, copying,
  remote compile/link/run, and evidence serialization only. Route identity,
  lowering/emission metadata, runtime ABI, family facts, dispatch semantics,
  and manifest ownership must remain in the C++/MLIR/plugin/target stack.

## Requirements

- Inventory the current fixture-free dry-run evidence from the archived
  frontend pipeline task and the live selected-object evidence from the
  archived selected dispatch object task.
- Extend or repair the existing runner/lit workflow so
  `--lower-linalg-frontend --arithmetic-family=i64-vmul` can run live against
  `ssh rvv`, export manifest-backed dispatch object/self-check object
  artifacts from compiler-produced selected metadata, copy/link/run remotely,
  and record structured evidence.
- Preserve the existing selected-fixture evidence path. New checks must remain
  explicit about whether they exercise selected fixture provenance or
  fixture-free frontend pipeline provenance.
- Evidence must record at least:
  `fixture_free_frontend_pipeline: true`, frontend lowering/pipeline command or
  label, selected route id, manifest route id, artifact kind, object/self-check
  artifact paths, artifact hash, remote compile/link/run status, remote link
  flags, stdout/stderr summary, and expected marker
  `tcrv_rvv_scalar_i64_vmul_dispatch_self_check_ok` or the manifest-provided
  marker for the chosen finite family.
- If the runner already supports the live fixture-free path, prefer focused
  lit/check evidence assertions over redundant helper layers.
- Keep the runtime/correctness claim bounded to fixture-free `i64-vmul`
  RVV+scalar dispatch self-check execution. Do not generalize to all families
  or a generic RVV backend.

## Acceptance Criteria

- [x] The fixture-free marked frontend `i64-vmul` path reaches selected
      RVV+scalar dispatch route metadata through
      `--tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline`.
- [x] Target-owned manifest lookup validates the selected route and exports the
      dispatch object and self-check object artifacts for the fixture-free
      pipeline path.
- [x] The self-check artifact is copied/linked/run on real `ssh rvv` and the
      recorded evidence proves the expected success marker.
- [x] Structured evidence distinguishes fixture-free compiler-pipeline
      provenance from the older selected-fixture path.
- [x] Existing selected-fixture object/self-check runtime evidence remains
      intact and explicitly scoped.
- [x] Focused checks pass, or the task remains open with the exact blocker and
      smallest next continuation point.

## Completion Result

The runner already supported the live fixture-free direct path, so this round
did not add a duplicate helper or move route ownership. The implemented change
tightened structured evidence for that path:

- direct evidence now records the frontend pipeline command and explicit
  `frontend_pipeline_args`, including
  `--tcrv-lower-linalg-rvv-binary-to-exec`;
- live `ssh rvv` evidence summary now preserves remote link flags and bounded
  stdout/stderr tails for compile/link/run steps;
- script self-test checks the new remote evidence summary fields;
- lit coverage now asserts the fixture-free `i64-vmul` direct path records the
  frontend lowering command and route provenance.

The passing live evidence artifact is:

- `artifacts/tmp/rvv_scalar_dispatch_e2e/codex-fixture-free-i64-vmul-runtime-live/evidence.json`

That evidence records `fixture_free_frontend_pipeline: true`, the marked
linalg-to-execution-planning command, selected/manifest route ids
`tcrv-export-rvv-scalar-i64-vmul-dispatch-object` and
`tcrv-export-rvv-scalar-i64-vmul-dispatch-self-check-object`, the generated
self-check object path/hash, remote link flags
`-O2 -march=rv64gcv -mabi=lp64d -no-pie`, successful remote compile/link/run
phases, bounded stdout/stderr summaries, and the expected marker
`tcrv_rvv_scalar_i64_vmul_dispatch_self_check_ok` for both the source-built
self-check executable and the exported self-check object executable.

The runtime/correctness claim is bounded to fixture-free `i64-vmul`
RVV+scalar dispatch self-check execution on real `ssh rvv`. Dispatch object
export is locally validated; the manifest self-check object is remotely linked
and run.

## Validation Performed

- `git diff --check`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Fixture-free dry run:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i64-vmul --lower-linalg-frontend --run-id codex-fixture-free-i64-vmul-runtime-dry --overwrite --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir`
- Fixture-free live run:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --arithmetic-family=i64-vmul --lower-linalg-frontend --run-id codex-fixture-free-i64-vmul-runtime-live --overwrite --input test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir --ssh-target rvv`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-scalar-i64-vmul-dispatch-object|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e'`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  `194/194` passed.

Self-repair performed:

- The first focused lit run failed because the new FileCheck assertions did not
  follow the sorted JSON field order. The test assertions were reordered and
  the same focused lit filter then passed.

Spec update judgment:

- No `.trellis/spec/` update was needed. This round did not introduce a new
  compiler contract, plugin boundary, route family, or testing convention; it
  tightened runner evidence fields for an existing bounded workflow.

## Non-goals

- No new RVV family, dtype, vector shape, mask/tail policy, generic vector
  dialect route, or generic RVV backend claim.
- No performance benchmarking or speedup claim.
- No Python implementation of compiler IR, passes, plugin registry, lowering,
  emission, route selection, runtime ABI, finite family facts, or dispatch
  semantics.
- No compute semantics in `tcrv.exec`.
- No RVV-specific branches in generic core passes.
- No broad smoke matrix, dashboard/status/report-only work, or
  evidence-schema-only change as the main result.
- No weakening of existing selected-fixture object/self-check evidence or
  fixture-free dry-run evidence.
- Do not store credentials, tokens, passwords, or connection strings.

## Minimal Validation Plan

- `git diff --check`
- Build affected tools/tests, at minimum `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-target-artifact-export-test`, and focused RVV planning/registry
  tests touched by the change.
- Run focused C++ test binaries covering target artifact export and RVV
  planning/manifest behavior when touched.
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` if the runner
  changes.
- Fixture-free dry run for `i64-vmul` proving local metadata/artifact
  generation still works.
- Live fixture-free `i64-vmul` run against `ssh rvv`, with expected marker in
  recorded evidence, before making any runtime/correctness claim.
- Focused lit from `artifacts/tmp/tianchenrv-build/test` covering
  fixture-free frontend pipeline evidence, selected dispatch object/self-check
  object export, script e2e, and bundle e2e filters.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if public tool, pass pipeline, target exporter, route registration, script,
  or lit surfaces change.
- Validate the active Trellis task before finish and the archived task after
  finish if completed.

## Technical Notes

- Read specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Read prior PRDs:
  `.trellis/tasks/archive/2026-05/05-10-rvv-selected-dispatch-object-evidence/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-frontend-pipeline-dispatch-object-evidence/prd.md`.
- Primary surfaces to inspect:
  `scripts/rvv_scalar_dispatch_e2e.py`,
  script lit tests, selected `i64-vmul` dispatch object tests,
  frontend lowering, execution planning, lowering boundary, emission readiness,
  RVV+scalar target artifact export, and tool front doors.
