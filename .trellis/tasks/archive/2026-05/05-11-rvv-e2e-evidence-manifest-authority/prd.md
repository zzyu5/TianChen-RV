# RVV e2e evidence manifest authority boundary

## Goal

Make generated compiler/exporter manifests, target artifact bundle indexes, and
sanitized command summaries the authority for the bounded RVV direct
microkernel and RVV+scalar dispatch `i32-vmul` e2e evidence workflows. Python
runners may orchestrate tools, copy/hash artifacts, execute remote commands,
and compare explicit CLI assertions, but they must not be the source of truth
for compiler-selected route ids, selected kernel/family, runtime ABI identity,
runtime glue role, vector shape/config, or bundle component metadata.

This is a runtime ABI / artifact evidence boundary task. It is not another
finite-family evidence slice.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state was clean on `main` at
  `b4d6f07 test(rvv): verify i32 vmul dispatch ssh evidence`.
- No `.trellis/.current-task` existed before this task was created.
- The immediately previous task
  `rvv-i32-vmul-frontend-dispatch-ssh-evidence` is archived at
  `.trellis/tasks/archive/2026-05/05-11-rvv-i32-vmul-frontend-dispatch-ssh-evidence/`
  and must not be reopened.
- The direct `i32-vmul` artifact evidence task is archived at
  `.trellis/tasks/archive/2026-05/05-11-rvv-i32-vmul-frontend-artifact-ssh-evidence/`.
- Those archived tasks proved fresh bounded `ssh rvv` runtime correctness for
  direct RVV `i32-vmul` and RVV+scalar dispatch `i32-vmul`; this task should
  improve evidence authority boundaries rather than repeat another
  one-family proof.
- Current `--tcrv-export-emission-manifest` already includes selected path
  runtime ABI metadata and target artifact records for direct RVV source,
  header, and object routes.
- Current target artifact bundle indexes already include dispatch
  source/header/object file names, route ids, component group/role,
  external ABI name, runtime ABI kind/name, ordered runtime ABI parameters,
  selected component variants/roles, selected plan metadata, and conservative
  evidence roles.
- The Python runners still contain family descriptor tables. Those tables may
  define allowed CLI families and expected assertions, but generated manifests
  and indexes must be parsed and used as the evidence facts.

## Requirements

- Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python changes are limited to evidence orchestration, manifest/index parsing,
  validation, hashing, command summary handling, and diagnostic self-tests.
- Do not implement compiler core, dialects, capability model, variant
  selection, lowering, emission, route selection, or runtime ABI semantics in
  Python.
- Preserve the architecture boundary: RVV and scalar behavior remain
  plugin/target-owned; `tcrv.exec` remains compute-free.
- Direct RVV evidence must consume generated emission manifest target artifact
  records for direct source/header/object route ids, selected variant/role,
  runtime ABI kind/name, runtime glue role where available, ordered ABI
  parameters, vector shape/config metadata, selected binary family metadata,
  artifact kind, owner, and evidence role.
- RVV+scalar dispatch bundle evidence must consume generated bundle index
  records for dispatch source/header/object file names, route ids, component
  group/role, external ABI name, selected dispatch components, runtime ABI
  kind/name, ordered dispatch ABI parameters, selected RVV/scalar family
  metadata, vector shape/config metadata, artifact kind, owner, and evidence
  role.
- Direct and dispatch runners must fail closed on missing or mismatched
  generated manifest/index fields for route id, runtime ABI kind/name, vector
  config, selected family, selected kernel, runtime ABI signature, component
  group/role, or selected component roles.
- Dry-run evidence JSON must remain explicitly local export/tooling evidence
  and must not include runtime success claims.
- Runtime success or correctness claims require real `ssh rvv` execution
  through the updated manifest-authority path and must remain limited to the
  exact rerun slice.
- Artifact hashes may be computed by the runners from emitted files, but the
  artifact paths and route/component identity they hash must come from
  generated artifacts, manifests, or bundle indexes.
- Command summaries must be sanitized and must distinguish local tool commands
  from remote compile/link/run commands and their statuses.

## Acceptance Criteria

- Direct RVV `i32-vmul` dry-run evidence records route id, selected kernel,
  selected variant/family, runtime ABI kind/name, runtime glue role, vector
  shape/config, artifact paths, artifact hashes, and dry-run command status
  from generated manifest/artifact data plus sanitized command summaries.
- RVV+scalar dispatch `i32-vmul` bundle dry-run evidence records route id,
  selected kernel, selected RVV/scalar component variants/family, runtime ABI
  kind/name, runtime glue role or dispatch ABI role, vector shape/config,
  artifact paths, artifact hashes, and dry-run command status from generated
  bundle index/artifact data plus sanitized command summaries.
- Missing generated manifest/index fields fail closed with clear diagnostics in
  focused runner self-tests and lit/FileCheck coverage.
- Mismatched route id, runtime ABI, vector config, or selected family fails
  closed with clear diagnostics in focused runner self-tests and lit/FileCheck
  coverage.
- Dry-run evidence does not emit `runtime_success=true`,
  `ssh_evidence_verified=true`, performance claims, throughput, or latency.
- Focused direct RVV `i32-vmul` dry-run validation still succeeds.
- Focused dispatch-bundle `i32-vmul` dry-run validation still succeeds through
  target-artifact bundle and plan-and-export front door.
- At least one fresh `ssh rvv` validation is rerun through the updated
  manifest-authority path, preferably the dispatch-bundle `i32-vmul` run.
- If C++ manifest/exporter fields are missing, the smallest target/exporter
  manifest-owned C++ change is added with focused C++ coverage.
- No extension-specific branches are added to core orchestration passes.

## Non-Goals

- No new RVV family, scalar family, finite-family matrix, benchmark,
  throughput/latency/performance claim, or generic RVV backend claim.
- No new lowering semantics, MLIR vector/scalable-vector route, or broad e2e
  runner rewrite.
- No IME, AME, Sophgo/offload, Template, or Toy work.
- No compute semantics in `tcrv.exec`.
- No Python compiler internals.
- No report-only, helper-only, smoke-only, or standalone evidence packaging
  closeout. The boundary must fail closed and demonstrably consume generated
  facts.

## Validation Plan

- Build focused targets as needed:
  `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-scalar-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Run focused direct dry-run:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vmul --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i32_vmul`.
- Run focused dispatch bundle dry-run:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vmul --expect-selected-kernel=frontend_bundle_i32_vmul`.
- Update/run focused lit coverage:
  `test/Scripts/rvv-microkernel-e2e.test` and
  `test/Scripts/rvv-scalar-dispatch-e2e.test`.
- If C++ fields change, update/run
  `test/Target/TargetArtifactExportTest.cpp` or the closest target/export
  test.
- Run one fresh `ssh rvv` validation through the updated path and store
  evidence under `artifacts/tmp/...` with a unique run id.
- Run `python3 -m py_compile` for modified Python runner files.
- Run `git diff --check`.
- Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-rvv-e2e-evidence-manifest-authority`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if practical after focused checks pass.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/validation/experiment-reference.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-i32-vmul-frontend-artifact-ssh-evidence/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-i32-vmul-frontend-dispatch-ssh-evidence/prd.md`.
- Primary implementation surfaces:
  `scripts/rvv_microkernel_e2e.py`,
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `lib/Target/EmissionManifest.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Scalar/ScalarMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Scripts/rvv-microkernel-e2e.test`,
  `test/Scripts/rvv-scalar-dispatch-e2e.test`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Definition Of Done

The task is done when both bounded `i32-vmul` evidence runners consume
compiler/exporter-generated manifest or bundle facts as authority, missing or
mismatched generated facts fail closed in focused tests, dry-run evidence
remains non-runtime, fresh `ssh rvv` evidence validates the updated path for
one bounded slice, focused/full checks pass, Trellis validation passes, the
task is finished/archived, and one coherent commit records the work.

If unfinished, leave this task open and record the exact continuation point:
C++ manifest field generation, command summary generation, direct runner
manifest consumption, dispatch runner manifest consumption, fail-closed
mismatch diagnostics, dry-run lit coverage, ssh compile/run, evidence JSON
capture, Trellis validation, or commit.

## Completion Notes

- Implemented the bounded evidence-authority boundary in the two Python e2e
  runners only; no C++ manifest/exporter fields were missing, so no C++ source
  change was needed.
- Direct RVV `i32-vmul` now consumes generated emission-manifest
  `target_artifacts` for source/header/object route ids, runtime ABI
  kind/name/signature, runtime glue role, selected family/vector metadata,
  artifact identity, and evidence role. Missing or stale generated fields fail
  closed in runner self-tests and focused lit coverage.
- RVV+scalar dispatch bundle `i32-vmul` now consumes the generated target
  artifact bundle index plus generated dispatch source comments for route ids,
  selected kernel/family/vector facts, runtime ABI signature, runtime glue
  roles, artifact paths, artifact hashes, and ssh command status boundaries.
- Dry-run evidence remains explicitly local export/tooling evidence and does
  not claim runtime success.
- Fresh `ssh rvv` evidence was collected at
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-manifest-authority-dispatch-ssh-20260511T043258Z/`.
  The claim is limited to the bounded RVV+scalar `i32-vmul` target-artifact
  bundle external caller correctness slice, with
  `ssh_evidence_verified=true`.
- Validation passed:
  `python3 -m py_compile scripts/rvv_microkernel_e2e.py scripts/rvv_scalar_dispatch_e2e.py`,
  both runner self-tests, focused direct and dispatch dry-runs, focused
  script lit tests, focused build/plugin/exporter tests, `git diff --check`,
  and `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  with 205/205 tests passing.
