# RVV executable artifact route metadata registration

## Goal

Adopt the existing `TargetArtifactRouteMetadata` registration contract for one
bounded executable RVV route so generic target artifact export can validate the
route's runtime ABI and selected-plan metadata before invoking RVV target-owned
export code. The proof route is the finite RVV `i32-vsub` runtime-callable
microkernel source route used by the recent `i32-vsub` / `i32m2` dispatch
bundle evidence path.

## Background

The archived task
`.trellis/tasks/archive/2026-05/05-11-extension-plugin-artifact-route-registration-template/`
added `TargetArtifactRouteMetadata` and proved metadata/no-claim Toy and
offload routes can register descriptor metadata through plugin-owned target
exporter bundles. The archived task
`.trellis/tasks/archive/2026-05/05-11-rvv-i32m2-selected-vl-dispatch-ssh-evidence/`
proved the `i32-vsub` / `i32m2` RVV+scalar dispatch bundle reaches real
`ssh rvv` compile/link/run correctness evidence. This task connects those two
threads by proving the generic route descriptor surface also works for a
bounded executable RVV route with selected RVV plan metadata and runtime ABI
boundaries.

## Requirements

- Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only orchestrate runner/evidence checks.
- Preserve `tcrv.exec` as execution organization only. Do not add compute ops
  or high-level tensor/tile semantics.
- Register an RVV route descriptor through the target artifact registry for the
  bounded `tcrv-export-rvv-i32-vsub-microkernel-c` executable route.
- The descriptor must declare runtime ABI, runtime ABI kind/name, runtime glue
  role, selected-plan metadata requirements, and conservative claim/evidence
  fields.
- Generic target artifact export must consume the descriptor before RVV
  route-local candidate validation and before target-owned source/bundle output.
- Route metadata must require selected-plan fields for RVV binary family,
  arithmetic operator, lowering descriptor, selected vector shape group,
  selected shape capability ids, runtime AVL/VL boundary markers, and runtime
  element-count ABI name where those fields already exist.
- RVV-specific validation of vector shape/operator/intrinsic/body semantics
  must remain in RVV target/plugin-owned code. Generic export may validate
  route metadata shape, runtime ABI equality, selected-plan metadata presence,
  static family fields, and bounded text only.
- Generated artifact or bundle index output must distinguish compiler/export
  success from runtime correctness evidence. Dry-run artifacts must not claim
  runtime correctness.
- Missing selected-plan metadata, stale runtime ABI kind, mismatched selected
  metadata, duplicate route registration, and unknown route must fail closed
  with useful diagnostics.

## Acceptance Criteria

- `tcrv-export-rvv-i32-vsub-microkernel-c` registers
  `TargetArtifactRouteMetadata` through the RVV plugin-owned target exporter
  bundle.
- Focused C++ coverage proves the route descriptor metadata is present,
  duplicate registration still fails, stale runtime ABI kind fails through
  generic route metadata preflight, and missing/stale selected-plan metadata
  fails before RVV source output.
- Focused lit/FileCheck coverage for the `i32-vsub` / `i32m2` generic route or
  bundle path checks that route, runtime ABI, selected RVV plan metadata,
  callable symbols, and no-runtime-claim boundaries remain visible.
- Existing RVV, scalar, Toy, offload, and generic target artifact export tests
  are not weakened.
- If any artifact or evidence JSON claims RVV runtime correctness, this task
  runs real `ssh rvv` evidence for the bounded proof route and records the
  artifact path/result. If this round only emits dry-run/compiler artifacts,
  it must explicitly carry no runtime correctness claim.
- Trellis task context validates before finishing/archive.

## Non-Goals

- No broad RVV dtype/LMUL/operator matrix.
- No new kernel family, generic RVV backend, MLIR vector/scalable-vector
  lowering route, or performance experiment.
- No Toy/offload route churn except compatibility fixes required by the shared
  metadata contract.
- No runtime correctness, throughput, latency, or generic RVV claim without
  real `ssh rvv` evidence.
- No compiler internals in Python.
- No docs-only, helper-only, smoke-only, report-only, or standalone evidence
  closeout without active C++ route registration/consumer behavior.

## Validation Plan

- Build focused targets: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-target-artifact-export-test`, and any touched RVV plugin/target
  tests.
- Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit/FileCheck tests for RVV scalar dispatch and target artifact
  export paths touched by route metadata adoption.
- Run `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` if script
  evidence surfaces are touched.
- Run a focused dry-run for `i32-vsub` / `i32m2` bundle metadata if bundle
  index output changes.
- Run real `ssh rvv` only if this task produces a new runtime correctness
  claim beyond the archived evidence.
- Run `git diff --check`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target
  check-tianchenrv -j2` if practical after focused checks pass.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/validation/experiment-reference.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-11-extension-plugin-artifact-route-registration-template/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-i32m2-selected-vl-dispatch-ssh-evidence/prd.md`.
- Initial code inspection found `TargetArtifactRouteMetadata` generic
  registration and preflight already implemented in
  `include/TianChenRV/Target/TargetArtifactExport.h` and
  `lib/Target/TargetArtifactExport.cpp`.
- Initial code inspection found RVV source route registration in
  `lib/Target/RVV/RVVMicrokernel.cpp` currently registers route id, artifact
  kind, origin, emission kind, runtime ABI role requirements, candidate
  validation, component group, and external ABI name, but does not yet attach
  route descriptor metadata.
- Existing RVV route-local validation already checks RVV selected shape,
  family/operator, capability ids, runtime AVL/VL boundary metadata, and body
  semantics; this task should expose the route-level descriptor to generic
  preflight without moving RVV semantics into generic export code.

## Definition Of Done

The task is done when the bounded RVV `i32-vsub` executable source route
publishes route descriptor metadata, generic target artifact preflight consumes
that descriptor and rejects stale route/runtime/selected-plan metadata,
focused C++/lit validation passes, dry-run artifacts remain explicit non-runtime
claims unless a fresh `ssh rvv` run is collected, Trellis validates and archives
the task, and one coherent commit records the module.

If unfinished, leave the task open and record the exact continuation point:
RVV route descriptor API, registry hookup, selected-plan metadata extraction,
generic validation consumer, artifact/bundle text emission, fail-closed
diagnostic, script evidence handoff, ssh rvv run, or lit/C++ test integration.

## Completion Notes

- Implemented a generic selected-plan metadata presence requirement in
  `TargetArtifactRouteMetadata` so route descriptors can require existing
  RVV-selected fields without hard-coding route-specific values in generic
  export.
- Registered route descriptor metadata for the bounded executable RVV
  `tcrv-export-rvv-i32-vsub-microkernel-c` source route from RVV target-owned
  code. The descriptor declares runtime ABI, ABI kind/name, runtime glue role,
  exact binary family/operator/lowering/runtime-VL requirements, presence
  requirements for selected shape/capability/runtime-control fields, and
  conservative no-runtime/no-performance claim fields.
- Kept RVV-specific vector shape, intrinsic/body, family, runtime-control, and
  selected-config extraction in `lib/Target/RVV/RVVMicrokernel.cpp`; generic
  export only consumes the route descriptor contract.
- Added focused C++ coverage for descriptor registration, presence
  requirements, generic stale-runtime-ABI rejection, generic stale selected
  binary family rejection, and fail-closed missing selected-shape metadata.
- Updated lit/FileCheck coverage where stale RVV vsub ABI now fails closed at
  generic route descriptor preflight before route-local RVV validation.
- Produced a local dry-run bundle for the bounded `i32-vsub` / `i32m2` route at
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-20260511-route-metadata-vsub-i32m2-dry/`.
  Its evidence JSON has `ssh_evidence: false` and the emitted artifact text
  remains explicit that correctness claims require a separate self-check
  harness plus `ssh rvv` evidence, so this task did not make a new RVV runtime
  correctness claim and did not need a fresh `ssh rvv` run.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tcrv-translate tcrv-opt -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir Target/RVVMicrokernel/rvv-microkernel-i32m2-object.mlir Scripts/rvv-scalar-dispatch-bundle-e2e.test Scripts/rvv-microkernel-e2e.test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-20260511-route-metadata-vsub-i32m2-dry --overwrite`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/ArtifactExport Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir Scripts/rvv-scalar-dispatch-bundle-e2e.test Scripts/rvv-microkernel-e2e.test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/target-artifact-bundle-guards.mlir`
- `git diff --check`
- `python3 .trellis/scripts/task.py validate .trellis/tasks/05-11-rvv-executable-artifact-route-metadata-registration`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

Self-repair performed: the initial full `check-tianchenrv` run exposed one
stale FileCheck expectation in
`test/Target/TargetArtifactBundleExport/target-artifact-bundle-guards.mlir`.
The expected diagnostic was updated to the new generic route descriptor
preflight error, then the focused lit test and full `check-tianchenrv` passed.
