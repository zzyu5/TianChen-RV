# Offload runtime plugin boundary template

## Goal

Complete one bounded C++/MLIR offload runtime boundary template proving that
runtime offload is integrated as an extension-plugin capability and
plugin-owned handoff route. The route must carry an offload runtime capability
through plugin proposal, variant selection, selected lowering-boundary
metadata, emission/readiness metadata, and a deterministic non-executable
artifact/manifest handoff without adding offload, Sophgo, vendor, or custom-ISA
branches to generic core passes.

## Background

The archived task
`.trellis/tasks/archive/2026-05/05-11-rvv-i32m2-selected-vl-dispatch-ssh-evidence/`
completed the non-m1 RVV selected-VL dispatch evidence path and collected real
`ssh rvv` correctness evidence. Repeating another RVV micro-evidence round
would exercise the same surface again. The next bottleneck is architectural:
the plugin system must show that the active path is not RVV-only.

The offload slice is intentionally metadata and descriptor oriented. It models
RISC-V host plus runtime/driver/C-ABI handoff to an accelerator as
`runtime-offload`, not as a custom RISC-V ISA extension, and it must not claim
local runtime execution, accelerator correctness, or performance.

## Requirements

- Use the stable first-slice offload names from the spec:
  - plugin name `offload-plugin`;
  - capability id `offload.runtime`;
  - capability kind `runtime-offload`;
  - preferred capability symbol `@offload_runtime`;
  - selected variant `@offload_runtime_first_slice`;
  - runtime ABI `generic-runtime-offload-c-abi-handoff.v1`;
  - handoff kind `runtime-offload`;
  - descriptor route id `tcrv-export-offload-runtime-descriptor`;
  - artifact kind `runtime-offload-handoff-descriptor`.
- Ensure an offload capability object participates in proposal legality and
  selection as a runtime-offload capability, not as an ISA/vector capability.
- The offload plugin must propose the variant through plugin-local logic and
  generic registry interfaces. Missing capability or malformed handoff metadata
  must fail closed or decline without materializing a fake offload route.
- The selected variant must materialize only `tcrv.exec` execution and variant
  metadata. No compute semantics may be added to `tcrv.exec`.
- Lowering boundary and emission/readiness metadata must be offload
  plugin/extension owned. Generic core passes may route and validate through
  registry interfaces, but must not gain offload, Sophgo, vendor, runtime, ISA,
  dtype, shape, or accelerator semantic branches.
- A `tcrv-translate` or existing target artifact route must emit a deterministic
  descriptor or manifest describing the offload handoff. It must include:
  selected kernel, selected variant, origin plugin, required capabilities,
  runtime ABI, handoff kind, artifact route/kind/status, typed runtime ABI
  parameters derived from IR-backed `mem_window` / `runtime_param` boundaries
  when the route requires them, and explicit non-claim fields.
- The artifact must explicitly state that it is an offload runtime handoff with
  no local runtime correctness claim, no hardware execution claim, and no
  performance claim.
- Add fail-closed diagnostics for:
  - missing offload runtime capability;
  - missing or stale runtime ABI / handoff metadata;
  - missing required runtime ABI role mirrors;
  - attempts to model offload as a custom RISC-V ISA extension.

## Acceptance Criteria

- Focused C++ and/or lit/FileCheck coverage demonstrates the positive path:
  offload capability -> plugin proposal -> selected variant ->
  plugin-owned lowering boundary -> emission/readiness metadata ->
  deterministic descriptor/manifest handoff.
- Negative coverage proves missing capability, malformed runtime ABI/handoff
  metadata, stale ABI role metadata, and custom-ISA misclassification fail
  closed before descriptor success.
- The descriptor or manifest contains deterministic bounded fields for kernel,
  variant, origin plugin, required capability refs, runtime ABI, handoff kind,
  artifact route/kind/status, runtime ABI parameters, and explicit no-claim
  metadata.
- Existing RVV/scalar paths remain valid. RVV changes are allowed only when
  shared generic interfaces need compatibility preservation.
- Generic core code remains target-neutral: any changed shared pass must only
  extend or consume generic registry/interface surfaces.
- `tcrv.exec` remains compute-free.
- Focused plugin/offload, variant selection/materialization, lowering
  boundary, emission readiness/plan, and target artifact export tests pass.
- `git diff --check` passes.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  is run if practical after focused checks pass.
- Trellis validates and archives only after an active C++/MLIR producer and
  consumer path exists. One coherent commit records the module.

## Non-Goals

- No Sophgo hardware execution claim.
- No accelerator runtime implementation, linking, device queue, DMA, or remote
  hardware requirement.
- No IME or AME implementation.
- No new RVV runtime evidence round and no RVV behavior changes except to
  preserve shared-interface compatibility.
- No generic compute ops or compute semantics inside `tcrv.exec`.
- No Python compiler internals. Python may remain only tooling, runner,
  artifact parser, or small support script if already used by tests.
- No docs-only, helper-only, smoke-only, report-only, or metadata-only closeout
  without active C++/MLIR producer and consumer changes.
- No broad plugin framework rewrite.

## Validation Plan

- Build focused targets that own the changed route, expected to include
  `tcrv-opt`, `tcrv-translate`, offload plugin tests, and target artifact
  export tests.
- Run focused C++ tests for the offload plugin registry/proposal/legality path.
- Run focused lit/FileCheck tests for variant materialization/selection,
  selected lowering boundary, emission readiness/plan, and descriptor or
  manifest artifact export.
- Run negative lit/FileCheck tests for missing capability, malformed handoff
  metadata, stale ABI role metadata, and ISA-extension misclassification.
- Run `git diff --check`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if practical after focused checks pass.
- Validate the Trellis task before finishing and archiving it.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/extension-plugins/offload-runtime-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Archived context read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-i32m2-selected-vl-dispatch-ssh-evidence/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-rvv-selected-config-boundary-family-contract/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-selected-config-vl-dataflow-materialization/prd.md`.
- Primary code surfaces to inspect before source edits:
  `include/TianChenRV/Plugin/Offload/`,
  `lib/Plugin/Offload/`,
  `test/Plugin/OffloadExtensionPluginTest.cpp`,
  `test/Plugin/offload-extension-plugin.test`,
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `lib/Transforms/VariantMaterialization.cpp`,
  `lib/Transforms/VariantSelection.cpp`,
  `lib/Transforms/LoweringBoundary.cpp`,
  `lib/Transforms/EmissionReadiness.cpp`,
  `lib/Target/EmissionManifest.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `tools/tcrv-opt/`,
  `tools/tcrv-translate/`, and existing offload metadata route tests under
  `test/Transforms/` and `test/Target/`.

## Definition Of Done

The task is complete when an active C++/MLIR offload producer/consumer path
selects the runtime-offload variant, materializes plugin-owned boundary and
emission metadata, exports a deterministic non-executable descriptor/manifest
with explicit no-runtime/no-performance claims, focused positive and negative
tests pass, generic core remains extension-neutral, Trellis validates and
archives the task, and one coherent commit records the module.

If unfinished, leave the task open and record the exact continuation point:
capability modeling, plugin proposal, legality/selection, variant
materialization, lowering boundary, emission readiness, artifact route,
negative diagnostic, or test integration.

## Completion Notes

- The active offload plugin path already carried the first-slice runtime
  offload capability through plugin registration, proposal collection,
  materialized `tcrv.exec.variant`, static selected marker planning,
  plugin-owned `tcrv_offload.lowering_boundary`, plugin emission plan metadata,
  and target artifact descriptor export.
- This round hardened the descriptor artifact claim boundary by adding explicit
  deterministic fields:
  `artifact_status = "non-executable-runtime-offload-handoff-metadata"`,
  `local_runtime_execution_claim = "none"`,
  `local_runtime_correctness_claim = "none"`,
  `hardware_execution_claim = "none"`, and
  `performance_claim = "none"`.
- The offload plugin C++ test now covers the case where `offload.runtime` is
  incorrectly declared as `kind = "custom-isa"`: proposal collection records a
  recoverable plugin-local decline, and legality rejects the materialized
  variant with a runtime-offload kind diagnostic.
- The lit emission-readiness negative test now exercises the same custom-ISA
  misclassification through the public pass route before emission-plan
  materialization.
- The descriptor artifact lit test now checks the explicit non-executable and
  no-claim fields in the `tcrv-translate --tcrv-export-target-artifact` output.
- Generic core passes were not given offload, Sophgo, vendor, runtime, dtype,
  shape, or target-specific branches. The changed producer/consumer behavior
  remains C++/MLIR/TableGen/CMake/lit owned.
- `tcrv.exec` stayed compute-free. No accelerator runtime, vendor call,
  hardware execution, local runtime correctness, or performance claim was
  added.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-offload-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-offload-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Plugin/offload-extension-plugin.test
  Transforms/EmissionReadiness/offload-fail-closed.mlir
  Target/ArtifactExport/offload-runtime-descriptor-artifact-route.test
  Target/EmissionManifest/emission-manifest-offload-pipeline.mlir`
- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-target-artifact-export-test tianchenrv-emission-readiness-test
  tianchenrv-variant-materialization-test tianchenrv-variant-selection-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emission-readiness-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-variant-materialization-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-variant-selection-test`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  200/200 lit tests passed.
