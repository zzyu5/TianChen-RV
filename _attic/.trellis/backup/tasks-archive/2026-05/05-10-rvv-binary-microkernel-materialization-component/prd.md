# Extract RVV binary microkernel materialization component

## Goal

Extract the RVV plugin-local binary microkernel materialization logic out of the
monolithic `RVVExtensionPlugin.cpp` into a dedicated C++ component that owns the
selected i32/i64 binary microkernel materialization behavior for the current
bounded RVV executable slice.

The goal is not to add a new hardware claim or broaden RVV support. The goal is
to keep the existing compiler path maintainable after the preceding
`RVVBinaryPlanning` extraction: selected RVV variants with finite binary
lowering descriptors must still validate, preflight callable runtime ABI
metadata, materialize `tcrv_rvv.lowering_boundary`, and attach the matching
plugin-local i32/i64 microkernel op through the RVV plugin boundary.

## What I already know

- Current HEAD is `8c3e3fc`, with a clean worktree before this task started.
- There is no `.trellis/.current-task`; the previous task
  `05-10-rvv-binary-planning-component-extraction` was archived in the latest
  supervisor audit.
- Latest supervisor review input is
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0037-20260510T041741Z/`.
  The fallback brief instructs the worker to create or repair a module-sized
  task from current evidence after malformed Hermes review JSON.
- The previous commit extracted selected-plan and vector-shape metadata helpers
  into `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h` and
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`.
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp` is still about 3000 lines and still
  owns the i32/i64 binary microkernel materialization structs, runtime ABI
  preflight helper, duplicate selected-path microkernel rejection, and
  materialization of `tcrv_rvv.i32_*_microkernel` /
  `tcrv_rvv.i64_*_microkernel` operations.
- The relevant long-term specs are `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.

## Requirements

- Add a plugin-local C++ component for RVV binary microkernel materialization,
  with a public header under `include/TianChenRV/Plugin/RVV/` and implementation
  under `lib/Plugin/RVV/`.
- Move selected i32/i64 binary microkernel materialization plan construction
  and operation construction out of `RVVExtensionPlugin.cpp` while preserving
  the RVV plugin as the only caller.
- Preserve existing behavior for:
  - i32 add/sub/mul microkernel descriptors;
  - i32m1 and i32m2 selected vector-shape configs;
  - i64 add/sub/mul microkernel descriptors;
  - selected `required_march` and optional selected MABI metadata;
  - runtime ABI mem_window/runtime_param preflight;
  - duplicate selected-path microkernel rejection;
  - selected vector-shape metadata on materialized microkernel ops.
- Keep all RVV-specific semantics inside the RVV plugin/component. Generic core
  passes, `tcrv.exec`, and target-neutral orchestration must not gain RVV,
  dtype, shape, or microkernel branches.
- Use C++ / MLIR / LLVM / CMake only for compiler implementation. Python may
  only remain in existing tooling/tests.
- Do not create new RVV correctness or performance claims. Any runtime claim
  still requires separate real `ssh rvv` evidence and is out of scope for this
  component extraction.

## Acceptance Criteria

- [x] `RVVExtensionPlugin.cpp` delegates selected binary microkernel plan and op
      materialization to the new RVV plugin-local component.
- [x] Existing selected-boundary materialization still emits exactly one
      matching `tcrv_rvv.lowering_boundary` and the matching i32 or i64
      microkernel op for descriptor-backed selected RVV binary paths.
- [x] Existing explicit microkernel and descriptor-backed paths continue to
      fail closed on stale boundary, duplicate microkernel, missing selected
      metadata, missing runtime ABI roles, and mismatched vector-shape metadata.
- [x] Focused C++ and lit checks covering RVV plugin/binary planning and RVV
      microkernel routes pass.
- [x] `git diff --check` passes.
- [x] Task status, Trellis context, and workspace journal are updated truthfully.

## Out of Scope

- Adding a new RVV family, dtype, vector shape, hardware probe, runtime
  benchmark, or performance story.
- Moving target exporters from `lib/Target/RVV/RVVMicrokernel.cpp`.
- Changing generic core pass behavior or adding target-family branches outside
  RVV plugin-local code.
- Reworking the entire 3000-line `RVVExtensionPlugin.cpp`; this task only
  extracts the selected binary microkernel materialization slice.

## Technical Notes

- Current extraction anchor:
  `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h` and
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`.
- Current monolithic materialization owners:
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp` functions around selected binary
  plan construction, callable ABI parameter collection, duplicate microkernel
  rejection, and `materializeRVVI32MicrokernelOp` /
  `materializeRVVI64MicrokernelOp`.
- Relevant focused checks are expected to include the RVV plugin tests,
  RVV binary planning tests, and RVV microkernel lit tests rather than broad
  unrelated matrices.
