# plan-and-export target artifact bundle front door

## Goal

Add a single C++ `tcrv-translate` front door that can take a minimally
unplanned TianChen-RV exec module with kernel/capability anchors, run the
existing built-in execution planning pipeline, and export the selected target
artifact bundle without requiring hand-authored selected-path diagnostics,
lowering-boundary metadata, or emission-plan diagnostics in the input MLIR.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- HEAD before this task is `ba1cbec feat: publish dispatch bundle runtime abi signature`.
- The worktree was clean at task start.
- Existing `--tcrv-export-target-artifact-bundle` is a coherence-gated exporter
  for already planned input and must keep that behavior unchanged.
- The existing execution planning pipeline already supports injected built-in
  extension plugin and target artifact exporter registries through
  `buildExecutionPlanningPipeline(pm, plugins, exporters)`.
- The existing bundle exporter already owns component_group, component_role,
  external_abi_name, and ordered runtime_abi_parameter[index] bundle index
  metadata for the RVV+scalar i32-vadd dispatch external ABI bundle.

## Requirements

- Add an explicit C++ translate front door, preferably
  `--tcrv-plan-and-export-target-artifact-bundle`.
- The new front door must register built-in extension plugins and built-in
  target artifact exporters at the tool boundary.
- The new front door must run the existing execution planning pipeline over the
  parsed `ModuleOp`, then call the existing target artifact bundle exporter with
  the existing output directory option.
- It must not duplicate planning, selection, lowering-boundary, emission-plan,
  coherence, bundle schema, or ABI logic in the tool.
- It must fail closed with MLIR diagnostics if planning fails, coherence fails,
  or bundle export fails, and it must not print bundle completion on failure.
- Keep the slice bounded to the existing built-in RVV+scalar i32-vadd bundle
  path.
- Python may only consume/orchestrate compiler outputs if touched; Python must
  not own compiler planning, target artifact semantics, or runtime ABI decisions.

## Acceptance Criteria

- [x] A lit/FileCheck test invokes the new front door on an input fixture that
      does not already contain selected-path diagnostics, lowering-boundary
      metadata, or emission-plan diagnostics.
- [x] The positive test proves the generated bundle index exists and preserves
      selected dispatch surface, component_group/component_role/external_abi_name
      metadata, and runtime_abi_parameter[index] signature fields.
- [x] A focused negative test proves the new front door fails before bundle
      completion when planning cannot produce a coherent/exportable path.
- [x] Existing already-planned `--tcrv-export-target-artifact-bundle` semantics
      remain unchanged.
- [x] Local checks pass: `git diff --check`, build `tcrv-opt` and
      `tcrv-translate`, and `check-tianchenrv`.

## Completion Evidence

- `git diff --check` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with 128/128 lit
  tests passing.
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py` passed.
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` passed.
- Real `ssh rvv` bundle external caller evidence through the new plan-and-export
  front door passed under
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/20260508T122009Z`.

## Out Of Scope

- Arbitrary kernels, generic target families, dynamic runtime loading, new
  backends, new dialect features, and performance measurement.
- Python compiler planning, selection, lowering, emission, bundle schema, or
  ABI semantics.
- New RVV runtime/correctness claims without real `ssh rvv` evidence.

## Technical Notes

- Core files inspected before editing include `AGENTS.md`, `README.md`,
  `CMakeLists.txt`, relevant `.trellis/spec/*` files,
  `include/TianChenRV/Transforms/Passes.h`, `include/TianChenRV/Transforms/Passes.td`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`, transform passes,
  target artifact export files, `tools/tcrv-translate/tcrv-translate.cpp`,
  `scripts/rvv_scalar_dispatch_e2e.py`, and existing execution planning /
  bundle tests.
