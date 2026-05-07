# Generic RVV+Scalar Dispatch Object Artifact Route

## Goal

Advance the TianChen-RV compiler path from planned RVV+scalar dispatch source export to a planned, artifact-kind-aware object export through the generic `tcrv-translate --tcrv-export-target-artifact` front door.

## What I Already Know

- The repository root is `/home/kingdom/phdworks/TianchenRV`.
- The worktree was clean before this task, at HEAD `59422f5 feat: export planned RVV scalar dispatch artifacts`.
- The latest Hermes audit is `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0035-20260507T224124Z/repo_audit.md`.
- The previous completed milestone added generic target source export for planned RVV+scalar i32-vadd dispatch.
- A dedicated `tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object` translator already emits a bounded RISC-V ELF relocatable object from the same validated dispatch self-check source.
- The generic artifact front door currently registers the RVV+scalar dispatch composite route only as source, not as a non-source object artifact.

## Requirements

- Register a target-owned RVV+scalar dispatch self-check object composite route in the built-in target artifact registry.
- Keep `tcrv-export-target-source-artifact` source-only and free of embedded self-check/object behavior.
- Make `tcrv-export-target-artifact` prefer a matching non-source composite artifact when both source and non-source composite routes are available for the same selected plan.
- Preserve fail-closed validation of selected path, lowering boundary, emission plan, runtime ABI roles, selected compile facts, and local clang/toolchain availability.
- Keep RVV/scalar-specific matching and object generation in the RVV+scalar target-owned exporter code, not in generic core routing.

## Acceptance Criteria

- [ ] `tianchenrv-target-artifact-export-test` sees both built-in composite routes and rejects duplicate registration.
- [ ] A focused lit test proves the generic `--tcrv-export-target-artifact` route can emit a non-empty RISC-V ELF relocatable object when local RVV object clang support is available.
- [ ] Existing source export tests still prove source-only output does not include a self-check harness.
- [ ] README/spec wording states the object route is bounded and does not claim linking, runtime integration, performance, or generic RVV lowering.
- [ ] The full `check-tianchenrv` target passes or reports exact unavailable toolchain diagnostics.

## Out Of Scope

- No generic RVV lowering.
- No dynamic linking/runtime loading.
- No automatic hardware probe in generated dispatch.
- No RVV correctness/performance claim without separate `ssh rvv` compile/run evidence.
- No broad negative fixture matrix beyond the focused route and registry checks.

## Technical Notes

- Relevant code: `lib/Target/TargetArtifactExport.cpp`, `lib/Target/Builtin/RVVScalarDispatch.cpp`, `test/Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-object.test`, `test/Target/TargetArtifactExportTest.cpp`.
- Relevant specs: `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`.
