# continue compiler owner from supervisor review fallback

## Goal

Close the current module-target-profile gap at the lowering/runtime artifact boundary: RVV selected compile metadata provided by a module-level `tcrv.exec.target` profile must remain usable after planning, when target-owned RVV source/object/dispatch exporters validate selected `-march` / optional `-mabi` facts.

## What I Already Know

* Repo root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree was clean at HEAD `5bb5f55 feat: consume target profile RVV capacity facts`.
* There is no stale active Trellis task; this task was created for the current serial worker round.
* Latest supervisor audit/review input is `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0088-20260508T223342Z/{repo_audit.md,review_input.md}`.
* The previous round made module-level profile capacity facts participate in RVV proposal, cost, selected boundary, microkernel descriptor, and emission-plan metadata.
* Current profile-only planning succeeds for `test/Transforms/ExecutionPlanning/execution-planning-pipeline-module-target-profile.mlir`.
* Current profile-only generic RVV source export fails because `lib/Target/RVV/RVVMicrokernel.cpp` still validates `rvv.probe.compile_run` / `rvv.toolchain.march` using exact `lookupByID` instead of relation-aware provider lookup.
* `lib/Target/Builtin/RVVScalarDispatch.cpp` has the same exact-only selected compile fact lookup for RVV+scalar dispatch object compilation.
* The user prompt requires single full-access serial execution and explicitly forbids subagents, spawned agents, parallel agents, and multi-agent workflows.

## Requirements

* Keep implementation in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
* Do not use Python for compiler internals; Python remains only runner/probe/artifact support.
* Preserve `tcrv.exec` as execution/capability/variant focused and compute-free.
* Keep RVV-specific selected compile fact interpretation target/plugin-local; do not add RVV branches to core orchestration.
* Direct RVV microkernel source/object exporters must accept selected march/mabi facts from exact capability providers or relation-provider target profiles in the kernel capability-provider scope.
* RVV+scalar dispatch object exporter must use the same relation-aware selected compile fact behavior for its RVV branch compile config.
* Preserve strict validation: selected `tcrv_rvv.required_march` must still match preserved selected march metadata, conflicting `selected_mabi` metadata must still fail, and missing selected march metadata must still fail.
* Update durable specs where they currently imply exact-only selected march/mabi capability consumption.
* Add focused tests that prove module-level target profile selected compile facts pass through planning into target-owned artifact export.

## Acceptance Criteria

* A module-level `tcrv.exec.target` profile that provides `rvv.probe.compile_run` and carries `selected_march` can drive `--tcrv-execution-planning-pipeline | --tcrv-export-target-source-artifact`.
* Direct RVV microkernel compile config uses relation-aware providers for `rvv.probe.compile_run`, `rvv.toolchain.march`, and `rvv.toolchain.mabi`.
* RVV+scalar dispatch object compile config uses relation-aware providers for the same compile fact ids.
* Exact capabilities remain supported and authoritative when present.
* Specs document relation-provider selected compile facts for artifact export.
* Focused lit/FileCheck or C++ tests pass, plus relevant build/check commands.
* No `ssh rvv` runtime/correctness/performance claim is made unless real `ssh rvv` evidence is run.
* Repo ends clean with one coherent commit if the round completes.

## Out Of Scope

* No new high-level tensor/tile IR or new generic RVV lowering.
* No new kernel family beyond the bounded i32-vadd slice.
* No new runtime probing, hardware execution, correctness claim, performance claim, or broad evidence packaging.
* No subagents or parallel worker workflow.

## Technical Notes

* Reproduced current failure:
  `artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Transforms/ExecutionPlanning/execution-planning-pipeline-module-target-profile.mlir --tcrv-execution-planning-pipeline | artifacts/tmp/tianchenrv-build/bin/tcrv-translate --tcrv-export-target-source-artifact`
  fails with missing preserved selected_march metadata, despite the module profile providing `rvv.probe.compile_run`.
* Implemented relation-aware selected compile metadata lookup in the direct RVV
  microkernel exporter and the RVV+scalar dispatch object compile config.
* Added source-export coverage for the profile-only direct RVV path and object
  preflight coverage for the profile-only dispatch path.
* Verification run:
  * `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
  * `artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Transforms/ExecutionPlanning/execution-planning-pipeline-module-target-profile.mlir --tcrv-execution-planning-pipeline | artifacts/tmp/tianchenrv-build/bin/tcrv-translate --tcrv-export-target-source-artifact`
  * `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='execution-planning-pipeline-module-target-profile|rvv-scalar-i32-vadd-dispatch-module-target-profile-object'` from `artifacts/tmp/tianchenrv-build/test`
  * `git diff --check`
  * `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
* `ssh rvv` was not run because this round makes no RVV runtime,
  correctness, or performance claim.
* Relevant specs:
  `.trellis/spec/capability-model/capability-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Relevant code:
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `test/Transforms/ExecutionPlanning/execution-planning-pipeline-module-target-profile.mlir`,
  `test/Target/RVVMicrokernel/`.
