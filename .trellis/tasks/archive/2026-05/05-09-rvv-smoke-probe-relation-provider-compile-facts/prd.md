# RVV smoke probe relation provider compile facts

## Goal

Make the RVV smoke-probe target source exporter consume selected RVV compile facts from explicit module-level `tcrv.exec.target` profile relation-providers, matching the relation-aware capability scope already used by planning, direct RVV microkernel export, and RVV+scalar dispatch export.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Live HEAD is `cb4fc0e feat: consume profile RVV compile facts in artifact export`.
* The previous round left `lib/Target/RVV/RVVSmokeProbe.cpp` as the remaining exact-only selected compile fact lookup path.
* The implementation must remain in target/export C++ and must not move RVV-specific decisions into generic core passes, `tcrv.exec` ODS, or Python compiler internals.

## Requirements

* Update `lib/Target/RVV/RVVSmokeProbe.cpp` so selected compile facts such as `rvv.probe.compile_run`, `rvv.toolchain.march`, and selected mabi metadata use relation-aware `TargetCapabilitySet` provider lookup semantics.
* Preserve exact-provider precedence through existing `TargetCapabilitySet` lookup behavior.
* Preserve strict diagnostics for missing selected march, required march mismatch, and any existing selected mabi or selected march metadata conflicts.
* Add or minimally extend lit/FileCheck coverage for a module-level target profile route:
  `tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-smoke-probe-c | FileCheck %s`.
* Keep the claim local to compiler/export behavior. Do not make RVV runtime correctness or performance claims without `ssh rvv` evidence.

## Acceptance Criteria

* [x] A profile-backed selected compile fact reaches emitted RVV smoke-probe C artifact metadata.
* [x] Negative coverage is added only if it directly proves the preserved mismatch diagnostic owner.
* [x] Existing direct RVV microkernel, RVV+scalar dispatch object, target artifact export, plugin proposal/planning, and capability model tests continue to pass.
* [x] `git diff --check` passes.
* [x] CMake configure with LLVM/MLIR 20 passes.
* [x] `check-tianchenrv` passes, unless blocked by a real local toolchain issue with exact command and error reported.

## Out of Scope

* New RVV lowering or runtime ABI work.
* IME, AME, Sophgo, offload ISA, or new backend target work.
* Python compiler-internal implementation.
* Broad test matrix or unrelated refactor.

## Technical Notes

* Primary implementation file: `lib/Target/RVV/RVVSmokeProbe.cpp`.
* Existing relation-aware consumers to compare: `lib/Target/RVV/RVVMicrokernel.cpp` and `lib/Target/Builtin/RVVScalarDispatch.cpp`.
* Relevant spec layers are capability model, core dialect, extension plugins, lowering/runtime, plugin protocol, and testing.
