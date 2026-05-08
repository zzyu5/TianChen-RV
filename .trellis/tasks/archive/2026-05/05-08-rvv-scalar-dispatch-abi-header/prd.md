# target-owned RVV scalar dispatch ABI header

## Goal

Make the bounded RVV+scalar i32-vadd dispatcher consumable by an external C runtime caller by exporting a target-owned ABI header derived from the same exec-IR-backed `DispatchABIPlan` as the generated dispatch source/object.

## What I Already Know

* The current HEAD is expected to include `3189ae4 feat: link scalar dispatch fallback to exec IR`.
* The worktree and the three supervisor-policy files were clean at worker start.
* The current gap is an ABI handoff surface: the generated C and relocatable object exist, but a downstream C caller should include an emitted header rather than reconstructing the prototype from comments or detached metadata.
* Implementation must stay in C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck; Python is allowed only for evidence orchestration.
* The header must preserve the same IR boundary validation as dispatch C/object export.

## Requirements

* Add a target-owned header export API for bounded RVV+scalar i32-vadd dispatch ABI.
* Add a `tcrv-translate` option for exporting that header.
* Generate the header from the same `DispatchABIPlan` used by dispatch source/object export.
* Emit include guard, `extern "C"` guard for C++, and a dispatcher prototype whose parameter order and C types match the generated dispatch definition.
* Preserve validations for selected RVV case runtime guard, scalar fallback link, route metadata mirror, required capability metadata, selected march/mabi metadata, and library object route constraints.
* Add FileCheck coverage for the header export.
* Add a focused negative test for an exec-IR boundary error shared with source/object export.
* Keep object evidence helper explicit and avoid routing it through the generic artifact front door.
* Update durable lowering-runtime/testing docs only if needed to describe the header/object/runtime-caller boundary.

## Acceptance Criteria

* [x] Header export prints the expected include guard, C++ guard, and exact dispatcher prototype.
* [x] Header export fails closed on at least one relevant exec-IR ABI/control boundary mismatch.
* [x] Existing dispatch C/object tests still pass.
* [x] `git diff --check` passes.
* [x] `cmake --build build --target tcrv-opt tcrv-translate -j2` passes.
* [x] `cmake --build build --target check-tianchenrv -- -j2` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` is run if present and not covered.
* [x] Manual `ssh rvv` evidence linked an external C driver against the generated header/object pair and ran the bounded guard=0/guard=1 self-check.

## Out of Scope

* No Python implementation of compiler decisions or ABI modeling.
* No generic compute operations in `tcrv.exec`.
* No broad RVV/IME/offload/arbitrary-kernel lowering.
* No automatic runtime probing or hidden hardware detection in generated artifacts.
* No docs-only, status-only, manifest-only, or evidence-only closeout.

## Technical Notes

* Primary code paths to inspect and likely modify: `include/TianChenRV/Target/RVVScalarDispatch.h`, `lib/Target/Builtin/RVVScalarDispatch.cpp`, `tools/tcrv-translate/tcrv-translate.cpp`.
* Test paths to inspect and update: `test/Target/RVVScalarDispatch/*`, `test/Target/TargetArtifactExportTest.cpp`, `test/Scripts/rvv-scalar-dispatch-e2e.test`.
* Specs to consult: core exec contract, capability contract, variant generation/selection/tuning, RVV/scalar fallback plugin contracts, lowering-runtime emission contract, and MLIR testing contract.
