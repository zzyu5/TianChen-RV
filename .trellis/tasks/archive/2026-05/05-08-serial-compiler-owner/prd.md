# RVV Microkernel Object Artifact Route

## Goal

Advance the bounded selected RVV i32-vadd microkernel emission path from
runtime-callable C source export to a target-owned RISC-V ELF relocatable
object artifact that can be selected through the generic target artifact
front door.

## What I Already Know

* Current HEAD is `6907446 feat: route dispatch header through target
  artifacts`.
* Latest completed supervisor audit is
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0045-20260508T040135Z/`;
  the next r0046 run directory has no completed `repo_audit.md` or
  `review_input.md`.
* The repo is clean before this task except the new Trellis task files.
* Existing RVV microkernel support already validates one selected
  `tcrv_rvv.i32_vadd_microkernel` and exports a library-style C source route
  `tcrv-export-rvv-microkernel-c`.
* Existing RVV+scalar dispatch support already compiles a composite dispatch
  object, but a direct selected RVV microkernel path does not yet have a
  first-class object artifact route.
* The object route must remain target-owned and must not make source-only
  export ambiguous.

## Requirements

* Add a C++ API for exporting one selected RVV i32-vadd microkernel as a
  RISC-V ELF relocatable object.
* Register a target-owned RVV microkernel object composite exporter that
  matches the existing supported RVV microkernel source candidate and lets
  `--tcrv-export-target-artifact` select the object for direct RVV paths.
* Add a direct `tcrv-translate` command for the bounded object route.
* Compile the already-validated library-style RVV microkernel C source with
  local `clang`, using selected `march` / optional `mabi` metadata preserved
  from structured target capabilities and selected variant metadata.
* Validate that the generated object is non-empty and has an ELF object
  signature before writing it.
* Keep self-check harness exports explicit and out of the generic artifact
  front door.
* Update durable lowering-runtime/testing specs to describe the direct RVV
  microkernel object artifact boundary.
* Add focused lit/FileCheck coverage for the direct and generic object routes
  when local RVV object clang support is available.

## Acceptance Criteria

* `git diff --check` passes.
* `cmake --build build --target tcrv-opt tcrv-translate -j2` passes.
* `cmake --build build --target check-tianchenrv -- -j2` passes.
* The new object route test proves a RISC-V relocatable object is produced,
  exposes the RVV microkernel callable symbol, and does not define `main`.
* No RVV runtime/correctness/performance claim is made unless separate
  `ssh rvv` evidence is collected.

## Out Of Scope

* No generic RVV lowering to LLVM/RISC-V IR.
* No arbitrary kernel/object export.
* No automatic hardware probing or hidden runtime dispatch.
* No promotion of self-check object/source helpers into the generic target
  artifact front door.
* No Python implementation of compiler IR, plugin decisions, lowering,
  emission, target artifact selection, or capability modeling.

## Technical Notes

* Primary code paths:
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`.
* Focused tests:
  `test/Target/RVVMicrokernel/*`,
  optional C++ target artifact registry tests if selection behavior needs an
  in-process check.
* Relevant specs:
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
