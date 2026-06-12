# RVV exact-family runtime ABI hardware proof

## Goal

Carry one existing descriptor-free typed RVV `i32-vadd` M1/agnostic runtime-callable production artifact path from selected `rvv-plugin` boundary through generated source, header, relocatable object, and a real `ssh rvv` external caller invocation. The proof must exercise the production exact-family source/header/object routes and validate the emitted runtime ABI, not a standalone compute source.

## Requirements

* Use the existing typed RVV `i32-vadd` route owned by selected `tcrv_rvv.i32_vadd_microkernel` body authority.
* Export source and header through the manifest-backed exact RVV family routes, preserving common EmitC/interface provenance and IR-backed runtime ABI metadata.
* Export or otherwise consume the production object route for the same selected exact family.
* Compile and run a bounded external caller on the real RVV target through `ssh rvv`; the caller may only verify the generated runtime-callable ABI.
* Keep all compiler behavior in C++/MLIR/TableGen/CMake/lit. Python is allowed only as evidence tooling/runner orchestration.
* Fix the first production owner boundary that fails: selected boundary, header ABI, object export, EmitC call mapping, runtime ABI order, local object compilation, or remote compile/run invocation.
* If hardware or toolchain access blocks execution, do not fake evidence; leave the task open or report the exact blocker with coherent production-path fixes only.

## Acceptance Criteria

* [x] A descriptor-free typed RVV `i32-vadd` input exports runtime-callable source with common EmitC source authority, generated op-interface provenance, and ordered runtime ABI metadata.
* [x] The same selected exact RVV family exports a declaration-only header whose prototype matches the source ABI order and C type spelling.
* [x] The same selected exact RVV family exports a non-empty RISC-V ELF relocatable object through the production object route.
* [x] A bounded external caller compiles and runs on `ssh rvv` against both source-built and generated-object paths, observes the expected success marker, and checks add results for bounded runtime `n` values.
* [x] Focused build, C++ test, focused lit, evidence helper run, `git diff --check`, `git diff --cached --check`, and Trellis validation pass.
* [x] The task is finished/archived and one coherent commit records the round if complete.

## Definition Of Done

* No descriptor-driven compute or direct descriptor-to-C fallback is introduced.
* No computation semantics are added to `tcrv.exec`.
* No RVV-specific branches are added to shared/common passes.
* No new arithmetic family, dtype expansion, broad smoke matrix, benchmark, performance claim, or generic RVV lowering claim is introduced.
* Generated binary/runtime artifacts remain under `artifacts/tmp` or remote scratch space and are not committed.
* Hardware evidence is reported only as bounded runtime ABI correctness for this exact RVV `i32-vadd` slice.

## Technical Approach

Use the existing production route as the owner of the proof:

1. Run the focused build targets for target/export, transforms, translate, and target artifact C++ tests.
2. Use `tcrv-opt` to materialize the selected typed RVV path from the checked-in microkernel fixture.
3. Use `tcrv-translate` exact-family route commands to emit runtime-callable source, header, and object.
4. Use `scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vadd` in non-dry-run mode to collect the bounded `ssh rvv` external ABI evidence from generated source/header/object and generated caller.
5. If a production boundary fails, patch that boundary rather than adding a separate smoke owner.

## Out Of Scope

* RVV sub/mul/i64 expansion.
* Dispatch or scalar fallback proof.
* Performance measurement.
* Generic linalg lowering proof.
* New runtime ABI families.
* Descriptor-only or descriptor-to-C production paths.

## Technical Notes

* Specs read: `.trellis/spec/index.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
* Previous task read: `.trellis/tasks/archive/2026-05/05-13-rvv-descriptor-authority-quarantine/prd.md`.
* Source surfaces inspected: `include/TianChenRV/Target/RVV/RVVMicrokernel.h`, `lib/Target/RVV/RVVMicrokernel.cpp`, `lib/Target/TargetArtifactExport.cpp`, `tools/tcrv-translate/tcrv-translate.cpp`, `scripts/rvv_microkernel_e2e.py`, and `test/Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir`.
* The evidence helper is runner tooling only. The production authority remains the exact-family artifact exporters and common EmitC route.

## Completion Notes

* No production code change was required. Current `HEAD` already carries the descriptor-free typed RVV `i32-vadd` runtime ABI through the production exact-family routes.
* Manual exact-route artifacts were emitted under `artifacts/tmp/rvv_exact_family_runtime_abi_manual/`:
  * `post_planning.mlir` from `./build/bin/tcrv-opt test/Target/EmissionManifest/emission-manifest-rvv-microkernel.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`.
  * `rvv_microkernel.c` from `./build/bin/tcrv-translate --tcrv-export-rvv-microkernel-c`.
  * `rvv_microkernel.h` from `./build/bin/tcrv-translate --tcrv-export-rvv-microkernel-header`.
  * `rvv_microkernel.o` from `./build/bin/tcrv-translate --tcrv-export-rvv-microkernel-object`.
* Manual artifact checks observed:
  * Source contains `TCRVLowerToEmitCSourceAuthority`, `TCRVEmitCLowerableOpInterface`, `__riscv_vadd_vv_i32m1`, `callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param`, and runtime ABI `lhs, rhs, out, n`.
  * Header prototype is `void tcrv_rvv_i32_vadd_microkernel_rvv_microkernel_manifest_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);`.
  * Object file is `ELF 64-bit LSB relocatable, UCB RISC-V, RVC, double-float ABI`.
* Real hardware evidence was collected with:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vadd --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --run-id 20260513T-rvv-exact-family-i32-vadd --overwrite --timeout 120`.
* Hardware evidence artifact:
  `artifacts/tmp/rvv_microkernel_e2e/20260513T-rvv-exact-family-i32-vadd/evidence.json`.
* `ssh rvv` evidence summary:
  * `ssh_compile_external_caller_object`: exit 0.
  * `ssh_compile_external_source_object`: exit 0.
  * `ssh_link_external_source_caller`: exit 0.
  * `ssh_run_external_source_caller`: exit 0, stdout `tcrv_rvv_microkernel_external_abi_ok counts=7,16`.
  * `ssh_link_external_object_caller`: exit 0.
  * `ssh_run_external_object_caller`: exit 0, stdout `tcrv_rvv_microkernel_external_abi_ok counts=7,16`.
* Claim scope remains bounded to generated RVV `i32-vadd` direct helper artifact handoff plus header/object external caller correctness only. No performance, generic lowering, or broad RVV runtime claim is made.

## Validation

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`: `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVMicrokernel|target-source-artifact-routes|target-artifact-export-registry'` with 33/33 selected tests passed.
* `python3 scripts/rvv_microkernel_e2e.py --self-test`
* `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vadd --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --run-id 20260513T-rvv-exact-family-i32-vadd --overwrite --timeout 120`
* `git diff --check`
* `git diff --cached --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-exact-family-runtime-abi-hardware-proof`
