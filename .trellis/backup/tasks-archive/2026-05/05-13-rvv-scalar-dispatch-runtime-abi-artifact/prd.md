# RVV+scalar dispatch runtime ABI artifact path

## Goal

Validate and, if needed, repair the production RVV+scalar dispatch artifact
route so one bounded `i32-vadd` dispatch slice carries selected exact-family
RVV source authority, scalar fallback source authority, dispatch-control EmitC
provenance, ordered runtime ABI bindings, generated source/header/object
artifacts, and a real external runtime invocation on `ssh rvv`.

This task is the next workflow boundary after the completed direct RVV
exact-family runtime ABI hardware proof. It must prove the composite dispatch
artifact path, not repeat a standalone RVV microkernel proof.

## What I Already Know

* Previous archived task
  `.trellis/tasks/archive/2026-05/05-13-rvv-exact-family-runtime-abi-hardware-proof/prd.md`
  proved the direct typed RVV `i32-vadd` exact-family source/header/object path
  with external caller execution on `ssh rvv`.
* Current dispatch ownership is in the RVV+scalar target artifact route:
  selected RVV dispatch-case callable candidate plus selected scalar
  dispatch-fallback callable candidate, then a target-owned composite
  dispatch source/header/object route.
* The lowering-runtime specs require callable parameters to come from
  `tcrv.exec.mem_window` and `tcrv.exec.runtime_param`, and require the
  dispatch availability guard to come from selected `tcrv.exec.case`
  `runtime_guard` linkage to a `dispatch-availability-guard`
  `tcrv.exec.runtime_param`.
* The dispatch source route already emits bounded metadata for component
  selected paths, component callable ABI fields, dispatch mem windows,
  runtime params, `dispatch_runtime_abi_parameter[index]`, dispatch fallback
  linkage, runtime guard linkage, and the common EmitC dispatch-control route.
* The bundle evidence helper can be used only as runner/evidence tooling. It
  must consume compiler-emitted bundle index metadata and generated artifacts;
  it must not define compiler internals or become the production owner.

## Requirements

* The dispatch export must reject an RVV component unless the component carries
  exact selected-family authority through compiler-emitted selected-plan
  metadata and route/ABI consistency checks.
* The generated dispatch source must record:
  * dispatch-control EmitC call/control provenance;
  * selected RVV component identity and source authority;
  * selected scalar component identity and source authority;
  * runtime ABI parameter binding order;
  * dispatch availability guard linkage;
  * selected RVV/scalar component identities.
* The generated dispatch header must expose exactly the dispatcher external C
  ABI prototype with the same parameter order and C type spelling as the
  generated dispatcher source.
* The generated dispatch object must be the runtime-callable library object
  for the dispatcher and embedded callable components, not a hidden self-check
  executable.
* A bounded external caller must compile, link, and run on real `ssh rvv`
  hardware against both source-built dispatch object and generated dispatch
  object when the current dispatch policy permits controllable branch coverage.
* The runtime invocation must exercise both explicit scalar fallback behavior
  and RVV-selected behavior by passing the host-provided dispatch guard values.
* If a production boundary fails, fix the first owner boundary that fails:
  dispatch ABI metadata, embedded source generation, component family
  selection, header/prototype mismatch, call operand order, route registration,
  object generation, or remote invocation setup.
* If full hardware invocation is blocked, do not fake evidence. Commit only
  coherent production-path fixes plus the exact blocker.

## Acceptance Criteria

* [ ] Direct and generic dispatch source/header/object commands emit the
      bounded `i32-vadd` RVV+scalar dispatch artifacts.
* [ ] Source artifact embeds the exact selected RVV `i32-vadd` family artifact
      and selected scalar fallback source under one dispatch function.
* [ ] Source artifact records EmitC dispatch-control provenance, component
      source authority, runtime ABI order, dispatch guard linkage, and selected
      RVV/scalar identities.
* [ ] Header prototype matches the dispatcher source ABI parameter order and C
      type spelling.
* [ ] Object artifact is non-empty RISC-V ELF relocatable library output with
      dispatcher, RVV callable, and scalar callable symbols and no `main`.
* [ ] Target artifact bundle mode records source/header/object dispatch
      records, component group, external ABI name, component roles, and ordered
      runtime ABI parameters.
* [ ] A bounded external caller compiles, links, and runs on `ssh rvv` against
      both source-built and generated dispatch object paths, observing the
      expected dispatch external ABI success marker for runtime counts 7 and
      16 and guard values 0 and 1, or an exact blocker is recorded.
* [ ] Focused build, C++ target artifact export test, focused lit, helper
      self-test or dry-run where relevant, exact artifact generation commands,
      `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass.
* [ ] The task is finished/archived and one coherent commit records the round
      if complete.

## Definition Of Done

* No descriptor-driven computation or descriptor-to-C fallback is introduced.
* No computation semantics are added to `tcrv.exec`.
* No RVV-specific branches are added to core common passes.
* No new arithmetic or dtype family is the headline result.
* No broad smoke matrix, benchmark, performance claim, or report-only task is
  treated as completion.
* Python remains bounded to evidence orchestration and artifact parsing.
* Generated runtime/binary artifacts remain under `artifacts/tmp` or remote
  scratch space and are not committed.
* Any RVV runtime/correctness claim is backed by real `ssh rvv` evidence and
  scoped only to the bounded RVV+scalar `i32-vadd` dispatch external ABI path.

## Out Of Scope

* New arithmetic or dtype expansion.
* Broad dispatch family matrix validation.
* Performance measurement.
* Generic high-level lowering correctness claims.
* Dynamic runtime hardware probing.
* Descriptor-selected compute semantics.
* Python implementation of compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission.

## Technical Notes

* Specs read for this PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-13-rvv-exact-family-runtime-abi-hardware-proof/prd.md`.
* Production source surfaces inspected before writing this PRD:
  RVV+scalar dispatch target route, direct RVV microkernel route, scalar
  fallback route, `tcrv-translate`, RVV+scalar dispatch lit tests, bundle
  route tests, and `rvv_scalar_dispatch_e2e.py` as evidence tooling reference.
* Minimal starting hypothesis: current production route likely already emits
  source/header/object and bundle metadata; this round must prove the external
  runtime-callable dispatch ABI on `ssh rvv` and repair any first failing
  production boundary instead of adding a parallel helper.

## Completion Notes

* Repaired the first production owner boundary that failed in the local manual
  dispatch object export: RVV+scalar dispatch object export and scalar fallback
  object export now fall back from `clang` to `clang-20`, matching the existing
  direct RVV microkernel object exporter behavior.
* Manual direct artifact commands emitted:
  * `artifacts/tmp/rvv_scalar_dispatch_runtime_abi_manual/post_planning.mlir`
    from `./build/bin/tcrv-opt test/Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`.
  * `artifacts/tmp/rvv_scalar_dispatch_runtime_abi_manual/rvv_scalar_dispatch.c`
    from `./build/bin/tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c`.
  * `artifacts/tmp/rvv_scalar_dispatch_runtime_abi_manual/rvv_scalar_dispatch.h`
    from `./build/bin/tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-header`.
  * `artifacts/tmp/rvv_scalar_dispatch_runtime_abi_manual/rvv_scalar_dispatch.o`
    from `./build/bin/tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-object`.
  * Generic source/header/object from
    `--tcrv-export-target-source-artifact`,
    `--tcrv-export-target-header-artifact`, and
    `--tcrv-export-target-artifact`.
  * Bundle source/header/object under
    `artifacts/tmp/rvv_scalar_dispatch_runtime_abi_manual/target_artifact_bundle/`
    from `./build/bin/tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=...`.
* Manual artifact checks observed:
  * Source records `dispatch_emitc_common_lower_to_emitc_boundary:
    TCRVLowerToEmitCSourceAuthority`, `dispatch_runtime_guard_link`,
    `rvv_callable_symbol`, `scalar_callable_symbol`, ordered
    `dispatch_runtime_abi_parameter[0..4]`, `__riscv_vadd_vv_i32m1`, and
    scalar `tcrv_scalar_i32_add`.
  * Header prototype is
    `void tcrv_dispatch_i32_vadd_dispatch_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);`.
  * Object is `ELF 64-bit LSB relocatable, UCB RISC-V, RVC, double-float ABI`
    and exposes the RVV callable, scalar callable, and dispatch symbols with
    no `main`.
  * Bundle index records three complete dispatch records: source, header, and
    object, with `component_group =
    "rvv-scalar-i32-vadd-dispatch-external-abi.v1"` and ordered runtime ABI
    parameters including `dispatch-availability-guard`.
* Real hardware evidence was collected with:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vadd --run-id 20260513T-rvv-scalar-dispatch-i32-vadd-external-abi --overwrite --timeout 120`.
* Hardware evidence artifact:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/20260513T-rvv-scalar-dispatch-i32-vadd-external-abi/evidence.json`.
* `ssh rvv` evidence summary:
  * `ssh_compile_bundle_external_caller_object`: exit 0.
  * `ssh_compile_bundle_dispatch_source_object`: exit 0.
  * `ssh_link_bundle_source_external_caller`: exit 0.
  * `ssh_run_bundle_source_external_caller`: exit 0, stdout
    `tcrv_rvv_scalar_i32_vadd_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
  * `ssh_link_bundle_index_object_external_caller`: exit 0.
  * `ssh_run_bundle_index_object_external_caller`: exit 0, stdout
    `tcrv_rvv_scalar_i32_vadd_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
* Claim scope remains bounded to RVV+scalar `i32-vadd` target-artifact bundle
  external caller correctness only. No performance, generic lowering, broad
  RVV runtime, or dynamic runtime integration claim is made.
* Spec update judgment: no `.trellis/spec/` change was needed because the
  existing specs already cover the dispatch ABI, source/header/object bundle,
  evidence-helper, and `ssh rvv` evidence boundaries used here.

## Validation

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVScalarDispatch|RVVMicrokernel|target-source-artifact-routes|target-artifact-export-registry|TargetArtifactBundleExport|rvv-scalar-dispatch-bundle-e2e'`
  with 62/62 selected tests passed.
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'scalar-target-header-object-artifact-routes|scalar-target-source-artifact-routes|scalar-target-vmul-source-artifact-routes|RVVScalarDispatch'`
  with 16/16 selected tests passed.
* `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
* `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vadd --run-id 20260513T-rvv-scalar-dispatch-i32-vadd-dry --overwrite`
* `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vadd --run-id 20260513T-rvv-scalar-dispatch-i32-vadd-external-abi --overwrite --timeout 120`
