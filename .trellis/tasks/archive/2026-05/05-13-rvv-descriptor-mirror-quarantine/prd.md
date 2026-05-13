# RVV descriptor-mirror quarantine for artifact emission

## Goal

Quarantine descriptor-mirror authority from the existing dynamic RVV artifact
emission route now that typed `tcrv_rvv` family ops and the common EmitC
lowerable route own emitted computation. The primary path is dynamic i32m1
`i32-vsub` direct source/header/object emission and plan-and-export bundle
emission; dynamic i32m1 `i32-vadd` is the required regression.

This round should make any retained descriptor-shaped metadata visibly
compatibility, diagnostic, or bounded component-capacity metadata. It must not
select arithmetic family, intrinsic spelling, selected vector shape, runtime
AVL/VL authority, runtime ABI parameter names, or artifact route identity.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree state was clean.
* Initial HEAD was `16eee6e fix(rvv): derive emitc intrinsics from typed ops`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the supplied Hermes direction brief before source edits.
* The previous completed task moved RVV arithmetic intrinsic selection to typed
  compute source-op provenance and selected-family agreement.
* Specs require the production route to be extension family ops -> EmitC ->
  RVV intrinsic C/C++, with descriptors only as bounded transition debt.
* The current code already validates stale `tcrv_rvv.lowering_descriptor`
  against typed microkernel bodies; the remaining confusing surface is that
  `tcrv_rvv.descriptor_element_count` is still treated with the generic legacy
  descriptor-mirror role even when typed selected-plan authority owns the route.

## Scope

In scope:

* Dynamic i32m1 `i32-vsub` direct source/header/object artifact export.
* Dynamic i32m1 `i32-vsub` plan-and-export target artifact bundle export.
* Dynamic i32m1 `i32-vadd` as a regression for the same direct and bundle path.
* RVV selected-plan metadata and route preflight around descriptor mirror versus
  typed source authority.
* Explicit validation that retained descriptor-local element count is bounded
  component capacity metadata only.
* Focused tests for stale/missing/mismatched descriptor metadata not selecting
  arithmetic, shape, runtime count authority, intrinsic spelling, or ABI names.

Out of scope:

* New vmul, i64, LMUL, dtype, family, or vector-shape expansion as the main
  result.
* Generic plugin framework rewrite or route-registration-only cleanup.
* Descriptor-driven computation or direct descriptor-to-C export.
* Moving compute semantics into `tcrv.exec`.
* Python compiler implementation.
* Performance claims or broad benchmark matrices.

## Requirements

* Compiler implementation remains C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
* Typed `tcrv_rvv` body and selected config/runtime ABI contracts remain
  production authority for compute, shape, runtime AVL/VL, and ABI parameter
  binding.
* `tcrv_rvv.lowering_descriptor`, when present, is an optional legacy mirror
  validated only after typed authority exists; stale or descriptor-only cases
  must fail closed before artifact output.
* `tcrv_rvv.descriptor_element_count` must be named or commented as bounded
  descriptor-local component capacity metadata, not as generic descriptor
  mirror authority.
* Direct source/header/object and plan-and-export bundle must consume the same
  typed selected-config/runtime AVL contract.
* Plugin-manifest route activation, family registry lookup, selected-config
  fail-closed validation, fixed-vector vadd extent enforcement, and dynamic
  tail/runtime-VL authority must remain intact.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as current before implementation.
* [x] Dynamic i32-vsub direct source/header/object export does not require
      descriptor mirror metadata to choose family, intrinsic spelling, selected
      shape, runtime AVL/VL authority, or ABI parameter names.
* [x] Dynamic i32-vadd remains a regression on the same typed authority path.
* [x] Plan-and-export bundle emission for vsub and vadd uses the same typed
      selected-config/runtime AVL authority as direct export.
* [x] Retained `descriptor_element_count` metadata is explicitly bounded
      descriptor-local component capacity metadata and is not validated with the
      legacy descriptor-mirror role.
* [x] Stale or mismatched `tcrv_rvv.lowering_descriptor` remains fail-closed
      before source/header/object/bundle output.
* [x] Tests or C++ coverage prove descriptor mirrors cannot select add versus
      sub, selected shape, runtime count authority, intrinsic spelling, or ABI
      parameter names.
* [x] Focused build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test`.
* [x] Focused lit/C++ checks cover dynamic vector vsub/vadd, RVV microkernel
      materialization, selected-config fail-closed regressions,
      `TargetArtifactBundleExport`, `EmissionManifest`, fixed-vector vadd,
      dynamic-tail/runtime-VL authority, plugin route activation, family
      registry, and representative linalg compatibility if touched.
* [x] Exact direct and bundle artifact commands are recorded for dynamic
      i32-vsub and the dynamic i32-vadd regression.
* [x] Fresh `ssh rvv` evidence is collected if emitted source/object runtime
      behavior changes. If the round only changes metadata quarantine and
      preflight wording, the report states the narrower claim.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] If complete, the task is finished, archived, and committed as one
      coherent commit. If incomplete, it remains open with the exact next
      continuation point.

## Definition of Done

The dynamic i32-vsub artifact route has no descriptor-mirror authority in its
default direct or bundle emission path. Dynamic i32-vadd stays green on the same
route, descriptor-local element count is quarantined as bounded component
capacity metadata only, and stale descriptor mirrors fail closed as diagnostic
compatibility checks rather than influencing emitted code.

## Completion Notes

Production authority change:

* Added a distinct `rvv-descriptor-local-component-capacity` role and note for
  retained `tcrv_rvv.descriptor_element_count` selected-plan metadata.
* Rewired RVV selected-emission planning to emit descriptor element count under
  that bounded capacity role instead of the legacy descriptor-mirror role.
* Rewired RVV direct microkernel and scalar-dispatch preflight validation so
  `descriptor_element_count` is accepted only as bounded component capacity
  metadata. The typed RVV source-op family/config and runtime ABI contract
  remain the production authority for add/sub selection, intrinsic spelling,
  vector shape, runtime AVL/VL, and ABI names.
* Added a target artifact export regression that rejects the legacy descriptor
  mirror role for `tcrv_rvv.descriptor_element_count`.
* Fixed the `tianchenrv-rvv-binary-planning-test` link dependencies so the
  route registration symbols provided by `TianChenRVTarget` are available after
  `TianChenRVRVVTarget`.

Artifact evidence root:

```text
artifacts/tmp/rvv_descriptor_mirror_quarantine_20260513T082941Z
```

Focused checks:

```bash
cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test -j2
./build/bin/tianchenrv-target-artifact-export-test
./build/bin/tianchenrv-i32-binary-family-registry-test
./build/bin/tianchenrv-rvv-binary-planning-test
./build/bin/tianchenrv-rvv-extension-plugin-test
./build/bin/tianchenrv-emitc-lowerable-interface-test
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir Target/RVVMicrokernel Target/EmissionManifest Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-i32-vadd-and-export-target-artifact-bundle.mlir Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir Transforms/LinalgToExec/linalg-i32-vsub-to-exec.mlir
git diff --check
```

The focused lit run passed 50/50 with the existing warning that
`Transforms/LinalgToExec/linalg-i32-vsub-to-exec.mlir` contains no tests.
No fresh `ssh rvv` run was required because this round changed metadata
quarantine and preflight validation, not emitted runtime arithmetic behavior.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/core-dialect/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-op-owned-emitc-artifact-route/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-extension-op-emitc-source-authority/prd.md`
* `.trellis/workspace/codex/journal-5.md`

Initial code inspection targets:

* `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
* `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
* `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
* `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/RVV/RVVScalarDispatch.cpp`
* `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
* dynamic vector add/sub tests and RVV artifact bundle tests.
