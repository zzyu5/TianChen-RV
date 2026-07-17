# RVV extension-op EmitC source authority

## Goal

Make the default RVV finite-binary source artifact emission for dynamic
i32-vsub derive emitted intrinsic computation from materialized `tcrv_rvv`
extension-family ops and the common EmitC lowerable route, with dynamic
i32-vadd preserved as the required regression. This round closes the next
production-route gap after RVV target-support route activation: route ownership
is already plugin/manifest-owned, but source emission still needs to prove that
typed selected RVV family ops, not descriptor-local compute metadata, are the
authority for emitted arithmetic semantics.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree state for this task was clean.
* Initial HEAD was `c7239d0 refactor(rvv): activate target support through
  plugin manifest`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the supplied Direction Brief before source edits.
* The previous completed route-activation task moved RVV target-support
  artifact and translate route activation through `RVVExtensionPlugin` manifest
  hooks.
* The current architectural mainline is extension family ops -> common EmitC
  lowerable route -> RVV intrinsic C/C++ -> clang/LLVM/native compiler.
* Descriptor-driven computation is implementation debt. Any remaining
  `tcrv_rvv.lowering_descriptor` mirror is compatibility/diagnostic metadata
  only after typed family/body authority is established.
* Current production scope is the existing bounded i32m1 add/sub path,
  prioritizing dynamic i32-vsub and keeping dynamic i32-vadd as a regression.

## Scope

In scope:

* Existing dynamic vector i32-vsub direct source/header/object artifact export.
* Existing dynamic vector i32-vsub plan-and-export target artifact bundle
  export.
* Dynamic vector i32-vadd as the required regression on the same source
  authority boundary.
* Materialized `tcrv_rvv` selected microkernel body checks for setvl/with_vl,
  load/load/arithmetic/store dataflow, selected vector shape, tail/mask policy,
  runtime AVL/VL boundary, and ABI role/name mapping.
* Common EmitC lowerable route evidence for the selected RVV family ops,
  including generated route comments and family-correct intrinsic calls.
* Fail-closed tests proving stale descriptor/operator metadata cannot change
  emitted vsub/vadd semantics.

Out of scope:

* New vmul, i64, LMUL, dtype, vector-shape, or family expansion as the main
  result.
* Route-registration-only cleanup or a broad plugin framework rewrite.
* Descriptor-driven computation or direct descriptor-to-C export as a default
  route.
* Moving computation semantics into `tcrv.exec`.
* Python implementation of compiler core, dialects, passes, plugin registry,
  lowering, or emission.
* Performance claims, broad benchmark matrices, standalone smoke/evidence
  packaging, or report-only closeout.

## Requirements

* Compiler implementation remains in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* The selected materialized RVV body must be the compute source of truth for
  emitted arithmetic. For this task, `tcrv_rvv.i32_sub` must select vsub and
  `tcrv_rvv.i32_add` must select vadd before any descriptor mirror is
  considered.
* The target/export path must consume a common EmitC lowerable route payload
  derived from the verified typed RVV family body, not independently choose
  arithmetic semantics from `tcrv_rvv.lowering_descriptor` or detached route
  metadata.
* Selected shape, SEW, LMUL, tail policy, mask policy, runtime AVL/VL
  boundary, and ABI parameter names/types must remain read from the
  materialized selected RVV op path or the shared selected-config/runtime ABI
  contract.
* Stale descriptor/operator metadata must fail closed before source/header/
  object/bundle output. Descriptor-only paths must not emit executable source.
* Any descriptor mirror that remains must be documented and validated only as a
  compatibility/diagnostic mirror after typed family/body authority is known.
* Direct source/header/object export and plan-and-export bundle export must
  agree on family, selected shape, route id, runtime ABI metadata, and emitted
  intrinsic semantics.
* Plugin-manifest route activation, selected-config fail-closed validation,
  finite family registry behavior, fixed-vector vadd extent enforcement, and
  dynamic transfer-tail authority must remain intact.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as current before implementation.
* [x] Dynamic i32-vsub direct source export emits `__riscv_vsub_vv_i32m1`
      because the verified materialized body contains `tcrv_rvv.i32_sub`, not
      because a descriptor string selected vsub.
* [x] Dynamic i32-vadd direct source export still emits
      `__riscv_vadd_vv_i32m1` because the verified materialized body contains
      `tcrv_rvv.i32_add`.
* [x] Generated source records common EmitC route metadata for the selected
      source ops, including source op names and `emitc.call_opaque` intrinsic
      mappings, for both vsub and vadd.
* [x] Stale descriptor/body mismatch tests prove a descriptor mirror cannot
      override the materialized RVV arithmetic op and cannot silently emit the
      wrong intrinsic.
* [x] Descriptor-only selected RVV executable emission remains unsupported or
      fails closed before source/header/object/bundle output.
* [x] Selected vector-shape metadata, tail/mask policy, runtime AVL/VL
      metadata, source-tail authority, and ABI parameter name/type ownership
      continue to be validated through the selected RVV op path or shared
      selected-config/runtime ABI contract.
* [x] Dynamic i32-vsub and i32-vadd plan-and-export bundles agree with direct
      artifacts on RVV+scalar source/header/object route metadata, selected
      config, runtime VL boundary, and emitted arithmetic semantics.
* [x] Focused build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test`.
* [x] Focused tests cover dynamic vector vsub/vadd, RVV microkernel
      materialization, selected-config fail-closed regressions,
      `TargetArtifactBundleExport`, `EmissionManifest`, family registry,
      fixed-vector vadd extent enforcement, dynamic-tail authority, and
      representative linalg compatibility if touched.
* [x] Exact direct and bundle artifact commands are recorded for dynamic
      i32-vsub plus at least one i32-vadd regression.
* [x] Fresh `ssh rvv` evidence is collected for dynamic i32-vsub if emitted
      source/object/bundle materialization changes; otherwise the final report
      states why no new runtime claim was made.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] If complete, the task is finished, archived, and committed as one
      coherent commit. If incomplete, it remains open with the exact next
      continuation point.

## Definition Of Done

* RVV finite-binary source artifact emission for dynamic i32-vsub is a
  production route whose emitted arithmetic semantics are derived from
  materialized `tcrv_rvv` extension ops and a common EmitC lowerable route.
* Dynamic i32-vadd remains a regression on the same authority boundary.
* Descriptor-local compute metadata is removed from, bypassed by, or explicitly
  staged outside the default compute-authority path.
* Evidence stays bounded to the changed source-emission authority and does not
  claim generic RVV lowering, full runtime integration, or performance.

## Technical Approach

Inspect the current RVV materialization/export path, then migrate the emission
payload so target/export source generation validates and consumes the typed RVV
body through the existing common EmitC lowerable route surface. Keep RVV family
specific intrinsic names, selected vector suffixes, ABI spellings, and route ids
inside RVV target/plugin code; keep shared orchestration target-neutral. Add
negative coverage around descriptor/body mismatch and descriptor-only paths so
descriptor metadata cannot be the default computation authority.

## Decision (ADR-lite)

**Context**: RVV route ownership is already activated through plugin/manifest
hooks, and selected-config/runtime-VL validation is already fail-closed at the
artifact boundary. The remaining risk is that emitted C arithmetic can still
appear to come from descriptor-local compute metadata rather than materialized
extension family ops.

**Decision**: Use the typed materialized `tcrv_rvv` body plus common EmitC
lowerable route payload as the source-emission authority for the existing
i32-vsub/i32-vadd direct and bundle artifact paths. Treat descriptor text only
as optional compatibility/diagnostic mirror metadata after typed authority is
known.

**Consequences**: This preserves the bounded add/sub route while moving the
default production path toward the documented extension-op -> EmitC -> intrinsic
mainline. Any remaining descriptor cleanup outside this route must be staged
explicitly instead of left as hidden compute authority.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/core-dialect/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-13-05-13-extension-manifest-target-support-route-activation/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-target-support-artifact-bundle-extraction/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-selected-config-fail-closed-artifact-validation/prd.md`
* `.trellis/workspace/codex/journal-5.md`

Initial code inspection targets:

* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `include/TianChenRV/Target/RVV/`
* `lib/Target/RVV/`
* `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
* `tools/tcrv-translate/tcrv-translate.cpp`
* RVV microkernel, dynamic vector add/sub, target artifact bundle, and
  selected-config fail-closed tests.

## Completion Notes

Implementation result:

* `lib/Target/RVV/RVVMicrokernel.cpp` now prints
  `descriptor_mirror_status` in both source and header artifacts. The comment
  records that any legacy descriptor mirror is compatibility/diagnostic metadata
  only after typed RVV body authority and cannot select emitted compute
  semantics.
* `test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir` now
  asserts the dynamic vsub selected-plan `tcrv_rvv.emitc_source_op`,
  `TCRVEmitCLowerableOpInterface`, typed source op dataflow step, common EmitC
  route source ops, and `__riscv_vsub_vv_i32m1` mapping from
  `tcrv_rvv.i32_sub`. It also injects stale `i32-vadd-microkernel.v1`
  descriptor metadata and requires fail-closed rejection before export.
* `test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir` keeps
  dynamic vadd as the regression and asserts the same authority chain for
  `tcrv_rvv.i32_add` -> `__riscv_vadd_vv_i32m1`, with stale
  `i32-vsub-microkernel.v1` descriptor metadata rejected before export.

Production source-emission authority before/after:

* Before this round, inspection showed production already built the generated C
  route from verified materialized RVV family bodies through
  `TCRVEmitCLowerableRoute`, with selected-plan metadata for
  `tcrv_rvv.emitc_source_op` and `TCRVEmitCLowerableOpInterface`; dynamic lit
  coverage did not fully assert that chain and generated source/header artifacts
  did not explicitly document the descriptor mirror status.
* After this round, dynamic vsub/vadd tests lock the typed family op and common
  EmitC route as the emitted intrinsic authority, and generated source/header
  artifacts explicitly mark descriptor mirrors as non-authoritative diagnostics.

Descriptor authority status:

* Removed from default compute authority: no emitted add/sub intrinsic is
  selected from `tcrv_rvv.lowering_descriptor`.
* Bypassed/fail-closed: stale descriptor mirrors on dynamic vsub/vadd fail before
  source export with diagnostics that the typed RVV body names the authoritative
  family.
* Left staged: descriptor mirror metadata remains only as legacy compatibility
  and diagnostic metadata after typed selected-plan authority is established;
  descriptor-only production emission remains covered by existing fail-closed
  tests.

Artifact evidence root:

```text
artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z
```

Direct artifact commands:

```bash
build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.h
build/bin/tcrv-translate --tcrv-export-rvv-i32-vsub-microkernel-object artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.o

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.h
build/bin/tcrv-translate --tcrv-export-rvv-microkernel-object artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.o
```

Bundle commands:

```bash
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/bundle/vector_dynamic_i32_vsub test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/bundle/vector_dynamic_i32_vsub/stdout.txt
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/bundle/vector_dynamic_i32_vadd test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/bundle/vector_dynamic_i32_vadd/stdout.txt
```

RVV hardware evidence:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --input test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --expect-selected-kernel frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --artifact-root artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/e2e --run-id 20260513T064534Z-rvv-extension-op-emitc-source-authority-vsub --overwrite --timeout 120
```

Result: success. Evidence lives at
`artifacts/tmp/rvv_extension_op_emitc_source_authority_20260513T064534Z/e2e/20260513T064534Z-rvv-extension-op-emitc-source-authority-vsub/evidence.json`.
Both bundle source and bundle object external caller paths ran on `ssh rvv` and
reported `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16,23`.

Checks run:

```bash
cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel Target/EmissionManifest Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-i32-vadd-and-export-target-artifact-bundle.mlir Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir Transforms/LinalgToExec/linalg-i32-vsub-to-exec.mlir
./build/bin/tianchenrv-target-artifact-export-test
./build/bin/tianchenrv-i32-binary-family-registry-test
./build/bin/tianchenrv-rvv-binary-planning-test
./build/bin/tianchenrv-rvv-extension-plugin-test
./build/bin/tianchenrv-rvv-selected-lowering-boundary-test
./build/bin/tianchenrv-emitc-lowerable-interface-test
```

Self-repair:

* Initial focused lit run from repo root failed because the generated
  `build/test/lit.site.cfg.py` loads `../../test/lit.cfg.py` relative to
  `build/test`; reran from `build/test` and the focused tests passed.
* Initial artifact inspection used bare `llvm-readobj`, which was not on
  `PATH`; reran object inspection with `/usr/lib/llvm-20/bin/llvm-readobj`.
* The broad focused lit invocation warned that
  `Transforms/LinalgToExec/linalg-i32-vsub-to-exec.mlir` contains no tests; the
  representative linalg compatibility path that actually ran was
  `Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`, and vsub bundle/linalg
  route coverage remains under `TargetArtifactBundleExport`/RVV microkernel
  focused tests.
