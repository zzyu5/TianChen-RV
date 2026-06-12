# RVV compiler-produced artifact e2e closure

## Goal

Close one coherent end-to-end RVV compiler-produced artifact route for the
existing `vector-dynamic-i32-vadd` slice. The route must start from the selected
frontend boundary, materialize a concrete `tcrv_rvv.i32_vadd_microkernel` with
op-owned source identity, selected SEW/LMUL/tail/mask config, and runtime
AVL/VL role data, then reach generated RVV source/header/object artifacts and
focused `ssh rvv` compile/run evidence or a precise non-code blocker after
local artifact generation succeeds.

This task deliberately raises the bar beyond the previous bundle/FileCheck
closure: selected-state metadata alone is not the module result. The production
route must prove that compiler-produced artifacts are generated and can be used
by the existing RVV evidence path, without falling back to descriptor-owned
compute/config/runtime authority.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repo state for this round: clean worktree, HEAD
  `5030786 feat(rvv): close selected bundle state contract`.
* No `.trellis/.current-task` existed before this task; this task was created
  from the Hermes Direction Brief.
* The previous completed task closed the selected bundle state contract around
  source identity, selected config, runtime length authority, and bundle
  metadata for the dynamic i32 vadd route.
* The next bottleneck is generated artifact and hardware evidence, not another
  isolated contract, metadata, or FileCheck-only refinement.
* The specs require the current RVV route to be extension family ops ->
  EmitC/RVV intrinsic C/C++ -> clang/LLVM by default; descriptor-to-C remains
  bounded debt, and RVV runtime/correctness claims require real `ssh rvv`
  evidence.

## Requirements

* Keep scope bounded to the existing compiler-produced
  `vector-dynamic-i32-vadd` / finite RVV binary `i32-vadd` selected route.
* Do not add a new RVV family, dtype, LMUL matrix, broad smoke matrix, or
  performance-tuning objective.
* The upstream selected route must materialize exactly the expected
  `tcrv_rvv.i32_vadd_microkernel` with:
  * op-owned selected source identity;
  * selected SEW 32, LMUL, tail policy, mask policy, vector type/suffix, and
    setvl suffix;
  * runtime element-count ABI parameter, runtime AVL source/role, runtime VL
    source/scope, and dynamic source-tail authority;
  * structured `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` / load-add-store dataflow.
* Selected emission planning and artifact export must consume the same
  materialized state through `RVVSelectedConfigContract` and
  `RVVRuntimeLengthContract`.
* Generated C must use clang/LLVM-compatible RVV C intrinsics through
  `riscv_vector.h` and must preserve EmitC-route provenance from the verified
  `tcrv_rvv` family body.
* The existing e2e runner path must operate on compiler-produced generated
  source/header/object artifacts for this vadd route, not only a hand-authored
  target fixture.
* If `ssh rvv` cannot complete, local artifact generation must still succeed
  first and the blocker must be exact: ssh, scp, compiler/header/sysroot,
  object/link/run, timeout, or success-marker failure.
* Descriptor-only or stale descriptor-driven source/config/runtime authority
  must be rejected, bypassed, deleted, or quarantined for this migrated vadd
  slice before source/header/object/bundle output.
* Core `tcrv.exec` remains orchestration-only. Generic transforms and shared
  artifact routing must not gain RVV semantic branches.
* Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck. Python is allowed only for the existing runner/evidence
  tooling and small support logic.

## Acceptance Criteria

* [x] The compiler-produced dynamic vadd path reaches post-planning MLIR with
      one selected RVV path, one matching `tcrv_rvv.lowering_boundary`, and one
      concrete `tcrv_rvv.i32_vadd_microkernel` carrying selected source
      identity, selected config, runtime AVL/VL role data, and callable ABI
      role state.
* [x] Selected emission planning, direct target artifact export, and bundle
      export consume the same selected config and runtime length contracts for
      source/header/object records.
* [x] Generated source uses `#include <riscv_vector.h>`, selected `vsetvl`,
      load, add, and store RVV intrinsics derived from the verified family-op
      body, not from descriptor text alone.
* [x] Generated header exposes only the external callable ABI prototype, and
      generated object output is a non-empty RISC-V ELF relocatable object with
      the expected exported symbol and no hidden `main`.
* [x] The e2e path has a compiler-produced artifact-bundle mode for the
      dynamic vadd route that discovers generated source/header/object files
      from the bundle index, validates their selected-state metadata, compiles
      or links them on `ssh rvv`, runs the bounded external caller, and records
      sanitized command logs and evidence JSON.
* [x] Real `ssh rvv` evidence succeeded; the final report must state only the
      bounded claim: the finite compiler-generated i32-vadd RVV artifact route
      compiled/linked/ran through the selected external ABI on that host. It
      must not claim generic RVV lowering, performance, or broad correctness.
* [x] No `ssh rvv` blocker occurred. The runner still records exact command
      phase summaries and would fail closed after local generation if ssh, scp,
      compile, link, run, timeout, or success-marker validation failed.
* [x] Fail-closed tests cover missing/stale/wrong selected source identity,
      missing/conflicting selected config, missing/stale runtime length role
      data, descriptor-only production attempts, and at least one stale typed
      body or EmitC-route mismatch before artifact output.
* [x] Focused regressions for the previous selected-route surfaces still pass:
      op-owned object artifact, runtime-length consumption, selected-config
      consumption, selected-source bundle export, full-state bundle contract,
      RVV microkernel, scalar dispatch, VectorToExec, LinalgToExec, and target
      artifact tests touched by this route.
* [x] A bounded ref-scan confirms there is no new RVV semantic branch in core
      `tcrv.exec`, generic transforms, or generic artifact bundle routing, and
      no descriptor-only compute/config/runtime default route for the migrated
      vadd slice.
* [ ] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean worktree, and one coherent
      commit complete the round if the task is finished.

## Definition Of Done

The default compiler-produced `vector-dynamic-i32-vadd` selected route is a
single connected production path from selected frontend boundary to generated
RVV source/header/object artifacts, with the same materialized `tcrv_rvv`
selected state consumed throughout planning/export/evidence. The task is not
done with contract-only, FileCheck-only, metadata-only, e2e-runner-only, or
workspace-journal evidence. Real hardware evidence is collected through
`ssh rvv`, or the exact non-code blocker is recorded after local generation
passes.

## Out Of Scope

* New RVV family, i64 expansion, dtype matrix, broad LMUL matrix, broad smoke
  matrix, or performance tuning.
* Descriptor-to-C production export or descriptor element count/vector shape as
  authoritative compute, selected config, or runtime control.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic core orchestration or generic artifact
  bundle routing.
* GCC/vendor compiler as the default route.
* Template, Toy, TensorExtLite, IME, Offload, or unrelated plugin changes
  except narrow regressions caused by shared validation.
* Runtime, correctness, or performance claims beyond focused evidence actually
  run for this finite compiler-generated i32-vadd artifact route.

## Technical Notes

Specs read before PRD:

* `.trellis/spec/index.md`
* `.trellis/spec/architecture/design-boundaries.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/capability-model/capability-contract.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-bundle-full-state-closure/prd.md`
* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-route-closure/prd.md`
* `.trellis/workspace/codex/journal-5.md` selected RVV route/config/runtime
  and bundle closure sections.

Likely implementation surface:

* `include/TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h`
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
* `include/TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h`
* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/RVV/RVVScalarDispatch.cpp`
* `scripts/rvv_microkernel_e2e.py`
* `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
* Focused RVV microkernel, scalar-dispatch, VectorToExec, LinalgToExec, and
  target artifact tests touched in the previous selected-route rounds.

## Implementation Summary

Completed in this round:

* Confirmed the production C++/MLIR route at HEAD already had the required
  selected-state plumbing for the dynamic vector i32-vadd slice:
  `VectorToExec` frontend lowering -> selected RVV path ->
  `tcrv_rvv.i32_vadd_microkernel` -> plan-and-export target artifact bundle ->
  generated source/header/object.
* Added focused lit coverage to
  `test/Scripts/rvv-microkernel-bundle-e2e.test` for the exact
  `--lower-vector-i32-vadd-frontend` compiler-produced route, using the C++
  `--tcrv-plan-and-export-target-artifact-bundle` front door, runtime counts
  `7,16,23`, and selected kernel
  `frontend_vector_dynamic_i32_vadd`.
* The new script test checks the generated evidence JSON, generated RVV source,
  generated header inclusion through the external caller, dynamic source-tail
  metadata, `riscv_vector.h` RVV intrinsic route, and no runtime/performance
  claims in dry-run mode.
* No production C++ code change was needed: the observed route already consumed
  `RVVSelectedConfigContract` and `RVVRuntimeLengthContract` and emitted the
  required source/header/object artifacts. This round changed only focused test
  coverage plus Trellis task records, then collected real hardware evidence.

No core `tcrv.exec` dialect, generic orchestration pass, generic artifact
bundle router, descriptor-to-C production path, dtype matrix, LMUL matrix, or
performance path was added.

## Evidence Summary

Main generated artifact and `ssh rvv` evidence:

```text
python3 scripts/rvv_microkernel_e2e.py \
  --arithmetic-family i32-vadd \
  --lower-vector-i32-vadd-frontend \
  --use-target-artifact-bundle \
  --use-plan-and-export-bundle-front-door \
  --expect-selected-kernel frontend_vector_dynamic_i32_vadd \
  --runtime-count 7 --runtime-count 16 --runtime-count 23 \
  --run-id codex-current-ssh --overwrite \
  --timeout 120 --connect-timeout 10
```

Result:

* status: `success`
* mode: `ssh`
* repo commit: `50307866202b24e5d3b950da27684c3c924bb7f6`
* input: `test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir`
* front door: `tcrv-plan-and-export-target-artifact-bundle`
* source route: `tcrv-export-rvv-microkernel-c`
* selected kernel: `frontend_vector_dynamic_i32_vadd`
* selected config: `i32m1`, SEW 32, LMUL m1, tail/mask agnostic
* generated source:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-current-ssh/target_artifact_bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-microkernel-c.c`
* generated header:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-current-ssh/target_artifact_bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-microkernel-header.h`
* generated object:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-current-ssh/target_artifact_bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o`
* generated external caller:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-current-ssh/rvv_microkernel_external_caller.c`
* ssh host facts: `uname -m` = `riscv64`, clang =
  `Ubuntu clang version 18.1.3 (1ubuntu1)`
* remote compile/link/run: both source-built object and bundle object paths
  compiled, linked, and ran successfully
* stdout marker for both runs:
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16,23`

Bounded claim: this proves only that the finite compiler-generated RVV
i32-vadd target artifact bundle source/header/object external ABI compiles,
links, and runs on `ssh rvv` for runtime counts `7,16,23`. It is not generic
RVV lowering, broad correctness, dynamic runtime integration, throughput,
latency, or performance evidence.

## Validation Summary

Repository and task setup:

* `pwd`
* `git status --short`
* `git log --oneline -8`
* `.trellis/spec/index.md` plus the relevant specs listed above were read
* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-bundle-full-state-closure/prd.md`
* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-route-closure/prd.md`

Focused build:

* `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`

Focused C++/script tests:

* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `python3 scripts/rvv_microkernel_e2e.py --self-test`

Focused lit:

* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport Target/RVVMicrokernel Target/RVVScalarDispatch Target/ArtifactExport Transforms/VectorToExec Transforms/LinalgToExec`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-microkernel-bundle-e2e.test`

Bounded core-neutrality scan:

```text
rg -n "rvv|RVV|tcrv_rvv|selected_vector|runtime_avl|runtime_vl|descriptor_element_count" \
  lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec
```

The matches are the expected RVV frontend lowering owner
`lib/Transforms/LowerSourceRVVBinaryToExec.cpp` and Exec dialect documentation
text. No new generic core or `tcrv.exec` RVV semantic branch was added in this
round.

## Spec Update Judgment

No `.trellis/spec/` update is needed. The existing architecture, RVV plugin,
EmitC route, lowering/runtime, capability, and testing specs already require
exactly this behavior: selected family ops feed EmitC/RVV intrinsic source,
target artifacts consume selected state through contracts, descriptor-only
authority remains quarantined, and real RVV runtime/correctness claims require
`ssh rvv` evidence. This round added focused test coverage and evidence for an
existing specified route rather than introducing a new contract.
