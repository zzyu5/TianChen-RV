# Descriptor-driven RVV binary runtime ABI boundary

## Goal

Make the selected RVV binary microkernel runtime ABI boundary compiler-owned and descriptor-consumed for the finite RVV add/sub/mul families across i32 and i64. A selected family descriptor must drive runtime ABI identity, callable C parameter types, parameter roles/ownership, external ABI component identity, and runtime element-count ABI metadata through planning, diagnostics, manifests, and artifact export.

## Why Now

The previous module added real `ssh rvv` profile-replay evidence for `i64-vsub` and `i64-vmul` at commit `5a37883`. More replay script expansion would be evidence plumbing. The next compiler bottleneck is removing active reliance on i32-vadd/add compatibility defaults when a concrete selected RVV binary family is available.

## Current Repository Facts

- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD before this task: `5a37883 feat(rvv): add i64 sub mul profile replay evidence`.
- Worktree was clean before creating this task.
- Existing RVV family descriptors live in `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`.
- Existing generic RVV binary helper APIs live in `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h` and already expose family-owned C types, runtime ABI names, route ids, and component groups for i32/i64 families.
- Existing support-layer contract in `include/TianChenRV/Support/RuntimeABIContract.h` is still i32-specific and has active compatibility functions that default through add.
- Existing emission/readiness/export paths already carry `runtime_abi_parameters`, but selected i64 coverage must prove those parameters are descriptor-owned and not inherited from i32-vadd defaults.

## Requirements

- Identify the current owner of RVV binary family descriptors and runtime ABI parameter construction before source changes.
- Add or refactor the smallest C++ support/target API so the selected `RVVBinaryFamilyDescriptor` owns or derives:
  - runtime ABI kind/name and runtime glue role;
  - external ABI component group;
  - ordered callable C parameter list;
  - runtime ABI parameter roles and ownership;
  - runtime element-count parameter spec.
- Ensure selected RVV emission plans, readiness diagnostics, execution-plan coherence, emission manifests, and target artifact export consume descriptor-owned ABI metadata when a concrete family is available.
- Preserve existing i32-vadd compatibility wrappers only as wrappers, not as active source of truth for non-vadd or i64 selected families.
- Add focused C++ tests proving at least `i64-vadd`, `i64-vsub`, and `i64-vmul` expose `const int64_t *`, `const int64_t *`, `int64_t *`, and `size_t` ABI parameters with correct roles/ownership and family-specific runtime ABI names/routes.
- Preserve coherent i32 add/sub/mul behavior.
- Add or update manifest/export tests so selected i64 runtime ABI parameters observed in diagnostics/manifests match the descriptor contract.

## Acceptance Criteria

- [x] There is a single descriptor-driven C++ API for RVV binary callable ABI shape and role requirements across i32/i64 add/sub/mul.
- [x] RVV plugin emission-plan construction for supported selected i32/i64 binary families uses the selected descriptor contract for runtime ABI parameters and identities.
- [x] RVV target artifact exporter registration and validation uses descriptor-derived required role/type/ownership lists and component identity for all supported RVV binary families.
- [x] C++ tests cover i64 add/sub/mul ABI parameter order, roles, C spellings, ownership, and family-specific runtime ABI names/routes.
- [x] Existing i32 add/sub/mul tests continue to pass.
- [x] A manifest or export-facing test observes i64 runtime ABI parameter metadata and rejects/avoids stale i32-vadd defaults.
- [x] `git diff --check` passes.
- [x] Focused build/tests for changed C++ and lit surfaces pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes.
- [x] The task is finished/archived and validated with `python3 ./.trellis/scripts/task.py validate <archived-task-path>`.
- [x] One coherent commit is created.

## Completion Notes

- Added `RVVBinaryRuntimeABIContract` in the RVV target descriptor layer and delegated existing RVV binary callable ABI helpers through it.
- Routed RVV plugin supported emission-plan runtime ABI identity through the selected RVV binary contract.
- Extended C++ tests for i64 add/sub/mul selected plans, emission plans, target exporter routes, and stale i32-vadd metadata rejection.
- Extended i64 RVV microkernel lit checks to assert descriptor-owned `runtime_abi_parameters` in emission-plan diagnostics.
- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` with the target-layer RVV binary runtime ABI contract.
- No new `ssh rvv` evidence was needed because generated runtime-callable source/header/object behavior and runtime correctness claims were not changed.

## Non-Goals

- Do not add a broad profile-replay smoke matrix.
- Do not add new RVV op families, dynamic VL variants, masks, tuning, or cost-model work.
- Do not implement compiler internals in Python.
- Do not move compute semantics into `tcrv.exec`.
- Do not hard-code RVV family semantics in core orchestration passes.
- Do not work on IME, AME, Sophgo/offload, scalar fallback dispatch, or unrelated provider29 artifacts.
- Do not claim new runtime correctness/performance without real `ssh rvv` evidence.

## Validation Plan

- `git diff --check`
- Configure/build with `artifacts/tmp/tianchenrv-build` if needed.
- Focused C++ tests:
  - RVV binary planning/plugin ABI metadata tests.
  - Target artifact export ABI role/type registration tests.
- Focused lit/FileCheck tests for i64 manifest/export ABI metadata if C++ tests do not fully cover the diagnostic/manifest surface.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- No new `ssh rvv` evidence is required if this round only changes compiler-owned metadata/tests and does not change generated RVV runtime-callable source/header/object behavior or make a new runtime claim.

## Technical Notes

- Relevant specs: `.trellis/spec/index.md`, `lowering-runtime/emission-runtime-contract.md`, `extension-plugins/rvv-plugin.md`, `plugin-protocol/interfaces-and-registry.md`, `variant-pipeline/generation-selection-tuning.md`, `testing/mlir-testing-contract.md`.
- Relevant implementation areas: support runtime ABI headers/sources, RVV binary family descriptors, RVV plugin emission plan construction, RVV microkernel target export, target artifact export registry, emission manifest serialization, execution-plan coherence preflight.
- Compatibility names such as `getI32VAddRuntimeABI*` may remain, but new active non-vadd/i64 paths must consume descriptor-driven RVV binary APIs.
