# rvv-selected-config-vl-artifact-contract

## Goal

Make the selected RVV compile-time config and runtime AVL/VL role boundary explicit, validated, and consumed by the plugin-owned target artifact/export path for one finite proven route: `i64-vmul` / `i64m1`. The round should turn the current selected metadata from route-adjacent convention into a fail-closed compiler contract at the artifact/export consumer boundary.

## Owner

RVV selected config and runtime AVL/VL contract consumed by the RVV or RVV+scalar target artifact exporters. Generic core/export code may enforce only generic presence and consistency contracts; RVV SEW/LMUL/vector-shape/family/runtime-role semantics must stay RVV plugin-local or target-owned.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- The starting branch is `main`, with a clean worktree at `ee0ec3a chore(trellis): record i64 vmul dispatch ssh evidence`.
- No `.trellis/.current-task` existed at session start, so this task was created from the Hermes brief before source changes.
- The previous archived task proved the existing `i64-vmul` RVV+scalar dispatch bundle through real `ssh rvv` evidence, but it did not require compiler source changes.
- This round must avoid another evidence-only/family-proof pass and instead strengthen active C++ compiler structure.
- Relevant specs require C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck implementation, plugin locality, target-neutral generic routing, and fail-closed artifact/export preflight behavior.

## Boundaries

- Compiler core, dialect, pass, plugin registry, lowering, emission, artifact export, and validation logic must remain in C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck.
- Python may only orchestrate checks/evidence and parse bounded artifacts; it must not own compiler contracts, selected config validation, route selection, descriptor semantics, or artifact export behavior.
- `tcrv.exec` remains compute-free. Do not add RVV compute semantics, tensor/tile semantics, or new high-level compute ops to the core dialect.
- Do not build or claim a generic RVV backend.
- Do not add a family matrix. The validation vehicle is exactly one existing finite route, preferably `i64-vmul` / `i64m1`.
- Do not add target-family branches to generic core passes or generic target artifact code.
- Do not change offload/scalar/toy behavior except for mechanical build compatibility.
- Do not make throughput, latency, performance, ratio, or broad runtime correctness claims.

## Required Contract

The selected artifact/export path must distinguish and validate these layers:

1. Hardware capability facts: examples include VLEN, vlenb, ELEN, hart count, selected march/mabi, toolchain/probe facts, and capability-provider provenance.
2. Compile-time selected RVV config: selected family/dtype, vector shape, SEW, LMUL, tail policy, mask policy, selected vector type/suffix, and selected setvl suffix.
3. Runtime SSA/control inputs: pointer arguments, runtime element count/AVL role, dispatch availability guard, and any generated VL/control role that is represented as actual IR/control/ABI data.
4. Descriptor-local artifact metadata: finite descriptor ids, bounded fixture element counts, external ABI names, component roles/groups, route ids, file names, and bundle index records.

## Requirements

1. Inspect the current selected-config/VL surfaces before implementing and do not duplicate existing abstractions.
2. Add or tighten C++ data structures and validation at the existing owner boundary so selected RVV metadata is carried into artifact/export candidate validation.
3. Valid `i64-vmul` / `i64m1` selected metadata must continue to export the chosen RVV or RVV+scalar source/header/object bundle.
4. Missing selected config metadata must fail before source/header/object/bundle output with focused diagnostics.
5. Stale or mismatched selected config metadata must fail before export, including mismatched SEW, LMUL, vector shape, family/dtype, selected vector spelling, and runtime AVL/element-count role.
6. Dispatch export must continue to validate the target/export-owned dispatch availability guard separately from callable runtime element-count/AVL input.
7. Generic target artifact routing may only enforce target-neutral selected-path, route, artifact-kind, runtime ABI presence, and consistency contracts.
8. RVV-specific selected config/VL semantics must be implemented plugin-locally or in RVV target/export code.
9. Focused tests must cover positive export and fail-closed negative cases for the selected config/VL contract.

## Acceptance Criteria

- [ ] Trellis task exists, is current, and records this PRD plus relevant spec context.
- [ ] Existing selected-config/VL surfaces are inspected and the implementation reuses or tightens them rather than duplicating route-name inference.
- [ ] C++ validation distinguishes hardware capability facts, compile-time RVV selected config, runtime SSA/control ABI roles, and descriptor-local artifact metadata.
- [ ] Valid `i64-vmul` / `i64m1` selected metadata exports through the RVV or RVV+scalar artifact bundle path.
- [ ] Missing selected config metadata fails closed before artifact output with a focused diagnostic.
- [ ] Stale/mismatched SEW, LMUL, vector shape, family/dtype, selected vector spelling, or runtime AVL/element-count role fails closed before artifact output with focused diagnostics.
- [ ] Generic core/export code remains target-neutral and does not branch on RVV family, dtype, shape, runtime, toolchain, or microarchitecture semantics.
- [ ] Offload/scalar/toy behavior is unchanged except for mechanical build compatibility if required.
- [ ] Focused lit/C++ tests cover valid export and the new fail-closed cases.
- [ ] `git diff --check` passes.
- [ ] Touched tools/tests build, including at minimum `tcrv-opt`, `tcrv-translate`, RVV plugin tests, scalar plugin tests if dispatch is touched, and target artifact export tests.
- [ ] Focused lit tests for the selected RVV/RVV+scalar artifact path and new fail-closed cases pass.
- [ ] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes before finish/archive.
- [ ] Trellis task validation passes before finish/archive.
- [ ] If source/header/object emission changes or the final report makes a runtime correctness claim, bounded single-route `ssh rvv` evidence is rerun and recorded without secrets.

## Minimal Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Scalar plugin test build/run if RVV+scalar dispatch validation touches scalar candidate behavior.
- Focused lit tests under `test/Plugin/`, `test/Target/RVVScalarDispatch/`, `test/Target/TargetArtifactBundleExport/`, and/or `test/Transforms/LinalgToExec/` for the changed contract.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-05-11-rvv-selected-config-vl-artifact-contract`

## Out of Scope

- Generic RVV backend claims.
- Broad add/sub/mul or i32/i64 family matrix expansion.
- Python-owned compiler semantics.
- Compute semantics in `tcrv.exec`.
- Prompt/report/journal-only completion.
- Performance or throughput evidence.
- RVV runtime correctness claims unless real `ssh rvv` evidence is explicitly rerun for the selected route.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Archived task read:
  - `.trellis/tasks/archive/2026-05/05-11-i64m1-rvv-scalar-dispatch-artifact-path-ssh-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-11-i64m1-rvv-scalar-dispatch-artifact-path-ssh-evidence/task.json`
- Starting evidence from the archived task proves the route exists; this task must strengthen the selected config/VL artifact contract rather than repackage that evidence as progress.

## Continuation Rule If Unfinished

Keep this task open and record the exact failed boundary: planning metadata, lowering boundary, emission readiness, emission plan, target candidate collection, exporter preflight, source/header/object emission, lit validation, aggregate check, or ssh evidence. Do not archive an incomplete task.
