# RVV LMUL m2 selected-config artifact closure

## Goal

Prove the RVV selected config profile is a real compiler boundary for one
non-default LMUL instance by routing i32m2 vadd through the bounded RVV binary
source-frontdoor/selected-boundary/materialization/artifact/runtime path. The
round must keep hardware/profile facts, compile-time vector config, runtime
AVL/VL roles, and descriptor-local element-count metadata separate.

## Background

The previous selected-config profile task converged at commit `28923a1`. It
made `RVVBinarySelectedConfigContract` and selected profile metadata consumed
by selected emission planning, RVVMicrokernel artifact generation,
TargetArtifactExport validation, RuntimeABICallablePlan, and
RVVScalarDispatch for existing i32 vadd/vsub/vmul behavior. The next bounded
gap is proving that this profile is not only metadata over the default m1
shape by carrying exactly one non-default LMUL axis, i32m2 vadd, through the
same production consumers.

## Scope

- RVV capability/profile facts needed to expose finite i32m2 config facts while
  preserving i32m1 as the default shape when no selector is present.
- The existing bounded RVV binary family path for exactly i32 vadd with
  `selected_vector_shape = i32m2`.
- Source-frontdoor or selected RVV boundary inputs that route through RVV
  plugin proposal/selection, selected lowering-boundary materialization,
  selected emission planning, RVVMicrokernel/TargetArtifactExport,
  RuntimeABICallablePlan, and RVVScalarDispatch.
- Focused C++/lit/script/local-generated-artifact evidence and focused
  `ssh rvv` compile/run evidence if the generated intrinsic route remains
  complete.

## Required Behavior

1. i32m2 selected config facts are represented as capability/profile facts:
   `rvv.i32_m2.sew32`, `rvv.i32_m2.lmul_m2`,
   `rvv.i32_m2.tail_policy.agnostic`, and
   `rvv.i32_m2.mask_policy.agnostic`. These are compile-time finite config
   facts, not runtime AVL/VL values and not descriptor element-count authority.

2. The bounded selected-shape selector remains explicit. If no
   `rvv.i32_binary.selected_vector_shape` capability is present, default i32
   proposal behavior continues to choose the existing valid i32m1 shape before
   trying i32m2. If the selector requests `i32m2`, the plugin validates only the
   i32m2 config set and fails closed when it is incomplete or inconsistent.

3. i32m2 vadd reaches selected boundary, materialized RVV microkernel body,
   selected emission plan metadata, generated source/header/object or bundle
   artifact export, RuntimeABICallablePlan, and RVVScalarDispatch using the
   same selected config profile consumers as the existing m1/default families.

4. Generated i32m2 vadd artifacts must show LMUL=m2, SEW=32, agnostic tail and
   mask policy, `vint32m2_t`, `i32m2`, `e32m2`,
   `__riscv_vsetvl_e32m2`, `__riscv_vle32_v_i32m2`,
   `__riscv_vadd_vv_i32m2`, `__riscv_vse32_v_i32m2`, the ABI signature, runtime
   `n`/AVL/VL role metadata, and selected source identity from the selected
   config profile.

5. Descriptor-local `tcrv_rvv.element_count` may remain bounded artifact
   capacity metadata only. It must not become selected config, runtime
   trip-count, source identity, or computation authority.

6. Existing i32 vadd/vsub/vmul m1/default behavior remains green, including the
   previous selected-config profile regressions.

## Acceptance Criteria

- Production C++ changes in the minimal profile/plugin/target/runtime owners
  needed to make i32m2 available as a selected finite config fact and keep
  i32m2 vadd routed through existing selected-config consumers.
- Focused lit/FileCheck evidence proves i32m2 vadd reaches selected boundary,
  selected profile metadata, materialized RVV body, artifact export, generated
  intrinsic names, and dispatch/runtime validation.
- RuntimeABICallablePlan and RVVScalarDispatch reuse the selected config
  profile rather than reconstructing shape/ABI/runtime roles from descriptor
  state.
- Fail-closed checks cover missing/conflicting i32m2 config facts, mismatched
  LMUL/shape metadata, missing runtime length role data, stale ABI/profile
  metadata, source-identity mismatch, descriptor-only production attempts, and
  explicit-only route misuse where this round touches the path.
- Focused build covers touched support/plugin/target/export tools and tests,
  including generated headers, RVV plugin/target libraries,
  `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, `tcrv-translate`,
  RuntimeABI callable-plan test, target artifact export test, and affected RVV
  target tests.
- Local generated-artifact/runtime evidence for i32m2 vadd is recorded.
- Focused `ssh rvv` clang compile/run evidence for generated i32m2 vadd is
  recorded if reachable; otherwise the exact blocker is recorded and no RVV
  runtime claim is made.
- `git diff --check`, `git diff --cached --check`, Trellis validation before
  finish, Trellis validation after archive, and final clean worktree.

## Non-Goals

- No broad dtype, SEW, LMUL, tail, mask, or op-family matrix.
- No second new op family beyond i32m2 vadd.
- No performance tuning or performance claim.
- No linalg/tensor frontend expansion beyond the existing bounded RVV binary
  frontdoor.
- No Python compiler semantics and no descriptor-to-C production exporter.
- No descriptor element-count or vector shape as selected config/runtime
  authority.
- No computation semantics in `tcrv.exec`.
- No RVV semantic special case in generic core orchestration.
- No replacing clang/LLVM as the default native route.
- No runtime/correctness/performance claim without focused local generated
  artifact or `ssh rvv` evidence.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`, and
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-14-rvv-selected-config-profile-contract/prd.md`.
- Current code already has generic selected-shape support for i32m2 and direct
  i32m2 typed-body planning checks. This round must close the vadd source/
  artifact/runtime evidence gap and keep the capability profile/replay surface
  honest for finite i32m2 facts.
