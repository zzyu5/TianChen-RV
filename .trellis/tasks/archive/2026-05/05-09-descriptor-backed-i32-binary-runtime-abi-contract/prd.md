# Descriptor-backed i32 binary runtime ABI contract

## Goal

Replace the stale vadd-owned support runtime ABI boundary with a family-aware
i32 binary ABI contract and callable plan that is consumed by the existing
add/sub/mul RVV, scalar, offload descriptor, RVV+scalar dispatch, emission
readiness, manifest, and artifact export paths.

## What I already know

- The previous Trellis task is complete and archived at commit `d616b43`.
- This task starts from a clean `main` worktree with no previous active task.
- `I32BinaryFamilyRegistry.h` already owns add/sub/mul family descriptors,
  including RVV, scalar, and dispatch runtime ABI strings.
- `Support` still exposes vadd-shaped APIs such as
  `I32VAddRuntimeABIContract`, `I32VAddCallableABIPlan`,
  `buildI32VAddCallableABIPlan`, and vadd-named role binding helpers.
- RVV, scalar, offload, RVV+scalar dispatch, and artifact export code all
  reuse the vadd-shaped support API, including some sub/mul paths.
- The common callable parameter shape must remain `lhs`, `rhs`, `out`, `n`,
  with dispatch adding one availability guard parameter where applicable.

## Requirements

- Introduce a family-aware support API, centered on an
  `I32BinaryRuntimeABIContract` and `I32BinaryCallableABIPlan`, driven by
  `I32BinaryFamilyDescriptor` or its RVV/scalar/dispatch subdescriptors.
- Derive RVV, scalar, and dispatch runtime ABI identities from the selected
  add/sub/mul family descriptor instead of vadd-only support defaults.
- Preserve the existing callable ABI role ordering, C names, C types, and
  ownership for lhs input buffer, rhs input buffer, output buffer, and runtime
  element count.
- Preserve dispatch availability guard semantics and append the guard only to
  dispatch ABI plans.
- Migrate active add/sub/mul RVV, scalar, offload, RVV+scalar dispatch,
  emission-plan, manifest, and artifact-export consumers touched by this
  module to the family-aware API.
- If temporary vadd compatibility wrappers remain, they must be explicitly
  named as temporary wrappers around the family-aware i32 binary API, and no
  active add/sub/mul path touched in this task may depend on them for ABI
  ownership.
- Update diagnostics so family-generic ABI failures say `i32 binary` or name
  the concrete family key, not `i32-vadd`, when the failing selected path is
  vsub or vmul.
- Repair tests that mask stale ABI ownership through sed-based vadd
  substitutions when they should exercise family-correct descriptor-backed
  behavior directly.
- Update only durable spec text needed to record descriptor-backed i32 binary
  ABI ownership.

## Acceptance Criteria

- [x] C++ support APIs expose family-aware i32 binary runtime ABI contract,
      callable plan, role requirements, parameter binding, and dispatch guard
      helpers.
- [x] Positive C++ coverage proves add, sub, and mul expose distinct
      descriptor-derived RVV/scalar/dispatch ABI identities with the common
      parameter shape.
- [x] Negative C++ coverage rejects stale mismatched family/runtime ABI names.
- [x] At least one vsub and one vmul lit/FileCheck path carries
      family-correct runtime ABI metadata from planning/readiness to
      artifact/export without vadd-only substitution masking.
- [x] Active add/sub/mul RVV, scalar, offload, RVV+scalar dispatch, manifest,
      and artifact export paths no longer call the vadd-named ABI helpers for
      ABI ownership.
- [x] `git diff --check` passes.
- [x] CMake configure passes with LLVM/MLIR 20 paths.
- [x] `check-tianchenrv` passes, or any failure is documented with the exact
      next continuation point.

## Non-goals

- Do not add new arithmetic families, dtypes, vector widths, SEW/LMUL modes,
  RVV intrinsics, or generated arithmetic semantics.
- Do not change runtime parameter layering, dispatch guard semantics, or
  self-check meaning except to correct stale ABI identity.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission in Python.
- Do not add RVV, IME, Sophgo, AME, or vendor-specific branches to core
  support code.
- Do not claim RVV runtime correctness or performance without fresh `ssh rvv`
  evidence. This task is an ABI metadata/contract refactor unless generated C
  behavior or runtime signatures change.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/extension-plugins/offload-runtime-plugin.md`,
  `.trellis/spec/capability-model/capability-contract.md`.
- Primary source owners inspected:
  `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`,
  `include/TianChenRV/Support/RuntimeABI*.h`,
  `lib/Support/RuntimeABI*.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Scalar/ScalarMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`,
  `lib/Plugin/Offload/OffloadExtensionPlugin.cpp`,
  and focused target/export/lowering tests.
