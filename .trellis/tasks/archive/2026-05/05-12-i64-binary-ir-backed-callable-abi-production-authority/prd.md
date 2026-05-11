# Make i64 binary callable ABI parameters production-owned by exec IR

## Goal

Make supported finite i64 add/sub/mul callable ABI parameters for RVV direct
and scalar fallback production/export paths derive from direct
`tcrv.exec.mem_window` and `tcrv.exec.runtime_param` ABI boundaries. Runtime ABI
metadata may remain as a mirror, and finite family metadata may still identify
route, dtype, ABI identity, and C types, but descriptor/default parameter lists
must not act as production callable ABI authority when exec IR is missing,
duplicate, malformed, or stale.

## What I Already Know

- Repository HEAD at task start is `99933fb feat(abi): derive i32 callable params from exec IR`.
- `.trellis/.current-task` was absent, so this task was created as
  `05-12-i64-binary-ir-backed-callable-abi-production-authority`.
- The archived i32 PRD confirms the previous round completed i32 RVV/scalar ABI
  authority through `support::buildI32BinaryCallableABIPlan`.
- `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp` still has a non-i32
  branch that locally assembles ABI parameters from collected windows and
  runtime params while hard-coding callable C names such as `lhs`, `rhs`, and
  `out`.
- `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` still has a non-i32 branch that
  starts from descriptor/default scalar ABI parameters, consumes collection
  errors, and returns fallback parameter metadata.
- `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Scalar/ScalarMicrokernel.cpp`, and
  `lib/Target/Builtin/RVVScalarDispatch.cpp` still contain route-local finite
  binary ABI plan or mirror logic that should reuse the shared exec-IR-backed
  plan for i64 candidate/source/header/object preflight.
- `tcrv.exec` must remain compute-free; compute source stays in typed RVV/scalar
  family ops and the common EmitC route.

## Requirements

- Add or generalize a shared support-layer finite binary callable ABI plan that
  covers the existing bounded RVV binary i32/i64 add/sub/mul family descriptors.
- The shared plan must inspect the materialized `tcrv.exec.kernel` body, collect
  exactly the required buffer `mem_window` roles plus one runtime-element-count
  `runtime_param`, and fail closed on missing, duplicate, malformed, stale, or
  empty C-name/type/ownership ABI metadata.
- Preserve existing i32 wrappers and behavior while adding i64 coverage.
- Rewire RVV i64 direct emission planning and RVV source/header/object preflight
  to consume the shared plan rather than local descriptor/default ABI assembly.
- Rewire scalar i64 fallback emission planning and scalar source/header/object
  preflight to consume the same shared plan and mirror validation.
- Keep `runtime_abi_parameters` emission-plan metadata as a mirror only. It may
  be emitted and checked, but generated callable signatures must come from the
  exec-IR-backed plan.
- Preserve typed family op and common EmitC route authority for production
  bodies.
- Quarantine or remove obsolete descriptor/default ABI construction from
  production i64 paths.

## Non-Goals

- Do not redo the completed i32 task except to preserve wrappers and reuse the
  new finite binary support API.
- Do not add descriptor-to-C, descriptor-to-ABI, or descriptor-to-compute
  authority.
- Do not add new dtype families or broaden generic RVV lowering.
- Do not add ssh evidence, runtime correctness claims, throughput claims,
  latency claims, dashboards, or evidence packaging.
- Do not implement compiler internals in Python.
- Do not add compute semantics to `tcrv.exec` or family-specific branches in
  shared core passes.

## Acceptance Criteria

- [x] Support C++ tests cover valid i64 add/sub/mul callable plans and
  fail-closed missing, duplicate, malformed, stale, or empty-C-name
  `mem_window` / `runtime_param` cases.
- [x] Existing i32 ABI tests continue passing through compatibility wrappers.
- [x] RVV i64 direct route emits source/header callable ABI parameters from the
  shared exec-IR-backed plan and fails closed when required ABI boundary
  metadata is missing or stale.
- [x] Scalar i64 fallback emission planning/export no longer falls back to
  descriptor/default ABI parameters when exec IR ABI boundaries are missing or
  malformed.
- [x] Runtime ABI metadata mirror validation uses the shared support-layer
  finite binary plan for RVV/scalar i64 candidates.
- [x] `git diff --check` passes.
- [x] The focused support test binary and focused lit tests for the changed
  RVV/scalar export paths pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  runs if the build directory is usable, or the report states the exact blocker.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant archived task read:
  - `.trellis/tasks/archive/2026-05/05-12-i32-binary-ir-backed-callable-abi-production-authority/prd.md`
- Main source entry points:
  - `include/TianChenRV/Support/RuntimeABICallablePlan.h`
  - `lib/Support/RuntimeABICallablePlan.cpp`
  - `include/TianChenRV/Support/RuntimeABIContract.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  - `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `test/Support/RuntimeABICallablePlanTest.cpp`
  - `test/Target/RVVMicrokernel/`
  - `test/Target/ArtifactExport/scalar-target-source-artifact-routes.test`

## Task Status Notes

- Shared support-layer finite binary callable ABI plan now serves RVV binary
  i32/i64 add/sub/mul family descriptors while preserving existing i32
  compatibility wrappers.
- RVV direct i64 and scalar fallback i64 production/export preflight consume the
  shared exec-IR-backed plan instead of descriptor/default ABI parameter
  construction.
- RVV+scalar dispatch component preflight also consumes the same shared plan so
  finite binary component signatures stay aligned with direct callable route
  contracts.
- `runtime_abi_parameters` remains mirror metadata. The i64 tests now prove
  non-default runtime `c_name = "len64"` flows from `tcrv.exec.runtime_param`
  into source/header/export surfaces.
- `tcrv.exec` stayed compute-free; compute body authority stayed in typed
  RVV/scalar family ops through the common EmitC route.
- No RVV runtime, correctness, throughput, latency, or performance claim was
  made.
- Focused checks, `git diff --check`, and full `check-tianchenrv` passed.
