# Role-based RVV scalar dispatch ABI binding

## Goal

Make the bounded RVV+scalar i32-vadd dispatch C/header/caller export resolve
runtime ABI parameters by typed `RuntimeABIParameterRole` instead of incidental
array positions. The runtime element-count ABI input and explicit
dispatch-availability guard remain separate target/export-owned control inputs.

## Requirements

- Remove practical reliance on `dispatchParameters[3]`, `parameters[4]`, and
  `back()` in `lib/Target/Builtin/RVVScalarDispatch.cpp` when binding runtime
  element count or dispatch guard semantics.
- Add the smallest reusable C++ support-layer helper for finding exactly one
  `RuntimeABIParameter` by `RuntimeABIParameterRole` with clear diagnostics.
- Preserve generated callable/dispatcher parameter ordering unless the existing
  tests force a change.
- Fail closed on malformed runtime ABI plans: missing or duplicate role,
  empty/invalid C name, unsupported C type, and ownership mismatch where already
  modeled.
- Keep Python as runner/evidence tooling only.
- Avoid broad runtime ABI, variant selection, plugin registry, lowering, or
  documentation churn.

## Acceptance Criteria

- C++ export owner code binds runtime element count and dispatch guard by
  `RuntimeABIParameterRole::RuntimeElementCount` and
  `RuntimeABIParameterRole::DispatchAvailabilityGuard`.
- Focused C++ and/or lit coverage proves role lookup works when the parameter
  vector order would otherwise make positional lookup wrong, and that missing
  or duplicate roles fail with diagnostics.
- Existing RVV+scalar dispatch FileCheck expectations still prove explicit
  `n` and `rvv_available` ABI parameters stay separate.
- Required local checks pass:
  `git diff --check`, CMake configure under `artifacts/tmp/tianchenrv-build`,
  and `cmake --build ... --target check-tianchenrv -j2`.
- The Trellis task is archived before final commit if task state is created.

## Out Of Scope

- No Python compiler/runtime ABI semantics.
- No generic RVV backend, performance claim, object/link/runtime integration
  broadening, automatic hardware probing, or new RVV correctness claim unless
  real `ssh rvv` evidence is run.
- No changes to `tcrv.exec` compute scope or extension/plugin ownership
  boundaries.

## Technical Notes

- Required initial inspection confirmed repo root
  `/home/kingdom/phdworks/TianchenRV`, clean worktree, and HEAD
  `d10dd8e feat: exercise runtime element count in rvv dispatch ABI`.
- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Existing dispatch exporter already validates IR-backed runtime_param roles
  through support-layer collection, but final C emission still performs
  positional semantic binding in `RVVScalarDispatch.cpp`.
