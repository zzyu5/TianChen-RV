# Journal - codex (Part 6)

> Continuation from `journal-5.md` (archived at ~2000 lines)
> Started: 2026-05-14

---



## Session 62: RVV artifact export source-frontdoor contract

**Date**: 2026-05-14
**Task**: RVV artifact export source-frontdoor contract
**Branch**: `main`

### Summary

Moved selected RVV source identity into RVVMicrokernel artifact/export validation and generated source/header comments; verified vadd/vsub bundle route locally and vsub on ssh rvv.

### Main Changes

- Added RVVMicrokernel source/header artifact comments exposing the selected source identity in the same compact contract form consumed by dispatch validation.
- Required finite typed RVV target artifact routes to carry `tcrv_rvv.selected_binary_source_kind` and the expected selected microkernel op in selected-plan metadata before source/header/object export succeeds.
- Extended RVVMicrokernel selected-plan source identity validation beyond `frontend-lowering`, so direct typed vsub/vadd source identity is checked through the same C++ contract.
- Updated focused C++ and lit coverage for vadd/vsub source-frontdoor bundle output, vsub source/header bundle output, and target artifact export fixture metadata.

### Git Commits

Final session commit recorded in git history for this round.

### Testing

- [OK] Built `tcrv-opt`, `tcrv-translate`, RVV target/plugin libraries, `tianchenrv-rvv-extension-plugin-test`, and `tianchenrv-target-artifact-export-test`.
- [OK] Ran `tianchenrv-rvv-extension-plugin-test` and `tianchenrv-target-artifact-export-test`.
- [OK] Ran focused lit for `RVVMicrokernel`, `TargetArtifactBundleExport`, and `RVVScalarDispatch` surfaces.
- [OK] Ran `rvv_microkernel_e2e.py --self-test` and `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] Ran local vadd/vsub source-frontdoor bundle dry-runs and one `ssh rvv` vsub generated artifact invocation.
- [OK] Ran `git diff --check` and Trellis task validation.

### Status

[OK] **Completed**

### Next Steps

- None - task complete

---

## Session 63: RVV exported artifact runtime consumption

**Date**: 2026-05-14
**Task**: RVV exported artifact runtime consumption
**Branch**: `main`

### Summary

Closed the downstream consumption gap for source-frontdoor RVV generated
artifacts. `RVVScalarDispatch` now validates the embedded RVVMicrokernel
artifact's exported selected-source identity and runtime ABI invocation
contract before dispatch source/header/object export can succeed.

### Main Changes

- Added production C++ validation in `RVVScalarDispatch.cpp` for embedded RVV
  callable source contracts:
  `rvv_microkernel_selected_source_identity`,
  `runtime_abi_invocation_contract`, and runtime length authority.
- Added `dispatch_embedded_rvv_artifact_contract_consumed` comments to
  generated dispatch source/header artifacts.
- Updated `rvv_scalar_dispatch_e2e.py` to parse and record
  `embedded_rvv_artifact_contract` from generated dispatch artifacts as a
  runner-only consumer.
- Tightened vector dynamic vadd/vsub bundle FileCheck coverage around embedded
  RVVMicrokernel selected-source and runtime ABI contracts.

### Testing

- [OK] Focused C++/tool build for RVV target/export/plugin and `tcrv-opt` /
  `tcrv-translate`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit for dynamic vector vadd/vsub bundle artifacts.
- [OK] Focused lit for `Target/RVVScalarDispatch`, `Target/RVVMicrokernel`, and
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test` (`46/46`).
- [OK] `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] `rvv_microkernel_e2e.py --self-test`.
- [OK] Local generated bundle dry-runs for vector i32-vadd and i32-vsub.
- [OK] `ssh rvv` generated bundle invocation for vector i32-vsub:
  `codex-exported-contract-vsub-ssh`, `ssh_evidence_verified=true`.
- [OK] `git diff --check`, `git diff --cached --check`, Trellis validation.

### Status

[OK] **Completed**

### Next Steps

- None - task complete
