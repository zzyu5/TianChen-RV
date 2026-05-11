# Journal - codex (Part 3)

> Continuation from `journal-2.md` (archived at ~2000 lines)
> Started: 2026-05-11

---



## Session 21: RVV selected config VL dataflow materialization

**Date**: 2026-05-11
**Task**: RVV selected config VL dataflow materialization
**Branch**: `main`

### Summary

Added a selected-config-contract-facing RVV VL dataflow materialization helper and routed finite RVV microkernel body construction through it, with i32m2/i64m1 positive and stale/missing contract negative C++ coverage.

### Main Changes

### Main Changes

- Added `RVVBinaryVLDataflowMaterialization` as the explicit bridge from `RVVBinarySelectedConfigContract` to bounded `tcrv_rvv` setvl/with_vl/load-op-store materialization facts.
- Routed `materializeRVVBinaryMicrokernelOp` through the new helper so descriptor shape/type/op data must match the selected config contract before IR is created.
- Added C++ coverage for i32m2 `i32-vsub` and i64m1 `i64-vmul` using the same selected-config-to-VL dataflow path, plus stale descriptor and missing contract negatives.

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: 200/200 lit tests passed.

No `ssh rvv` evidence was collected and no runtime correctness or performance claim was made.

### Status

[OK] **Completed**


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete

---

## Session 22: RVV i32m2 selected VL dispatch ssh evidence

**Date**: 2026-05-11
**Task**: RVV i32m2 selected VL dispatch ssh evidence
**Branch**: `main`

### Summary

Proved the non-m1 `i32-vsub` / `i32m2` selected-config path through
plan-and-export target artifact bundle generation, RVV+scalar dispatch wrapper
emission, and real `ssh rvv` compile/link/run correctness.

### Main Changes

- Created Trellis task
  `rvv-i32m2-selected-vl-dispatch-ssh-evidence` and wrote the PRD.
- Strengthened `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test` for the
  i32m2 vsub bundle path: selected binary config, runtime `n`, dispatch guard
  `rvv_available`, descriptor-local i32-vsub metadata, bundle index ABI,
  scalar subtract source, and branch evidence are now FileChecked.
- Confirmed the C++/MLIR/plugin/target path already carries
  `frontend_bundle_i32m2_vsub` through selected-config/VL dataflow into the
  generated dispatch source and bundle object.

### Evidence

- Real RVV evidence:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-20260511-i32m2-vsub-bundle-ssh/evidence.json`
- `ssh rvv` host facts: `riscv64`, `/usr/bin/clang`, Ubuntu clang 18.1.3.
- Generated RVV source uses `__riscv_vsetvl_e32m2`,
  `__riscv_vle32_v_i32m2`, `__riscv_vsub_vv_i32m2`, and
  `__riscv_vse32_v_i32m2`.
- Runtime checks covered `rvv_available=0` and `rvv_available=1` with counts
  `7` and `16`, and observed
  `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16
  branches=scalar_and_rvv`.

### Testing

- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i32-vsub --vector-shape=i32m2 --lower-linalg-frontend --run-id codex-20260511-i32m2-vsub-dispatch-dry --overwrite --expect-selected-kernel frontend_dispatch_i32m2_vsub`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-20260511-i32m2-vsub-bundle-dry --overwrite --expect-selected-kernel frontend_bundle_i32m2_vsub`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-20260511-i32m2-vsub-bundle-ssh --overwrite --expect-selected-kernel frontend_bundle_i32m2_vsub --timeout 180 --connect-timeout 20`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Focused lit:
  `rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir`,
  `linalg-i32-vsub-to-rvv-artifact.mlir`,
  `rvv-scalar-dispatch-e2e.test`,
  `rvv-scalar-dispatch-bundle-e2e.test`,
  `i32m2-dataflow.mlir`, and `rvv-microkernel-e2e.test`.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  200/200 lit tests passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete.

---

## Session 24: Extension plugin artifact route registration template

**Date**: 2026-05-11
**Task**: Extension plugin artifact route registration template
**Branch**: `main`

### Summary

Completed the bounded plugin-owned artifact route registration template by
lifting route metadata into the C++ target artifact exporter registration
contract and making generic target artifact preflight consume it.

### Main Changes

- Created and archived Trellis task
  `extension-plugin-artifact-route-registration-template` with a PRD aligned to
  plugin-protocol, lowering-runtime, offload-plugin, and testing specs.
- Added `TargetArtifactRouteMetadata` for route-level runtime ABI metadata,
  selected-plan metadata requirements, and explicit conservative claim fields.
- Extended generic target artifact preflight to reject stale registered runtime
  ABI metadata and stale selected-plan/handoff metadata before route-local
  export.
- Registered descriptor metadata for both Toy metadata and offload runtime
  descriptor plugin-owned target exporter bundles.
- Emitted offload route claim fields in the generic target artifact bundle
  index while preserving deterministic descriptor output and no-claim wording.

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='(offload-runtime-descriptor-artifact-route|offload-runtime-descriptor-bundle|toy-metadata-artifact-route|toy-metadata-artifact-runtime-abi-kind-fails|target-artifact-export-registry)'`
  from `artifacts/tmp/tianchenrv-build/test`: 5 focused lit tests passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-toy-extension-plugin-test tianchenrv-offload-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-toy-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-offload-extension-plugin-test`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-extension-plugin-artifact-route-registration-template`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  200/200 lit tests passed.

No runtime correctness, hardware execution, or performance evidence was
claimed.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Status

[OK] **Completed**

### Next Steps

- None - task complete.

---

## Session 23: Offload runtime plugin boundary template

**Date**: 2026-05-11
**Task**: Offload runtime plugin boundary template
**Branch**: `main`

### Summary

Completed the bounded offload runtime plugin boundary template by tightening
the descriptor handoff claim boundary and adding fail-closed coverage for
offload runtime capability misclassification as a custom ISA.

### Main Changes

- Created and archived Trellis task
  `offload-runtime-plugin-boundary-template` with a PRD aligned to the offload
  runtime plugin spec.
- Added explicit descriptor output fields for non-executable handoff status,
  no local runtime execution claim, no local runtime correctness claim, no
  hardware execution claim, and no performance claim.
- Extended the offload plugin C++ test so `offload.runtime` declared as
  `kind = "custom-isa"` records a recoverable proposal decline and fails
  plugin legality as runtime-offload misclassification.
- Added lit coverage for the same custom-ISA misclassification through the
  public emission-plan materialization route.
- Updated descriptor artifact FileCheck coverage for the new explicit no-claim
  fields.

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-offload-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-offload-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Plugin/offload-extension-plugin.test Transforms/EmissionReadiness/offload-fail-closed.mlir Target/ArtifactExport/offload-runtime-descriptor-artifact-route.test Target/EmissionManifest/emission-manifest-offload-pipeline.mlir`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-emission-readiness-test tianchenrv-variant-materialization-test tianchenrv-variant-selection-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emission-readiness-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-variant-materialization-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-variant-selection-test`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  200/200 lit tests passed.

No offload runtime, hardware correctness, or performance evidence was claimed.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Status

[OK] **Completed**

### Next Steps

- None - task complete.
