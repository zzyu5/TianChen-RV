# Evidence Summary

## Result

Completed the bounded Stage 1 typed RVV MAcc route-family derivation task.
Plain MAcc, scalar-broadcast MAcc, and computed-mask MAcc route-family plans now
carry selected typed-config snapshots and validate MAcc-related leaves as typed
fact-derived outputs before provider materialization.

## Behavior Completed

- Added typed config snapshot fields to plain MAcc, scalar-broadcast MAcc, and
  computed-mask accumulation family plans.
- Replaced owner-local exact `i32m1`/`i32m2` MAcc helper authority with typed
  SEW/LMUL/predicate derivation for MAcc, setvl, scalar splat, compare, merge,
  store, vector type, and mask type leaves.
- Made computed-mask MAcc description mirrors consume `plan.maccIntrinsic` and
  `plan.maskedMergeIntrinsic`, and made materialization facts consume those
  owner-derived leaves.
- Synchronized target artifact validators, generated-bundle script mirrors, and
  lit expectations from old `e32m1` MAcc profile/type strings to typed mirror
  profile/type strings.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the executable
  MAcc typed-config snapshot and leaf-derivation contract.

## Checks

- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- Filtered lit:
  `cd artifacts/tmp/tianchenrv-build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='explicit-selected-body-artifact-macc-add|pre-realized-selected-body-artifact-macc-add|explicit-selected-body-artifact-scalar-broadcast-macc-add|pre-realized-selected-body-artifact-scalar-broadcast-macc-add|rvv-generated-bundle-abi-e2e-pre-realized-macc-add-dry-run|rvv-generated-bundle-abi-e2e-macc-add-dry-run' .`
- `git diff --check`

## Scan Classification

Bounded scan command:

```bash
rg -n "__riscv_.*_i32m[12]|vint32m[12]_t|RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m" \
  include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h \
  lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp \
  lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp \
  lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp \
  lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp \
  lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp \
  test/Plugin/RVVExtensionPluginTest.cpp \
  test/Target/RVV/explicit-selected-body-artifact-macc-add.mlir \
  test/Target/RVV/pre-realized-selected-body-artifact-macc-add.mlir \
  test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir \
  test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-macc-add.mlir \
  test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-macc-add-dry-run.test \
  test/Scripts/rvv-generated-bundle-abi-e2e-macc-add-dry-run.test \
  scripts/rvv_generated_bundle_abi_e2e.py \
  .trellis/spec/extension-plugins/rvv-plugin.md \
  .trellis/spec/testing/mlir-testing-contract.md
```

Classification:

- `RVVEmitCMAccRouteFamilyPlanOwners.cpp` and `RVVEmitCRoutePlanning.h`: no
  complete legacy authority hits. The owner retains only compositional
  derivation stems such as `__riscv_vmacc_vv_` plus typed suffix.
- `RVVEmitCRoutePlanning.cpp`: remaining hits are unrelated route families,
  central typed-profile helpers, fail-closed legacy i32 detection, or derived
  output helpers outside MAcc owner-local authority.
- `RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` and
  `RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`: no MAcc authority hits.
- `RVVTargetArtifactRouteFamilyValidation.cpp` and
  `scripts/rvv_generated_bundle_abi_e2e.py`: remaining hits are derived output
  validation for target/export families, widening/contraction/segment/memory
  evidence, or unrelated remaining Stage 1 debt. Plain/scalar MAcc profile and
  type-mapping mirrors were changed to typed strings.
- `test/Plugin/RVVExtensionPluginTest.cpp` and changed lit/script tests:
  remaining hits are expected derived output, negative stale-field checks, or
  parseable-only/deprecated legacy i32 fail-closed tests.
- `.trellis/spec/`: remaining hits are Stage 1 rule text, negative examples,
  and testing-contract guidance.

## Runtime Evidence

No runtime correctness or performance claim was made. No `ssh rvv` evidence was
required for this compiler-planning and lit/FileCheck task.

