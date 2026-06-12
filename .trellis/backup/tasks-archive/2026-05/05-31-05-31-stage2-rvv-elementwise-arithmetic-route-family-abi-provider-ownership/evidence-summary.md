# Evidence Summary

## Module Boundary Closed

- Added scalar-broadcast elementwise typed config snapshot ownership to
  `RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan`.
- The scalar-broadcast route-family provider now derives and validates element
  type, signed C type, element bit width, SEW, LMUL, tail policy, mask policy,
  config contract, vector type, vector C type, setvl/load/store leaves, and
  route description dtype/config mirrors from selected typed RVV body/config
  facts before provider materialization.
- Generated-bundle evidence for `strided_add` now requires elementwise
  route-family provider mirrors: runtime control plan, source/destination memory
  forms, elementwise route-family plan, target leaf profile,
  provider_supported_mirror, header declarations, and C type mapping.

## Covered Elementwise Forms

- Plain elementwise arithmetic: `add`, `sub`, `mul`.
- Masked elementwise arithmetic: `masked_add`.
- Strided elementwise arithmetic: `strided_add`.
- Scalar-broadcast elementwise arithmetic: `scalar_broadcast_add`,
  `scalar_broadcast_sub`, `scalar_broadcast_mul`.

## Direct / Pre-Realized Shortcut Status

- `--direct-pre-realized-route-entry` remains fail-closed for selected
  pre-realized elementwise forms.
- Positive selected-boundary generated-bundle evidence uses selected-body
  realization and provider-built routes, not direct route-entry shortcut
  authority.

## Checks Run

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body ... --op-kind add --op-kind sub --op-kind mul --op-kind masked_add --op-kind strided_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run ... --op-kind scalar_broadcast_add --op-kind scalar_broadcast_sub --op-kind scalar_broadcast_mul --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry ... --op-kind add --op-kind strided_add` failed closed as expected.
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body ...` for segment2 five-form and indexed memory non-regression.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body ... --op-kind add --op-kind sub --op-kind mul --op-kind masked_add --op-kind strided_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py ... --op-kind scalar_broadcast_add --op-kind scalar_broadcast_sub --op-kind scalar_broadcast_mul --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91`
- Bounded authority-leak scan over touched files and diff-only scan. Diff-only
  scan added no legacy i32, descriptor, source-front-door, route-id,
  artifact-name, exact-intrinsic, or bare supported authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 464/464.

## Artifact Roots

- `artifacts/tmp/05-31-stage2-rvv-elementwise-arithmetic-route-family-abi-provider-ownership/dry-run/pre-realized-elementwise/pre-realized-elementwise`
- `artifacts/tmp/05-31-stage2-rvv-elementwise-arithmetic-route-family-abi-provider-ownership/dry-run/scalar-broadcast-elementwise/scalar-broadcast-elementwise`
- `artifacts/tmp/05-31-stage2-rvv-elementwise-arithmetic-route-family-abi-provider-ownership/dry-run/direct-pre-realized-fail-closed/direct-pre-realized-fail-closed`
- `artifacts/tmp/05-31-stage2-rvv-elementwise-arithmetic-route-family-abi-provider-ownership/dry-run/non-regression-memory/non-regression-memory`
- `artifacts/tmp/05-31-stage2-rvv-elementwise-arithmetic-route-family-abi-provider-ownership/ssh/pre-realized-elementwise/pre-realized-elementwise`
- `artifacts/tmp/05-31-stage2-rvv-elementwise-arithmetic-route-family-abi-provider-ownership/ssh/scalar-broadcast-elementwise/scalar-broadcast-elementwise`
