# Stage2 RVV Computed-Mask Widening Dot Reduction Round Notes

## Implementation Summary

- Added one bounded computed-mask signed widening dot-product reduction slice:
  `out_i32[0] = acc_seed + sum_i((cmp_lhs_i32[i] < cmp_rhs_i32[i]) ? (int32)lhs_i16[i] * (int32)rhs_i16[i] : 0)`.
- The current bounded compare side uses `i32/m1` compare inputs because the existing generic compare verifier supports the Stage 2 compare-mask surface at `SEW32 LMUL m1`; dot operands remain `i16/mf2`, and the reduction result/seed remain scalar `i32`.
- Added dot-lhs/dot-rhs runtime ABI roles, typed pre-realized body op, realized `tcrv_rvv.masked_widening_dot_reduce`, selected-body realization, route planning/provider emission, construction protocol metadata, target header metadata, generated-bundle script support, positive target artifact fixture, and negative fail-closed verifier tests.
- Provider lowering uses RVV plugin-owned planning: compare mask is produced from typed compare facts; masked widening multiply uses the real RVV `_m(mask,lhs,rhs,vl)` form and then merges inactive lanes to zero before horizontal reduction, so inactive lanes cannot contribute.

## Evidence

- Focused build: `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`.
- Focused C++ checks: `build/bin/tianchenrv-rvv-extension-plugin-test && build/bin/tianchenrv-construction-protocol-common-test && build/bin/tianchenrv-target-artifact-export-test`.
- Negative verifier check: `build/bin/tcrv-opt test/Dialect/RVV/computed-mask-widening-dot-reduction-negative.mlir --split-input-file --verify-diagnostics`.
- Positive selected-body checks:
  - `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries | FileCheck ... --check-prefix=REALIZED`
  - `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck ... --check-prefix=PLAN`
  - `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | FileCheck ... --check-prefix=HEADER`
- Generated-bundle dry-run: `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_widening_dot_reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --dry-run --overwrite` passed at `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260520T125418Z`.
- Real RVV correctness: `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_widening_dot_reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --overwrite` passed at `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260520T125428Z`.
- Full regression: `cmake --build build --target check-tianchenrv -j2` passed `239/239`.
- Formatting: `git diff --check` passed.

## Authority Scan

- No positive/default route authority was added for `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor/direct-C/source-export, or common/export RVV semantics.
- The only new `rvv-i32m1` mention is a negative verifier test that asserts stale `route_id` authority is rejected.
