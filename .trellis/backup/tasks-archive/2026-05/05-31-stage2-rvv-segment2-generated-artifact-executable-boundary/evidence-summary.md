# Evidence Summary

## Scope

Task: `05-31-stage2-rvv-segment2-generated-artifact-executable-boundary`

This round validated the executable generated-artifact/runtime ABI boundary for
five RVV segment2 forms:

- `segment2_deinterleave_unit_store`
- `segment2_interleave_unit_load`
- `computed_masked_segment2_load_unit_store`
- `computed_masked_segment2_store_unit_load`
- `computed_masked_segment2_update_unit_load`

## Production Outcome

No production code repair was required. Current HEAD already carries the
provider-preflight, target artifact route-family validator, generated-bundle
fixtures, ABI order, harness, and real `ssh rvv` executable path needed for the
five forms.

This task therefore closes the executable claim by evidence, not by adding a
new route or helper beside the existing production path.

## Artifact Roots

- Dry-run:
  `artifacts/tmp/stage2_segment2_generated_artifact_executable_boundary/selected-boundary-dry-run`
- Real RVV hardware:
  `artifacts/tmp/stage2_segment2_generated_artifact_executable_boundary/selected-boundary-ssh-rvv`
- Direct route-entry fail-closed:
  `artifacts/tmp/stage2_segment2_generated_artifact_executable_boundary/direct-route-entry-fail-closed`

## Runtime Counts

All five forms used:

```text
0, 1, 16, 23, 257
```

These cover zero elements, one element, exact VL-sized execution, tail
execution, and a stress case.

## ABI / Route Facts

Observed generated header prototypes and runtime ABI order:

```text
segment2_deinterleave_unit_store:
  void tcrv_emitc_pre_realized_body_segment2_deinterleave_unit_store_kernel_pre_realized_body_rvv_segment2_deinterleave_unit_store(const int32_t *src, int32_t *out0, int32_t *out1, size_t n);
  ABI order: src,out0,out1,n

segment2_interleave_unit_load:
  void tcrv_emitc_pre_realized_body_segment2_interleave_unit_load_kernel_pre_realized_body_rvv_segment2_interleave_unit_load(const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);
  ABI order: src0,src1,dst,n

computed_masked_segment2_load_unit_store:
  void tcrv_emitc_pre_realized_body_cmseg_load_kernel_pre_realized_body_rvv_cmseg_load(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, int32_t *out0, int32_t *out1, size_t n);
  ABI order: cmp_lhs,cmp_rhs,src,out0,out1,n

computed_masked_segment2_store_unit_load:
  void tcrv_emitc_pre_realized_body_cmseg_store_kernel_pre_realized_body_rvv_cmseg_store(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);
  ABI order: cmp_lhs,cmp_rhs,src0,src1,dst,n

computed_masked_segment2_update_unit_load:
  void tcrv_emitc_pre_realized_body_cmseg_update_kernel_pre_realized_body_rvv_cmseg_update(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);
  ABI order: cmp_lhs,cmp_rhs,src0,src1,dst,n
```

## Hardware Correctness

The real `ssh rvv` run compiled and executed all generated artifacts. The
harness output reported:

- plain deinterleave: field order distinguishing lanes and tail preservation;
- plain interleave: interleaved layout, field order distinguishing lanes, and
  tail preservation;
- computed-mask segment2 load: active lanes updated, inactive lanes preserved,
  source preserved, field order checked, and tail preserved;
- computed-mask segment2 store: active lanes updated, inactive lanes preserved,
  source preserved, field order checked, and tail preserved;
- computed-mask segment2 update: active lanes updated by the structural update,
  inactive lanes preserved, source preserved, field order checked, and tail
  preserved.

Every form printed `PASS op=<form> counts=0,1,16,23,257`.

## Direct Route-Entry Retirement

The direct pre-realized route-entry probe exited nonzero with:

```text
--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): segment2_deinterleave_unit_store, segment2_interleave_unit_load, computed_masked_segment2_load_unit_store, computed_masked_segment2_store_unit_load, computed_masked_segment2_update_unit_load; the direct route-entry shortcut is retired and these fixtures must use the public selected lowering-boundary producer before target bundle export
```

## Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-31-stage2-rvv-segment2-generated-artifact-executable-boundary`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Five-form selected-boundary dry-run: passed.
- Five-form `ssh rvv` generated-bundle run: passed.
- Direct route-entry fail-closed probe: failed closed as expected.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused segment2 generated-bundle lit filter: 12/12 passed.
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `git diff --check`
- Bounded authority scan over relevant segment2 production/test/evidence
  surfaces: no new production authority drift; matches were existing
  guardrails, fail-closed diagnostics, or negative tests.
- `cmake --build build --target check-tianchenrv -j2`: 464/464 passed.

## Spec Update Judgment

No `.trellis/spec/` update was needed. This round did not create a new
contract; it verified the already-specified segment2 target export consumer
contract, provider-route rebuild requirement, mirror-only metadata role,
direct-route-entry retirement, and RVV hardware evidence requirement.
