# Evidence Summary

## Scope

Task:
`05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership`

This round hardened the RVV segment2 target artifact ABI/provider consumer
boundary for plain segment2 candidate mirrors while preserving the existing
computed-mask segment2, generated-bundle, and indexed memory paths.

Covered route forms:

- `segment2_deinterleave_unit_store`
- `segment2_interleave_unit_load`
- `computed_masked_segment2_load_unit_store`
- `computed_masked_segment2_store_unit_load`
- `computed_masked_segment2_update_unit_load`

## Production Outcome

Production code changed in:

- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`

The segment2 target artifact candidate mirror validator now consumes
provider-derived plain segment2 facts for:

- segment tuple C type;
- deinterleave segment-load and field-extract mirrors;
- interleave tuple-create and segment-store mirrors;
- field0/field1 roles;
- field0/field1 result or payload names;
- deinterleave field destination memory forms;
- interleave field source memory forms;
- absence of stale non-applicable segment load/store/tuple/extract mirrors;
- absence of stale computed-mask segment2 update mirrors on plain segment2.

This is target artifact validation after route construction. It does not make
metadata, route ids, exact intrinsic spellings, artifact names, scripts, direct
route-entry fixtures, or common EmitC semantic authority.

## Test Coverage

Changed C++ target artifact coverage in:

- `test/Target/TargetArtifactExportTest.cpp`

The test now checks stale or missing plain segment2 candidate mirrors for:

- deinterleave tuple C type, field role, field name, field destination memory
  form, segment-load mirror, missing field-extract mirror, and stale segment
  store mirror;
- interleave tuple C type, field role, field name, field source memory form,
  missing segment-store mirror, tuple-create mirror, and stale segment-load
  mirror.

Existing computed-mask segment2 checks continue to cover stale mask source,
field role, update arithmetic, runtime ABI order, provider mirror, route
operand binding, source/destination memory form, header, C type mapping,
runtime control, and stale plain route-family mirrors.

## Artifact Roots

- Segment2 selected-boundary dry-run:
  `artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/dry-run/pre-realized-segment2`
- Segment2 real RVV hardware:
  `artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/ssh-rvv/pre-realized-segment2`
- Segment2 direct route-entry fail-closed:
  `artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/direct-route-entry-fail-closed/direct-pre-realized-segment2`
- Indexed memory non-regression dry-run:
  `artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/non-regression/pre-realized-indexed-memory`

## Runtime Counts

Generated-bundle dry-run and `ssh rvv` segment2 evidence used:

```text
0, 1, 16, 23, 257
```

These cover zero elements, one element, exact-VL execution, tail execution, and
a stress case.

## Generated-Bundle Evidence

The selected-boundary dry-run passed for all five segment2 forms:

```text
rvv_generated_bundle_abi_e2e: dry_run_success
artifact_dir: artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/dry-run/pre-realized-segment2
```

The direct pre-realized route-entry probe failed closed as expected:

```text
--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): segment2_deinterleave_unit_store, segment2_interleave_unit_load, computed_masked_segment2_load_unit_store, computed_masked_segment2_store_unit_load, computed_masked_segment2_update_unit_load; the direct route-entry shortcut is retired and these fixtures must use the public selected lowering-boundary producer before target bundle export
```

## Hardware Correctness

The real `ssh rvv` run compiled and executed all five generated artifacts. Each
form printed `PASS op=<form> counts=0,1,16,23,257`.

Representative output included:

```text
PASS op=segment2_deinterleave_unit_store counts=0,1,16,23,257
PASS op=segment2_interleave_unit_load counts=0,1,16,23,257
PASS op=computed_masked_segment2_load_unit_store counts=0,1,16,23,257
PASS op=computed_masked_segment2_store_unit_load counts=0,1,16,23,257
PASS op=computed_masked_segment2_update_unit_load counts=0,1,16,23,257
```

The harness reported field-order checks, source preservation, inactive-lane
preservation for computed-mask forms, and tail preservation.

## Indexed Memory Non-Regression

The indexed memory generated-bundle dry-run passed for:

- `indexed_gather_unit_store`
- `indexed_scatter_unit_load`
- `computed_masked_indexed_gather_load_unit_store`
- `computed_masked_indexed_scatter_store_unit_load`

Output:

```text
rvv_generated_bundle_abi_e2e: dry_run_success
artifact_dir: artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/non-regression/pre-realized-indexed-memory
```

## Checks

- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body ...` for segment2 five-form dry-run
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry ...` failed closed as expected
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --ssh-target rvv ...` for segment2 five-form hardware run
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body ...` for indexed memory non-regression
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`: 464/464 passed

## Authority Scan

Bounded scan over touched production/test/task files found:

- The production diff only adds provider/candidate mirror consumption for plain
  segment2 tuple, field, memory-form, and segment callee mirrors.
- New `metadata-derived-*` strings appear only in negative test mutations.
- New exact `__riscv_*` strings appear only as expected mirror values in
  negative target-validator tests; they do not select route support.
- No new central ad hoc, descriptor-derived, ABI-string-derived,
  script-derived, artifact-name-derived, common-EmitC-derived,
  source-front-door-derived, route-id-derived, direct-route-entry-only,
  pre-realized-fixture-only, or legacy-i32-derived production authority was
  introduced.

## Spec Update Judgment

No `.trellis/spec/` update was needed. This round implemented an already-stated
contract: RVV plugin/provider route facts are authority, target artifact
metadata is mirror-only after provider route construction, and common EmitC
remains neutral.
