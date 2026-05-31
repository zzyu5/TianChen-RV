# Evidence Summary

## Scope

Task:
`05-31-stage2-rvv-indexed-memory-route-family-abi-provider-ownership`

This round hardened the RVV indexed memory target artifact consumer boundary for
computed-mask indexed scatter while preserving the existing base indexed
gather/scatter and computed-mask indexed gather paths.

## Production Outcome

Production code changed in:

- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`

The compare/select-mask target artifact route-family validator now consumes
`indexUniqueness` as a provider-derived route fact:

- `ComputedMaskIndexedScatterStoreUnitLoad` requires `indexUniqueness =
  "unique"`.
- Non-scatter compare/select-mask routes reject stale non-empty index
  uniqueness facts.
- Candidate metadata `tcrv_rvv.index_uniqueness` is validated as a mirror of
  the provider description after route construction, not as route authority.

Common EmitC was not changed. The route semantics remain provider-owned.

## Test Coverage

Changed target C++ coverage in:

- `test/Target/TargetArtifactExportTest.cpp`

The test adds a manual computed-mask indexed scatter route/candidate fixture
for the target artifact registry and checks:

- positive provider/candidate validation for computed-mask indexed scatter;
- stale computed-mask indexed scatter provider `indexUniqueness` rejection;
- stale computed-mask indexed scatter candidate `tcrv_rvv.index_uniqueness`
  mirror rejection;
- stale computed-mask indexed gather uniqueness rejection, proving gather stays
  accepted with no uniqueness authority.

Self-repair performed during testing:

- An attempted generic `RVVTargetArtifactCandidateFixture` positive for
  computed-mask indexed scatter failed during fixture parsing because that
  helper generated a `tcrv_rvv.binary` body for this op class.
- The test was repaired by using the existing manual target artifact
  provider/route/candidate style already used in this file, keeping the round
  focused on target artifact validation rather than broadening into fixture
  generator repair.

## Artifact Roots

- Selected-boundary dry-run:
  `artifacts/tmp/05-31-stage2-rvv-indexed-memory-route-family-abi-provider-ownership/dry-run/pre-realized-indexed-memory`
- Real RVV hardware:
  `artifacts/tmp/05-31-stage2-rvv-indexed-memory-route-family-abi-provider-ownership/ssh-rvv/pre-realized-indexed-memory`
- Direct route-entry fail-closed:
  `artifacts/tmp/05-31-stage2-rvv-indexed-memory-route-family-abi-provider-ownership/direct-route-entry-fail-closed/direct-pre-realized-indexed-scatter`

## Runtime Counts

All generated-bundle dry-run and `ssh rvv` evidence used:

```text
0, 1, 16, 23, 257
```

These cover zero elements, one element, exact-VL execution, tail execution, and
a stress case.

## Generated-Bundle Evidence

The selected-boundary generated-bundle dry-run passed for:

- `indexed_gather_unit_store`
- `indexed_scatter_unit_load`
- `computed_masked_indexed_gather_load_unit_store`
- `computed_masked_indexed_scatter_store_unit_load`

Output:

```text
rvv_generated_bundle_abi_e2e: dry_run_success
artifact_dir: artifacts/tmp/05-31-stage2-rvv-indexed-memory-route-family-abi-provider-ownership/dry-run/pre-realized-indexed-memory
```

The real `ssh rvv` run compiled and executed all four generated artifacts. All
forms printed `PASS op=<form> counts=0,1,16,23,257`.

Computed-mask indexed scatter hardware output included:

```text
computed_masked_indexed_scatter_store_unit_load case n=16 ok computed_mask indexed_scatter active_lanes=8 inactive_lanes=8 inactive_preserved_lanes=8 noncontiguous_index_lanes=14 source_preserved tail_preserved
computed_masked_indexed_scatter_store_unit_load case n=23 ok computed_mask indexed_scatter active_lanes=11 inactive_lanes=12 inactive_preserved_lanes=12 noncontiguous_index_lanes=22 source_preserved tail_preserved
computed_masked_indexed_scatter_store_unit_load case n=257 ok computed_mask indexed_scatter active_lanes=129 inactive_lanes=128 inactive_preserved_lanes=128 noncontiguous_index_lanes=256 source_preserved tail_preserved
PASS op=computed_masked_indexed_scatter_store_unit_load counts=0,1,16,23,257
```

## Direct Route-Entry Retirement

The retired direct pre-realized route-entry probe failed closed as expected:

```text
--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): computed_masked_indexed_scatter_store_unit_load; the direct route-entry shortcut is retired and these fixtures must use the public selected lowering-boundary producer before target bundle export
```

## Checks

- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body ...`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --ssh-target rvv ...`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry ...` failed closed as expected.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `cmake --build build --target check-tianchenrv -j2`: 464/464 passed.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-31-stage2-rvv-indexed-memory-route-family-abi-provider-ownership`

## Authority Scan

Bounded scan over the production file, target C++ test, and task context found:

- The production diff only adds provider/candidate consumption of
  `index_uniqueness`.
- Existing exact intrinsic strings remain in target validator/test fixture data
  as provider route leaves and validation expectations; the new production code
  does not infer support from exact intrinsic spelling.
- `metadata-derived-*` and `metadata-only-*` matches are negative-test mutation
  labels or PRD non-goal text.
- No new descriptor-derived, source-front-door-derived, source-artifact-derived,
  common-EmitC-derived, direct-route-entry-derived, artifact-name-derived,
  route-id-derived, or legacy-i32-derived production authority was introduced.

## Spec Update Judgment

No `.trellis/spec/` update was needed. This round implemented an already-stated
contract: RVV plugin/provider facts are authority, target artifact metadata is
mirror-only after provider route construction, and common EmitC remains neutral.
