# Stage2 RVV widen i32-to-i64 conversion artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path fail-closed and executable:

```text
selected tcrv.exec RVV variant
  -> typed_widening_conversion_pre_realized_body for widen_i32_to_i64
  -> RVV plugin-local realization into setvl / with_vl / i32m1 source load /
     i32m1-to-i64m2 widening_convert / i64m2 result store
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

Route authority must remain the typed/realized `tcrv_rvv` body and RVV
plugin/provider facts. Route ids, artifact names, metadata mirrors, exact
intrinsic spellings, descriptors, C strings, scripts, runtime counts, test
names, direct route-entry support, source-front-door markers, or common EmitC
logic are mirrors/evidence only.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV widen i32-to-i64 conversion artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no file entries, so the worktree started
  clean.
* Initial `git log --oneline -8` started at
  `8ef27cdf rvv: validate computed masked strided dot artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require selected typed RVV
  body -> RVV plugin-local realization/provider facts -> common EmitC ->
  target artifact -> real `ssh rvv` evidence for runtime/correctness claims.
* The archived computed-mask strided widening dot task provides the latest
  completed production pattern: provider-owned facts, target validation
  consumption, generated-bundle dry-run evidence, real `ssh rvv` execution,
  task archive, and one coherent commit.
* The direction brief names existing code and fixtures for this route:
  `TypedWideningConversionPreRealizedBodyOp`, `widening_convert`, load/store
  and vector config ODS, RVV provider/planning route-family files,
  `RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and the pre-realized/explicit
  widen i32-to-i64 target artifact fixtures.

## Requirements

* Support exactly the bounded `widen_i32_to_i64` generated artifact route.
* Keep support rooted in selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, widening conversion route-family plan,
  materialization facts, math operand binding facts, route-control provider
  plan, RVV-owned widening conversion statement plan, provider-built route
  facts, and target validator consumption.
* Validate runtime ABI order `lhs,out,n`.
* Validate source type policy:
  i32 element type, source SEW `32`, source LMUL `m1`, source vector/C type,
  and unit-stride source load.
* Validate result type policy:
  i64 element type, destination SEW `64`, destination LMUL `m2`, result
  vector/C type, destination-config runtime AVL/VL contract, and result store.
* Validate conversion facts:
  typed compute op `tcrv_rvv.widening_convert`,
  conversion kind `widen_signed_integer`, conversion relation
  `signed-i32m1-to-i64m2`, widening conversion route-family plan, conversion
  intrinsic, required header declarations, and C type mapping.
* Provider route description must carry route operand binding plan/summary,
  widening conversion route-family plan, runtime AVL/VL contract, required
  headers, C type mapping, target leaf/profile facts, provider target
  validation facts, and explicit `provider_supported_mirror` or equivalent
  mirror labeling.
* Route operand binding summary must represent every generated
  header/prototype ABI parameter as:
  `<logical>=<role>:<c-name>:abi|...|hdr`.
* Target artifact validation must consume provider-owned facts for this
  conversion route family and reject stale candidate/provider mirrors before
  bundle acceptance.
* Unsupported or stale combinations must fail closed with targeted diagnostics
  rather than falling back to names, strings, descriptors, artifact metadata,
  intrinsic spelling authority, elementwise-route residue, or common EmitC RVV
  semantic inference.
* Generated-bundle dry-run evidence must record provider-derived conversion
  facts, binding order, source/result SEW-LMUL/type policy, conversion
  relation, route-family plan, header/type facts, capability/profile facts,
  target validator consumption, provider mirror, runtime AVL/VL facts, and
  harness coverage contract.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,23,257` and at least two input patterns proving signed
  i32-to-i64 sign extension, wide-magnitude values beyond i16 range, output
  store, tail sentinel preservation, and runtime `n`/AVL honoring.

## Acceptance Criteria

* [x] Focused production diff strengthens the RVV plugin/provider and target
      validator fact surface for `widen_i32_to_i64`; no metadata-only
      closeout.
* [x] Provider-owned widening conversion facts are consumed by target
      validation rather than relying on target-local stale route truth.
* [x] Route operand binding plan/summary includes `lhs`, `out`, and `n` with
      role, C name, `abi`, required use tokens, and `hdr`.
* [x] Selected-body realization/FileCheck proves pre-realized body consumption
      into setvl/with_vl/source load/widening_convert/store.
* [x] Target artifact tests fail closed for stale or missing provider mirror,
      target leaf/profile, ABI order/roles, binding plan/summary,
      source/result SEW-LMUL/type facts, conversion relation, route-family
      plan, header/type facts, stale provider conversion mirror, stale
      elementwise-route residue, and target profile mismatch.
* [x] Existing or tightened REALIZED/PLAN/HEADER checks for the pre-realized
      and explicit selected-body widen i32-to-i64 fixtures pass.
* [x] Generated-bundle dry-run records provider-derived
      `conversion_sew_policy_boundary`, complete route operand binding, target
      validator consumption, target leaf/profile, headers/types,
      `provider_supported_mirror`, conversion relation, source/result
      SEW-LMUL policy, and runtime AVL/VL facts.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,23,257` and at least two signed input patterns.
* [x] Smallest relevant build/test/script commands, direct FileCheck
      equivalents if lit is unavailable, `git diff --check`, and a bounded
      old-authority scan over touched files pass.
* [x] Trellis task is finished/archived and one coherent commit is created if
      the task completes.

## Completion Evidence

Completed on 2026-06-03.

Production changes:

* Added provider-owned `RVVWideningConversionRouteFacts` and
  `getRVVWideningConversionRouteFacts(...)` for selected-body widening
  conversion facts.
* Rewired RVV route planning validation/derivation to copy or validate
  widening conversion shared constants from the provider-owned fact surface.
* Rewired target artifact conversion dtype-policy validation to consume the
  same provider fact accessor instead of a target-local conversion facts table.
* Strengthened target artifact tests with `widen_i32_to_i64` fail-closed cases
  for stale provider mirror, source/destination SEW-LMUL policy, target leaf
  profile, and route operand binding summary.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  widening conversion fact-surface contract.

Checks run:

* `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter widen-i32-to-i64` from `build/test` (`5` passed, `472` excluded)
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --ssh-target rvv --op-kind widen_i32_to_i64 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj llvm-readobj-20`
* `git diff --check`
* Bounded added-diff old-authority scan for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, exact `__riscv_*_i32m1`, descriptor/direct-C/source-export,
  source-front-door, and source-artifact residue.

Real RVV evidence:

```text
widen_i32_to_i64 case n=0 pattern=0 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
widen_i32_to_i64 case n=0 pattern=1 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
widen_i32_to_i64 case n=1 pattern=0 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
widen_i32_to_i64 case n=1 pattern=1 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
widen_i32_to_i64 case n=16 pattern=0 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
widen_i32_to_i64 case n=16 pattern=1 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
widen_i32_to_i64 case n=23 pattern=0 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
widen_i32_to_i64 case n=23 pattern=1 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
widen_i32_to_i64 case n=257 pattern=0 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
widen_i32_to_i64 case n=257 pattern=1 ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked
tcrv_rvv_generated_bundle_abi_widen_i32_to_i64_ok counts=0,1,16,23,257
PASS op=widen_i32_to_i64 counts=0,1,16,23,257
```

## Definition Of Done

* Production owner behavior is implemented in the RVV plugin/provider and
  target validation path, not only in tests, scripts, metadata, or reports.
* Focused MLIR/FileCheck, C++ target/provider checks, generated-bundle dry-run,
  and real RVV generated-bundle evidence pass.
* No new broad conversion matrix, dtype/LMUL clone batch, source-front-door
  positive route, descriptor/direct-C exporter, direct route-entry support, or
  common EmitC RVV semantic inference is introduced.
* Task status, completion notes, archive, final clean worktree, and commit are
  truthful.

## Technical Approach

1. Inspect the existing `widen_i32_to_i64` ODS, selected-body realization,
   route-family planning/provider, target artifact validation, generated-bundle
   script, fixtures, and fail-closed tests.
2. Identify whether provider fact surface, target validation, evidence JSON,
   or runtime harness is missing any PRD-required conversion ABI boundary fact.
3. Implement the smallest production diff that makes provider facts and target
   validation authoritative for source/destination SEW-LMUL, conversion
   relation, ABI order, operand binding, headers/types, target profile, and
   provider mirror.
4. Tighten fixtures and tests to prove good path and stale/mismatch
   fail-closed behavior.
5. Run focused build/tests/dry-run and real `ssh rvv` generated-bundle
   correctness, then finish/archive and commit.

## Decision (ADR-lite)

**Context**: Stage 2 is expanding corrected typed RVV coverage. The brief asks
for one conversion route boundary, not broad conversion coverage.

**Decision**: Use a bounded provider-owned fact surface for
`widen_i32_to_i64`, consumed by both RVV provider and target validation, with
generated-bundle and real RVV evidence as mirrors/evidence after provider
route construction.

**Consequences**: This keeps conversion relation, source/result type policy,
runtime ABI, headers/types, and route support structural and plugin-owned. It
does not add narrowing, unsigned, float, frontend lowering, or dtype/LMUL clone
matrices.

## Out Of Scope

* Broad conversion matrix, narrowing, saturating, unsigned, float/integer
  conversion batches, dtype/LMUL clone families, high-level Linalg/Vector
  frontend lowering, source-front-door positive route, one-intrinsic wrapper
  dialect, report-only commit, common EmitC RVV semantics, dot-reduce redo,
  macc redo, compare/select redo, reduction, segment/indexed/broadcast routes,
  dashboards, descriptor routes, direct-C/source exporters, or direct
  route-entry positive support.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names,
  fixture names, or harness constants as the source of conversion relation,
  dtype/config, runtime ABI, policy, route support, or evidence authority.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi-boundary/implement.jsonl`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi-boundary/check.jsonl`

Repository files to inspect while deriving implementation:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* RVV provider/planning route-family headers and implementations
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-widen-i32-to-i64.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir`
* Matching dry-run and direct pre-realized fail-closed script tests
