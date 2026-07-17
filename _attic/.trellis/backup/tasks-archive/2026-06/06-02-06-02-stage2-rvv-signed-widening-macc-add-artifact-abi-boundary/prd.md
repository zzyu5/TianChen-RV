# Stage2 RVV signed widening macc-add artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path fail-closed and executable:

```text
selected tcrv.exec RVV variant
  -> typed_widening_macc_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / i16mf2 lhs load /
     i16mf2 rhs load / i32m1 accumulator load / signed widening_macc add /
     i32m1 result store
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

Route authority must remain the typed/realized `tcrv_rvv` body and RVV
plugin/provider facts. Route ids, artifact names, metadata mirrors, exact
intrinsic spellings, descriptors, C strings, scripts, runtime counts, test
names, direct route-entry support, or source-front-door markers are
mirrors/evidence only.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV signed widening macc-add artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no entries.
* Initial `git log --oneline -8` started at
  `e354dc7b rvv: validate computed masked macc artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require selected typed RVV
  body -> RVV plugin-local realization/provider facts -> common EmitC -> target
  artifact -> real `ssh rvv` evidence for runtime/correctness claims.
* The archived computed-masked macc-add task provides the closest provider fact
  surface pattern: canonical provider-owned facts consumed by provider plan,
  route binding, target artifact validation, generated-bundle dry-run, and real
  `ssh rvv` execution.
* The active route class is direct-provider contraction. The RVV plugin must
  consume verified contraction family/materialization/math facts into a direct
  contraction provider plan before constructing `TCRVEmitCLowerableRoute`; the
  statement owner then consumes that provider plan instead of rediscovering
  route facts from operation names, ABI strings, route ids, or metadata.
* The testing contract already defines `widening_macc_boundary` evidence for
  executable `widening_macc_add` and requires provider-derived widening facts,
  target validator consumption, explicit source/accumulator/result ABI roles,
  stale mirror rejection, and real `ssh rvv` evidence.
* The bounded production gap is the existing signed i16mf2 x i16mf2 + i32m1 ->
  i32m1 widening MAcc relation. This task must not become a widening matrix,
  unsigned/bfloat/dtype batch, LMUL clone batch, frontend task, or source
  artifact front door.

## Requirements

* Keep support rooted in the selected typed/pre-realized `tcrv_rvv` body, RVV
  plugin-local realization, contraction route-family plan, direct contraction
  provider plan, provider-built route facts, and target validator consumption.
* Support exactly the existing signed relation:
  `i16mf2 lhs x i16mf2 rhs + i32m1 accumulator -> i32m1 result`.
* Validate runtime ABI order `lhs,rhs,acc,out,n`; every generated
  header/prototype runtime parameter must be represented in provider-derived
  operand binding and marked as exported.
* Validate source SEW/LMUL `16/mf2`, accumulator/result SEW/LMUL `32/m1`,
  policy, unit-stride memory form, runtime AVL/VL contract, setvl placement,
  and with-vl scope.
* Validate typed compute op `tcrv_rvv.widening_macc`, MAcc kind `add`, signed
  widening relation
  `signed-i16mf2xi16mf2-plus-i32m1-to-i32m1`, accumulator layout
  `separate-i32-vector-accumulator-input`, and result layout
  `store-widening-multiply-accumulate-result-to-output-buffer`.
* Provider route description must carry route operand binding plan/summary,
  direct contraction route-family plan, target capability/config/profile facts,
  required headers, C type mapping, provider-derived widening facts, and an
  explicit `provider_supported_mirror`.
* Target artifact validation must consume the provider fact surface and reject
  stale or missing provider/candidate mirrors before bundle acceptance.
* Fail closed for wrong ABI order, missing `lhs`/`rhs`/`acc`/`out`/`n`,
  stale non-widening MAcc facts, stale widening-dot facts, wrong source or
  accumulator/result SEW/LMUL, wrong signedness, wrong widening relation,
  wrong accumulator layout, wrong result layout, missing route-family plan,
  missing header/type facts, stale target capability/profile facts, stale
  binding summary, stale provider mirror, descriptor/direct-C/source-front-door
  residue, direct route-entry residue, route-id/intrinsic authority, or common
  EmitC RVV semantic inference.
* Generated-bundle dry-run evidence must record provider-derived
  `widening_macc_boundary`, selected source ABI roles, route operand binding,
  target validator consumed facts, direct contraction plan, target
  leaf/profile/config facts, headers/types, `provider_supported_mirror`,
  source/accumulator/result SEW/LMUL, widening relation, accumulator layout,
  result layout, statement operand order, and runtime AVL/VL facts.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,7,16,23,257` and at least two input patterns proving signed i16
  widening products accumulate into i32, positive/negative products are both
  exercised, products exceed source-width range in multi-lane cases,
  accumulator/source/tail sentinels are preserved, and runtime `n`/AVL is
  honored.

## Acceptance Criteria

* [x] Focused production diff strengthens signed widening macc-add
      provider/direct-contraction/target validation; no metadata-only closeout.
* [x] Selected-body realization/FileCheck proves pre-realized body consumption
      into setvl/with_vl/i16mf2 loads/i32m1 accumulator load/widening_macc/store.
* [x] Target artifact tests fail closed for stale or missing provider mirror,
      ABI order/roles, binding plan/summary, source/accumulator/result
      SEW/LMUL, signed widening relation, typed compute op, accumulator layout,
      result layout, route-family/direct-contraction plan, header/type facts,
      target capability/profile facts, stale non-widening MAcc facts, stale
      widening-dot facts, descriptor/direct-C/source-front-door residue, direct
      route-entry residue, and intrinsic/route-id authority.
* [x] Existing or tightened REALIZED/PLAN/HEADER checks for the pre-realized
      widening macc-add fixture pass.
* [x] Generated-bundle dry-run records provider-derived widening MAcc facts,
      route operand binding, target validator consumption, target leaf/profile,
      headers/types, `provider_supported_mirror`, source/accumulator/result
      SEW/LMUL, signed widening relation, accumulator layout, result layout,
      statement operand order, and runtime AVL/VL facts.
* [x] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,7,16,23,257` and at least two input patterns.
* [x] Smallest relevant build/test/script commands, direct FileCheck
      equivalents if lit is unavailable, `git diff --check`, and a bounded
      old-authority scan over touched files pass.
* [x] Trellis task is finished/archived and one coherent commit is created if
      the task completes.

## Completion Notes

Implemented a provider-owned `RVVWideningMAccRouteFacts` surface and consumed
it from provider plan validation, target artifact validation, generated-bundle
evidence, dry-run checks, and runtime harness validation. The bounded supported
relation is exactly signed `i16mf2 x i16mf2 + i32m1 -> i32m1` with ABI order
`lhs,rhs,acc,out,n`, `tcrv_rvv.widening_macc`, source/accumulator/result
SEW/LMUL facts, contraction route-family plan, route operand binding summary,
required headers, C type mapping, and explicit `provider_supported_mirror`.

Self-repair performed during check:

* Fixed the generated widening MAcc harness so the two-pattern runtime loop
  passes `pattern` into each case and verifies source, accumulator, output tail,
  and signed positive/negative widening products.
* Tightened target validator diagnostics so stale source/result SEW/LMUL reports
  the missing `i16mf2` source and `i32m1` accumulator/result facts instead of a
  generic layout/relation error.
* Updated target C++ fail-closed assertions to match the more precise
  `i16mf2`/`i32m1` diagnostics.

Final runtime evidence:

```text
artifacts/tmp/stage2-rvv-signed-widening-macc-add/final-ssh-rvv/pre-realized-widening-macc-add
tcrv_rvv_generated_bundle_abi_widening_macc_add_ok counts=0,1,7,16,23,257 patterns=0,1
PASS op=widening_macc_add counts=0,1,7,16,23,257 patterns=0,1
```

## Out Of Scope

* Broad widening matrix, unsigned/signed/bfloat/dtype batch, LMUL clone batch,
  high-level matmul/Linalg frontend, source-front-door positive route,
  performance autotuning, report-only commit, common EmitC RVV semantics,
  computed-masked macc redo, runtime-scalar macc redo, reductions, segment or
  indexed memory, widening dot-reduction redo, q4/q8 kernels, dashboards,
  descriptor routes, direct-C/source exporters, or direct route-entry positive
  support.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names,
  fixture names, or harness constants as the source of widening relation,
  dtype/config, accumulator layout, result layout, runtime ABI, policy, or
  route support.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-macc-add-artifact-abi-boundary/prd.md`

Repository files to inspect while deriving implementation:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* widening/contraction route-family owner files under `lib/Plugin/RVV/EmitC/`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-widening-macc-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-macc-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-widening-macc-add-fail-closed.test`
