# Stage2 RVV strided add artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV strided elementwise add compiler path
fail-closed and executable:

```text
selected tcrv.exec RVV variant
  -> typed_binary_pre_realized_body with strided-load-store memory form
  -> RVV plugin-local realization into setvl / with_vl /
     strided_load(lhs, lhs_stride) / strided_load(rhs, rhs_stride) /
     binary add / strided_store(out, out_stride)
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

Route authority must remain the selected typed/realized `tcrv_rvv` body and
RVV plugin/provider facts. Common EmitC/export may carry provider payloads and
mirrors only; they must not invent strided memory semantics, stride roles, ABI
order, dtype/config, operation kind, or fallback unit-stride behavior.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV strided add artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no file entries; `rtk` reported `ok`, so
  the worktree started clean.
* Initial `git log --oneline -8` started at
  `22c2683b rvv: validate scalar broadcast add artifact ABI`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require selected typed RVV
  body -> RVV plugin-local realization/provider facts -> common EmitC ->
  target artifact -> real `ssh rvv` evidence for runtime/correctness claims.
* The archived scalar-broadcast add task completed the latest provider-owned
  ABI/header binding pattern for a Stage 2 elementwise path.
* Existing strided add files already expose the core selected-body route shape:
  `typed_binary_pre_realized_body` with `memory_form = "strided-load-store"`,
  explicit selected-body `tcrv_rvv.strided_load` / `tcrv_rvv.binary` /
  `tcrv_rvv.strided_store`, generated-bundle dry-run tests, and basic target
  header stale-layout / stale-ABI fail-closed checks.
* Current strided add route operand binding summaries still use `header` in
  several provider and fixture tokens. The current route contract requires
  generated header/prototype ABI participants to carry explicit `hdr`
  participation markers.
* Current strided add generated-bundle harness defaults to one stride triple
  `lhs_stride=2`, `rhs_stride=3`, `out_stride=2`; this does not yet prove
  independent lhs/rhs/out runtime stride roles across multiple non-unit
  patterns such as `2:3:2` and `3:2:4`.

## Requirements

* Support exactly the bounded `strided_add` generated artifact route.
* Keep support rooted in selected typed/pre-realized `tcrv_rvv` body, RVV
  plugin-local realization, elementwise arithmetic route-family plan,
  materialization facts, route operand binding facts, route-control provider
  plan, RVV-owned statement plan, provider-built route facts, and target
  validator consumption.
* Validate runtime ABI order `lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride`.
* Validate dtype/config policy: source/result element type `i32`, SEW `32`,
  LMUL `m1`, tail/mask agnostic policy, VL type `size_t`, and RVV vector C
  type mapping.
* Validate operation and body relation: typed compute op `tcrv_rvv.binary`, op
  kind `add`, memory form `strided-load-store`, source memory form
  `strided-load`, destination memory form `strided-store`, lhs/rhs
  `tcrv_rvv.strided_load`, output `tcrv_rvv.strided_store`, and runtime
  AVL/VL loop control.
* Validate strided memory layout
  `element-strided-lhs-rhs-output-runtime-abi`.
* Validate independent runtime stride roles:
  `lhs_stride=lhs-input-stride`, `rhs_stride=rhs-input-stride`, and
  `out_stride=output-stride`, with provider-owned source facts
  `runtime_abi:lhs_stride`, `runtime_abi:rhs_stride`, and
  `runtime_abi:out_stride`.
* Route operand binding summary must represent every generated
  header/prototype ABI parameter as:
  `<logical>=<role>:<c-name>:abi|...|hdr`.
* Provider route description must carry strided elementwise route-family plan,
  route operand binding plan/summary, runtime AVL/VL contract, required
  headers, C type mapping, target leaf/profile facts, explicit
  `provider_supported_mirror` labeling, source/destination memory forms, and
  stride source mirrors derived from provider facts.
* Target artifact validation must consume provider-owned facts and reject stale
  candidate/provider mirrors before bundle acceptance.
* Unsupported or stale combinations must fail closed with targeted diagnostics
  rather than falling back to names, strings, descriptors, artifact metadata,
  intrinsic spelling authority, unit-stride residue, or common EmitC semantic
  inference.
* Generated-bundle dry-run evidence must record provider-derived strided memory
  facts, binding order, stride role/source facts, route-family plan,
  header/type facts, target validator consumption, provider mirror, runtime
  AVL/VL facts, and harness coverage contract.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,23,257` and at least stride triples `2:3:2` and `3:2:4`,
  proving lhs/rhs/out strides are independent, non-stride slots and tails
  preserve sentinels, output stores use `out_stride`, and runtime `n`/AVL is
  honored.

## Acceptance Criteria

* [ ] Focused production diff strengthens RVV plugin/provider and target
      validator fact surfaces for `strided_add`; no metadata-only closeout.
* [ ] Provider-owned strided add binding summary uses explicit
      `abi|...|hdr` participation markers for `lhs`, `rhs`, `out`, `n`,
      `lhs_stride`, `rhs_stride`, and `out_stride`.
* [ ] Target artifact tests fail closed for stale or missing provider mirror,
      route-family plan, ABI order, binding plan/summary, stride layout,
      independent stride role/source, header/type facts, target leaf/profile,
      and accidental unit-stride fallback.
* [ ] REALIZED/PLAN/HEADER checks for both pre-realized and explicit strided
      add fixtures prove setvl/with_vl/strided_load/strided_load/binary/
      strided_store and provider-derived route facts.
* [ ] Generated-bundle dry-run records provider-derived strided memory facts,
      complete route operand binding, target validator consumption, target
      leaf/profile, headers/types, `provider_supported_mirror`, runtime
      stride sources, independent stride triple cases, and runtime AVL/VL
      facts.
* [ ] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,23,257` and stride triples `2:3:2` and `3:2:4`.
* [ ] Smallest relevant build/test/script commands, direct FileCheck
      equivalents if lit is unavailable, `git diff --check`, and a bounded
      old-authority scan over touched files pass.
* [ ] Trellis task is finished/archived and one coherent commit is created if
      the task completes.

## Definition Of Done

* Production owner behavior is implemented in the RVV plugin/provider and
  target validation path, not only in tests, scripts, metadata, or reports.
* Focused MLIR/FileCheck, C++ target/provider checks, generated-bundle dry-run,
  and real RVV generated-bundle evidence pass.
* No broad add/sub/mul strided matrix, dtype/LMUL clone batch, source-front-door
  positive route, descriptor/direct-C exporter, direct route-entry support,
  common EmitC RVV semantic inference, scalar broadcast redo, conversion redo,
  dot-reduce redo, macc redo, compare/select redo, segment/indexed/masked
  memory route, dashboard, or unrelated Stage 2 route redo is introduced.
* Task status, completion notes, archive, final clean worktree, and commit are
  truthful.

## Technical Approach

1. Inspect existing strided add ODS, selected-body realization,
   route-family planning/provider, target artifact validation,
   generated-bundle script, fixtures, and fail-closed tests.
2. Tighten provider strided add operand binding facts so the generated summary
   carries `abi` and `hdr` participation markers.
3. Tighten target/dry-run/FileCheck expectations and C++ fail-closed coverage
   for stale or missing strided add route facts.
4. Extend generated-bundle strided add harness/evidence to execute at least
   stride triples `2:3:2` and `3:2:4` while preserving sentinels in skipped
   input/output/tail slots.
5. Run focused build/tests/dry-run and real `ssh rvv` generated-bundle
   correctness, then finish/archive and commit.

## Decision (ADR-lite)

**Context**: Stage 2 is expanding corrected typed RVV coverage. The brief asks
for one independent runtime-stride elementwise path, not broad strided
coverage.

**Decision**: Use the existing strided elementwise provider route family and
strengthen its provider-owned fact surface, especially the ABI/header binding
summary, independent stride roles, target validator consumption, and generated
runtime harness coverage.

**Consequences**: LHS/RHS/output strides remain explicit runtime SSA values
consumed by `tcrv_rvv.strided_load` and `tcrv_rvv.strided_store`; common
EmitC/export still only carries provider payloads and mirrors. This does not
add frontend lowering, new dtype families, source-front-door positive routes,
or extra add/sub/mul matrices.

## Out Of Scope

* Broad strided add/sub/mul matrix, dtype/LMUL clone families, high-level
  Linalg/Vector frontend lowering, source-front-door positive route,
  one-intrinsic wrapper dialect, report-only commit, common EmitC RVV
  semantics, scalar broadcast redo, widening conversion redo, dot-reduce redo,
  macc redo, compare/select redo, reduction, segment/indexed/masked memory
  routes, dashboards, descriptor routes, direct-C/source exporters, or direct
  route-entry positive support.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names,
  fixture names, stride constants, or harness constants as the source of stride
  roles, memory form, op kind, dtype/config, runtime ABI, policy, route
  support, or evidence authority.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-scalar-broadcast-add-artifact-abi-boundary/prd.md`

Repository files inspected while deriving implementation:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-strided-add.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-strided-add.mlir`
* `test/Conversion/EmitC/rvv-generic-stage2-strided-add-materialization.mlir`
* `test/Conversion/EmitC/rvv-generic-stage2-strided-add-negative.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-strided-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-strided-add-dry-run.test`
* `test/Plugin/RVVExtensionPluginTest.cpp`
