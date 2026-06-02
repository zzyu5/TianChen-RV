# Stage2 RVV scalar-broadcast add artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path fail-closed and executable:

```text
selected tcrv.exec RVV variant
  -> typed_binary_pre_realized_body for scalar_broadcast_add
  -> RVV plugin-local realization into setvl / with_vl / lhs load /
     rhs_scalar splat / binary add / result store
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

Route authority must remain the selected typed/realized `tcrv_rvv` body and
RVV plugin/provider facts. Common EmitC and target export may carry provider
payloads and mirrors only; they must not invent scalar broadcast semantics.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV scalar-broadcast add artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no file entries, so the worktree started
  clean.
* Initial `git log --oneline -8` started at
  `4e1dcda3 rvv: validate widening conversion artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require selected typed RVV
  body -> RVV plugin-local realization/provider facts -> common EmitC ->
  target artifact -> real `ssh rvv` evidence for runtime/correctness claims.
* The archived widen conversion task completed the latest provider-owned fact
  -> target validation -> generated bundle -> ssh rvv pattern.
* Existing scalar-broadcast add files already cover the core selected-body
  route shape:
  `typed_binary_pre_realized_body` with `memory_form =
  "rhs-scalar-broadcast"` and `op_kind = "add"`, realized `tcrv_rvv.splat`,
  explicit selected-body `tcrv_rvv.splat` + `tcrv_rvv.binary`, generated-bundle
  dry-run tests, and direct pre-realized fail-closed script tests.
* Current route operand binding summaries for scalar-broadcast elementwise use
  old `runtime-abi-mirror` / `header-mirror` tokens, while the route contract
  requires provider ABI/header participation facts with explicit `abi` and
  `hdr` markers.

## Requirements

* Support exactly the bounded `scalar_broadcast_add` generated artifact route.
* Keep support rooted in selected typed/pre-realized `tcrv_rvv` body, RVV
  plugin-local realization, scalar-broadcast elementwise route-family plan,
  materialization facts, route operand binding facts, route-control provider
  plan, RVV-owned elementwise statement plan, provider-built route facts, and
  target validator consumption.
* Validate runtime ABI order `lhs,rhs_scalar,out,n`.
* Validate dtype/config policy: source/result element type `i32`, SEW `32`,
  LMUL `m1`, tail/mask agnostic policy, VL type `size_t`, vector C type, and
  scalar C type `int32_t`.
* Validate operation and body relation: typed compute op `tcrv_rvv.binary`,
  op kind `add`, memory form `rhs-scalar-broadcast`, lhs vector load,
  `rhs_scalar` ABI value with role `rhs-scalar-value`, RHS scalar splat through
  `tcrv_rvv.splat`, binary add consuming lhs load and splat result, output
  store, and runtime AVL/VL loop control.
* Route operand binding summary must represent every generated
  header/prototype ABI parameter as:
  `<logical>=<role>:<c-name>:abi|...|hdr`.
* Provider route description must carry scalar-broadcast route-family plan,
  route operand binding plan/summary, runtime AVL/VL contract, required
  headers, C type mapping, target leaf/profile facts, and explicit
  `provider_supported_mirror` labeling.
* Target artifact validation must consume provider-owned facts and reject stale
  candidate/provider mirrors before bundle acceptance.
* Unsupported or stale combinations must fail closed with targeted diagnostics
  rather than falling back to names, strings, descriptors, artifact metadata,
  intrinsic spelling authority, vector-RHS residue, or common EmitC semantic
  inference.
* Generated-bundle dry-run evidence must record provider-derived scalar
  broadcast facts, binding order, rhs scalar role/type, route-family plan,
  header/type facts, target validator consumption, provider mirror, runtime
  AVL/VL facts, and harness coverage contract.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,23,257` and RHS scalar values including negative and positive
  cases such as `-37` and `91`, proving scalar value broadcast across all
  active lanes, lhs and rhs signs are handled, output store and tail sentinel
  preservation hold, and runtime `n`/AVL is honored.

## Acceptance Criteria

* [ ] Focused production diff strengthens RVV plugin/provider and target
      validator fact surfaces for `scalar_broadcast_add`; no metadata-only
      closeout.
* [ ] Provider-owned scalar-broadcast binding summary uses explicit
      `abi|...|hdr` participation markers for `lhs`, `rhs_scalar`, `out`, and
      `n`.
* [ ] Target artifact tests fail closed for stale or missing provider mirror,
      route-family plan, ABI order, binding plan/summary, scalar role/type,
      header/type facts, target leaf/profile, and vector-RHS/non-splat residue.
* [ ] REALIZED/PLAN/HEADER checks for both pre-realized and explicit
      scalar-broadcast add fixtures prove setvl/with_vl/load/splat/binary/store
      and provider-derived route facts.
* [ ] Generated-bundle dry-run records provider-derived scalar broadcast facts,
      complete route operand binding, target validator consumption, target
      leaf/profile, headers/types, `provider_supported_mirror`, RHS scalar
      role/type, and runtime AVL/VL facts.
* [ ] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,23,257` and RHS scalars `-37` and `91`.
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
* No broad add/sub/mul broadcast matrix, dtype/LMUL clone batch, source-front
  door positive route, descriptor/direct-C exporter, direct route-entry
  support, common EmitC RVV semantic inference, or unrelated Stage 2 route redo
  is introduced.
* Task status, completion notes, archive, final clean worktree, and commit are
  truthful.

## Technical Approach

1. Inspect existing scalar-broadcast add ODS, selected-body realization,
   route-family planning/provider, target artifact validation,
   generated-bundle script, fixtures, and fail-closed tests.
2. Tighten provider scalar-broadcast elementwise operand binding facts so the
   generated summary carries `abi` and `hdr` participation markers.
3. Tighten target/dry-run/FileCheck expectations and C++ fail-closed coverage
   for stale or missing scalar-broadcast route facts.
4. Run focused build/tests/dry-run and real `ssh rvv` generated-bundle
   correctness, then finish/archive and commit.

## Decision (ADR-lite)

**Context**: Stage 2 is expanding corrected typed RVV coverage. The brief asks
for one scalar-broadcast add ABI boundary, not broad broadcast coverage.

**Decision**: Use the existing scalar-broadcast elementwise provider route
family and strengthen its provider-owned fact surface, especially the ABI/header
binding summary and target validator consumption.

**Consequences**: RHS scalar broadcast remains explicit typed `tcrv_rvv.splat`
dataflow consumed by `tcrv_rvv.binary`; common EmitC/export still only carries
provider payloads and mirrors. This does not add frontend lowering, new dtype
families, source-front-door positive routes, or extra add/sub/mul matrices.

## Out Of Scope

* Broad add/sub/mul broadcast matrix beyond touched shared facts, dtype/LMUL
  clone families, high-level Linalg/Vector frontend lowering, source-front-door
  positive route, one-intrinsic wrapper dialect, report-only commit, common
  EmitC RVV semantics, widening conversion redo, dot-reduce redo, macc redo,
  compare/select redo, reduction, segment/indexed/masked memory routes,
  dashboards, descriptor routes, direct-C/source exporters, or direct
  route-entry positive support.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names,
  fixture names, or harness constants as the source of scalar role, op kind,
  dtype/config, runtime ABI, policy, route support, or evidence authority.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-widen-i32-to-i64-conversion-artifact-abi-boundary/prd.md`

Repository files inspected while deriving implementation:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-scalar-broadcast-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-scalar-broadcast-add-fail-closed.test`
