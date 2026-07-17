# Stage2 RVV Selected-Body Multi-Family Route Composition Owner

## Goal

Implement one bounded Stage 2 RVV selected-body route that composes multiple
already-landed typed RVV families inside one executable selected body. The
representative is a masked segment2 update: a compare-produced mask, segment2
field/source movement, one typed arithmetic step, inactive-lane passthrough
through a masked segment2 store, shared runtime n/AVL/VL control, provider-owned
route facts, neutral EmitC materialization, generated artifact evidence, and
real `ssh rvv` correctness evidence.

## What I Already Know

- Current HEAD is `8d2d1778 rvv: route computed-mask segment2 store entry owner`.
- There is no `.trellis/.current-task` at session start; this task was created
  from the Hermes direction brief.
- Existing archived work has individually landed computed-mask segment2 load,
  computed-mask segment2 store, plain segment2 deinterleave/interleave, masked
  elementwise, reductions, scalar broadcast, conversions, MAcc, contraction, and
  base memory route-entry paths.
- Current specs require the RVV path to flow through:
  `tcrv.exec` selected RVV variant -> typed low-level `tcrv_rvv` body ->
  RVV plugin-owned selected-body realization/route-family plans/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> generated artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- `tcrv.exec` ABI/runtime roles are not dtype, compute, route, policy, or
  intrinsic authority. Route ids, artifact names, exact intrinsic spelling,
  descriptors, source-front-door markers, and mirror metadata are not authority.
- Relevant existing implementation surfaces include:
  `RVVSelectedBodyRealization.cpp`,
  `RVVEmitCRoutePlanning.h`,
  `RVVEmitCRoutePlanning.cpp`,
  `RVVEmitCRouteProvider.cpp`,
  `RVVOps.td`,
  `RVVDialect.cpp`,
  `RVVConfigContract.cpp`,
  `RVVExtensionPluginTest.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Requirements

- Add exactly one bounded composed route representative centered on a masked
  segment2 update.
- The selected pre-realized typed body must structurally carry:
  compare lhs/rhs, field/source operands, destination operand, runtime `n`, op
  kind, arithmetic kind, predicate kind, segment factor 2, dtype/config, memory
  forms, field roles, mask role/source/form, inactive-lane passthrough policy,
  and RVV policy.
- RVV selected-body realization must consume the pre-realized op into explicit
  typed `tcrv_rvv` body structure before provider route construction. The
  realized body must include ordered same-VL setvl/with_vl, compare-producing
  loads, field loads, arithmetic, compare mask, and masked segment2 store.
- Planning/provider code must validate and carry the composition facts through
  RVV-owned family/provider facts before constructing the route. Provider
  mirrors may be emitted only after route construction.
- Unsupported or stale composition inputs must fail closed with targeted
  diagnostics, including wrong/missing producer dependency, wrong mask producer,
  missing arithmetic, wrong arithmetic op, missing destination passthrough,
  segment factor mismatch, ABI role/order mismatch, memory-form mismatch,
  dtype/config mismatch, runtime AVL/VL mismatch, policy mismatch, stale mirror
  metadata, and missing route/provider plan dependencies.
- Update script evidence only as a direct consumer of the production route. The
  script must not become route, dtype, policy, intrinsic, or executable
  authority.

## Acceptance Criteria

- [ ] Production RVV plugin changes add one composed selected-body route path in
      selected-body realization and route planning/provider boundaries.
- [ ] Focused C++ tests prove the composed pre-realized selected body is consumed
      into explicit typed `tcrv_rvv` body structure and reaches provider-built
      route facts.
- [ ] Focused C++ negative tests prove fail-closed diagnostics for stale or
      malformed composition facts before provider/common EmitC construction.
- [ ] Generated-bundle dry-run evidence shows typed-body/config/runtime/mask/
      segment2/arithmetic facts, provider-supported mirrors only after provider
      route construction, and no descriptor/direct-C/source-export/legacy-i32
      authority.
- [ ] At least one real `ssh rvv` run covers runtime counts including 0, small,
      exact, tail, and stress cases, with mask true/false lanes and sentinels
      proving true composed update behavior, inactive-lane preservation, stream
      order, and tail preservation.
- [ ] Existing non-regression surfaces for computed-mask segment2 store/load,
      plain segment2 interleave/deinterleave, masked elementwise, reduction,
      scalar-broadcast, conversion, runtime-scalar, compare/select, MAcc,
      contraction, and base-memory route entries remain green or any blocker is
      reported with exact command output.
- [ ] Bounded authority scan over touched files finds no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived, or
      legacy-i32-derived route authority.
- [ ] `git diff --check` passes and final `git status --short` is clean after a
      coherent commit, unless an exact blocker prevents completion.

## Out Of Scope

- Broad route-composition matrix.
- Dtype/LMUL clone batches.
- Gather/scatter expansion beyond the chosen representative.
- High-level frontend, Linalg, source-front-door positive route, dashboard, or
  report-only task.
- New one-intrinsic wrapper dialects.
- Common EmitC choosing RVV semantics.
- Weakening existing single-family RVV owners.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/implementation-stack/compiler-stack-contract.md`, and
  `.trellis/spec/guides/index.md`.
- Bounded inspection found existing computed-mask segment2 store/load route
  entries but no operation kind that composes segment2 store with a separate
  arithmetic step under one selected body.
