# Stage2 RVV Plugin-Local Selected-Body Realization Closure

## Context

This task starts from the Hermes direction brief for Stage 2 RVV
plugin-local selected-body realization. The previous archived task closed the
bounded `scalar_broadcast_macc_add` composition route for an already realized
typed `tcrv_rvv` body: explicit `setvl`, `with_vl`, `load`, `splat`,
accumulator `load`, `macc`, and `store` flow through RVV-owned route analysis,
provider validation, common EmitC materialization, generated artifact mirrors,
and real `ssh rvv` correctness evidence.

Current source inspection shows the remaining gap:

- `scalar_broadcast_macc_add` is already route-supported for explicit realized
  typed bodies.
- `TypedMAccPreRealizedBodyOp` only accepts `memory_form = "vector-rhs-load"`
  in selected-body realization today.
- The production route-entry realization bridge currently admits
  elementwise/compare-select and base-memory pre-realized families, but not the
  MAcc route family.
- The provider correctly fails closed if it sees a pre-realized body before
  selected-body realization runs.

The bounded module goal is therefore to make one already route-supported
multi-family path start from a typed pre-realized selected body and be realized
inside the RVV plugin before provider/common route construction.

## Module Goal

Support one bounded pre-realized selected-body path:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv body:
       typed_macc_pre_realized_body(
         lhs, rhs_scalar, acc, out, n,
         op_kind = scalar_broadcast_macc_add,
         memory_form = rhs-scalar-broadcast-macc,
         accumulator/result/config/policy facts)
  -> RVV plugin-local selected-body realization:
       setvl(runtime n)
       with_vl
       load lhs
       splat rhs_scalar
       load accumulator
       macc(lhs_vec, rhs_broadcast_vec, acc_vec)
       store out
  -> existing RVV provider route facts and TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact mirrors
  -> focused generated-bundle and ssh rvv evidence
```

The concrete operation name for this task is
`scalar_broadcast_macc_add`.

## Required Authority Boundaries

- The pre-realized typed body must structurally carry operation kind, memory
  form, lhs, RHS scalar value, accumulator, output, runtime n/AVL, element
  type/config, policy, accumulator layout, and result layout facts.
- RVV plugin realization must create only legal execution structure:
  `setvl`, `with_vl`, `load`, `splat`, `load`, `macc`, and `store`.
- Realization must not change computation semantics, dtype semantics,
  parameter roles, required capabilities, dispatch/fallback behavior, or
  runtime `n`/AVL values.
- The route provider must consume the realized body through the existing
  scalar-broadcast MAcc route analysis and statement plan. It must not infer
  missing structure from route ids, ABI names, artifact names, manifests,
  descriptors, helper names, or harness constants.
- Common EmitC/export must remain neutral and only materialize provider-built
  route payloads.
- Artifact and emission-plan metadata are mirrors only. Support-like fields
  touched by this task must use explicit mirror labels such as
  `provider_supported_mirror`.

## Functional Requirements

- Allow a bounded scalar-broadcast MAcc shape through
  `TypedMAccPreRealizedBodyOp` using `memory_form =
  "rhs-scalar-broadcast-macc"` and `op_kind =
  "scalar_broadcast_macc_add"`.
- Validate explicit runtime ABI roles for `lhs`, `rhs_scalar`, `acc`, `out`,
  and `n`, including RHS scalar value ownership/type.
- Validate SEW32, LMUL m1, agnostic policy, accumulator role/layout, result
  layout, and selected variant `requires` metadata before mutation.
- Realize RHS scalar into `tcrv_rvv.splat` and require `tcrv_rvv.macc` to
  consume that splat result as its RHS operand.
- Reuse the existing explicit scalar-broadcast MAcc provider route facts,
  operand-binding summary, statement plan, and EmitC materialization path
  after realization.
- Extend the production route-entry selected-body realization bridge so a
  direct pre-realized `scalar_broadcast_macc_add` route reaches provider
  construction without first requiring an explicit external
  `--tcrv-materialize-selected-lowering-boundaries` pass.
- Keep unsupported or malformed MAcc pre-realized bodies fail-closed before
  route/artifact authority.

## Non-Goals

- No high-level Linalg, Vector, StableHLO, TOSA, or source-front-door lowering.
- No broad dtype/LMUL/op matrix, global tiling, autotuning, dashboard, or
  readiness state machine.
- No new dtype-prefixed `tcrv_rvv.i32_*` helper ops, `!tcrv_rvv.i32m*`
  authority, `RVVI32M1*` slices, or `rvv-i32m1` route ids.
- No one-op-per-intrinsic wrapper growth.
- No movement of selected-body realization semantics into common EmitC or
  target artifact plumbing.
- No reliance on prompt edits, reports, metadata, manifests, route labels, or
  helper-only changes as the main achievement.

## Acceptance Criteria

- A focused pre-realized `scalar_broadcast_macc_add` fixture proves that
  `typed_macc_pre_realized_body` is consumed and replaced by realized
  `setvl`/`with_vl`/`load`/`splat`/`load`/`macc`/`store` structure.
- Focused provider/EmitC route-entry coverage proves a direct pre-realized
  selected body is realized by the RVV production route-entry bridge before
  route facts are collected, without requiring
  `--tcrv-materialize-selected-lowering-boundaries`.
- Focused fail-closed tests cover at least missing or inconsistent RHS scalar
  producer/role/type, wrong memory form/op kind, invalid ABI binding, and
  already-realized/pre-realized body mixing before route construction.
- FileCheck evidence shows emitted RVV code order and operands derive from
  typed body/config/runtime facts: `vsetvl`, `vle lhs`, scalar broadcast,
  `vle acc`, `vmacc`, and `vse out`.
- Generated-bundle dry-run evidence records pre-realized input mode,
  route-entry realization, pre-realized body consumption, explicit mirror
  labels, runtime ABI order, typed compute op, route operand bindings, and no
  descriptor/direct-C/source-export residue.
- If executable behavior is claimed, run real `ssh rvv` generated-bundle
  compile/run evidence with deterministic correctness across at least three
  runtime counts, including a tail case; include count 0 if the harness
  supports it.
- Run a bounded scan over touched RVV realization/planning/provider/target/
  script/fixture files for name-, metadata-, descriptor-, ABI-, harness-, or
  legacy i32-derived realization authority.
- Run `git diff --check`.
- Run focused lit/script/C++ checks for changed behavior and
  `check-tianchenrv`, or document the exact blocker if the full check cannot
  run.
- Finish/archive the Trellis task only after source, tests, evidence, task
  status, and commit are coherent.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/variant-pipeline/index.md`, and
  `.trellis/spec/implementation-stack/index.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-multifamily-selected-body-composition/prd.md`.
- Likely implementation files:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and focused fixtures under
  `test/Target/RVV`, `test/Conversion/EmitC`, `test/Scripts`, and
  `test/Plugin`.
- Existing explicit realized evidence exists in
  `test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir`,
  `test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-macc-materialization.mlir`,
  and `test/Scripts/rvv-generated-bundle-abi-e2e-scalar-broadcast-macc-add-dry-run.test`.
- Existing pre-realized MAcc evidence exists for vector RHS in
  `test/Target/RVV/pre-realized-selected-body-artifact-macc-add.mlir`; this
  task adds the scalar-broadcast MAcc counterpart rather than another explicit
  realized route.
