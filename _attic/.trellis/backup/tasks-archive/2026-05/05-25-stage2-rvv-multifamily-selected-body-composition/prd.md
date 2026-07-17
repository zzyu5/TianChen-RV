# Stage2 RVV Multi-Family Selected-Body Composition Closure

## Context

This task starts from the Hermes direction brief for Stage 2 RVV multi-family
selected-body composition. The prior archived task closed the bounded
`scalar_broadcast_sub` route: a typed `tcrv_rvv.splat` plus elementwise binary
statement plan flowed through RVV-owned planning, common EmitC materialization,
generated bundle artifacts, and real `ssh rvv` correctness evidence.

The current repository also has a bounded plain MAcc route. The next module
closure is not another single-family clone. It must prove that one selected RVV
variant can contain two already route-supported statement families in one typed
low-level `tcrv_rvv` body, and that RVV plugin analysis, statement planning,
provider route construction, generated artifact mirrors, and executable
evidence all preserve typed-body authority.

Current evidence from source inspection:

- Existing migrated statement-plan orchestration intentionally rejects multiple
  unrelated route-family plans unless there is a deliberate aggregate route.
- A MAcc body with a scalar broadcast RHS is currently fail-closed as outside
  the bounded slice.
- Existing typed RVV operations already cover `setvl`, vector load, scalar
  splat, MAcc, store, policy, SEW/LMUL, and runtime AVL/VL facts.

## Module Goal

Make one bounded multi-family selected-body composition route supported and
executable:

```text
selected tcrv.exec RVV variant
  -> typed tcrv_rvv body:
       setvl(runtime n)
       vector load lhs
       scalar broadcast rhs_scalar with the same VL
       vector load accumulator
       macc(lhs_vec, rhs_broadcast_vec, acc_vec)
       store out
  -> RVV plugin-owned legality and selected-body analysis
  -> ordered RVV composition statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> generated RVV artifact mirror
  -> focused ssh rvv correctness evidence
```

The concrete operation name for this task is `scalar_broadcast_macc_add`.

## Required Authority Boundaries

- The selected typed `tcrv_rvv` body must structurally carry operation kind,
  operand roles, RHS scalar broadcast source, accumulator and output bindings,
  element type, SEW, LMUL, policy, runtime AVL/VL, and the producer-to-consumer
  value boundary.
- RVV plugin code owns composition legality, producer/consumer validation,
  statement ordering, intrinsic/type/header mapping, and fail-closed
  diagnostics.
- Common EmitC lowering and target artifact plumbing must stay neutral; they
  may materialize the provider route but must not infer cross-family RVV
  semantics.
- Artifact and emission-plan fields are mirrors only. Any support/result/status
  style fields touched by this task must use explicit mirror labels.
- ABI names, route ids, test names, artifact names, descriptor residue, and
  harness constants must not become route authority.

## Functional Requirements

- Add a bounded `scalar_broadcast_macc_add` selected-body route surface using
  existing generic typed RVV concepts rather than legacy `i32_*` op or
  `!tcrv_rvv.i32m*` authority.
- Validate that the RHS scalar splat is the producer consumed by the MAcc RHS
  operand, not an inferred route-name convention.
- Validate that the lhs load, RHS scalar splat, accumulator load, MAcc, and
  store share the expected runtime VL scope.
- Validate explicit ABI/runtime bindings for `lhs`, `rhs_scalar`, `acc`, `out`,
  and `n`.
- Fail closed before route/artifact authority when producer facts, consumer
  facts, VL scope, type/config facts, ABI bindings, or temporary/output
  ownership are missing or ambiguous.
- Build one ordered RVV composition statement plan:
  `vsetvl`, `vle lhs`, `vmv_v_x rhs_scalar`, `vle acc`, `vmacc`, `vse out`.
- Emit RVV code through the common EmitC route using only provider-built route
  statements and plugin-owned RVV facts.
- Extend generated-bundle tooling only as an evidence harness for this bounded
  route, not as semantic authority.

## Non-Goals

- No high-level Linalg, Vector, StableHLO, or source-front-door lowering.
- No global tiling, autotuning, dashboard, readiness state machine, or broad
  operation matrix.
- No new dtype-prefixed `tcrv_rvv.i32_*` helper ops, `!tcrv_rvv.i32m*` route
  authority, `RVVI32M1*` slices, or exact intrinsic spelling as semantic
  authority.
- No one-op-per-intrinsic wrapper growth.
- No movement of RVV composition semantics into common EmitC or target artifact
  plumbing.

## Acceptance Criteria

- Focused provider/planning tests prove a positive
  `scalar_broadcast_macc_add` typed selected-body route and its ordered
  statement plan.
- Focused fail-closed tests cover representative invalid cases, including at
  least missing RHS scalar broadcast producer, invalid producer/consumer value
  boundary, mismatched or missing VL scope, and wrong or incomplete ABI/runtime
  binding.
- FileCheck evidence shows emitted RVV code order and operands derive from the
  typed body/config/runtime facts:
  `vsetvl`, `vle lhs`, scalar broadcast, `vle acc`, `vmacc`, and `vse out`.
- Generated-bundle dry-run evidence records explicit mirror labels for route,
  provider support, typed body operations, ABI/runtime roles, and statement
  order.
- If executable behavior is claimed, run a real `ssh rvv` generated-bundle
  compile/run with deterministic correctness across at least three runtime
  counts, including non-multiple-tail coverage.
- Run a bounded scan over touched RVV planning/provider/target/script/fixture
  files for stale name-, metadata-, descriptor-, ABI-, harness-, or legacy
  `i32m1`-derived composition authority.
- Run `git diff --check`.
- Run focused tests for changed behavior and `check-tianchenrv`, or document
  the exact blocker if the full check cannot run.
- Finish/archive the Trellis task only after source, tests, evidence, task
  notes, and commit are coherent.
