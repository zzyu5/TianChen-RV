# Stage2 RVV Typed Runtime-Scalar Compare-Masked Standalone Reduction Route-Family Derivation

## Task Source

Hermes Direction Brief:
`Switch: Stage2 RVV typed runtime-scalar compare-masked standalone reduction route-family derivation`.

Repository facts at task creation:

- Root: `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree: clean.
- Initial HEAD:
  `0fe6e74f rvv: derive typed runtime scalar cmp masked memory route facts`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief.
- The immediately preceding archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-typed-runtime-scalar-cmp-masked-memory-route-family-derivation`
  completed typed runtime-scalar compare-masked memory derivation for baseline
  SEW32 LMUL m1, SEW64 LMUL m1, and SEW32 LMUL m2 witnesses.
- The older archived standalone-reduction boundary task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-boundary`
  proved baseline selected-boundary executable evidence for
  `runtime_scalar_cmp_masked_standalone_reduce_add`, but did not close typed
  i64 or LMUL m2 derivation for the standalone reduction family.

## Problem

`runtime_scalar_cmp_masked_standalone_reduce_add` is the adjacent
runtime-scalar computed-mask reduction path after the typed masked-memory
family. It imports a compare LHS vector input, a runtime scalar threshold,
source payload, scalar accumulator seed, scalar output, and runtime `n`; it then
realizes a compare-produced mask, excludes inactive lanes from the reduction,
and carries a scalar accumulator/result across runtime VL chunks.

The current bottleneck is not to add another reduction family. The bottleneck
is to make this bounded family derive all executable facts from typed
`tcrv_rvv` body/config/runtime structure instead of baseline i32m1 or
exact-intrinsic assumptions. The derived facts include:

- scalar C type, vector C type, mask C type, accumulator/result type, SEW,
  LMUL, policy, runtime ABI order, and runtime scalar threshold ABI;
- compare predicate, mask provenance, inactive/neutral policy, source payload
  load, accumulator seed role, output role, reduction leaf, statement leaves,
  provider mirrors, and target artifact ABI;
- generated-bundle evidence with `route_entry_realization = false` and
  `pre_realized_body_consumed = true`.

## Goal

Implement one coherent production module for bounded typed
`runtime_scalar_cmp_masked_standalone_reduce_add` route-family derivation:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized runtime-scalar compare masked standalone reduction body
  -> RVV plugin-local selected-body realization
  -> realized setvl/with_vl/load/splat/compare/mask/neutralize/reduce/store body
  -> standalone/computed-mask reduction route-family facts from typed body/config/runtime facts
  -> route materialization + math operand-binding + route-control provider facts
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC
  -> generated target artifact and harness
  -> ssh rvv evidence for typed witnesses
```

Representative positive witnesses:

- baseline SEW32 LMUL m1 non-regression;
- SEW64 LMUL m1 typed witness.

If a typed witness is blocked by a real legality or production contract
constraint, this task must fail closed with a targeted diagnostic and record the
exact blocker instead of claiming support.

This round treats SEW32 LMUL m2 as a targeted fail-closed witness rather than a
positive executable witness. RVV reduction intrinsic typing for an m2 source
uses an m1 scalar reduction accumulator/result vector
(`vredsum_vs_i32m2_i32m1`), while the current bounded standalone reduction
route materializer exposes one vector result/store channel tied to the selected
body config. Claiming executable LMUL m2 support without first adding a
separate scalar-result vector channel would be an invalid route claim.

## Requirements

- Preserve `tcrv.exec` as ABI/envelope authority only. `cmp_lhs`,
  `rhs_scalar`, `src`, `acc`, `out`, and `n` roles may be declared by
  `mem_window` / `runtime_param`, but RVV dtype/config/reduction semantics must
  come from typed/realized `tcrv_rvv` structure.
- Runtime ABI order must be
  `cmp_lhs,rhs_scalar,src,acc,out,n`. The runtime scalar threshold C type,
  source payload type, accumulator/result type, and output type must derive
  from the typed element/config facts.
- Selected-body realization must consume the pre-realized body and derive
  vector/mask/scalar types, `setvl`, compare LHS load, runtime scalar splat,
  compare mask, source payload load, inactive-lane neutralization, standalone
  reduction, scalar seed/result layout, output store, runtime AVL/VL, source
  provenance, SEW, LMUL, and policy from structural facts.
- Route planning/provider must derive provider facts from the realized typed
  body and same-analysis config/runtime facts: scalar/vector/mask C types,
  reduction intrinsic leaf, compare/splat/reduction/store statement leaves,
  route-control plan, math operand-binding facts, provider-supported mirrors,
  runtime ABI mirrors, and target header/prototype mirrors.
- Unsupported dtype/config/policy/reduction or inconsistent runtime bindings
  must fail closed before executable route construction. No fallback to i32m1,
  route id, artifact name, ABI string, script-derived facts, common EmitC
  choices, source-front-door residue, descriptor residue, or exact intrinsic
  spelling is allowed.
- Direct pre-realized route-entry must remain fail-closed for
  `runtime_scalar_cmp_masked_standalone_reduce_add`; this task must not reopen a
  direct route-entry shortcut.

## Acceptance Criteria

- Production diff includes movement in the owning boundaries as needed:
  RVV config/type contract, selected-body realization, route planning/provider
  facts, generated-bundle ABI, target artifact boundary, or focused tests. Tests
  alone are sufficient only if inspection proves production already derives the
  typed facts and the missing piece is evidence/tooling.
- Typed selected-boundary witnesses cover SEW64 LMUL m1 for
  `runtime_scalar_cmp_masked_standalone_reduce_add`; SEW32 LMUL m2 is rejected
  with an exact fail-closed diagnostic explaining the missing separate LMUL m1
  scalar reduction accumulator/result channel.
- Baseline SEW32 LMUL m1 remains selected-boundary executable.
- Generated-bundle dry-run validates `route_entry_realization = false`,
  `pre_realized_body_consumed = true`, typed scalar/vector/mask/reduction facts,
  selected ABI order, route-control mirrors, math operand-binding mirrors,
  provider metadata mirrors, target header/prototype ABI, statement leaves, and
  emitted RVV C/C++ leaves.
- Direct pre-realized route-entry remains fail-closed with selected-boundary
  diagnostics for the standalone reduction family.
- Focused FileCheck or C++ coverage proves typed facts are derived/consumed
  before common EmitC, including at least scalar/vector/mask/reduction
  accumulator facts and provider route consumption.
- Real `ssh rvv` generated-bundle compile/run/correctness passes for counts
  `0`, `1`, an exact-vector count, a tail count, and a stress count with signed
  data and distinct runtime scalar thresholds.
- Runtime harness coverage checks mixed active/inactive lanes, inactive nonzero
  payload exclusion, active positive/negative payloads, scalar seed/result
  behavior, output sentinel preservation, and tail preservation.
- Focused non-regression passes for the just-completed typed masked-memory
  family and existing runtime-scalar compare/select family.
- Any touched Python tooling passes `python3 -m py_compile` and the relevant
  script self-test when behavior changes.
- `git diff --check` passes.
- `cmake --build build --target check-tianchenrv -j2` passes, or an exact
  blocker is recorded after focused checks pass.
- Bounded authority scan over touched production/test/script files shows no new
  central ad hoc, name-derived, metadata-derived, descriptor-derived,
  ABI-string-derived, script-derived, artifact-name-derived,
  common-EmitC-derived, source-front-door-derived, route-id-derived,
  exact-intrinsic-derived, direct-route-entry-only,
  pre-realized-fixture-only, or legacy-i32-derived route authority.

## Definition Of Done

- Trellis PRD/context are truthful and current.
- The bounded standalone reduction route family derives typed facts across the
  accepted witness set, or exact fail-closed blockers are recorded.
- Direct route-entry remains fail-closed.
- Focused checks, ssh rvv evidence, non-regression, authority scan, and final
  status are recorded.
- The task is finished/archived only after acceptance is met.
- One coherent commit records the completed round.

## Non-Goals

- No standalone_reduce_min/max or computed_mask_standalone_reduce_min/max
  expansion.
- No segment2, compare/select redesign, masked-memory rework, MAcc/widening-dot
  follow-up, source-front-door route, high-level Linalg/frontend lowering,
  one-intrinsic wrapper dialect, dashboard/report work, broad smoke matrix, or
  evidence-only substitution for a production mismatch.
- No direct pre-realized route-entry positive support.
- No dtype-prefixed helper op family, descriptor compute path, route-id
  authority, artifact-name authority, exact-intrinsic authority, or common EmitC
  semantic selection.

## Technical Notes

Read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/index.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-typed-runtime-scalar-cmp-masked-memory-route-family-derivation/prd.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-boundary/prd.md`

Likely production owners:

- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`

Likely focused evidence:

- existing runtime-scalar compare-masked standalone-reduce target tests;
- generated-bundle dry-runs for baseline and SEW64 LMUL m1;
- a fail-closed SEW32 LMUL m2 verifier/provider probe documenting the scalar
  reduction accumulator/result channel blocker;
- direct pre-realized route-entry fail-closed probe;
- non-regression dry-runs for typed masked-memory and runtime-scalar
  compare/select families.
