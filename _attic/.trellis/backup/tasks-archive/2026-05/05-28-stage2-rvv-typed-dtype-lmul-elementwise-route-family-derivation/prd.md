# Stage2 RVV Typed Dtype/LMUL Elementwise Route-Family Derivation

## Task Source

Hermes Direction Brief: `Switch: Stage2 RVV typed dtype/LMUL elementwise route-family derivation`.

Repository facts at task creation:

- Root: `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree: clean.
- Initial HEAD: `20929ede rvv: close runtime scalar masked standalone reduction boundary`.
- No `.trellis/.current-task` existed, so this task was created from the brief.

## Problem

RVV Stage2 elementwise arithmetic must move from bounded i32m1-era or witness-local authority toward a generic typed route-family derivation. The provider must derive route facts for non-i32m1 selected bodies from typed `tcrv_rvv` body/config/capability/runtime facts, not from route ids, artifact names, ABI strings, exact intrinsic spelling, generated-bundle allowlists, descriptor residue, common EmitC, or test names.

The current code already has selected-body typed config analysis and existing `i64_add` / `lmul_m2_add` witness fixtures. This round must make the production owner stronger by carrying those typed facts through the elementwise route family plan, verifying provider materialization before `TCRVEmitCLowerableRoute` construction, and making the target artifact consumer fail closed when rebuilt route payload facts or mirror metadata disagree with the selected typed body.

## Goal

Implement one coherent production module for RVV plugin-owned elementwise arithmetic route-family derivation:

```text
selected tcrv.exec RVV variant
  -> typed tcrv_rvv elementwise arithmetic body/config/runtime facts
  -> RVV provider-derived dtype/SEW/LMUL/policy/type/intrinsic/header/ABI facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC
  -> target artifact
  -> generated-bundle evidence and RVV runtime evidence when executable claims are made
```

`i64_add` and `lmul_m2_add` are bounded witnesses of the same generic route-family path. They must not become separate clone-table route authority.

## Requirements

- The RVV elementwise arithmetic route-family plan records and validates the typed config facts that determine element type, signed C type, SEW, LMUL, tail policy, mask policy, vector type spelling, vector C type, VL C type, intrinsic family, headers, runtime ABI roles, and capability requirements.
- Route planning verifies the elementwise plan against `RVVSelectedBodyTypedConfigFacts`, selected body operation/memory form, runtime AVL/VL relation, route operand binding, and capability leaves.
- The route provider fails closed before constructing `TCRVEmitCLowerableRoute` when elementwise materialization facts no longer match the validated typed plan.
- The target artifact consumer rebuilds the provider route and validates elementwise payload facts: required headers, C type mappings, ABI mappings, statement callees, runtime `n` loop bound, selected-body provenance, explicit mirror metadata, and stale non-elementwise metadata.
- Unsupported or inconsistent dtype/SEW/LMUL/policy/body/runtime/capability/ABI/metadata combinations fail closed with targeted diagnostics.
- Common EmitC/export remains neutral. It must only materialize provider-supplied route facts and must not choose RVV semantics.

## Acceptance Criteria

- Production diff includes RVV route planning/provider files and directly affected target artifact boundary code.
- `i64_add` dry-run evidence shows provider-derived `i64`, `SEW=64`, `LMUL=m1`, signed C type, vector C type, e64m1 intrinsic family, runtime AVL/VL facts, explicit provider mirror labels, and mirror-only artifact metadata.
- `lmul_m2_add` dry-run evidence shows provider-derived `i32`, `SEW=32`, `LMUL=m2`, signed C type, vector C type, m2 intrinsic family, runtime AVL/VL facts, explicit provider mirror labels, and mirror-only artifact metadata.
- Fail-closed diagnostics cover mismatched element type, SEW, LMUL, policy, missing capability, stale route id/metadata, wrong ABI roles, wrong AVL/VL relation, and common-EmitC semantic invention or equivalent bounded checks in this owner.
- Focused non-regression keeps existing i32m1 elementwise behavior and the recent selected-boundary reduction/mask paths working.
- RVV executable claims for the witnesses are backed by `ssh rvv` compile/run/correctness for counts including 0, 1, exact, tail, and stress signed payloads, or the exact blocker is reported without claiming runtime correctness.
- `git diff --check` passes.
- `check-tianchenrv` passes, or an exact non-task blocker is recorded with focused checks passing.
- Task state is updated truthfully and the task is finished/archived only after implementation and validation are complete.

## Non-Goals

- No broad dtype/LMUL clone batch.
- No one-intrinsic wrapper dialect.
- No high-level frontend, Linalg, Vector, StableHLO, or per-Linalg-op lowering expansion.
- No conversion/widening/reduction/broadcast route expansion beyond what is needed to keep this elementwise owner coherent.
- No direct-route-entry work.
- No report/dashboard/evidence-only round.
- No route authority from source-front-door, descriptors, route ids, artifact names, ABI strings, exact intrinsic spelling, scripts, or common EmitC.

## Implementation Notes

Likely production owners:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`

Likely focused evidence consumers:

- `test/Target/RVV/pre-realized-selected-body-artifact-i64-add.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-lmul-m2-add.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-i64-add-dry-run.test`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-lmul-m2-add-dry-run.test`
- Focused route-planning/provider/target negative tests as needed.
