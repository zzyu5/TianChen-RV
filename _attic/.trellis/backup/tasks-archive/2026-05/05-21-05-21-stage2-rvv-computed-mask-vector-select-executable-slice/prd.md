# Stage2 RVV computed-mask vector select executable slice

## Goal

Implement one bounded Stage2 RVV executable slice for a compare-produced mask feeding vector select:

```text
out[i] = (cmp_lhs[i] < cmp_rhs[i]) ? true_value[i] : false_value[i]
```

The slice must start from a selected/pre-realized `tcrv.exec` RVV boundary, realize a typed RVV compare-mask plus select/merge body under RVV plugin ownership, plan and emit the route from typed body/config/runtime facts, generate an artifact, and prove correctness on the real `ssh rvv` target.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- `git status --short` was clean before task creation.
- Recent HEAD is `27fa65d2 rvv: consolidate contraction selected-body realization`.
- No `.trellis/.current-task` existed before this task was created.
- The previous completed task archived at `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-contraction-selected-body-family/` consolidated contraction selected-body realization through RVV plugin-local family boundaries.
- Existing Stage2 RVV support already has compare/select history, but this task is specifically a computed-mask vector select executable slice with explicit compare operands, predicate, mask provenance, true/false operands, runtime n/AVL, policy, artifact generation, and `ssh rvv` evidence.
- Specs require the authority chain to stay: `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV plugin selected-body realization / legality / route provider -> common EmitC materializer -> target artifact -> `ssh rvv` evidence.

## Requirements

- Selected/pre-realized RVV body structurally carries:
  - compare lhs/rhs vector input roles;
  - compare predicate for one bounded ordinary case, initially signed less-than unless current code shows a better existing predicate spelling;
  - compare-produced mask provenance;
  - true vector operand role;
  - false vector operand role;
  - output vector role;
  - runtime `n`/AVL value use;
  - typed vector config, including element type, SEW, LMUL, and policy;
  - tail/mask inactive-lane policy.
- RVV selected-body realization materializes or validates a realized typed `tcrv_rvv` structure equivalent to:
  - setvl/with-vl for runtime n;
  - load compare lhs/rhs;
  - compare to mask;
  - load true/false vectors;
  - select/merge under the computed mask;
  - store output;
  - preserve tail sentinels outside runtime n.
- RVV route planning derives mask/select/source/result C type facts and route support from typed body/config/runtime facts.
- RVV route provider validates the route and chooses intrinsic leaves only after typed body/config/runtime validation.
- Common EmitC/export remains neutral and does not own RVV compare/select semantics.
- Unsupported or incomplete compare/select structures fail closed with targeted diagnostics.
- Generated bundle dry-run covers representative counts such as 7, 16, and 23.
- Real `ssh rvv` execution proves mixed active/inactive lanes choose true/false values correctly, runtime n is honored, and tail sentinels are preserved.

## Acceptance Criteria

- [x] Positive dialect/pass/FileCheck or equivalent coverage shows selected-body realization for the computed-mask select path.
- [x] Positive route-planning/provider evidence shows route support is derived from typed body/config/runtime facts.
- [x] Positive generated header/artifact evidence exists for the computed-mask select path.
- [x] Negative fail-closed tests cover missing mask provenance, mask/select type mismatch, true/false operand mismatch, missing n/AVL, invalid policy/config, stale route-id authority, and incomplete typed body structure.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` or the appropriate generated-bundle path supports and dry-runs the new computed-mask select op for counts 7, 16, and 23.
- [x] Real `ssh rvv` PASS evidence exists for mixed lane patterns, runtime n, and preserved tail sentinels.
- [x] Active-authority scan shows no reintroduced `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*` route authority, `!tcrv_rvv.i32m*` lowerable authority, source-front-door/source-seed authority, descriptor/direct-C/source-export authority, public exact intrinsic route authority, or common/export RVV semantic authority.
- [x] Focused build/tests for touched RVV dialect/config, selected-body realization, route planning/provider, construction protocol, target support, script, and artifact paths pass.
- [x] `check-tianchenrv`, `git diff --check`, and final clean `git status --short` pass before commit if practical in the environment.

## Non-Goals

- No broad select framework.
- No compare predicate matrix.
- No dtype or LMUL clone batch.
- No broadcast-select, strided-select, gather/scatter/indexed memory, contraction, matmul, Linalg, or frontend lowering.
- No source-front-door positive route.
- No one-intrinsic wrapper dialect.
- No dashboard/report-only/helper-only work as the main result.
- No revival of legacy i32 route authority. An e32/i32 element case is allowed only as an ordinary instance of the generic typed RVV surface, not through `RVVI32M1*`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, artifact names, source-front-door markers, or exact intrinsic spelling as authority.
- No RVV compare/select semantics in common EmitC/export, target metadata, artifact names, route ids, descriptors, ABI strings, tests, or exact intrinsic spellings.

## Technical Notes

- Read first:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-contraction-selected-body-family/`
  - prior compare/select and computed-mask archived tasks under `.trellis/tasks/archive/2026-05/`
- Likely implementation surfaces:
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - existing computed-mask and generated-bundle artifact tests

## Current Phase

Finish / archive. The bounded computed-mask vector select executable slice is implemented, verified, and ready for one coherent commit.
