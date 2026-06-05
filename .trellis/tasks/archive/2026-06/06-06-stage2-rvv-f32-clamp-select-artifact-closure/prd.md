# Stage2 RVV f32 clamp/select selected-body artifact closure

## Goal

Make one RVV f32 clamp/select route truthful from selected `tcrv.exec` RVV variant and typed `tcrv_rvv` selected body through RVV plugin-owned route planning, common EmitC materialization, and target artifact validation. The task should close the concrete f32 clamp/select route boundary rather than add construction-protocol metadata or route-string authority.

## What I already know

- The repository is on `main`, worktree was clean at task creation, and HEAD is `1fc5ead6 plugin: repair construction protocol route boundary`.
- The prior task repaired the shared construction protocol route boundary, including f32 clamp/select role-step and runtime ABI mirrors plus Template provider route value/reset contract.
- Current Stage 2 authority chain is selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin selected-body realization/route provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
- `tcrv.exec` binds ABI/runtime roles and selected variants only; it must not infer clamp/select compute semantics, dtype, SEW/LMUL, predicate, VL placement, result layout, intrinsic spelling, or route support.
- RVV plugin code must validate or derive clamp/select operation kind, predicate/min/max facts, f32 dtype/config, operand binding, runtime ABI order, result layout, selected route family, and header/intrinsic mirrors from typed body/config/runtime facts.

## Assumptions

- A focused f32 clamp/select fixture or nearby selected-body route already exists and can be repaired instead of inventing a new broad frontend route.
- If artifact closure reaches generated bundle form, ssh RVV correctness is in scope; otherwise the bounded finish point is route-supported plus target-validation closure with generated-bundle ssh RVV as the next continuation.

## Requirements

- Establish, or prove already existing, production provider/route/target support for one f32 clamp/select route from typed selected body to target artifact.
- Keep compute authority in RVV plugin-owned typed selected body and route planning, not common EmitC, artifact names, route ids, ABI strings, test names, manifests, or construction-protocol metadata.
- Ensure stale or missing operation kind, predicate/min/max facts, f32 dtype/config, operand binding, runtime ABI order, result layout, selected route family, and intrinsic/header mirrors fail closed with targeted diagnostics.
- Preserve the repaired construction-protocol route value/reset contract.
- Keep edits scoped to RVV plugin/provider/target route closure, tests, task context, and any necessary spec note.

## Acceptance Criteria

- [x] Positive route evidence shows f32 clamp/select support from typed selected body through `TCRVEmitCLowerableRoute`.
- [x] Positive artifact/header evidence shows the target/export side mirrors provider-derived f32 clamp/select route facts.
- [x] Negative checks cover stale or missing operation kind, predicate/min/max facts, f32 dtype/config, operand binding, runtime ABI order, result layout, selected route family, and intrinsic/header mirrors.
- [x] Common EmitC/export remains neutral and does not invent clamp/select semantics.
- [x] Construction-protocol regression covering the route value/reset contract still passes.
- [x] Focused `tcrv-opt`/`tcrv-translate`, RVV plugin, target artifact/export, and generated-bundle checks run as applicable.
- [x] `git diff --check`, `git diff --cached --check`, and a bounded old-authority scan over touched files/added diff lines pass.
- [x] If ssh RVV correctness is claimed, evidence includes counts 0, 1, VL-boundary, tail, and multi-chunk sizes with scalar oracle and tail preservation.

## Out of Scope

- Broad f32/dequant route matrix.
- High-level Linalg, Vector, StableHLO, or source frontend work.
- Per-Linalg route authority or one-intrinsic wrapper dialect work.
- Template/Toy feature work.
- Construction-protocol-only metadata completion.
- Performance, autotuning, tuning databases, or broad readiness dashboards.
- Common EmitC invention of RVV clamp/select semantics.
- Route-string, artifact-name, ABI-string, test-name, descriptor, or manifest authority.

## Technical Notes

- Read first: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
- Read prior task: `.trellis/tasks/archive/2026-06/06-06-extension-plugin-construction-protocol-common-repair/`.
- Likely code owners: `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h`, `lib/Conversion/EmitC/TCRVEmitCLowerableInterface.cpp`, `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`, `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`, `lib/Plugin/RVV/EmitC/`, `lib/Plugin/RVV/Construction/`, `lib/Target/RVV/`, `scripts/rvv_generated_bundle_abi_e2e.py`, and nearby f32 clamp/select fixtures.

## Completion Evidence

Completed as a bounded f32 clamp/select selected-body artifact closure.

Implementation completed:

- Target artifact candidate validation now requires compare/select producer
  metadata mirrors for `tcrv_rvv.lower_bound_role`,
  `tcrv_rvv.upper_bound_role`, `tcrv_rvv.lower_bound_c_type`,
  `tcrv_rvv.upper_bound_c_type`, `tcrv_rvv.bound_order`, and
  `tcrv_rvv.clamp_relation` to match provider-derived selected-body route
  facts. Empty provider facts also fail closed if stale clamp mirrors appear on
  non-clamp compare/select producers.
- The f32 clamp/select target fixture now asserts exact provider/header mirrors
  for route operand binding summary, computed-mask select route-family plan,
  mask producer source, mask/tail route-family plan, required headers, C type
  mapping, lower/upper bound roles and C types, bound order, clamp relation, and
  secondary compare predicate.
- The fixture now contains stale-mirror negative checks for provider support,
  runtime ABI order, lower bound role, bound order, route operand binding
  summary, route-family plan, upper bound C type, secondary compare predicate,
  and required header declarations.
- `.trellis/spec/testing/mlir-testing-contract.md` records the executable f32
  clamp/select target/header mirror-key contract and expected stale mirror
  negative coverage.

Checks run:

- `cmake --build build --target tcrv-opt tcrv-translate`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact`
- Direct stale-mirror checks equivalent to the fixture `RUN:` lines for
  provider mirror, runtime ABI order, lower bound role, bound order, route
  operand binding summary, route-family plan, upper bound C type, secondary
  compare predicate, and required header declarations.
- `build/bin/tcrv-opt test/Dialect/RVV/pre-realized-f32-clamp-select-negative.mlir --split-input-file --verify-diagnostics`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind f32_clamp_select --dry-run`
  produced `rvv_generated_bundle_abi_e2e: dry_run_success` under
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T175557Z`.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `build/bin/tianchenrv-construction-protocol-common-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 -m lit -sv ...` was attempted for focused f32 clamp/select and
  construction-protocol tests but unavailable locally:
  `/usr/bin/python3: No module named lit`. `FileCheck` was also not present in
  the local build or PATH, so direct command and diagnostic-string equivalents
  were used for local verification.
- `git diff --check`
- `git diff --cached --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-f32-clamp-select-artifact-closure`
- Bounded old-authority scan over added diff lines found no new `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, exact
  `__riscv_*_i32m1`, source-front-door/source-artifact, descriptor, or route-id
  authority.

ssh RVV correctness:

- Not claimed in this round. The route reached generated bundle dry-run and
  target-validation closure; real `ssh rvv` correctness is the exact
  continuation if executable evidence is requested.
