# Stage2 RVV plain compare-select route-control provider-plan integration

## Goal

Make the existing plain compare/select RVV route family consume the shared
`RVVSelectedBodyRouteControlProviderPlan` before compare/select statement-plan
construction. This must route the already-supported plain compare/select path
through the same typed config, runtime AVL/VL, policy, selected capability,
runtime ABI, materialization, operand-binding, and same-analysis boundary that
recently became authoritative for ordinary elementwise, scalar-broadcast
elementwise, base memory, standalone reduction, and scalar MAcc paths.

This is a production-path migration of an existing route family. It must not
add new compare/select operation kinds, new dtype/LMUL clone batches,
computed-mask coverage, frontend lowering, descriptor/source-front-door
positive paths, dashboards, or evidence-only fixture copying.

## Source Direction

- Direction title: `Stage2 RVV plain compare-select route-control provider-plan integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by the
  existing plain compare/select route-family/provider path.
- Initial HEAD: `7a55ce6c rvv: consume route control plan in scalar broadcast elementwise`.
- Initial worktree: clean.
- No `.trellis/.current-task` existed at session start, so this task records
  the Hermes brief as the Trellis work item before source edits.

## What Is Already Known

- The shared route-control provider-plan boundary is defined as RVV plugin-local
  provider input after route-family verification and materialization facts, and
  before statement-plan construction.
- Current required consumers in the spec include ordinary elementwise
  arithmetic, scalar-broadcast elementwise arithmetic, base memory movement,
  standalone reduction, and scalar-broadcast MAcc.
- Recent commits completed route-control consumption for scalar MAcc,
  ordinary elementwise arithmetic, and scalar-broadcast elementwise arithmetic.
- The plain compare/select family already has family/provider, operand-binding,
  statement-plan, generated-bundle predicate-boundary, and ssh rvv evidence
  history. This task must reuse that route surface and move it through the
  control boundary.
- The compare/select statement plan currently derives from verified family
  plans, materialization facts, and RVV-owned elementwise/select operand-binding
  facts. This task adds route-control consumption for the plain sub-family
  before that statement plan is returned.

## Requirements

1. Production C++ planning/provider behavior must make only the existing plain
   compare/select route-control consumer use the shared
   `RVVSelectedBodyRouteControlProviderPlan` before compare/select statement
   construction.
2. The route-control provider plan for plain compare/select must validate or
   expose:
   - same-analysis typed config facts;
   - same-analysis selected target capability facts;
   - the owning runtime AVL/VL control plan;
   - runtime ABI order and runtime `n` binding;
   - SEW, LMUL, tail policy, and mask policy;
   - selected legality/capability provider mirrors;
   - route description mirror consistency.
3. The compare/select statement-plan boundary must fail closed before route
   construction when plain compare/select requires route-control facts but any
   of the control plan, materialization facts, family facts, operand-binding
   facts, runtime ABI facts, policy facts, selected capability facts, or mirrors
   are stale, missing, or mismatched.
4. The provider route must still attach provider-built compare/select statement
   plans to `TCRVEmitCLowerableRoute`; common EmitC and target artifact code
   must remain mirror/materialization consumers only.
5. Any emitted object/header/artifact metadata touched by this task must use
   explicit mirror labels. Metadata must remain mirror-only after provider route
   construction.
6. The route-control boundary spec must be updated to list plain compare/select
   as a current consumer once the production path and tests prove it.

## Acceptance Criteria

- [x] `RVVSelectedBodyRouteControlProviderPlan` includes a plain compare/select
      consumer flag or equivalent structural marker.
- [x] Plain compare/select route-control plan construction joins the verified
      plain compare/select family plan, same-analysis materialization facts,
      typed config facts, selected target capability facts, runtime AVL/VL
      control plan, and route description mirrors.
- [x] `getRVVSelectedBodyCompareSelectRouteStatementPlan(...)` consumes the
      route-control provider plan for plain compare/select before building the
      setvl/load/compare/select/store statement plan.
- [x] Non-plain compare/select sub-families remain unchanged unless they already
      have their own current boundaries; this task must not silently migrate
      computed-mask, runtime-scalar, or dual-mask compare/select coverage.
- [x] Focused C++ tests prove positive plain compare/select route-control
      consumption and provider/statement-plan attachment.
- [x] Focused negative C++ tests fail closed for representative stale or
      missing control dependencies, including:
      - stale materialization facts or stale same-analysis ownership;
      - missing runtime AVL/VL control plan;
      - wrong runtime AVL role or runtime ABI mirror mismatch;
      - policy mismatch;
      - unsupported selected capability/config;
      - missing plain compare/select family or materialization plan;
      - stale plain compare/select operand binding.
- [x] Focused lit/FileCheck or generated-header checks cover explicit
      mirror-only labels if route metadata/header mirrors change.
- [x] Existing explicit or pre-realized plain compare/select generated-bundle
      dry-run evidence is rerun if emitted output changes. Existing ssh rvv
      evidence may be reused if executable generated code is unchanged; rerun
      affected plain compare/select ssh rvv correctness if output or runtime
      behavior changes.
- [x] Run `git diff --check`.
- [x] Run focused RVV plugin tests and focused lit/FileCheck for touched
      compare/select route-control behavior.
- [x] Run `check-tianchenrv`, or report the exact blocker.
- [x] Run a bounded authority scan over touched planning/provider/test/spec or
      target/script files for name-, route-id-, metadata-, descriptor-,
      ABI-string-, script-, artifact-, common-EmitC-, source-front-door-, or
      legacy-i32-derived AVL/VL/policy authority.

## Out Of Scope

- New compare/select operation kinds or predicates.
- Computed-mask select, runtime-scalar compare/select, dual compare-mask-and
  select migration onto route-control unless directly needed for isolation
  tests.
- Conversion, contraction, reduction expansion, memory expansion, dtype/LMUL
  clone batches, high-level frontend lowering, source-front-door positive
  routes, dashboards, broad smoke matrices, or evidence-only fixture copying.
- Treating runtime counts, route ids, metadata fields, manifests, artifact
  names, ABI strings, descriptors, scripts, tests, common EmitC, target artifact
  code, or legacy i32 spellings as AVL/VL, policy, dtype, predicate, or compute
  authority.

## Implementation Notes

- Follow the existing route-control consumer patterns from base memory,
  standalone reduction, scalar MAcc, ordinary elementwise arithmetic, and
  scalar-broadcast elementwise arithmetic.
- Keep the change local to RVV planning/provider and focused tests unless code
  inspection proves selected-body realization or target mirror code must change.
- Prefer adding a plain compare/select consumer flag and passing the control
  plan into the existing compare/select statement-plan builder over adding a
  parallel statement path.
- The statement plan should not repair or infer SEW/LMUL, tail/mask policy,
  runtime AVL, ABI order, or selected capability from names or mirrors.

## Evidence Plan

1. Inspect current planning/provider/test code:
   - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
   - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
   - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
   - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
   - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
   - `test/Plugin/RVV/RVVExtensionPluginTest.cpp`
   - existing plain compare/select lit fixtures.
2. Implement the smallest production-path migration that makes plain
   compare/select a route-control consumer.
3. Add focused positive and fail-closed tests in the existing RVV plugin test
   style.
4. Update the RVV plugin spec after the production path and tests establish the
   new consumer.
5. Run focused build/tests, authority scan, `git diff --check`, and
   `check-tianchenrv`.

## Completion Evidence

- Production C++ changed:
  - `RVVSelectedBodyRouteControlProviderPlan` now exposes
    `controlsPlainCompareSelect`.
  - `getRVVSelectedBodyRouteControlProviderPlan(...)` now treats
    `CmpSelect` as a route-control consumer and requires same-analysis
    `plainCompareSelectRouteFamilyPlan` materialization facts before exposing
    runtime AVL/VL, typed config, selected target capability, policy, runtime
    ABI, and mirror facts.
  - `getRVVSelectedBodyCompareSelectRouteStatementPlan(...)` now requires the
    plain compare/select route-control provider plan before building the
    setvl/load/compare/select/store statement sequence.
- Focused C++ coverage:
  - positive plain compare/select route-control joins typed config, selected
    target capability facts, runtime AVL/VL control plan, and mirror facts;
  - provider route still attaches the RVV-owned compare/select statement plan;
  - fail-closed tests cover missing plain family plan, stale runtime AVL role,
    policy mismatch, unsupported selected capability/config, stale
    same-analysis materialization facts, runtime ABI mirror mismatch, and
    stale plain compare/select operand binding.
- Spec updated:
  - `.trellis/spec/extension-plugins/rvv-plugin.md` now lists plain
    compare/select as an explicit route-control provider-plan consumer and
    documents the statement-plan control dependency.
- Generated evidence:
  - dry-run generated-bundle for pre-realized `cmp_select` passed under
    `artifacts/tmp/stage2_plain_compare_select_route_control/pre-realized-cmp-select-route-control-dry`.
  - evidence includes `runtime_avl_vl_boundary`,
    `tcrv_rvv.runtime_control_plan`, `tcrv_rvv.runtime_avl_source`,
    `tcrv_rvv.runtime_abi_order`,
    `tcrv_rvv.target_capability_provider_mirror`,
    `tcrv_rvv.target_capability_legality_mirror`,
    `tcrv_rvv.plain_compare_select_route_family_plan`, and
    `tcrv_rvv.provider_supported_mirror`.
- Real RVV evidence:
  - `ssh rvv` generated-bundle ABI/e2e for pre-realized `cmp_select` passed
    counts `7,16,23` with both predicate-true and predicate-false lanes.
- Checks run:
  - `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-plain-compare-select-route-control-plan`
  - `git diff --check`
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  - `./build/bin/tianchenrv-rvv-extension-plugin-test`
  - `cmake --build build --target tcrv-opt tcrv-translate -j2`
  - `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='cmp-select|compare-select'`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_plain_compare_select_route_control --run-id pre-realized-cmp-select-route-control-dry --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_plain_compare_select_route_control --run-id pre-realized-cmp-select-ssh-rvv --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --connect-timeout 10 --timeout 120`
  - `cmake --build build --target check-tianchenrv -j2`
  - Added-line authority scan over touched planning/provider/test/spec/task
    files.
