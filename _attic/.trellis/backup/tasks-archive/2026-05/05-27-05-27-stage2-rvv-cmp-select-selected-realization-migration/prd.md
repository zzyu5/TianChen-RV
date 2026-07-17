# Stage2 RVV cmp_select compare/select Selected-Body Realization Migration

## Goal

Move the generated artifact path for the plain `cmp_select` compare/select
family behind the public RVV selected lowering-boundary realization producer
and remove its active direct pre-realized route-entry shortcut authority.

The production path for this task must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv plain compare/select body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv body preserving lhs/rhs runtime ABI imports,
     predicate relation, true/false selected value behavior, output binding,
     i32 element dtype, SEW32, LMUL m1, policy, runtime n/AVL/VL, setvl and
     with_vl placement, selected requires, and ABI order
  -> plain compare-select route-family facts
  -> route materialization facts
  -> elementwise/select operand-binding facts
  -> route-control provider plan
  -> migrated compare/select statement-plan owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

Direct pre-realized route-entry support for plain `cmp_select` must be demoted
or deleted. Because the generated-bundle script currently groups
`cmp_select` and `cmp_select_sle` under the same `is_cmp_select` predicate,
`cmp_select_sle` is in scope for the same direct-route-entry demotion and
selected-boundary evidence. The generated artifact must not be accepted from a
direct route-entry shortcut, route id, artifact name, script option, ABI
string, exact intrinsic spelling, common EmitC behavior, descriptor residue,
source-front-door metadata, or legacy i32 helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV cmp_select compare/select
  selected-body realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `524d62fd rvv: demote strided load route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no subagents or parallel agent
  workflows.

## Current Repository Facts

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` has an
  `elementwise/compare-select` selected-body realization owner. It recognizes
  `TypedCompareSelectPreRealizedBodyOp` in the selected-body realization
  cluster and realizes it into `setvl`, `with_vl`, `load`, `compare`,
  `select`, and `store`.
- The same owner currently marks plain compare/select as direct route-entry
  eligible through `isPreRealizedRVVCompareSelectRouteEntryOp` and
  `isPreRealizedRVVElementwiseCompareSelectRouteEntryOp`.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently lets direct
  pre-realized route-entry mode accept `self.is_cmp_select`, and
  `is_cmp_select` covers both `cmp_select` and `cmp_select_sle`.
- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-route-entry-dry-run.test`
  is the remaining positive direct pre-realized route-entry dry-run for
  `cmp_select`; it asserts `materializer:
  rvv-route-entry-selected-body-realization` and `route_entry_realization:
  true`.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run.test`
  already uses the selected lowering-boundary path for `cmp_select`; this task
  should strengthen that path and make it the generated-bundle authority.
- Existing compare/select route-provider code requires RVV-owned plain
  compare-select operand-binding facts before provider statement construction,
  and emits compare/select statements from provider facts into
  `TCRVEmitCLowerableRoute`.
- Archived task
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-strided-load-unit-store-selected-realization/`
  completed the same direct-route-entry demotion pattern for a selected-body
  family: selected-boundary realization stayed positive, direct mode failed
  closed, generated-bundle dry-run recorded `route_entry_realization: false`,
  real `ssh rvv` passed, and `check-tianchenrv` passed.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling, harnesses,
   and generated-bundle guardrails.
2. Keep plain `cmp_select` / `cmp_select_sle` under the RVV
   `elementwise/compare-select` selected-body realization owner. The selected
   lowering-boundary producer must realize the typed pre-realized body before
   provider route facts are collected.
3. Demote or delete direct pre-realized route-entry support for plain
   `cmp_select` and `cmp_select_sle`. Direct shortcut requests must fail
   closed before bundle generation or provider route construction.
4. Preserve typed facts for operation kind `cmp_select`, predicate relation
   `eq` or `sle`, lhs/rhs runtime ABI imports, true-lane `lhs` selected value,
   false-lane `rhs` selected value, output runtime ABI binding, i32 element
   type, SEW32 LMUL m1, tail/mask policy, runtime `n`/AVL/VL values,
   `setvl`/`with_vl` placement, selected variant `requires`, plain
   compare-select route-family plan, elementwise/select operand-binding facts,
   route-control facts, provider route facts, and artifact ABI order
   `lhs,rhs,out,n`.
5. Fail closed for unsupported or inconsistent selected-boundary input,
   including missing runtime_param, missing mem_window/runtime ABI value, wrong
   predicate relation, wrong operand binding, missing true/false lane evidence,
   wrong dtype/config/policy, wrong AVL/VL relation, wrong setvl placement,
   stale route id or mirror metadata, direct-route-entry-only authority,
   artifact-name/script-derived authority, exact-intrinsic-as-authority, and
   common-EmitC semantic invention.
6. Reuse existing selected-body realization, plain compare-select route-family,
   route materialization, route-control provider, elementwise/select
   operand-binding, migrated statement-plan, target artifact, and
   generated-bundle boundaries where they already express the required facts.
   Do not introduce a new central route table or common EmitC semantic branch.
7. Do not start standalone reduction, segment2 deinterleave/interleave,
   computed-masked segment2 load/store/update, computed-mask select,
   runtime-scalar compare/select, new memory movement cleanup, high-level
   Linalg/frontend lowering, new dtype/LMUL clone batches, one-intrinsic
   wrapper dialects, selected-body realization framework rewrites,
   dashboard/report work, broad smoke matrices, or evidence-only tasks.
8. Do not add proof-only tests for completed strided load/unit-store,
   conversion, widening dot, MAcc, base-memory, or computed-mask memory paths
   except as bounded non-regression needed by touched files.

## Acceptance Criteria

- [x] Production code no longer treats plain `cmp_select` /
      `cmp_select_sle` as direct pre-realized route-entry eligible, while the
      `elementwise/compare-select` selected-body realization owner still
      realizes them through the public selected lowering-boundary producer.
- [x] C++ tests prove a typed pre-realized plain compare/select body belongs
      to the `elementwise/compare-select` realization owner, is not a direct
      route-entry consumer, and fails closed when
      `realizePreRealizedRVVRouteEntrySelectedBody` is used as a direct
      route-entry shortcut.
- [x] C++ or lit tests prove the selected-boundary path consumes realized
      typed facts, including `setvl`, `with_vl`, `load`, `compare`, `select`,
      `store`, lhs/rhs/output/runtime ABI roles, runtime `n`/AVL/VL, plain
      compare-select family plan, elementwise/select operand-binding facts,
      route-control facts, ABI order, provider-supported mirrors, predicate
      relation, select layout, and true/false operand roles.
- [x] The generated-bundle script rejects
      `--direct-pre-realized-route-entry --op-kind cmp_select` and
      `--op-kind cmp_select_sle` before route-entry materialization or bundle
      generation.
- [x] Generated-bundle dry-runs for pre-realized `cmp_select` and
      `cmp_select_sle` pass through
      `--tcrv-materialize-selected-lowering-boundaries`, record
      `route_entry_realization: false`, record selected-body producer
      evidence, record realized compare/select/provider facts, and record no
      direct route-entry materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts `0`, `1`, `7`,
      `16`, `23`, and `257` with predicate-true and predicate-false lanes for
      the selected compare/select path.
- [x] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including at minimum the RVV extension plugin
      smoke test and focused lit/script checks for selected-boundary
      compare/select.
- [x] A bounded touched-file authority scan finds no new executable or route
      claim depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [x] Trellis status, journal/archive, and final commit are truthful.

## Completion Results

- Production movement:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` no longer registers
  `TypedCompareSelectPreRealizedBodyOp` as direct route-entry eligible. Plain
  compare/select remains under the `elementwise/compare-select` selected-body
  realization owner and is consumed by the selected lowering-boundary producer.
- Generated-bundle boundary:
  `scripts/rvv_generated_bundle_abi_e2e.py` no longer treats
  `self.is_cmp_select` as `supports_direct_pre_realized_route_entry`. Direct
  `cmp_select` and `cmp_select_sle` requests fail closed before bundle
  generation with a bounded direct-route-entry allowlist diagnostic.
- Selected-boundary facts:
  focused C++ and lit coverage now checks producer-realized `setvl`,
  `with_vl`, two loads, compare, select, store, predicate `eq`/`sle`, true
  value `lhs`, false value `rhs`, output `out`, ABI order `lhs,rhs,out,n`,
  `rvv-plain-compare-select-route-family-plan.v1`, provider mirrors, and
  `route_entry_realization: false`.
- Target and conversion consumers:
  target artifact/header tests for pre-realized `cmp_select` now run selected
  lowering-boundary materialization before emission planning/export, and the
  conversion EmitC test for pre-realized plain compare/select does the same
  before building a `TCRVEmitCLowerableRoute`.
- Dry-run evidence:
  pre-realized selected-body generated-bundle dry-run passed for
  `cmp_select` and `cmp_select_sle` with counts `0,1,7,16,23,257`, recording
  `materializer: tcrv-materialize-selected-lowering-boundaries` and
  `route_entry_realization: false`.
- Real RVV evidence:
  `ssh rvv` generated-bundle execution passed for `cmp_select` and
  `cmp_select_sle` over counts `0,1,7,16,23,257`. Predicate coverage included
  true and false lanes for the multi-lane cases:
  `cmp_select` true/false counts were `2/5`, `4/12`, `6/17`, and `65/192`
  for counts `7`, `16`, `23`, and `257`; `cmp_select_sle` true/false counts
  were `5/2`, `10/6`, `14/9`, and `155/102`.
- Checks:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
  generated-bundle self-test, direct fail-closed manual checks for
  `cmp_select` and `cmp_select_sle`, selected-boundary dry-run,
  `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test -j2`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`, focused lit for
  compare/select and conversion route materialization, `git diff --check`,
  task validation, bounded authority scans, and
  `cmake --build build --target check-tianchenrv -j2` passed with
  `400/400` tests.
- Self-repair:
  full `check-tianchenrv` first exposed an obsolete conversion test that still
  expected direct route-entry materialization for pre-realized plain
  `cmp_select`. The test was split so direct arithmetic add remains direct,
  while plain compare/select now has its own selected-boundary route
  materialization fixture.
- Spec-update judgment:
  no `.trellis/spec/**` edit was needed. This round enforced existing
  Stage2 selected-body realization, provider-authority, common EmitC
  neutrality, and testing contracts without adding a new durable rule.

## Validation Plan

1. Validate Trellis task context.
2. Inspect current selected-body realization, route planning/provider,
   target-support bundle, generated-bundle script, focused plugin tests, and
   target/script tests for active direct compare/select route-entry authority.
3. Narrow plain compare/select route-entry eligibility so `cmp_select` and
   `cmp_select_sle` remain selected-boundary capable but are not direct
   route-entry eligible.
4. Update generated-bundle tooling/lit coverage so direct pre-realized
   route-entry mode fails closed for both `cmp_select` and `cmp_select_sle`.
5. Strengthen selected-boundary C++/lit/script coverage so evidence explicitly
   checks `route_entry_realization: false`, selected-body producer evidence,
   compare/select predicate facts, provider mirrors, true/false operand roles,
   and artifact ABI order.
6. Run focused C++ plugin tests and focused lit/script tests.
7. Run generated-bundle selected-boundary dry-runs for `cmp_select` and
   `cmp_select_sle`.
8. Run real `ssh rvv` generated-bundle evidence for counts `0,1,7,16,23,257`
   through the selected lowering-boundary producer path.
9. Run focused adjacent non-regression for explicit selected-body
   compare/select and unrelated remaining direct route-entry cases that share
   the owner registry boundary.
10. Run authority scan, `git diff --check`, task validation, and
    `check-tianchenrv` or record the exact blocker.

## Out Of Scope

- Standalone reduction, segment2 memory, computed-mask select,
  runtime-scalar compare/select, computed-mask segment2 load/store/update,
  additional dtype or LMUL clone batches, high-level
  Linalg/Vector/StableHLO frontend work, source-front-door construction,
  descriptor-driven compute, direct C/source export paths, one-intrinsic
  wrapper dialects, selected-body framework rewrite, dashboards, prompt-only
  work, and legacy `RVVI32M1` / `rvv-i32m1` compatibility.
- Additional proof-only work for completed strided load/unit-store, widening
  conversion, widening dot, widening MAcc, computed-mask widening dot, strided
  widening dot, computed-mask strided widening dot, or base-memory paths
  except as bounded non-regression.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Relevant predecessor tasks read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-strided-load-unit-store-selected-realization/prd.md`,
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-direct-cmp-select-executable-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-plain-cmp-select-route-family-ownership/prd.md`,
  `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-cmp-select-executable-abi/prd.md`.
- Initial production owner files inspected:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-route-entry-dry-run.test`.
