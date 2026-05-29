# Stage2 RVV base memory-movement artifact ABI statement-plan closure

## Goal

Close the target artifact ABI statement-plan validation boundary for the
selected typed RVV base memory-movement routes:
`strided_load_unit_store`, `unit_load_strided_store`,
`indexed_gather_unit_store`, `indexed_scatter_unit_load`,
`masked_unit_load_store`, and `masked_unit_store`.

The route-supported artifact claim must flow from the selected `tcrv.exec` RVV
variant and typed `tcrv_rvv` base memory body, through RVV provider-derived
route facts and the rebuilt `TCRVEmitCLowerableRoute` statement plan, into
target artifact validation. Tests, generated-bundle scripts, route ids,
artifact names, exact intrinsic spellings, and candidate metadata are evidence
consumers or mirrors only; they must not become authority.

This task is complete only if production validation in
`lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` moves for the base
memory-movement consumer. Test-only evidence is not sufficient.

## Direction Source

- Direction title: `Switch: Stage2 RVV base memory-movement artifact ABI
  statement-plan closure`.
- Module owner: RVV target artifact route-family validation for the base
  memory-movement route family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `115ded51 rvv: close widening macc ABI statement validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker, no spawned,
  parallel-agent, or multi-agent workflow.

## Current Repository Facts

- The previous widening MAcc task closed exact target artifact ABI
  statement-plan validation for `widening_macc_add`; the validator now checks
  exact pre-loop setvl, loop setvl, source/accumulator loads, compute, store,
  runtime AVL/VL, ABI pointer bindings, C types, results, and selected typed
  RVV provenance.
- The earlier vector reduction and widening dot-reduction tasks established
  the same target-side pattern: target artifact validation consumes rebuilt
  provider route statement facts instead of accepting loop payloads by callee
  presence or candidate mirrors.
- `RVVTargetArtifactRouteFamilyValidation.cpp` already has a base-memory
  target artifact consumer with provider-owned route id, memory form,
  provider-supported mirror, route-family plan, binding plan, runtime
  control-plan, runtime ABI order, dtype/policy/setvl/result facts, stale
  non-family rejection, layout facts, headers, type mappings, ABI mappings,
  and candidate mirror checks.
- The remaining production gap is inside
  `validateRVVBaseMemoryMovementRouteStatementPlan`: after basic pre-loop
  setvl, loop bounds, loop setvl, and selected typed RVV provenance checks, it
  still accepts required loop payload statements by
  `routeLoopContainsCallee` / `requireLoopCallee`. That proves a callee string
  appears somewhere, but not exact operands, results, VL use, pointer
  expressions, stride/index/mask bindings, memory form, ABI mapping/order, or
  selected-body provenance per required statement.
- `TargetArtifactExportTest.cpp` has positive base-memory coverage for
  strided load/unit store, indexed gather/unit store, and masked unit
  load/store, plus negative tests for provider mirrors and metadata mirrors.
  It does not yet mutate rebuilt route statement operands/results/pointers/VL
  for base-memory routes, and it does not yet cover the full six-route owner
  set as positive target artifact consumers.

## Requirements

1. Keep provider-built route facts as authority. Candidate metadata may only be
   checked as mirrors after provider route construction.
2. Replace base-memory callee-presence acceptance with exact rebuilt statement
   validation for:
   - full-chunk pre-loop `setvl` from runtime `n`;
   - exactly one runtime AVL/VL loop with provider-derived induction, bounds,
     step, and runtime `n` relation;
   - loop `setvl` from remaining runtime AVL;
   - unit, strided, indexed, and masked load/store statement callees;
   - pointer expressions and ABI C types for source/data/destination/index/mask
     buffers;
   - source and destination stride ABI bindings and stride operands;
   - index load result, index scale operands/results, indexed gather/scatter
     operands, index EEW, and offset-unit derived statement facts;
   - mask predicate/load result, mask C type, passthrough load result, masked
     load/store operands, inactive-lane/passthrough memory contracts, and
     per-iteration VL use;
   - result names and result C types for every required provider-built
     statement;
   - selected typed RVV source provenance on every required pre-loop and loop
     statement.
3. Preserve existing checks for runtime ABI order/roles, route operand binding
   summary, memory form/layout, source/destination stride/index/mask sources,
   target leaf profile, provider-supported mirror labels, route-family plan,
   headers, type mappings, ABI mappings, and stale non-base route-family facts.
4. Add focused C++ target/export route-clone mutation coverage in
   `test/Target/TargetArtifactExportTest.cpp` proving stale rebuilt route
   operands/results/pointers/VL/stride/index/mask/memory-form/ABI/provenance
   fail closed for the six base memory-movement subfamilies, or for a coherent
   subset with a truthful unfinished continuation point.
5. Add positive target/export validation for existing base memory-movement
   routes. If fixture support is missing for any of the six owner routes, add
   the minimal typed selected-body fixture needed to validate the production
   owner path rather than accepting a narrower implicit owner set.
6. Keep common EmitC/export neutral. Common code must not infer RVV semantics,
   dtype/config, ABI roles, mask policy, stride/index policy, memory form,
   route support, statement order, or intrinsic choices.
7. Generated-bundle or target dry-run evidence should exercise affected
   selected-boundary routes and expose base-memory statement-plan facts when
   available.
8. Real `ssh rvv` evidence is required only if this round changes executable,
   runtime, or correctness claims for touched routes, or if a touched route
   needs renewed hardware proof. Otherwise target/export and generated-bundle
   dry-run evidence are sufficient for statement-plan validation.
9. Preserve non-regression for completed widening MAcc and widening
   dot-reduction statement-plan validators.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the
      bounded module, specs, precedent, non-goals, and validation plan.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` validates exact rebuilt
      provider route statement order, operands, result names, result C types,
      runtime AVL/VL facts, pointer expressions, stride/index/mask bindings,
      memory form facts, ABI mapping/order, and selected typed RVV provenance
      for the base memory-movement target artifact consumer.
- [x] The production validator no longer uses
      `routeLoopContainsCallee` / `requireLoopCallee` as acceptance for the
      base-memory statement payload.
- [x] Positive C++ target/export coverage proves artifact acceptance for
      `strided_load_unit_store`, `unit_load_strided_store`,
      `indexed_gather_unit_store`, `indexed_scatter_unit_load`,
      `masked_unit_load_store`, and `masked_unit_store`, or records the exact
      coherent subset and continuation if completion is impossible in this
      round.
- [x] Focused route-clone mutations fail closed for stale or missing pre-loop
      setvl AVL, loop setvl remaining AVL, unit/strided/indexed/masked
      load/store operands/results, source/output pointer expressions, stride
      operands, index load/scale/gather/scatter operands/results, mask
      predicate/passthrough/masked load/store operands/results, per-iteration
      VL operands, runtime ABI order/roles, type mappings, stale candidate
      mirrors, stale route-family mirrors, and selected typed RVV provenance.
- [x] Direct-route-entry-only, artifact-name/script-derived, route-id-derived,
      exact-intrinsic-as-authority, common-EmitC semantic invention,
      descriptor/source-front-door residue, pre-realized-fixture-only, and
      legacy-i32-derived authority remain fail-closed or absent.
- [x] Generated-bundle or target dry-run evidence passes for affected
      selected-boundary base memory routes.
- [x] Completed widening MAcc and widening dot-reduction statement-plan
      validation remains non-regressed through focused evidence.
- [x] Bounded touched-file authority scan finds no new central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, callee-presence-only,
      or legacy-i32-derived executable authority.
- [x] `git diff --check` passes.
- [x] Focused checks pass; `check-tianchenrv` passes, or an exact blocker is
      recorded.
- [x] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Technical Approach

1. Inspect the rebuilt base-memory routes for all six operation kinds and
   encode target-side expected statement sequences from provider description
   facts, not from artifact metadata.
2. Reuse `validateRVVProviderBuiltRouteStep` for callee, operand expression,
   operand C type, result name, and result C type comparisons. Add small
   target-local helpers only where they reduce repeated pointer, stride, index,
   or mask expression assembly.
3. Replace the base-memory `requireLoopCallee` switch with exact sequence
   validators for each subfamily:
   - strided load -> unit store;
   - unit load -> strided store;
   - index load -> index scale -> indexed gather -> unit store;
   - unit load -> index load -> index scale -> indexed scatter;
   - mask load/predicate path -> passthrough load -> masked load -> unit store;
   - source load -> mask predicate path -> masked store.
4. Extend the target/export fixture builder with minimal selected-body
   fixtures for any missing owner subfamilies.
5. Add route-clone negative tests by mutating rebuilt route steps after route
   construction, mirroring the recent vector reduction, widening dot, and
   widening MAcc statement-plan closure style.
6. Run focused target artifact tests, generated-bundle dry-runs for affected
   routes, non-regression for widening MAcc/widening dot, authority scan,
   `git diff --check`, and `check-tianchenrv` if feasible.

## Out of Scope

- Do not start new RVV operation coverage outside the six base memory-movement
  owner routes.
- Do not add new dtype/LMUL clone batches, high-level Linalg/Vector/StableHLO
  frontend lowering, selected-body realization rewrites, provider route
  expansion, generated-bundle feature expansion, dashboards, broad smoke
  matrices, or evidence-only tasks.
- Do not rework completed widening MAcc or widening dot-reduction
  statement-plan validation except for focused non-regression.
- Do not move RVV semantics into common EmitC/export code.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
- Do not make descriptors, ABI strings, artifact names, route ids, test names,
  metadata, exact intrinsic spellings, common EmitC code, source-front-door
  markers, direct route entries, pre-realized fixtures, callee presence, or
  legacy i32 helper names route authority.

## Validation Plan

1. `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. `cmake --build build --target tcrv-opt tcrv-translate -j2` if
   generated-bundle scripts need fresh tools.
4. Generated-bundle selected-boundary dry-run for affected base
   memory-movement routes.
5. Focused non-regression for `widening_macc_add` and the four widening
   dot-reduction statement-plan routes.
6. Bounded touched-file authority scan.
7. `git diff --check`
8. `cmake --build build --target check-tianchenrv -j2`, or exact blocker.

## Completion Evidence

- Production owner changed:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
  `validateRVVBaseMemoryMovementRouteStatementPlan` now consumes rebuilt
  provider-built route statement facts for all six base memory-movement
  subfamilies and no longer accepts their loop payload by
  `routeLoopContainsCallee` / `requireLoopCallee`.
- Exact facts now validated include runtime ABI count/order/roles, pre-loop
  `setvl(n)`, loop `setvl(n - i)`, loop bound/step/VL facts, pointer
  expressions, operand C types, result names, result C types, source/dest
  stride operands, index load/scale operands and results, indexed gather/scatter
  base/value operands, mask load/predicate/passthrough facts, masked load/store
  operands, unit store VL, memory form, and selected typed RVV provenance.
- Positive C++ target/export coverage now includes all six owner routes:
  `strided_load_unit_store`, `unit_load_strided_store`,
  `indexed_gather_unit_store`, `indexed_scatter_unit_load`,
  `masked_unit_load_store`, and `masked_unit_store`.
- Focused route-clone negative coverage mutates stale pre-loop AVL, loop AVL,
  unit/strided pointer operands, stride operands, indexed pointer/scale/base/
  result/value operands, mask pointer/result/passthrough/masked operands, store
  VL, source result names, and selected source provenance.
- Checks run:
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - `cmake --build build --target tcrv-opt tcrv-translate -j2`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_base_memory_statement_plan_closure --run-id pre-realized-base-memory-dry --overwrite --op-kind strided_load_unit_store --op-kind unit_load_strided_store --op-kind indexed_gather_unit_store --op-kind indexed_scatter_unit_load --op-kind masked_unit_load_store --op-kind masked_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --stride-bytes 8 --stride-bytes 12 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_base_memory_statement_plan_closure --run-id widening-macc-dot-nonregression-dry --overwrite --op-kind widening_macc_add --op-kind widening_dot_reduce_add --op-kind strided_input_widening_dot_reduce_add --op-kind computed_masked_widening_dot_reduce_add --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  - touched production added-line authority scan: no forbidden authority hits.
    Test-file hits were intentional `metadata_derived_*` stale route-clone
    negative inputs only.
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`: 459/459 passed.
- Self-repair performed: the first widening MAcc/dot non-regression dry-run
  command incorrectly passed `--stride-bytes` to non-strided route kinds; the
  command failed before validation, was corrected by removing the unsupported
  option, and the corrected non-regression dry-run passed.
- `ssh rvv` was not rerun because this task changes target artifact
  statement-plan validation only and does not change executable/runtime/
  correctness claims for the affected routes.
- Spec update judgment: no `.trellis/spec/` update is needed. The work applies
  the existing RVV provider-built statement-plan and mirror-only metadata
  contract to the base memory target artifact consumer without adding new
  route API fields, command signatures, dialect semantics, or durable
  conventions.

## Spec Update Judgment

Initial expectation: no `.trellis/spec/` update is required. The existing RVV
plugin, core dialect, EmitC route, and testing specs already require
provider-derived selected-body route facts, target artifact route-family
validators, mirror-only metadata, common EmitC neutrality, and evidence-bound
runtime claims. This task applies those existing contracts to the
base-memory target artifact validator.
