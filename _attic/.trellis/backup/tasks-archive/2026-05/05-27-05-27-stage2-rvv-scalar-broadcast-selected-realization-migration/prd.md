# Stage2 RVV Scalar Broadcast Selected-Body Realization Production Migration

## Goal

Move `scalar_broadcast_add` generated artifact authority back to the public
selected lowering-boundary realization producer path, and demote the
direct pre-realized route-entry shortcut for this consumer.

The bounded production chain for this round is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv scalar_broadcast_add body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv setvl / with_vl / load / splat / binary / store body
  -> scalar-broadcast elementwise route-family provider facts
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

The generated artifact path for `scalar_broadcast_add` must go through
`--tcrv-materialize-selected-lowering-boundaries`; it must not claim support
from direct route-entry acceptance, pre-realized fixture names, route ids,
artifact names, script options, ABI strings, exact intrinsic spelling, common
EmitC behavior, descriptor residue, source-front-door metadata, or legacy i32
helper authority.

## Direction Source

- Direction title: `Redirect: Stage2 RVV selected-body realization production migration for scalar_broadcast_add`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `46f24f81 rvv: prove runtime memory selected-body realization`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- The selected-boundary `scalar_broadcast_add` fixture already exists at
  `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`.
  Its positive RUN lines materialize selected lowering boundaries before
  emission plans and target artifact export.
- `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body
  --op-kind scalar_broadcast_add` already uses the selected-boundary producer
  path and records `materializer = tcrv-materialize-selected-lowering-boundaries`.
- A later direct route-entry proof made `scalar_broadcast_add` eligible for
  `--direct-pre-realized-route-entry` and added a positive script lit test.
  This shortcut is now stale for this consumer under the redirect brief.
- The RVV selected-body realization owner registry currently treats the
  elementwise/compare-select family as route-entry eligible for typed binary
  pre-realized bodies, including RHS scalar-broadcast memory form.
- Existing provider planning already validates the scalar-broadcast elementwise
  route-family plan, route-control plan, runtime ABI order
  `lhs,rhs_scalar,out,n`, route operand bindings, RHS scalar splat leaf,
  provider-supported mirror, headers, and C type mapping.
- Common EmitC and target artifact code must remain consumers of provider-built
  facts only; they must not infer scalar-broadcast semantics.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling and script
   guardrails.
2. Keep the selected-boundary `scalar_broadcast_add` producer path positive.
   It must realize the typed pre-realized body into `setvl`, `with_vl`, unit
   `load`, RHS scalar `splat`, `binary add`, and `store` before provider route
   facts are collected.
3. Demote or delete the direct pre-realized route-entry shortcut for
   `scalar_broadcast_add`. The shortcut must fail closed with a targeted
   diagnostic or script error instead of producing a generated artifact.
4. Preserve direct route-entry behavior for unrelated supported route-entry
   families unless they are outside this task's bounded consumer.
5. Preserve typed facts for operation kind, scalar/broadcast RHS binding,
   memory roles, runtime `n`/AVL/VL values, dtype/config/policy, setvl
   placement, loop relation, required RVV capability, provider route facts,
   and artifact ABI order.
6. Unsupported or inconsistent selected-boundary input must fail closed before
   provider/common route construction where exposed by current hooks: missing
   runtime param, missing mem_window/runtime ABI value, wrong RHS scalar role,
   wrong output binding, dtype/config/policy mismatch, wrong AVL/VL relation,
   wrong setvl placement, missing capability, stale route id, stale mirror
   metadata, artifact-name/script-derived authority, exact-intrinsic-as-
   authority, direct-route-entry-only authority, and common-EmitC semantic
   invention.
7. Do not add broad route coverage, a new route family, dtype/LMUL clone set,
   high-level frontend lowering, one-intrinsic wrapper dialect, selected-body
   framework rewrite, descriptor/source-front-door positive route, dashboard,
   or proof-only helper task.

## Acceptance Criteria

- [x] Production code demotes `scalar_broadcast_add` direct pre-realized
      route-entry eligibility while keeping the selected-boundary producer path
      positive.
- [x] C++ tests prove `scalar_broadcast_add` is not route-entry eligible for
      direct route-entry realization, and that the selected-boundary producer
      still realizes it before provider route construction.
- [x] C++ or lit tests prove provider route facts for selected-boundary
      `scalar_broadcast_add` consume realized typed body facts and include the
      scalar-broadcast elementwise route-family plan, runtime control plan,
      route operand binding plan, RHS scalar splat leaf, ABI order, and
      provider-supported mirror.
- [x] Focused fail-closed coverage rejects missing or wrong RHS scalar binding,
      wrong output/runtime binding, stale scalar-broadcast route-family mirror,
      stale ABI order, and direct-route-entry-only authority.
- [x] The direct pre-realized route-entry script path for
      `scalar_broadcast_add` fails closed or is removed from positive lit
      coverage.
- [x] Generated-bundle dry-run for `scalar_broadcast_add` passes through the
      selected lowering-boundary path and records no direct route-entry
      materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts including `0`,
      `1`, exact-VL, tail, and stress cases with at least two RHS scalar
      values.
- [x] Focused non-regression covers completed selected-body producer paths for
      `computed_mask_select` and `runtime_i32_splat_store`, plus adjacent
      route-family paths changed or risked by this demotion.
- [x] A bounded touched-file authority scan finds no new executable or route
      claim depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [x] Trellis status, journal/archive, and the final commit are truthful.

## Validation Plan

1. Validate Trellis context.
2. Update the RVV selected-body realization route-entry predicate so
   `scalar_broadcast_add` uses the selected-boundary producer path only.
3. Update generated-bundle tooling and script tests so
   `--direct-pre-realized-route-entry --op-kind scalar_broadcast_add` no longer
   succeeds.
4. Keep or strengthen selected-boundary lit coverage for
   `pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`.
5. Run focused C++ plugin tests and focused lit/script tests.
6. Run generated-bundle selected-boundary dry-run for `scalar_broadcast_add`.
7. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,17,257`
   and RHS scalars `-37,19`.
8. Run focused non-regression for `computed_mask_select`,
   `runtime_i32_splat_store`, and direct route-entry families whose predicate
   remains supported.
9. Run authority scan, `git diff --check`, task validation, and
   `check-tianchenrv`.

## Out Of Scope

- Stage2 coverage expansion beyond `scalar_broadcast_add`.
- New scalar-broadcast operation families, dtype/LMUL clone batches, or
  additional intrinsic wrappers.
- High-level Linalg/Vector/StableHLO frontend work.
- Source-front-door, descriptor-driven compute, direct C/source-export paths,
  or legacy `RVVI32M1` / `rvv-i32m1` compatibility.
- Dashboard/report-only/prompt-only work as the main achievement.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Relevant archived tasks read:
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-selected-body-realization-producer`,
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-runtime-memory-realization`,
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-scalar-broadcast-elementwise-route-entry-owner`,
  and `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-scalar-broadcast-add-executable-slice`.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and focused target/script tests.

## Implementation Result

- Demoted `scalar_broadcast_add` from direct pre-realized route-entry
  eligibility in `RVVSelectedBodyRealization.cpp` by rejecting
  RHS scalar-broadcast typed binary bodies at the direct route-entry predicate.
- Preserved the selected-boundary producer path for `scalar_broadcast_add`:
  `tcrv.exec` selected variant -> typed pre-realized `tcrv_rvv` body ->
  selected lowering-boundary materializer -> realized `setvl` / `with_vl` /
  `load` / `splat` / `binary` / `store` -> scalar-broadcast provider facts.
- Updated the generated-bundle evidence tool so
  `--direct-pre-realized-route-entry --op-kind scalar_broadcast_add` fails
  closed before bundle generation, while normal
  `--pre-realized-selected-body --op-kind scalar_broadcast_add` still uses
  `--tcrv-materialize-selected-lowering-boundaries`.
- Updated C++ plugin coverage so `rvv_pre_route_scalar_broadcast_add` is
  selected-boundary producer eligible, not route-entry eligible, and direct
  route-entry realization fails with the route-entry family diagnostic.
- Replaced the positive direct route-entry lit test with a fail-closed test,
  and strengthened the selected-boundary dry-run test to require
  `route_entry_realization = false` plus the selected-body realization
  producer marker.

## Validation Evidence

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-27-05-27-stage2-rvv-scalar-broadcast-selected-realization-migration`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Selected-boundary scalar-broadcast dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_scalar_broadcast_selected_realization_migration --run-id selected-boundary-scalar-broadcast-add-dry --overwrite --op-kind scalar_broadcast_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 19 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- Direct route-entry fail-closed command:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --artifact-root artifacts/tmp/stage2_rvv_scalar_broadcast_selected_realization_migration --run-id direct-route-entry-scalar-broadcast-add-fail --overwrite --op-kind scalar_broadcast_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 19 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  failed as expected with `--direct-pre-realized-route-entry is bounded to ... got ['scalar_broadcast_add']`.
- Focused lit:
  `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-scalar-broadcast-add|direct-pre-realized-scalar-broadcast-add'`
  passed 2/2 selected tests.
- Wider scalar-broadcast lit:
  `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'scalar-broadcast-add'`
  passed 6/6 selected tests.
- Focused selected-body producer non-regression:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_scalar_broadcast_selected_realization_migration --run-id selected-boundary-producer-nonregression-dry --overwrite --op-kind computed_mask_select --op-kind runtime_i32_splat_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 19 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- Real `ssh rvv` scalar-broadcast evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_scalar_broadcast_selected_realization_migration --run-id selected-boundary-scalar-broadcast-add-ssh --overwrite --op-kind scalar_broadcast_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 19 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --connect-timeout 8 --timeout 120`
  reported `PASS op=scalar_broadcast_add counts=0,1,16,17,257 rhs_scalars=-37,19`.
- Production-only authority scan over `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  and `scripts/rvv_generated_bundle_abi_e2e.py` found no new forbidden
  authority additions. Full touched-file scan only matched negative
  `implicit-check-not` assertions in lit.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 391/391 on the
  final single-instance rerun.

Self-repair notes:

- One selected-boundary dry-run was accidentally duplicated with the same
  artifact run-id, causing a transient empty materialized module in one racing
  process. The single rerun passed.
- Two `check-tianchenrv` instances were accidentally launched concurrently.
  One passed 391/391; the racing one failed with `Text file busy` and artifact
  write conflicts. A final single-instance rerun passed 391/391 and is the
  recorded result.

## Spec Update Judgment

No `.trellis/spec/**` update was needed. This task implements the existing
RVV selected-body realization and route-entry boundary: scalar-broadcast
elementwise generated artifacts must flow through the selected lowering-boundary
producer unless a future explicit owner reintroduces route-entry support with a
new contract. No new command/API signature, cross-layer contract, dialect
syntax, provider payload field, or validation rule was introduced beyond the
bounded demotion and regression tests.
