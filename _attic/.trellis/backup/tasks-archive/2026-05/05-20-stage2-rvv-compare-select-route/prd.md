# Stage2 RVV compare/select selected-body route path

## Goal

Add one bounded Stage2 compare/select selected-body route path on the corrected
generic typed `tcrv_rvv` surface. A selected RVV variant may carry explicit
pre-realized compare/select facts; `RVVSelectedBodyRealization` must consume
those facts into concrete `setvl` / `with_vl` / `load` / `compare` / `select` /
`store` `tcrv_rvv` body structure, and `RVVEmitCRoutePlanning` / the RVV route
provider must then route that realized body through the existing
provider-built `TCRVEmitCLowerableRoute`.

This is not a broad predicate framework. The slice is one i32 SEW32 LMUL m1
compare-eq producing a generic `!tcrv_rvv.mask<i32, "m1">` and one select that
uses that mask over the loaded lhs/rhs vector values.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`, `git status --short`, and `git log --oneline -8` showed a
  clean worktree at `647c0ee6 rvv: extract route provider planning`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief as
  `.trellis/tasks/05-20-stage2-rvv-compare-select-route`.
- `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require the RVV authority
  chain:
  selected `tcrv.exec` RVV variant -> typed/realized low-level `tcrv_rvv` body
  -> RVV plugin legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral common EmitC/export.
- Current generic `tcrv_rvv.compare` and `tcrv_rvv.select` ops already exist
  on the corrected typed surface. Deprecated `tcrv_rvv.i32_cmp_eq`,
  `tcrv_rvv.i32_select`, and `!tcrv_rvv.i32m1_mask` remain parse-only
  Stage1 inventory and must not become positive route authority.
- Current explicit selected-body compare/select artifact coverage reaches
  route planning and target header export through generic `tcrv_rvv.compare`
  and `tcrv_rvv.select`.
- Current pre-realized selected-body realization supports generic add/sub/mul,
  strided add, masked add, macc add, and reduce add, but does not expose a
  pre-realized compare/select body owner.
- `RVVEmitCRoutePlanning` already recognizes realized generic compare/select
  structure and maps operation `cmp_select` to `tcrv_rvv.select`,
  `rvv-generic-cmp-select-emitc-route`, generic mask type/C type, compare
  intrinsic, and select/merge intrinsic.
- `scripts/rvv_generated_bundle_abi_e2e.py` supports explicit selected-body
  `cmp_select`, but pre-realized selected-body expectations do not currently
  include `cmp_select`.

## Requirements

1. Add one pre-realized compare/select selected-body surface in the RVV dialect
   that carries only typed RVV operation/config/memory/mask/runtime SSA facts.
2. Bound the surface to `op_kind = "cmp_select"`, `predicate_kind = "eq"`,
   `memory_form = "vector-rhs-load"`, `mask_source =
   "compare-produced-mask-same-vl-scope"`, `select_layout =
   "select-lhs-when-mask-else-rhs"`, SEW32, LMUL m1, and tail/mask agnostic
   policy.
3. Verify explicit runtime ABI roles for lhs, rhs, out, and runtime n/AVL.
   Missing, wrong, or non-`runtime_abi_value` roles must fail closed before
   realization.
4. Extend `RVVSelectedBodyRealization` so it detects exactly one pre-realized
   compare/select body, rejects mixtures with already realized RVV body ops,
   creates `setvl`, `with_vl`, two generic loads, generic compare, generic
   select, and generic store, then erases the pre-realized marker.
5. Keep `RVVEmitCRouteProvider` as the public route authority and common EmitC
   as a neutral materialization consumer. Do not move RVV semantic decisions
   into common/export code.
6. Preserve existing realized explicit compare/select, masked add, reduce,
   macc, strided, and unit-stride add/sub/mul behavior.
7. Add positive evidence that pre-realized compare/select realizes to generic
   typed compare/select structure and reaches emission-plan / target header
   export through the existing route planning/provider path.
8. Add negative fail-closed coverage for unsupported predicate kind, unsupported
   select layout, unsupported memory form, wrong dtype/config/policy, missing
   or wrong runtime roles, and mixed pre-realized plus realized body structure.
9. Extend generated-bundle dry-run evidence only enough to cover
   `--pre-realized-selected-body --op-kind cmp_select`. Do not claim runtime
   correctness unless real `ssh rvv` evidence is collected.
10. Run active-authority scans to confirm no positive legacy `RVVI32M1`,
    `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
    source-front-door/source-seed, descriptor/direct-C/source-export, artifact
    metadata, or common/export RVV semantic authority is reintroduced.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata match this bounded
      compare/select selected-body realization route task.
- [x] A new generic typed pre-realized compare/select RVV body op exists and
      is verifier-bounded to the accepted op/predicate/mask/select/runtime
      facts.
- [x] `RVVSelectedBodyRealization` consumes the new pre-realized op and creates
      explicit generic `tcrv_rvv.compare` plus `tcrv_rvv.select` under one
      selected `with_vl` body.
- [x] `RVVEmitCRoutePlanning` / provider route support is exercised through
      the realized generic body and reports `cmp_select`,
      `tcrv_rvv.select`, generic mask metadata, compare intrinsic, and
      `rvv-generic-cmp-select-callable-c-abi.v1`.
- [x] Unsupported predicate kind, mask/source/layout facts, memory form,
      dtype/config/policy, runtime ABI roles, and mixed realized/pre-realized
      structure fail closed with focused diagnostics.
- [x] Common EmitC/export code remains neutral; no common RVV semantic branch
      or artifact-name authority is added.
- [x] Generated-bundle dry-run covers pre-realized compare/select.
- [x] Focused build/lit/script checks for touched RVV dialect, selected-body
      realization, route planning/provider, target artifact, and script paths
      pass.
- [x] Active-authority scan confirms old i32/source/front-door/descriptor/common
      authority was not reintroduced.

## Non-Goals

- No broad dtype/LMUL expansion, predicate algebra, compare kinds beyond eq,
  select layouts beyond lhs-when-mask-else-rhs, broadcast compare/select,
  reductions, macc, masked arithmetic, or extra Stage2 classes as side quests.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive route or source-seed evidence mode.
- No one-intrinsic wrapper dialect or new dtype-prefixed op namespace.
- No compatibility wrapper preserving legacy `tcrv_rvv.i32_*` or
  `!tcrv_rvv.i32m*` authority.
- No descriptor-driven computation or descriptor-driven C/source export.
- No runtime, correctness, or performance claim without fresh real `ssh rvv`
  evidence.

## Validation Plan

1. Validate Trellis task context.
2. Build focused targets:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused lit for:
   pre-realized compare/select positive artifact;
   compare/select negative realization/verifier cases;
   existing explicit compare/select artifact;
   existing pre-realized masked/reduce/macc/strided/unit-stride regression
   fixtures.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
5. Run local generated-bundle dry-run for
   `--pre-realized-selected-body --op-kind cmp_select`.
6. Run real `ssh rvv` compare/select evidence only if this round claims
   executable correctness. Otherwise report route-supported/static evidence
   only.
7. Run `git diff --check`.
8. Run active-authority scans over active RVV include/lib/script/test paths.
9. Run broader `check-tianchenrv` if focused failures or shared behavior
   changes justify the broader gate.

## Implementation Results

- Added `tcrv_rvv.typed_compare_select_pre_realized_body` as a bounded generic
  typed pre-realized selected-body op. Its verifier accepts only the current
  `cmp_select` / `eq` / `vector-rhs-load` / same-VL compare mask /
  lhs-when-mask-else-rhs select layout with SEW32 LMUL m1 and agnostic policy.
- Extended `RVVSelectedBodyRealization` to discover the new pre-realized body,
  validate explicit lhs/rhs/out/n runtime ABI roles, reject mixed
  pre-realized and realized RVV body ops, and realize the marker into generic
  `setvl`, `with_vl`, two `load`s, `compare`, `select`, and `store`.
- Reused existing `RVVEmitCRoutePlanning` and `RVVEmitCRouteProvider` generic
  compare/select support. The provider remains the route authority; common
  EmitC/export were not modified.
- Added positive target artifact lit coverage for the pre-realized
  compare/select path and negative selected-body coverage for bad predicate,
  mask source, select layout, memory form, config, policy, runtime ABI role,
  and mixed-body cases.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` and added a dry-run lit
  fixture for `--pre-realized-selected-body --op-kind cmp_select`.
- No runtime/correctness/performance claim was made, so no `ssh rvv` run was
  required for this round.

## Validation Results

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-compare-select-route`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [x] `build/bin/tianchenrv-rvv-dialect-test`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-construction-protocol-common-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] Focused lit for the new pre-realized compare/select target, negative,
      and dry-run tests: 3/3 passed.
- [x] Focused lit regression for explicit compare/select, existing
      pre-realized add/sub/mul/strided/masked/reduce/macc, and related
      provider negatives: 14/14 passed.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Local dry-run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root /tmp/tianchenrv-pre-realized-cmp-select --run-id dryrun --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [x] `git diff --check`
- [x] Added-authority scan over include/lib/script/test diffs found only
      negative prose forbidding descriptor/source-front-door/direct-C authority.
- [x] `cmake --build build --target check-tianchenrv -j2`: 179/179 passed.
