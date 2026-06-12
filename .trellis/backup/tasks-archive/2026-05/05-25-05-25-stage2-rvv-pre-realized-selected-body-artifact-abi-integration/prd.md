# Stage2 RVV pre-realized selected-body artifact ABI integration

## Goal

Prove that one bounded, statement-plan-backed pre-realized RVV selected-body
cluster reaches the production target artifact and generated-bundle ABI path
through the RVV plugin-owned route-entry realization bridge. The selected
`tcrv.exec` RVV variant must start with a typed pre-realized `tcrv_rvv` body,
realize before route facts are collected, feed the existing RVV facts,
operand-binding facts, statement plans, provider-built
`TCRVEmitCLowerableRoute`, common EmitC, and RVV target artifact generation,
then produce bounded ABI evidence.

This round is artifact/ABI integration evidence for existing route-entry
behavior. It is not new RVV coverage.

## Direction Source

- Direction title: `Stage2 RVV pre-realized selected-body artifact ABI integration`.
- Module owner: RVV plugin-owned pre-realized selected-body route-entry path
  carried through common EmitC and RVV target artifact generation for one
  bounded statement-plan-backed kernel cluster.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `666d294a rvv: add family-neutral selected-body route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` requires the RVV-first chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
- `.trellis/spec/extension-plugins/rvv-plugin.md` defines the route-entry
  realization bridge for supported pre-realized elementwise/compare-select and
  base-memory selected bodies. The bridge must realize before route facts are
  collected and must fail closed for unsupported route-entry families.
- `.trellis/spec/lowering-runtime/emitc-route.md` says target artifact export
  consumes materialized EmitC plus provider route mirrors. Common EmitC/target
  export must not infer RVV dtype, operation kind, memory form, schedule,
  intrinsic spelling, ABI roles, or route support from metadata.
- The archived task
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-selected-body-family-neutral-route-entry-bridge/prd.md`
  completed the route-entry bridge and proved direct route/emission entry
  materialization for compare/select and one base-memory form.
- Current code evidence:
  - `RVVExtensionPlugin::buildVariantEmissionPlan(...)` and
    `RVVExtensionPlugin::buildVariantEmitCLowerableRoute(...)` already call
    `getOrRealizeRVVSelectedBodyRouteBoundary(...)`.
  - `RVVSelectedBodyRealization.cpp` exposes
    `realizePreRealizedRVVRouteEntrySelectedBody(...)` and currently supports
    elementwise/compare-select plus base memory route-entry selected bodies.
  - `test/Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir` and
    `test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir`
    still run the explicit `--tcrv-materialize-selected-lowering-boundaries`
    pass before artifact emission.
  - `scripts/rvv_generated_bundle_abi_e2e.py` still records pre-realized
    selected-body bundle generation as using the public selected-boundary
    materializer, even for route-entry-supported fixtures.

## Requirements

1. Prove the production Target/RVV artifact path can start from selected
   pre-realized `tcrv_rvv` bodies for:
   - one compare/select fixture;
   - one base-memory fixture, `strided_load_unit_store`.
2. The direct path must run through `--tcrv-materialize-emission-plans` without
   first invoking `--tcrv-materialize-selected-lowering-boundaries`.
3. The realized body must still feed existing RVV route facts,
   operand-binding facts, family statement plans, provider route construction,
   common EmitC, and target artifact/header generation.
4. Generated-bundle ABI evidence tooling must be able to exercise the bounded
   direct pre-realized route-entry path and label it truthfully as route-entry
   realization, not as the explicit selected-boundary materializer.
5. Existing explicit selected-body artifact behavior must remain unchanged.
6. Existing pre-realized selected-body evidence modes that intentionally use
   the explicit selected-boundary pass for other families must remain
   available; this task must not silently claim direct route-entry support for
   unbounded families.
7. Unsupported or incomplete pre-realized route-entry integration must fail
   closed before artifact generation; common EmitC/target export must remain
   neutral consumers.

## Acceptance Criteria

- [x] Trellis PRD, `implement.jsonl`, and `check.jsonl` reference the relevant
      RVV plugin, EmitC route, plugin protocol, testing specs, and prior
      route-entry PRD.
- [x] `pre-realized-selected-body-artifact-cmp-select.mlir` proves direct
      pre-realized route-entry artifact planning/header export without
      `--tcrv-materialize-selected-lowering-boundaries`.
- [x] `pre-realized-selected-body-artifact-strided-load-unit-store.mlir` proves
      direct pre-realized route-entry artifact planning/header export without
      `--tcrv-materialize-selected-lowering-boundaries`.
- [x] The explicit selected-boundary materialization pass coverage for those
      fixtures remains present so the legacy explicit-boundary path is still
      regression tested.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` has a bounded direct
      pre-realized route-entry evidence mode for the chosen artifact/ABI path.
- [x] A script lit/FileCheck test proves generated-bundle ABI dry-run evidence
      for `cmp_select` and `strided_load_unit_store`, including materialized
      body checks, target bundle checks, generated harness checks, and absence
      of descriptor/direct-C/source-export/source-front-door authority.
- [x] Existing generated-bundle script tests still pass for explicit selected
      bodies and existing pre-realized selected-boundary mode.
- [x] Bounded scans over touched script/test/compiler files show no new legacy
      `RVVI32M1`, `rvv-i32m1`, positive finite `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/source-front-door/direct-C/source-export,
      or mirror-only route authority drift.
- [x] Bounded provider/common/target diff scan shows common EmitC and target
      artifact code did not gain RVV semantic inference.
- [x] `git diff --check` passes.
- [x] Focused lit/FileCheck, script dry-run, generated-bundle ABI dry-run, and
      `check-tianchenrv` pass, or exact blockers are documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Out Of Scope

- No new RVV operation family, route coverage, reduction, contraction, dtype,
  LMUL, frontend, source-front-door, descriptor/direct-C/source-export, or
  artifact dashboard work.
- No runtime ABI, dispatch/fallback, runtime `n`/AVL/VL, emitted semantics, or
  performance claim changes.
- No movement of RVV semantics into common EmitC, target export, or Python
  compiler-core structures.
- No broad matrix sweep. Evidence is bounded to the chosen compare/select and
  base-memory artifact/ABI path plus necessary regressions.
- No `ssh rvv` correctness/performance claim unless the generated artifact is
  actually compiled/run on the RVV machine.

## Technical Approach

Keep compiler semantics where they already belong:

- RVV plugin code owns route-entry selected-body realization.
- RVV route planning/provider code owns route facts, operand bindings,
  statement plans, headers, C type/intrinsic choices, and ABI mappings.
- Common EmitC and target export consume provider-built routes and mirrors.

Implementation should therefore be narrowly scoped:

- Update the two representative Target/RVV pre-realized fixtures so
  `--tcrv-materialize-emission-plans` alone proves direct route-entry artifact
  generation, while keeping explicit selected-boundary materialization
  regression lines.
- Extend `rvv_generated_bundle_abi_e2e.py` with a bounded option for direct
  pre-realized route-entry bundle generation. That option must require
  `--pre-realized-selected-body` and reject non-supported op kinds instead of
  pretending all pre-realized families have direct route-entry artifact
  support.
- Add a focused script test for direct generated-bundle ABI dry-run evidence
  over `cmp_select` and `strided_load_unit_store`.

## Validation Plan

1. Validate and start this Trellis task.
2. Run focused direct artifact FileCheck for:
   - `test/Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir`;
   - `test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir`.
3. Run focused generated-bundle ABI script dry-run for direct route-entry
   `cmp_select` and `strided_load_unit_store`.
4. Run the new script lit test.
5. Run representative existing script tests for explicit selected bodies and
   old pre-realized selected-boundary mode.
6. Run active-authority scans over touched files.
7. Run provider/common/target diff scan.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Implementation Result

- Updated
  `test/Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir` so
  `--tcrv-materialize-emission-plans` alone proves direct pre-realized
  route-entry artifact planning and header export, while retaining the explicit
  `--tcrv-materialize-selected-lowering-boundaries` regression path.
- Updated
  `test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir`
  with the same direct artifact/header route-entry coverage for one
  base-memory statement-plan-backed fixture.
- Added `--direct-pre-realized-route-entry` to
  `scripts/rvv_generated_bundle_abi_e2e.py`. The option requires
  `--pre-realized-selected-body`, skips the public selected-boundary
  materializer, labels evidence as `rvv-route-entry-selected-body-realization`,
  and rejects non-bounded op kinds instead of widening support claims.
- Added
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-route-entry-dry-run.test`
  covering direct generated-bundle ABI dry-run evidence for `cmp_select` and
  `strided_load_unit_store`.
- Updated `.trellis/spec/testing/mlir-testing-contract.md` with the durable
  direct pre-realized route-entry generated-bundle evidence command contract,
  validation/error matrix, and tests required.
- Common EmitC, RVV route provider/planning, and target artifact C++ code were
  not changed in this round; they remain neutral consumers of provider-built
  route outputs.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-pre-realized-selected-body-artifact-abi-integration`
  passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- Focused direct artifact checks passed:
  - `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck ... --check-prefix=PLAN`
  - `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck ... --check-prefix=HEADER`
  - `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck ... --check-prefix=PLAN`
  - `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck ... --check-prefix=HEADER`
- Direct generated-bundle ABI dry-run passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind cmp_select --op-kind strided_load_unit_store ...`.
- Manual FileCheck of the new script test's ROOT/CMP/STRIDED/HARNESS prefixes
  passed against the generated dry-run artifacts.
- Existing script modes still passed for:
  - pre-realized `cmp_select` with the explicit selected-boundary materializer;
  - pre-realized `strided_load_unit_store` with the explicit selected-boundary
    materializer;
  - explicit selected-body add/sub/mul.
- Direct option fail-closed checks passed:
  - `--direct-pre-realized-route-entry` without
    `--pre-realized-selected-body` rejects before bundle generation;
  - `--direct-pre-realized-route-entry --op-kind reduce_add` rejects as outside
    the bounded direct route-entry evidence set.
- `ssh rvv` generated-bundle ABI evidence passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind cmp_select --op-kind strided_load_unit_store ...`.
  The remote output reported:
  - `PASS op=cmp_select counts=7,16,23` with true/false predicate-lane
    coverage;
  - `PASS op=strided_load_unit_store counts=7,16,23 stride_bytes=4,8,12` with
    byte-strided load, contiguous output, and tail preservation checks.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with 365/365 lit
  tests.
- Bounded provider/common/target diff scan found no changed common EmitC,
  RVV provider/planning, or target artifact C++ files.
- Added-line active-authority scan over touched tracked files found no new
  legacy i32/source-front-door/descriptor/mirror-only authority drift. The new
  script lit test uses descriptor/direct-C/source-export/source-front-door only
  as `implicit-check-not` negative assertions.
- `git diff --check` passed.
- Final commit is created after this PRD update in the same round.
