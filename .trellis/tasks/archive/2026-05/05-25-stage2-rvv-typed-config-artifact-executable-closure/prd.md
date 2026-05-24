# Stage2 RVV typed config artifact executable closure

## Goal

Prove that the typed RVV dtype/config authority consumed by provider emission
survives generated target artifact packaging and real RVV execution for one
bounded existing route-supported cluster. The round must connect a selected
`tcrv.exec` RVV variant through the typed `tcrv_rvv` body, provider-built
`TCRVEmitCLowerableRoute`, common EmitC, generated RVV bundle, and one real
`ssh rvv` compile/run correctness oracle.

This is an executable-evidence closure for the already-supported typed-config
emission path. It is not new RVV family coverage, not frontend work, not a
performance claim, and not a helper/report-only task.

## Direction Source

- Direction title: `Stage2 RVV typed config artifact executable closure`.
- Module owner: RVV typed dtype/config emission path carried through generated
  RVV artifacts and one real `ssh rvv` correctness run for existing
  route-supported families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `8bcf3d31 rvv: consume typed config facts in emission`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` defines the current RVV-first authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider -> `TCRVEmitCLowerableRoute`
  -> common EmitC -> target artifact -> `ssh rvv` evidence when runtime or
  correctness is claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires RVV dtype/config,
  intrinsic spelling, C vector types, ABI mapping, legality, and fail-closed
  diagnostics to remain RVV-plugin-owned and derived from typed body/config/
  runtime facts, not route ids, artifact names, helper names, ABI strings,
  metadata, or common EmitC/export code.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC/export
  to materialize provider-built route payloads neutrally and not infer RVV
  dtype, SEW, LMUL, policy, operation kind, schedules, ABI, headers, or
  intrinsic names from mirrors or metadata.
- `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  output for RVV runtime/correctness claims and requires generated-bundle
  evidence over runtime `n` for memory-writing routes to verify active lanes
  and guard/tail sentinel preservation.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-typed-config-emission-consumption-closure/prd.md`
  closed provider/emission consumption of `RVVSelectedBodyTypedConfigFacts`
  and added focused plugin/target tests, but intentionally stopped short of
  proving generated artifact ABI and real RVV execution.
- `scripts/rvv_generated_bundle_abi_e2e.py` already owns generated RVV bundle
  ABI checks, local dry-run validation, optional non-dry-run `ssh rvv`
  compile/run, and deterministic route-family correctness oracles.
- Existing dry-run lit coverage includes typed-config-relevant families such as
  pre-realized `i64_add`, `lmul_m2_add`, `cmp_select`, and memory movement
  routes. The implementation should choose one existing cluster that is already
  route-supported rather than expanding operation-family coverage.

## Requirements

1. Select one bounded existing typed-config route-supported cluster for the
   executable closure. Prefer a non-i32 arithmetic or compare/select case if
   it is already ready; include one memory case only if it is already supported
   by the existing generated-bundle script and does not broaden the task.
2. Generated artifact bundle verification must explicitly prove ABI metadata
   mirrors the provider-derived typed route facts for the selected route:
   element type, SEW, LMUL, policy/config contract, vector C type/header,
   operation/memory form, intrinsic spelling, runtime ABI order, and runtime
   count use where applicable.
3. The emitted RVV C type, required header, and intrinsic spelling must be
   checked as consequences of typed body/config/capability/runtime facts, not
   route ids, artifact names, helper names, ABI strings, test names, manifests,
   emission-plan residue, or common EmitC/export logic.
4. The generated artifact must compile and run on `ssh rvv` with deterministic
   input/output correctness. For memory-writing routes over runtime `n`, the
   oracle must check active lanes and guard/tail sentinel preservation.
5. Unsupported or inconsistent typed config combinations relevant to the chosen
   closure must fail closed before artifact authority. The failure may be
   covered by an existing target/provider diagnostic if it directly guards the
   new executable path; otherwise add a focused negative check.
6. Common EmitC and target mechanics remain neutral consumers. Do not move RVV
   dtype/config/intrinsic decisions into common EmitC/export or script-side
   authority.
7. Do not introduce or preserve positive legacy `RVVI32M1*`, `rvv-i32m1-*`,
   `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
   descriptor/direct-C/source-export, or exact-intrinsic-name authority.

## Acceptance Criteria

- [x] PRD and context files reference the RVV plugin, EmitC route, testing,
      validation, and previous typed-config emission-consumption task.
- [x] One focused generated-bundle ABI/e2e path checks typed route fact mirrors
      for the selected executable route artifact.
- [x] The same path reaches real non-dry-run `ssh rvv` compile/run with a
      deterministic correctness oracle and records the evidence location.
- [x] Unsupported or stale typed config metadata relevant to the selected
      artifact path fails closed before artifact authority.
- [x] Focused lit/unit coverage is added or updated only for the changed
      generated-bundle/artifact behavior.
- [x] Bounded authority scan over touched planning/provider/target/script/
      fixture files shows no new name-, metadata-, route-id-, descriptor-, or
      source-front-door-derived RVV dtype/config authority.
- [x] `git diff --check` passes.
- [x] Focused script/tests pass and `check-tianchenrv` passes, or an exact
      blocker is documented.
- [x] Trellis task status, journal, archive, and commit are truthful if the
      task completes.

## Non-Goals

- No broad new operation-family coverage, reductions, contractions, high-level
  frontend lowering, source-front-door positive routes, legacy i32 authority,
  one-op-per-intrinsic wrappers, dtype/LMUL clone batches, descriptor/direct-C/
  source-export paths, dashboards, broad smoke matrices, or performance claims.
- No script-side inference of route support or typed config from artifact names,
  route ids, status/result wording, manifests, ABI strings, helper names, test
  names, or exact intrinsic spellings.
- No movement of RVV semantic choices into common EmitC/export.
- No Python implementation of compiler core, dialect, legality, route planning,
  provider, lowering, or emission behavior.

## Technical Approach

1. Start the Trellis task and validate context.
2. Inspect the generated-bundle ABI script, existing dry-run lit tests, target
   artifact fixtures, and provider/artifact metadata fields for typed config
   mirrors.
3. Choose the narrow executable cluster with the least new compiler surface.
   Candidate priority: pre-realized `i64_add` or `cmp_select`, falling back to
   another already-supported typed-config route only if evidence shows those
   are not viable.
4. Add or tighten generated-bundle script validation so the chosen route's
   artifact metadata, emitted C/header, runtime ABI, and correctness oracle are
   checked as a single closure.
5. Add a focused dry-run lit/FileCheck and a focused negative fail-closed check
   if existing diagnostics do not already cover the changed path.
6. Run one real `ssh rvv` generated-bundle compile/run for the selected route
   and record the artifact/evidence path in the PRD.
7. Run focused checks, bounded authority scans, `git diff --check`, and
   `check-tianchenrv` if feasible.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-typed-config-artifact-executable-closure`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
4. Focused dry-run generated-bundle test for the chosen selected-body route.
5. One non-dry-run generated-bundle `ssh rvv` compile/run for the chosen route
   with runtime counts such as `7,16,23`.
6. Focused negative fail-closed check for stale/unsupported typed config in the
   selected path if needed.
7. Bounded authority scan over touched script, fixture, provider/planning, and
   target files.
8. `git diff --check`
9. `cmake --build build --target check-tianchenrv -j2`, unless an exact blocker
   is encountered and recorded.

## Definition Of Done

The selected typed-config RVV route has a generated artifact whose mirrors,
ABI, emitted RVV C/header/intrinsics, and correctness oracle are tied to
provider-derived typed body/config/runtime facts; the artifact compiles and
runs on real `ssh rvv`; stale or unsupported typed config does not become
artifact authority; focused and full checks pass or a precise blocker is
documented; the task is finished/archived; and one coherent commit records the
work.

## Implementation Result

- Chosen executable cluster: pre-realized `i64_add`. This is a bounded
  non-i32 elementwise arithmetic route that already flows through the selected
  `tcrv.exec` RVV variant, typed pre-realized `tcrv_rvv` body, selected-boundary
  realization, provider-built route, common EmitC, generated object/header
  bundle, and external C ABI harness.
- `scripts/rvv_generated_bundle_abi_e2e.py` now exports the materialized RVV
  EmitC C/C++ source beside the generated bundle and verifies the typed emitted
  source for plain elementwise arithmetic. For `i64_add` this checks
  `vint64m1_t`, `__riscv_vsetvl_e64m1`, `__riscv_vle64_v_i64m1`,
  `__riscv_vadd_vv_i64m1`, and `__riscv_vse64_v_i64m1`.
- Generated-bundle metadata verification now requires `tcrv_rvv.element_type`
  as an expected typed config mirror, and plain non-RHS-broadcast elementwise
  arithmetic now checks the provider-derived elementwise family plan,
  target-leaf profile, provider-supported mirror, required header declarations,
  C type mapping, runtime-control plan, source/destination memory form, and
  route-operand binding mirrors.
- Generated header verification now checks typed config comments for config
  contract, element type, SEW, LMUL, policy, required headers, and C type
  mapping. Header checks remain declaration-only and still reject intrinsic
  bodies in public headers.
- Per-op evidence now records `emitted_rvv_cpp_checks` and
  `typed_config_artifact_closure`, explicitly labeling artifact metadata as
  `mirror-only-after-provider-route`.
- The script self-test now mutates `i64_add` element-type metadata from `i64`
  to `i32` and requires bundle verification to fail closed.
- The focused lit test
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-i64-add-dry-run.test`
  now checks the emitted RVV C/C++ source, typed metadata mirrors, and typed
  closure evidence for the `i64_add` generated bundle.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-typed-config-artifact-executable-closure`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Focused dry-runs:
      - pre-realized `i64_add`
      - explicit `add`
      - pre-realized `lmul_m2_add`
      - pre-realized `cmp_select`
      - RHS-broadcast `add`
      - LMUL m2 selected-body `add`
- [x] Focused lit:
      `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-i64-add-dry-run|rhs-broadcast-dry-run|lmul-m2-dry-run|self-test)'`
      from `build/test`, 4/4 passed.
- [x] Real `ssh rvv` generated-bundle run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_typed_config_artifact_executable_closure --run-id pre-realized-i64-add-ssh-rvv --overwrite --op-kind i64_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv`
- [x] `ssh rvv` evidence path:
      `artifacts/tmp/stage2_typed_config_artifact_executable_closure/pre-realized-i64-add-ssh-rvv`
- [x] Remote compile/run evidence:
      - `remote_arch=riscv64`
      - `clang_path=/usr/bin/clang`
      - `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`
      - `i64_add case n=7 ok`
      - `i64_add case n=16 ok`
      - `i64_add case n=23 ok`
      - `PASS op=i64_add counts=7,16,23`
- [x] First `check-tianchenrv` attempt caught a real overreach: plain
      elementwise metadata checks were applied to RHS-broadcast `add`. The fix
      restricted the new plain elementwise closure to non-RHS-broadcast routes.
- [x] Bounded authority scan over touched task/script/test files found only
      PRD non-goals, FileCheck `implicit-check-not` clauses, and existing
      negative/self-test residue checks; no new positive legacy i32,
      source-front-door/source-artifact, descriptor, direct-C/source-export, or
      name/metadata-derived route authority was introduced.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`, 365/365 passed.

## Spec Sync

No `.trellis/spec/**` change is required. This round applies existing RVV
plugin, unified EmitC route, emission-runtime, and testing contracts to the
generated-bundle evidence tooling. The one self-repair was an implementation
scoping issue in script validation, not a new durable project rule.
