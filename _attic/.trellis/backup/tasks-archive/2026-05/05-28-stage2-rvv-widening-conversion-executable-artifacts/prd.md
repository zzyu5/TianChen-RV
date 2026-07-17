# Stage2 RVV widening conversion executable artifacts

## Goal

Repair stale Trellis-only residue from the previous no-op computed-mask
segment2 round, then close the executable generated-bundle path for the two
existing RVV widening conversion selected-boundary fixtures:

- `widen_i16_to_i32`
- `widen_i32_to_i64`

The production path under test is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv widening conversion body
  -> RVV plugin-local selected-body realization
  -> realized typed load / widening_convert / store structure
  -> RVV-owned conversion dtype/SEW/LMUL/runtime facts
  -> provider route facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC
  -> target artifact ABI/header/object bundle
  -> generated RVV C artifact
  -> ssh rvv compile/run/correctness evidence
```

## Direction Source

Hermes Direction Brief:

- Direction title: `Redirect: repair stale Trellis state, then close RVV
  widening conversion executable artifacts`.
- Module owner: first restore git/Trellis coherence from uncommitted no-op
  computed-mask segment2 task residue, then make the selected-boundary widening
  conversion fixtures executable through plugin-owned typed RVV facts, provider
  route construction, neutral EmitC, target artifact ABI, generated C artifact,
  and real `ssh rvv` evidence.
- Why now: the last run left HEAD unchanged at
  `8a75f2cd rvv: close plain segment2 executable artifacts` and left only dirty
  Trellis metadata. Current inspection also shows direct pre-realized
  route-entry support is disabled while the widening conversion
  selected-boundary generated-bundle checks need executable evidence for both
  conversion fixtures.

## Repository Facts Read Before Implementation

- `pwd` is `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` showed:
  - modified `.trellis/workspace/codex/journal-17.md`
  - untracked archived no-op computed-mask segment2 task directory
- Initial `git log --oneline -8` showed HEAD at
  `8a75f2cd rvv: close plain segment2 executable artifacts`, followed by
  recent RVV demotion commits.
- `.trellis/.current-task` did not exist, so this task was created from the
  Hermes brief.
- Gate 0 state repair has removed the untracked archived no-op task directory
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-computed-mask-segment2-executable-artifact-path/`
  and restored the journal to no diff. No metadata-only success commit was
  created for that stale residue.

## Continuation Facts From This Round

- The current task existed at
  `.trellis/tasks/05-28-stage2-rvv-widening-conversion-executable-artifacts/`
  and `.trellis/.current-task` pointed to it.
- The dirty worktree contained one shared script diagnostic change plus direct
  pre-realized FileCheck updates. These changes were retained because they keep
  selected-boundary-only direct route-entry probes op-specific, including
  `widen_i16_to_i32` and `widen_i32_to_i64`; no unrelated production owner was
  added.
- Selected-boundary dry-run evidence for both widening conversion fixtures was
  regenerated under
  `artifacts/tmp/stage2_widening_conversion_executable_closure/selected-boundary-dry-run/`.
- Non-dry-run generated-bundle evidence for both widening conversion fixtures
  compiled and ran on `ssh rvv` under
  `artifacts/tmp/stage2_widening_conversion_executable_closure/ssh-rvv-executable/`.
- The stale no-op computed-mask segment2 archive path remained absent.

## Requirements

1. Keep direct pre-realized route-entry disabled for the two widening
   conversion fixtures unless production evidence proves a scoped route-entry
   owner change is required. Selected-boundary materialization is the intended
   executable path for this task.
2. The selected-boundary dry-run path for `widen_i16_to_i32` and
   `widen_i32_to_i64` must show `route_entry_realization: false`,
   `pre_realized_body_consumed: true`, and selected-body materialization before
   route/provider/export evidence.
3. The realized typed body must preserve and expose source/result dtype,
   source/result SEW, source/result LMUL, widening relation, memory form,
   runtime `n`/AVL/VL, setvl placement, policy, source load, conversion op, and
   result store from the typed RVV structure.
4. Route/provider/statement-plan construction must consume RVV-owned typed
   body/config/runtime facts, route-control provider facts, conversion
   route-family facts, operand-binding facts, and statement-plan facts before
   producing `TCRVEmitCLowerableRoute`.
5. Common EmitC/export must remain neutral. Conversion semantics must not be
   inferred from route ids, operation names alone, artifact names, ABI strings,
   test names, exact intrinsic spellings, script defaults, metadata mirrors,
   descriptors, source-front-door markers, or common EmitC code.
6. Target artifact export must consume provider route plus provider description
   before accepting object/header/bundle claims. Artifact/header metadata may
   mirror conversion facts only after provider route construction.
7. Non-dry-run generated-bundle execution on real `ssh rvv` must pass for both
   fixtures with runtime counts covering `0`, `1`, an exact vector-sized case,
   a tail case, and stress cases.
8. Correctness evidence must check lane conversion values, negative signed
   extension behavior, output sentinel preservation for inactive/tail lanes,
   runtime ABI order, pointer roles, source preservation, setvl/VL loop use,
   and generated C compile/run on RVV hardware.
9. Negative coverage must reject or fail closed for wrong source/result dtype or
   SEW relation, missing runtime/mem binding, stale route id, metadata-derived
   authority, exact-intrinsic-as-authority, and common-EmitC semantic invention.
10. Recently closed segment2 and contraction selected-body paths must not
    regress under focused checks.

## Acceptance Criteria

- [x] Gate 0 residue repair is preserved: no old no-op computed-mask segment2
      archive remains, and stale journal-only residue is not committed as a
      standalone success artifact.
- [x] Selected-boundary dry-run generated-bundle evidence for
      `widen_i16_to_i32` and `widen_i32_to_i64` reports
      `route_entry_realization: false`, `pre_realized_body_consumed: true`, and
      selected-boundary materialization.
- [x] Direct pre-realized route-entry remains unsupported or fails closed for
      the widening conversion fixtures unless this task explicitly changes a
      scoped route-entry owner with matching production/provider/test evidence.
- [x] Non-dry-run generated-bundle execution on `ssh rvv` passes for both
      widening conversion fixtures with counts including `0`, `1`, exact, tail,
      and stress cases.
- [x] Evidence JSON/header/generated C/harness output mirrors provider-derived
      conversion facts, source/result type policy, route-family facts,
      operand-binding facts, statement-plan facts, runtime ABI order, and
      runtime AVL/VL usage.
- [x] Negative coverage exercises wrong dtype/SEW relation, missing runtime or
      mem binding, metadata-derived authority, stale route-id/ABI/artifact-name
      authority, exact-intrinsic-only authority, and common EmitC semantic
      invention for the conversion boundary.
- [x] Focused non-regression covers recently closed segment2 and contraction
      selected-body paths.
- [x] Bounded authority scan over touched production files and directly related
      tests/scripts finds no new descriptor, source-front-door, route-id,
      artifact-name, ABI-string, metadata, script-default, exact-intrinsic,
      common-EmitC, or legacy-i32 authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded.
- [x] Task is finished/archived and one coherent commit is created if
      production progress is made.

## Completion Evidence

- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --op-kind widen_i16_to_i32 ...`: passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --op-kind widen_i32_to_i64 ...`: passed.
- Direct pre-realized route-entry probes for both widening conversion fixtures
  failed closed before target bundle export with op-specific selected-boundary
  diagnostics.
- `ssh rvv` non-dry-run generated-bundle execution passed for
  `widen_i16_to_i32` with counts `0,1,16,23,257`; remote output reported
  `tcrv_rvv_generated_bundle_abi_widen_i16_to_i32_ok` and `PASS`.
- `ssh rvv` non-dry-run generated-bundle execution passed for
  `widen_i32_to_i64` with counts `0,1,16,23,257`; remote output reported
  `tcrv_rvv_generated_bundle_abi_widen_i32_to_i64_ok` and `PASS`.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`: passed.
- Focused lit filter
  `direct-pre-realized|pre-realized-widen-i(16|32)|pre-realized-selected-body-artifact-widen-i(16|32)`:
  25/405 selected tests passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  and `build/bin/tianchenrv-rvv-extension-plugin-test`: passed.
- Focused segment2 and contraction-family selected-body dry-run
  non-regression: passed.
- Changed-line authority scan over touched script/tests found no new
  descriptor, direct-C, source-front-door/export, route-id-derived,
  artifact-name-derived, metadata-derived, exact-intrinsic-derived, or legacy
  i32 authority.
- `git diff --check`: passed.
- `cmake --build build --target check-tianchenrv -j2`: passed, 405/405.

## Spec Update Decision

No `.trellis/spec/` update was made. This round did not introduce a new
architecture contract: it completed evidence for the existing conversion/SEW
policy selected-boundary path and kept the already-specified direct
pre-realized route-entry fail-closed boundary op-specific in script diagnostics.

## Out Of Scope

- Reopening computed-mask segment2 or plain segment2 work.
- Committing the stale no-op Trellis archive as a standalone success artifact.
- Source-front-door positive routes, high-level Linalg/frontend lowering,
  one-intrinsic wrapper dialects, global autotuning, broad dashboard/report
  work, future plugin work, or dtype/LMUL clone batches detached from the two
  existing widening conversion fixtures.
- Moving conversion semantics into common EmitC/export or test/script metadata.

## Technical Approach

1. Validate the clean post-Gate-0 baseline with `git status --short`.
2. Inspect and, if needed, repair the selected-body realization, route planning
   and provider, target bundle, generated-bundle script, and focused tests for
   widening conversion:
   - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
   - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
   - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
   - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
   - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
   - `scripts/rvv_generated_bundle_abi_e2e.py`
   - `test/Plugin/RVVExtensionPluginTest.cpp`
   - `test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir`
   - `test/Target/RVV/pre-realized-selected-body-artifact-widen-i32-to-i64.mlir`
3. Run selected-boundary dry-run and direct fail-closed probes for both
   conversion fixtures.
4. Run non-dry-run generated-bundle executions on `ssh rvv` for both
   conversion fixtures with runtime counts covering zero, one, exact, tail, and
   stress cases.
5. Add or adjust production/test evidence only where a real blocker appears.
6. Run focused C++/lit/script checks, segment2/contraction non-regression,
   authority scan, `git diff --check`, and `check-tianchenrv` when feasible.

## Technical Notes

Specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`

Key spec constraints:

- RVV runtime/correctness claims require `ssh rvv` evidence.
- Conversion route support must come from typed `tcrv_rvv` conversion
  body/config/runtime facts, then RVV plugin legality/realization,
  route-family facts, operand-binding facts, statement-plan facts, provider
  route construction, common EmitC, and mirror metadata.
- `tcrv.exec` ABI declarations are runtime/export organization only and must
  not define RVV dtype, operation kind, vector length policy, or route identity.
- Common EmitC/export must not invent RVV dtype, SEW/LMUL, schedule, intrinsic
  choice, or ABI role semantics.
