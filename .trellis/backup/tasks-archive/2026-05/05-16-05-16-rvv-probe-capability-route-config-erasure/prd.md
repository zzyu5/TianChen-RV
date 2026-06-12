# RVV probe capability route-config erasure

## Goal

Remove probe-derived RVV route/config authority from the C++ capability-profile
surface. RVV probe facts may record raw hardware and toolchain evidence, but
they must not manufacture selected RVV SEW/LMUL/tail/mask route capabilities or
finite-family config semantics.

This is a Wrong Logic Deletion Campaign round. It is deletion/refactor-only:
remove obsolete authority and update tests/docs that protect it. Do not rebuild
RVV EmitC, add a replacement generator, or introduce compatibility routes in
this round.

## What I already know

- Current HEAD is `6d5f663 chore(target): erase metadata artifact skeletons`.
- Worktree was clean before this task was created.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.
- The active surface named by the brief is:
  - `include/TianChenRV/Plugin/RVV/RVVCapabilityProfile.h`
  - `lib/Plugin/RVV/RVVCapabilityProfile.cpp`
  - direct consumers in `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - RVV plugin/profile and variant-selection tests that consume probe-derived
    config capabilities
  - `scripts/rvv_remote_probe.py` only to preserve its evidence-only boundary
- Probe evidence should remain raw hardware/toolchain evidence such as
  architecture, harts, VLEN/vlenb, clang/cmake availability, march/mabi, and
  bounded compile/run evidence.

## Requirements

- Delete `RVVProbeCapabilityFacts` fields that encode selected route/config
  semantics, including first-slice and i32/i64 finite-family SEW/LMUL/tail/mask
  fields.
- Remove validation/build logic that validates probe-derived selected RVV
  route/config fields.
- Ensure `buildRVVTargetCapabilitiesFromProbeFacts` no longer emits RVV vector
  config capabilities from probe facts.
- Remove or rewrite public helpers, tests, fixtures, comments, and docs that
  treat probe facts or default probe fields as the source of RVV SEW/LMUL/tail/
  mask/i32/i64 finite-family capability production.
- Preserve valid explicit RVV extension-family syntax and typed RVV IR just
  because it mentions SEW/LMUL; the target is probe-derived authority, not RVV
  semantics owned by the RVV extension.

## Non-goals

- No RVV EmitC rebuild.
- No new lowering route.
- No new selected-boundary implementation.
- No new capability model feature.
- No replacement route-config generator.
- No Python route revival.
- No compatibility adapter or legacy mode.
- No expansion of `ssh rvv` evidence collection.
- No deletion of valid explicit `tcrv_rvv` extension-family syntax solely
  because it mentions SEW/LMUL under typed RVV IR.

## Acceptance Criteria

- [x] `RVVProbeCapabilityFacts` no longer carries first-slice/i32m2/i64m1
      SEW/LMUL/tail/mask route/config fields.
- [x] `buildRVVTargetCapabilitiesFromProbeFacts` no longer emits
      `rvv.i32_m1.*`, `rvv.i32_m2.*`, `rvv.i64_m1.*`, or probe-derived
      `isa-vector-config` capabilities from probe facts.
- [x] Tests no longer rely on probe facts to make metadata-only/no-body RVV
      proposal or variant-selection paths viable.
- [x] Retained RVV probe profile data only records raw hardware/toolchain
      evidence.
- [x] Focused ref-scan finds no active-surface residue for:
      `firstSliceSEWBits`, `firstSliceLMUL`, `firstSliceTailPolicy`,
      `firstSliceMaskPolicy`, `i32M2SEWBits`, `i32M2LMUL`, `i64M1SEWBits`,
      `i64M1LMUL`, `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
      `rvv.i32_m2.sew32`, `rvv.i32_m2.lmul_m2`, `rvv.i64_m1.sew64`, and
      probe-derived `isa-vector-config`, excluding `.trellis/tasks/archive`,
      `.trellis/workspace`, `artifacts/tmp`, `build`, and `.git`.
- [x] Focused build/checks run:
      `ninja -C build tcrv-opt tcrv-translate`, affected RVV plugin/profile or
      variant-selection tests, `ninja -C build check-tianchenrv` attempted,
      `git diff --check`, and Trellis validation.
- [x] Task is finished/archived and one coherent commit is created, unless an
      expected deletion gap prevents completion and is documented precisely.

## Completion Evidence

- `ninja -C build tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-variant-selection-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-variant-selection-test`
- `python3 scripts/rvv_remote_probe.py --self-test`
- `ninja -C build check-tianchenrv` passed 75/75 lit tests.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-16-05-16-rvv-probe-capability-route-config-erasure`
- Focused active-surface ref-scan found no retained probe config fields,
  profile config helper APIs, or `isa-vector-config` production in
  `RVVCapabilityProfile`.

## Technical Notes

- Specs to read before implementation:
  - `.trellis/spec/index.md`
  - `.trellis/spec/capability-model/profiles.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- This PRD follows the deletion campaign rule: do not restore wrong logic just
  to make tests pass. If deletion exposes a missing new-architecture gap, report
  it as such and leave the wrong probe-derived route/config authority deleted.
