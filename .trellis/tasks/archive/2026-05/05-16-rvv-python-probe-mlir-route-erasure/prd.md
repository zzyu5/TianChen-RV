# RVV Python probe-to-MLIR route erasure

## Goal

Remove the Python compiler-core route authority that replays RVV probe JSON into
`tcrv.exec` MLIR capability, target, kernel, route, or fallback modeling. After
this deletion round, RVV probe tooling may only collect bounded raw
hardware/toolchain evidence; compiler-visible RVV capability/profile behavior
must come from the C++/MLIR extension-family/plugin path.

## Campaign Mode

This is part of the Wrong Logic Deletion Campaign. The round is
deletion/refactor-only:

- do not add replacement frontend lowering;
- do not add Common EmitC rebuild work;
- do not add executable RVV plugin emission;
- do not add route selection, target profile generation, compatibility
  wrappers, legacy modes, or schema adapters that preserve the old replay path.

If deletion exposes missing rebuild architecture, record that as a gap instead
of restoring the Python route.

## Scope

- Delete `scripts/rvv_probe_to_mlir.py` as an active or hidden compiler path.
- Delete lit coverage that pipes Python-generated RVV replay MLIR into
  `tcrv-opt` as compiler-path evidence.
- Delete directly related RVV probe replay fixtures that only protected that
  Python-to-MLIR path.
- Refactor `scripts/rvv_remote_probe.py` so retained `capability_facts` are raw
  bounded evidence facts only, not compiler-route config facts such as
  first-slice SEW/LMUL/tail/mask policy.
- Update specs/docs that still authorize Python RVV probe replay as
  `tcrv.exec` capability/target/kernel MLIR.
- Keep remote probe local self-test coverage for schema/parser/sanitizer
  behavior without contacting `ssh rvv`.

## Out Of Scope

- New RVV lowering, emission, or runtime glue.
- New extension-family op implementation.
- New target profile generator.
- New Python schema adapter that still emits `tcrv.exec` MLIR.
- New compatibility wrapper or deprecated hidden replay mode.
- Expanded `ssh rvv` evidence.
- Broad unrelated test matrix work.

## Acceptance Criteria

- [ ] No Python script emits `tcrv.exec` MLIR or selected
  route/capability/target/kernel modeling from RVV probe evidence.
- [ ] No active lit test pipes Python-generated RVV replay MLIR into `tcrv-opt`
  as compiler-path evidence.
- [ ] `scripts/rvv_remote_probe.py` no longer fabricates compiler-route config
  facts such as `first_slice_sew_bits`, `first_slice_lmul`,
  `first_slice_tail_policy`, or `first_slice_mask_policy`.
- [ ] Retained Python RVV probe code remains bounded evidence collection only.
- [ ] Specs/docs no longer describe a live RVV probe-to-MLIR replay helper as an
  allowed compiler-path fixture source.
- [ ] Focused ref-scan covers `rvv_probe_to_mlir`, `rvv-probe-to-mlir`,
  `emit_replay_mlir`, `rvv_probe_replay`, `rvv_probe_i64_replay`,
  first-slice capability fact keys, and Python-generated `tcrv.exec.capability`
  surfaces, excluding archive/workspace/build/tmp paths.
- [ ] `python3 scripts/rvv_remote_probe.py --self-test` passes.
- [ ] Affected script/lit coverage no longer references the deleted replay
  route.
- [ ] `ninja -C build tcrv-opt tcrv-translate` passes.
- [ ] Remaining focused lit/FileCheck suites pass where applicable.
- [ ] `ninja -C build check-tianchenrv` is attempted; deletion gaps are reported
  without restoring Python route authority if failures appear.
- [ ] `git diff --check` passes.
- [ ] Trellis task validates, finishes/archives, and one coherent commit is
  created.

## Relevant Specs Read

- `.trellis/spec/index.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/capability-model/profiles.md`

## Initial Evidence

- Current HEAD: `fd44b54 chore(plugin): erase metadata-only route contract`.
- Worktree was clean before task creation.
- Active replay surface exists in `scripts/rvv_probe_to_mlir.py`.
- Active lit coverage exists in `test/Scripts/rvv-probe-to-mlir.test`.
- Probe replay-only fixtures exist under `test/Fixtures/rvv_probe/`.
- `scripts/rvv_remote_probe.py` currently emits first-slice config/policy facts
  in `capability_facts`.
- Specs still contain replay-helper permission text and must be brought in line
  with deletion-campaign direction.
