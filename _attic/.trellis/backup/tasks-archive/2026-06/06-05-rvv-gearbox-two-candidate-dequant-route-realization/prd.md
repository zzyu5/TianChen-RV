# Stage2 RVV Gearbox two-candidate dequant route realization

## Goal

Extend the existing typed `dequantize_i32_to_f32` RVV Gearbox path from a
one-candidate selection contract into a bounded two-candidate route-realization
slice. The Gearbox pass must derive the legal candidate set from the selected
typed `tcrv_rvv` body/config/runtime facts, deterministically select one legal
candidate, and the RVV provider must validate that selected schedule before it
materializes the route/statement plan. The selected schedule must affect real
route planning or emitted statement structure, not only metadata.

## What I already know

- No `.trellis/.current-task` existed when this round began, so this task was
  created from the Hermes Direction Brief.
- Commit `5f80e167` and the archived Gearbox candidate-selection task already
  added candidate-set, selected-candidate, selection-reason, legality-scope,
  provider membership validation, target mirrors, and focused fail-closed tests.
- The existing route remains bounded to the typed
  `dequantize_i32_to_f32` body with runtime ABI order `lhs,scale,out,n`,
  source `i32/m1`, result `f32/m1`, runtime AVL source `n`, and provider-owned
  dequantization facts.
- Current production code exposes only one legal candidate:
  `rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1`.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires RVV schedule and
  selected-body realization decisions to remain plugin-owned and derived from
  typed body/config/runtime/capability facts.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires Common EmitC to
  carry provider-built route payloads neutrally and requires target artifacts
  to treat candidate metadata as provider-derived mirrors only.
- The selected route behavior will be implemented in RVV pass/provider/route
  planning/statement-plan/target mirror code. Common EmitC must not choose a
  Gearbox schedule.

## Requirements

- Production pass, provider, route-planning/statement-plan, and target mirror
  changes are required.
- The proof route remains the existing selected typed
  `dequantize_i32_to_f32` route. Do not add a new frontend, new high-level op,
  or broad route surface.
- Add exactly one additional legal Gearbox candidate for the same typed body.
  The candidate must be derived from structural route facts:
  operation, source/result SEW and LMUL, runtime AVL source, memory form,
  dequantization relation, and ABI roles.
- Deterministic selection must choose one candidate for a structural reason.
  This task may choose the bounded two-slice/unroll candidate when the typed
  dequant route satisfies the existing SEW32/LMUL m1/runtime-AVL contract.
- The selected candidate must change route/statement-plan behavior. The
  expected bounded behavior is a two-slice dequant statement plan: the route
  still preserves semantics, but the loop body contains the selected schedule's
  additional runtime-VL slice steps and target validation expects the selected
  schedule's statement count.
- Provider validation must prove before route construction that:
  - candidate inventory exists;
  - selected candidate exists;
  - selected candidate belongs to the legal candidate set;
  - selected schedule dimensions match the selected candidate;
  - operation, legality scope, dtype/config, memory form, runtime AVL source,
    and ABI facts match the typed body/provider facts;
  - unsupported schedule dimensions fail closed.
- Target artifact validation must consume provider-owned route facts and mirror
  contracts. It must reject stale candidate-set, selected-candidate, selected
  schedule, and statement-plan-count mirrors.
- Common EmitC/export remains neutral. It may materialize provider-built
  statement steps but must not infer or choose RVV schedules.
- Do not use route strings, artifact names, q-names, ABI strings, test names,
  descriptor residue, or intrinsic spelling as Gearbox authority.

## Acceptance criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      two-candidate route-realization task and its spec basis.
- [x] The Gearbox pass materializes at least two legal candidates for the
      positive `dequantize_i32_to_f32` typed body and records the deterministic
      selected candidate.
- [x] The selected candidate is validated as a member of the legal set before
      route construction, and stale/mismatched schedule dimensions fail closed.
- [x] The selected candidate changes the dequant route/statement plan in
      production code, with focused evidence for the selected schedule's body
      step count or emitted route structure.
- [x] Target validation mirrors only provider-derived candidate and selected
      schedule facts, and rejects stale mirrors.
- [x] Focused positive tests show the legal candidate set, selected candidate,
      selection reason, schedule dimensions, and route-plan difference.
- [x] Negative tests cover missing candidate set, selected candidate not in
      candidate set, unsupported schedule dimension, stale selected schedule
      metadata, stale target mirror, missing route-plan materialization, and
      route-string/artifact-name/intrinsic-spelling authority attempts.
- [x] Relevant `tcrv-opt`, `tcrv-translate`, target artifact, and RVV plugin
      tests pass for the touched behavior.
- [x] `git diff --check`, `git diff --cached --check`, and a bounded
      old-authority/q-name scan over touched files pass.
- [x] `ssh rvv` is reported as not applicable unless this task makes executable
      correctness/runtime/performance claims.
- [x] The Trellis task status, archive state, journal, commit hash, and final
      worktree status are recorded truthfully.

## Out of scope

- Runtime benchmarking, persistent tuning cache, assembly feedback loop, or
  cross-kernel autotuning.
- New high-level Linalg/Vector/StableHLO frontend work.
- Common/core RVV schedule selection.
- Direct C/intrinsic macro autotuning as route authority.
- Broad route-surface expansion or one-intrinsic wrapper growth.
- q-name, benchmark-name, artifact-name, ABI-string, descriptor, or intrinsic
  spelling authority.
- Claiming executable correctness, runtime speedup, or performance without
  `ssh rvv` evidence.

## Technical notes

- Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/testing/index.md`.
- Continuation context:
  `.trellis/tasks/archive/2026-06/06-05-06-05-rvv-gearbox-bounded-candidate-selection-contract/prd.md`.
- Main production owners to inspect/update:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
- Focused tests to inspect/update:
  `test/Transforms/RVV/rvv-gearbox-dequantize-i32-to-f32.mlir`,
  `test/Transforms/RVV/rvv-gearbox-dequantize-i32-to-f32-negative.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`,
  `test/Target/TargetArtifactExportTest.cpp`, and RVV plugin/provider tests
  that exercise route planning.

## Decision

Implement one real additional schedule candidate as a bounded Gearbox
two-slice/unroll alternative for the existing typed dequant route, and make the
provider-selected schedule drive the dequant statement plan and target
validation contract. If repository evidence shows the two-slice candidate
cannot be expressed safely without semantic or materializer changes outside
this module, narrow to the smallest legal schedule alternative that changes
route-plan structure and record the rejected alternative as unsupported rather
than faking metadata.

## Completion evidence

- Added legal Gearbox candidate `rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1`
  alongside the retained legal u1 candidate, and deterministically selected u2
  for the existing typed `dequantize_i32_to_f32` body.
- The selected u2 schedule changes production route planning: dequant loop step
  becomes `full_chunk_vl * 2`, and the loop body grows from five steps to ten
  steps with a second runtime-VL setvl/load/convert/scale/store slice.
- Provider validation checks selected candidate membership, selected schedule
  dimensions, u2 loop-step/body-step materialization, and second-slice
  statement facts before route construction.
- Target validation consumes provider-derived dequant Gearbox facts, validates
  the u2 loop step and second-slice operands/results, and rejects stale
  candidate/selected-schedule mirrors and stale route-plan materialization.
- Checks run:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`;
  `lit -sv . --filter 'rvv-gearbox-dequantize-i32-to-f32|explicit-selected-body-artifact-dequantize-i32-to-f32'`
  from `build/test`; `./build/bin/tianchenrv-target-artifact-export-test`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`; `git diff --check`;
  `git diff --cached --check`; bounded old-authority/q-name scan over touched
  files and added diff lines.
- `ssh rvv` was not run because this task does not claim executable
  correctness, runtime behavior, or performance.
