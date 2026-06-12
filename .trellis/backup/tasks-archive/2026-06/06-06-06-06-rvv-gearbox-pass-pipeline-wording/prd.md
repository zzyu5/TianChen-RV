# Update RVV Gearbox Pass Pipeline Wording

## Goal

Apply the Pro feedback as a small documentation/spec delta. Clarify that RVV
Gearbox/resource-aware selected-body realization is implemented as an RVV
plugin-local MLIR pass pipeline, while "realization" names the transformation
result/stage. Keep the update aligned with current implementation facts and do
not rewrite the architecture.

## What I Already Know

- Pro recommends small wording calibration, not a broad Trellis audit.
- The current registered MLIR pass is
  `--tcrv-rvv-materialize-gearbox-schedules`.
- That pass is an MVP static Gearbox schedule materializer for bounded
  dequantize i32-to-f32 scheduling, not a full resource-aware autotuner.
- Commit `31bbcade` seeded a low-precision direct-contraction resource candidate
  contract in provider/target validation, but it is not a standalone registered
  MLIR pass yet.
- Commits `6877677e` and `ccd029b2` added executable ABI evidence for product
  dequant/dequant-clamp paths; they do not claim performance or llama.cpp
  parity.

## Requirements

- Update `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md` with the
  current implementation calibration.
- Update Trellis spec wording to state that selected-body realization is a pass
  pipeline stage, not the opposite of a pass.
- Define resource-aware tuning as typed-body/capability candidate construction
  plus static legality/resource pruning before optional compile/asm/runtime
  feedback.
- Describe static/AOT, offline profile, and JIT/runtime modes without claiming
  all modes are implemented now.
- Preserve the existing authority model: typed body and provider facts are
  authority; artifact metadata, benchmark names, q8/q4 labels, and route ids are
  not.

## Acceptance Criteria

- [x] Gearbox v3 artifact distinguishes current MVP pass from future full
      resource-aware pass pipeline.
- [x] Variant pipeline spec names the pass-pipeline shape and tuning modes.
- [x] RVV plugin spec records that resource candidate seed is a transitional
      provider-owned contract until a dedicated pass materializes it.
- [x] No compiler source is changed.
- [x] `git diff --check` passes.

## Out Of Scope

- Do not implement or rename compiler passes in this task.
- Do not add runtime/benchmark claims.
- Do not run a full spec audit.
- Do not commit unless explicitly requested.

## Technical Notes

- Attached Pro suggestion:
  `/home/kingdom/.codex/attachments/9b5d9aa5-efe4-42ff-af36-6b79aecc4f7d/pasted-text.txt`
- Target files:
  `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
