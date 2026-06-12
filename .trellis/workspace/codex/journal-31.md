> Continuation from `journal-30.md` (archived at ~1972 lines)

## Session 582: RVV same-target measured candidate feedback loop

**Date**: 2026-06-12
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the real same-target measured
candidate feedback milestone for the bounded
`widening_product_reduce_dequantize_f32` low-precision Gearbox family. The
candidate-labelled bridge from the previous slice was exercised on real
`ssh rvv` for both `grouped-u2` and `packed-i4` selected-body fixtures.

### Boundary Classification

- Compiler authority remains the selected typed `tcrv_rvv` body/config/runtime
  facts and provider-owned low-precision resource candidate facts.
- Candidate labels, measurement classifications, speedup ranges, object/header
  paths, artifact directories, and root feedback status are policy/evidence
  routing state only.
- Both candidate feedback records keep
  `measurement_result_is_route_authority = false`; no measured outcome was
  promoted into route, schedule, typed-body, artifact/header, provider maturity,
  or dispatch authority.

### Main Changes

- Ran non-dry-run `rvv_generated_bundle_same_target_measure.py` on `ssh rvv`
  with both `--candidate-input grouped-u2=...` and
  `--candidate-input packed-i4=...` under one
  `widening_product_reduce_dequantize_f32` op family.
- Recorded evidence under
  `artifacts/gate4-candidate-feedback-ssh/gate4-candidate-feedback-ssh`.
- Hardened the same-target measurement bridge so non-dry-run per-candidate
  evidence roots carry `same_target_measurement = true`, candidate feedback
  records use `same-target-measured`, and root `candidate_feedback_loop.status`
  uses `candidate-same-target-measured`.
- Kept dry-run feedback wording unchanged:
  `ready-for-same-target-measurement`.
- Updated the RVV plugin and EmitC route specs with the measured-feedback status
  convention and the non-authority boundary.
- Updated the active PRD and task metadata while keeping the macro task
  `in_progress`.

### Evidence

- Root evidence:
  `artifacts/gate4-candidate-feedback-ssh/gate4-candidate-feedback-ssh/evidence.json`
  records `status = success`, `dry_run = false`, `ssh_evidence = true`,
  `same_target_measurement = true`, and
  `candidate_feedback_loop.status = candidate-same-target-measured`.
- `grouped-u2`: selected candidate index `2`, candidate/legal count `3/3`,
  baseline `scalar-c-reference/product-reduction-dequant-v1`, 12 summary
  records, 60 measurement records, classification `regression`, best-speedup
  range `0.501820..0.610294`.
- `packed-i4`: selected candidate index `3`, candidate/legal count `3/3`,
  baseline `scalar-c-reference/product-reduction-dequant-packed-i4-v1`, 12
  summary records, 60 measurement records, classification `no-win`,
  best-speedup range `0.901869..1.021921`.

### Self-Repair

- The first real measurement attempt failed before bundle generation because
  `llvm-readobj` was not on PATH. Reran with
  `/usr/lib/llvm-20/bin/llvm-readobj`.
- The first successful evidence inspection showed non-dry-run candidate feedback
  still used dry-run/exportability status wording at the root/per-candidate
  boundary. The script now distinguishes measured status while keeping the
  result non-authoritative.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] Non-dry-run `ssh rvv` candidate measurement command with counts
  `257,4096,65536`, warmups `2`, repeats `5`, iterations `8`.
- [OK] Final evidence jq checks for root status, per-candidate selected indexes,
  object/header SHA256s, parsed timing counts, remote compile/run success, and
  `measurement_result_is_route_authority = false`.
- Remaining focused lit, Trellis validation, scans, diff whitespace checks, and
  staged checks are run after this journal entry.

### Status

[OPEN MACRO TASK] The measured candidate feedback milestone is complete. The
macro task remains active for reviewing whether the source-backed grouped-u2
regression and packed-i4 no-win evidence require explicit provider-side
admission/remediation-policy updates.

### Git Commits

Final coherent commit is created after this journal entry.
