# Stage2 RVV Gearbox-selected dequant executable ABI closure

## Goal

Close executable evidence for the existing typed
`dequantize_i32_to_f32` RVV Gearbox route after the previous round made the
selected u2 schedule affect provider route planning. The selected
`tcrv.exec` RVV variant must flow through Gearbox schedule materialization,
provider route construction, neutral target bundle export, external ABI
compilation, and real `ssh rvv` correctness execution for the selected u2
two-slice schedule.

## What I already know

- No `.trellis/.current-task` existed when this round began, so this task was
  created from the Hermes Direction Brief.
- Commit `b5bfa269` completed route-supported two-candidate Gearbox
  realization for `dequantize_i32_to_f32`: u2 is selected, loop step becomes
  `full_chunk_vl * 2`, and the route statement plan contains a second
  setvl/load/convert/scale/store slice.
- The route authority remains the selected typed `tcrv_rvv` body/config/runtime
  facts and RVV provider validation. Candidate metadata and exported bundle
  fields are mirrors only.
- The existing positive fixture
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`
  already checks provider-derived Gearbox metadata and stale mirror rejection.
- The existing script `scripts/rvv_generated_bundle_abi_e2e.py` already has
  `dequantize_i32_to_f32` support, default runtime counts, runtime scale cases,
  f32 absolute tolerance, and generated external ABI harness plumbing.

## Requirements

- Reuse the existing typed `dequantize_i32_to_f32` route and generated bundle
  tooling. Do not add a new route family, frontend, benchmark path, descriptor
  path, or common/core Gearbox selection.
- Generated artifact evidence must prove before runtime that the provider
  selected u2 schedule is present: candidate set, selected candidate, schedule
  id, u2 unroll, u2 loop step, and second-slice statement plan.
- ABI binding must cover `lhs`, `scale`, `out`, and `n` from provider-derived
  runtime ABI facts. The harness must not derive semantics from artifact name,
  route string, ABI string, q-name, descriptor residue, or intrinsic spelling.
- Runtime correctness must compare the generated RVV function output against a
  host/reference dequant computation `((float)lhs[i]) * scale`.
- Runtime cases must include small, full-chunk, two-chunk, and tail counts, and
  at least two nonzero scale values. The f32 tolerance policy must be explicit.
- Negative/preflight checks must reject stale u1/u2 mirrors, missing selected
  schedule materialization, stale u2 loop structure, selected candidate
  mismatch, and route-string/artifact-name authority before execution.
- The implementation may minimally extend the evidence script or tests if the
  existing harness lacks u2-specific preflight checks or count coverage. Keep
  production route authority in provider facts.
- Finish the Trellis round truthfully, including focused checks, `ssh rvv`
  evidence, archive status, and one coherent commit if complete.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      executable-closure task and its spec basis.
- [x] The generated bundle/evidence path verifies selected u2 Gearbox route
      facts before runtime: legal candidate set, selected candidate, schedule
      id, unroll `2`, loop step `full_chunk_vl * 2`, and second-slice
      setvl/load/convert/scale/store plan.
- [x] The external ABI harness binds and calls the generated function with
      `lhs`, `scale`, `out`, and `n`, and records that runtime count/scale
      cases are execution evidence rather than route authority.
- [x] `ssh rvv` correctness runs compare generated RVV output with the host
      reference for small, full-chunk, two-chunk, and tail runtime counts across
      at least two nonzero scale values.
- [x] The tolerance policy for f32 dequant comparison is recorded in the
      evidence output and enforced by the harness.
- [x] Negative/preflight tests or self-tests reject stale/missing Gearbox
      selection, stale u2 loop-step/body-plan materialization, selected
      candidate mismatch, and route-string/artifact-name authority before
      execution.
- [x] Focused build/test checks pass for `tcrv-opt`, `tcrv-translate`,
      relevant RVV target/plugin tests, and the generated bundle ABI evidence
      script.
- [x] `git diff --check`, `git diff --cached --check`, and bounded
      old-authority/q-name scans over touched files pass.
- [x] Trellis task status, archive state, journal entry, commit hash, and final
      worktree status are recorded truthfully.

## Out of Scope

- Runtime benchmarking, performance claims, persistent tuning cache, assembly
  feedback, cross-kernel autotuning, or broad smoke matrices.
- New high-level Linalg/Vector/StableHLO frontend work.
- New route families beyond `dequantize_i32_to_f32`.
- Direct C/intrinsic macro schedule authority.
- Common/core RVV schedule selection.
- Treating artifact metadata, route strings, ABI names, q-names, descriptors,
  or intrinsic spellings as route authority.

## Technical Notes

- Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`.
- Previous task:
  `.trellis/tasks/archive/2026-06/06-05-rvv-gearbox-two-candidate-dequant-route-realization/`.
- Main files to inspect:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Decision

Use the existing generated bundle ABI evidence script as the production-facing
evidence bridge, and extend it only where necessary to make the selected u2
Gearbox schedule a mandatory preflight condition for
`dequantize_i32_to_f32`. If the generated artifact lacks enough u2 structure
to verify before runtime, repair the provider/export/test path so those facts
come from provider route construction and target validation, not from harness
name matching.

## Completion Evidence

- Production repair: Common EmitC now materializes safe left-associative
  `+`/`-` expression chains so provider-owned statement-plan expressions such
  as `n - offset - vl`, `lhs + offset + vl`, and `out + offset + vl` lower to
  scoped EmitC SSA instead of unsafe literal text.
- Provider route-plan repair: the dequant u2 second-slice statement plan and
  route validation now use safely spaced expressions, and target artifact tests
  expect the same provider-derived expressions.
- Evidence bridge: `scripts/rvv_generated_bundle_abi_e2e.py` runs
  `--tcrv-rvv-materialize-gearbox-schedules` for
  `dequantize_i32_to_f32`, records Gearbox candidate/selected/schedule facts,
  requires `full_chunk_vl * 2`, requires the second
  setvl/load/convert/scale/store slice before runtime, and records f32
  tolerance `1e-05`.
- Added focused script test:
  `test/Scripts/rvv-generated-bundle-abi-e2e-dequantize-i32-to-f32-gearbox-dry-run.test`.
- Artifact evidence path used for real runtime:
  `artifacts/tmp/rvv_gearbox_dequant_executable/ssh-rvv-final`.
- Real RVV command:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_gearbox_dequant_executable --run-id ssh-rvv-final --overwrite --op-kind dequantize_i32_to_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --dequant-scale -0.125 --dequant-scale 0.375 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 120 --connect-timeout 15`.
- `ssh rvv` result: PASS for counts `0,1,16,17,257`, patterns `0,1`,
  scales `-0.125,0.375`, source preservation, tail sentinel preservation, and
  tolerance `1e-05`.
- Commit hash and final worktree status are recorded in the final report rather
  than embedded in this PRD, because embedding the final commit hash in the
  committed file would be self-invalidating.
