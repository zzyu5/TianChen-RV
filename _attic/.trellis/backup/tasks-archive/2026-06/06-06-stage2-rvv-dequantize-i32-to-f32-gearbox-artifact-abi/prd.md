# Stage2 RVV dequantize i32-to-f32 gearbox artifact ABI boundary

## Goal

Make the existing `dequantize_i32_to_f32` selected-body RVV route family
truthfully executable as a generated RVV target artifact, or fail closed at
the exact missing artifact/ABI boundary. The owned boundary is typed
`tcrv_rvv` dequantization body facts, i32 source and f32 result element types,
runtime f32 scale parameter role, Gearbox schedule facts, SEW/LMUL/config and
policy, runtime AVL/VL, per-operand ABI/header bindings, RVV plugin-owned
route validation, common EmitC materialization, target artifact export,
generated bundle ABI, and `ssh rvv` correctness evidence.

## Why Now

Commit `08d9dd5b` closed the widening integer conversion executable-evidence
gap with `ssh rvv` proof for `widen_i16_to_i32` and `widen_i32_to_i64`. The
next bounded Stage 2 bottleneck should remain in the conversion/dtype class
but add runtime-scale conversion and Gearbox scheduling, rather than only SEW
widening.

## What I Already Know

- There was no active `.trellis/.current-task`; this task was created from the
  Hermes Direction Brief.
- The work is Stage 2 RVV coverage on corrected typed low-level `tcrv_rvv`
  selected bodies, not Stage 1 legacy `i32m1` route-authority growth.
- The authority chain is `tcrv.exec` selected RVV variant -> typed or realized
  `tcrv_rvv` body -> RVV plugin legality/selected-body realization/route
  provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact ->
  generated bundle -> `ssh rvv` evidence when runtime correctness is claimed.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles only. Source/result dtype, runtime scale semantics, Gearbox schedule,
  intrinsic spelling, ABI order, and route support must come from RVV provider
  facts derived from the selected typed body/config/runtime facts.
- Common EmitC/export may carry provider-built payloads and mirrors, but must
  not infer dequantization semantics from route ids, ABI strings, artifact
  metadata, test names, helper names, or exact intrinsic spellings.
- `RVVGearboxSchedules.cpp` validates the explicit selected dequantization body
  before adding Gearbox facts: i32 source vector, f32 result vector, source
  load from `lhs`, scale from `dequant-scale-value` named `scale`, runtime AVL
  from `n`, and one same-scope store to `out`.
- `RVVEmitCRouteProvider.cpp` calls the dequantization statement-plan owner and
  provider-facts verifier after selected-body analysis and before route
  construction.
- `RVVTargetArtifactRouteFamilyValidation.cpp` routes dequantization through
  the conversion dtype-policy validator, including runtime scale facts and
  Gearbox u2 second-slice statement validation.
- The existing script dry-run fixture already verifies materialized boundary,
  emission plan, target artifact export, generated harness, Gearbox u2 loop
  shape, runtime scale cases, and tail-preservation harness text. This round
  must audit whether a non-dry-run `ssh rvv` evidence gap or a production
  validation gap remains.

## Requirements

- Preserve RVV plugin ownership for dequantization family/provider facts,
  i32 source/f32 result dtype policy, runtime scale ABI role/type/name,
  Gearbox candidate/schedule facts, operand binding, statement planning,
  runtime AVL/VL, header/type summaries, and target validation contracts.
- Prove or repair the explicit selected-body `dequantize_i32_to_f32` path from
  Gearbox schedule materialization through provider route facts, common EmitC
  materialization, target artifact export, generated bundle compile, and
  `ssh rvv` correctness when executable behavior is claimed.
- Harden the production seam if audit shows dry-run-only support, stale route
  validation, missing provider preflight, missing conversion dtype-policy
  target validation, wrong source/result element type, wrong runtime scale
  binding, stale Gearbox schedule, wrong ABI/header binding, wrong generated C
  type, incorrect loop/VL mapping, or metadata/route-id/common-EmitC semantic
  authority.
- Add or retain focused fail-closed evidence for at least one stale or missing
  executable-boundary fact such as source/result dtype, scale runtime binding,
  Gearbox candidate/schedule, conversion relation, header/type summary, ABI
  order, runtime AVL/VL, or statement-plan leaf.
- Keep support levels separate: parseable/verifier-legal is not
  route-supported; route-supported is not executable without complete
  ABI/runtime/export support and real `ssh rvv` evidence for runtime claims.

## Acceptance Criteria

- [x] PRD and Trellis context identify the dequantize i32-to-f32 Gearbox
  executable artifact ABI boundary, non-goals, and relevant specs.
- [x] Repository audit records whether current explicit selected-body
  dequantization is dry-run-only, stale, under-validated, already
  production-valid, or too broad for a single round.
- [x] Production code is changed when the executable boundary is incomplete or
  under-validated; otherwise the PRD records a precise no-source-change
  justification backed by focused evidence.
- [x] Positive evidence covers the selected scope through Gearbox schedule
  materialization, emission plan, target artifact export, generated bundle
  compile, and `ssh rvv` correctness if runtime behavior is claimed.
- [x] Focused fail-closed evidence rejects at least one stale/missing
  dequantization executable-boundary fact before artifact acceptance or
  executable-route claim.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] Relevant generated-bundle dry-run test or underlying command passes.
- [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*` route
  authority, source-front-door route, descriptor compute path, or exact
  intrinsic spelling as route authority.
- [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean after commit.

## Out of Scope

- No broad dequantization matrix, dtype/LMUL clone batch, high-level
  Linalg/Vector/StableHLO frontend, per-Linalg route authority,
  source-front-door positive route, performance tuning database, dashboard, or
  report-only closeout.
- No product-reduce or clamp route expansion except as bounded references for
  the dequantization/runtime-scale seam.
- No widening integer conversion rework except as reference.
- No compare/select, MAcc, reduction, memory, segment2, contraction, or
  unrelated mask route rework.
- No common EmitC invention of RVV source/result dtype, runtime scale,
  Gearbox schedule, conversion relation, runtime VL, ABI, intrinsic spelling,
  or route support semantics.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, emission, or route semantics. Python
  may support generated-bundle tooling and evidence collection only.

## Technical Approach

1. Audit the Gearbox schedule pass, dequantization route-provider preflight,
   statement plan, target validator, generated-bundle script, and named tests.
2. Run focused dry-run and non-dry-run generated-bundle evidence for explicit
   selected-body `dequantize_i32_to_f32` with at least two nonzero scale values
   and counts crossing zero, single-lane, one-VL, tail, and multi-VL cases.
3. Patch production C++/MLIR/test/tooling only where the executable
   artifact/ABI seam is missing or stale.
4. Run focused C++ tests, generated-bundle dry-run/fail-closed checks, and
   non-dry-run `ssh rvv` generated-bundle evidence for any claimed executable
   behavior.
5. Finish/archive the Trellis task and commit one coherent change.

## Technical Notes

- Read specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Reference archive:
  - `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-widening-conversion-artifact-abi/`
- Bounded source/test targets from the Direction Brief:
  - `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`
  - `lib/Plugin/RVV/RVVGearboxSchedules.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.h`
  - `lib/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-dequantize-i32-to-f32-gearbox-dry-run.test`
  - `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue.mlir`

## Audit Conclusion

The production dequantize i32-to-f32 Gearbox artifact/ABI seam is already
provider-owned and fail-closed at the route and target artifact boundary. No
compiler source change was needed in this round.

- `RVVGearboxSchedules.cpp` validates the selected dequantization body before
  provider route construction. It requires the SEW32/LMUL m1 config, the
  `i32_to_f32_scaled` kind, relation
  `signed-i32m1-to-f32m1-scale-f32`, an i32 source vector, an f32 result
  vector, source load from `lhs`, runtime scale from
  `dequant-scale-value` named `scale`, runtime AVL from `n`, and exactly one
  same-scope store to `out`. It rejects stale Gearbox facts instead of
  overwriting them silently.
- `RVVEmitCRouteProvider.cpp` obtains the dequantization statement plan and
  calls `verifyRVVSelectedBodyDequantizationRouteProviderFacts(...)` before
  constructing the lowerable route.
- `RVVEmitCRoutePlanning.cpp` carries the dequantization route-family plan and
  verifies source/result dtype policy, source/result SEW-LMUL, runtime scale
  role/type/name, convert/scale intrinsics, route operand binding, runtime
  ABI order, runtime AVL/VL control, provider support mirror, Gearbox
  candidate/schedule facts, and stale widening/non-dequant residue.
- `RVVTargetArtifactRouteFamilyValidation.cpp` dispatches dequantization
  through the conversion dtype-policy target validator. The validator checks
  the runtime AVL/VL selected-boundary contract, runtime ABI parameters,
  source/result dtype facts, runtime scale facts, header/type/binding mirrors,
  and the Gearbox u2 second-slice route statement plan before accepting the
  target artifact.
- `scripts/rvv_generated_bundle_abi_e2e.py` already supports explicit
  selected-body `dequantize_i32_to_f32` dry-run and non-dry-run flows, inserts
  `--tcrv-rvv-materialize-gearbox-schedules`, verifies emitted Gearbox u2
  loop shape, generates the external C ABI harness, runs two signed i32 source
  patterns across two nonzero scale values, and checks source and output-tail
  preservation.

The current gap was executable evidence, not source behavior. This round
produced non-dry-run `ssh rvv` correctness evidence for the explicit
selected-body `dequantize_i32_to_f32` Gearbox route.

## Implementation Summary

- Created the Trellis task and bounded PRD from the Hermes Direction Brief.
- Audited the Gearbox schedule pass, route provider preflight,
  dequantization statement-plan owner, conversion dtype-policy target
  validator, generated-bundle harness, and named fixtures.
- No production source files were changed because the audited seam already
  satisfies the provider-owned route/ABI/header/type/runtime-scale/Gearbox
  validation contract and focused checks did not expose a stale or
  under-validated production boundary.
- No `.trellis/spec/` update was needed: the existing RVV plugin, EmitC route,
  emission runtime, and testing specs already cover provider-owned route
  authority, runtime ABI/header mirrors, Gearbox schedule validation, target
  artifact fail-closed behavior, and `ssh rvv` evidence requirements.

## Evidence

Positive executable `ssh rvv` evidence:

- Command:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/stage2-rvv-dequantize-i32-to-f32-gearbox-audit --run-id explicit-dequant-ssh --overwrite --op-kind dequantize_i32_to_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --dequant-scale -0.125 --dequant-scale 0.375 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
- Result: `rvv_generated_bundle_abi_e2e: success`.
- Artifact root:
  `artifacts/tmp/stage2-rvv-dequantize-i32-to-f32-gearbox-audit/explicit-dequant-ssh`.
- `dequantize_i32_to_f32` executed on `ssh rvv` for counts `0,1,16,17,257`,
  patterns `0,1`, and scales `-0.125,0.375`.
- Every runtime case reported `source_preserved` and `tail_preserved` with
  f32 tolerance `1e-05`.
- Final markers:
  `tcrv_rvv_generated_bundle_abi_dequantize_i32_to_f32_ok counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 tolerance=1e-05`
  and
  `PASS op=dequantize_i32_to_f32 counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 tolerance=1e-05`.

Positive dry-run evidence:

- Initial command using unversioned `llvm-readobj` failed with
  `required tool not found: llvm-readobj`. This was a host tool-path issue,
  not a dequantization route failure.
- Re-run command:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/stage2-rvv-dequantize-i32-to-f32-gearbox-audit --run-id explicit-dequant-dry-run --overwrite --op-kind dequantize_i32_to_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --dequant-scale -0.125 --dequant-scale 0.375 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
- Result: `rvv_generated_bundle_abi_e2e: dry_run_success`.
- Artifact root:
  `artifacts/tmp/stage2-rvv-dequantize-i32-to-f32-gearbox-audit/explicit-dequant-dry-run`.

Focused fail-closed evidence:

- Missing Gearbox schedule materialization failed before provider route
  construction with:
  `requires pass-produced RVV Gearbox candidate-selection fact 'tcrv_rvv.gearbox.candidate_set' on tcrv_rvv.with_vl before provider route construction`.
- Stale runtime scale role failed during target artifact candidate validation
  with:
  `candidate tcrv_rvv.dequant_scale_role provenance must mirror selected typed RVV dequantization scale ABI role 'dequant-scale-value' but was 'output-buffer'`.
- Stale Gearbox schedule id failed during target artifact candidate validation
  with:
  `candidate tcrv_rvv.gearbox.schedule_id provenance must mirror selected typed RVV dequantization Gearbox schedule id 'rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1' but was 'artifact-name-derived-gear'`.
- Stale dequantization relation failed during target artifact candidate
  validation with:
  `candidate tcrv_rvv.dequantization_relation provenance must mirror selected typed RVV dequantization relation 'signed-i32m1-to-f32m1-scale-f32' but was 'script-derived-dequant'`.

Build and local checks:

- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `./build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 -m lit --version` failed because local `lit` is not installed, so
  the relevant lit checks were validated through the underlying script/tool
  commands instead of the lit runner.
- Bounded old-authority scan found only existing provider-derived intrinsic
  spellings and negative test markers in already-present script/test/source
  files; this round added no production code and no positive legacy
  route-authority surface.

