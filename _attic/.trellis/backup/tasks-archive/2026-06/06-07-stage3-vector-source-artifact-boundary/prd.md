# Stage3 bounded Vector source-front-door RVV executable artifact boundary

## Goal

Carry the newly materialized bounded MLIR Vector add source-front-door path one
step farther through generated artifact export, generated bundle ABI
consumption, and `ssh rvv` correctness evidence. The workflow must start from
the explicit source-front-door input, materialize a selected `tcrv.exec` RVV
dispatch case with a typed generic `tcrv_rvv` add body, use the RVV
plugin-owned route/provider facts, export the target artifact bundle, and
execute the generated external ABI harness on the RVV target when correctness
is claimed.

## What I Already Know

* The worktree was clean before this task was created; current HEAD is
  `1dd838b0 rvv: materialize vector source boundary`.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-07-stage3-vector-selected-rvv-boundary/`
  added the bounded `tcrv-rvv-materialize-vector-add-source-front-door`
  materializer and stopped at materialization/provider/export/header smoke
  evidence without runtime correctness.
* `.trellis/spec/index.md` and `.trellis/spec/extension-plugins/rvv-plugin.md`
  keep the RVV authority chain as selected `tcrv.exec` envelope -> typed
  `tcrv_rvv` body -> RVV provider -> `TCRVEmitCLowerableRoute` -> Common
  EmitC -> target artifact -> `ssh rvv` evidence for runtime/correctness
  claims.
* `.trellis/spec/lowering-runtime/emitc-route.md` says Common EmitC/export may
  carry provider payloads but must not infer RVV dtype, SEW, LMUL, operation,
  ABI, schedule, or intrinsic semantics from source marker names, route ids, or
  artifact metadata.
* `scripts/rvv_generated_bundle_abi_e2e.py` already proves generated
  object/header bundle ABI consumption for explicit selected typed RVV bodies
  and can run the generated harness on `ssh rvv` when not in dry-run mode.
* The same script still treats legacy `--source-seed` as unsupported. This
  task must not revive that legacy source route; it needs a distinct bounded
  opt-in mode for the corrected Vector add source-front-door materializer.

## Requirements

* Add one bounded generated-bundle evidence path for the corrected Vector add
  source-front-door materializer. It must be explicit opt-in and limited to the
  already materialized rank-1 i32 Vector add source pattern.
* The evidence path must run the production materializer pass before emission
  planning and target bundle export:
  source MLIR -> `--tcrv-rvv-materialize-vector-add-source-front-door` ->
  `--tcrv-materialize-emission-plans` -> RVV EmitC C/C++ export -> target
  artifact bundle export -> external ABI harness -> optional `ssh rvv`.
* The source marker may select the bounded materializer, but evidence and
  acceptance must remain tied to the selected typed `tcrv_rvv` body/provider
  route and target artifact validator, not marker text, Vector op names,
  route-id authority, diagnostics, artifact metadata, or Common EmitC inference.
* The generated harness must cover runtime `n` edge cases and tail/source
  preservation. Runtime counts are execution cases only and must not be treated
  as dispatch, dtype, or route authority.
* Keep legacy `--source-seed` fail-closed and do not broaden Vector/Linalg or
  add new RVV ops/dtype/LMUL coverage.
* Add focused fail-closed evidence for stale or unsupported source-front-door
  mode use, such as requesting non-add op kinds in the bounded source mode or
  mixing the source-front-door mode with selected-body modes.

## Acceptance Criteria

* [x] The source-front-door mode accepts only `--op-kind add` and starts from
      `test/Transforms/RVV/rvv-vector-add-source-front-door.mlir` or a
      compatible single `--input` override.
* [x] Dry-run generated-bundle evidence proves the source input was copied,
      materialized through `tcrv-rvv-materialize-vector-add-source-front-door`,
      exported as a target bundle, and validated with object/header/prototype
      metadata for `@rvv_vector_add`.
* [x] Evidence JSON records that the source marker is only an opt-in
      materialization boundary, while route/runtime facts come from the
      selected typed `tcrv_rvv` body and RVV provider.
* [x] The generated harness checks active-lane add correctness, output tail
      preservation beyond runtime `n`, and source buffer preservation.
* [x] Non-dry-run execution on `ssh rvv` passes before any runtime/correctness
      claim is made.
* [x] A focused negative test proves unsupported/stale source-front-door mode
      usage fails before bundle generation.
* [x] Existing legacy `--source-seed` unsupported behavior remains unchanged.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit/script tests for the source-front-door generated-bundle path
      pass.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] A bounded old-authority scan over touched files and added diff lines
      shows no new positive legacy `RVVI32M1`, `rvv-i32m1`,
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor-driven compute,
      source-front-door route-id authority, or Common EmitC semantic branch.

## Out Of Scope

* No broad Vector, Linalg, StableHLO, tensor/tile, frontend, or per-Linalg
  route generalization.
* No additional Vector source ops beyond the bounded i32 add pattern.
* No old `i32m1` compatibility route and no dtype/LMUL clone batch.
* No source-front-door route authority from names, markers, diagnostics, route
  ids, artifact metadata, or Common EmitC.
* No performance claim or tuning database work.
* No report-only closeout; the round must either add the executable artifact
  boundary seam or leave the exact blocked continuation point.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on
  2026-06-07.
* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  and the shared capability/plugin/compute guides.
* Primary implementation candidate:
  `scripts/rvv_generated_bundle_abi_e2e.py`.
* Primary focused test candidates:
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-source-front-door-dry-run.test`
  and a companion fail-closed script test.
* Reference implementation/evidence files:
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `test/Transforms/RVV/rvv-vector-add-source-front-door.mlir`,
  `test/Transforms/RVV/rvv-vector-add-source-front-door-negative.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test`,
  and `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`.

## Completion Evidence

* Added `--vector-source-front-door` generated-bundle mode to
  `scripts/rvv_generated_bundle_abi_e2e.py`. The mode is mutually exclusive
  with selected-body modes, accepts only the bounded add expectation, runs
  `--tcrv-rvv-materialize-vector-add-source-front-door`, and then uses the
  existing selected typed-body route/export path.
* The evidence JSON records the source-front-door marker as an explicit opt-in
  materialization boundary only. The recorded route authority remains the
  selected typed `tcrv_rvv` body/config/runtime facts consumed by the RVV
  provider.
* The generated C harness now checks active-lane add results for runtime counts
  `0`, `1`, `17`, and `257`; verifies that `lhs` and `rhs` are preserved; and
  verifies output tail sentinel preservation beyond runtime `n`.
* Added dry-run lit coverage for source copy, materializer command, bundle
  export metadata, provider-supported mirror fields, selected dispatch mirrors,
  prototype/header mapping, and harness source.
* Added fail-closed lit coverage for unsupported source-front-door op kinds and
  mixed source-front-door/selected-body modes before bundle generation.
* Non-dry-run `ssh rvv` execution passed with:
  `PASS op=add counts=0,1,17,257 source_preserved tail_preserved`.
