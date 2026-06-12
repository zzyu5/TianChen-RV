# Stage2 RVV i32-to-f32 dequant executable ABI closure

## Goal

Close the executable ABI boundary for the just-added typed RVV
i32-to-f32 runtime-scale dequantization route. The task must prove that one
selected `tcrv.exec` RVV variant with a typed `tcrv_rvv.dequantize` body can
produce a generated object/header bundle, be consumed by an external C ABI
harness, compile on the real `ssh rvv` target, and match a host/reference
calculation at runtime.

This is an executable-closure task for the existing route-supported dequant
slice introduced by commit `4f39033d`; it is not a new route-coverage task.

## What I Already Know

- The session began in `/home/kingdom/phdworks/TianchenRV` on `main` with a
  clean worktree at `4f39033d rvv: add i32 to f32 dequant route foundation`.
- No current Trellis task existed, so this task was created from the Hermes
  Direction Brief.
- The archived task
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-i32-to-f32-dequant-route-foundation/`
  completed route-supported typed `tcrv_rvv.dequantize` support and explicitly
  did not claim executable `ssh rvv` correctness.
- `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  and `.trellis/spec/lowering-runtime/emitc-route.md` require the route
  authority to remain in selected typed `tcrv_rvv` body/config/runtime facts and
  provider-owned route facts. Common EmitC/export and artifact metadata are
  mirrors only.
- Existing fixtures:
  `test/Dialect/RVV/generic-i32-to-f32-dequantization-dataflow.mlir` verifies
  the dialect/dataflow boundary, and
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`
  verifies emission-plan/header artifact facts for the selected-body route.
- `scripts/rvv_generated_bundle_abi_e2e.py` already provides generated-bundle
  dry-run and `ssh rvv` executable evidence for many selected-body RVV routes,
  but it does not yet include `dequantize_i32_to_f32` as an executable op-kind.

## Requirements

- Use the existing typed dequant selected-body route as authority:
  `tcrv_rvv.load` i32 source, `tcrv_rvv.dequantize` with runtime scale role
  `dequant-scale-value`, f32 result vector, `tcrv_rvv.store` f32 output, and
  runtime `n`/AVL.
- Add generated-bundle ABI evidence support for exactly one bounded dequant
  artifact:
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`.
- Bind and verify the generated C ABI parameter order and types:
  `const int32_t *lhs`, `float scale`, `float *out`, `size_t n`.
- The external harness must compare against a host/reference calculation over
  multiple runtime counts, signed i32 source patterns, and at least two
  nontrivial scale values.
- The f32 tolerance must be explicit and checked in the harness.
- The harness must verify output tail sentinel preservation and source buffer
  preservation.
- Run a dry-run artifact generation path before remote execution.
- Compile and run on `ssh rvv` before claiming executable correctness.
- If evidence exposes a production route/verifier/provider/target bug, repair
  production code and rerun focused checks.
- If evidence does not expose a production bug, keep source changes to neutral
  evidence/harness support and task metadata.

## Acceptance Criteria

- [ ] `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind dequantize_i32_to_f32`
  generates and verifies the representative dequant object/header bundle.
- [ ] The script accepts at least two runtime scale values for dequant evidence
  and rejects insufficient scale coverage for `dequantize_i32_to_f32`.
- [ ] The generated external C harness calls the produced C ABI symbol with
  `lhs, scale, out, n`, checks `fabsf(out[i] - ((float)lhs[i] * scale))` within
  an explicit tolerance, and verifies source/tail sentinel preservation.
- [ ] Real `ssh rvv` compile/run evidence passes for representative counts,
  signed source patterns, and scale variants.
- [ ] If compiler production files are not changed, no new route coverage or
  route authority is added through script names, ABI names, artifact names, or
  metadata mirrors.
- [ ] If the dequant fixture is touched, the focused `tcrv-opt`/FileCheck path
  for it passes.
- [ ] If provider/target production code changes, `tianchenrv-rvv-extension-plugin-test`
  and `tianchenrv-target-artifact-export-test` pass.
- [ ] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and the
  script self-test pass.
- [ ] Bounded old-authority and q-name-authority scan over touched files finds
  no new positive legacy authority.
- [ ] `git diff --check` and `git diff --cached --check` pass.
- [ ] The task is finished/archived and one coherent commit is created if all
  acceptance criteria are satisfied; otherwise the task remains open with the
  exact continuation point.

## Definition Of Done

- One generated typed RVV i32-to-f32 dequantization artifact has executable ABI
  evidence on the real RVV target.
- The final report states whether production code changed or only neutral
  harness/evidence support changed.
- The final report includes the artifact/op exercised, dry-run and `ssh rvv`
  command/result summary, scale/tolerance/reference details, checks, old
  authority scan, task archive status, commit hash, and worktree status.

## Out Of Scope

- Zero-point, clamp, fused product-reduction-plus-dequant full-kernel closure,
  q8/q4/llama benchmark-specific routes, high-level Linalg/frontend scope, or
  broad dtype/LMUL clone batches.
- New route coverage, new dequant operation forms, or compatibility wrappers
  preserving old i32m1 authority.
- Handwritten C demos as the main deliverable; the harness must consume the
  generated header/object bundle.
- Repeated product-reduction executable evidence or broad smoke matrices.
- Inferring dtype, compute, schedule, scale semantics, or support from route
  ids, artifact names, op-kind strings, ABI names, common EmitC, or metadata
  mirrors.

## Technical Approach

Add dequant support to the existing generated-bundle evidence tool:

- Add a `dequantize_i32_to_f32` expectation for the explicit selected-body
  dequant fixture and generated C ABI symbol.
- Extend expectation predicates and metadata/header verification for f32 output
  and provider-derived dequant fields.
- Add a dequant-specific harness branch that allocates signed i32 input,
  f32 output sentinels, loops over counts/source patterns/scale values, calls
  the generated function, compares to a reference expression with explicit
  tolerance, and checks preservation invariants.
- Add CLI/self-test support for repeated `--dequant-scale` values while keeping
  existing integer route evidence unchanged.
- Run dry-run first, then real `ssh rvv` compile/run evidence.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-i32-to-f32-dequant-route-foundation/prd.md`.
- Primary implementation target:
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Primary fixture:
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`.
