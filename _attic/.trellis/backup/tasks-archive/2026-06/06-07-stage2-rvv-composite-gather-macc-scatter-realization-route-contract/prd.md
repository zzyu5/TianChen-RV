# Stage2 RVV composite gather-MAcc-scatter selected-body realization and route contract

## Goal

Replace the newly exposed fail-closed composite RVV seam with a bounded
plugin-owned production workflow for the base masked indexed gather -> masked
MAcc/add accumulation -> indexed scatter path. The selected or pre-realized
composite `tcrv_rvv` body must be recognized by RVV-owned code, realized into
one coherent selected-body form when needed, and consumed by the RVV route
provider to build one `TCRVEmitCLowerableRoute` from typed body/config/runtime
facts. If any required fact is missing or stale, the same owner must fail
closed with a precise diagnostic before Common EmitC or target artifact export
can infer semantics.

## What I already know

- Repository started clean on `main` at
  `7910fa71 rvv: fail close composite gather macc scatter boundary`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
- The archived predecessor task
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-masked-indexed-gather-macc-scatter-abi-boundary/`
  deliberately added fail-closed diagnostics for explicit and pre-realized
  runtime-scalar compare + computed-mask + indexed gather + masked MAcc +
  indexed scatter composites.
- Specs require the authority chain to remain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/selected-body realization/route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral Common EmitC
  materialization.
- Common EmitC, artifact metadata, route ids, helper names, test names,
  descriptor residue, and C ABI strings must not become RVV semantic
  authority.

## Requirements

- Recognize the base composite body shape in RVV-owned selected-body analysis:
  runtime scalar compare, computed mask, masked indexed gather load, masked
  MAcc/add accumulation with accumulator role, masked indexed scatter store,
  runtime AVL/VL, inactive-lane/scatter policy, and ordered ABI/runtime values.
- Add or wire a plugin-local composite selected-body realization path for the
  corresponding pre-realized multi-family composite boundary.
- Build exactly one RVV-provider-owned route contract for the positive base
  composite path, using typed gather/index/mask/MAcc/scatter/accumulator,
  dtype/config, ABI/header order, policy, and AVL/VL facts.
- Keep unsupported or stale variants fail-closed with targeted diagnostics for
  missing gather index, runtime scalar compare, computed mask, inactive-lane
  policy, MAcc accumulator/operand role, scatter destination, ABI order,
  dtype/config, header/prototype, or AVL/VL facts.
- Preserve Common EmitC neutrality: it may only consume the provider-built
  route payload and must not derive RVV semantics.
- If full artifact/export materialization is too large for this round, finish
  the positive selected-body realization plus route-supported contract first
  and record target artifact/generated-bundle/`ssh rvv` evidence as the next
  continuation point.

## Acceptance Criteria

- [x] Explicit composite typed body no longer reports the predecessor
  fail-closed "missing composite owner/provider boundary" diagnostic when all
  required facts are present; it reaches the intended plugin-owned composite
  route-supported boundary.
- [x] Pre-realized multi-family composite body is consumed by a plugin-local
  composite realization path and reaches the same route-supported boundary, or
  fails closed only for a named missing fact. This round keeps it fail-closed
  at the named composite realization-owner boundary.
- [x] Focused negative tests cover stale or missing facts for gather index,
  runtime scalar compare, computed mask, inactive-lane/scatter policy, MAcc
  accumulator/operand role, scatter destination, ABI order, dtype/config,
  header/prototype, or AVL/VL facts where the implementation touches those
  boundaries. This round adds a stale scatter-value negative case and preserves
  existing focused coverage for the other touched boundaries.
- [x] Existing isolated runtime-scalar indexed gather, computed-mask MAcc, and
  indexed scatter behavior continues to pass.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target
  artifact/export code or mirrors are touched; otherwise this PRD records why
  it was not required.
- [x] Bounded old-authority scan over touched files and added diff lines shows
  no new descriptor/direct-C/source-front-door/legacy `tcrv_rvv.i32_*`,
  `RVVI32M1`, or `rvv-i32m1` route authority. Added exact intrinsic strings
  appear only in provider-derived leaf mirror assertions, not route selection
  authority.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

- PRD, task context, and journal truthfully describe whether this round reached
  positive route-supported contract only, artifact/export, or runtime evidence.
- Source and test changes are focused on the composite RVV owner boundary.
- Trellis task is completed and archived only after focused checks pass.
- One coherent commit is created when the task is complete; otherwise the task
  remains open with an exact continuation point.

## Out of Scope

- Broad composite-kernel framework.
- High-level Linalg/Vector/StableHLO frontend work.
- Source-front-door positive routes or per-Linalg route authority.
- Generic autotuning/performance databases or dashboards.
- Broad dtype/LMUL clone batches.
- Unrelated gather/scatter/memory/mask/MAcc matrix expansion.
- Common EmitC invention of RVV semantics.
- Compatibility wrappers that hide the fail-closed gap.
- Executable artifact or `ssh rvv` correctness claims until the plugin-owned
  composite realization and provider route contract structurally exist.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/variant-pipeline/index.md`
- Predecessor task read:
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-masked-indexed-gather-macc-scatter-abi-boundary/task.json`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-masked-indexed-gather-macc-scatter-abi-boundary/prd.md`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-masked-indexed-gather-macc-scatter-abi-boundary/implement.jsonl`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-masked-indexed-gather-macc-scatter-abi-boundary/check.jsonl`
- Current code owner files to inspect before implementation:
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - directly related gather, MAcc, compare/mask, scatter, statement-plan, and
    route-provider owner files as needed.

## Implementation Scope Decision

This round will finish the positive plugin-owned selected-body realization and
route-supported contract only. It will not claim target artifact/export or
`ssh rvv` correctness for the composite unless the provider route can be
materialized through the existing target exporter without broad artifact work.
The expected next continuation point, if export is not reached, is generated
bundle/header mirror evidence for the already provider-built composite route.

## Completion Notes

- Completed a positive explicit selected-body route-supported contract for the
  runtime-scalar computed-mask indexed gather-MAcc-scatter path.
- The pre-realized multi-family composite remains fail-closed at the named
  composite realization-owner boundary; the next compiler continuation is a
  plugin-local realization owner that rewrites those pre-realized family bodies
  into the explicit realized body shape.
- No target artifact/generated-bundle or `ssh rvv` runtime correctness claim is
  made in this round.
