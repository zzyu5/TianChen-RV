# Stage2 RVV Gearbox low-precision resource-fact generated artifact boundary

## Goal

Prove and, if needed, repair the generated artifact/runtime boundary for the
existing Gearbox low-precision resource-fact seam on
`widening_product_reduce_dequantize_f32`.

The bounded workflow is:

```text
selected/pre-realized typed tcrv_rvv body
  -> RVV Gearbox pass-produced tcrv_rvv.low_precision_resource facts
  -> selected-body realization fact propagation
  -> RVV provider-owned resource validation and route construction
  -> common EmitC materialization
  -> generated target artifact/bundle ABI
  -> ssh rvv correctness evidence when executable behavior is claimed
```

The task is not to broaden low-precision coverage. It is to make this single
representative seam executable through generated artifacts, or to fail closed
with targeted diagnostics when the required resource facts, schedule legality,
memory form, budget, ABI/header binding, runtime AVL/VL, or provider mirror
facts are missing, stale, or inconsistent.

## What I Already Know

- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief before source edits.
- Initial repo state was `/home/kingdom/phdworks/TianchenRV`, branch `main`,
  clean `git status --short`, with HEAD
  `48a9a812 feat(rvv): consume gearbox low precision resource facts`.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-06-rvv-gearbox-low-precision-resource-candidate-pass/`
  introduced a plugin-local Gearbox schedule/resource pass seam, provider
  consumption/validation, selected-body realization propagation, target export
  mirrors, and focused tests.
- The previous task explicitly made no `ssh rvv` runtime claim. This round must
  close or repair the generated artifact/runtime boundary before claiming
  executable behavior.
- Long-term authority remains structural: route/resource facts come from typed
  `tcrv_rvv` body/config/capability/runtime facts and RVV plugin validation,
  not from route ids, metadata, helper names, test names, artifact names, ABI
  strings, common EmitC branches, descriptor residue, or old i32 route
  authority.
- Memory-derived context reinforces that low-precision q8/q4 examples should
  be handled as broader RVV Stage 2 low-precision/quantized contraction maturity
  work, not q8-named or helper-named route authority.

## Requirements

- Generated-bundle materialization for
  `widening_product_reduce_dequantize_f32` must run the Gearbox resource-fact
  pass before provider planning when the workflow claims executable behavior.
- Generated artifacts must carry provider/resource mirror facts only. Mirrors
  are validation evidence, not route/resource authority.
- The RVV provider must accept the route only after validating pass-produced
  resource facts against the selected body/config/capability/runtime facts.
- Pre-realized selected-body realization must preserve the pass-produced
  low-precision resource facts into the realized body used by provider
  planning.
- Missing or stale resource facts must fail closed with targeted diagnostics for
  at least one focused negative case.
- If artifact generation or runtime execution already works, this task may
  close with focused executable evidence and no source changes beyond truthful
  Trellis notes.
- If evidence exposes a real production seam bug, fix the smallest production
  path that owns the bug; do not hide it behind report-only updates, broad smoke
  matrices, helper-only code, or common EmitC semantic inference.

## Acceptance Criteria

- [x] PRD and task context identify the bounded module owner and non-goals.
- [x] Relevant specs and previous task artifacts are read before source edits.
- [x] Generated explicit selected-body bundle evidence shows the Gearbox pass
      enabled before provider planning for
      `widening_product_reduce_dequantize_f32`.
- [x] Generated pre-realized selected-body bundle evidence shows realization
      propagates low-precision resource facts before provider planning.
- [x] Generated artifacts expose only explicit provider/resource mirrors, not
      route-id, test-name, helper-name, artifact-name, or common-EmitC semantic
      authority.
- [x] Non-dry-run `ssh rvv` correctness evidence is collected if runtime
      correctness is claimed.
- [x] At least one stale or missing resource-fact case fails closed with a
      targeted diagnostic. Acceptable examples include absent
      `low_precision_resource`, stale schedule candidate, illegal resource
      budget, mismatched memory form, missing propagated pre-realized fact,
      stale provider mirror, or unsupported executable route claim.
- [x] Focused tests/checks pass:
      `build/bin/tianchenrv-rvv-extension-plugin-test`,
      `build/bin/tianchenrv-target-artifact-export-test`, targeted lit for the
      Gearbox pass and target artifact fixtures, plus generated bundle ABI
      evidence.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] A bounded old-authority scan over touched source/test/script lines shows
      no new positive legacy `i32m1`, descriptor, source-front-door, direct-C,
      route-id, artifact-name, bare status/readiness, or common EmitC RVV
      semantic authority.
- [x] Task status, journal/context notes, archive state, and final commit are
      truthful.

## Implementation Notes

- Found a real generated-bundle seam bug in
  `scripts/rvv_generated_bundle_abi_e2e.py`: for pre-realized
  `widening_product_reduce_dequantize_f32`, the script ran
  `--tcrv-materialize-selected-lowering-boundaries` before
  `--tcrv-rvv-materialize-gearbox-schedules`. That order fails closed because
  provider planning sees no pass-produced
  `tcrv_rvv.low_precision_resource.*` facts.
- Repaired the materialization order for this representative so the Gearbox
  pass runs before selected-body realization. Plain `dequantize_i32_to_f32`
  remains unchanged: it still runs Gearbox after selected-boundary
  materialization because that MVP schedule path consumes realized `with_vl`
  structure.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record this
  generated-bundle ordering contract for future runs.

## Evidence

- Explicit dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-gearbox-resource-fact-product-dequant-explicit-dry/evidence.json`
- Pre-realized dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-gearbox-resource-fact-product-dequant-prerealized-dry/evidence.json`
- Explicit `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-gearbox-resource-fact-product-dequant-explicit-ssh/evidence.json`
- Pre-realized `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-gearbox-resource-fact-product-dequant-prerealized-ssh/evidence.json`
- Both non-dry-run runs reported `status=success`, `ssh_evidence=true`,
  `ssh_target=rvv`, remote compile success, remote run success, runtime counts
  `1,7,16,17,257`, scale values `-0.125,0.375`, two data patterns, and
  `source_preserved accumulator_preserved tail_preserved`.

## Checks Run

- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter rvv-gearbox-widening-product-reduce-dequantize-f32`
- [x] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter explicit-selected-body-artifact-widening-product-reduce-dequantize-f32`
- [x] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Explicit and pre-realized generated-bundle dry-runs for
      `widening_product_reduce_dequantize_f32`
- [x] Explicit and pre-realized generated-bundle non-dry-run `ssh rvv`
      correctness runs for `widening_product_reduce_dequantize_f32`
- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-rvv-gearbox-resource-fact-artifact-boundary`
- [x] `git diff --check`
- [x] `git diff --cached --check`
- [x] Bounded added-line old-authority scan over touched script lines

## Out Of Scope

- No product-reduce-dequant-clamp expansion.
- No MAcc, mask, broadcast, conversion, reduction, memory-family, dtype/LMUL
  clone batch, or new low-precision candidate-family expansion.
- No high-level Linalg/Vector/StableHLO frontend work.
- No source-front-door positive route.
- No common EmitC invention of RVV resource semantics.
- No dashboard, index, report-only closeout, or broad pass-pipeline refactor
  outside this Gearbox resource-fact artifact seam.
- No runtime/performance claim without real `ssh rvv` evidence.

## Technical Notes

- Required specs from brief:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Required previous-task context:
  `.trellis/tasks/archive/2026-06/06-06-rvv-gearbox-low-precision-resource-candidate-pass/`.
- Required implementation files from brief:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Required focused fixtures from brief:
  `test/Transforms/RVV/rvv-gearbox-widening-product-reduce-dequantize-f32.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`,
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`.
