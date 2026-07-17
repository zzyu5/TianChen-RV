# Stage2 RVV Gearbox pass MVP for typed conversion route control

## Goal

Make a real RVV plugin-local Gearbox pass boundary visible in the compiler for
one existing typed conversion route. The bounded MVP starts from an already
selected/realized `dequantize_i32_to_f32` typed `tcrv_rvv` body, derives the
only currently legal static gear from structural body/config facts, writes
explicit gear/schedule facts onto IR, requires the RVV provider to validate and
consume those facts before route construction, and mirrors the provider-derived
Gearbox facts into target metadata only after provider acceptance.

The Gearbox v3 artifact is reference intake, not an exact implementation
contract. This task intentionally implements one coherent pass-plus-provider
consumption slice instead of runtime benchmarking, candidate databases, assembly
feedback, or cross-kernel autotuning.

## What I Already Know

- The worktree starts on `main` after `238f6e6e rvv: validate typed conversion
  dtype policy route`.
- No `.trellis/.current-task` existed when this round began, but
  `.trellis/tasks/06-05-rvv-gearbox-autotuning-pass-reference/` already existed
  as a planning/reference task. This task has been started and narrowed instead
  of creating an unrelated task.
- The reference document
  `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md` frames Gearbox as an
  MLIR pass pipeline that turns selected/pre-tuned typed `tcrv_rvv` into
  concrete gear/schedule decisions before the RVV route provider.
- `.trellis/spec/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md` require code-affecting
  hints/config/profile to be consumed by the RVV plugin into real
  `tcrv_rvv` structure or provider-validated route-control facts before
  route construction.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires Common EmitC and
  target artifact export to treat provider facts and metadata as mirrors only.
- The previous archived task made `dequantize_i32_to_f32` conversion facts
  visible to target validation: source/destination dtype/SEW/LMUL, scale ABI,
  conversion kind/relation, typed compute op, runtime ABI, and stale mirror
  rejection.
- Existing tests include
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`
  and C++ provider/target tests around dequant conversion facts.

## Requirements

- Add a production MLIR pass with an explicit pass name, initially scoped to
  already realized selected RVV `dequantize_i32_to_f32` bodies.
- The pass must derive a bounded gear/schedule choice only from structural
  body/config facts: dequant op kind/relation, source/result vector type,
  selected `setvl` SEW/LMUL/policy, runtime AVL/VL use, and target/plugin-owned
  route facts. It must not use route ids, artifact names, q-names, ABI strings,
  intrinsic spellings, filenames, or benchmark names as authority.
- The MVP legal gear is a static provider-owned decision for the current
  dequant proof body, for example `e32_m1_u1`, with explicit schedule facts
  such as static selector, source/result SEW/LMUL, unroll, VL policy, and
  runtime AVL source.
- The pass must materialize the decision as explicit IR attrs on the selected
  `tcrv_rvv` body/ops so later provider code can validate freshness. This is
  an IR/route-control fact, not artifact metadata authority.
- The RVV provider must require and consume the Gearbox facts for
  `dequantize_i32_to_f32` route construction. Missing, stale, unsupported, or
  string-derived Gearbox facts must fail closed before the lowerable route is
  accepted.
- Target artifact metadata may include Gearbox facts only as mirrors of the
  provider-derived/validated route description. Target validation must reject
  stale mirrors when metadata is touched.
- Keep Common EmitC neutral. Do not add common/core RVV schedule selection or
  a target-local Gearbox table.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` truthfully describe this
      bounded Gearbox pass/provider/target task and its spec basis.
- [x] A real production pass is registered in `tcrv-opt`, e.g.
      `--tcrv-rvv-materialize-gearbox-schedules`, and focused lit coverage
      proves an existing dequant selected body gains explicit Gearbox facts.
- [x] The pass fails closed for unsupported bodies/configs and for stale
      preexisting Gearbox attrs instead of silently accepting metadata.
- [x] RVV provider route planning for `dequantize_i32_to_f32` requires the
      pass-produced Gearbox facts and copies them into provider-owned route
      description/validation surfaces before route construction.
- [x] Positive route/provider evidence proves Gearbox facts are consumed before
      route creation and appear in emission-plan or target metadata only as
      provider-derived mirrors.
- [x] Negative tests prove missing/stale/unsupported Gearbox facts fail closed,
      including route-string/artifact-name/intrinsic-spelling authority attempts.
- [x] Focused target artifact or emission-plan checks reject stale Gearbox
      metadata mirrors if target metadata is updated.
- [x] Checks include relevant `tcrv-opt` lit/FileCheck tests, focused
      `tianchenrv-rvv-extension-plugin-test` and/or
      `tianchenrv-target-artifact-export-test` when provider/target C++ changes,
      `git diff --check`, `git diff --cached --check`, and a bounded scan over
      touched files for old-authority/q-name drift.
- [x] No runtime/correctness/performance claim is made without `ssh rvv`
      evidence. `ssh rvv` is not expected for this pass/provider metadata slice.

## Completion Evidence

- Registered and implemented production pass
  `--tcrv-rvv-materialize-gearbox-schedules` in the RVV plugin-local pass
  pipeline. The pass matches the existing selected
  `dequantize_i32_to_f32` typed body and derives a single bounded static
  schedule from typed body/config/runtime facts.
- Added shared provider-owned Gearbox schedule constants and extended RVV
  dialect attr verification so only the intended `with_vl`/`dequantize`
  schedule attrs are accepted.
- Extended dequant provider route planning, route description validation,
  conversion dtype-policy contract, construction metadata, and target artifact
  validation so missing or stale Gearbox facts fail closed before route export.
- Updated target artifact header/metadata checks so Gearbox values are emitted
  only as provider-derived mirrors.
- Focused checks run:
  - `cmake --build build --target tcrv-opt tcrv-translate`
  - `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test`
  - `cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-gearbox-dequantize-i32-to-f32|explicit-selected-body-artifact-dequantize-i32-to-f32'`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - `./build/bin/tianchenrv-rvv-extension-plugin-test`
  - `git diff --check`
  - bounded old-authority/q-name scan over touched files
- Self-repair performed during implementation:
  - moved Gearbox pass matching from loose attrs into typed body/config/runtime
    structural checks;
  - required provider route construction to reject missing pass-produced facts;
  - added stale target-metadata mirror rejection for artifact-name-derived
    Gearbox facts.
- `ssh rvv` was not run because this round makes no executable correctness or
  performance claim.

## Definition of Done

- Production C++/MLIR behavior exists; report-only, prompt-only, helper-only,
  or metadata-only work is not sufficient.
- Focused tests pass or missing local tools are reported explicitly.
- Task status and journal record the implemented owner, checks, old-authority
  scan, ssh applicability, and final worktree status.
- One coherent commit is created if the task is complete.

## Technical Approach

1. Register a new RVV Gearbox pass in the existing MLIR pass infrastructure.
2. Implement one bounded pass over selected RVV variants/`tcrv_rvv.with_vl`
   bodies. Match exactly one realized dequant route shape:
   `setvl -> with_vl -> load -> dequantize -> store`, SEW32 LMUL m1,
   agnostic policy, runtime AVL from `n`, source `i32m1`, result `f32m1`,
   dequant scale ABI role `dequant-scale-value`.
3. Materialize attrs such as `tcrv_rvv.gearbox.schedule_id`,
   `tcrv_rvv.gearbox.selector`, `tcrv_rvv.gearbox.source`,
   `tcrv_rvv.gearbox.unroll`, `tcrv_rvv.gearbox.vl_policy`,
   `tcrv_rvv.gearbox.source_sew/source_lmul`,
   `tcrv_rvv.gearbox.dest_sew/dest_lmul`, and
   `tcrv_rvv.gearbox.runtime_avl_source`.
4. Extend RVV provider dequantization route facts, route-family plan,
   description, conversion dtype-policy validation contract, and metadata mirror
   emission to validate/consume those attrs.
5. Add positive and negative lit/FileCheck tests for the pass and provider
   fail-closed path, plus focused target metadata mirror checks if metadata is
   updated.
6. Run focused checks and repair failures.

## Decision (ADR-lite)

Context: The v3 reference describes a broad Gearbox/autotuning pipeline, but the
current repository already has a mature bounded dequant conversion route and
expects Stage2 selected-body realization/provider ownership.

Decision: Implement the first Gearbox slice as a static RVV plugin-local pass
for one existing realized `dequantize_i32_to_f32` route, then make provider
route construction require the pass-produced gear facts.

Consequences: This creates a real pass/provider boundary now without promising
runtime autotuning. Later tasks can add candidate enumeration, pruning,
assembly feedback, and runtime profiling behind the same Gearbox fact surface.

## Out of Scope

- Runtime benchmarking, persistent tuning cache, assembly feedback, or
  cross-kernel autotuning.
- Full Gearbox candidate enumeration or profile-guided selection.
- High-level frontend/Linalg/Vector/StableHLO work.
- Common/core RVV schedule selection.
- Direct C/intrinsic macro autotune as route authority.
- Broad route-surface expansion, dtype/LMUL clone batches, or one-intrinsic
  wrapper dialect growth.
- q8/q4/llama benchmark authority.
- Any executable/correctness/performance claim without `ssh rvv` evidence.

## Technical Notes

- Reference intake:
  `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md`.
- Previous task context:
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-typed-conversion-sew-policy-route-foundation/`.
- Relevant specs:
  `.trellis/spec/index.md`;
  `.trellis/spec/extension-plugins/rvv-plugin.md`;
  `.trellis/spec/lowering-runtime/emitc-route.md`;
  `.trellis/spec/variant-pipeline/index.md`;
  `.trellis/spec/testing/index.md`.
- Main implementation candidates:
  `include/TianChenRV/Transforms/Passes.td`;
  `include/TianChenRV/Transforms/Passes.h`;
  `lib/Transforms/`;
  `tools/tcrv-opt/tcrv-opt.cpp`;
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`;
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`;
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`;
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`;
  `test/Transforms/RVV/`;
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`.
