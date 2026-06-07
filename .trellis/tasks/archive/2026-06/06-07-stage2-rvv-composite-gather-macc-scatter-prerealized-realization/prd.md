# Stage2 RVV composite gather-MAcc-scatter pre-realized realization owner

## Goal

Implement the RVV plugin-local selected-body realization owner that consumes the
bounded pre-realized runtime-scalar computed-mask indexed gather, masked MAcc,
and indexed scatter family bodies and rewrites them into the explicit
composite `tcrv_rvv` body shape that is already provider route-supported.
The owner must preserve typed body/config/runtime facts and fail closed before
provider route construction when the three family bodies cannot be fused
without changing mask, index, accumulator, output, dtype/config, ABI, or AVL/VL
semantics.

## What I already know

- Repository started clean on `main` at
  `33d43d69 rvv: add composite gather macc scatter route contract`.
- No `.trellis/.current-task` existed; this task was created from the Hermes
  Direction Brief.
- The archived predecessor task
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-gather-macc-scatter-realization-route-contract/`
  completed the explicit selected composite route-supported contract and left
  pre-realized multi-family bodies fail-closed at the named composite
  realization-owner boundary.
- `.trellis/spec/extension-plugins/rvv-plugin.md` now defines the
  runtime-scalar computed-mask indexed gather-MAcc-scatter route contract:
  runtime ABI order is
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`; the body contains
  one selected `with_vl` with load/splat/load/load/old-dst-load/index-load,
  compare, masked indexed gather, masked MAcc, and masked indexed scatter.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires Common EmitC to
  consume only provider-built `TCRVEmitCLowerableRoute` payloads; it must not
  infer RVV composite semantics from route ids, metadata, helper names, ABI
  names, or artifact strings.
- Current code has the explicit route recognizer and recorder in
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`; the named fail-closed
  pre-realized boundary is in
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.

## Requirements

- Add a bounded RVV plugin-local composite selected-body realization path for
  exactly one runtime-scalar computed-mask indexed gather pre-realized body,
  one runtime-scalar computed-mask MAcc pre-realized body, and one
  runtime-scalar computed-mask indexed scatter pre-realized body under the same
  selected RVV variant.
- Realize the three family bodies into one explicit selected `tcrv_rvv`
  structure:
  `setvl`, `with_vl`, compare-lhs load, runtime scalar splat, payload load,
  accumulator load, old destination load, index load, compare, masked indexed
  gather, masked MAcc, and masked indexed scatter.
- Validate cross-family compatibility before erasing the pre-realized bodies:
  shared compare lhs, runtime scalar, predicate, mask role/source/memory form,
  runtime `n`/AVL, SEW, LMUL, policy, index value, index EEW, offset unit,
  output destination, gather inactive-lane policy, scatter inactive-lane
  policy, MAcc accumulator/result layout, and runtime ABI roles.
- Preserve the provider-owned explicit composite route contract after
  realization; pre-realized and explicit bodies must reach the same operation
  kind, memory form, typed compute chain, ABI order, operand-binding facts,
  leaf role facts, inactive-lane policy, and provider support mirror.
- Keep incompatible pre-realized family shapes fail-closed with targeted RVV
  diagnostics before Common EmitC or target artifact export.
- Do not add Common EmitC semantic branches, descriptor/source-front-door
  authority, route-id shortcuts, q8/llama-specific logic, or broad route-family
  rewrites.

## Acceptance Criteria

- [x] Positive explicit composite route-contract test remains passing.
- [x] Positive pre-realized composite test realizes the three family bodies
  into the explicit composite body shape and then reaches the same
  provider-owned route-supported contract as the explicit fixture.
- [x] Focused negative pre-realized coverage proves at least one stale or
  incompatible family binding fails closed before route construction, with a
  diagnostic naming the composite realization owner boundary and the stale fact.
- [x] Existing single-family gather, computed-mask MAcc, scatter, and
  realization owner tests continue to pass.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes, or the final
  report explains why no target artifact/export code was touched.
- [x] Bounded old-authority scan over touched files and added diff lines shows
  no new descriptor/direct-C/source-front-door/legacy `tcrv_rvv.i32_*`,
  `RVVI32M1`, or `rvv-i32m1` route authority.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

- PRD, context JSONL, task metadata, and journal truthfully describe the owner
  implemented and whether artifact/export remains the next continuation point.
- Source and test changes are focused on RVV plugin-local selected-body
  realization and composite route-contract evidence.
- The Trellis task is completed and archived only after focused checks pass.
- One coherent commit is created when the task is complete; if unfinished, the
  task remains open with the exact continuation point.

## Technical Approach

Add a composite realization owner path beside the selected-body realization
registry rather than forcing the existing one-body owner interface to represent
three sibling family bodies. The composite detector will collect direct child
pre-realized RVV bodies in the selected variant, require exactly the supported
gather/MAcc/scatter runtime-scalar computed-mask trio, validate cross-family
facts, create the realized explicit body before the first family op, erase the
three pre-realized family ops, and return the realized `with_vl`. Route
planning then reuses the already-supported explicit composite provider
contract.

## Decision (ADR-lite)

**Context**: The predecessor route contract deliberately left three
pre-realized family bodies fail-closed because the existing selected-body
realization registry owns one pre-realized op at a time.

**Decision**: Implement a bounded composite realization path in RVV
plugin-local selected-body realization, with its own cross-family validator and
materializer, then hand the realized body to the existing explicit route
provider.

**Consequences**: This avoids Common EmitC semantics and avoids route-id
shortcuts, but it intentionally handles only the bounded runtime-scalar
computed-mask indexed gather-MAcc-scatter trio. Target artifact/generated-bundle
evidence remains a separate continuation unless this round touches export code.

## Out of Scope

- Target artifact/generated-bundle/`ssh rvv` correctness or performance claims.
- Broad composite route framework or unrelated gather/scatter/MAcc family
  expansion.
- High-level Linalg/Vector/StableHLO frontend work.
- Gearbox/autotuning/pass framework work.
- Common EmitC invention of RVV composite semantics.
- Descriptor, source-front-door, helper-name, route-id, metadata, or artifact
  authority.
- New dtype-prefixed op families or legacy `i32m1` route authority.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/variant-pipeline/index.md`
- Previous task read:
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-gather-macc-scatter-realization-route-contract/task.json`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-gather-macc-scatter-realization-route-contract/prd.md`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-gather-macc-scatter-realization-route-contract/implement.jsonl`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-gather-macc-scatter-realization-route-contract/check.jsonl`
- Code/test entry points inspected:
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`

## Completion Notes

- Added `RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner`, a bounded
  RVV plugin-local owner for the runtime-scalar computed-mask indexed
  gather/MAcc/scatter pre-realized trio.
- The owner validates shared compare lhs, rhs scalar, runtime n/AVL, index,
  output destination, mask facts, inactive-lane policies, SEW/LMUL/policy,
  accumulator/result layout, and required ABI roles before materializing one
  explicit `setvl`/`with_vl` body.
- The realized body reuses the explicit provider contract:
  `tcrv_rvv.masked_indexed_load+tcrv_rvv.masked_macc+
  tcrv_rvv.masked_indexed_store`, runtime ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, and the existing
  provider-owned route facts.
- Family-only placeholder ABI values for pre-realized MAcc lhs and scatter
  source are required to be local to their family bodies and are erased when
  the bodies are fused, so they do not pollute the provider ABI order.
- C++ tests now prove explicit and pre-realized composite bodies reach the
  same route-supported contract, and a stale scatter index binding fails closed
  in the composite realization owner before route construction.
- No target artifact/generated-bundle or `ssh rvv` correctness claim is made;
  the exact next continuation point is composite artifact/export mirror and
  generated-bundle/header evidence over this newly realized route.
