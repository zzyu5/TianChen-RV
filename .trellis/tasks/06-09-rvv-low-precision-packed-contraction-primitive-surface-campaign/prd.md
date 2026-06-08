# RVV low-precision packed-contraction primitive surface campaign

## Goal

Build a production RVV plugin boundary for packed low-bit contraction operands.
The campaign starts from the existing typed low-precision `tcrv_rvv` body and
RVV-owned resource/primitive facts, then makes packed operand form,
signedness, packing width, and unpack/dequant intent explicit before selected
body realization, route planning, target validation, or artifact claims can
accept the path.

This is a macro task. This first round implements Gate 1 and the first bounded
Gate 2 production slice: classify the current gap from repository evidence and
land the smallest coherent typed-fact/fail-closed boundary for packed low-bit
operands. The task remains active after this slice unless all campaign gates are
genuinely complete.

## Direction Brief Source

Hermes direction title:

```text
RVV low-precision packed-contraction primitive surface campaign
```

The brief asks for a new macro owner after commit `dc0ab535` closed the grouped
`u2` low-precision performance campaign. The new owner is not another
generated-bundle or `ssh rvv` evidence closeout. It is the production compiler
primitive surface for packed low-bit contraction operands and the fail-closed
boundary for unsupported q4/q8-like pressure-test shapes.

## What I Already Know

- The worktree was clean before task creation. Recent history ends with
  `dc0ab535 chore(task): close rvv low precision campaign`.
- The archived grouped `u2` campaign closed all of its gates with active
  production consumers, `ssh rvv` correctness evidence, and same-target timing
  that remained a regression/no-win signal.
- Current specs require the RVV authority chain:
  selected typed `tcrv_rvv` body -> RVV plugin-owned legality, selected-body
  realization, primitive/resource facts, and route provider ->
  `TCRVEmitCLowerableRoute` -> neutral Common EmitC -> target artifact.
- Current repository code already has positive byte low-precision surfaces:
  signed `i8mf4 -> i16mf2 -> i32m1` product-reduction primitive facts,
  unsigned `u8mf4 -> u16mf2` widening-product coverage, product-dequant and
  product-dequant-clamp realization, and grouped `u2` resource candidates.
- The gap for this campaign is therefore not "add q8" or "add unsigned u8".
  The first gap is that packed sub-byte operand form is not a structural
  production fact. Existing low-precision resource facts carry source/product/
  accumulator/result dtype and resource shape, but not explicit operand form,
  signedness, packed element width, packing layout, or unpack intent.
- q4/q2-style packed operands must not be accepted because a benchmark name,
  route id, artifact name, ABI string, wrapper, descriptor, or Common EmitC path
  suggests them. Until typed packed support exists, such claims must fail closed
  at an RVV-owned primitive/resource/provider boundary.

## Campaign Gates

- [x] Gate 1: attribute the next low-precision production-kernel gap to a
      named typed primitive/resource owner using current code and archived
      grouped `u2` evidence, not benchmark names.
- [x] Gate 2: land production-source primitive-surface or fail-closed
      validation for one coherent packed low-precision contraction operand
      family.
- [ ] Gate 3: make selected-body realization and RVV route/statement planning
      consume the primitive/resource facts without Common EmitC semantic
      inference.
- [ ] Gate 4: add focused artifact/runtime evidence only for production support
      changed by this campaign.
- [ ] Gate 5: when executable support exists, run same-target correctness and
      timing against a named scalar baseline and report win/regression honestly.

## Current Round Slice

Complete the first Gate 2 production slice by adding provider-owned packed
operand primitive/resource facts and a fail-closed guard:

- existing byte `i8` product-reduction dequant/dequant-clamp resource
  candidates must explicitly carry `unpacked byte elements`, signed source
  operands, 8-bit storage/effective element width, and no-unpack widening
  product intent;
- stale or unsupported packed/sub-byte claims such as `packed_i4`, `q4`, or
  `q2` in the same RVV resource fact surface must be rejected before selected
  body realization or route construction can treat them as executable;
- Common EmitC and target artifact code may only carry provider mirrors after
  the provider-owned boundary accepts them; they must not infer packing.

This slice may leave positive packed q4/q2 support unimplemented. That is
intentional for the first milestone if the production boundary and targeted
fail-closed diagnostics are landed.

## Requirements

- Keep implementation in C++/MLIR/TableGen/CMake/lit/FileCheck. Python remains
  only tooling.
- Do not add q4/q8/llama.cpp route authority, source-front-door positive
  routes, descriptor-driven compute, wrappers, or benchmark-name acceptance.
- Do not make Common EmitC choose dtype, packing, unpack/dequant, schedule,
  intrinsic spelling, or compute semantics.
- Preserve current signed `i8` and unsigned `u8` low-precision behavior unless
  a new typed fact proves it must fail closed.
- Add focused coverage for accepted byte facts and stale/unsupported packed
  facts.
- Do not claim runtime correctness or performance unless this slice changes
  executable generated artifacts and collects the required `ssh rvv` evidence.

## Acceptance Criteria

- [x] Production RVV primitive/resource/provider code carries explicit
      low-precision operand form, source signedness, packed storage/effective
      element width, packing layout, and unpack intent facts for the active
      byte product-reduction dequant/dequant-clamp path.
- [x] The RVV-owned resource/realization/provider boundary fails closed when
      those facts are missing, stale, or claim unsupported packed sub-byte
      operands.
- [x] Focused tests cover the positive byte fact surface and at least one
      unsupported packed/stale fact diagnostic before route construction or
      realization support.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass after the change.
- [x] `tcrv-opt` / `tcrv-translate` are built or verified available if touched
      lit/export tests need them.
- [x] Bounded authority scan over touched files and added diff lines shows no
      new legacy i32/source-front-door/descriptor/Common-EmitC semantic
      authority drift.
- [ ] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean after the commit.

## Out Of Scope

- No generated-bundle-only or `ssh rvv` evidence-only round.
- No positive packed q4/q2 executable route unless the full typed
  primitive/resource/provider/target path is implemented and tested.
- No high-level Linalg/Vector/StableHLO frontend work.
- No per-Linalg route authority, benchmark-name route authority, or one wrapper
  per intrinsic.
- No performance-win or llama.cpp parity claim without same-target raw evidence.

## Technical Notes

- Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and archived PRD
  `.trellis/tasks/archive/2026-06/06-08-06-08-rvv-low-precision-production-kernel-performance-optimization-campaign/prd.md`.
- Production owners under inspection:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Current implementation evidence already includes
  `RVVLowPrecisionWideningReductionPrimitiveFacts`,
  `RVVLowPrecisionContractionResourceCandidate`,
  `RVVLowPrecisionContractionResourceSelection`, selected-body realization
  consumers, route metadata mirrors, and target validation mirrors.

## Continuation Point

After this slice, the next milestone should add positive typed packed sub-byte
primitive support for one narrow operand family, then extend the same
provider-owned packed fact boundary through statement planning, generated-bundle
evidence, and same-target evidence only after route support is executable.

## Current Round Result

This slice completes Gate 1 and the first Gate 2 fail-closed production
boundary. The RVV low-precision resource candidate/selection surface now carries
provider-owned:

- `operand_form`;
- `source_signedness`;
- `storage_element_width`;
- `effective_element_width`;
- `packing_layout`;
- `unpack_intent`.

The active byte product-reduction dequant/dequant-clamp path is explicitly
classified as signed, unpacked byte operands with 8-bit storage/effective
element width and no-unpack direct widening-product intent. Stale or unsupported
packed claims such as `packed-i4-nibbles` fail closed in the Gearbox/resource
boundary and in target artifact mirror validation. Common EmitC remains a
metadata/payload carrier only.

Focused validation completed:

- built `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`;
- ran both required C++ test binaries successfully;
- ran focused lit for the Gearbox pass and pre-realized target artifact path
  from `build/test` with both tests passing;
- ran `git diff --check`;
- ran a bounded added-line authority scan over touched production/test files
  with no new legacy i32/source-front-door/descriptor/Common-EmitC authority
  matches.

The macro task remains active. This round does not claim positive q4/q2 packed
execution, generated-bundle evidence, runtime correctness, performance, or
llama.cpp parity.
