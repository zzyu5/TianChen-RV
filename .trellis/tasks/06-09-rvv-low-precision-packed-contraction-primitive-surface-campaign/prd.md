# RVV low-precision packed-contraction primitive surface campaign

## Goal

Build a production RVV plugin boundary for packed low-bit contraction operands.
The campaign starts from the existing typed low-precision `tcrv_rvv` body and
RVV-owned resource/primitive facts, then makes packed operand form,
signedness, packing width, and unpack/dequant intent explicit before selected
body realization, route planning, target validation, or artifact claims can
accept the path.

This is a macro task. Gate 1 and the first Gate 2 production boundary are
complete. The current round is a bounded Gate 3 slice: make the selected-body
realization, route-provider, and statement-planning consumers rely on
provider-owned low-precision primitive/resource facts for the accepted byte
representative, and fail closed when stale packed/sub-byte facts reach a
production consumer. The task remains active after this slice unless all
campaign gates are genuinely complete.

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

Complete one bounded Gate 3 production slice by making the current accepted
low-precision product-reduction dequant/dequant-clamp path consume the
provider-owned packed/operand resource facts at the realization and
route/statement-planning boundary:

- selected-body realization must continue to compare pass-produced resource
  facts against provider-owned widening-reduction primitive facts before it
  materializes the realized producer/consumer `with_vl` regions;
- direct-contraction route/provider and statement-planning consumers must
  require the selected resource facts for operand form, source signedness,
  storage/effective element width, packing layout, unpack intent,
  source/product/accumulator/result dtype, runtime AVL/VL, runtime ABI order,
  target capability mirrors, and resource shape before constructing statement
  plans or `TCRVEmitCLowerableRoute`;
- stale packed/sub-byte claims such as `packed-i4-nibbles`, `q4`, or `q2` must
  fail closed at the RVV consumer boundary before Common EmitC or target
  artifact code can treat them as executable;
- Common EmitC and target artifact code may only carry provider mirrors after
  the RVV provider-owned boundary accepts them; they must not infer packing.

This slice may still leave positive packed q4/q2 support unimplemented. If the
repository evidence does not yet contain a full typed packed sub-byte primitive
family, this round should land the exact production fail-closed consumer
boundary and leave positive packed support as a continuation point.

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

- [x] Gate 2: production RVV primitive/resource/provider code carries explicit
      low-precision operand form, source signedness, packed storage/effective
      element width, packing layout, and unpack intent facts for the active
      byte product-reduction dequant/dequant-clamp path.
- [x] Gate 2: the RVV-owned resource/realization/provider boundary fails closed when
      those facts are missing, stale, or claim unsupported packed sub-byte
      operands.
- [x] Gate 2: focused tests cover the positive byte fact surface and at least one
      unsupported packed/stale fact diagnostic before route construction or
      realization support.
- [x] Gate 3: selected-body realization, direct-contraction provider planning,
      and statement-plan owner selection consume the provider-owned
      low-precision primitive/resource facts for the accepted byte
      product-reduction representative before statement construction or route
      construction.
- [x] Gate 3: focused tests prove stale packed/sub-byte operand facts at the
      route/statement consumer boundary fail closed with diagnostics naming the
      low-precision direct-contraction resource selection instead of relying on
      route ids, artifact names, descriptors, or Common EmitC inference.
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

## Current Gate 3 Round Result

This slice completes the bounded Gate 3 route/statement consumer boundary for
the current accepted byte product-reduction representative. The
direct-contraction statement-plan owner now requires product-reduction
dequant/dequant-clamp routes to consume the same provider-owned
low-precision resource selection as the route-family/provider plan before
constructing statement plans. It cross-checks the selection mirror and
fail-closes stale packed/sub-byte operand facts such as `packed-i4-nibbles`
before Common EmitC or target artifact export can treat them as executable.

Focused validation completed:

- built `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`;
- ran `build/bin/tianchenrv-rvv-extension-plugin-test`;
- ran `build/bin/tianchenrv-target-artifact-export-test`;
- ran focused lit from `build/test` for
  `rvv-gearbox-widening-product-reduce-dequantize-f32` and
  `pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32`;
- ran `git diff --check`;
- ran a bounded added-line authority scan over touched production/test files
  with no new legacy i32/source-front-door/descriptor/Common-EmitC authority
  matches.
- updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the
  product-reduction statement consumer contract and the non-goal of applying
  byte operand-form checks to sibling low-precision resource representatives.

The macro task remains active. Positive packed q4/q2 executable support,
generated-bundle evidence, same-target correctness/timing, and parity claims
remain open.
