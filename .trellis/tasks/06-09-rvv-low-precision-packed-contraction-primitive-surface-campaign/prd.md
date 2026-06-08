# RVV low-precision packed-contraction primitive surface campaign

## Goal

Build a production RVV plugin boundary for packed low-bit contraction operands.
The campaign starts from the existing typed low-precision `tcrv_rvv` body and
RVV-owned resource/primitive facts, then makes packed operand form,
signedness, packing width, and unpack/dequant intent explicit before selected
body realization, route planning, target validation, or artifact claims can
accept the path.

This is a macro task. Gate 1, the first Gate 2 production boundary, the byte
representative Gate 3 consumer boundary, and the first Gate 3 packed-i4
structural resource/provider slice are complete. The current round is a bounded
Gate 3 nibble unpack/sign-extension statement slice: consume the selected signed
packed-i4/nibble resource facts in RVV direct-contraction statement planning,
materialize provider-owned unpack/sign-extension statement steps before the
widening product, and keep target artifact/runtime claims fail-closed until the
Gate 4 artifact/export path and evidence are implemented. The task remains
active after this slice unless all campaign gates are genuinely complete.

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
- [x] Gate 3: make selected-body realization and RVV route/statement planning
      consume the primitive/resource facts without Common EmitC semantic
      inference.
- [ ] Gate 4: add focused artifact/runtime evidence only for production support
      changed by this campaign.
- [ ] Gate 5: when executable support exists, run same-target correctness and
      timing against a named scalar baseline and report win/regression honestly.

## Current Round Slice

Complete one bounded Gate 3 production slice for the accepted signed packed-i4
resource family:

- keep the previously selected packed-i4 resource facts as the only authority:
  byte storage width, four-bit effective width, low/high nibble packing layout,
  signed source interpretation, sign-extension unpack intent, dtype/runtime
  facts, resource shape, provider mirrors, and target/export fail-closed
  boundary contract;
- make the direct-contraction statement-plan owner build a provider-owned
  packed-i4 unpack/sign-extension sub-plan before the widening product. The
  sub-plan must unpack low and high nibbles from both operands using RVV-owned
  statement leaves, feed the resulting signed i8 vectors into the widening
  product/reduction sequence, and preserve the default byte path when no
  explicit packed-i4 candidate is selected;
- continue to reject stale, missing, mismatched, or unsupported packed facts at
  the RVV-owned route/statement boundary. Do not infer packing or unpack
  semantics from q4, q8, llama.cpp, route ids, artifact names, benchmark names,
  descriptors, or Common EmitC;
- add a targeted target-artifact/export fail-closed boundary for packed-i4 until
  Gate 4 implements provider-owned artifact validation and evidence. A positive
  statement plan is not a generated-bundle, runtime correctness, timing, or
  parity claim;
- update focused C++ and MLIR/FileCheck coverage for the positive statement
  sub-plan and the remaining target/export fail-closed boundary.

This slice does not close Gate 4 or Gate 5. Generated-bundle/runtime evidence,
same-target correctness/timing, and parity claims remain open until the
packed-i4 path has real executable support.

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
- [x] Gate 3 current packed-i4 slice: provider-owned resource candidates and
      route-family validation recognize one signed packed-i4/nibble operand
      family with storage/effective width, packing layout, unpack intent,
      dtype, runtime AVL/VL, resource shape, provider mirror, and target/export
      fail-closed boundary facts.
- [x] Gate 3 current packed-i4 slice: selected-body realization and provider
      planning consume explicit packed-i4 facts when present, preserve default
      byte selection when absent, and reject stale/missing/mismatched packed
      facts before Common EmitC or target artifact export.
- [x] Gate 3 current packed-i4 slice: statement planning fails closed at the
      missing packed-i4 unpack/statement boundary without claiming executable
      generated artifacts, runtime correctness, timing, or parity.
- [x] Gate 3 current nibble-unpack slice: direct-contraction statement planning
      consumes the selected packed-i4 resource facts and constructs RVV-owned
      low/high nibble sign-extension statements before widening product and
      reduction, while the default unpacked-byte path remains stable.
- [x] Gate 3 current nibble-unpack slice: stale or mismatched packed facts still
      fail closed at the RVV-owned route/statement boundary, and target artifact
      export fails closed with a targeted packed-i4 Gate 4 boundary instead of
      claiming executable artifacts or runtime evidence.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass after the change.
- [x] `tcrv-opt` / `tcrv-translate` are built or verified available if touched
      lit/export tests need them.
- [x] Bounded authority scan over touched files and added diff lines shows no
      new legacy i32/source-front-door/descriptor/Common-EmitC semantic
      authority drift.
- [x] `git diff --check`, `git diff --cached --check`, and final
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

After this slice, the next milestone should extend the provider-owned packed-i4
statement boundary through target artifact validation and generated-bundle
evidence, then collect same-target evidence only after route support is
executable.

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

## Current Gate 3 Packed-I4 Round Result

This slice adds the first positive packed sub-byte primitive/resource family at
the structural resource/provider boundary. The RVV resource candidate set now
includes a signed packed-i4-in-i8 product-reduction dequant/dequant-clamp
candidate with 8-bit storage width, 4-bit effective width, low/high nibble
layout, sign-extension unpack intent, bounded live-vector/resource shape, and
its own realization decision. Explicit selected packed-i4 facts are consumed by
selected-body realization, route-family validation, provider planning, route
mirrors, and the statement-plan fail-closed target/export boundary.

The default byte path remains selected when no explicit packed-i4 candidate is
present. Mismatched packed facts fail closed before Common EmitC or target
artifact export. A true packed-i4 statement plan remains deliberately
unsupported: statement planning now fails closed with a diagnostic naming the
missing RVV-owned nibble unpack/sign-extension boundary before widening
product. This slice does not claim executable generated artifacts, `ssh rvv`
correctness, timing, or parity.

Focused validation completed:

- built `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`;
- ran `build/bin/tianchenrv-rvv-extension-plugin-test`;
- ran `build/bin/tianchenrv-target-artifact-export-test`;
- manually executed the focused lit RUN pipelines for
  `rvv-gearbox-widening-product-reduce-dequantize-f32` and
  `pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32`
  because this environment has no `llvm-lit`/Python `lit` module;
- updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the packed-i4
  resource/statement-boundary contract.

The macro task remains active. Gate 4 generated-bundle/runtime evidence and
Gate 5 same-target correctness/timing remain open. This was the previous
continuation point; the current nibble-unpack slice advances it below.

## Current Gate 3 Packed-I4 Nibble-Unpack Round Result

This slice completes the bounded provider-owned packed-i4 statement boundary for
the accepted signed packed-i4 product-reduction-dequant representative. The
direct-contraction statement-plan owner now consumes the selected packed-i4
resource facts, loads packed i8 operand vectors, emits RVV-owned low/high nibble
sign-extension steps for both operands, runs the low-nibble and high-nibble
widening product/reduction chains, and carries the second reduction result into
`dot_acc_vec`. The default unpacked-byte path remains unchanged when no
explicit packed-i4 candidate is selected.

The path now reaches provider-built `TCRVEmitCLowerableRoute` eligibility, but
target artifact export remains deliberately fail-closed at the packed-i4 Gate 4
boundary until artifact validation and runtime evidence exist. This slice does
not claim generated-bundle support, `ssh rvv` correctness, timing, or parity.

Focused validation completed:

- built `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`;
- ran `build/bin/tianchenrv-rvv-extension-plugin-test`;
- ran `build/bin/tianchenrv-target-artifact-export-test`;
- manually executed the focused FileCheck pipelines for
  `rvv-gearbox-widening-product-reduce-dequantize-f32` and
  `pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32`
  because this environment has no `llvm-lit`/Python `lit` module.

The macro task remains active. The exact next continuation point is Gate 4:
validate the provider-owned packed-i4 statement payload through target artifact
export and generated-bundle evidence before any runtime/correctness/performance
claim.
