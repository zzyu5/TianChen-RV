# RVV low-precision packed-contraction primitive surface campaign

## Goal

Build a production RVV plugin boundary for packed low-bit contraction operands.
The campaign starts from the existing typed low-precision `tcrv_rvv` body and
RVV-owned resource/primitive facts, then makes packed operand form,
signedness, packing width, and unpack/dequant intent explicit before selected
body realization, route planning, target validation, or artifact claims can
accept the path.

This is a macro task. Gate 1, Gate 2, Gate 3, and Gate 4 are complete for the
accepted signed packed-i4/nibble representative through selected-body
realization, provider planning, provider-owned route statement payloads, target
artifact/export validation, and generated-bundle dry-run evidence. The current
round is a bounded Gate 5 executable same-target comparison slice: carry the
accepted packed-i4 path from typed `tcrv_rvv`/resource facts through generated
artifact execution on `ssh rvv`, correct the external harness/reference oracle
only where it consumes provider-owned packed-i4 facts, and record same-target
correctness plus scalar/reference comparison evidence. If executable support
cannot truthfully proceed, the slice must land the smallest production
compiler/runtime/export fix that removes the blocker or leave a precise active
continuation point. The task remains active until Gate 5 is genuinely complete.

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
- [x] Gate 4: add focused artifact/runtime evidence only for production support
      changed by this campaign.
- [ ] Gate 5: when executable support exists, run same-target correctness and
      timing against a named scalar baseline and report win/regression honestly.

## Current Round Slice

Complete one bounded Gate 5 executable same-target comparison slice for the
accepted signed packed-i4 resource family:

- keep the selected packed-i4 resource facts as the only authority: byte
  storage width, four-bit effective width, low/high nibble packing layout,
  signed source interpretation, sign-extension unpack intent, dtype/runtime
  facts, resource shape, provider mirrors, and provider-built statement
  payloads;
- use the public pre-realized selected-body generated-bundle route and its
  rebuilt RVV provider route, not fixture names, metadata, route ids, artifact
  names, or script-side packed semantics, to detect whether the generated
  artifact consumes the packed-i4 representative;
- make the generated-bundle harness/reference oracle distinguish default
  unpacked-byte product-dequant from explicit signed packed-i4 product-dequant
  by consuming the provider-owned low-precision resource metadata already
  validated in Gate 4;
- for packed-i4 runtime evidence, generate inputs whose bytes carry two signed
  i4 lanes, compare the generated artifact against a scalar/reference
  same-target calculation that sign-extends low and high nibbles, accumulates
  both lanes per packed byte, applies the runtime scale, checks source and
  accumulator preservation, and preserves non-scalar output sentinels;
- run non-dry-run `ssh rvv` correctness for the accepted packed-i4
  representative when the harness and generated artifact are executable;
- record a same-target scalar/reference comparison boundary. If timing is not
  yet implemented or not production-ready, leave Gate 5 open with the exact
  next continuation point instead of claiming performance or parity;
- if executable support cannot truthfully proceed, identify and fix or
  fail-close the exact production blocker without claiming runtime correctness,
  timing, parity, or performance.

This slice may complete Gate 5 correctness and scalar/reference comparison for
the accepted packed-i4 representative. It does not claim performance win,
llama.cpp parity, or q4/q8 route authority unless same-target timing evidence
and a named baseline are present.

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
- [x] Gate 4 current packed-i4 artifact/export slice: target route-family
      validation consumes provider-owned packed-i4 resource facts and accepts
      artifact export only when the rebuilt route carries the exact low/high
      nibble sign-extension statement payload, low/high product/reduction
      chain, and final carry assignment.
- [x] Gate 4 current packed-i4 artifact/export slice: focused target/export
      tests cover positive packed-i4 artifact acceptance plus stale
      payload/mirror fail-closed diagnostics before any generated-bundle or
      runtime claim.
- [x] Gate 4 evidence boundary: generated-bundle and `ssh rvv` correctness or
      timing remain unclaimed unless executable support is actually reached.
- [x] Gate 4 current generated-bundle slice: the public selected-body
      realization and generated-bundle dry-run path accepts an explicit signed
      packed-i4 product-reduction-dequant selected resource and carries the
      provider-owned packed resource mirrors and statement payload into the
      generated artifact evidence.
- [x] Gate 4 current generated-bundle slice: generated-bundle evidence checks
      packed lhs/rhs loads, low/high nibble sign-extension, low/high widening
      product/reduction, final carry into `dot_acc_vec`, ABI/header/export
      facts, and stale resource/payload rejection without inferring packed-i4
      semantics from names, route ids, descriptors, or Common EmitC.
- [x] Gate 4 current generated-bundle slice: if executable packed-i4 support is
      still not present, the evidence remains dry-run/artifact-only and Gate 5
      remains open with no `ssh rvv` correctness, timing, parity, or
      performance claim.
- [x] Gate 5 current packed-i4 executable slice: generated-bundle harness
      selection consumes provider-owned packed-i4 low-precision resource
      metadata and switches only the external reference/oracle behavior needed
      to compare packed low/high signed i4 nibbles against the generated
      artifact.
- [x] Gate 5 current packed-i4 executable slice: non-dry-run generated-bundle
      evidence runs on `ssh rvv`, passes correctness for the accepted packed-i4
      representative, and records a same-target scalar/reference comparison
      boundary without treating fixture names, route ids, artifact names,
      provider mirrors, or dry-run JSON as runtime authority.
- [x] Gate 5 current packed-i4 executable slice: if same-target timing is not
      implemented in this round, the task remains open with an exact
      continuation point and no performance win, parity, or llama.cpp claim.
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

This slice should either close the generated-bundle artifact evidence part of
Gate 4 for the artifact-supported packed-i4 path or name the exact production
blocker that prevents it. Same-target correctness or timing evidence must be
collected only after executable packed-i4 generated artifact support exists.

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

## Current Gate 4 Packed-I4 Artifact/Export Round Result

This slice completes the target artifact/export seam for the accepted signed
packed-i4 product-reduction-dequant representative. The previous target-layer
hard fail-close for selected packed-i4 resources was replaced with target
route-family validation that consumes provider-owned packed-i4 resource facts
and accepts artifact export only when the rebuilt route carries the exact
provider-owned statement payload:

- packed i8 source-vector loads for lhs/rhs;
- low-nibble shift-left and arithmetic shift-right sign-extension for both
  operands;
- high-nibble arithmetic shift-right sign-extension for both operands;
- low-nibble widening product/reduction into `reduced_i32_vec`;
- high-nibble widening product/reduction into `reduced_i32_vec_i4_high`;
- final loop carry assignment from the high-nibble reduction result into
  `dot_acc_vec`.

The target/export test now builds the packed-i4 fixture from the typed
pre-realized selected body, runs selected-body realization in the fixture, then
rebuilds the provider route and validates artifact acceptance from the realized
body. Stale packed resource facts, stale unpack-intent metadata mirrors, and
stale low/high nibble statement operands/results fail closed in the target
artifact bridge.

This slice does not claim generated-bundle support, `ssh rvv` correctness,
timing, or parity. The macro task remains active. The exact next continuation
point is still Gate 4, narrowed to focused generated-bundle evidence for the
artifact-supported packed-i4 path; Gate 5 same-target correctness/performance
remains blocked until executable generated artifacts exist.

## Current Gate 4 Packed-I4 Generated-Bundle Round Result

This slice completes focused generated-bundle artifact evidence for the
accepted signed packed-i4 product-reduction-dequant representative without
claiming executable/runtime correctness. The public pre-realized selected-body
path now accepts the explicit packed-i4 resource decision through selected-body
realization, provider route rebuild, target artifact export, and generated
bundle dry-run evidence.

The generated-bundle evidence script now treats low-precision resource facts as
actual provider-owned metadata. The default byte/grouped path still validates
the `u2` resource profile, while an explicit signed packed-i4 selected
candidate switches the evidence profile to:

- `operand_form = packed-i4-nibbles`;
- signed i4 nibbles stored in i8 with storage width 8 and effective width 4;
- low/high nibble layout and sign-extension unpack intent;
- two-region Gearbox handoff with `load-product-reduce` ->
  `dequant-store`;
- packed lhs/rhs source-vector loads;
- low/high nibble sign-extension;
- low/high widening product and reduction;
- final high-nibble reduction carry into `dot_acc_vec`.

Focused generated-bundle validation completed with
`scripts/rvv_generated_bundle_abi_e2e.py --dry-run
--pre-realized-selected-body --op-kind
widening_product_reduce_dequantize_f32 --input
test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`,
and the evidence JSON records `ssh_evidence = false`. The default byte/grouped
product-dequant dry-run also still passes. This slice intentionally does not
claim `ssh rvv` correctness, timing, parity, or performance.

The macro task remains active. Gate 4 artifact/generated-bundle evidence is
now complete for this representative, but Gate 5 same-target
correctness/performance remains blocked until executable packed-i4 generated
artifact support and a truthful scalar-baseline comparison are available.

## Current Gate 5 Packed-I4 Executable Correctness Round Result

This slice completes the executable correctness and same-target scalar
reference comparison path for the accepted signed packed-i4
product-reduction-dequant representative. The generated-bundle evidence script
now detects the packed-i4 representative only after the object/header bundle
metadata has been validated against provider-owned low-precision resource facts.
For that representative, the external C ABI harness switches from the default
unpacked-byte `i8*i8` oracle to a packed-i4 scalar oracle that:

- packs two signed i4 lanes into each int8_t input byte;
- sign-extends low and high nibbles independently;
- accumulates low-nibble and high-nibble products per packed byte into the
  i32 scalar seed;
- applies the runtime f32 scale after accumulation;
- checks source and accumulator preservation plus non-scalar output sentinels.

The default byte/grouped product-dequant dry-run remains unchanged and does not
record a packed-i4 oracle.

Focused validation completed:

- built `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and
  `tcrv-translate`;
- ran `build/bin/tianchenrv-rvv-extension-plugin-test`;
- ran `build/bin/tianchenrv-target-artifact-export-test`;
- ran `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
- ran `scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
- ran the default byte/grouped product-dequant generated-bundle dry-run;
- ran the explicit packed-i4 product-dequant generated-bundle dry-run;
- ran the explicit packed-i4 product-dequant non-dry-run `ssh rvv` evidence
  with `ssh_evidence = true`, remote `clang` compile success, remote run
  success, and counts `1,7,16,17,257` across two patterns and two runtime
  scale values.

The macro task remains active. Gate 5 now has executable packed-i4 correctness
and a same-target scalar/reference comparison path for the accepted
representative, but it still does not claim timing, performance win, llama.cpp
parity, or q4/q8 route authority. The exact next continuation point is Gate 5
same-target timing: add a named scalar/baseline timing measurement on `ssh rvv`
for the accepted packed-i4 representative and report win/regression honestly.
