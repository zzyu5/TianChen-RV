# Stage2 RVV Low-Precision Contraction Primitive Surface Campaign

## Goal

Create one active macro owner for RVV low-precision contraction primitive
surface maturity. The campaign moves q8/q4-style pressure into typed
low-level `tcrv_rvv` primitive facts and RVV plugin-owned legality/planning,
not q8/q4 route ids, artifact names, source-front-door markers, or
generated-bundle-only evidence.

Gates 1-3 are complete. The current round implements a Gate 4 slice:
Gearbox/resource-aware selected-body realization and measured same-target
comparison must consume the low-precision primitive source facts,
widening-product multiplicand/extension facts, widening reduction facts,
config/policy/VL facts, source-backed artifact identity, and measurement
provenance before route, target, or policy surfaces can claim maturity.

## What I Already Know

- Commit `3ebe7adc rvv: close gearbox realization campaign` completed and
  archived the Gearbox resource-aware selected-body realization campaign.
- No `.trellis/.current-task` existed at session start, so this task is created
  from the Hermes Direction Brief.
- Specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral Common EmitC -> target
  artifact -> `ssh rvv` evidence only when runtime/correctness/performance is
  claimed.
- `tcrv.exec` binds ABI/runtime roles; it does not own low-precision compute,
  dtype, SEW/LMUL, sign-extension, or intrinsic spelling.
- Common EmitC/export must not infer signedness, byte-load form, extension
  behavior, widening, schedule, mask, contraction semantics, or target
  support. It may only carry provider-built payloads and mirrors.
- Historical q8/q4 and llama.cpp examples are pressure tests that expose
  broader Stage 2 low-precision/quantized contraction gaps. They are not route
  authority.

## Requirements

- Keep this Trellis task active across rounds until all macro campaign gates
  are complete or human steering redirects the campaign.
- Complete one coherent unfinished gate slice per round and update this PRD
  with completed and remaining gates plus a precise continuation point.
- Gate 1 must establish or harden a production surface where typed i8/u8
  vector/config, byte load, and sign-or-zero-extension facts are modeled and
  consumed by RVV plugin-owned admission/planning before later widening
  product or reduction routes depend on them.
- Gate 1 admission must derive from typed `tcrv_rvv` body/config, target
  capability, and runtime ABI facts. It must fail closed when required facts
  are missing, stale, or metadata-derived.
- Gate 2 must structurally represent signed i8 and unsigned u8 widening-product
  multiplicand roles, extension/sign policy, product dtype/SEW/LMUL, typed
  config/policy, runtime ABI operand facts, route-family validation facts, and
  target artifact mirrors.
- Gate 2 facts must be provider-owned and derived from typed body/config,
  runtime ABI, and target capability facts. Artifact metadata, route ids,
  helper names, ABI strings, or test names must not invent support.
- Gate 3 must structurally represent widening reduction/accumulation facts
  derived from typed low-precision product-reduction body/config/runtime ABI
  facts, including source signedness, source/load/extension policy,
  product dtype/config, accumulator/result dtype/config, chain relation,
  widening product intrinsic, widening reduction intrinsic, seed splat,
  accumulator/result layout, and reduction store VL.
- Gate 3 signed i8 and unsigned u8 accepted cases must be provider-owned and
  fail closed when stale, missing, mismatched, or metadata-derived primitive
  facts attempt to claim widening reduction support.
- Gate 4 must wire the Gate 1-3 primitive facts through Gearbox resource
  candidate/admission, selected-body realization and handoff, route metadata,
  target artifact mirrors, target support bundle export, same-target
  measurement records, and production-pressure policy inputs.
- Gate 4 must fail closed when primitive source/load/extension facts,
  widening-product role/policy facts, resource schedule/admission facts,
  target mirrors, artifact identity, or same-target measurement provenance are
  missing, stale, or metadata-only.
- Targeted diagnostics must distinguish missing typed vector/config facts,
  unsupported signedness/extension form, missing byte-load facts, and
  metadata/route-id/artifact-name authority attempts where the code surface can
  observe them.
- Retain Common EmitC/export neutrality.
- Do not add new dtype-prefixed op families, q8/q4-named routes, one-intrinsic
  wrapper dialects, high-level Linalg/Vector/StableHLO frontends, or
  descriptor-driven C/source export.

## Macro Campaign Gates

- [x] Gate 1: typed i8/u8 RVV primitive surface and byte-load
  sign-or-zero-extension admission are modeled and consumed by RVV plugin
  legality/planning with fail-closed diagnostics.
- [x] Gate 2: low-precision widening product facts are structurally
  represented and provider-consumed without route-id or artifact-name
  authority.
- [x] Gate 3: widening reduction/accumulation facts are represented and
  consumed for contraction-style kernels.
- [ ] Gate 4: Gearbox/resource-aware selected-body realization and measured
  same-target comparison consume these primitive facts with source-backed
  evidence. Current slice wires primitive source and widening-product facts
  through Gearbox/resource admission, realization handoff, route/target
  metadata, and same-target measurement records; remaining Gate 4 work is
  measured same-target production-kernel comparison maturity and any further
  resource-schedule improvement needed before performance-preferred dispatch.

## Current Round Slice: Gate 4 Primitive-Fact Provenance

Implementation starts from the completed Gate 1-3 low-precision primitive
surface. This Gate 4 slice wires primitive source/load/extension facts and
widening-product multiplicand/extension policy facts into Gearbox resource
candidates, selected-body realization, route-family validation, target artifact
mirrors, target support bundle export, same-target measurement records, and
production-pressure policy inputs.

Acceptance criteria:

- [x] Gearbox resource candidates and selected-body realization carry
  primitive source load, primitive source extension, widening-product
  multiplicand roles, and widening-product extension policy into realized
  handoff attributes and reject stale/missing values.
- [x] RVV route-family plan validation, route description validation, route
  metadata, target artifact validation, and target support bundle export
  consume those facts as provider-owned mirrors.
- [x] Same-target measurement records, performance-policy inputs, and
  production-pressure profiles require the same primitive source and
  widening-product facts and reject stale source-backed artifact records.
- [x] Existing source-backed packed-i4 dequant-clamp same-target evidence is
  updated to preserve these provider facts in its measurement record without
  turning the measured regression into a performance-preferred dispatch claim.
- [x] Focused tests prove accepted Gearbox realization/target metadata and
  fail-closed stale primitive source-extension, schedule, resource, target
  mirror, and measurement-provenance paths.
- [x] Bounded scan confirms touched code and added diff do not introduce legacy
  RVV route-authority markers as positive support.
- [x] Relevant focused build/test targets pass.
- [x] Task remains active with Gates 1-3 marked complete and Gate 4 remaining,
  unless repository evidence proves all gates are already complete.

Completed Gate 1 slice:

- Added provider-owned low-precision primitive source-load and
  source-extension facts:
  `lowPrecisionPrimitiveSourceLoadKind` and
  `lowPrecisionPrimitiveSourceExtensionKind`.
- Signed i8 primitive facts derive `unit-stride-byte-load` and
  `sign-extend-i8-to-i16-product`; unsigned u8 widening-product facts derive
  `unit-stride-byte-load` and `zero-extend-u8-to-u16-product`.
- RVV contraction route-family plan validation, route description mirror
  verification, widening-product route facts, widening-reduction primitive
  facts, target validation contracts, and target artifact metadata mirror
  checks now consume these fields.
- Target artifact support bundles export the new mirror keys:
  `tcrv_rvv.low_precision_primitive.source_load` and
  `tcrv_rvv.low_precision_primitive.source_extension`.

Completed Gate 2 slice:

- Added provider-owned standalone widening-product fact fields:
  `wideningProductMultiplicandRoleSummary` and
  `wideningProductExtensionPolicy`.
- Signed i8 widening-product facts now explicitly mirror:
  `lhs=lhs-input-buffer:wprod-lhs:src-i8mf4`,
  `rhs=rhs-input-buffer:wprod-rhs:src-i8mf4`, and
  `source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2`.
- Unsigned u8 widening-product facts now explicitly mirror:
  `lhs=lhs-input-buffer:wprod-lhs:src-u8mf4`,
  `rhs=rhs-input-buffer:wprod-rhs:src-u8mf4`, and
  `source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2`.
- RVV contraction route-family plan validation, route description validation,
  target provider-facts validation, target support bundle metadata, and target
  artifact mirror validation all consume the new fields.
- Signed and unsigned widening-product lit fixtures now prove accepted mirrors
  and stale metadata rejection for multiplicand roles and extension policy.

Completed Gate 3 slice:

- Extended bounded typed RVV product-reduction verification so unsigned u8
  chains can carry `unsigned_widening_product` into
  `unsigned_widening_reduce_add` with `u8mf4 -> u16mf2 -> u32m1` typed facts.
- Added unsigned product-reduction runtime ABI acceptance:
  `const uint8_t *` lhs/rhs, `const uint32_t *` accumulator, `uint32_t *`
  output, and `size_t` runtime count.
- Made RVV contraction route-family planning derive signed or unsigned
  widening-reduction primitive facts from the selected typed body/config and
  route description rather than route ids, fixture names, artifact metadata, or
  Common EmitC inference.
- Required route-family plan validation, route description mirror validation,
  route construction, statement planning, and target artifact validation to
  consume product/reduction chain relation, source signedness, product/result
  vector C types, accumulator/result dtype, widening product intrinsic,
  widening reduction intrinsic, scalar seed splat, C type mapping, target leaf,
  and low-precision primitive facts.
- Added a selected-body unsigned u8 product-reduction artifact fixture with
  accepted mirror checks and stale source signedness, source extension,
  accumulator dtype, reduction intrinsic, and C type mapping rejection coverage.

Completed Gate 4 slice in this round:

- Added Gearbox resource candidate fields for primitive source load,
  primitive source extension, widening-product multiplicand roles, and
  widening-product extension policy.
- Required selected-body realization and `gearbox_cross_region_handoff` to
  materialize and verify those facts before route planning can consume the
  handoff.
- Propagated the same facts through RVV route-family plan validation, route
  description validation, route metadata, target artifact mirror validation,
  and target support bundle header export.
- Extended same-target measurement record parsing, policy input construction,
  production-pressure profile construction, script-generated evidence records,
  and stale-evidence checks to require source-backed primitive provenance.
- Updated the packed-i4 dequant-clamp source-backed evidence JSON so the
  measurement record preserves provider primitive source and widening-product
  facts while still denying performance-preferred dispatch for the measured
  no-win/regression outcome.

Current Gate 4 measured-comparison slice:

- Audited the source-backed packed-i4 dequant/dequant-clamp same-target
  comparison path against the production compiler route.
- No production compiler source change was made in this slice because the
  current dequant-clamp generated C++ already materializes the selected
  packed-i4 resource schedule as low/high nibble sign-extension, two widening
  products, an i16 pair-sum, one `vwredsum`, and the f32 dequant/clamp epilogue
  before target artifact export. The Gate 4 blocker is not missing codegen for
  this representative; it is preventing the measured regression/no-win evidence
  from being confused with route unsupported, missing evidence, or a
  performance-preferred dispatch claim.
- Hardened the dequant-clamp packed-i4 target fixture so the generated C++
  path must expose the low/high nibble unpack, two-product pair-sum,
  single-reduction schedule, and f32 clamp/select/store sequence. This ties the
  source-backed same-target no-win policy to the actual emitted statement
  structure, not just packed-i4 metadata or artifact names.
- That audit's parsed dequant-clamp evidence reported regression/no-win and
  was later superseded by the current schedule/resource repair slice's fresh
  same-target timing below. The policy rule remains the same: no-win/regression
  evidence may preserve route support and correctness execution, but must deny
  performance preference and performance-win claims.
- The correct current dispatch/policy outcome remains:
  route support and correctness execution are preserved, while performance
  preference and performance-win claims are denied with
  `same-target-measurement-no-win-or-regression`.

Current Gate 4 measured-comparison acceptance:

- [x] Audit shows no safe resource-schedule improvement is available without a
  new production schedule/resource repair and new same-target timing.
- [x] Dequant-clamp packed-i4 generated C++ must prove the selected resource
  schedule, not only metadata: load packed bytes, sign-extend low/high signed
  i4 nibbles, compute two widening products, pair-sum them, reduce once, then
  apply dequant/clamp/store.
- [x] Parsed dequant-clamp same-target evidence remains source-backed and
  candidate-sensitive, with measurement counts and speedup range bound to the
  dequant-clamp packed-i4 candidate.
- [x] Performance policy keeps the current no-win/regression as an explicit
  performance-preference denial, not route unsupported and not missing
  evidence.
- [x] Macro task remains active because Gate 4 still needs a future
  provider-owned schedule/resource repair plus fresh same-target timing before
  any performance-preferred dispatch can be claimed.

Remaining Gate 4 continuation:

- Use the now-source-backed primitive/resource/measurement provenance to
  improve or audit the low-precision production-kernel same-target comparison
  path. The next owner should focus on whether the measured packed-i4
  dequant/dequant-clamp resource schedule can be improved with a real
  provider-owned schedule/resource repair and fresh same-target timing. Until
  then, the source-backed regression evidence remains the explicit
  performance-preference denial. Do not switch to new route ids, q8/q4 named
  wrappers, or generated-bundle-only evidence.

Current Gate 4 dispatch-preference denial slice:

- This round adds a production-consumed selected-dispatch policy boundary for
  the existing source-backed packed-i4 no-win/regression result. The goal is
  not to claim a new schedule win, but to make the compiler reject any selected
  dispatch case/mirror that tries to carry `performance-preferred` semantics
  while the measured policy decision selects the conservative
  correctness-fallback path.
- Acceptance:
  - [x] Selected-dispatch policy validation fails closed when the RVV dispatch
    case policy or case mirror claims performance preference for the current
    no-win/regression packed-i4 measurement.
  - [x] Route support and correctness execution remain allowed for the
    accepted source-backed record, while performance selection and performance
    win claims remain denied.
  - [x] Focused C++ coverage proves accepted no-win dispatch policy, stale
    dispatch-case policy rejection, and stale dispatch-case mirror rejection.
  - [x] No new same-target measurement is required because this slice changes
    policy consumption only, not generated schedule/resource behavior.
  - [x] The macro task remains active after the slice unless Gate 4 is fully
    closed with a provider-owned schedule/resource repair and fresh
    same-target timing.

Completed Gate 4 dispatch-preference denial slice:

- Added fail-closed selected-dispatch no-win policy validation so a route-
  supported packed-i4 dispatch case cannot carry `performance-preferred`
  policy or mirror text unless same-target measured-win evidence and provider
  maturity facts have promoted the decision.
- Added focused plugin tests for the accepted no-win correctness-fallback path,
  stale selected-dispatch case policy rejection, and stale selected-dispatch
  case mirror rejection.
- No new `ssh rvv` measurement was rerun in this slice because generated
  schedule/resource behavior did not change; existing source-backed
  no-win/regression evidence still denies performance preference.

Current Gate 4 schedule/resource repair slice:

- This round changes production RVV plugin statement planning for the
  provider-owned packed-i4 product-reduction-dequant/dequant-clamp path. The
  concrete bottleneck is the old statement order: it unpacked all low/high i4
  nibble vectors for lhs/rhs before building either widening product, extending
  vector live ranges before the pair-sum/single-reduction schedule.
- The intended repair keeps the same typed primitive semantics and pair-sum
  single-`vwredsum` reduction strategy, but makes the statement schedule
  consume low-nibble facts earlier: load packed lhs/rhs, unpack low lhs/rhs,
  build the low widening product, then unpack high lhs/rhs, build the high
  widening product, pair-sum, and reduce.
- Because this changes generated schedule/resource behavior, the slice must run
  fresh same-target timing for the affected packed-i4 candidate after focused
  compiler tests pass. A measured no-win/regression keeps correctness fallback
  and updates the denial with fresh source-backed evidence; a measured win may
  become `performance-preferred` only through matching provider-owned maturity,
  remediation, target mirror, and policy facts.

Acceptance:

- [x] Packed-i4 statement planning emits the repaired low-product-before-high-
  product order from provider-owned low-precision resource facts, without
  Common EmitC inventing RVV semantics.
- [x] Focused C++ and FileCheck coverage proves the repaired statement order,
  target artifact metadata, and existing stale policy rejection boundaries.
- [x] Fresh `ssh rvv` same-target timing is collected for the changed
  packed-i4 path, or a precise blocker is recorded before policy update.
- [x] Performance policy remains `correctness-fallback` on fresh no-win/
  regression evidence, or selects `performance-preferred` only if measured-win
  evidence and provider-owned maturity facts agree.
- [x] Bounded scans show no new legacy RVV route authority, q8/q4 route naming,
  source-front-door positive route, descriptor-driven compute, or Common EmitC
  semantic inference.

Completed Gate 4 schedule/resource repair slice:

- Repaired the packed-i4 statement plan so the selected-body/provider path
  emits low-nibble sign-extension and the low widening product before high-
  nibble sign-extension/product construction. This reduces the old all-unpack-
  before-product live-range shape while preserving the provider-owned typed
  primitive facts, pair-sum, and single-`vwredsum` semantics.
- Updated provider resource schedule/remediation facts, target artifact mirror
  validation, generated-bundle script checks, and focused C++/FileCheck tests
  to require `low-product-before-high-unpack` schedule evidence instead of the
  older pair-sum-only schedule label.
- Collected fresh `ssh rvv` same-target timing after the production schedule
  change:
  `widening_product_reduce_dequantize_f32` remains regression/no-win with
  best speedup range `0.688202..0.705410`, 12 summary records, 60 measurement
  records, and 12 correctness records.
  `widening_product_reduce_dequant_clamp_f32` remains regression/no-win with
  best speedup range `0.683721..0.705212`, 24 summary records, 120 measurement
  records, and 24 correctness records.
- Because both fresh measurements remain below 1.0, the consumed performance
  policy keeps `dispatchPolicyPath = correctness-fallback`,
  `performanceSelectionAllowed = false`, and denial reason
  `same-target-measurement-no-win-or-regression`. No performance-preferred
  marker is allowed from this slice.
- Gate 4 remains open because this repair improved the production schedule
  shape but did not produce a measured win. The next slice should choose a
  different provider-owned packed-i4 schedule/resource bottleneck or record a
  precise blocker before any further performance policy change.

## Non-Goals

- No generated-bundle-only or `ssh rvv`-only closeout unless it validates
  production code changed in this slice.
- No q8/q4-named route ids, artifact names, wrappers, or llama.cpp-specific
  authority.
- No high-level frontend, per-Linalg route authority, dtype/LMUL clone batch,
  one-intrinsic wrapper dialect, broad MAcc evidence matrix, or source-front
  door positive route.
- No Common EmitC invention of dtype, widening, schedule, mask, extension, or
  contraction semantics.
- No campaign archive after only Gate 1.

## Technical Notes

- Read `.trellis/spec/index.md`.
- Read `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Read `.trellis/spec/lowering-runtime/emitc-route.md`.
- Read `.trellis/spec/variant-pipeline/index.md`.
- Read `.trellis/spec/testing/mlir-testing-contract.md`.
- Read archived Gearbox campaign PRD:
  `.trellis/tasks/archive/2026-06/06-10-stage2-rvv-gearbox-realization-campaign/prd.md`.
- Relevant source areas to inspect:
  `include/TianChenRV/Dialect/RVV/`,
  `lib/Dialect/RVV/`,
  `include/TianChenRV/Plugin/RVV/`,
  `lib/Plugin/RVV/`,
  `lib/Plugin/RVV/EmitC/`,
  `lib/Target/RVV/`,
  and focused tests under `test/Plugin/` and `test/Target/RVV/`.

## Verification Plan

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Target/RVV/explicit-selected-body-artifact-widening-product.mlir
  Target/RVV/explicit-selected-body-artifact-widening-product-unsigned-u8.mlir
  Target/RVV/explicit-selected-body-artifact-widening-product-reduce-add.mlir
  Target/RVV/explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8.mlir`
- Additional focused dialect/target tests if the implementation changes
  dialect syntax, verifier behavior, or target artifact validation.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-gate1`
- `git diff --check`
- `git diff --cached --check`
- Bounded scan over touched files and added diff lines for legacy RVV
  route-authority markers.

## Verification Results

- `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8|
  explicit-selected-body-artifact-widening-product-reduce-add|
  explicit-selected-body-artifact-widening-product-unsigned-u8|
  explicit-selected-body-artifact-widening-product\\.mlir'` passed from
  `build/test`.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-gate1`
  passed.
- `git diff --check` passed.
- `git diff --cached --check` passed.
- Bounded changed-file and staged-diff scan for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, source-front-door, q8/q4 route naming,
- `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test` passed for this Gate 4 slice.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- Focused manual FileCheck positive checks passed for
  `test/Transforms/RVV/rvv-gearbox-widening-product-reduce-dequantize-f32.mlir`
  and `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
  covering Gearbox schedule, selected-body realization, emission-plan metadata,
  and target header artifact export.
- Focused manual FileCheck negative checks passed for stale primitive source
  extension in Gearbox schedule and stale primitive source extension in
  `gearbox_cross_region_handoff`.
- `git diff --check` passed.
- Bounded added-diff scan for `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m`, q8/q4/llama route naming, source-front-door,
  descriptor-driven, and direct C exporter authority found no new positive
  legacy authority. Added `__riscv_*_i32m1` matches are the expected
  provider-owned widening-reduction primitive intrinsic mirrors for typed
  i8/i16/i32 product-reduction facts, not old i32m1 route authority.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test` passed
  for the Gate 4 selected-dispatch denial slice.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed, including stale
  selected-dispatch case policy and selected-dispatch case mirror rejection.
- `cmake --build build --target tianchenrv-target-artifact-export-test
  tcrv-opt tcrv-translate` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-
  f32-packed-i4` passed from `build/test`.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-
  gate1` passed.
- `git diff --check` passed.

Current Gate 4 schedule/resource repair verification:

- `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test` passed after the packed-i4
  statement-order repair and realization-admission schedule fact sync.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed after self-repairing
  stale realization-admission schedule-reason evidence consumption.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-
  packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-
  dequant-clamp-f32-packed-i4|rvv-generated-bundle-same-target-measure-gate4-
  dry-run|rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-
  dequantize-f32-packed-i4-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-
  widening-product-reduce-dequant-clamp-f32-packed-i4-dry-run'` passed from
  `build/test`.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- Fresh `ssh rvv` same-target timing after the production schedule repair
  passed correctness and produced regression/no-win evidence:
  dequant `0.688202..0.705410` with 12 summaries / 60 measurements / 12
  correctness records, and dequant-clamp `0.683721..0.705212` with 24
  summaries / 120 measurements / 24 correctness records.
- Bounded scan over touched source/tests/spec/current Gate 4 evidence found no
  old Gate 3 packed-i4 evidence ID, old pair-sum-only schedule decision,
  stale schedule-decision reason, q8/q4 route authority, source-front-door
  positive route, descriptor-driven compute, or Common EmitC RVV semantic
  inference.

## Spec Update Decision

- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record the
  signed i8 and unsigned u8 low-precision widening-reduction primitive-fact
  contract, including unsigned source/product/accumulator/result facts,
  intrinsics, scalar seed splat, target mirror validation, and Common EmitC
  neutrality.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the
  low-precision no-win dispatch preference boundary: route-supported
  packed-i4 no-win/regression cases must select correctness fallback and reject
  selected-dispatch case or mirror text that claims `performance-preferred`
  without measured-win evidence.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the packed-i4
  low-product-before-high-unpack schedule/resource repair contract, the
  provider-mirror vs fresh-measurement range distinction, current same-target
  evidence IDs, and the strict no-win/regression dispatch policy ranges.

## Continuation Point

Keep this macro task active. Gates 1-3 are complete. The next unfinished
milestone is Gate 4: Gearbox/resource-aware selected-body realization plus
measured same-target comparison must consume the provider-owned low-precision
primitive facts with source-backed evidence, without q8/q4 wrappers,
artifact-name authority, or Common EmitC semantic inference.
