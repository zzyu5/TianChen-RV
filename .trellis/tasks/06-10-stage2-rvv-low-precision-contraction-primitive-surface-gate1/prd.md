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
  to require `low-shifted-product-rescale` schedule evidence instead of the
  older pair-sum-only schedule label.
- Collected fresh `ssh rvv` same-target timing after the production schedule
  change:
  `widening_product_reduce_dequantize_f32` remains regression/no-win with
  best speedup range `0.688202..0.705133`, 12 summary records, 60 measurement
  records, and 12 correctness records.
  `widening_product_reduce_dequant_clamp_f32` remains regression/no-win with
  best speedup range `0.677994..0.704931`, 24 summary records, 120 measurement
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

Current Gate 4 evidence-root policy-ingestion slice:

- This round refines the production policy denial boundary after the fresh
  low-shifted-product-rescale measurements remained regression/no-win. The
  bounded improvement is not another generated-bundle closeout and not another
  schedule rewrite; it makes the RVV low-precision performance policy consume
  the complete same-target evidence JSON root before selected-dispatch
  acceptance, instead of trusting only a helper-built record or an isolated
  nested `same_target_measurement_record`.
- The concrete blocker carried forward is still the fresh same-target
  regression/no-win evidence. The policy seam now verifies that the root
  `status`, `ssh` target/provenance, timing method, root-level
  `result_classification`, measurement harness schedule mirrors, packed-i4
  oracle selection, maturity evidence input, provider feedback tie-back, and
  nested source-backed record all agree with the provider-owned packed-i4
  resource/schedule facts before correctness fallback is accepted.
- This earlier evidence-root policy slice did not itself change the runtime
  statement schedule. After the current low-shifted-product-rescale schedule
  repair, the final source-backed dequant and dequant-clamp measurement roots
  are the current policy inputs and continue to deny performance preference.

Acceptance:

- [x] Production policy exposes evidence-root ingestion APIs for parsing the
  complete same-target evidence JSON root into a measurement record and policy
  input.
- [x] Evidence-root ingestion rejects stale root-level result/schedule facts
  before selected dispatch can consume a no-win record.
- [x] Focused C++ coverage feeds both current source-backed dequant and
  dequant-clamp Gate 4 JSON roots through the evidence-root policy path and
  proves correctness fallback plus performance-preference denial.
- [x] Existing helper-built and record-level policy paths remain covered and
  continue to reject stale source-backed, primitive, schedule, artifact, and
  correctness facts.
- [x] Macro task remains active because Gate 4 still needs a later
  provider-owned schedule/resource repair plus fresh same-target timing before
  any performance-preferred dispatch claim.

Completed Gate 4 evidence-root policy-ingestion slice:

- Added `buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceRoot`,
  `buildRVVLowPrecisionSameTargetMeasurementPolicyInputFromEvidenceRoot`, and
  selected-dispatch policy/verify overloads that consume complete evidence
  roots while preserving provider-owned route and maturity authority.
- The root verifier now requires `status = success`, `ssh_evidence = true`,
  `ssh_target = rvv`, `dry_run = false`, the monotonic raw timing method,
  candidate-specific op kind/baseline identity, validated packed-i4 metadata
  selection, root `result_classification` agreement, measurement harness
  schedule agreement, packed oracle agreement, maturity input agreement, and
  provider feedback tie-back agreement.
- Updated plugin coverage so the current dequant JSON root and current
  dequant-clamp JSON root both drive selected-dispatch policy evaluation. Stale
  root-level speedup and harness schedule facts now fail before the no-win
  record can authorize a selected-dispatch decision.
- The consumed policy decision remains `correctness-fallback` with
  `performanceSelectionAllowed = false`,
  `performanceWinClaimAllowed = false`, and
  `same-target-measurement-no-win-or-regression`.

Current Gate 4 low-shifted-product schedule/resource repair slice:

- This round changes production RVV plugin statement planning for the
  provider-owned packed-i4 product-reduction-dequant/dequant-clamp path. The
  concrete bottleneck is the remaining low-nibble unpack schedule: after
  loading packed bytes, the current path sign-extends each low nibble with
  shift-left 4 followed by arithmetic shift-right 4 for both operands before
  the low widening product. That keeps extra i8 low-nibble vectors live and
  spends two i8 arithmetic shifts before the product can be formed.
- The intended repair keeps packed-i4 semantics and the single pair-sum
  `vwredsum` reduction strategy, but changes the provider-owned schedule to
  compute the low product from the shift-left low-nibble operands directly and
  arithmetic-shift the widened i16 low-product vector right by 8 before the
  high-nibble product, pair-sum, and reduction. This preserves
  `(sign_extend_i4(lhs_low) * sign_extend_i4(rhs_low))` while reducing the
  low-nibble unpack/product statement count and the selected peak-live resource
  estimate.
- Because this changes generated RVV statement/runtime behavior, the slice must
  run fresh same-target timing for the affected packed-i4 candidate after
  focused compiler tests pass. A measured no-win/regression keeps correctness
  fallback and records a sharper provider-owned blocker; a measured win may
  become `performance-preferred` only through matching provider-owned maturity,
  target mirror, measurement, and policy facts.

Acceptance:

- [x] Packed-i4 provider/resource facts identify the low-shifted-product
  rescale schedule, selected vector budget, and remaining no-win blocker without
  relying on route ids, artifact names, q8/q4 labels, or Common EmitC
  inference.
- [x] Statement planning emits the low-shifted-product/rescale schedule from
  provider-owned packed-i4 resource facts and target artifact validation rejects
  stale old low-nibble sign-extension or metadata-only schedule mirrors.
- [x] Focused C++/FileCheck/script coverage proves the repaired statement order,
  target artifact metadata, dry-run harness evidence, and stale policy
  rejection boundaries.
- [x] Fresh `ssh rvv` same-target timing is collected for dequant and
  dequant-clamp packed-i4 paths, or a precise hardware/build blocker is
  recorded before policy update.
- [x] Performance policy remains `correctness-fallback` on fresh no-win/
  regression evidence, or selects `performance-preferred` only if measured-win
  evidence and provider-owned maturity facts agree.
- [x] Bounded scans show no new legacy RVV route authority, q8/q4 route naming,
  source-front-door positive route, descriptor-driven compute, or Common EmitC
  semantic inference.

Current Gate 4 resource-cost blocker/admission slice:

- This round keeps the active macro task open and adds a production
  resource-cost/admission boundary for the current packed-i4
  low-shifted-product-rescale path. The compiler path must carry a
  provider-owned cost contract/model, loop-body step count, blocker, and
  admission decision from resource candidate selection through selected-body
  realization, route facts, target mirrors, generated evidence inputs, and
  low-precision performance policy.
- The slice does not change the generated RVV statement schedule by default:
  the existing same-target evidence still shows dequant and dequant-clamp
  packed-i4 no-win/regression. The expected production behavior is therefore a
  sharper correctness-fallback denial that selected dispatch and performance
  policy consume from provider-owned resource facts rather than from reports,
  artifact names, q8/q4 labels, or Common EmitC inference.
- If implementation discovers a narrowly justified schedule/resource repair
  beyond the low-shifted-product-rescale path, that repair must be part of this
  same boundary and must be followed by fresh same-target timing. Otherwise,
  runtime behavior is unchanged and the existing evidence roots are consumed as
  the admission denial evidence.

Acceptance:

- [x] Packed-i4 provider/resource facts include a resource-cost contract/model,
  loop-body step count, resource-cost blocker, and performance-admission
  decision for the low-shifted-product-rescale schedule.
- [x] Selected-body realization, route fact derivation, cross-region handoff,
  and target artifact mirrors require or preserve those facts for packed-i4
  dequant and dequant-clamp.
- [x] Same-target measurement records/evidence inputs and
  `RVVLowPrecisionPerformancePolicy` consume the provider-owned resource-cost
  facts and deny performance-preferred admission for current no-win evidence;
  stale or missing resource-cost facts are rejected.
- [x] Focused C++/FileCheck/script coverage proves the new admission boundary
  and stale resource-cost rejection without relying on report-only or
  generated-bundle-only evidence.
- [x] Because generated runtime schedule is unchanged, no fresh `ssh rvv`
  timing is required; if runtime generation changes, fresh dequant and
  dequant-clamp same-target timing must be collected before policy claims.
- [x] Bounded scans show no new legacy RVV route authority, q8/q4 route naming,
  source-front-door positive route, descriptor-driven compute, or Common EmitC
  semantic inference.

Completed Gate 4 resource-cost blocker/admission slice:

- Added provider-owned packed-i4 resource-cost facts for the current
  low-shifted-product-rescale path:
  `rvv-low-precision-packed-i4-resource-cost-contract.v1`,
  `low-shifted-product-rescale-loop-12-peak-live-6of32-two-region-vsetvl.v1`,
  loop-body step count `12`,
  blocker `packed-i4-low-shifted-product-rescale-loop-12-budget-6of32-no-win`,
  and admission decision
  `deny-performance-preferred-with-resource-cost-no-win-blocker`.
- Required those facts through Gearbox resource candidates, selected-body
  realization, `gearbox_cross_region_handoff` verification, route-family
  selection, statement planning consistency checks, route metadata, target
  artifact mirrors, generated-bundle metadata, same-target measurement records,
  evidence roots, and production-pressure policy inputs.
- Updated the current source-backed dequant and dequant-clamp evidence roots to
  carry the provider resource-cost/admission facts without changing timing
  data or generated runtime schedule. Since both current roots remain
  regression/no-win, the policy still selects correctness fallback and denies
  performance-preferred dispatch.
- Added focused plugin, target, script, and evidence-root checks for accepted
  resource-cost facts and stale/missing resource-cost/admission rejection.
- Self-repaired the target exporter test stack pressure exposed by the larger
  route/provider fact surface: route-control and segment2 provider temporary
  plans now avoid large stack copies, and the monolithic target artifact test
  raises its own soft stack limit before constructing the full RVV fixture
  matrix.
- Gate 4 remains open. The next continuation point is a provider-owned
  packed-i4 schedule/resource bottleneck beyond the current 12-step
  low-shifted-product-rescale path, or a precise production-consumed blocker if
  no measured-win repair is found.

Current Gate 4 resource-cost decision-consumption slice:

- This round turns the packed-i4 resource-cost admission field into an explicit
  dispatch/performance decision input. The current low-shifted-product-rescale
  path still carries the no-win denial
  `deny-performance-preferred-with-resource-cost-no-win-blocker`, but future
  measured-win policy evaluation may select `performance-preferred` only if the
  RVV provider also changes the resource-cost admission decision to
  `admit-performance-preferred-with-resource-cost-measured-win`.
- The slice is policy/provider-local and does not change generated RVV
  statement scheduling or target runtime code. The existing dequant and
  dequant-clamp same-target regression/no-win evidence roots remain the
  current evidence inputs.

Acceptance:

- [x] Direct performance measurement outcome policy evaluation consumes
  provider resource-cost and realization-admission tie-backs, not only
  record/input/evidence-root paths.
- [x] Current no-win/regression policy requires the resource-cost admission
  decision to remain
  `deny-performance-preferred-with-resource-cost-no-win-blocker`.
- [x] A measured-win policy path fails closed if provider maturity and
  measurement facts claim `performance-preferred` while the resource-cost
  admission decision still carries the no-win denial.
- [x] Focused C++ coverage proves the accepted measured-win fixture needs an
  explicit provider-owned admission update and that stale no-win admission
  blocks performance preference.
- [x] No fresh `ssh rvv` timing is required because generated runtime schedule
  is unchanged.

Completed Gate 4 resource-cost decision-consumption slice:

- Added a distinct provider-owned measured-win admission value:
  `admit-performance-preferred-with-resource-cost-measured-win`.
- `verifyPackedI4SelectionFacts` now treats performance admission as a
  resource-cost decision enum, while no-win policy consistency requires the
  existing deny value and measured-win policy requires the new admit value.
- Direct `RVVLowPrecisionPerformanceMeasurementOutcome` evaluation now checks
  resource-cost and realization-admission tie-backs, closing the gap where
  only record/input/evidence-root paths consumed those facts.
- Plugin and target artifact measured-win fixtures now update the provider
  admission decision explicitly before `performance-preferred` can be selected.
  A measured-win record that leaves the no-win denial in place fails strict
  policy verification.
- Gate 4 remains open because this is a policy/provider decision-consumption
  closure, not a new schedule repair or measured runtime win.

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
  dequant `0.688202..0.705133` with 12 summaries / 60 measurements / 12
  correctness records, and dequant-clamp `0.677994..0.704931` with 24
  summaries / 120 measurements / 24 correctness records.
- Bounded scan over touched source/tests/spec/current Gate 4 evidence found no
  old Gate 3 packed-i4 evidence ID, old pair-sum-only schedule decision,
  stale schedule-decision reason, q8/q4 route authority, source-front-door
  positive route, descriptor-driven compute, or Common EmitC RVV semantic
  inference.

Current Gate 4 evidence-root policy-ingestion verification:

- `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test` passed after adding the evidence-root
  policy ingestion API and focused coverage. The target artifact test target
  still emits its existing switch-coverage warnings.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed, including dequant
  and dequant-clamp source-backed JSON root policy ingestion, root-level stale
  speedup rejection, and root-level stale harness schedule rejection.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-
  packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-
  dequant-clamp-f32-packed-i4|rvv-generated-bundle-same-target-measure-gate4-
  dry-run|rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-
  dequantize-f32-packed-i4-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-
  widening-product-reduce-dequant-clamp-f32-packed-i4-dry-run'` passed from
  `build/test` with 5 focused tests.
- Fresh `ssh rvv` timing was rerun after the generated RVV statement schedule
  changed. The final dequant `0.688202..0.705133` and dequant-clamp
  `0.677994..0.704931` regression/no-win records are the consumed evidence
  inputs and still deny performance preference.

Current Gate 4 resource-cost blocker/admission verification:

- `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test` passed after adding provider-owned
  packed-i4 resource-cost/admission facts and the target exporter stack
  self-repair. The target artifact test target still emits its existing
  switch-coverage warnings.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed with the default
  command after the test binary raises its own soft stack limit.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed after adding resource-cost/admission tie-back checks.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed after
  limiting packed-i4 resource-cost metadata to packed-i4 resource profiles.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --dry-run
  --artifact-root /tmp/tcrv-gate4-packed-i4-dry-run --run-id
  codex-gate4-resource-cost --overwrite --op-kind
  widening_product_reduce_dequantize_f32 --op-kind
  widening_product_reduce_dequant_clamp_f32` passed.
- `git diff --check` passed.
- Bounded added-diff scan for `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m`, `__riscv_*_i32m1`, source-front-door, and descriptor
  markers found only the PRD non-goal wording; no new positive legacy route
  authority was added.

Current Gate 4 resource-cost decision-consumption verification:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed, including the
  measured-win admission update and stale no-win resource-cost admission
  rejection.
- `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test` passed. The target artifact test
  target still emits its existing switch-coverage warnings.
- `build/bin/tianchenrv-target-artifact-export-test` initially failed because
  its measured-win fixture still carried the no-win resource-cost admission
  denial; after updating that target-side fixture to the new provider-owned
  measured-win admission value, the test passed.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-
  gate1` passed.
- `git diff --check` passed.
- No fresh `ssh rvv` timing was rerun because this slice changes policy
  admission consumption only, not generated RVV runtime scheduling.

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
  low-shifted-product-rescale schedule/resource repair contract, the
  provider-mirror vs fresh-measurement range distinction, current same-target
  evidence IDs, and the strict no-win/regression dispatch policy ranges.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the complete
  same-target evidence-root policy ingestion contract: root status/ssh/timing,
  result classification, measurement harness schedule, packed oracle,
  maturity-input, and provider-feedback tie-backs must agree with the nested
  source-backed record before selected-dispatch policy evaluation.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the
  resource-cost admission decision-consumption rule: no-win/regression policy
  must keep
  `deny-performance-preferred-with-resource-cost-no-win-blocker`, while a
  measured-win performance-preferred path requires the provider-owned
  `admit-performance-preferred-with-resource-cost-measured-win` decision and
  fails closed if measurement or metadata attempts to promote performance
  without that provider update.

## Continuation Point

Keep this macro task active. Gates 1-3 are complete. The next unfinished
milestone is Gate 4: choose the next provider-owned packed-i4 schedule/resource
bottleneck beyond the current 12-step low-shifted-product-rescale path, or
record a precise production-consumed blocker if no measured-win repair is
available. Do not claim performance-preferred dispatch until a source-backed
same-target measured win agrees with provider maturity, target mirrors,
measurement roots, and policy facts.
