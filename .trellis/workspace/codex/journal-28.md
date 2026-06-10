> Continuation from `journal-27.md` (archived at ~1982 lines)

## Session 574: RVV low-precision Gate 2 Gearbox signedness handoff

**Date**: 2026-06-10
**Task**: RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task for Gate 2. The completed slice makes the Gate
1 low-precision primitive source signedness fact part of the realized Gearbox
cross-region handoff, then requires the same fact in dialect verification,
Gearbox body validation, route planning, and contraction route-family provider
checks before Common EmitC or target artifact mirrors can accept the route.

The macro task remains open. This round completes the signedness handoff
sub-slice, not the whole Gate 2 campaign. Additional resource/remediation
facts still need to drive selected-body realization decisions before moving to
generated artifact and `ssh rvv` evidence.

### Main Changes

- Added required `primitive_source_signedness` to
  `tcrv_rvv.gearbox_cross_region_handoff`.
- Materialized the handoff signedness from the selected low-precision resource
  candidate after validating it against primitive facts.
- Rejected missing or stale handoff signedness in RVV dialect verification,
  Gearbox schedule validation, selected-body route planning, and
  contraction-family route owner checks.
- Updated focused product/dequant, dequant-clamp, and packed-i4 selected-body
  tests so realized handoff structure carries the signedness fact.
- Added a stale signedness negative test for the pre-realized product/dequant
  path and a C++ plugin smoke assertion for the handoff fact.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  passed after self-repair.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Manual `tcrv-opt`/`tcrv-translate`/`rg` checks passed for pre-realized,
  explicit, dequant-clamp, and packed-i4 selected-body paths. The project image
  did not provide `FileCheck`, so focused RUN-line intent was verified with the
  built tools.
- Stale handoff signedness and missing handoff signedness both failed closed
  before route/export acceptance.
- No generated-bundle, `ssh rvv`, correctness, timing, or parity claim was
  made.

### Self-Repair

- The first rebuild failed because the plugin C++ test referenced
  `kRVVLowPrecisionResourceSourceSignednessSigned` without the
  `tianchenrv::plugin::rvv::` namespace qualifier. Qualified the constant and
  reran the build successfully.
- An initial shell batch check failed because debugging markers were written
  into temporary MLIR, and a later `rg -q` plus `pipefail` wrapper tripped
  SIGPIPE on large compiler output. Re-ran the same checks with clean temporary
  files and explicit fixed-string assertions.

### Spec Update Decision

[NO CHANGE] `.trellis/spec/extension-plugins/rvv-plugin.md` already required
Gearbox cross-region handoff to carry primitive-chain source signedness for
product-reduction selected-body realization. This round implements that
existing contract in ODS, verifier, realization, provider, and focused tests.

### Status

[OPEN] Macro task remains active. Gate 1 is complete. Gate 2 signedness handoff
sub-slice is complete. Continue Gate 2 with broader resource/remediation
consumption: selected candidate/resource budget, unpack or packed operand
policy, remediation intent, and realized region shape checks. Gates 3 and 4
remain later.

## Session 575: RVV low-precision Gate 2 resource/remediation handoff

**Date**: 2026-06-10
**Task**: RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the same active macro task at Gate 2. This slice extends the previous
signedness-only Gearbox handoff into a provider-owned resource/remediation
handoff: the realized `tcrv_rvv.gearbox_cross_region_handoff` now carries the
selected resource candidate, candidate set, peak-live vector groups, vector
register budget, operand form, packing layout, unpack intent, product/dequant
region indexes, and packed-i4 remediation plan facts when applicable.

The macro task remains open. Gate 1 is complete and this Gate 2
resource/remediation handoff slice is complete for the product/dequant and
packed-i4 representatives. Gate 3 and Gate 4 remain later.

### Main Changes

- Added required resource/remediation attributes to
  `tcrv_rvv.gearbox_cross_region_handoff`.
- Materialized those attributes from the selected low-precision resource
  candidate during RVV selected-body realization.
- Made dialect verification, Gearbox schedule validation, selected-body route
  planning, and contraction route-family owner validation reject stale,
  missing, contradictory, or metadata-only handoff resource/remediation facts.
- Kept Common EmitC and target artifact export as mirror/mechanics consumers
  after provider-owned handoff validation succeeds.
- Updated product/dequant and packed-i4 focused MLIR fixtures plus C++ plugin
  assertions to cover realized handoff resource/remediation facts and
  stale/missing negative cases.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Manual `tcrv-opt`/`tcrv-translate`/`rg` positive checks passed for explicit
  product/dequant, pre-realized product/dequant, packed-i4, and dequant-clamp
  paths. The project image did not provide `FileCheck` or `llvm-lit`.
- Manual negative checks passed for stale candidate, missing candidate, stale
  peak-live, stale unpack policy, stale region index, stale packed-i4
  remediation, and missing packed-i4 remediation.
- `rtk git diff --check` and `rtk git diff --cached --check` passed.
- Bounded added-line old-authority scan found no new `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, source-front-door, or source-artifact
  authority. The only broader `descriptor` match was negative ODS wording.
- No generated-bundle, `ssh rvv`, correctness, timing, parity, or performance
  claim was made in this Gate 2 slice.

### Self-Repair

- An initial positive fixed-string check used a regex pattern with unescaped
  bracket ranges. Re-ran the checks with `rg -F`.
- The dequant-clamp positive check initially searched for the product/dequant
  candidate id; the actual accepted candidate is the dequant-clamp candidate.
  The route/export itself passed and confirmed the provider-owned resource
  facts are still accepted.

### Spec Update Decision

[NO CHANGE] The existing RVV plugin and EmitC-route specs already require RVV
plugin-owned selected-body realization/provider facts before Common EmitC or
target artifact mirrors can accept a route. This round implements that contract
for the Gate 2 resource/remediation handoff and does not introduce a new
cross-project rule.

### Status

[OPEN] Macro task remains active. Gate 1 is complete. Gate 2
resource/remediation handoff slice is complete for product/dequant and packed-i4
representatives. Continue with Gate 3 only after confirming the changed
production path is the one used for generated artifact evidence: produce the
bounded artifact and `ssh rvv` evidence for at least one path changed by this
campaign. Gate 4 remains same-target measurement and dispatch/performance
policy consumption.

## Session 576: RVV low-precision Gate 3 packed-i4 artifact evidence

**Date**: 2026-06-10
**Task**: RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the same active macro task at Gate 3. This slice selected the
campaign-changed pre-realized
`widening_product_reduce_dequantize_f32` signed packed-i4 path and produced
bounded generated artifact plus real `ssh rvv` correctness evidence for that
path. The evidence is tied to provider-owned Gate 2 resource/remediation
handoff facts rather than fixture names, route ids, artifact names, or Common
EmitC inference.

The macro task remains open. Gates 1-3 are complete; Gate 4 remains same-target
measurement and dispatch/performance policy consumption.

### Main Changes

- Hardened `scripts/rvv_generated_bundle_abi_e2e.py` so packed-i4 remediation
  plan contract, remediation plan, statement strategy, and vector budget are
  explicit low-precision resource metadata facts in fake bundles, generated
  header checks, bundle validation, evidence summaries, and self-test negative
  cases.
- Extended generated artifact evidence with actual/expected selected candidate,
  resource decision mirror, operand form, packing layout, unpack intent,
  peak-live vector groups, vector-register budget, product/dequant region
  indexes, primitive chain facts, target capability mirrors, and packed-i4
  remediation facts after object/header agreement.
- Extended the external harness evidence with a metadata gate showing the
  packed-i4 scalar reference oracle is selected from validated provider-owned
  low-precision resource metadata.
- Updated the macro PRD to mark Gates 2 and 3 complete and leave Gate 4 as the
  precise continuation.

### Evidence

- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed,
  including missing packed-i4 remediation plan metadata and stale packed-i4
  remediation statement metadata negative cases.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir ... --dry-run --run-id gate3-packed-i4-resource-handoff-dry-run --overwrite` passed and wrote evidence under
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/gate3-packed-i4-resource-handoff-dry-run/`.
- `rtk ninja -C build tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Direct stale/missing remediation checks passed:
  stale handoff `remediation_statement_strategy`, missing handoff
  `remediation_vector_budget`, and stale artifact
  `low_precision_resource.remediation_statement_strategy` all fail closed with
  targeted diagnostics before stale route/export acceptance.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir ... --run-id gate3-packed-i4-resource-handoff-ssh --overwrite` passed and wrote evidence under
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/gate3-packed-i4-resource-handoff-ssh/`.
- The `ssh rvv` evidence reports remote `riscv64`, `clang` 18.1.3, remote
  compile success, remote run success, `ssh_evidence: true`, `status:
  success`, and packed-i4 correctness for runtime counts `0,1,17,257`, two data
  patterns, and scales `-0.125,0.375`.

### Self-Repair

- The project image still does not provide `FileCheck` or `llvm-lit`, so the
  focused fixture negative cases were executed directly with built
  `tcrv-opt`/`tcrv-translate` pipelines plus fixed diagnostic matches.

### Spec Update Decision

[NO CHANGE] The existing RVV plugin and EmitC-route specs already require
provider-owned route facts before Common EmitC or target artifact mirrors can
accept a route. This round tightens the Gate 3 evidence tool and task record; it
does not introduce a new durable architecture rule.

### Status

[OPEN] Macro task remains active. Gates 1-3 are complete. Continue with Gate 4:
same-target measurement and dispatch/performance policy consumption using
provider-owned evidence for the same production capability family, including
conservative fallback when no win is measured.

## Session 577: RVV low-precision Gate 4 measurement-to-policy consumption

**Date**: 2026-06-10
**Task**: RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the same active macro task at Gate 4. This slice used the Gate 3
packed-i4 `widening_product_reduce_dequantize_f32` generated artifact path to
produce real same-target `ssh rvv` measurement evidence, then updated the
production low-precision dispatch/performance policy boundary to consume that
provider-owned evidence. The measured result is a regression/no-win, so the
policy remains conservative: correctness route support is preserved, but
performance preference and performance-win claims are denied.

All campaign gates are now complete.

### Main Changes

- Updated the accepted packed-i4 Gate 4 measurement identity to
  `gate4-low-precision-packed-i4-same-target-measure-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`.
- Updated the accepted same-target best-speedup range to
  `0.689938..0.705891` and kept the previous `0.683805..0.705257` range as a
  stale-evidence negative case.
- Kept dispatch/performance policy consumption tied to provider-owned
  low-precision resource/Gearbox facts, measurement target signature,
  primitive chain, correctness records, and remediation evidence.
- Preserved fail-closed policy rejection for missing, stale, cross-target,
  untrusted, measurement-only, provider-disconnected, stale primitive, stale
  remediation, and stale sibling evidence.
- Synchronized focused script, RVV plugin, target artifact, generated-bundle,
  and RVV plugin spec expectations with the current Gate 4 evidence.

### Evidence

- Real same-target measurement passed with:
  `summaries=12`, `measurements=60`, `correctness=12`,
  `classification=regression`, `best_speedup_range=0.689938..0.705891`,
  `selection_eligible=false`, and `performance_win_claim_allowed=false`.
- The evidence is under
  `artifacts/tmp/rvv_generated_bundle_same_target_measure/gate4-low-precision-packed-i4-same-target-measure-ssh/`
  and reports `ssh_evidence: true`, `status: success`,
  `provider_maturity_outcome: regression`,
  `provider_dispatch_preference: not-performance-preferred`, and
  `contract_alignment: matches-provider-maturity-outcome`.
- Focused positive and negative checks cover current evidence acceptance,
  stale speedup range, stale evidence id, missing measurement evidence,
  cross-target evidence, untrusted/non-`ssh rvv` evidence, stale provider
  tie-back, measurement-only win promotion, stale primitive facts, stale
  remediation facts, and stale sibling evidence.

### Self-Repair

- The first same-target measurement attempt passed `--llvm-readobj llvm-readobj`
  and failed because the host image does not provide `llvm-readobj`. Re-running
  without that optional object-introspection argument produced the real
  same-target `ssh rvv` measurement evidence used by this slice.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the
current accepted Gate 4 measurement evidence id and best-speedup range so the
durable RVV policy contract matches the production constants and focused tests.

### Status

[READY TO ARCHIVE] Gates 1-4 are complete. The task is ready to finish/archive
after focused checks, Trellis validation, old-authority scans, whitespace
checks, and one coherent Gate 4 commit.

## Session 578: RVV low-precision production selected-dispatch Gate 1

**Date**: 2026-06-10
**Task**: RVV low-precision production-kernel selected-dispatch campaign
**Branch**: `main`

### Summary

Created the new macro Trellis task for the Stage 2 RVV low-precision
production-kernel selected-dispatch campaign and completed Gate 1 only. This
slice connected selected `tcrv.exec.dispatch` case/fallback facts to the
low-precision performance policy and RVV contraction route-family provider
validation, so the bounded packed-i4 widening product-reduce-dequantize path
must consume provider-owned selected-dispatch boundary facts before policy
acceptance.

The Gate 4 same-target measurement remains a no-win/regression input. Gate 1
therefore preserves route/correctness support but keeps performance preference
denied and requires the conservative fallback path.

### Main Changes

- Added `RVVLowPrecisionSelectedDispatchPolicyBoundary` to carry selected
  dispatch case/fallback facts through RVV route analysis.
- Collected the boundary from real `tcrv.exec.dispatch` case/fallback ops:
  selected variant, role, origin, policy, runtime guard, fallback variant,
  fallback role/origin/policy, and selected-dispatch mirrors.
- Added strict low-precision policy overloads that verify selected-dispatch
  boundary facts together with accepted primitive/resource/measurement handoff.
- Made RVV contraction route-family provider validation consume the boundary
  when low-precision resource selection is present.
- Added focused positive and negative tests for accepted structural facts,
  missing fallback facts, stale/metadata-only fallback origin, and plan/header
  mirror propagation.

### Evidence

- `ninja -C build tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `tcrv-opt` plan checks for selected dispatch case/fallback mirrors on the
  packed-i4 pre-realized selected-body artifact fixture.
- `tcrv-opt | tcrv-translate --tcrv-export-target-header-artifact` header
  checks for the same selected dispatch case/fallback mirrors.

### Self-Repair

- An initial attempt to analyze the dispatch-case fixture directly inside the
  C++ unit test used a helper fixture that did not carry a materialized
  `tcrv.exec.dispatch`. The repair kept C++ coverage focused on strict policy
  and provider validation, then used the MLIR artifact fixture to prove real
  dispatch case/fallback collection and export propagation.

### Spec Update Decision

[NO SPEC UPDATE] Gate 1 implements the active campaign PRD and existing RVV
plugin/EmitC route contract. The campaign-level durable rule should be captured
after Gates 2-4 prove the full production selected-dispatch workflow, not after
this policy-boundary slice alone.

### Status

[OPEN] Gate 1 is complete. Gates 2-4 remain unchecked. Continue with Gate 2:
carry the selected/pre-realized low-precision production-kernel body through
route/export/artifact validation while preserving conservative fallback
semantics and structural primitive/resource/measurement policy input.

## Session 579: RVV low-precision selected-dispatch Gate 2

**Date**: 2026-06-10
**Task**: RVV low-precision production-kernel selected-dispatch campaign
**Branch**: `main`

### Summary

Completed Gate 2 only for the active macro task. The selected/pre-realized
packed-i4 widening product-reduction/dequantization route now carries the
provider-owned selected-dispatch case/fallback policy boundary through RVV
route validation into target artifact validation. Target artifact acceptance
checks the selected dispatch case and conservative fallback mirrors against
that provider boundary while preserving the existing primitive signedness/type,
Gearbox resource/remediation, artifact-evidence, same-target measurement,
runtime AVL/VL, ABI/header, and provider mirror checks.

The accepted same-target measurement remains a no-win/regression input, so
Gate 2 preserves correctness support and conservative fallback behavior without
claiming a new performance-preferred route.

### Main Changes

- Added `RVVLowPrecisionSelectedDispatchPolicyBoundary` to the widening
  dot-reduce route validation contract and populated it from the RVV route
  description.
- Added target artifact validation for
  `tcrv_rvv.selected_dispatch_case_mirror` and
  `tcrv_rvv.selected_dispatch_fallback_mirror` against provider-owned
  selected-dispatch boundary facts.
- Routed packed-i4 product-reduction/dequantization target artifact
  performance-policy verification through the selected-dispatch policy
  overload when boundary facts are present.
- Added focused target artifact positive/negative coverage for selected case
  and fallback mirrors, missing fallback mirror metadata, and provider
  boundary missing fallback facts.
- Added selected pre-realized fixture checks for stale case/fallback mirror
  rejection during target header artifact export.

### Evidence

- `ninja -C build tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `tcrv-opt` plan check for selected dispatch case/fallback mirrors on the
  packed-i4 pre-realized selected-body artifact fixture.
- `tcrv-opt | tcrv-translate --tcrv-export-target-header-artifact` positive
  header artifact check for selected dispatch mirrors, runtime AVL, header,
  provider support, and `not-performance-preferred` facts.
- `tcrv-opt | tcrv-translate --tcrv-rvv-emitc-to-cpp` check for the low
  precision RVV load, widening product, widening reduction, seed, and store
  intrinsics.
- Stale selected-dispatch case and fallback mirror target header artifact
  checks fail closed with provider-boundary diagnostics.

### Self-Repair

- `clang-format` is not available in this environment, so formatting was
  checked by source inspection and `git diff --check`.
- `FileCheck`, `not`, and `llvm-lit` are not available in this environment, so
  the new MLIR `RUN` lines were manually exercised with the local `tcrv-opt`
  and `tcrv-translate` binaries.

### Spec Update Decision

[NO SPEC UPDATE] Gate 2 implements the active PRD and existing RVV
plugin/EmitC route/target-artifact contracts. No new durable cross-task rule was
introduced; the full campaign rule should only be promoted to spec after Gates
3-4 close the production selected-dispatch workflow.

### Status

[OPEN] Gate 2 is complete. Gates 3-4 remain unchecked. Continue with Gate 3:
update only the bounded same-target measurement policy input and demonstrate
preferred-route or conservative-fallback behavior from provider-owned facts,
without making measurement evidence or artifact metadata route authority.

## Session 580: RVV low-precision selected-dispatch Gate 3

**Date**: 2026-06-10
**Task**: RVV low-precision production-kernel selected-dispatch campaign
**Branch**: `main`

### Summary

Completed Gate 3 only for the active macro task. Same-target measurement
evidence is now represented as bounded low-precision selected-dispatch policy
input before it can affect dispatch preference. The RVV policy validates that
the input is tied to provider-owned selected candidate, route-family plan,
provider-supported mirror, runtime ABI order, primitive chain facts,
remediation handoff, target capability mirrors, and selected dispatch
case/fallback boundary facts.

The accepted packed-i4 no-win/regression evidence still preserves route and
correctness support while selecting the conservative fallback path. A matched
measured-win input can select a performance-preferred path only after provider
resource, maturity, remediation, target, and dispatch facts already agree.
Gate 3 makes no new runtime, correctness, or performance claim.

### Main Changes

- Added `RVVLowPrecisionSameTargetMeasurementPolicyInput` plus builder,
  consumer, diagnose/evaluate/resolve/verify overloads in the RVV
  low-precision performance policy.
- Routed packed-i4 resource remediation validation and selected-dispatch
  provider validation through the bounded same-target policy input.
- Extended the same-target measurement script maturity input with authority,
  correctness count, same-target/ssh evidence, target profile, and provider
  tie-back fields aligned with the C++ contract.
- Added focused C++ tests for accepted fallback, matched measured-win
  preference, missing measurement identity, cross-target input, metadata-only
  provider support, route-family-only claims, disconnected candidates,
  measurement-only primitive claims, and stale runtime ABI.

### Evidence

- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- `ninja -C build tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `tcrv-opt` plan check for the packed-i4 pre-realized selected-body artifact.
- `tcrv-opt | tcrv-translate --tcrv-export-target-header-artifact` positive
  header check for `not-performance-preferred` and selected dispatch mirrors.
- `git diff --check`
- `git diff --cached --check`
- Added-lines old-authority scan over changed source/test/script files had no
  hits; the only broader scan hits were PRD prohibition text.

### Spec Update Decision

[NO SPEC UPDATE] Gate 3 implements the active macro PRD and existing RVV
plugin/EmitC route/target-artifact contracts. The campaign-level rule should be
promoted only after Gate 4 closes final `ssh rvv` evidence and task closeout.

### Status

[OPEN] Gate 3 is complete. Gate 4 remains unchecked. Continue with final
`ssh rvv` correctness/performance evidence and campaign closeout only when the
macro acceptance gates are genuinely satisfied.

## Session 581: RVV low-precision selected-dispatch Gate 4 closeout

**Date**: 2026-06-10
**Task**: RVV low-precision production-kernel selected-dispatch campaign
**Branch**: `main`

### Summary

Completed Gate 4 for the active macro task. The final same-target `ssh rvv`
measurement for the bounded packed-i4
`widening_product_reduce_dequantize_f32` selected-dispatch path completed
successfully, but still classifies as a regression/no-win. The selected-dispatch
policy therefore preserves route/correctness support and chooses the
`correctness-fallback` path while denying performance selection and
performance-win claims.

### Main Changes

- Updated the accepted Gate 4 measurement identity to
  `gate4-selected-dispatch-final-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`.
- Updated the accepted same-target best-speedup range to
  `0.684318..0.708057` and retained the previous accepted range
  `0.689938..0.705891` as a stale-evidence negative case.
- Synchronized provider resource constants, low-precision dispatch/performance
  policy constants, generated-bundle script mirrors, target header/artifact
  fixture expectations, and RVV plugin spec text.
- Preserved the measured-win path only for a matched provider maturity,
  selection eligibility, remediation, dispatch, target, and same-target
  measurement fixture; measurement-only win promotion remains fail-closed.
- Updated the active PRD to mark Gates 1-4 complete and record the exact
  archive-ready closeout state.

### Evidence

- Real same-target `ssh rvv` measurement:
  `python3 scripts/rvv_generated_bundle_same_target_measure.py --artifact-root artifacts/tmp/rvv_generated_bundle_same_target_measure --run-id gate4-selected-dispatch-final-ssh --overwrite --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`
- Result: `summaries=12`, `measurements=60`, `correctness=12`,
  `classification=regression`, `best_speedup_range=0.684318..0.708057`,
  `provider_performance_selection_eligible=false`, and
  `performance_win_claim_allowed=false`.
- Rebuilt focused binaries:
  `ninja -C build tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Same-target dry-run after contract sync:
  `python3 scripts/rvv_generated_bundle_same_target_measure.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_same_target_measure --run-id gate4-selected-dispatch-final-dry-run --overwrite --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`
- Generated-bundle ABI dry-run after contract sync:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id gate4-selected-dispatch-final-dry-run --dry-run --overwrite`
- Direct `tcrv-opt | tcrv-translate --tcrv-export-target-header-artifact`
  check showed header mirrors for
  `gate4-selected-dispatch-final-ssh` and `0.684318..0.708057`.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the final
selected-dispatch Gate 4 evidence id and best-speedup range. This is the
campaign-level promotion deferred by Gate 3.

### Status

[READY TO ARCHIVE] Gates 1-4 are complete. The remaining work is mechanical
final verification, old-authority scan, whitespace checks, archive, and one
coherent Gate 4 closeout commit.

## Session 582: Packed-i4 performance remediation Gate 2b schedule decision

**Date**: 2026-06-10
**Task**: Stage2 RVV packed-i4 performance-remediation campaign
**Branch**: `main`

### Summary

Completed the Gate 2b production slice for the active macro task. Packed-i4
remediation schedule facts now drive an RVV-owned Gearbox/resource-aware
schedule decision instead of remaining only mirrored metadata. The chosen
decision is accepted only when the provider-owned low/high unpack plan, pair-sum
product plan, single-`vwredsum` reduction plan, two-region runtime-AVL VL plan,
and 7-of-32 vector-group budget all agree.

### Main Changes

- Added `schedule_decision_contract`, `schedule_decision`, and
  `schedule_decision_reason` as provider-owned packed-i4 resource facts.
- Threaded the selected decision through the RVV resource candidate,
  selected-body realization, Gearbox cross-region handoff, route-family
  selection, statement/route planning, target artifact validation, target header
  bundle, and generated-bundle dry-run evidence scripts.
- Added fail-closed coverage for stale or missing schedule-decision facts in
  RVV plugin/provider planning and target artifact mirror validation.
- Preserved the prior no-win/regression policy input:
  `correctness-fallback`, `not-performance-preferred`, and no performance-win
  claim.

### Evidence

- Rebuilt focused binaries:
  `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
  scripts/rvv_generated_bundle_same_target_measure.py`.
- Manual packed-i4 `tcrv-opt` pipeline verified realized body and handoff
  schedule-decision facts, rejected a stale handoff decision, materialized route
  metadata, exported target header mirrors, and rejected a stale artifact mirror.
- Generated-bundle ABI dry-run verified the schedule decision in evidence,
  bundle index, and target header.
- Same-target measurement dry-run verified `schedule_decision` and
  `provider_schedule_decision` fields without executing `ssh rvv`.
- Added-line legacy-authority scan found no new `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, source-front-door, or descriptor authority.
- `git diff --check` passed.

### Spec Update Decision

[NO SPEC UPDATE] Gate 2b implements the active macro PRD using existing RVV
plugin-local selected-body realization, provider-owned route planning, and
target-artifact validation rules. Promote only if Gate 3/4 evidence establishes
a durable new policy rule.

### Status

[OPEN] Gate 2b is complete. Gate 3 generated artifact correctness plus real
same-target `ssh rvv` measurement and Gate 4 measured selected-dispatch policy
consumption remain open. `.trellis/.current-task` stays active.

## Session 583: Packed-i4 performance remediation Gate 3 evidence and measurement

**Date**: 2026-06-10
**Task**: Stage2 RVV packed-i4 performance-remediation campaign
**Branch**: `main`

### Summary

Completed Gate 3 for the active macro task. The Gate 2b packed-i4 schedule
decision now appears explicitly in generated-bundle evidence and same-target
measurement evidence, and the updated packed-i4
`widening_product_reduce_dequantize_f32` path has real same-target `ssh rvv`
correctness and timing evidence.

The real measurement still classifies as a regression. The path remains
correctness-supported but not performance-preferred:
`provider_performance_selection_eligible=false` and
`performance_win_claim_allowed=false`.

### Main Changes

- Added the remediation schedule facts and Gate 2b schedule-decision facts to
  `packed_i4_resource_remediation_evidence` in the generated-bundle ABI
  evidence summary.
- Added `measurement_schedule_decision_evidence` and
  `same_target_schedule_decision_evidence` to the same-target measurement
  evidence so the measurement record carries both `schedule_decision` and
  `provider_schedule_decision`.
- Updated focused script dry-run tests to assert the new generated-artifact and
  same-target schedule-decision evidence fields.
- Updated the active PRD to mark Gate 3 complete, record the measurement path,
  and leave Gate 4 open.

### Evidence

- Rebuilt focused binaries:
  `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
  scripts/rvv_generated_bundle_same_target_measure.py`.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`.
- Generated-bundle ABI dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id gate3-packed-i4-schedule-decision-dry-run --overwrite --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`.
- Same-target dry-run:
  `python3 scripts/rvv_generated_bundle_same_target_measure.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_same_target_measure --run-id gate3-packed-i4-schedule-decision-dry-run --overwrite --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`.
- Real same-target `ssh rvv` measurement:
  `python3 scripts/rvv_generated_bundle_same_target_measure.py --artifact-root artifacts/tmp/rvv_generated_bundle_same_target_measure --run-id gate3-packed-i4-schedule-decision-ssh --overwrite --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20 --ssh-target rvv`.
- Result: `summaries=12`, `measurements=60`, `correctness=12`,
  `classification=regression`, `best_speedup_range=0.689815..0.705331`,
  `provider_performance_selection_eligible=false`, and
  `performance_win_claim_allowed=false`.
- Evidence path:
  `artifacts/tmp/rvv_generated_bundle_same_target_measure/gate3-packed-i4-schedule-decision-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`.
- `git diff --check` passed before journal/PRD updates.

### Spec Update Decision

[NO SPEC UPDATE] Gate 3 completed the active macro PRD using existing RVV
plugin/testing contracts. The durable policy rule should be promoted only after
Gate 4 consumes this measurement in selected-dispatch policy.

### Status

[OPEN] Gate 3 is complete. Gate 4 remains open: consume the Gate 3 measurement
evidence in selected-dispatch policy and preserve correctness fallback unless a
future provider-backed same-target win justifies `performance-preferred`.


## Session 574: Session 584: Packed-i4 remediation Gate 4 policy consumption

**Date**: 2026-06-10
**Task**: Session 584: Packed-i4 remediation Gate 4 policy consumption
**Branch**: `main`

### Summary

Completed the active packed-i4 performance remediation macro task by threading Gate 2b schedule-decision facts and Gate 3 same-target regression evidence into the RVV low-precision selected-dispatch performance policy; preserved correctness fallback and archived the task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending-gate4-closeout` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 576: Stage2 RVV Gearbox resource-planning Gate 1

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel Gearbox resource-aware selected-body realization campaign
**Branch**: `main`

### Summary

Completed Gate 1 of the new open macro campaign: introduced a named RVV-owned
low-precision production resource-planning contract and connected it through
Gearbox resource facts, selected-body realization, provider validation, route
metadata mirrors, and target artifact validation. Gates 2-4 remain open.

### Main Changes

- Added `tcrv_rvv.low_precision_resource.planning_contract =
  "rvv-low-precision-production-resource-planning-contract.v1"` as the explicit
  Gate 1 handoff.
- Populated the contract from the selected low-precision resource candidate and
  copied it into realized product-reduction/dequant selected-body facts.
- Made provider route-family validation and target artifact validation reject
  stale or missing contract facts before route/artifact acceptance.
- Updated RVV plugin spec and focused plugin/target/FileCheck coverage.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- Positive FileCheck for
  `test/Transforms/RVV/rvv-gearbox-widening-product-reduce-dequantize-f32.mlir`.
- Negative FileCheck for stale pre-realized planning contract diagnostics.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-10-stage2-rvv-production-kernel-gearbox-resource-realization-campaign`.
- `git diff --check` and `git diff --cached --check`.
- Bounded added-line scan found no new source/test old RVV route authority,
  descriptor/source-front-door authority, or q8/q4/llama route authority.

### Spec Update Decision

[SPEC UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the
production resource-planning contract id, attr name, provider/target validation
behavior, failure matrix, and test obligations.

### Status

[OPEN MACRO] Gate 1 is complete. Keep the Trellis task active for Gates 2-4.

### Next Steps

- Gate 2: make selected-body realization use the resource plan as the
  implementation driver for a representative low-precision/contraction body
  while preserving semantics, runtime AVL/VL, ABI roles, variant origin,
  dispatch, and fallback behavior.


## Session 585: Stage2 RVV low-precision primitive-surface Gate 1

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Created the macro Trellis task for the Stage2 RVV low-precision contraction
primitive-surface campaign and completed the Gate 1 slice. The production
change hardens provider-owned product-reduction primitive-surface validation
instead of adding another generated-bundle or `ssh rvv` evidence closeout.

### Main Changes

- Added a provider-side primitive-surface validator that compares
  `RVVLowPrecisionContractionResourceSelection` against
  `RVVLowPrecisionWideningReductionPrimitiveFacts` for source/product/
  accumulator/reduction dtype, signedness, SEW/LMUL, and final result dtype.
- Moved primitive-chain resource validation earlier in route-family plan and
  route-description validation so stale primitive facts fail before route
  acceptance.
- Added a focused RVV plugin C++ negative test for stale primitive product SEW:
  expected `16`, stale `32`.
- Seeded the macro PRD and JSONL context, leaving Gates 2-4 open.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- Signed widening-product export:
  `build/bin/tcrv-opt test/Target/RVV/explicit-selected-body-artifact-widening-product.mlir --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact`.
- Unsigned u8 widening-product export:
  `build/bin/tcrv-opt test/Target/RVV/explicit-selected-body-artifact-widening-product-unsigned-u8.mlir --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact`.
- Unsigned u8 stale primitive signedness negative chain rejected as expected.
- `llvm-lit` was not available in the local build tree, so the key lit RUN
  chains were executed directly.
- `git diff --check` and `git diff --cached --check` passed.

### Spec Update Decision

[NO SPEC UPDATE] This slice implements an existing RVV plugin spec contract:
low-precision widening-reduction primitive facts must be provider-owned and
fail-closed before route/artifact acceptance. The PRD records the campaign
state; no new durable architecture rule was introduced.

### Status

[OPEN] Gate 1 is complete. Gates 2-4 remain open. Next continuation point:
make RVV plugin-local selected-body realization/Gearbox scheduling consume the
reusable primitive surface for a coherent low-precision contraction slice.


## Session 586: Stage2 RVV low-precision primitive-surface Gate 2a

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Completed one Gate 2 production sub-slice for the active macro task. The RVV
Gearbox schedule pass now consumes the provider-owned low-precision
widening-reduction primitive-surface contract before it writes selected
resource/schedule facts onto a pre-realized product-reduction-dequantization
body.

### Main Changes

- Added a Gearbox-pass primitive-surface validator in
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp` that maps bounded
  product-reduction dequant/dequant-clamp resource operations to provider
  `RVVSelectedBodyOperationKind`, obtains
  `RVVLowPrecisionWideningReductionPrimitiveFacts`, and compares candidate
  source/product/accumulator/result dtype, SEW/LMUL, signedness, primitive
  contracts, chain kind, product/reduction relations, intrinsics, seed splat,
  layouts, and store-VL before schedule attrs are materialized.
- Kept selected-body realization/provider ownership intact. Common EmitC was
  not changed, and no q8/q4/llama label, artifact name, route id, descriptor,
  or status field became route authority.
- Added focused FileCheck coverage showing Gearbox schedule materialization
  carries primitive-chain resource facts on the pre-realized body, plus a stale
  schedule primitive negative chain that fails in selected-body realization
  before route construction.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- Gearbox primitive schedule positive chain:
  `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir --tcrv-rvv-materialize-gearbox-schedules | /usr/bin/FileCheck-20 test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir --check-prefix=GEARBOX-SCHEDULE-PRIMITIVE`.
- Stale schedule primitive negative chain:
  `build/bin/tcrv-opt ... --tcrv-rvv-materialize-gearbox-schedules | sed ... | build/bin/tcrv-opt --tcrv-materialize-selected-lowering-boundaries`,
  checked with `/usr/bin/FileCheck-20 --check-prefix=STALE-SCHEDULE-PRIMITIVE`
  and explicit nonzero exit status.
- `git diff --check` and `git diff --cached --check` passed.
- Bounded authority scan over touched files found only PRD non-goal wording and
  the existing negative `rvv-i32m1` stale-authority test.
- Local `FileCheck` and `not` were not on PATH; `/usr/bin/FileCheck-20` and
  explicit shell exit-code checks were used. `llvm-lit` was not available
  locally.

### Spec Update Decision

[NO SPEC UPDATE] This slice implements the existing RVV plugin and variant
pipeline contracts requiring low-precision Gearbox scheduling and selected-body
realization to consume provider-owned primitive/resource facts before route
construction. No new durable architecture rule was introduced.

### Status

[OPEN] Gate 2a is complete. The macro task remains active: Gate 2 still needs
remaining representative coverage before Gate 3, especially dequant-clamp and
packed-i4 resource paths where provider/statement-plan/target boundaries must
prove they consume selected resource facts rather than mirrors. Gates 3 and 4
remain future work.


## Session 587: Stage2 RVV low-precision primitive-surface Gate 2b

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Completed one Gate 2b representative resource-consumption slice for the active
macro task. The RVV Gearbox schedules pass now preserves the dequant-clamp
resource operation when re-consuming an already realized product-reduction
cross-region handoff, and provider plan verification no longer lets the
packed-i4 Gate 4 performance-policy gate block non-packed dequant-clamp Gate 2
realization.

### Main Changes

- Updated `lib/Plugin/RVV/RVVGearboxSchedules.cpp` so
  `validateLowPrecisionProductDequantGearboxBody` reports whether the realized
  consumer is a direct dequant store or a lower/upper dequant-clamp
  compare/select/store chain. `materializeGearboxForWithVL` now uses that
  body-derived fact to write the matching dequant or dequant-clamp
  low-precision resource memory form onto both producer and consumer
  `with_vl` regions.
- Updated
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp` so
  selected-dispatch low-precision performance-policy verification is entered
  only for provider-selected packed-i4 resource candidates. Packed-i4 policy
  checks remain active; non-packed dequant/dequant-clamp paths are not forced
  through packed-i4 measurement evidence.
- Added dequant-clamp FileCheck coverage for the realized handoff path after a
  second Gearbox schedules pass, plus a stale memory-form negative chain that
  fails before emission-plan route construction with the expected clamp memory
  form, stale dequantize form, and selected dequant-clamp candidate.
- Refreshed the packed-i4 representative fixture checks to match actual printed
  provider-owned attr order, then reran selected-body, statement-plan,
  target-header, CPP, handoff-remediation stale, and artifact-schedule stale
  checks. No packed-i4 production source gap was found in this seam.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- Dequant-clamp positive selected-body chain:
  `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32.mlir --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | /usr/bin/FileCheck-20 ... --check-prefix=REALIZED`.
- Dequant-clamp realized handoff resource-consumption chain:
  `build/bin/tcrv-opt ... --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries --tcrv-rvv-materialize-gearbox-schedules | /usr/bin/FileCheck-20 ... --check-prefix=GEARBOX-CONSUME`.
- Dequant-clamp stale resource memory negative chain:
  same realized handoff pipeline with a sed-injected dequantize memory form,
  then `build/bin/tcrv-opt --tcrv-materialize-emission-plans`, checked with
  `/usr/bin/FileCheck-20 --check-prefix=STALE-RESOURCE-MEMORY` and explicit
  nonzero exit status.
- Dequant-clamp statement-plan and target-header chains:
  `--tcrv-materialize-emission-plans` with `PLAN`, then
  `build/bin/tcrv-translate --tcrv-export-target-header-artifact` with
  `HEADER`.
- Packed-i4 representative checks: `REALIZED`, `PLAN`, `HEADER`, `CPP`,
  `STALE-PACKED-HANDOFF-REMEDIATION`, and
  `STALE-ARTIFACT-SCHEDULE-DECISION`.

### Spec Update Decision

[NO SPEC UPDATE] This slice implements existing RVV plugin and selected-dispatch
contracts: representative low-precision resource paths must consume
provider-owned primitive/resource facts, and packed-i4 measurement policy must
remain scoped to packed-i4 candidates. No new durable architecture rule was
introduced.

### Status

[OPEN] Gate 2b representative dequant-clamp and packed-i4 resource consumption
is complete. Gate 2 still needs a final closure audit for any remaining
low-precision product-reduction representative source gaps. Gates 3 and 4
remain future work. No runtime/correctness/performance claim was made, so no
`ssh rvv` evidence was required in this slice.


## Session 588: Stage2 RVV low-precision primitive-surface Gate 2 closure audit

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Closed Gate 2 for the active macro task by auditing the campaign's bounded
low-precision product-reduction representatives against direct production
source consumers. No missing selected-body, Gearbox/resource,
provider/statement-plan, or target-boundary consumer was found, so this slice
updates the macro PRD/journal and leaves production source unchanged.

### Source Inventory

- Plain widening product-reduce add is covered by provider primitive-fact
  validation and target route/statement validation. It does not require
  Gearbox resource scheduling, so absence of a Gearbox resource consumer is not
  a Gate 2 gap for that representative.
- Product-reduce dequantize f32 and product-reduce dequant-clamp f32 are
  consumed by `RVVContractionSelectedBodyRealizationOwner.cpp`,
  `RVVGearboxSchedules.cpp`,
  `RVVEmitCContractionRouteFamilyPlanOwners.cpp`, and
  `RVVTargetArtifactRouteFamilyValidation.cpp`.
- The explicit dequant-clamp selected-body fixture is
  `test/Target/RVV/explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32.mlir`.
  It validates the same provider/target boundary after selected-body
  realization.
- Packed-i4 product-reduce dequantize remains provider-owned across operand
  packing/unpack facts, remediation schedule, statement payload, target
  mirrors, and selected-dispatch policy scope.
- `RVVEmitCReductionAccumulationStatementPlanOwners.cpp` was audited as a
  relevant generic reduction seam. It is not the owner for this Gate 2
  low-precision product-reduction representative path; the contraction
  route-family plan owner contains the needed consumers.

### Evidence

- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- Plain product-reduce add: `PLAN`, `HEADER`, and `STALE-PRIM-ACC`.
- Product-reduce dequantize f32: `REALIZED`, `PLAN`, `HEADER`, and
  `STALE-PROVIDER-PRIMITIVE-RESOURCE`.
- Product-reduce dequant-clamp f32: `REALIZED`, `GEARBOX-CONSUME`, `PLAN`,
  `HEADER`, and `STALE-RESOURCE-MEMORY`.
- Explicit dequant-clamp selected body: `REALIZED`, `PLAN`, `HEADER`, and
  `STALE-PROVIDER`.
- Packed-i4 product-reduce dequantize: `REALIZED`, `PLAN`, `HEADER`, `CPP`,
  `STALE-PACKED-HANDOFF-REMEDIATION`, and
  `STALE-ARTIFACT-SCHEDULE-DECISION`.
- Final housekeeping for the slice: `git diff --check`,
  `git diff --cached --check`, and a bounded old-authority scan over touched
  task/journal text.

### Spec Update Decision

[NO SPEC UPDATE] This audit applies the existing RVV plugin, EmitC route, and
variant-pipeline contracts. It did not introduce a new durable architecture
rule.

### Status

[OPEN MACRO TASK] Gate 2 is closed by source-backed audit. The macro task stays
active because Gate 3 generated artifact correctness plus same-target `ssh rvv`
evidence and Gate 4 selected-dispatch/performance-policy consumption remain
future milestones. The next continuation point is Gate 3 for the representative
generated artifact correctness path.


## Session 589: Stage2 RVV low-precision primitive-surface Gate 3

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Closed Gate 3 for the active macro task with generated artifact correctness
and same-target `ssh rvv` evidence for the representative pre-realized
`widening_product_reduce_dequant_clamp_f32` path. No production source gap was
found; the existing Gate 1/Gate 2 consumers remained connected through
selected-body realization, Gearbox/resource facts, provider route planning,
target artifact export, generated bundle ABI consumption, and remote runtime
oracle checks.

### Main Evidence

- Non-dry-run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e_gate3/gate3-pre-realized-widening-product-reduce-dequant-clamp-f32-ssh/`.
- Root evidence: `status=success`, `ssh_evidence=true`,
  `input_mode=pre-realized-selected-body`, and
  `pre_realized_selected_body=true`.
- Remote target profile from compile stdout: `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`, and `Ubuntu clang version 18.1.3`.
- Remote runtime stdout ended with
  `PASS op=widening_product_reduce_dequant_clamp_f32 counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05`.
- The per-case remote output covered below/inside/above clamp cases, mixed
  signed positive/negative widening products, `source_preserved`,
  `accumulator_preserved`, and `tail_preserved`.
- Explicit selected-body dry-run also passed for
  `widening_product_reduce_dequant_clamp_f32`, proving the explicit compound
  selected-body entrance still materializes through the same provider-owned
  route and target artifact boundary.

### Checks

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`.
- Pre-realized dequant-clamp generated-bundle dry-run with counts
  `0,1,16,17,257`.
- Explicit dequant-clamp generated-bundle dry-run with counts `0,1,16,17,257`.
- Pre-realized dequant-clamp non-dry-run `ssh rvv` generated-bundle ABI run
  with counts `0,1,16,17,257`.
- Stale resource-memory negative chain: replacing
  `unit-stride-widening-product-reduce-dequant-clamp-f32` with
  `unit-stride-widening-product-reduce-dequantize-f32` still fails before
  emission-plan route construction and matches `STALE-RESOURCE-MEMORY`.

### Spec Update Decision

[NO SPEC UPDATE] This slice applies the existing RVV plugin, EmitC route,
variant-pipeline, and testing contracts. It does not introduce a new durable
architecture rule; it supplies the named Gate 3 generated artifact and
same-target correctness evidence for the macro campaign.

### Status

[OPEN MACRO TASK] Gate 3 is complete. The macro task remains active because
Gate 4 is still open: selected-dispatch/performance policy must consume
measurement and schedule facts without promoting measurement-only wins or
claiming performance preference from executable correctness alone.


## Session 590: Stage2 RVV low-precision primitive-surface Gate 4

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Closed Gate 4 for the active macro task by tightening the production target
artifact validation path to consume the structured same-target measurement
policy input for packed-i4 low-precision product-reduction candidates. The
policy still preserves executable route support and correctness execution while
denying performance preference for the current no-win/regression maturity
contract.

### Main Changes

- Updated `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` so both
  packed-i4 target artifact performance-policy validation points construct
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` from the provider-owned
  resource selection and accepted same-target measurement outcome before
  invoking `verifyRVVLowPrecisionPerformancePolicy`.
- Updated `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp` so the
  policy-input overload preserves the targeted
  `stale-sibling-route-measurement` diagnosis before running the full input
  provider-fact gate.
- Updated `test/Target/TargetArtifactExportTest.cpp` to assert the Gate 4
  target policy input carries provider primitive-chain, Gate 2 schedule, and
  target capability facts before target policy evaluation.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`.
- Packed-i4 dry-run same-target measurement fixture:
  `python3 scripts/rvv_generated_bundle_same_target_measure.py --dry-run
  --artifact-root artifacts/tmp/gate4-policy-input-dry-run --run-id
  gate4-policy-input-dry-run --overwrite --op-kind
  widening_product_reduce_dequantize_f32 --input
  test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir
  --measure-count 257 --measure-count 4096 --measure-count 65536
  --warmup-count 2 --repeat-count 5 --measure-iterations 8 --tcrv-opt
  build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj
  /usr/bin/llvm-readobj-20`.
- Dry-run evidence input reported contract
  `packed-i4-same-target-performance-maturity-evidence-input.v1`,
  `measurement_classification=not-measured`,
  `provider_schedule_decision=select-packed-i4-pair-sum-single-reduce-u1-two-region-budget-7of32.v1`,
  `provider_primitive_chain_kind=signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-vwredsum.v1`,
  `provider_performance_selection_eligible=false`,
  `provider_dispatch_preference=not-performance-preferred`,
  `performance_win_claim_allowed=false`,
  `performance_preference_denied=true`, and
  `correctness_execution_allowed=true`.

### Spec Update Decision

[NO SPEC UPDATE] This slice implements the existing RVV plugin Gate 4
dispatch/performance policy contract. It did not introduce a new durable rule;
it changed the target validation consumer to use the already-specified
same-target policy input interface.

### Status

[ARCHIVED MACRO TASK] Gates 1-4 are complete. The task PRD and task metadata
were updated for completion, and the task is ready to move to the June 2026
archive.


## Session 591: Stage2 RVV production-kernel measurement Gate 1

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel same-target measurement and selected-dispatch campaign
**Branch**: `main`

### Summary

Opened the new macro campaign for production-kernel same-target measurement and
selected-dispatch policy. Completed Gate 1 by replacing the fixed accepted
outcome helper as the production source for packed-i4 same-target policy input
with an explicit source-backed measurement-record boundary.

### Main Changes

- Added `RVVLowPrecisionSameTargetMeasurementRecord` and record-to-outcome /
  record-to-policy-input builders in the RVV low-precision performance policy
  API.
- Updated RVV provider validation, resource-remediation handoff checks,
  resource-selection policy checks, and RVV target artifact validation to build
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` from measurement records
  instead of directly calling `getAcceptedRVVPackedI4Gate4MeasurementOutcome()`.
- Preserved stale sibling-route diagnosis by routing sibling records through
  the policy handoff classifier, while still fail-closing stale target/runtime
  ABI/provider tie-backs at the record/input boundary.
- Updated RVV plugin and target artifact tests so accepted and measured-win
  outcomes are materialized from records, and target-side negative cases mutate
  record fields for wrong target/runtime ABI evidence.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`.
- `git diff --check`.
- `git diff --cached --check`.
- Bounded touched-file authority scan for removed fixed outcome helper and
  legacy RVV route authority. It found no `getAccepted...Outcome` residue and
  only pre-existing fail-closed/source-front-door test strings in touched test
  files.

### Spec Update Decision

[NO SPEC UPDATE] Gate 1 implements the existing Stage 2 requirement that
same-target measurement evidence remains input-only and that provider-owned
primitive/resource facts remain authoritative. No new durable spec rule was
introduced.

### Status

[OPEN MACRO TASK] Gate 1 is complete and committed as this round's coherent
slice. Gates 2-4 remain open; the next continuation point is Gate 2 same-target
`ssh rvv` measurement evidence for a representative low-precision contraction
generated artifact.


## Session 592: Stage2 RVV production-kernel measurement Gate 2

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel same-target measurement and selected-dispatch campaign
**Branch**: `main`

### Summary

Completed Gate 2 for the active macro task by connecting the representative
packed-i4 generated artifact same-target measurement output to the Gate 1
measurement-record boundary. The script now emits an explicit
`same_target_measurement_record` object, and the RVV policy API can materialize
that object into `RVVLowPrecisionSameTargetMeasurementRecord` before building
`RVVLowPrecisionSameTargetMeasurementPolicyInput`.

### Main Changes

- Added `buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceInput` and
  `buildRVVLowPrecisionSameTargetMeasurementPolicyInputFromEvidenceInput` to
  the low-precision performance policy API.
- Updated `scripts/rvv_generated_bundle_same_target_measure.py` so packed-i4
  same-target evidence emits `same_target_measurement_record` at per-op,
  summary, and root levels while keeping reporting-only fields outside the C++
  record shape.
- Added focused plugin tests for evidence-object parsing, policy-input
  construction, and missing measurement id / stale target / stale runtime ABI /
  stale primitive-chain fail-closed cases.
- Updated the script dry-run FileCheck coverage to assert the new record object.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the executable
  contract for the evidence record payload and C++ API signatures.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`.
- Packed-i4 same-target dry-run:
  `python3 scripts/rvv_generated_bundle_same_target_measure.py --dry-run
  --artifact-root artifacts/tmp/gate2-packed-i4-same-target-dry-run --run-id
  gate2-packed-i4-record-dry-run --overwrite --op-kind
  widening_product_reduce_dequantize_f32 --input
  test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir
  --measure-count 257 --measure-count 4096 --measure-count 65536
  --warmup-count 2 --repeat-count 5 --measure-iterations 8 --tcrv-opt
  build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj
  /usr/bin/llvm-readobj-20`.
- Packed-i4 real same-target `ssh rvv` run:
  `python3 scripts/rvv_generated_bundle_same_target_measure.py
  --artifact-root artifacts/tmp/gate2-packed-i4-same-target-measurement
  --run-id gate2-packed-i4-generated-artifact-measure --overwrite --op-kind
  widening_product_reduce_dequantize_f32 --input
  test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir
  --measure-count 257 --measure-count 4096 --measure-count 65536
  --warmup-count 2 --repeat-count 5 --measure-iterations 8 --tcrv-opt
  build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj
  /usr/bin/llvm-readobj-20`.
- Real evidence path:
  `artifacts/tmp/gate2-packed-i4-same-target-measurement/gate2-packed-i4-generated-artifact-measure/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`.
- Real run result: `status=success`, `ssh_evidence=true`,
  `classification=regression`, `measurement_summary_record_count=12`,
  `measurement_record_count=60`, `correctness_record_count=12`,
  `measurement_best_speedup_range=0.689567..0.705465`,
  `provider_runtime_abi_order=lhs,rhs,acc,scale,out,n`,
  `provider_primitive_chain_kind=signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-vwredsum.v1`,
  `provider_schedule_decision=select-packed-i4-pair-sum-single-reduce-u1-two-region-budget-7of32.v1`,
  and `performance_win_claim_allowed=false`.
- `git diff --check`.
- Focused diff old-authority scan found no newly added legacy RVV authority;
  the only diff hit was the PRD negative q8/q4/no-descriptor acceptance line.

### Spec Update Decision

[SPEC UPDATED] This slice added a new executable API and payload contract, so
`.trellis/spec/extension-plugins/rvv-plugin.md` now documents
`same_target_measurement_record`, the C++ evidence-input builders, validation
errors, good/base/bad cases, and required tests.

### Status

[OPEN MACRO TASK] Gate 2 is complete. The macro task remains active because
Gate 3 and Gate 4 remain open. The next continuation point is Gate 3:
selected-dispatch/performance preference must consume source-backed
measurement records plus primitive/resource facts with full stale sibling-route,
wrong-target, wrong-ABI, and wrong-primitive-chain diagnostics.

## 2026-06-10 - Stage2 RVV same-target measurement campaign Gate 3

### Summary

Completed Gate 3 for the active macro task by making the selected-dispatch and
performance-preference policy boundary consume
`RVVLowPrecisionSameTargetMeasurementRecord` directly. The current packed-i4
regression/no-win record now produces an explicit conservative
`correctness-fallback` policy decision, and a bounded measured-win fixture
selects `performance-preferred` only after the measurement record and
provider-owned maturity, eligibility, dispatch, remediation, schedule, target,
and primitive facts agree.

### Main Changes

- Added record-based `evaluateRVVLowPrecisionPerformancePolicy`,
  `resolveRVVLowPrecisionDispatchPerformancePolicy`, and
  `verifyRVVLowPrecisionPerformancePolicy` overloads, including
  `RVVLowPrecisionSelectedDispatchPolicyBoundary` variants.
- Rewired selected-dispatch provider and target artifact validation to call the
  record-consumption policy boundary instead of first materializing a policy
  input as the selected-dispatch authority.
- Added focused plugin tests proving accepted record consumption, conservative
  fallback resolution, measured-win promotion only with matching provider
  facts, and fail-closed stale target/runtime ABI/primitive/dispatch-boundary
  diagnostics.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the executable
  record-based dispatch/performance policy API and test obligations.
- Updated the active PRD to mark Gate 3 complete and leave Gate 4 as the next
  continuation point.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`.
- `git diff --check`.
- Focused added-line old-authority scan over touched files found no new legacy
  RVV route-authority, stale source-construction, or narrow benchmark-label
  authority hits.

### Spec Update Decision

[SPEC UPDATED] This slice changed the production API signature and policy
contract, so `.trellis/spec/extension-plugins/rvv-plugin.md` now documents the
record-based Gate 3 policy surface, resolver behavior, selected-dispatch
boundary validation, and required tests.

### Status

[OPEN MACRO TASK] Gate 3 is complete. The macro task remains active because
Gate 4 remains open. The next continuation point is Gate 4: perform the final
campaign audit separating correctness execution evidence, same-target
measurement evidence, and selected-dispatch/performance policy authority before
archiving.

## 2026-06-10 - Stage2 RVV same-target measurement campaign Gate 4

### Summary

Completed the Gate 4 final audit for the active macro task. Gates 1-3 remain
truthful: source-backed measurement records reach the C++ policy boundary,
representative same-target `ssh rvv` evidence remains evidence-only, and
selected-dispatch/performance decisions consume measurement records together
with provider/resource/schedule/runtime/target/dispatch facts. The final audit
keeps correctness execution, performance selection, performance-preferred path
selection, and performance-win claims separate.

### Main Changes

- Added focused plugin test coverage proving a parsed generated-evidence JSON
  record feeds the selected-dispatch
  `RVVLowPrecisionSameTargetMeasurementRecord` policy overload directly and
  still selects conservative `correctness-fallback`.
- Added record-level negatives for correctness-disabled evidence and
  measurement-only win promotion before selected-dispatch performance
  preference can be accepted.
- Updated the active PRD to mark Gates 1-4 complete and record that there is no
  remaining macro continuation point.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` so future Gate 4
  audits require parsed evidence-record selected-dispatch coverage, not only a
  helper-built representative record.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`.
- `python3 ./.trellis/scripts/task.py validate
  06-10-06-10-stage2-rvv-production-kernel-same-target-measurement-selected-dispatch-campaign`.
- `rg -n "getAcceptedRVVPackedI4Gate4MeasurementOutcome" include lib test
  scripts` found no remaining source/test/script use.
- Focused added-line scan over the touched C++ test found no newly added legacy
  RVV route-authority, source-front-door, descriptor, or q8/q4/llama authority.
- `git diff --check`.

### Spec Update Decision

[SPEC UPDATED] Gate 4 made an executable test obligation explicit:
`.trellis/spec/extension-plugins/rvv-plugin.md` now requires parsed evidence
records to feed the selected-dispatch record overload directly and requires
record-level correctness-disabled and measurement-only win promotion negatives.

### Status

[READY TO ARCHIVE] All campaign gates are complete. The task can be finished
and archived after final validation and the coherent Gate 4 commit.

## 2026-06-10 - Stage2 RVV Gearbox resource-planning Gate 2 handoff consumer

### Summary

Continued the active production-kernel Gearbox/resource-aware selected-body
realization macro task at Gate 2. This sub-slice makes the selected-body
realization handoff consume the Gate 1 planning contract structurally: the
realized `tcrv_rvv.gearbox_cross_region_handoff` now carries the selected
resource planning contract, and RVV dialect/provider validation rejects missing
or stale handoff planning-contract facts before route construction.

### Main Changes

- Added `planning_contract =
  "rvv-low-precision-production-resource-planning-contract.v1"` to the realized
  product-reduction Gearbox cross-region handoff.
- Made `GearboxCrossRegionHandoffOp::verify` require that handoff planning
  contract and keep it bounded metadata.
- Made direct route-slice recording and contraction route-family validation
  compare the handoff planning contract with the provider-selected
  `RVVLowPrecisionContractionResourceSelection`.
- Updated the representative product-reduction-dequant FileCheck coverage with
  positive, stale, and missing handoff planning-contract checks, and repaired the
  explicit realized fixture.
- Updated the RVV plugin spec with the executable handoff planning-contract
  requirement.

### Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`.
- `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `build/bin/tianchenrv-target-artifact-export-test`.
- Manual positive `tcrv-opt` check for
  `pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
  showing handoff `planning_contract`.
- Manual stale/missing handoff planning-contract negative `tcrv-opt` checks.
- `build/bin/tcrv-opt` emission-plan checks for the explicit product-dequant
  fixture, dequant-clamp fixture, and packed-i4 fixture.

### Status

[OPEN MACRO TASK] Gate 1 is complete. Gate 2 handoff planning-contract consumer
sub-slice is complete, but Gate 2 remains partially open for marker-level
resource-plan checks or another production representative if required. Gates 3
and 4 remain open. The next continuation point is the remaining Gate 2
resource-plan realization surface, not generated artifact or same-target
measurement evidence.


## Session 575: Stage2 RVV same-target measurement Gate 4 audit

**Date**: 2026-06-10
**Task**: Stage2 RVV same-target measurement Gate 4 audit
**Branch**: `main`

### Summary

Closed the active same-target measurement selected-dispatch macro campaign with parsed-record selected-dispatch audit coverage, record-level performance-claim negatives, spec update, and Trellis archive.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | Gate 4 audit closeout commit |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
