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
