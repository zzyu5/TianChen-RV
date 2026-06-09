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
