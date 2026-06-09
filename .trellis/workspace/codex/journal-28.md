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
