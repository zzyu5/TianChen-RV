> Continuation from `journal-26.md` (archived at ~1976 lines)

## Session 571: RVV packed-i4 Gate 4 artifact/export support

**Date**: 2026-06-09
**Task**: RVV low-precision packed-contraction primitive surface campaign

### Summary

Continued the active macro task for Gate 4. Replaced the target artifact
hard fail-close for selected signed packed-i4 resources with target
route-family validation that accepts export only when the rebuilt provider
route carries the exact low/high nibble sign-extension payload, low/high
widening product-reduction chain, and final carry assignment.

The target artifact test now builds the packed-i4 fixture from the typed
pre-realized selected body, runs selected-body realization in the fixture, then
rebuilds the provider route for target validation. Stale packed resource facts,
stale unpack-intent mirrors, and stale low/high nibble statement payloads fail
closed in the target bridge.

### Testing

- [OK] built `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] manual `FileCheck-20` pipelines for the focused Gearbox positive path
  and target REALIZED/PLAN/HEADER/stale-packed-mirror paths.
- [OK] `llvm-lit` and Python `lit` were unavailable; focused RUN lines were
  executed manually.
- [OK] No generated-bundle, `ssh rvv`, correctness, timing, or parity claim was
  made.

### Status

[OPEN] Macro task remains active. Gate 4 target artifact/export support is
complete for the accepted packed-i4 representative, but focused generated-bundle
evidence remains the next continuation point. Gate 5 same-target
correctness/performance remains blocked until executable generated artifacts
exist. Final coherent commit is created after this journal entry.

## Session 572: RVV packed-i4 Gate 4 generated-bundle evidence

**Date**: 2026-06-09
**Task**: RVV low-precision packed-contraction primitive surface campaign

### Summary

Continued the active macro task for the narrowed Gate 4 continuation point:
focused generated-bundle artifact evidence for the accepted signed packed-i4
product-reduction-dequant representative. The public selected-body consumer
boundary now accepts the packed-i4 low-precision resource decision, allowing
the explicit packed-i4 pre-realized body to flow through selected-body
realization, provider route rebuild, target artifact export, and generated
bundle dry-run evidence.

The generated-bundle evidence script now validates low-precision resource facts
from actual provider-owned metadata. The default byte/grouped path still checks
the `u2` profile. An explicit packed-i4 selected candidate switches the
evidence profile to the packed operand form, signed nibble layout, storage
width 8, effective width 4, sign-extension unpack intent, two-region Gearbox
handoff, packed lhs/rhs source loads, low/high nibble sign-extension, low/high
widening product/reduction, and final carry into `dot_acc_vec`.

Added a focused packed-i4 pre-realized selected-body fixture so the generated
bundle evidence can be reproduced without temporary input rewriting. This
round intentionally remains dry-run/artifact-only and does not claim `ssh rvv`
correctness, timing, parity, or performance.

### Testing

- [OK] built `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] default byte/grouped product-dequant generated-bundle dry-run.
- [OK] explicit packed-i4 product-dequant generated-bundle dry-run with
  `ssh_evidence = false`.
- [OK] Focused realized/header/EmitC RUN-equivalent checks for the new
  packed-i4 fixture. `FileCheck`, `llvm-lit`, and Python `lit` were unavailable
  in this environment, so the pipelines were executed manually with `rg`
  checks over the generated outputs.
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded added-line and touched-file authority scans found no new legacy
  i32/source-front-door/descriptor route authority drift.

### Status

[OPEN] Macro task remains active. Gate 4 artifact/export and focused
generated-bundle artifact evidence are complete for the accepted packed-i4
representative. Gate 5 same-target correctness/performance remains open and
blocked until executable packed-i4 generated artifact support and a truthful
scalar-baseline comparison exist. Final coherent commit is created after this
journal entry.

## Session 573: RVV packed-i4 Gate 5 executable correctness

**Date**: 2026-06-09
**Task**: RVV low-precision packed-contraction primitive surface campaign

### Summary

Continued the active macro task for Gate 5. The non-dry-run generated-bundle
path was already able to stage, compile, and execute the packed-i4 generated
artifact on `ssh rvv`, but the external C harness still used the default
unpacked-byte `i8*i8` reference oracle. That produced a real remote correctness
failure at `n=1` because the generated packed-i4 code correctly accumulates
both low and high signed i4 nibbles per packed byte while the old oracle treated
each byte as one signed i8 lane.

Updated `scripts/rvv_generated_bundle_abi_e2e.py` so harness generation derives
the packed-i4 oracle switch from validated provider-owned object/header
low-precision resource metadata. For the accepted packed-i4 product-dequant
representative, the harness now packs two signed i4 lanes into each int8_t
source byte, sign-extends low/high nibbles, accumulates both products per
packed byte into the scalar seed, applies runtime f32 scale, and checks source,
accumulator, and output sentinel preservation. The default byte/grouped
product-dequant path remains unchanged.

Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the executable
packed-i4 harness/oracle contract so future Gate 5 work keeps the reference
calculation metadata-derived from provider facts and preserves the default byte
oracle.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] packed-i4 generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/gate5_packed_i4_after_oracle_fix_dry`
- [OK] packed-i4 generated-bundle non-dry-run on `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/gate5_packed_i4_after_oracle_fix_ssh`
- [OK] remote compile on `ssh rvv`: `riscv64`, `/usr/bin/clang`,
  Ubuntu clang 18.1.3.
- [OK] remote run passed counts `1,7,16,17,257`, patterns `0,1`, scales
  `-0.125,0.375`, with `packed_i4_reference_oracle` and `ssh_evidence = true`.
- [OK] default byte/grouped product-dequant dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/gate5_default_byte_grouped_regression_dry`
- [OK] built `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`

### Status

[OPEN] Macro task remains active. Gate 5 executable packed-i4 correctness and
same-target scalar/reference comparison are now present for the accepted
representative, but timing/performance is still not measured. No performance
win, llama.cpp parity, or q4/q8 route-authority claim is made. Next
continuation point: add a named same-target scalar/baseline timing measurement
on `ssh rvv` for the accepted packed-i4 representative and report
win/regression honestly.

## Session 574: RVV packed-i4 Gate 5 same-target timing

**Date**: 2026-06-09
**Task**: RVV low-precision packed-contraction primitive surface campaign

### Summary

Continued the active macro task for the final Gate 5 timing milestone. This
round did not change production compiler C++/MLIR/TableGen code because the
accepted packed-i4 generated RVV artifact, provider route, target export, and
non-dry-run correctness path were already complete. The remaining owner was
evidence tooling: measure that accepted generated artifact against a named
scalar C baseline on the same `ssh rvv` target.

Extended `scripts/rvv_generated_bundle_same_target_measure.py` so a single
`--input` pre-realized selected-body fixture can be measured. The packed-i4
timing baseline is selected only after generated object/header metadata
validates provider-owned low-precision resource facts. The default unpacked
byte product-dequant measurement remains on
`scalar-c-reference/product-reduction-dequant-v1`; the accepted packed-i4
representative switches to
`scalar-c-reference/product-reduction-dequant-packed-i4-v1`.

Collected same-target timing evidence on `ssh rvv` for counts
`257,4096,65536`, patterns `0,1`, scales `-0.125,0.375`, warmups `2`,
repeats `5`, and iterations `8`. The raw evidence includes target profile,
compile flags, correctness guards before timing, 60 raw `MEASURE` records, and
12 parsed `SUMMARY` records. The generated RVV artifact did not beat the
scalar packed-i4 baseline: best speedup ranged from `0.752187` to `0.804100`,
so this is a no-win/regression result, not a performance-win or llama.cpp
parity claim.

Updated `.trellis/spec/testing/mlir-testing-contract.md` and
`.trellis/spec/extension-plugins/rvv-plugin.md` with the packed-i4
same-target timing contract: `--input` is a fixture override for one op only,
packed baseline selection must come from validated provider metadata, and
default product-dequant timing must not emit `packed_i4_reference_oracle`.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`
- [OK] `scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] default byte/grouped same-target measurement dry-run:
  `artifacts/tmp/gate4-same-target-measurement/gate5_default_byte_grouped_timing_regression_dry`
- [OK] packed-i4 same-target measurement dry-run:
  `artifacts/tmp/gate4-same-target-measurement/gate5_packed_i4_timing_dry`
- [OK] packed-i4 same-target measurement on `ssh rvv`:
  `artifacts/tmp/gate4-same-target-measurement/gate5_packed_i4_same_target_timing_ssh`
- [OK] default byte/grouped ABI e2e dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/gate5_default_byte_grouped_regression_after_timing_dry`
- [OK] packed-i4 ABI e2e dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/gate5_packed_i4_after_timing_dry`
- [OK] packed-i4 ABI e2e non-dry-run on `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/gate5_packed_i4_after_timing_ssh`
- [OK] built `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`

### Status

[COMPLETE] Macro campaign gates are complete for the accepted signed packed-i4
product-reduction-dequant representative. The Trellis task is archived as
completed; final pre-commit checks and a coherent commit remain before
wrap-up.

## Session 575: RVV production-kernel Gate 1 packed resource realization

**Date**: 2026-06-09
**Task**: RVV production-kernel capability campaign: resource-aware packed low-precision contraction realization

### Summary

Created the new macro campaign task after the packed-i4 primitive surface
campaign was archived. The new owner is production compiler/resource-aware
selected-body realization, not another generated-bundle or timing evidence
seam.

Repository inspection showed the Gate 1 foundation already existed:
`RVVContractionSelectedBodyRealizationOwner.cpp` consumes pass-produced
low-precision resource facts into producer/consumer `with_vl` realization,
`vsetvl_region_marker`, and `gearbox_cross_region_handoff`, and
`RVVEmitCContractionRouteFamilyPlanOwners.cpp` re-consumes those realization
facts before route acceptance. The bounded production gap was narrower:
several verifier and route-collection helpers still treated packed-i4 as
"not grouped, therefore default two-region schedule". That worked only because
the current packed-i4 region count equals the base byte path.

This slice added shared RVV Gearbox helpers in
`RVVGearboxSchedule.h` to derive support class, expected `vsetvl` region
count, producer/dequant marker indices, and product phase from the selected
low-precision realization decision. The selected-body realizer, RVV dialect
handoff verifier, route-family realization-structure validator, and route
collector now call those helpers. Packed-i4 realization decision is therefore
explicitly consumed across producer/consumer scope and fails closed for stale
decision, stale region count, or stale `from_phase`.

### Testing

- [OK] built `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] manual packed-i4 fixture pipelines for realized body, target header,
  emitted C++, stale packed resource decision, stale packed region count, and
  stale packed `from_phase`.
- [OK] manual non-packed product-reduction-dequant fixture pipelines for
  realized body, target header, and missing-resource fail-close.
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded added-line authority scan found no new legacy i32,
  source-front-door, descriptor, or Common-EmitC authority drift.

`FileCheck` and `llvm-lit` are unavailable in this environment, so the fixture
checks were run as equivalent `tcrv-opt` / `tcrv-translate` pipelines with
targeted `rg` assertions over key output and diagnostics.

### Status

[OPEN] Gate 1 production compiler slice is complete and committed in the active
macro task. The macro task remains active because Gates 2-4 are not complete.
Next continuation point: Gate 2 route/statement/artifact validation must consume
the explicit packed-resource schedule facts without Common EmitC or metadata
mirrors becoming semantic authority.
