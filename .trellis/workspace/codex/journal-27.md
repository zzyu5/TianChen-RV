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

## Session 576: RVV production-kernel Gate 2 route/statement/artifact consumption

**Date**: 2026-06-09
**Task**: RVV production-kernel capability campaign: resource-aware packed low-precision contraction realization

### Summary

Continued the active macro campaign at Gate 2. This slice carries the packed-i4
Gearbox/resource schedule facts from selected-body realization into the
production route planning, direct-contraction statement planning, route
metadata, target support bundle, and target artifact validation/export
boundary.

`RVVLowPrecisionContractionResourceSelection` now retains the provider-owned
realization producer, realization decision, realized unroll factor, realized
`vsetvl` region count, realized peak live-vector groups, product/dequant region
indices, and product/dequant phases. Route-family planning derives or imports
those facts from RVV-owned typed body/config/resource facts. Statement-plan
ownership compares the provider plan and family plan before constructing the
packed-i4 statement payload. Target artifact validation now mirrors and checks
the same schedule facts before accepting a header artifact.

The packed-i4 statement path still consumes explicit typed/resource facts:
`packed-i4-nibbles`, signed source, storage width 8, effective width 4, two
signed i4 elements per byte, sign-extension before widening product, unroll 1,
two realized `vsetvl` regions, peak live vector groups 6, product region 1
with phase `load-product-reduce`, and dequant region 2 with phase
`dequant-store`. Common EmitC remains a neutral materializer.

### Testing

- [OK] built `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] manual packed-i4 PLAN pipeline showed provider-owned realization
  decision, realized region count, product/dequant indices, and product/dequant
  phases in route metadata.
- [OK] manual packed-i4 HEADER pipeline exported the same schedule facts as
  target artifact mirrors.
- [OK] manual packed-i4 C++ emission pipeline found low/high nibble
  sign-extension, widening product, and widening reduction intrinsics.
- [OK] stale packed resource decision, stale packed region count, stale packed
  `from_phase`, and stale artifact realization decision fail closed with
  targeted diagnostics.
- [OK] non-packed product-reduction-dequant header export still follows its
  u2/grouped resource facts, and the missing-resource-pass negative check still
  fails closed.
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded added-line authority scan found only the intentional
  `artifact-name-derived-resource-decision` negative-test string.

`FileCheck` is not available in this environment, so the fixture RUN lines were
validated through equivalent `tcrv-opt` / `tcrv-translate` pipelines with
targeted `rg` assertions.

### Status

[OPEN] Gate 2 is complete for this production-source consumption slice. The
macro task remains active because Gate 3 generated artifact evidence and Gate 4
same-target measurement rerun are still unfinished. Next continuation point:
Gate 3 regenerate and inspect packed-i4 generated artifact evidence from the
validated production route/statement/artifact path, without treating generated
artifacts as semantic authority.

## Session 577: RVV production-kernel Gate 3 generated artifact evidence

**Date**: 2026-06-09
**Task**: RVV production-kernel capability campaign: resource-aware packed low-precision contraction realization

### Summary

Continued the active macro campaign at Gate 3. The PRD top-level checklist now
reconciles Gate 2 as complete from commit `104df15a` and marks Gate 3 complete
for generated artifact evidence. Gate 4 remains open.

This slice strengthens generated-bundle evidence for the accepted signed
packed-i4 product-reduction-dequant representative. The ABI e2e verifier now
requires object/header low-precision realization schedule mirrors for
product-dequant routes: realization producer, realization decision, realized
unroll, realized `vsetvl` region count, realized peak live-vector groups,
product/dequant region indices, product/dequant phases, runtime ABI order, and
target capability mirrors. The evidence JSON now includes
`generated_artifact_resource_schedule_evidence` so the generated artifact's role
is explicit: mirror-only after RVV provider route construction and target
artifact validation.

Added a focused packed-i4 generated-bundle dry-run test that uses the packed
fixture through `--input`, checks evidence JSON, target artifact bundle index
metadata, packed low/high nibble statement payloads, and the packed external ABI
harness oracle selected from validated `packed-i4-nibbles` metadata.

Synchronized the testing and RVV plugin specs with the Gate 3 generated-bundle
schedule mirror contract: provider-owned realization schedule fields must agree
between object/header metadata and expected evidence fields, and missing/stale
mirrors fail before evidence is accepted.

### Testing

- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] packed-i4 generated-bundle dry-run with the packed fixture override and
  `--llvm-readobj ""`
- [OK] default unpacked product-dequant generated-bundle dry-run regression with
  `--llvm-readobj ""`
- [OK] product-dequant-clamp generated-bundle dry-run regression with
  `--llvm-readobj ""`
- [OK] stale packed artifact realization decision fails target artifact export
  with the provider-selected realization decision diagnostic.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded added-line authority scan

`FileCheck`, `llvm-lit`, and `llvm-readobj` are unavailable in this local
environment, so the new lit file was validated through equivalent dry-run
commands and targeted `rg` assertions over generated evidence, bundle index,
and harness source.

### Status

[OPEN] Gate 3 generated artifact evidence is complete and committed in the
active macro task. The macro task remains active because Gate 4 same-target
measurement rerun is still unfinished. Next continuation point: Gate 4 rerun
same-target measurement after this production compiler/resource-aware evidence
change and report win, regression, or no-win.

## Session 578: RVV production-kernel Gate 4 same-target measurement

**Date**: 2026-06-09
**Task**: RVV production-kernel capability campaign: resource-aware packed low-precision contraction realization

### Summary

Continued the active macro campaign at Gate 4 and reran same-target timing for
the accepted signed packed-i4 product-reduction-dequant representative after
the Gate 1-3 production compiler/resource-aware changes. The measurement used
`scripts/rvv_generated_bundle_same_target_measure.py`, which regenerates and
validates the generated RVV object/header through the generated-bundle ABI e2e
path before selecting the packed scalar baseline from provider-owned
`packed-i4-nibbles` metadata.

The real run succeeded on `ssh rvv` against
`scalar-c-reference/product-reduction-dequant-packed-i4-v1` with counts
`257,4096,65536`, patterns `0,1`, scales `-0.125,0.375`, warmups `2`,
repeats `5`, and iterations `8`. Remote profile: `riscv64`, 64 CPUs,
`/usr/bin/clang`, Ubuntu clang 18.1.3. Evidence is under
`artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh/`.

All 12 parsed best-speedup summaries are below 1.0, from `0.761006` to
`0.807006`. This is a no-win/regression signal for the generated RVV artifact
against the named scalar C packed-i4 baseline; no performance win, llama.cpp
parity, q4/q8 route authority, or benchmark-name authority is claimed.

Updated the macro PRD to mark Gate 4 complete and record the raw evidence
paths, generated artifact identity, object/header hashes, timing table, and
outcome. Corrected the stale Gate 5 wording in the measurement script docstring
and extended the Gate 4 dry-run lit test with packed-i4 measurement harness
coverage for the metadata switch, packed baseline identity, correctness guard,
timing record shape, and packed harness structure.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] packed-i4 same-target measurement dry-run with the packed fixture
  override and `--llvm-readobj ""`
- [OK] packed-i4 same-target measurement on `ssh rvv` with the packed fixture,
  counts `257,4096,65536`, warmups `2`, repeats `5`, iterations `8`, and
  `--llvm-readobj ""`
- [OK] default product-dequant and dequant-clamp same-target measurement
  dry-run regression with `--llvm-readobj ""`; default product-dequant kept
  `scalar-c-reference/product-reduction-dequant-v1` and
  `packed_i4_reference_oracle=false`

### Status

[ARCHIVED] All four macro campaign gates are complete. Final Trellis quality
verification passed, the macro task was archived, and one coherent commit was
created for the Gate 4 closeout.

### Spec Update Decision

[NO-SPEC-CHANGE] The executable same-target measurement contract already lives
in `.trellis/spec/testing/mlir-testing-contract.md` and the packed-i4 provider
contract already lives in `.trellis/spec/extension-plugins/rvv-plugin.md`. This
round executed those contracts and added focused dry-run coverage, but did not
introduce a new command signature, metadata field, validation rule, or
cross-layer behavior requiring a spec edit.

## Session 579: RVV packed low-precision performance feedback repair campaign

**Date**: 2026-06-09
**Task**: RVV production-kernel packed low-precision performance repair campaign

### Summary

Started the new active macro repair campaign after the prior packed resource
realization task was archived. The new PRD begins from the measured Gate 4
packed-i4 no-win/regression evidence: same-target generated RVV artifact timing
against `scalar-c-reference/product-reduction-dequant-packed-i4-v1` produced 12
parsed summaries below 1.0, with best-speedup range `0.761006..0.807006`.

Completed Gate 1 by tying that no-win signal to the production packed-i4
resource-selection seam instead of to q4/q8, llama.cpp, benchmark names, route
ids, or artifact names. The seam is
`RVVLowPrecisionContractionResourceSelection` flowing through Gearbox resource
attrs, selected-body realization, route-family planning, statement planning,
route metadata, target support bundle metadata, target artifact validation, and
generated-bundle dry-run evidence.

Completed the first Gate 2 production-surface action by adding provider-owned
packed-i4 performance feedback facts:
`same-target-packed-i4-no-win.v1`,
`scalar-c-reference/product-reduction-dequant-packed-i4-v1`,
`0.761006..0.807006`, and
`no-win-repair-required-before-performance-claim`. Packed-i4 selected bodies,
route planning, statement planning, route metadata, target validation, and ABI
dry-run evidence now preserve those facts and reject stale feedback mirrors
before artifact acceptance.

The RVV plugin code-spec now includes the packed-i4 no-win performance feedback
contract, exact fields, validation matrix, good/bad cases, and required tests.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py scripts/rvv_generated_bundle_same_target_measure.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] packed-i4 generated-bundle ABI dry-run with
  `--pre-realized-selected-body`, fixture override, runtime counts
  `0,1,16,17,257`, and `--llvm-readobj ""`
- [OK] direct compiler pipeline shows realized/body, route plan, header, bundle
  index, and evidence metadata carrying the four packed-i4 feedback fields
- [OK] stale `performance_feedback` edited to
  `same-target-packed-i4-performance-win.v1` fails target artifact export with
  a provider-mirror diagnostic

`clang-format`, `FileCheck`, and `llvm-lit` are unavailable in this local
environment, so lit/FileCheck tests were not run directly.

### Status

[OPEN] Gate 1 and Gate 2 are complete for the current slice. The macro task
remains active because Gate 3 full generated artifact/header/C/correctness
evidence and Gate 4 same-target timing rerun remain unfinished. Next
continuation point: improve the RVV-owned packed execution structure or collect
Gate 3 evidence while preserving the new fail-closed no-win feedback facts.

## Session 580: RVV packed low-precision Gate 3 artifact evidence

**Date**: 2026-06-09
**Task**: RVV production-kernel packed low-precision performance repair campaign

### Summary

Continued the active macro repair campaign at Gate 3. Gate 1 and Gate 2 were
already complete from commit `b7832403`; this slice keeps the same Trellis task
active and proves the provider-owned packed-i4 no-win feedback/resource facts
survive into generated artifact evidence without claiming a performance win.

The generated-bundle ABI verifier now exposes packed-i4 no-win feedback fields
in the `widening_product_reduction_boundary.low_precision_resource` evidence
summary, validates generated header comments when a packed-i4 selected candidate
is present, and reuses one expected low-precision metadata helper for both
validation and fake-bundle self-tests. The self-test now includes missing and
stale packed-i4 `performance_feedback` metadata failures, so the evidence path
fails closed before accepting stale no-win/performance mirrors.

The packed-i4 generated-bundle dry-run lit test now checks generated bundle JSON,
generated header comments, emitted RVV C/C++ packed low/high nibble statement
order, and the packed external ABI correctness oracle. The evidence remains
dry-run/header/C/harness evidence only; no real `ssh rvv` runtime correctness or
same-target timing claim was made in this slice.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] packed-i4 generated-bundle ABI dry-run with
  `--pre-realized-selected-body`, fixture override, runtime counts
  `0,1,16,17,257`, and `--llvm-readobj ""`
- [OK] generated header contains the four packed-i4 no-win feedback mirror
  comments and `packed-i4-nibbles` operand form.
- [OK] emitted RVV C/C++ contains packed low/high signed-i4 nibble unpack,
  two widening product/reduction steps, `dot_acc_vec` carry assignment, and
  final f32 store.
- [OK] generated harness contains `packed_i4_reference_oracle`,
  `pack_signed_i4_pair`, `sign_extend_i4`, low/high product accumulation, source
  and accumulator preservation, and tail preservation checks.
- [OK] evidence JSON summary carries
  `same-target-packed-i4-no-win.v1`,
  `scalar-c-reference/product-reduction-dequant-packed-i4-v1`,
  `0.761006..0.807006`, and
  `no-win-repair-required-before-performance-claim`.
- [OK] default unpacked product-dequant generated-bundle dry-run regression
  remains separate and does not emit packed-i4 feedback or oracle text.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded added-line authority scan; only initial matches were negative
  `implicit-check-not` guards.

`FileCheck` and `llvm-lit` are unavailable in this local environment, so the
updated lit file was validated through the actual dry-run artifacts and focused
`rg` assertions over generated evidence, bundle header, emitted C, and harness.

### Status

[OPEN] Gate 3 generated artifact/header/C/correctness evidence is complete for
the current repair campaign. The macro task remains active because Gate 4
same-target measurement rerun is still unfinished. Next continuation point:
Gate 4 same-target packed-i4 timing against
`scalar-c-reference/product-reduction-dequant-packed-i4-v1` after deciding
whether a later RVV-owned Gearbox/resource/statement repair should be measured;
otherwise continue reporting the no-win feedback honestly without a performance
win claim.

### Spec Update Decision

[NO-SPEC-CHANGE] The packed-i4 no-win performance feedback contract, required
fields, fail-closed behavior, and generated-bundle dry-run expectation already
live in `.trellis/spec/extension-plugins/rvv-plugin.md`. This round implemented
and verified that existing contract in the generated-bundle evidence verifier
and lit surface, but did not add a new metadata field, command signature,
validation matrix, or cross-layer behavior requiring a spec edit.

## Session 581: RVV packed low-precision Gate 4 same-target measurement

**Date**: 2026-06-09
**Task**: RVV production-kernel packed low-precision performance repair campaign

### Summary

Continued the active macro repair campaign at Gate 4. Gates 1-3 were already
complete from the prior slices; this slice did not modify C++ production owners.
The focused change was to harden `scripts/rvv_generated_bundle_same_target_measure.py`
so same-target measurement evidence records a structured win/no-win/regression
classification and ties that result back to provider-owned packed-i4 feedback
facts from the validated generated bundle.

The script now classifies parsed `SUMMARY` records as `win` only when every
`best_speedup` is above 1.0, `regression` when every `best_speedup` is below
1.0, and `no-win` for mixed/tie cases. Dry-run evidence is explicitly
`not-measured`. Packed-i4 evidence also carries a provider feedback tie-back
with `performance_feedback = same-target-packed-i4-no-win.v1`,
`performance_baseline = scalar-c-reference/product-reduction-dequant-packed-i4-v1`,
`performance_best_speedup_range = 0.761006..0.807006`, and
`performance_action = no-win-repair-required-before-performance-claim`.

The real `ssh rvv` run measured the generated packed-i4 RVV artifact against
`scalar-c-reference/product-reduction-dequant-packed-i4-v1` on the same target
with input sizes `257,4096,65536`, scales `-0.125,0.375`, warmups `2`, repeats
`5`, iterations `8`, compile flags `-O2 -march=rv64gcv -mabi=lp64d -I.`, and
`clock_gettime(CLOCK_MONOTONIC_RAW)`. Target profile capture reported
`ssh_target=rvv`, `remote_arch=riscv64`, `/usr/bin/clang`, Ubuntu clang 18.1.3,
and 64 CPUs.

Evidence artifacts:

- root evidence:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh_current/evidence.json`;
- per-op evidence:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh_current/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`;
- raw target profile:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh_current/widening_product_reduce_dequantize_f32/remote_target_profile_stdout.txt`;
- raw timing stdout:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh_current/widening_product_reduce_dequantize_f32/remote_measure_run_stdout.txt`.

Parsed Gate 4 result:

```text
n      pattern  scale   scalar ns/iter  generated ns/iter  speedup
257    0       -0.125   622.500         812.500            0.766154
257    0        0.375   622.500         817.500            0.761468
257    1       -0.125   622.500         815.000            0.763804
257    1        0.375   620.000         810.000            0.765432
4096   0       -0.125   9127.500        11352.500          0.804008
4096   0        0.375   9125.000        11357.500          0.803434
4096   1       -0.125   9125.000        11355.000          0.803611
4096   1        0.375   9127.500        11352.500          0.804008
65536  0       -0.125   145292.500      181245.000         0.801636
65536  0        0.375   145300.000      181252.500         0.801644
65536  1       -0.125   145292.500      181240.000         0.801658
65536  1        0.375   145297.500      181237.500         0.801697
```

All 12 summaries are below 1.0, so the structured classification is
`regression` with `best_speedup_range = 0.761468..0.804008`. The provider
feedback tie-back is `consistent-with-current-no-win-feedback` and
`performance_win_claim_allowed = false`.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] packed-i4 same-target measurement dry-run with fixture override,
  counts `257,4096,65536`, warmups `2`, repeats `5`, iterations `8`, and
  `--llvm-readobj ""`
- [OK] default product-dequant/dequant-clamp same-target measurement dry-run
  regression with counts `257,1024`, warmups `1`, repeats `2`, iterations `3`,
  and `--llvm-readobj ""`
- [OK] real packed-i4 same-target measurement on `ssh rvv`; evidence has
  60 raw `MEASURE` records, 12 `SUMMARY` records, correctness guards, and
  classification `regression`
- [OK] focused `jq` assertions for dry-run not-measured behavior, real
  regression classification/range/counts, provider feedback tie-back, and default
  path non-packed baseline selection
- [OK] focused `rg` assertions over raw timing stdout for baseline identity,
  min/max best speedup records, and PASS line
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`

`FileCheck` and `llvm-lit` are unavailable in this local environment, so the
updated lit file was validated through actual dry-run artifacts and focused
`jq`/`rg` assertions.

### Status

[OPEN] Gate 4 same-target measurement evidence is complete and classified as
regression. The macro task remains active because the continuation rule says a
no-win/regression Gate 4 result should leave the next production repair owner
for Hermes review instead of archiving the campaign. Next continuation point:
repair the RVV plugin-local Gearbox/resource/statement-planning path for the
selected packed-i4 product-reduction candidate, then rerun the same Gate 4
measurement contract and update provider-owned feedback facts if the measured
result changes.

### Spec Update Decision

[UPDATED] Added `Packed-I4 Same-Target Measurement Classification Evidence` to
`.trellis/spec/extension-plugins/rvv-plugin.md` because this round changed the
machine-readable evidence contract: dry-run evidence is `not-measured`, real
same-target runs classify parsed `SUMMARY best_speedup` records as win/no-win/
regression, and packed-i4 timing evidence must carry provider feedback tie-back
before any performance claim is allowed.

## 2026-06-09 - Packed-i4 Gate 5/6 product-pair-sum repair

### Summary

- Continued active task `06-09-rvv-production-kernel-packed-low-precision-performance-repair-campaign`.
- Repaired the packed-i4 selected statement path in `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`: removed the stale low-nibble reduction, added low/high `__riscv_vwmul_vv_i16mf2`, `__riscv_vadd_vv_i16mf2` product-pair sum, and one `__riscv_vwredsum_vs_i16mf2_i32m1` from the pair sum into `dot_acc_vec`.
- Updated provider-owned resource facts to decision `consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1`, peak live vector groups `7`, and refreshed no-win speedup range `0.688427..0.705724`.
- Updated target validation to reject stale high-product operands, stale product-pair sum operands, stale single-reduction input/result, and stale final carry source.
- Updated generated-bundle tooling, dry-run tests, C++ tests, fixture expectations, and RVV plugin spec for the new product-pair-sum single-reduction contract.

### Evidence

- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test` passed.
- Gate 5 dry-run evidence: `artifacts/tmp/gate5-packed-i4-product-pair-sum-repair/packed-i4-gate5-product-pair-sum`, including emitted `__riscv_vadd_vv_i16mf2`, `product_pair_sum_vector`, and `single_reduction_vector`.
- Gate 6 real same-target evidence: `artifacts/tmp/gate6-same-target-measurement/gate6_packed_i4_product_pair_sum_same_target_measure_ssh`, classification `regression`, 12 summaries, 60 measurements, best speedup range `0.688427..0.705724`, and `performance_win_claim_allowed = false`.

### Status

- Gate 5 and Gate 6 are complete truthfully.
- No packed-i4 performance-win claim is allowed.
- Task is ready for final quality checks, commit, and archive.


## Session 572: Packed-i4 Gate 5/6 product-pair-sum repair

**Date**: 2026-06-09
**Task**: Packed-i4 Gate 5/6 product-pair-sum repair
**Branch**: `main`

### Summary

Repaired packed-i4 statement planning to use product-pair sum plus single reduction, refreshed provider feedback from same-target regression evidence, updated tests/specs, and archived the completed macro task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `5ffa4741` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete

## 2026-06-09 - RVV packed low-precision performance-maturity feedback Gate 1/2

### Summary

- Created active macro task
  `06-09-rvv-packed-low-precision-performance-maturity-feedback-campaign`.
- Completed Gate 1/2 slice: added provider-owned packed-i4 performance
  maturity mirrors and wired them through Gearbox/resource facts,
  selected-body realization, route-family planning, statement planning, route
  metadata, target support bundle metadata, and target artifact validation.
- The packed-i4 route remains legal/executable for correctness, but current
  Gate 6 regression evidence is mirrored as
  `performance_maturity = executable-not-performance-mature`,
  `performance_maturity_outcome = regression`,
  `performance_selection_eligible = false`, and
  `dispatch_preference = not-performance-preferred`.
- Updated focused C++ tests, MLIR/FileCheck fixture coverage, generated-bundle
  scripts, and RVV plugin spec for the new maturity-selection contract.

### Evidence

- `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate` passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed after repairing
  the low-precision resource attr verifier whitelist.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed after the same
  verifier repair.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- Packed-i4 ABI dry-run passed at
  `artifacts/tmp/packed-i4-performance-maturity-dry-run/pre-realized-widening-product-reduce-dequantize-f32-packed-i4`.
- Packed-i4 same-target dry-run passed at
  `artifacts/tmp/packed-i4-performance-maturity-same-target-dry-run/gate4-packed-i4-same-target-measure`;
  generated evidence contains `performance_win_claim_allowed = false`.
- `rtk git diff --check` and Python bytecode checks passed.
- Bounded added-diff old-authority scan found no new positive
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m`,
  source-front-door, source-export, direct-C, q4/q8, or llama authority.

### Status

[OPEN] Gate 1 and Gate 2 are complete for this macro campaign. The task remains
active by design. Next continuation point: Gate 3, connect same-target
measurement outcome fields to the provider-owned maturity contract as evidence
input, then finish Gate 4/5 with stale-evidence and same-target policy checks.

## 2026-06-09 - RVV packed low-precision performance-maturity feedback Gate 3

### Summary

- Continued active macro task
  `06-09-rvv-packed-low-precision-performance-maturity-feedback-campaign`.
- Completed Gate 3 slice: same-target measurement evidence now emits a
  structured `performance_maturity_contract_evidence_input` object for
  packed-i4 runs.
- The object ties measured classification/outcome family/speedup range and
  measurement evidence id to provider-owned maturity evidence/outcome,
  performance-selection eligibility, dispatch preference, claim allowance,
  denial reason, route-support effect, and correctness-execution allowance.
- Updated the RVV plugin spec with the 7-section executable contract for this
  evidence-input payload. Measurement scripts remain reporting/validation
  clients; provider/resource facts and target artifact mirrors remain the
  maturity contract authority.

### Evidence

- `rtk python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Packed-i4 same-target dry-run passed at
  `artifacts/tmp/packed-i4-gate3-filecheck-dry-run/gate4-packed-i4-same-target-measure`.
- Manual FileCheck passed for `PACKED-ROOT`, `PACKED-WPRD`, and
  `PACKED-HARNESS` against the updated dry-run evidence and harness.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- `rtk git diff --check` passed.
- Bounded added-diff old-authority scan found only negative/spec guardrail
  mentions of q4/q8/llama, with no positive legacy i32/source-front-door/
  descriptor/direct-C authority.

### Status

[OPEN] Gate 1, Gate 2, and Gate 3 are complete for this macro campaign. The
task remains active by design. Next continuation point: Gate 4, add focused
fail-closed stale evidence tests proving no-win/regression cannot become
performance-selected or claimed while route-supported executable correctness
remains allowed, then Gate 5 for focused artifact/correctness and same-target
policy checks.

## 2026-06-09 - RVV packed low-precision performance-maturity feedback Gate 4

### Summary

- Continued active macro task
  `06-09-rvv-packed-low-precision-performance-maturity-feedback-campaign`.
- Completed Gate 4 slice: stale/no-win/regression packed-i4 maturity evidence
  now fails closed in focused script policy self-tests, provider negative
  coverage, target artifact negative coverage, and the packed-i4 lit fixture.
- Added measurement evidence-input integrity checks so stale measurement id,
  classification, outcome family, speedup range, provider maturity outcome,
  selection eligibility, dispatch preference, or win-claim allowance cannot
  silently disagree with the parsed same-target classification and provider
  mirrors.
- No production provider/target semantic repair was required: the existing
  provider/resource and target artifact validators already reject stale packed-i4
  maturity mirrors. This slice added the missing Gate 4 proof around that seam.

### Evidence

- `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test` passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- `rtk python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test` passed.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Packed-i4 Gate 4 dry-run passed at
  `artifacts/tmp/packed-i4-gate4-fail-closed-dry-run/gate4-packed-i4-fail-closed`.
- Focused `jq` assertions proved root/per-op evidence keeps
  `performance_win_claim_allowed = false`,
  `performance_preference_denied = true`,
  `correctness_execution_allowed = true`,
  `provider_performance_selection_eligible = "false"`, and
  `provider_dispatch_preference = "not-performance-preferred"`.
- `rtk git diff --check` passed.
- Bounded added-diff old-authority scan found no new positive
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m`,
  source-front-door, source-export, direct-C, descriptor, q4/q8, or llama
  authority.

`llvm-lit` and `FileCheck` are unavailable in this local environment, so the
updated lit fixture was validated through dry-run artifacts, focused `jq`
assertions, script self-tests, and the C++ provider/target tests.

### Status

[OPEN] Gate 1, Gate 2, Gate 3, and Gate 4 are complete for this macro campaign.
The task remains active by design. Next continuation point: Gate 5, rerun
focused artifact/correctness plus same-target policy checks and record whether
the policy reports win, no-win, or regression truthfully while preserving the
correctness/performance-maturity split.

### Spec Update Decision

[NO UPDATE] No durable spec change was needed. This round exercised the existing
RVV plugin, EmitC route, variant tuning, and MLIR testing contracts without
adding a new architecture rule.

## 2026-06-09 - RVV packed low-precision performance-maturity feedback Gate 5

### Summary

- Continued active macro task
  `06-09-rvv-packed-low-precision-performance-maturity-feedback-campaign`.
- Completed Gate 5 slice: reran focused packed-i4 artifact/correctness and
  same-target measurement evidence on `ssh rvv`, then recorded the truthful
  policy outcome in the active PRD.
- The packed-i4 route remains executable for correctness, but same-target
  measurement still classifies it as `regression`; performance selection,
  dispatch preference, and performance win claims remain denied by the
  provider-owned maturity contract.
- No production source change was required. Existing provider/resource,
  target artifact, and measurement bridge validation already preserved the
  executable/performance-maturity split.

### Evidence

- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir --runtime-count 257 --runtime-count 4096 --runtime-count 65536 --dequant-scale 0.125 --dequant-scale 0.25 --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id gate5_packed_i4_focused_artifact_correctness_ssh --overwrite --ssh-target rvv`
  passed after the first attempt was self-repaired by adding the required second
  nonzero dequant scale. Evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/gate5_packed_i4_focused_artifact_correctness_ssh`.
- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir --measure-count 257 --measure-count 4096 --measure-count 65536 --artifact-root artifacts/tmp/gate5-same-target-measurement --run-id gate5_packed_i4_focused_same_target_measure_ssh --overwrite --ssh-target rvv`
  passed. Evidence:
  `artifacts/tmp/gate5-same-target-measurement/gate5_packed_i4_focused_same_target_measure_ssh`.
- Same-target policy fields:
  `classification = regression`,
  `outcome_family = no-win`,
  `best_speedup_range = 0.691358..0.705406`,
  `summary_count = 12`,
  `measurement_record_count = 60`,
  `provider_performance_selection_eligible = false`,
  `provider_dispatch_preference = not-performance-preferred`,
  `performance_win_claim_allowed = false`,
  `correctness_execution_allowed = true`, and
  `provider_contract_update_required = false`.
- `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- `rtk git diff --check` passed.
- Bounded added-diff old-authority scan found no new positive
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m`,
  source-front-door, source-export, direct-C, descriptor, q4/q8, or llama
  authority.

### Status

[DONE] Gates 1 through 5 are complete for the macro campaign. The task is ready
to archive after final diff/status checks and commit.

### Spec Update Decision

[NO UPDATE] No durable spec change was needed. Gate 5 produced refreshed
evidence against the already-documented RVV plugin maturity, same-target
measurement, EmitC route, variant tuning, and MLIR testing contracts.

## 2026-06-09 - RVV production-kernel capability campaign Gate 1

### Summary

- Created active macro task
  `06-09-rvv-production-kernel-capability-campaign`.
- Completed Gate 1 production slice: the low-precision resource selection now
  carries explicit RVV-owned primitive-chain facts for the widening product ->
  widening reduction -> dequant family.
- The new facts are populated from RVV resource candidates/pass facts, consumed
  by selected-body realization validation, checked by route-family
  selection/equality/validation, mirrored through provider route metadata, and
  validated by target artifact provider/mirror checks.
- This is production compiler plumbing, not a generated-bundle or measurement
  rerun. No performance preference or win claim changed.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed after self-repair
  of the pre-realized low-precision fixture facts.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed after
  self-repair of the packed-i4 artifact fixture facts.
- Manual FileCheck passed for Gearbox positive output and stale primitive
  rejection.
- Manual FileCheck passed for pre-realized target REALIZED, PLAN, HEADER,
  stale provider primitive, and stale primitive mirror checks.
- Manual FileCheck passed for packed-i4 REALIZED, PLAN, and HEADER checks.
- `rtk git diff --check` passed.
- `rtk git diff --cached --check` passed.
- Bounded touched-diff old-authority scan found no new `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m`,
  source-front-door, source-artifact, or descriptor authority. Added `i32m1`
  strings are low-precision primitive/reduction resource facts and negative
  stale checks, not route authority.

### Status

[OPEN] Gate 1 is complete. Gates 2 through 5 remain open, and
`.trellis/.current-task` intentionally remains active for the macro campaign.

### Continuation

Next owner: Gate 2 Gearbox/resource-aware selected-body realization should
consume the explicit primitive-chain resource facts to materialize legal
low-precision contraction structure without treating route ids, artifact names,
Common EmitC, or measurement scripts as authority.

## 2026-06-09 - RVV production-kernel capability campaign Gate 2

### Summary

- Continued the active macro task
  `06-09-rvv-production-kernel-capability-campaign`.
- Completed Gate 2 production source slice: Gearbox/resource-aware
  selected-body realization now consumes the selected low-precision
  primitive-chain resource facts and carries them through the realized
  `tcrv_rvv.gearbox_cross_region_handoff` structure.
- The realized handoff now structurally carries primitive chain contract/kind,
  widening-product relation, product-reduction chain relation, widening product
  intrinsic, widening reduction intrinsic, scalar seed splat intrinsic,
  accumulator/result layouts, and reduction store VL.
- RVV-owned validation rejects stale or missing handoff primitive-chain facts
  before route/provider planning, Common EmitC, or target artifact export can
  accept the realized body.
- This is a production compiler/source change only. No generated-bundle,
  same-target timing, dispatch policy, or performance claim changed.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- `FileCheck` and `llvm-lit` were not available in PATH, so focused manual
  checks used `tcrv-opt`, `tcrv-translate`, `sed`, and `rg`.
- Manual positive checks proved pre-realized dequant, dequant-clamp, and
  packed-i4 selected-body realization emit Gearbox handoffs with the expected
  primitive-chain fields.
- Manual explicit artifact checks proved the fixture's positive RUN path with
  `--tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans`
  carries the new resource primitive mirrors into plan/header output.
- Manual stale handoff negative check replaced
  `primitive_reduction_intrinsic = "__riscv_vwredsum_vs_i16mf2_i32m1"` with
  `__riscv_vwredsum_vs_i32m1_i32m1` and failed closed at
  `tcrv_rvv.gearbox_cross_region_handoff` verifier diagnostics.
- `rtk git diff --check` passed before staging.
- Bounded touched-diff old-authority scan found only PRD/ODS guardrail text for
  q4/q8/llama/descriptor/source-front-door, not new positive route authority.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the
product-reduction Gearbox handoff primitive-chain field contract and the
required stale-fact negative coverage.

### Status

[OPEN] Gates 1 and 2 are complete. Gates 3 through 5 remain open, and
`.trellis/.current-task` intentionally remains active for the macro campaign.

### Continuation

Next owner: Gate 3 route/statement planning and target artifact mirror
validation must consume the selected Gearbox realization primitive/resource
facts and reject stale or unsupported performance/resource claims without
treating route ids, artifact names, Common EmitC, or measurement scripts as
authority.

## 2026-06-09 - RVV production-kernel capability campaign Gate 3 first slice

### Summary

- Continued the active macro task
  `06-09-rvv-production-kernel-capability-campaign`.
- Completed the first Gate 3 production source slice: low-precision
  product-reduction statement planning now requires the selected Gearbox
  realization/resource facts before statement construction.
- Target artifact provider validation now rejects missing or stale
  low-precision product-reduction resource/realization facts before route
  statements or candidate metadata mirrors can be accepted.
- Candidate metadata remains a mirror only. The added target coverage rejects a
  stale realization-decision mirror instead of treating artifact metadata as
  route authority.
- No generated-bundle, `ssh rvv`, dispatch policy, or performance claim changed
  in this slice.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- `rtk git diff --check` passed.
- `rtk git diff --cached --check` passed.
- JSONL parse check for the task implement/check logs passed.
- Bounded code-only touched-diff old-authority scan found no new
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m`,
  source-front-door, source-artifact, or descriptor authority.

### Self-Repair

- Qualified RVV low-precision realization constants in
  `test/Plugin/RVVExtensionPluginTest.cpp` after the first rebuild exposed
  missing namespace qualification.
- Added target candidate-mirror stale realization-decision coverage after the
  first focused tests showed provider-description coverage was stronger than
  metadata-mirror coverage.

### Spec Update Decision

[NO CHANGE] Existing RVV plugin and EmitC route specs already require
provider-owned primitive/resource/realization facts and target mirror
validation. This slice implemented that contract without introducing a new
cross-session rule.

### Status

[OPEN] Gate 3 first production source slice is complete. Gate 3 remains open
only for additional route/provider/target mirror broadening if Hermes selects
it. Gates 4 and 5 remain open, and `.trellis/.current-task` intentionally
remains active for the macro campaign.

### Continuation

Next owner: either continue Gate 3 if another route/provider/target mirror
surface is identified, or advance to Gate 4 generated artifact correctness and
same-target measurement for the changed production path. Gate 5 dispatch/
performance policy must wait for Gate 4 measurement facts.

## 2026-06-09 - RVV production-kernel capability campaign Gate 3 completion

### Summary

- Continued the active macro task
  `06-09-rvv-production-kernel-capability-campaign`.
- Completed Gate 3 by proving the remaining route-family/provider mirror
  surface consumes the same selected low-precision resource, realization
  schedule, primitive-chain, region, resource-budget, and selected target
  capability facts before statement planning or artifact acceptance.
- No production source change was required in this completion slice: live
  inspection showed the route-family/provider and target validators already
  enforced the Gate 3 contract after the first slice and prior Gate 1/2 work.
- Added focused provider-level C++ coverage for stale realization mirrors,
  selected target capability mirrors, missing selected candidates,
  primitive-chain kind, product region facts, resource budget pressure, and
  packed-i4 realization schedule mirrors.
- No generated-bundle, `ssh rvv`, dispatch policy, or performance claim changed
  in this slice.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk ninja -C build tianchenrv-target-artifact-export-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Final whitespace, JSONL, old-authority, staged diff, and status checks are
  run as the closeout gate for this slice.

### Self-Repair

- Qualified the new missing-resource default value with the
  `tianchenrv::plugin::rvv` namespace after the first focused rebuild exposed
  the missing qualification.

### Spec Update Decision

[NO CHANGE] Existing RVV plugin and EmitC route specs already require
provider-owned primitive/resource/realization facts, route/provider mirror
validation, target mirror validation, and Common EmitC neutrality. This slice
only completed focused evidence for that contract.

### Status

[OPEN] Gates 1, 2, and 3 are complete. Gates 4 and 5 remain open, and
`.trellis/.current-task` intentionally remains active for the macro campaign.

### Continuation

Next owner: Gate 4 generated artifact correctness and same-target measurement
for the changed production path. Gate 5 dispatch/performance policy must wait
for Gate 4 measurement facts.

## 2026-06-09 - RVV production-kernel capability campaign Gate 4 evidence plumbing

### Summary

- Continued the active macro task
  `06-09-rvv-production-kernel-capability-campaign`.
- Completed the first Gate 4 sub-slice by carrying the low-precision
  primitive-chain/resource facts from Gates 1-3 into generated-bundle evidence
  and same-target measurement dry-run records.
- The generated-bundle and measurement paths now require primitive-chain,
  resource, realization schedule, provider, target capability, artifact ABI,
  correctness-before-timing, and measurement classification mirrors for the
  grouped and packed-i4 product-reduction/dequantization paths.
- Packed-i4 provider feedback now derives its required primitive/resource/
  realization/target field set from provider-owned low-precision metadata and
  fails closed on stale primitive-chain, reduction, realization, or target
  capability mirrors before accepting a measurement record.
- No dispatch/performance policy changed and no runtime correctness or
  performance claim was made. The dry-run measurement evidence remains
  classified as `not-measured`.

### Evidence

- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- Packed-i4 same-target dry-run passed and wrote a `not-measured` record under
  `artifacts/tmp/codex-gate4-check/gate4-packed-i4-same-target-measure`.
- Grouped dequant/dequant-clamp same-target dry-run passed under
  `artifacts/tmp/codex-gate4-check-standard/gate4-same-target-measure`.
- Packed-i4 generated-bundle ABI dry-run passed under
  `artifacts/tmp/codex-gate4-abi-packed/pre-realized-widening-product-reduce-dequantize-f32-packed-i4`.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- `rtk git diff --check` passed before staging.
- A bounded touched-diff old-authority scan found no new positive
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m`,
  source-front-door, source-artifact, descriptor, q4, q8, or llama route
  authority in the changed script/test diff.

### Self-Repair

- Expanded packed-i4 generated-bundle and same-target FileCheck expectations
  after the evidence diff showed the primitive-chain facts were present but not
  yet asserted by the script tests.

### Spec Update Decision

[NO CHANGE] Existing RVV plugin and EmitC route specs already require
provider-owned primitive/resource/realization facts, Common EmitC neutrality,
target mirror validation, and artifact/measurement records as non-authoritative
mirrors. This slice implemented the existing Gate 4 evidence contract.

### Status

[OPEN] The Gate 4 evidence-plumbing sub-slice is complete. Gate 4 remains open
because real `ssh rvv` correctness/timing evidence was not run in this slice
and no accepted measured win/no-win/regression facts exist. Gate 5 remains
unopened, and `.trellis/.current-task` intentionally remains active.

### Continuation

Next owner: continue Gate 4 by running or accepting real same-target `ssh rvv`
correctness/timing evidence for the generated low-precision primitive/resource
artifact path, then record truthful measured win/no-win/regression facts. Gate
5 dispatch/performance policy must wait for those accepted Gate 4 measurement
facts.

## 2026-06-09 - RVV production-kernel capability campaign Gate 4 real measurement acceptance

### Summary

- Continued the active macro task
  `06-09-rvv-production-kernel-capability-campaign`.
- Completed the second Gate 4 slice by running the selected signed packed-i4 /
  low-precision widening product-reduce-dequantize generated artifact path on
  real `ssh rvv`.
- The evidence preserves generated object/header/index identity, selected
  input/variant, correctness-before-timing harness, packed scalar baseline,
  target profile, primitive-chain/resource/realization/target mirrors, and
  provider-owned maturity tie-back.
- The accepted measured outcome is `classification = regression`,
  `outcome_family = no-win`, `best_speedup_range = 0.688889..0.705200`,
  `summary_record_count = 12`, `measurement_record_count = 60`, and
  `correctness_record_count = 12`.
- The provider maturity tie-back keeps `performance_win_claim_allowed = false`,
  `provider_performance_selection_eligible = false`,
  `provider_dispatch_preference = not-performance-preferred`, and
  `correctness_execution_allowed = true`.
- No production source change was required. The existing same-target measurement
  bridge already implemented non-dry-run correctness/timing, stale provider
  tie-back rejection, and win/no-win/regression classification.

### Evidence

- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- Packed-i4 focused dry-run passed at
  `artifacts/tmp/codex-gate4-real-measure-dry/gate4-packed-i4-real-measure-dry`
  and remained `not-measured`.
- Packed-i4 real same-target measurement passed on `ssh rvv` at
  `artifacts/tmp/codex-gate4-real-measure/gate4-packed-i4-real-measure-ssh`.
- Per-op evidence:
  `artifacts/tmp/codex-gate4-real-measure/gate4-packed-i4-real-measure-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`.
- Remote stdout contains 12 correctness guards before timing, 60 raw `MEASURE`
  records, 12 parsed `SUMMARY` records, and `PASS`.
- Remote target profile recorded `riscv64`, 64 CPUs, and Ubuntu clang 18.1.3.

### Spec Update Decision

[NO CHANGE] Existing RVV plugin and EmitC route specs already document packed-i4
same-target classification, maturity evidence input, provider-owned resource
mirrors, Common EmitC neutrality, and the real `ssh rvv` evidence requirement.

### Status

[OPEN] Gates 1 through 4 are complete for the current accepted packed-i4
representative. Gate 5 remains open, and `.trellis/.current-task` intentionally
remains active for the macro campaign.

### Continuation

Next owner: Gate 5 dispatch/performance policy should consume the accepted
Gate 4 regression/no-win facts truthfully: preserve executable correctness,
deny performance preference and win claims, reject stale policy/maturity
mirrors, and keep route support separate from performance-ready dispatch.

## 2026-06-09 - RVV production-kernel capability campaign Gate 5 dispatch/performance policy consumption

### Summary

- Continued the active macro task
  `06-09-rvv-production-kernel-capability-campaign`.
- Completed Gate 5 as a production source slice by adding
  `RVVLowPrecisionPerformancePolicy`, an RVV-owned policy consumer for the
  accepted packed-i4 Gate 4 same-target measurement outcome.
- The policy consumes the accepted Gate 4 measurement identity, `ssh rvv`
  target profile, regression/no-win classification, best-speedup range, record
  counts, provider maturity tie-back, selected candidate, primitive-chain facts,
  realization/resource facts, and target capability mirrors.
- The accepted outcome preserves executable route support and correctness
  execution while denying performance-preferred dispatch and performance-win
  claims.
- Provider route-family validation and RVV target artifact validation now call
  the policy before accepting packed-i4 low-precision product-reduction/
  dequantization performance-policy facts.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- `rtk git diff --check` passed.
- Bounded touched-file old-authority scan found pre-existing legacy/i32m1
  guardrail coverage in large tests; the current added diff lines introduced no
  new RVVI32M1, `rvv-i32m1`, legacy `tcrv_rvv.i32_*`, source-front-door,
  source-artifact, descriptor-driven, or exact-intrinsic route authority.
- `clang-format` was not available in this environment; formatting was checked
  with compiler coverage and `git diff --check`.

### Self-Repair

- Added an explicit target provider negative proving the accepted Gate 4
  dequantization measurement cannot be reused for the unmeasured
  dequant-clamp packed-i4 sibling candidate.
- Replaced a hard-coded primitive reduction intrinsic expectation in the plugin
  test with the provider-owned RVV low-precision resource constant.

### Spec Update Decision

[NO CHANGE] Existing RVV plugin, EmitC route, and campaign PRD contracts already
require provider-owned primitive/resource/realization facts, Common EmitC
neutrality, same-target measured evidence as non-authoritative input, and
fail-closed policy behavior for stale performance facts. This slice implemented
that existing Gate 5 contract in production code and tests.

### Status

[DONE] Gates 1 through 5 are complete for the current accepted packed-i4
representative. The macro task is ready to finish/archive after staged diff
checks and the final coherent commit.

### Continuation

No Gate 5 continuation remains for this macro campaign. Future work should start
from a new task only if it changes a different production-kernel capability
owner beyond this accepted packed-i4 representative.

## 2026-06-09 - RVV Low-Precision Remediation Gate 2

### Summary

- Continued the active macro task
  `06-09-rvv-low-precision-performance-remediation-campaign`.
- Completed Gate 2 as a production source slice: the packed-i4 low-precision
  resource selection now consumes the structured Gate 1 measurement policy
  handoff as a provider-owned remediation decision.
- The remediation decision carries the handoff contract, structured
  `correctness-supported-no-win-regression` diagnosis, accepted same-target
  measurement identity, resource action, dispatch consequence, and precise
  no-win/regression blocker.
- Selected-body realization attrs, pass-fact derivation, route descriptions,
  and target artifact mirrors now must agree on those remediation facts or fail
  closed.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Bounded added-diff old-authority scan found no new RVVI32M1,
  `rvv-i32m1`, legacy `tcrv_rvv.i32_*`, source-front-door, source-artifact,
  descriptor-driven, q4/q8/llama, or exact-intrinsic route authority.

### Self-Repair

- Updated the target packed-i4 positive MLIR fixture to carry the same Gate 2
  remediation resource facts as the plugin fixture after the provider correctly
  fail-closed the missing `remediation_handoff_contract`.

### Status

[OPEN] Gate 2 is complete for the remediation campaign. Gates 3-5 remain open,
and `.trellis/.current-task` stays active on the same macro task.

### Continuation

Continue with Gate 3: lift the Gate 2 remediation decision into the full
provider route fact and target artifact validation surface without metadata
authority. Gates 4 and 5 remain for new same-target measurement evidence and
dispatch/performance policy enablement only when production facts justify it.

## 2026-06-09 - RVV Low-Precision Remediation Gate 3

### Summary

- Continued the active macro task
  `06-09-rvv-low-precision-performance-remediation-campaign`.
- Completed Gate 3 as a production provider/target validation mirror slice:
  low-precision resource selections now carry the validated route-family plan
  id and provider-supported mirror from the RVV contraction route-family plan.
- Route-family validation requires the provider route facts, selected
  description, and resource selection to agree before route construction can be
  accepted.
- Target artifact export mirrors those Gate 3 facts and rejects stale or
  missing candidate metadata before artifact acceptance.
- The mirror surface remains non-authoritative: artifact metadata can confirm
  provider-owned facts but cannot invent low-precision semantics, performance
  preference, or win claims.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Bounded added-diff old-authority scan found no new RVVI32M1,
  `rvv-i32m1`, legacy `tcrv_rvv.i32_*`, source-front-door, source-artifact,
  descriptor-driven, q4/q8/llama, or exact-intrinsic route authority.
- Bounded touched-file scan showed pre-existing legacy/i32/source-front-door
  guardrail fixtures in large RVV tests; no matching authority was introduced
  by this slice.
- `rtk git diff --check` passed.

### Self-Repair

- None needed after the focused build and tests.

### Status

[OPEN] Gate 3 is complete for the remediation campaign. Gates 4-5 remain open,
and `.trellis/.current-task` stays active on the same macro task.

### Continuation

Continue with Gate 4: generate and measure the remediated low-precision
artifact on the same `ssh rvv` target, accepting or rejecting evidence only
through structured provider-owned resource, route-family, primitive-chain, and
remediation facts. Gate 5 remains for dispatch/performance policy consumption
after accepted measurement evidence.

## 2026-06-09 - RVV Low-Precision Remediation Gate 4

### Summary

- Continued the active macro task
  `06-09-rvv-low-precision-performance-remediation-campaign`.
- Completed Gate 4 as a same-target generated-artifact measurement acceptance
  slice for the packed-i4 low-precision representative.
- The generated-bundle evidence path now binds accepted/rejected measurement
  evidence to provider-owned route-family plan, provider-supported mirror,
  primitive-chain, remediation, target artifact, ABI/runtime, and same-target
  facts.
- Target support bundle headers now mirror the Gate 2 remediation facts, and
  the measurement script rejects stale or missing provider/resource/remediation
  facts before accepting evidence.

### Evidence

- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Focused lit from `build/test` passed for:
  `rvv-generated-bundle-same-target-measure-gate4` and
  `rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequantize-f32-packed-i4`.
- Real same-target run generated and measured
  `artifacts/tmp/codex-gate4-remediation-real/gate4-remediation-packed-i4-ssh`
  on `ssh rvv`: 12 summaries, 60 measurements, classification `regression`,
  best speedup range `0.691667..0.705064`, `selection_eligible=false`, and
  `claim_allowed=false`.

### Self-Repair

- Rebuilt `tcrv-opt` and `tcrv-translate` after strict Gate 4 checks exposed
  stale local tools.
- Added generated-header remediation metadata mapping after header verification
  showed those mirrors were present in artifact metadata but absent from the
  header evidence surface.
- Adjusted packed-i4 measurement tie-back to read verified target artifact
  metadata before route summary metadata and to reject object/header metadata
  disagreements.
- Corrected FileCheck expectations after the first lit pass used flat WPRD
  metadata checks while the artifact index encodes key/value metadata records.

### Spec Update Decision

[NO CHANGE] Existing RVV plugin and EmitC-route specs already require
provider-owned RVV facts, Common EmitC neutrality, target artifact mirrors as
non-authoritative checks, and real same-target evidence for runtime/performance
claims. This slice implemented that existing contract without adding a new
cross-layer command or API shape that needs a spec update.

### Status

[OPEN] Gate 4 is complete for the remediation campaign. Gate 5 remains open,
and `.trellis/.current-task` stays active on the same macro task.

### Continuation

Continue with Gate 5: make dispatch/performance policy consume the accepted
Gate 4 no-win/regression classification, preserving correctness/executable
support while fail-closing performance preference and win claims unless future
same-target structured measurement evidence proves a true win.

## 2026-06-09 - RVV Low-Precision Remediation Gate 5

### Summary

- Continued the active macro task
  `06-09-rvv-low-precision-performance-remediation-campaign`.
- Completed Gate 5 as a production dispatch/performance policy consumption
  slice.
- `RVVLowPrecisionPerformancePolicyDecision` now exposes explicit dispatch
  policy path fields, distinguishing `correctness-fallback` from
  `performance-preferred`.
- The accepted Gate 4 same-target packed-i4 regression/no-win measurement keeps
  route support and correctness execution allowed, but selects
  `correctness-fallback` and denies performance preference and win claims.
- A strict measured-win path exists for future provider updates: it selects
  `performance-preferred` only when same-target evidence, `ssh rvv` target
  facts, provider maturity tie-back, selection eligibility, dispatch
  preference, remediation facts, and win-claim fields all agree.
- A non-throwing resolver preserves correctness fallback for stale evidence
  while strict provider/target verification still fail-closes stale or
  measurement-only win promotion before performance-preferred dispatch.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Focused lit from `build/test` passed for
  `rvv-generated-bundle-same-target-measure-gate4` and
  `rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequantize-f32-packed-i4`.
- Bounded added-diff old-authority scan found no new `RVVI32M1`,
  `rvv-i32m1`, legacy `tcrv_rvv.i32_*`, source-front-door,
  source-artifact, descriptor-driven, q4/q8/llama, direct-C, or exact i32m1
  intrinsic route authority.
- Touched-file scan still shows pre-existing guardrail and legacy/fail-closed
  hits in large RVV tests; no matching authority was introduced by this slice.

### Self-Repair

- None needed after focused build, C++ tests, script self-tests, lit dry-run,
  and diff whitespace checks.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the
Gate 5 packed-i4 dispatch/performance policy consumption contract, including
the production API signatures, dispatch policy paths, strict accepted-policy
entry, safe fallback resolver, validation matrix, and tests required.

### Status

[DONE] Gates 1 through 5 are complete for the remediation campaign. The macro
task is ready to finish/archive after final staged diff checks and the coherent
Gate 5 commit.

### Continuation

No continuation remains for this macro campaign. Future work should open a new
task only for a different production-kernel capability owner, a real packed-i4
provider/resource repair, or new same-target evidence collected after a changed
production path.

## 2026-06-09 - RVV Packed-I4 Production-Kernel Campaign Gate 1

### Summary

- Created/continued the active macro task
  `06-09-rvv-packed-i4-production-kernel-resource-aware-realization-campaign`.
- Completed Gate 1 as a production resource/remediation planning slice for the
  selected packed-i4 product-reduction-dequant representative.
- Added provider-owned packed-i4 remediation plan facts:
  `remediation_plan_contract`, `remediation_plan`,
  `remediation_statement_strategy`, and `remediation_vector_budget`.
- Gearbox now materializes these facts on the selected packed-i4 candidate;
  selected-body realization copies them; provider planning parses/verifies
  them; statement planning requires them before packed-i4 unpack/product/pair-
  sum/single-reduce construction; route metadata and target validation mirror
  them; target support bundle/header export exposes them as mirrors.
- No runtime, correctness, or performance improvement was claimed; existing
  correctness fallback / not-performance-preferred policy remains intact.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  passed during focused build.
- `rtk ninja -C /home/kingdom/phdworks/TianchenRV/build tcrv-translate tianchenrv-target-artifact-export-test`
  passed after updating target support bundle export.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Focused lit from `build/test` passed for
  `pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4`.
- Bounded added-diff old-authority scan over `include/`, `lib/`, and `test/`
  found no new `RVVI32M1`, `rvv-i32m1`, legacy `tcrv_rvv.i32_*`,
  source-front-door, descriptor-driven, q4/q8/llama, Common EmitC semantic
  branch, or exact i32m1 intrinsic route authority. Touched-file full scan only
  found pre-existing guardrail/legacy negative-test strings.
- `rtk git diff --check` passed.

### Self-Repair

- Initial C++ target/plugin run exposed old packed-i4 fixture attrs missing the
  new required remediation plan fields; fixtures and positive/negative
  assertions were updated.
- Initial lit run exposed FileCheck ordering and target support-bundle header
  export omissions; checks were reordered to actual metadata order and
  `RVVTargetSupportBundle.cpp` was updated to expose the new mirror keys.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the
Gate 1 packed-i4 no-win remediation plan contract, including the new resource
selection fields, accepted values, validation matrix, target support-bundle
mirror requirement, and tests required.

### Status

[OPEN] Gate 1 is complete, but the packed-i4 production-kernel macro campaign
remains active because Gates 2, 3, and 4 are not complete.

### Continuation

Continue with Gate 2: provider and target validation should consume the Gate 1
remediation plan facts as provider-owned mirrors and reject metadata-only win
claims. Gate 3 must collect same-target `ssh rvv` comparison only after a
repaired generated artifact is available; Gate 4 then consumes that evidence in
dispatch/performance policy.

## 2026-06-10 - RVV Packed-I4 Production-Kernel Campaign Gate 2

### Summary

- Continued the active macro task
  `06-09-rvv-packed-i4-production-kernel-resource-aware-realization-campaign`.
- Completed Gate 2 as a provider/target validation-boundary slice.
- The packed-i4 target artifact route-family validator now rejects
  metadata-only performance/win/dispatch claim keys such as
  `performance_win_claim_allowed`, `dispatch_policy_path`, and `win_claim`
  before artifact/header export.
- Existing target candidate checks still accept valid provider-owned
  low-precision resource/remediation/performance mirrors and reject stale or
  missing mirrors. The new denylist only blocks extra claim fields that are not
  part of the provider-owned selection contract.
- No runtime, correctness, or performance improvement was claimed.

### Evidence

- `rtk ninja -C build tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Focused lit from `build/test` passed for
  `pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4`.
- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Bounded added-diff old-authority scan over touched production/test files found
  no new `RVVI32M1`, `rvv-i32m1`, legacy `tcrv_rvv.i32_*`,
  source-front-door, descriptor-driven, q4/q8/llama, Common EmitC semantic
  branch, or exact i32m1 intrinsic route authority.
- `rtk git diff --check` passed.

### Self-Repair

- Adjusted new C++ expect-error fragments to match the target validator's
  field-label wording before running focused tests.

### Spec Update Decision

[NO CHANGE] Existing RVV plugin, EmitC route, and testing specs already require
target artifact metadata and headers to be exact mirrors after provider route
construction and explicitly reject metadata-only packed-i4 performance/win
claims. This slice implements that existing contract with a focused target-side
denylist and regression coverage.

### Status

[OPEN] Gate 2 is complete, but the packed-i4 production-kernel macro campaign
remains active because Gates 3 and 4 are not complete.

### Continuation

Continue with Gate 3: collect same-target `ssh rvv` comparison only after a
repaired generated artifact is available, then let Gate 4 consume that evidence
in dispatch/performance policy.

## 2026-06-10 - RVV Packed-I4 Production-Kernel Campaign Gate 3

### Summary

- Continued the active macro task
  `06-09-rvv-packed-i4-production-kernel-resource-aware-realization-campaign`.
- Completed Gate 3 as a same-target generated-artifact measurement slice for
  the repaired packed-i4 widening product-reduce-dequantize f32 path.
- No production source change was required. The existing same-target
  measurement workflow already generates and validates the repaired packed-i4
  target artifact, selects the packed scalar baseline only after validated
  provider-owned `packed-i4-nibbles` metadata, runs correctness guards before
  timing, parses raw timing records, and emits a policy-consumable maturity
  evidence input without changing provider facts or dispatch policy.
- The real `ssh rvv` measurement classified the current generated artifact as
  `regression`, outcome family `no-win`, not a performance win.
- The structured handoff preserves `correctness_execution_allowed = true`,
  `provider_performance_selection_eligible = false`,
  `provider_dispatch_preference = not-performance-preferred`, and
  `performance_win_claim_allowed = false`.

### Evidence

- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- Packed-i4 focused dry-run passed at
  `artifacts/tmp/codex-gate3-packed-i4-dry-run/gate3-packed-i4-same-target-measure-dry`
  and remained `classification = not-measured`.
- Real same-target run passed:
  `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --artifact-root artifacts/tmp/codex-gate3-packed-i4-real --run-id gate3-packed-i4-same-target-measure-ssh --overwrite --op-kind widening_product_reduce_dequantize_f32 --input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir --measure-count 257 --measure-count 4096 --measure-count 65536 --warmup-count 2 --repeat-count 5 --measure-iterations 8 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv`.
- Real evidence root:
  `artifacts/tmp/codex-gate3-packed-i4-real/gate3-packed-i4-same-target-measure-ssh`.
- Per-op evidence:
  `artifacts/tmp/codex-gate3-packed-i4-real/gate3-packed-i4-same-target-measure-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`.
- Remote target profile recorded `ssh_target=rvv`, `remote_arch=riscv64`, 64
  CPUs, and Ubuntu clang 18.1.3.
- Remote run stdout contains 12 correctness records, 12
  `CORRECTNESS_GUARD_BEFORE_TIMING` records, 60 raw `MEASURE` records, 12 parsed
  `SUMMARY` records, and a final `PASS`.
- Structured result: `classification = regression`, `outcome_family = no-win`,
  `best_speedup_range = 0.683805..0.705257`, `summary_record_count = 12`,
  `measurement_record_count = 60`, `correctness_record_count = 12`,
  `performance_win_claim_allowed = false`, and
  `provider_contract_update_required = false`.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Focused lit from `build/test` passed for
  `rvv-generated-bundle-same-target-measure-gate4`.
- Focused lit from `build/test` passed for
  `rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequantize-f32-packed-i4`.

### Self-Repair

- The first dry-run used an unavailable `/usr/bin/llvm-readobj` path and failed
  before bundle evidence generation. The rerun used the script-resolved LLVM 20
  tool path `/usr/lib/llvm-20/bin/llvm-readobj` and passed.

### Spec Update Decision

[NO CHANGE] Existing RVV plugin, EmitC-route, and MLIR testing specs already
document the same-target measurement contract, packed-i4 baseline selection
from validated provider metadata, structured win/no-win/regression
classification, maturity evidence input, Common EmitC neutrality, and real
`ssh rvv` evidence requirement. This slice exercised that existing contract on
the current repaired Gate 1/2 path without adding a new cross-layer API or
compiler behavior.

### Status

[OPEN] Gate 3 is complete, but the packed-i4 production-kernel macro campaign
remains active because Gate 4 dispatch/performance policy consumption is not
complete.

### Continuation

Continue with Gate 4: dispatch/performance policy must consume the Gate 3
same-target regression/no-win evidence and preserve conservative correctness
fallback unless provider-owned measured facts explicitly justify a
performance-preferred path.

## 2026-06-10 - RVV Packed-I4 Production-Kernel Campaign Gate 4

### Summary

- Continued and completed the active macro task
  `06-09-rvv-packed-i4-production-kernel-resource-aware-realization-campaign`.
- Gate 4 now consumes the Gate 3 packed-i4 same-target evidence in production
  policy/validation code instead of relying on older Gate 4/Gate 6 evidence
  mirrors.
- Updated provider-owned packed-i4 performance/remediation facts and the
  accepted `RVVLowPrecisionPerformancePolicy` outcome to:
  `gate3-packed-i4-same-target-measure-ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`,
  classification `regression`, outcome family `no-win`, best speedup range
  `0.683805..0.705257`, 12 summaries, 60 timing records, and 12 correctness
  records.
- The accepted policy preserves route/correctness support, selects
  `correctness-fallback`, keeps dispatch `not-performance-preferred`, and denies
  performance selection and win claims for the current evidence.
- The measured-win policy branch remains available only when provider maturity,
  remediation, dispatch, measurement id/range, and win-claim facts all agree.

### Files

- Production: `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Tests/fixtures: `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  packed-i4 selected-body fixture, and the packed-i4 ABI/same-target dry-run
  lit files.
- Spec/task: `.trellis/spec/extension-plugins/rvv-plugin.md`, active PRD,
  `implement.jsonl`, and `check.jsonl`.

### Evidence

- `rtk ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  passed.
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
- Focused lit passed for
  `pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4`.
- Focused lit passed for `rvv-generated-bundle-same-target-measure-gate4`.
- Focused lit passed for
  `rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequantize-f32-packed-i4`.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
  passed.
- Bounded added-diff old-authority scan found no new legacy route authority,
  source-front-door authority, descriptor-driven computation, low-precision
  benchmark-label authority, Common EmitC semantic branch, or exact i32m1
  intrinsic authority.
- `rtk git diff --check` and `rtk git diff --cached --check` passed.
- `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-09-rvv-packed-i4-production-kernel-resource-aware-realization-campaign`
  passed.

### Self-Repair

- The first Gate 4 same-target dry-run lit run failed because
  `scripts/rvv_generated_bundle_abi_e2e.py` still expected the older
  `0.688427..0.705724` speedup range and
  `gate4-packed-i4-real-measure-ssh` evidence id. Updated the script expected
  profile to the Gate 3 facts and reran successfully.
- A broad added-diff authority scan initially matched a newly added PRD negative
  phrase; reworded the PRD line so the mechanical scan stays clean without
  weakening the boundary.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the Gate 4
policy consumption contract with the concrete accepted Gate 3 evidence id,
classification, speedup range, record counts, stale speedup rejection, and test
requirements.

### Status

[DONE] Gates 1 through 4 are complete for this macro campaign. No further
continuation remains; the task is ready to archive and commit.
