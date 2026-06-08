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
