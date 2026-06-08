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
