> Continuation from `journal-28.md` (archived at ~1980 lines)

## Session 577: Stage2 RVV low-precision contraction primitive Gate 1

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Created the new macro Trellis task for the RVV Stage 2 low-precision
contraction primitive-surface campaign and completed Gate 1 only. Source
inspection showed the signed i8 product-reduction primitive facts, unsigned u8
widening-product path, selected-body resource facts, and target mirrors already
exist in production. The remaining Gate 1 blocker was the target artifact
provider-facts preflight: it did not directly compare the selected
low-precision resource primitive surface fields against
`RVVLowPrecisionWideningReductionPrimitiveFacts` before artifact export.

### Main Changes

- Hardened target artifact provider-facts validation so
  `lowPrecisionResourceSelection` source/product/accumulator/reduction/final
  primitive facts must match provider-owned widening-reduction primitive facts
  before artifact export.
- Added focused target artifact C++ coverage mutating a packed-i4
  product-reduction dequant resource selection's primitive product SEW to prove
  stale resource primitive facts fail closed.
- Created and left open the macro Trellis task with Gate 1 complete and Gates
  2-4 remaining.

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign`
- [OK] `git diff --check`

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin spec already requires low-precision primitive
facts to be provider-owned and target artifact validation to reject stale
primitive/resource mirrors. This slice implements that existing contract at one
missing target provider-facts preflight.

### Status

[OPEN MACRO TASK] Gate 1 is complete. Gates 2-4 remain open. The next
continuation point is Gate 2 selected-body realization consumption of these
typed primitive facts for a representative low-precision contraction/dequant
path without changing compute semantics, ABI roles, runtime AVL/VL, dispatch,
or fallback behavior.

### Git Commits

Final coherent commit is created after this journal entry.
