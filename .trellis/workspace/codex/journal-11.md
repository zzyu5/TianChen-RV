# Journal - codex (Part 11)

> Continuation from `journal-10.md` (archived at ~2000 lines)
> Started: 2026-05-19

---



## Session 133: Stage2 generic RVV reduction accumulation route

**Date**: 2026-05-19
**Task**: Stage2 generic RVV reduction accumulation route
**Branch**: `main`

### Summary

Added a bounded generic typed RVV reduce(add) selected-body route skeleton with verifier, construction protocol, provider materialization, artifact/header dry-run, fail-closed negatives, and no ssh-rvv runtime claim.

### Main Changes

- Added generic `tcrv_rvv.reduce` for a typed vector input plus typed accumulator/result under matching `tcrv_rvv.with_vl`; the bounded supported reduction kind is `add`.
- Extended RVV construction protocol route recognition with `reduce_add`, generic `tcrv_rvv.reduce`, and provider-owned `rvv-generic-reduce-add-*` route/runtime ABI identity.
- Extended the RVV EmitC route provider to derive a reduction leaf from typed body/config/runtime facts and to reject RHS-broadcast reduction in this slice.
- Added positive EmitC materialization and target header/artifact dry-run fixtures plus negative verifier/provider coverage.

### Testing

- [OK] Trellis task context validation.
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused reduction lit set: 5/5 passed.
- [OK] focused existing first-slice EmitC regression lit set: 5/5 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 154/154.
- [OK] `git diff --check`
- [OK] active-authority scan: diff-only legacy-shaped addition is provider-derived `__riscv_vredsum_vs_i32m1_i32m1`; no new `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`, or `tcrv_rvv.i32_macc` helper surface.

### Status

[OK] **Completed** as route-supported and artifact/header dry-run evidence. No `ssh rvv` correctness or performance claim was made in this round.

### Next Steps

- Future continuation, if requested: turn the reduction skeleton into executable runtime correctness evidence on `ssh rvv` and refine accumulator/result layout beyond this bounded vector accumulator form.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |


## Session 134: Stage1 Gate A RVV route identity cleanup

**Date**: 2026-05-19
**Task**: `stage1-gate-a-rvv-route-identity-cleanup`
**Branch**: `main`

### Summary

Closed the bounded Stage1 Gate A route-identity residues for the active RVV selected-body route path. Route ids, runtime ABI names, artifact route/header/bundle names, config-contract APIs, source-front-door pass naming, and pre-realized selected-body entry points now use generic selected-body / typed-body authority rather than `rvv-i32m1`, `RVVI32M1`, or `i32_binary_pre_realized_body` as positive route identity.

### Main Changes

- Renamed active RVV construction identities to `rvv-generic-typed-body-*` and per-operation provider-derived `rvv-generic-binary-*`, `rvv-generic-cmp-select-*`, and `rvv-generic-reduce-add-*` route/runtime ABI labels.
- Renamed active config-contract APIs and metadata to selected-body generic names while preserving SEW32/LMUL/policy facts as typed config facts.
- Replaced positive `tcrv_rvv.i32_binary_pre_realized_body` fixtures with `tcrv_rvv.typed_binary_pre_realized_body`.
- Kept retained i32 add/sub/mul as ordinary generic typed `tcrv_rvv.binary {kind = ...}` instances, and kept legacy `tcrv_rvv.i32_*` selected-body snippets only as negative fail-closed fixtures.
- Renamed the RVV source-front-door pass to a fail-closed legacy vector-source front-door identity.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] focused lit filter for `Target/RVV`, RVV lowering/source-front-door transforms, and RVV script dry-runs: 40/40 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 154/154 passed.
- [OK] `git diff --check`
- [OK] exact Stage1 Gate A brief scan over `include/TianChenRV lib/Plugin/RVV lib/Dialect/RVV test/Target/RVV`: no `rvv-i32m1`, `RVVI32M1`, or `i32_binary_pre_realized_body` matches.
- [OK] remaining extended legacy matches are classified as negative fail-closed Target/RVV fixtures, dialect parse/verifier debt, and provider-derived intrinsic leaf spellings.

### Status

[OK] **Completed and ready to archive**. No `ssh rvv` runtime, correctness, or performance claim was made.

### Next Steps

- Future continuation, if requested: delete or further fail-close the remaining parseable legacy dialect debt under a separate Stage1 deletion task; do not treat it as Stage2 coverage work.


## Session 135: Stage2 executable closure for generic RVV reduction

**Date**: 2026-05-19
**Task**: `stage2-generic-rvv-reduction-executable-closure`
**Branch**: `main`

### Summary

Closed the existing generic `tcrv_rvv.reduce {kind = "add"}` selected-body
path as executable RVV evidence. The provider now makes the bounded reduction
layout explicit: RHS is the vector seed, lane 0 carries the per-VL chunk result,
and generated code stores only lane 0 to `out[offset]`.

### Main Changes

- Added provider-derived reduction accumulator/result layout fields to
  `RVVSelectedBodyEmitCRouteDescription`.
- Updated the RVV EmitC route provider so reduce-add stores with VL `1`, making
  non-result lanes untouched and giving the external ABI harness a precise
  correctness contract.
- Added reduce-add metadata mirrors to target artifact bundle metadata.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with `--op-kind
  reduce_add` dry-run and `ssh rvv` correctness support.
- Updated focused reduce-add materialization and artifact fixtures.

### Testing

- [OK] Trellis context validation.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused C++ build for `tcrv-opt`, `tcrv-translate`, RVV plugin,
  construction protocol, and target artifact export tests.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] reduce-add generated bundle dry-run.
- [OK] existing add/sub/mul/cmp_select generated bundle dry-run.
- [OK] real `ssh rvv` run for reduce-add counts `1,7,16,17,257`:
  `tcrv_rvv_generated_bundle_abi_reduce_add_ok counts=1,7,16,17,257`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 154/154 passed.
- [OK] `git diff --check`
- [OK] diff-only active-authority scan introduced no legacy RVV route authority.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for the bounded reduce-add chunk layout described above.

### Next Steps

- Future continuation, if requested: broaden reduction/accumulation semantics
  beyond this bounded per-VL chunk layout under a separate Stage2 coverage task.


## Session 136: Stage2 generic RVV masked add route semantics

**Date**: 2026-05-19
**Task**: `stage2-rvv-masked-add-route-semantics`
**Branch**: `main`

### Summary

Added one bounded generic `tcrv_rvv.masked_binary {kind = "add"}` selected-body
route. The route carries compare-produced mask flow, passthrough/result vector
role, VL, typed vector/mask shape, and policy facts through RVV dialect
verification into RVV provider route construction and target artifact evidence.

### Main Changes

- Added `tcrv_rvv.masked_binary` and verifier checks for compare mask source,
  same `with_vl` scope, VL token consistency, vector/mask compatibility,
  passthrough/lhs/rhs/result agreement, and bounded `kind = "add"`.
- Added RVV construction protocol support for the `masked_add` selected-body
  route, typed op identity, role sequence, ABI mapping, and artifact metadata.
- Added provider-side masked-add route validation and materialization:
  compare mask, active add, passthrough merge, and store. Common EmitC/export
  remains neutral.
- Extended target artifact export tests and `rvv_generated_bundle_abi_e2e.py`
  with `--op-kind masked_add` dry-run and executable harness support.
- Added positive/negative dialect, materialization, artifact, and script tests;
  updated stale generic-op diagnostic fixtures to include `tcrv_rvv.masked_binary`.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV plugin,
  construction protocol, and target artifact export tests.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused masked-add dialect/materialization/negative/target artifact
  FileCheck commands.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] masked-add generated-bundle dry-run.
- [OK] real `ssh rvv` masked-add correctness evidence for counts
  `1,7,16,17,257`:
  `tcrv_rvv_generated_bundle_abi_masked_add_ok counts=1,7,16,17,257`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 157/157 passed.
- [OK] `git diff --check`
- [OK] diff active-authority scan introduced no legacy RVV route authority;
  remaining exact i32m1 intrinsic matches are provider-derived leaf checks.

### Self-Repair

- Replaced an initially attempted five-argument masked add intrinsic shape after
  current RVV clang rejected it during generated object packaging. The final
  bounded route uses active add plus provider-derived merge with passthrough.
- Repaired stale broad lit diagnostics that omitted the newly allowed
  `tcrv_rvv.masked_binary` op from generic selected-body op lists.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for bounded masked add counts `1,7,16,17,257`.

### Next Steps

- Future continuation, if requested: expand Stage2 masked execution classes
  beyond this single masked add submodule under a separate bounded task.


## Session 137: Stage2 RVV typed SEW/LMUL config derivation for generic arithmetic

**Date**: 2026-05-19
**Task**: `stage2-rvv-typed-config-arithmetic`
**Branch**: `main`

### Summary

Closed the bounded Stage2 generic arithmetic typed-config task by repairing the
RVV provider-local resolver to cross-check typed vector/mask element width and
LMUL against selected `setvl`/`with_vl` config before route/profile/intrinsic
selection. Existing m2 arithmetic support was confirmed as a real generic typed
selected-body route, not stale artifact metadata, and was revalidated through
local artifact dry-run plus real `ssh rvv` correctness evidence.

### Main Changes

- Added provider-side typed-config validation for generic
  `!tcrv_rvv.vector` and `!tcrv_rvv.mask` values before deriving route
  profiles and intrinsic/header payloads.
- Added C++ plugin coverage that proves LMUL m2 selected-body add derives m2
  RVV intrinsics and then fails closed if the same typed m2 body is paired with
  stale LMUL m1 config metadata.
- Kept the change inside the RVV provider/test surface; common EmitC/export,
  source front doors, reduction/mask/broadcast coverage, and route-family
  breadth were not expanded.

### Testing

- [OK] Trellis task context validation.
- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, RVV plugin,
  construction protocol, and target artifact export tests.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] focused m2 target artifact/script lit commands: 4/4 passed.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] m2 add/sub/mul generated-bundle dry-run at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-typed-config-m2-dry`.
- [OK] real `ssh rvv` m2 add/sub/mul correctness evidence at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-typed-config-m2-ssh`:
  `PASS op=add counts=7,16,23`, `PASS op=sub counts=7,16,23`,
  and `PASS op=mul counts=7,16,23`.
- [OK] `git diff --check`
- [OK] diff active-authority scan introduced no legacy RVV route authority.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 157/157 passed.

### Status

[OK] **Completed and ready to archive**. This round makes a real RVV
correctness claim only for bounded m2 generic arithmetic add/sub/mul counts
`7,16,23`.

### Next Steps

- Future continuation, if requested: broaden typed-config derivation beyond
  this bounded generic arithmetic m1/m2 surface under a separate Stage2 task.
