# Stage2 RVV runtime-scalar-cmp masked indexed scatter-store ABI boundary

## Goal

Make the existing RVV runtime-scalar-cmp masked indexed scatter-store selected-body route executable as generated RVV artifacts with truthful ABI/runtime evidence, or harden the exact executable artifact boundary to fail closed when stale or missing facts would otherwise make the route appear executable.

This task owns the artifact boundary for the scatter-store side of the runtime-scalar-cmp indexed-memory family. It must preserve the TianChen-RV authority chain: selected typed `tcrv_rvv` body facts are consumed by the RVV plugin provider, lowered through `TCRVEmitCLowerableRoute` and common EmitC mechanics, exported as a target artifact bundle, and proven with generated-bundle evidence when runtime behavior is claimed.

## Requirements

- Keep route authority in typed `tcrv_rvv` body/config/runtime facts and RVV plugin validation, not route ids, helper names, tests, metadata, or common EmitC semantic branching.
- Cover the runtime-scalar-cmp masked indexed scatter-store route family, including explicit selected-body and pre-realized selected-body inputs.
- Align runtime scalar comparison operands, computed predicate/mask facts, masked active/inactive lane policy, indexed address/data memory roles, store side-effect role, dtype/SEW/LMUL/config/policy, runtime AVL/VL, header/prototype binding, ABI order, and generated statement facts.
- If the executable seam is currently dry-run-only, stale, or under-validated, make a focused production or target-boundary change rather than recording Trellis-only evidence.
- If the seam is already executable, close the exact remaining target-boundary or fail-closed blocker for this route family and document why no broader source change is needed.
- Preserve common EmitC/export neutrality: the common materializer may emit neutral mechanics, but it must not invent RVV scatter-store semantics.
- Add or update focused positive and negative evidence for the executable artifact boundary.

## Acceptance Criteria

- [x] Explicit selected-body runtime-scalar-cmp masked indexed scatter-store artifact path reaches materialized selected boundary, emission plan, target artifact export, generated bundle compile, and `ssh rvv` correctness evidence if runtime correctness is claimed.
- [x] Pre-realized selected-body runtime-scalar-cmp masked indexed scatter-store artifact path reaches the same executable evidence level when claimed.
- [x] At least one stale or missing executable-boundary fact for this seam fails closed with a targeted diagnostic or focused test.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Relevant generated-bundle dry-run tests for explicit and pre-realized runtime-scalar-cmp masked indexed scatter-store pass.
- [x] A bounded old-authority scan over touched files and added diff lines shows no new route authority from legacy i32/helper/metadata/source-front-door patterns.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Worktree is clean after final commit if the task is finished.

## Definition of Done

- Task context and PRD are truthful for this bounded module owner.
- Source or test changes are committed as one coherent commit, unless repository evidence proves a no-source-change finish is the only correct outcome.
- Trellis task status is finished and archived when the implementation and checks are complete.
- Workspace journal records the completed round and evidence.

## Out of Scope

- Broad indexed-memory matrix expansion.
- Dtype/LMUL clone batches.
- Vector gather/scatter or computed-mask-only rework except as bounded reference.
- MAcc, product-reduce, dequant, clamp, segment2, reduction, compare/select, conversion, or unrelated mask route rewrites.
- High-level Linalg/Vector/StableHLO frontend work or per-Linalg route authority.
- Performance tuning databases, dashboards, reports-only work, or broad smoke matrices.
- Source-front-door positive routes.
- Common EmitC invention of RVV semantics.

## Technical Notes

- Read first:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-masked-indexed-gather-load-abi/`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `lib/Plugin/RVV/EmitC/`
  - `lib/Target/RVV/`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-scatter-store-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-indexed-scatter-store-dry-run.test`
  - `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-scatter-store.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-scatter-store.mlir`
- Initial repository state:
  - `pwd`: `/home/kingdom/phdworks/TianchenRV`
  - `git status --short`: clean
  - recent head: `5dce0aa0 rvv: harden runtime scalar indexed gather ABI boundary`
- Repository research:
  - The runtime-scalar-cmp masked indexed scatter-store selected/pre-realized
    MLIR fixtures and generated-bundle dry-run tests already exist.
  - The production route-planning layer already exposes
    `RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad` facts, including
    runtime ABI order `lhs,rhs_scalar,src,index,dst,n`, RHS scalar splat facts,
    masked indexed store leaf, empty ordinary store residue, and
    provider-owned computed-mask indexed validation contract.
  - `test/Target/TargetArtifactExportTest.cpp` has manual target-boundary
    coverage for vector computed-mask indexed scatter and for runtime-scalar
    indexed gather, but it lacks the equivalent runtime-scalar indexed scatter
    manual route/candidate contract coverage. This is the bounded
    target-boundary blocker for this round.
- Completed behavior:
  - Added manual target-boundary coverage for runtime-scalar indexed scatter:
    provider validation contract access, positive provider/candidate
    validation, stale runtime-scalar binding summary rejection, stale mask
    producer rejection, stale runtime ABI order rejection, and candidate mirror
    rejection for the same stale facts.
  - No production route semantics changed; the route was already executable.
    This round closes the missing target-boundary/fail-closed coverage seam.
- Evidence:
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`: pass.
  - `build/bin/tianchenrv-target-artifact-export-test`: pass.
  - `build/bin/tianchenrv-rvv-extension-plugin-test`: pass.
  - `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter runtime-scalar-cmp-masked-indexed-scatter-store` from `build/test`: pass, 4/4.
  - Explicit selected-body generated bundle on `ssh rvv`: pass for counts
    `0,1,16,17,257`, `rhs_scalar=-37,91`, patterns `0,1`, with
    active/inactive lane coverage, noncontiguous indexed scatter,
    `source_preserved`, and `tail_preserved`.
  - Pre-realized selected-body generated bundle on `ssh rvv`: pass for the
    same counts, scalar values, and patterns with the same preservation
    evidence.
  - Bounded added-code old-authority scan: no hits for legacy i32/helper,
    descriptor/direct-C/source-export/source-front-door route authority terms.

## Current Phase

finish
