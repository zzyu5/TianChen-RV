# Stage 3 换心 — ComputedMaskMemory family: converted + string-plan owner deleted

Base HEAD: a0a601d3 (3rd owner, BaseMemoryMovement, retired). This change converts the
**ComputedMaskMemory** family through the real DialectConversion (RVVToEmitC.cpp) and
deletes its string statement-plan owner from the shared
`lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`. Segment2 (incl. its own
computed-mask variants) stays string-plan owned and INTACT.

## What converted (10 op-kinds, all PASS on real ssh rvv riscv64)

Vector-compare computed mask:
- computed_masked_unit_load_store        (compare -> masked_load _tumu -> store)
- computed_masked_strided_store          (compare -> masked_strided_store _m)
- computed_masked_strided_load_unit_store(compare -> masked_strided_load _tumu -> store)
- computed_masked_indexed_gather_load_unit_store (compare -> masked gather vluxei _tumu -> store)
- computed_masked_indexed_scatter_store_unit_load (compare -> masked scatter vsoxei _m)

Runtime-scalar compare (rhs splat) computed mask:
- runtime_scalar_cmp_masked_store        (splat -> compare -> masked_store _m, undisturbed policy)
- runtime_scalar_cmp_masked_load_store   (splat -> compare -> masked_load _tumu -> store)
- runtime_scalar_cmp_masked_indexed_gather_load_unit_store
- runtime_scalar_cmp_masked_indexed_scatter_store_unit_load
- runtime_scalar_cmp_masked_indexed_gather_macc_scatter (gather -> macc -> scatter compound)

PASS lines: ssh-rvv-computedmask-converted.log (genuine harness output, NO --dry-run,
counts 1/7/16/17/257, rhs_scalars -37/91).

## Conversion patterns added (RVVToEmitC.cpp, +553 net)

- Relaxed the masked_load / masked_store mask-source guard: accept a mask from a
  tcrv_rvv.compare (computed mask) OR a tcrv_rvv.mask_load (base-memory buffer mask) —
  both lower to the byte-identical `_tumu`/`_m` forms (`isMaskFromMaskLoadOrCompare`).
- 4 new emitters: emitMaskedStridedLoad (vlse _tumu), emitMaskedStridedStore (vsse _m),
  emitMaskedIndexedLoad (vluxei _tumu), emitMaskedIndexedStore (vsoxei _m) + their
  intrinsic-name helpers, all guarded (form/unit/policy/byte-stride/EEW/mask-source).
- Policy gate exception `isComputedMaskMaskedStoreBody`: the runtime-scalar (splat)
  unit masked-store body carries undisturbed scope policy honored by the `_m` store;
  REQUIRES the compare RHS to be a splat (the only legitimate unit store-only family) —
  a vector-vector compare unit store-only body still falls back (the stage2-masked-store
  negative contract is preserved).
- Index-early ordering reorder: for a computed-mask indexed body, move index_load right
  after the first load and emit the element->byte scale at index_load time, so the
  rendered C matches the legacy string-plan byte order (and the macc-scatter ordered-
  token harness chain).

## Byte-identity

21 of 23 in-scope fixtures (.cpp via --tcrv-rvv-emitc-to-cpp) are BYTE-IDENTICAL to the
legacy string-plan oracle (the index-early reorder achieved exact identity for the
indexed/strided/unit shapes). The 2 macc-scatter fixtures differ by ONE redundant-but-
correct `vmerge` before the masked scatter (emitMaskedMAcc applies the predicate via a
merge; the scatter is already masked, so the scattered active-lane values are identical
— confirmed by the macc-scatter hardware PASS and the passing ordered-token dry-run).

## String-plan code DELETED (ComputedMaskMemory-exclusive)

Shared owner file `RVVEmitCMemoryStatementPlanOwners.cpp`: 1435 -> 667 lines.
Across the 3 string-plan files: -863 deletions / +66 retirement comments.
Deleted: the 5 owner-only helpers (requireRVV...{Leaf,ABI},
getRVV...SourceProvenance, makeRVV...Step, addRVV...LoopStep), the orchestration
(buildComputedMaskMemoryMigratedRouteStatementPlan), the public
buildRVVSelectedBodyComputedMaskMemoryMigratedRouteStatementPlan, the ~640-line
getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan, the dispatch-table entry, the
consumer predicate (isRVVSelectedBodyComputedMaskMemoryStatementPlanConsumer), and the
3 header decls. Each removal grep-gated to 0 live refs.

KEPT (grep-proven shared/live): the Segment2 builder + the shared
stringify/setRVVSelectedBodyMemoryMigratedRouteStatementPlan helpers it still uses; the
ComputedMaskMemory enum case (used by stringify); the route-family provider machine
(getRVVSelectedBodyComputedMaskMemoryRouteFamilyPlan,
verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts) + the
RVVSelectedBodyComputedMaskMemoryRouteStatementPlan struct — these live in
RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners and are the description/provider source
of truth (the struct could not be cleanly removed: the route-family verify function
still takes it; NOT isolable, kept as reported).

## Validation

- Full clean rebuild (ninja -t clean && ninja && ninja bin/tcrv-opt bin/tcrv-translate);
  tools newer than libs.
- `cd build/test && lit.py -q .` -> EXACTLY 3 environmental reds (same as HEAD baseline:
  2x computed-masked-strided-input-widening-dot-reduce-add-dry-run + self-test). 3 new
  structural lit tests added under test/Conversion/RVV (565 total).
- Adversarial guard probes: vector-vector compare unit store-only (negative) refuses;
  non-byte-stride masked_strided_load refuses; legitimate runtime-scalar masked-store
  converts.
- ssh rvv (riscv64, NO --dry-run): all 10 converted op-kinds PASS; all 7 Segment2 ops
  PASS (owner intact). Logs: ssh-rvv-computedmask-converted.log,
  ssh-rvv-segment2-intact.log.
