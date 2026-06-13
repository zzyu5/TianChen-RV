# RoutePlanning analysis restructure — flip the Gearbox dequant realization to typed IR, retire the contraction-owner body path

Author: Opus (senior MLIR architect). Date: 2026-06-13.
Task: `.trellis/tasks/06-12-stage3-replace-string-machine/`. Parent ADR: 06-12-mlir-audit-refactor.
Builds on (does NOT duplicate): `research/gearbox-dequant-retirement-plan.md` (Fable, realization-rewrite verdict,
ODS op + conversion handlers — **now BUILT**, commit 7bfd6012). This is the NEXT rung: with the conversion
already done and clean, resolve the metadata-source question Fable left implicit and a4 surfaced, plus the
RoutePlanning analysis relaxation. The 40k-line `RVVEmitCRoutePlanning.cpp` analysis is the real remaining work.

---

## Verdict: **re-baseline-then-flip** (bounded — NOT genuinely unbounded, NOT a clean bounded-flip)

Relax the always-run RoutePlanning analysis to admit the typed single-scope nibble dequant body, flip the
realization to emit it, re-home the gearbox resource facts off the deleted handoff op onto `with_vl` (fact
VALUES unchanged), re-baseline the 2 packed-i4 body-C fixtures, gate correctness on `ssh rvv` (tolerance=1e-05),
then delete the legacy contraction body-emission path.

It is **not** a clean `bounded-flip` because the 2 packed-i4 fixtures lock emitted C and the emission path
changes — re-baseline is forced. It is **not** `genuinely-unbounded`: a4's "metadata-derivation layer that does
not exist" is wrong — the ~70 facts are sourced from the gearbox `selectedResourceCandidate`, not invented at
the handoff (`RVVContractionSelectedBodyRealizationOwner.cpp:2089-2093`, facts at `:1342-1345`,
`RVVGearboxSchedules.cpp:244-251,423-429,753-758`). The handoff is a *carrier*, not a source. Re-homing the
carry onto `with_vl` is additive plumbing (the realization already stamps a subset there — `realized_unroll_factor`,
`realized_vsetvl_region_count`, `realized_peak_live_vector_groups` — explicit fixture line 39), not a new
derivation engine.

---

## real_blocker: the always-run ANALYSIS + the non-gated metadata source. NOT the 28-site plan-emission.

Reconciling the four findings agents:
- **a1 / stage3 under-counted.** Correct that the analysis single-compute relaxation is bounded, but stopped
  before the metadata layer.
- **a4 over-counted.** Correct that the metadata layer is real and non-gate-skipped; wrong that it is unbounded
  or an I5 violation (see I5 ruling below).
- **a2 is closest** and supplies the piece a1 missed: the null `wideningProductOp` deref in
  `deriveRVVSelectedBodyContractionRouteFamilyPlan` (`RVVEmitCContractionRouteFamilyPlanOwners.cpp:9860/9863/…`)
  runs inside the always-run analysis (called at `RVVEmitCRoutePlanning.cpp:35407` from `analyzeRVVSelectedBodyRoute`),
  so a nibble head (`slice.wideningProductOp` null) crashes it. Fold this in.

So the real blocker is the **always-run analysis** (`analyzeRVVSelectedBodyRoute`, `RVVEmitCRoutePlanning.cpp:34168`),
which runs on EVERY path including the gate-true converting path (`RVVEmitCRouteProvider.cpp:449` calls it
unconditionally; only the route BUILD at `:453-456` is gated). Three analysis sub-blockers + one non-gated
metadata source, all bounded:
1. **Missing nibble dispatch.** `recordRVVSelectedBodyScopedRouteOp` has no arm for
   `PackedI4NibbleUnpackProductOp` (grep count 0 in `RVVEmitCRoutePlanning.cpp`); committed code fails FIRST at
   the unrecognized-op fallthrough (`:17636`), not the single-compute message.
2. **Single-compute hatches assume a WideningProduct head.** Once a dispatch arm records a non-WideningProduct
   kind, the standalone-reduce hatch (`:16480-16513`) and dequant hatch (`:17263-17291`) reject it (they hardcode
   `arithmeticKind == WideningProduct` and deref `slice.wideningProductOp`).
3. **Family-plan null deref** (a2): `deriveRVVSelectedBodyContractionRouteFamilyPlan` derefs the null
   `slice.wideningProductOp` when `isProductReductionChain` holds for the nibble-headed dequant family.
4. **Non-gated metadata source.** The PLAN/HEADER `low_precision_resource.*` + `gearbox.*` config metadata
   (`RVVEmitCRoutePlanning.cpp:39540-39584`, emitted into the description at `RVVExtensionPlugin.cpp:637-648`,
   description returned on the gate-true path at `RVVEmitCRouteProvider.cpp:458`) is sourced from the handoff
   (`handoff.get*()` at `:17117/17121/…`). The single-scope target deletes the handoff, so the facts lose their
   current source. They are re-derivable from the gearbox candidate; the fix is to re-home the carry.

---

## gate_skip_finding: the 28-site plan-emission coupling IS gate-skipped — NOT a blocker (CONFIRMED)

The convert-gate `rvvSelectedBodyFullyConvertsToEmitC` (`RVVEmitCRouteProvider.cpp:422-442`) sets
`ownerRoute = nullptr` on convert (`RVVExtensionPlugin.cpp:616-617`). `describeRVVSelectedBodyEmitCRoute` then
runs `analyzeRVVSelectedBodyRoute` unconditionally but runs `buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis`
(the 28 `slice.wideningProductOp` derefs in the statement-plan owner build path) only `if (verifiedRoute)`
(`:453-456`). For a converting family `verifiedRoute` is null, so the 28-site build path is **unreachable** —
both because it is gated AND because the analysis hard-fails upstream first. The commit 7bfd6012 message blaming
the 28-site coupling as THE blocker is a red herring. This is the one point a1/a2/stage3/a4 all agree on, and it
is correct. Do not spend effort on the statement-plan owner build path for the converting families.

---

## single_compute_fix: one unified analysis workstream (covers a1 + a2). Blast radius: the dequant-chain sites only.

The "exactly one selected compute op" guard is a single-pointer slot (`slice.arithmeticOp`); ~70 other
"exactly one" sites are per-op-kind slot guards for unrelated routes (mask_and, setvl, loads/stores, segment2,
…) and are NOT touched by adding a product variant. The fix is local to the dequant-chain sites that assume a
WideningProduct head:

1. **Add a `slice.nibbleProductOp` slot + recorder + dispatch arm.** Add a
   `dyn_cast<PackedI4NibbleUnpackProductOp>` arm in `recordRVVSelectedBodyScopedRouteOp` (after the
   widening_product arm ~`:17581`) calling a new `recordRVVSelectedBodyNibbleUnpackProduct` that mirrors
   `recordRVVSelectedBodyWideningProduct` (`:16716-16740`) but stores into `slice.nibbleProductOp` and sets a
   `NibbleProduct` arithmeticKind.
2. **Widen the hatch predicates** at `:16480-16513` (standalone-reduce) and `:17263-17291` (dequantize) from
   `== WideningProduct` / `== WideningProductReduceAdd` to also accept the nibble-headed kinds; replace the
   `slice.wideningProductOp` SSA-link/`getResult()` derefs (`:16489/16499/17266`) with a small
   `productSlotResult(slice)` accessor that returns whichever of `wideningProductOp` / `nibbleProductOp` is
   populated. Mirror for the select/clamp hatches (`:16341-16384`) for the clamp family.
3. **Null-guard the family-plan derivation** (a2): in `deriveRVVSelectedBodyContractionRouteFamilyPlan`
   (`RVVEmitCContractionRouteFamilyPlanOwners.cpp:9856-9892,9972,10164,10179`) route LhsValue/RhsValue and the
   product-relation reads through the same `productSlotResult` accessor so a null `wideningProductOp` no longer
   crashes. The always-run description deref at `RVVEmitCRoutePlanning.cpp:34211` is already null-guarded
   (`if (analysis.slice.wideningProductOp) …`) — a nibble body just leaves `wideningProductRelation` unset; the
   converted-path metadata must not require it (read the relation off the nibble op's `product_relation` attr).

**Blast radius (analysis):** bounded to the ~7 `wideningProductOp` deref sites
(`RVVEmitCRoutePlanning.cpp:16348,16371,16489,16499,16897,17266,34211` + the family-plan owner sites). The
single-slot invariant is preserved. The existing `WideningProduct->StandaloneReduce->Dequantize` chain already
ships through these exact hatches via hardcoded escape hatches — the nibble flip is an additive variant of a
template the codebase already implements.

---

## I5 ruling (the load-bearing interpretation — a4's tie reversed, confirmed from spec + code)

`.trellis/spec/architecture/core-invariants.md:21-23` (I5): executable dtype/config/operation must be
**structural** in the typed body or consumed into the realized body before route construction; it FORBIDS
*inferring* dtype/config/operation from ABI strings, route ids, artifact/test names, intrinsic spelling, etc.
It does NOT forbid CARRYING resource facts as reporting metadata that the conversion never reads to shape compute.

a4 conflated "any `with_vl` resource attr" with the violation. The real violation is a compute-shaping read.
Confirmed from code: the conversion reads ZERO resource mirrors — `grep` of `RVVToEmitC.cpp` for
`low_precision_resource|candidate_fact|operand_form|unpack_intent|gearbox_cross_region_handoff` returns one hit,
a COMMENT (`:2564`) stating it does NOT read those strings. The packed-i4 correctness comes from the op TYPE
(`packed_i4_nibble_unpack_product` carries `product_relation`; the fixed vsll/vsra/vwmul/vsra/vwmacc chain is the
op's lowering, fixture `rvv-to-emitc-...-packed-i4.mlir:42-48`), not from any candidate string. So re-homing the
~70 reporting facts onto `with_vl` is I5-LEGAL: it is the N3 PLAN/HEADER reporting path, not the compute path.

---

## Metadata-source design: BRANCH (b) — re-home facts onto with_vl (decided from as-built conversion)

The as-built conversion's `isLowPrecisionDequantBody` walk (`RVVToEmitC.cpp:1440-1460`) REJECTS the handoff and
any leftover marker/second-scope (`:1454-1459` "the gearbox handoff … is not the fully-typed dequant body …
fall back to legacy"). It owns ONLY the single-scope, no-handoff body: multiple inline product+reduce slices,
then inline dequant + store. So the realization flip MUST emit single-scope/no-handoff (branch b, not Fable's
implicit handoff-retention branch a).

Therefore:
- **Realization** (`RVVContractionSelectedBodyRealizationOwner.cpp` ~`:1996-2169`): drop the 3
  `vsetvl_region_marker`s, the `createRealizedGearboxCrossRegionHandoff` call (`:2089-2093`), and the second
  `with_vl` consumer scope (`:2095-2098`); inline the dequant/clamp chain in the single scope; emit
  `packed_i4_nibble_unpack_product` for the packed candidate; add `unroll_factor` on `with_vl`. EXTEND the
  existing `realized_*` stamping on `with_vl` to carry the remaining ~70 gearbox/primitive/candidate facts
  (same `selectedResourceCandidate` source as today — `:1342-1345` etc.), so fact VALUES are unchanged.
- **Analysis** (`RVVEmitCRoutePlanning.cpp`): redirect the ~70 `handoff.get*()` reads (`:17105-17162` and the
  `recordRVVSelectedBodyGearboxCrossRegionHandoff` block ~`:16886`) to read the same attrs off the `with_vl`
  scope when no handoff is present. The `selection.*` fields and the PLAN/HEADER emission (`:39540-39584`) are
  unchanged downstream.

Because the fact VALUES are identical (same gearbox-candidate source, just a different IR carrier), the PLAN and
HEADER fixtures' resource/gearbox lines (`explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir:77-107`)
do **not** need re-baselining. Only the 2 packed-i4 body-C fixtures move (emission path changed).

**Caveat — vestigial two-scope facts.** Most facts survive cleanly (SEW/lmul/route-id/construction). But the two
facts that DESCRIBE the deleted structure — `gearbox.producer_scope = "gearbox-scope:product-reduction"` and
`consumer_scope = "gearbox-scope:dequant-store"` (fixture lines 83-84), sourced from the handoff's
`from_phase`/`to_phase` (`RVVContractionSelectedBodyRealizationOwner.cpp:1161-1165`) via
`selection.producerScope/consumerScope` — describe a producer/consumer SPLIT that a single scope no longer has.
They survive byte-stable ONLY if the re-home stamps those exact literal strings onto the lone `with_vl` as
deliberate vestigial N3 reporting facts (I5-safe: the conversion never reads them — `RVVToEmitC.cpp:2564`). State
this explicitly; do not assert clean survival, or step 4's byte-diff surprises you and the re-baseline scope
silently widens.

---

## diagnostic_handling: survive (PLAN/HEADER) + re-baseline only the 2 packed-i4 body-C fixtures

- **SURVIVE:** the dequant diagnostics and the N3 PLAN/HEADER resource contract are analysis-derived from
  unchanged fact values; with branch (b) re-homing them onto `with_vl`, the `low_precision_resource.*` /
  `gearbox.*` PLAN (lines 77-86) and HEADER (lines 101-107) lines, the `STALE-*` validation RUN lines
  (`:144-146`), and the construction/config metadata (`getRVVSelectedBodyConstructionMetadataFacts`
  `:35884-35894`, derived from the description's operation kind / typedComputeOpName / route ids — never
  handoff-sourced) all survive unchanged.
- **RE-BASELINE (forced):** the 2 packed-i4 body-C fixtures only — the old golden C was produced by the legacy
  string machine being retired, so it is NOT an independent authority; the I5-honest authority is the
  structurally-derived converted emission. Re-baseline those CPP `CHECK` lines to the converted output, gated on
  the HW lamp.
  - `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4.mlir`
  The two grouped fixtures and the explicit-realization fixtures carry no body-C gates.

---

## ordered_steps

1. **Baseline.** Confirm build + lit green; capture legacy golden C for all 4 dequant fixtures and the grouped +
   packed-i4 HW PASS (tolerance=1e-05) as the equivalence + correctness oracle.
2. **Realization flip** (`RVVContractionSelectedBodyRealizationOwner.cpp` ~`:1996-2169`): emit the single-scope
   no-handoff typed body (drop markers + handoff + second scope; inline dequant/clamp; emit
   `packed_i4_nibble_unpack_product`; add `unroll_factor` + the re-homed ~70 facts on `with_vl`). Add a
   structural lit test asserting the realized IR carries the typed ops + the facts on `with_vl`, no handoff.
3. **Analysis relax** (`RVVEmitCRoutePlanning.cpp` + `RVVEmitCContractionRouteFamilyPlanOwners.cpp`): add the
   nibble dispatch arm + `recordRVVSelectedBodyNibbleUnpackProduct` + `slice.nibbleProductOp`; widen the
   standalone-reduce / dequant / select hatch predicates and route product-slot derefs through
   `productSlotResult`; null-guard the family-plan derivation; redirect the ~70 handoff reads to the `with_vl`
   carrier when no handoff is present.
4. **Validate (local).** `tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans`
   on all 4 dequant fixtures: analysis admits the typed body, no crash, PLAN/HEADER resource lines byte-match
   the pre-flip values. Run the conversion fixtures (`test/Conversion/RVV/rvv-to-emitc-...dequant*.mlir`) green.
   Re-green full lit; expect ONLY the 2 packed-i4 body-C fixtures red (emission changed).
5. **Re-baseline** the 2 packed-i4 body-C CPP `CHECK` lines to the converted output; document the old golden as
   the retired string machine (not an authority).
6. **ssh rvv hardware lamp** (CORRECTNESS AUTHORITY — non-negotiable gate): re-run the bundle-ABI e2e for the
   grouped AND packed-i4 dequant + dequant-clamp WITHOUT `--dry-run`; require
   `PASS op=... tolerance=1e-05` + `ssh_evidence=true`. Record `evidence.json` + remote stdout. The nibble
   unpack is correctness-load-bearing and, post-re-baseline, golden ONLY via this lamp.
7. **Delete the legacy path** (gated on step 6 green): remove the shared `RVVEmitCStatementPlanOwners.cpp`
   DirectContraction DEQUANT body-emission path (the `usesGroupedLowPrecisionProductReduction` /
   `usesPackedI4LowPrecisionProductReduction` branches + the dot_acc_vec / grouped-tail / nibble string
   synthesis + the 8 gate-skipped `wideningProductOp` derefs that only ran in the route build). KEEP
   `RVVEmitCContractionRouteFamilyPlanOwners.cpp` (route-family / provider-facts / diagnostic layer; 0
   statement-plan steps; all 6 prior owner retirements kept their `*RouteFamilyPlanOwners.cpp`). Re-green
   build + lit (honest-green).

---

## blast_radius

- **Analysis (bounded):** ~7 `wideningProductOp` deref sites in `RVVEmitCRoutePlanning.cpp`
  (`:16348,16371,16489,16499,16897,17266,34211`) + the family-plan owner sites
  (`RVVEmitCContractionRouteFamilyPlanOwners.cpp:9860/9863/9881/9892/9972/10164/10179`); one new dispatch arm +
  recorder + slot. The ~70 unrelated "exactly one" guards are untouched.
- **Metadata re-home (bounded, additive):** ~70 `handoff.get*()` reads redirected to `with_vl`; the realization
  stamping site already carries 3 `realized_*` facts there. Fact VALUES unchanged → PLAN/HEADER fixtures don't
  move.
- **Conversion (already built):** zero new work; the single-scope dequant body handler ships (commit 7bfd6012),
  rejects the handoff form by design (`RVVToEmitC.cpp:1454-1459`).
- **Realization (one file, bounded):** `RVVContractionSelectedBodyRealizationOwner.cpp ~:1996-2169`.
- **Fixtures — two classes, fork in step 1 (this sets the TRUE blast radius):**
  - *Realization-RUNNING fixtures* (input is a pre-realized/explicit selected body the owner re-emits): the 2
    packed-i4 body-C fixtures re-baseline; the explicit-realization fixture's `realized_*` attrs (line 39) and
    any structurally-asserted handoff/second-scope lines track the new with_vl-carried single-scope form — part
    of the bounded re-baseline, not new engine work.
  - *Frozen two-scope INPUT fixtures* (`pre-realized-selected-body-artifact-*` supplying a handoff/two-scope body
    as INPUT): the convert-gate makes these load-bearing. A frozen two-scope handoff input → the conversion
    rejects the handoff (`RVVToEmitC.cpp:1454-1459`) → `rvvSelectedBodyFullyConvertsToEmitC` returns false → the
    fixture stays on the LEGACY route, so its PLAN/HEADER never moves BUT it no longer exercises the new path; OR
    it goes STALE, asserting a realized form the owner no longer emits. Step 1 determines per-fixture which:
    rewrite the INPUT to single-scope to exercise the new path, or accept it as a legacy-route regression fixture.
    This fork is the difference between "2 body-C fixtures" and a larger re-baseline set — pin it before
    estimating cost.
  - ~23 handoff/dequant fixtures otherwise stable because fact values are preserved (subject to the vestigial
    producer/consumer-scope caveat above).
- **Owner deletion:** the shared `RVVEmitCStatementPlanOwners.cpp` DirectContraction dequant body path only;
  NOT the 11,745-line `RVVEmitCContractionRouteFamilyPlanOwners.cpp`.

---

## risks

1. **packed-i4 numeric correctness (top risk).** The nibble unpack is correctness-load-bearing and, after
   re-baseline, golden ONLY via the legacy emission until step 6. A wrong shift / sign-extend = silently wrong
   result. Mitigation: hard-gate legacy-path deletion on a real `ssh rvv` PASS (tolerance=1e-05,
   `ssh_evidence=true`); never assert from `--dry-run`.
2. **Allowed-read ledger drift / I5 regression.** The re-homed `with_vl` facts must stay REPORTING-only; if any
   future conversion read pulls a compute decision from them it re-creates the mirror coupling Stage-3 removes.
   Mitigation: assert in the structural lit test that the conversion derives compute from the typed ops only;
   keep the re-homed attrs evidence/reporting-only; the existing `RVVToEmitC.cpp:2564` no-read invariant is the
   guard.
3. **Fact-value drift on re-home.** Re-homing must preserve EXACT fact strings (same `selectedResourceCandidate`
   source) or PLAN/HEADER fixtures break and the re-baseline scope silently widens. Mitigation: step 4 byte-diffs
   PLAN/HEADER resource lines pre/post flip before touching any fixture.
4. **Analysis hatch over-narrowing.** Widening the hatch predicates risks either admitting an illegal body or
   still rejecting a legal nibble one. Mitigation: drive all 4 dequant fixtures through the analysis in step 4;
   keep the single-slot `arithmeticOp` invariant; the `productSlotResult` accessor centralizes the head choice.
5. **a2's null deref recurrence.** If any family-plan site is missed, the nibble body segfaults in the always-run
   analysis on the converting path. Mitigation: route ALL product derefs through `productSlotResult`; add a
   negative-ish lit asserting analysis success (no crash) on the nibble fixture.
6. **HW access flakiness.** `ssh rvv` (ProxyJump) can be down. Mitigation: gate deletion on real evidence; do not
   close the rung on a dry-run.
7. **Scope creep into Fable's broader ODS.** Fable's plan proposed a second `grouped_product_reduce_loop` ODS op;
   the as-built conversion handles multiple inline slices in one scope WITHOUT a new control op, so do NOT add it
   unless step 4 shows the grouped unroll-2 cannot reach emission parity as inline slices. Resist generalizing.
