# Item ② — budget-divergence: the honest, code-grounded finding

**Question (user):** make budget-divergence *real* — let genuine register-pressure /
live-vector-group counts push the budget down so the **narrow** arm fires on a *real*
pressured kernel, NOT synthetic injection.

**Answer:** On a single RVV chip this is **architecturally impossible without synthetic
injection** — and faking it would be a weaker paper position than the honest
characterization below. The resource-aware pruning that *does* fire on every real RVV
selection is the **m8 EMUL cap** (+ the VLEN/Zvl capability strip), not the vreg budget.

## Proof the budget axis is dormant on RVV (4× headroom)

The narrow arm gate (`RVVGearboxSchedules.cpp:1674`):
```
handoff.getPeakLiveVectorGroups() > handoff.getVectorRegisterBudget()   // narrow fires
```

- **Budget** is uniformly **32** for *every* quant shape — `kRVVQ40/Q41/Q50/Q51/Q80
  ShapeVectorRegisterBudget = 32` (`RVVGearboxSchedule.h:2123/2681/2743/2805/2867`),
  default `kRVVLowPrecisionResourceVectorRegisterBudget = 32` (line 752). This is the
  architectural RVV vreg-file fact: every RVV profile (incl. zve32x, RVV0.7) has 32
  vector registers — there is no RVV chip with fewer.
- **Real `peakLiveVectorGroups`** for *every* in-scope gearbox candidate
  (`getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups`, `RVVGearboxSchedule.h:1254`):
  | candidate | peakLiveVectorGroups | vs budget 32 |
  |---|---|---|
  | default contraction | **4** (line 742) | 8× headroom |
  | packed-i4 | **5** (line 750) | 6.4× |
  | grouped | **7** (line 746) | 4.6× |
  | composite gather-macc-scatter | **8** (line 1701) | 4× |

  The widest real candidate (8) is **4× under** the budget. `peakLive > 32` cannot be
  reached by any real kernel — the narrow arm is **unreachable** except by an op-attr
  that synthetically lowers the budget below the footprint
  (`resolveGearboxVectorRegisterBudget` reads `..VectorRegisterBudgetAttrName` if present
  — that override IS the synthetic injection the user rejected).

## The resource axis that IS live: the m8 EMUL cap

The rung enumeration (`enumerateRVVLowPrecisionAccumulatorLMULRungs`,
`RVVGearboxSchedule.h:1964–1992`) walks source LMUL `{mf4, mf2, m1, m2}`. Each widens to
an i16 product (EMUL 2×) and an i32 accumulator (EMUL 4×). **m4 source is excluded by the
EMUL cap, not the budget** (lines 1968–1979):
```
// m2 is the widest whose i32/m8 accumulator exists
// (m4 source would need an i32/m16 accumulator, beyond the m8 cap).
accumulatorLMUL = getRVVNextWiderLMUL(productLMUL);
if (accumulatorLMUL.empty())
    continue; // i32 accumulator would exceed the m8 LMUL cap -> not a rung.
```
For the widest legal rung m2→i16m4→i32m8: footprint = acc(m8)=8 + product(m4)=4 +
reserve ≈ **12** ≪ budget 32. So `isLegal = totalRegisterCost <= budget` (line 1990) is
**true for every rung regardless of the budget** — the budget never prunes; the **EMUL
cap is what bounds the selection at m2**. That is the real, always-firing resource
constraint behind the N3 "resource-aware" claim.

## Honest position (no synthetic injection)

| axis | status on RVV | why |
|---|---|---|
| **m8 EMUL cap** | **LIVE** — fires on every selection | excludes m4 source (i32m16 absent); fixes m2→i16m4→i32m8 as widest legal rung |
| **VLEN / Zvl128b capability** | **LIVE** — structural | VLEN=128 → e16m1 holds 8 i16 lanes → 16-block group splits into two 8-lane halves (the strip we emit) |
| **vreg budget (32)** | **DORMANT on RVV** | every real footprint (≤12) is ≥2.6× under 32; binds only under op-attr injection |

**Budget divergence is a cross-profile phenomenon, not a single-RVV-chip one.** It becomes
real only on a family whose vreg file is < the kernel footprint — which no RVV profile is
(RVV mandates 32). So the budget axis is correctly *modeled* (self-consistent, stamped,
verifier-checked) but its *divergence* belongs to **④ (a non-RVV second family with a
smaller/typed register file)** or a deliberately constrained profile — NOT a faked
single-chip ablation. Headlining the EMUL cap (real) and honestly labelling budget
dormant-on-RVV is the stronger, reviewer-proof N3 position (per advisor).

## What this means for the campaign

- ② is **closed as an honest characterization**, not a code change that forces the narrow
  arm. The deliverable is this finding + keeping the resource-aware narrative pinned to
  the EMUL cap + VLEN capability.
- The budget-narrow ablation stays **parked/dormant** (not headlined), with the
  cross-family realization deferred to ④. Surface this to the user — it contradicts the
  literal "make budget real" wish, and the honest reason matters.
