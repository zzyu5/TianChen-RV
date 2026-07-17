# P-B7 STEP-1 ceiling — 2nd kernel family, hand-variant 3-way on real `ssh rvv`

- **What**: before any compiler change, hand-wrote the wide-LMUL deferred variant of two
  candidate 2nd families + a genuine-scalar + TWO competent-naive baselines, and measured the
  N-way on real `ssh rvv`. Decides whether the wide variant beats scalar AND a *competent* naive
  for the family (the discriminating bar — beating genuine scalar alone is automatic for any RVV).
- **Date**: 2026-06-15. **Board**: real riscv64 (`ssh rvv`), Ubuntu clang 18.1.3, isa
  `rv64imafdcv...zvfh` (probed this session, `remote_target_profile.txt`). VLEN=128 inherited from
  the P-B1 board profile (this session did not re-probe `/sys/.../vlen`; the LMUL-rung reasoning
  below assumes the same 128-bit board P-B1 measured). genuine-scalar TU compiled `-march=rv64gc`
  (no vector ISA); `objdump` of the scalar object on the board found ZERO vector ops → genuine.
- **Sources** (durable): `dot-reduce-i16/{ceiling.c,ceiling_rvv.c}` and
  `u8-product-reduce/{ceiling.c,ceiling_rvv.c}`. Each links all variants into ONE binary, times
  them interleaved on the SAME buffers, best-of-11×16, integer-EXACT correctness guard before
  timing (scalar oracle).

## The four columns (the honesty crux — naive baseline definition)

Beating genuine-scalar is automatic. The discriminating columns are the two naive RVV bars:
- **per-iter-reduce** = the algorithm the COMPILER EMITS TODAY for the dot-reduce family
  (`emitWideningDotReduce`: vwmul + a `vredsum` into a running seed EVERY iteration — the
  latency-bound cross-lane reduction on the critical path). The real shipping default.
- **narrow-deferred** = the "naive done right" at NARROW LMUL: one persistent vector accumulator
  (vadd.vv / vwaddu.wv), ONE trailing vredsum, but i16mf2 / u8mf4 source (under-vectorized on a
  128-bit board). The *competent* naive — the bar P-B1 warned is parity-vs-wide for the byte kernel.
- **wide-deferred** = the TUNED candidate: widest-legal source LMUL (i16m4 / u8m2), persistent
  i32m8 / u32m8 accumulator, ONE trailing vredsum.

## Results (best_per_iter_ns ratios; >1.0 = wide-deferred is faster)

### Family (a) signed i16 widening dot-reduce  `sum = acc + Σ lhs·rhs`

| n | wide vs scalar | wide vs per-iter (compiler default) | wide vs narrow-deferred (competent naive) |
|---|---|---|---|
| 257 | 3.96 | 6.16 | 2.99 |
| 256 | 5.38 | 7.85 | 3.12 |
| 1024 | 5.96 | 8.80 | 3.05 |
| 4096 | 6.31 | 9.35 | 3.18 |
| 16384 | 4.97 | 7.30 | 2.50 |
| 65536 | 3.66 | 5.41 | 2.01 |

→ **CLEAN WIN at every n against all three measured bars.** The win is TWO INDEPENDENT LEVERS
(don't conflate them — narrow-deferred and wide-deferred differ in LMUL ONLY; both use vadd.vv
deferred accumulate + one trailing vredsum):

- **The compiler's CURRENT dot-reduce emission is `per-iter-reduce`** (`emitWideningDotReduce`), and
  it is a **regression vs genuine scalar: 0.64–0.69×** at every n — same pathology class as the byte
  kernel's auto-selected grouped-u2 (a vector schedule that loses to scalar). Here the cause is
  different: a `vredsum` on the critical path EVERY iteration (latency-bound), not under-LMUL.
- **Lever 1 — deferred single-reduce** (per-iter-reduce → narrow-deferred): ~2.9–3.1× (n=4096:
  15879→5390 ns). Hoists the cross-lane reduction out of the loop into ONE trailing vredsum. Fixes
  the regression. This lever is ABSENT from the byte kernel's story (the byte default was already
  deferred; its pathology was under-LMUL).
- **Lever 2 — max-legal LMUL** (narrow-deferred → wide-deferred): ~2.0–3.2× (n=4096: 5390→1697 ns).
  The SAME LMUL-width effect P-B1 attributed the byte win to (mf2→m4 source). Pure width.
- **Headline honest win = vs the compiler's CURRENT emission (per-iter-reduce) = 5.4–9.4×** — the
  product of both levers, selector-driven once built.
- **Vs a competent *wide*-LMUL naive: PARITY by construction** (not a measured column) — wide-deferred
  IS that naive; there is no latency-hiding headroom beyond selecting the wide rung (consistent with
  P-B1). The N3 value is *selecting* the deferred + wide schedule the compiler currently does not.

Win compresses at 64KB (memory pressure rising: 2-byte operands) but persists strongly. NOT
bandwidth-saturated. The dot-reduce family's distinctness vs byte: its current default fails a
DIFFERENT way (latency-bound per-iter reduce), so resource-aware selection generalizes — different
pathology, different fix (two levers vs one), different op (vadd.vv vs vwadd.wv).

### Family (c) unsigned u8 widening product-reduce (insurance / comparative)

| n | wide vs scalar | wide vs per-iter | wide vs narrow-deferred |
|---|---|---|---|
| 257 | 4.57 | 8.11 | 3.69 |
| 4096 | 9.92 | 14.70 | 4.99 |
| 65536 | 7.89 | 11.68 | 3.94 |

→ Even larger margins (1-byte operands → less bandwidth pressure, two widening steps → more compute
to hide). A near-clone of the proven signed-i8 byte kernel (unsigned vwmulu/vwaddu).

## The pick: family (a) signed i16 widening dot-reduce

Both win, so the criterion is "most defensibly-distinct family + lowest-friction reuse":
- (a) is a **genuinely distinct family** (different typed op `tcrv_rvv.widening_dot_reduce`, dtype
  i16, two input buffers + scalar acc-seed ABI) → stronger N3 generalization evidence; (c) is a
  near-clone of the existing signed-i8 byte path (same shape, just unsigned).
- (a) needs a **NEW non-widening accumulate op** (i16 product is ALREADY i32 == acc width → the
  deferred accumulate is `vadd.vv`, NOT the byte path's widening `vwadd.wv` / `widening_accumulate`).
  So this is NOT clean reuse of the byte machinery — the task's "prefer (a) IF clean reuse" condition
  is only half-met; (a) is kept for distinctness + the two-lever N3 story, eyes open on friction.
- (c) u8 reuses `widening_accumulate` (vwaddu.wv) as-is and is a near-clone of the proven signed-i8
  byte path → it would likely reach full e2e faster, but it is weaker N3 evidence (replays the byte
  shape). Decision: build (a); keep (c) as durable comparative ceiling evidence.
- The dot-reduce front-end already exists (`widening_dot_reduce` op + binding + header ABI + route
  family), so the front half is partial reuse; the realization/conversion/bundle get a parallel
  wide-deferred path (like the byte kernel got at P-B3..P-B6).

Decision: extend the autotuner + emission to family (a). (c) stays as durable comparative evidence.

## Staged build plan (each stop reportable; do NOT force the e2e green)

- **2a (dialect + conversion), hand-fed IR → ssh-rvv numeric-correct.** New non-widening accumulate
  op (`vadd.vv` i32m8); i16m4→i32m8 widening-product predicate; new `isDeferredWideDotReduceBody`
  recognizer + emitter (NO dequant; store i32 lane0 via the existing standalone store). The
  P-B3-equivalent milestone — durable on its own. Narrow i16mf2 dot-reduce path byte-untouched.
- **2b (realization owner produces it selector-driven).** Wall risk: the owner validates dozens of
  stamped narrow facts.
- **2c (bundle export).** Highest friction.

Get 2a ssh-rvv-correct + durable before 2b. Report 灯 status precisely at whatever stage walls.
