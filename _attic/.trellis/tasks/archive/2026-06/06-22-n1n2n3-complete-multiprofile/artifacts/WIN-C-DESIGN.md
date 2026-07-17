# Win-C design + honest verdict (2026-06-24)

**Verdict: a genuine Win-C is buildable for MICRO only; its llama-e2e arm is architecturally BLOCKED.**

**The transform:** deferred-wide accumulation vs per-iteration reduction on the i16 dot-reduce (and i8
product-reduce-dequant) sub-op contraction bodies. OFF = per-strip `vredsum`-into-scalar (reduction on the
loop-carried dependency every iter, `RVVContractionSelectedBodyRealizationOwner.cpp:2548`). ON =
`widening_product`+`deferred_accumulate`(loop-carried wide vector acc)+`standalone_reduce`(ONE trailing
vredsum, `:2053`). Different op set, different dependency chain — genuinely STRUCTURAL, not a knob. Both
already emit + lower (op-identity-recognized: `isDeferredWideDotReduceBody`, RVVToEmitCInternal.h:99).

**Why it's not claimed yet (3 honest constraints):**
1. NOT a toggleable pass today — the structural choice is CONFLATED with the LMUL/budget (Win-A) axis in one
   `if` at `:2629`. Win-C work = SEPARATE the structural axis from LMUL, expose an independent on/off.
2. The headline 5.9–11.2× is CONTAMINATED (mixes defer + wide-LMUL). Honest Win-C = deferred÷per-iter at the
   SAME LMUL ≈ (5.9–11.2)÷(2.3–3.8) ≈ **1.5–3× expected, UNMEASURED**. This same-LMUL ablation is make-or-break:
   if marginal, the effect was Win-A in costume and there is NO Win-C. (Judged unlikely — per-iter vredsum on
   the loop-carried chain is a known latency bottleneck — but MUST be measured, not assumed.)
3. e2e BLOCKED: the only structure-exposed kernels (TypedWideningDotReduce…) are "N/A — not a llama hot-path
   kernel" (matrix §1b). Every llama-routed kernel (repack GEMV/GEMM, q4_0 block-dot) is a MONOLITHIC emitter
   op — reduction baked into C++ string emitters, not toggleable IR. e2e Win-C needs a large emitter refactor
   (out of scope; risks the q4_0 byte-exact e2e seals).

**Build plan (micro Win-C):** (1) stamp a `tcrv_rvv.reduction_structure ∈ {per_iteration, deferred_accumulate}`
body fact from a pass option (mirror the no-clobber autotuner stamps, RVVGearboxSchedules.cpp:148-181); (2)
split the realization-owner branch (`:2629`) to read it FIRST and emit the chosen structure AT THE OFF arm's
LMUL (so the ablation isolates structure from width); (3) lit test both arms at fixed LMUL; (4) rvv micro:
deferred÷per-iter at fixed LMUL. Both arms 100% tcrv-opt output (no hand kernel) → valid Win-C. I5: realize
into op identity, never a mirror string. I7: unstampable → fall closed to per_iteration. Effort ~1–3 days.

**R1 (do FIRST, cheap, disqualifying-if-marginal):** emit deferred + per-iteration at the SAME fixed LMUL,
compile, time on rvv. This is the make-or-break number. Files: RVVContractionSelectedBodyRealizationOwner.cpp
(:2053/:2548/:2629), RVVGearboxSchedules.cpp (:1227/:148), Passes.td (:756), RVVToEmitC.cpp (:1559/:523).

---

## RESULT — R1 make-or-break MEASURED (2026-06-24, ssh rvv SG2044 VLEN=128, clang18)

**VERDICT: Win-C (the pass ON/OFF) is REAL and reproduced at 3.0–3.3×.**
Well above the 1.5× gate → the pass ON/OFF effect is NOT Win-A in costume. Proceed to build the pass.

> **CORRECTION (see "3× DECOMPOSITION" section at the bottom of this file).** The phrase "Structural-only
> ratio = 3.0–3.3×" used below is SUPERSEDED. A later hand-written register-kept analysis kernel decomposed the
> 3× and found the PURE reduction-structure delta ≈ **1.00×** on SG2044 — the measured 3× is the per-iteration
> `out[0]` MEMORY round-trip in the compiler's OFF arm, NOT a cross-lane-reduction latency benefit. The pass
> ON/OFF 3× is still real (the OFF arm genuinely has the round-trip), but it must NOT be called "structural-only."
> Read the decomposition section before quoting any number from here as a "reduction-structure" benefit.

### How the same-LMUL ablation was obtained with ZERO code change (measurement-only)
The structural choice is conflated with the budget at `:2629`, BUT the budget crosses the SMALLEST deferred
rung's legality threshold at exactly the m1 boundary, which hands us both structures at the SAME LMUL:
- `enumerateRVVDotReduceDeferredWideLMULRungs` smallest rung = mf2→m1, cost = footprint(m1)=1 + reserve 8 = **9**.
- **budget 9** → mf2→m1 rung LEGAL → `selected` set → `realizeDeferredWideDotReduceBody` (DEFERRED, ON).
- **budget 8** → ALL rungs pruned → `selected==nullopt` → fall-through `:2548`
  `createRealizedGenericWideningDotReduceCompute` → per-iteration `WideningDotReduceOp` → `emitWideningDotReduce`
  (which hard-requires result LMUL m1) → PER-ITERATION (OFF).

Both are i16mf2 source / i32m1 product+result. **At VLEN=128, e16mf2 VLMAX=4 AND e32m1 VLMAX=4 (verified on
board) → both strip 4 elements/iter, identical loads (`vle16_v_i16mf2`), identical product (`vwmul_vv_i32m1`),
identical loop trip count.** The ONLY difference is the reduction structure (confirmed in compiled asm):
- DEFERRED: `vwmul` → `vadd.vv` (loop-carried i32m1 vector acc) per iter, then ONE trailing `vredsum`.
- PER-ITER: `vwmul` → `vredsum.vs` on the loop-carried SCALAR seed EVERY iter (0 `vadd.vv`).
So the ratio isolates the structural axis from the LMUL/budget (Win-A) axis. Both arms are 100% verbatim
tcrv-opt output (selector → `--tcrv-rvv-lower-to-emitc` → `mlir-translate --mlir-to-cpp`); no hand kernel.

### Numbers (best-of-11×16, taskset 1 core; back-to-back same-load; 2 runs)
| n | deferred_ns (ON) | periter_ns (OFF) | periter÷deferred |
|---|---|---|---|
| 256 | 263.75 | 798.75 | 3.03× |
| 1024 | 962.50 | 3126.25 | 3.25× |
| 4096 | 3755.00 | 12435.00 | 3.31× |
| 16384 | 15086.25 | 49841.25 | 3.30× |
| 65536 | 64042.50 | 200812.50 | 3.14× |

Run 2 reproduced 3.03–3.31× (absolute ns shifts with board load; ratio stable — the back-to-back same-load
ratio is the metric). Both arms NUMERICALLY EXACT vs the genuine-scalar oracle on every n, INCLUDING the odd
n=257/1000 that exercise the tail remainder (driver TU compiled -march=rv64gc, objdump vector-op count = 0).

### Honest framing
This is the structural axis cleanly isolated: per-iteration `vredsum` on the loop-carried reduction chain is
the latency bottleneck; deferring it to a loop-carried vector accumulator + one trailing reduce removes it —
at the SAME LMUL. This is DISTINCT from Win-A (the LMUL-width sweep, 2.27–3.79× wide÷narrow, also same
deferred algorithm). Win-C = the reduction-structure axis; Win-A = the LMUL-width axis. The contaminated
"6–11×" headline = Win-C × Win-A composed; factoring out the LMUL-width (Win-A) axis leaves the pass-ON/OFF reduction-structure number ~3× (see disclosure: bundles genuine structure + a co-occurring round-trip; NOT a pure-structure isolate).

### Honest disclosure (so it is not discovered)
The per-iteration fall-through (`:2548` → `emitWideningDotReduce`) round-trips the running scalar seed through
MEMORY every iteration: it reads `out[0]` (`vmv_v_x` splat), `vredsum`s into it, and stores `out[0]` back each
iter (confirmed in the emitted C / asm). So part of the 3× is that load/store round-trip on the loop-carried
dependency, NOT pure `vredsum` latency. This round-trip naturally ACCOMPANIES the per-iteration-to-scalar lowering the compiler emits today — it is
SEPARABLE in principle (a register-kept-scalar accumulator would avoid it), NOT fundamental to the structure.
So the 3× is a **pass-ON/OFF** number bundling TWO costs the deferred path removes: (a) the per-iteration
`vredsum` on the loop-carried reduction chain — GENUINE structure that survives a register-kept accumulator
(a low-latency `vadd.vv` chain vs a cross-lane-reduction chain on the critical path); and (b) the baseline's
scalar memory round-trip — a CO-OCCURRING emit cost of today's per-iter lowering. By the user's definition
Win-C IS pass-on/off, so 3× is the legitimate Win-C number and the verdict stands — but it must NOT be
reported as "pure reduction-structure benefit = 3×"; the isolated structural delta is smaller, and is
decomposed by a planned register-kept analysis kernel (hand-written, analysis-only — a footnote, NOT a Win-C
arm). The task defines OFF as the `:2548` fall-through, which is what we measured.

Artifacts: `winC-ablation/{deferred_body.cpp,periter_body.cpp,winc_driver.c,winc-rvv-stdout-run1.txt}`.

---

## RESULT — STEP 2 the minimal Win-C pass BUILT (2026-06-24)

The make-or-break (3×) cleared the gate, so the reduction-structure axis is now an INDEPENDENT,
compiler-toggleable knob, ORTHOGONAL to the LMUL/budget (Win-A) axis. Both arms emit at the SAME m1 LMUL from
the SAME pass with only the structure fact flipped.

### What the pass does
- New body fact `tcrv_rvv.low_precision_resource.reduction_structure ∈ {deferred_accumulate, per_iteration}`,
  stamped by the gearbox pass from a new `reduction-structure` pass option, gated to the mf2/m1 signed
  dot-reduce strip the selector serves (mirrors `materializeDeferredWideBudgetForDotReduceBody`).
- The realization owner reads the fact FIRST (before the budget rung logic, the now-separated structural
  axis): `per_iteration` -> the per-strip-vredsum family realizer (OFF); `deferred_accumulate` -> the deferred
  chain at the budget-selected LMUL, composing with Win-A, falling to the always-available minimal mf2->m1
  rung if the budget pruned every rung (ON). ABSENT fact -> falls closed (I7) to the existing budget-driven
  behavior with ZERO change.
- I5: the executable structure lives in the realized op identity (`deferred_accumulate` op vs the per-iteration
  `widening_dot_reduce` op), NEVER in a mirror string. The fact is a PRE-realization body attr (allowed on the
  pre-realized op, held OFF the `isRVVLowPrecisionResourceAttrName` copy allowlist), consumed and erased with
  the body at realization, so it can never survive onto a realized op.
- I7: an unstampable/unrecognized option value is a bounded pass error; an absent fact falls closed to
  per-iteration-or-existing-behavior (no silent wrong structure).

### Files changed
- `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h` — the `reduction_structure` attr name + the two value
  constants + `makeRVVDotReduceMinimalDeferredM1Rung()` (the always-available m1 deferred rung).
- `include/TianChenRV/Transforms/Passes.td` — `reduction-structure` Option on the gearbox pass.
- `lib/Plugin/RVV/RVVGearboxSchedules.cpp` — `materializeReductionStructureForDotReduceBody` stamp + threaded
  into the dot-reduce stamping loop from the pass option.
- `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp` — optional string-fact reader +
  the realization-owner branch split (structure axis read FIRST, orthogonal to budget).
- `lib/Dialect/RVV/IR/RVVDialect.cpp` — allow the pre-realization `reduction_structure` body fact on the
  dot-reduce pre-realized op verifier (like the budget fact; NOT a copyable low_precision_resource mirror).
- `test/Target/RVV/pre-realized-selected-body-realize-winc-reduction-structure-ablation.mlir` — new lit.

### Verification
- New lit PASSES: option stamps the fact (STAMP), both arms emit at fixed m1 with distinct op structure
  (DEFERRED has deferred_accumulate + ONE standalone_reduce, NO per-iteration widening_dot_reduce; PERITER has
  the per-iteration widening_dot_reduce, NO deferred_accumulate/standalone_reduce; EMITC arms confirm the
  intrinsics), and a bad option value is a bounded error (BADOPT).
- REGRESSION (the #1 risk): the existing `...deferred-wide-dot-reduce-autotuner-e2e.mlir` (the ABSENT-fact
  path) PASSES byte-identical — WIDE/EMITC/NARROW/NARROW-EMITC/NARROWEST-EMITC all green. The 3 plugin
  selection unit tests (incl. `rvv-low-precision-lmul-selection.test`) PASS.
- LOOP-CLOSING (decisive): the deferred/per-iter C emitted via the NEW reduction_structure pass path is
  STRUCTURALLY byte-identical to the budget-injected bodies measured at 3.0-3.3× (full intrinsic + loop-body
  diff = empty). So the pass exposes EXACTLY the measured ablation. Artifacts:
  `winC-ablation/{deferred_body_via_pass_fact.cpp,periter_body_via_pass_fact.cpp}`.
- I5 on the ACTIVE-COPY path: the per_iteration arm routes through `realizePreRealizedRVVSelectedContraction
  Family` which calls `copyLowPrecisionResourceAttrs` (:2219). With budget 9 + per_iteration injected (so the
  copy is actively moving attrs — budget mirrors through), the realized output carries ZERO `reduction_structure`
  strings: the copy allowlist (`isRVVLowPrecisionResourceAttrName`, which the fact is deliberately NOT in)
  filters it. The executable structure is in op identity (`widening_dot_reduce` for per-iter,
  `deferred_accumulate` for deferred), never a mirror string.
- ORTHOGONALITY proven (structure beats budget): per_iteration at budget 32 (which would otherwise pick the
  wide deferred m8 rung) still emits per-iter m1; deferred at budget 8 (all rungs pruned) falls to the minimal
  m1 deferred rung. So the structure axis is genuinely independent of the LMUL/budget (Win-A) axis.
- I7 hardening: an unrecognized value fails closed at BOTH entry points — the pass option (BADOPT) and a
  directly-injected body fact (BADFACT, a bounded realizer diagnostic; no silent fall-through to budget logic).
- FULL SUITE: 692/695 `check-tianchenrv` pass; the 3 failures are PRE-EXISTING (verified identical on clean
  main HEAD with no Win-C changes): 2× computed-masked-strided-dot (a different, untouched kernel family) +
  1× product-reduction-dequant self-test (different family). Win-C adds ZERO new failures.

---

## RESULT — 3× DECOMPOSITION (analysis kernel) (2026-06-24, ssh rvv SG2044 VLEN=128, clang18)

**HEADLINE: the pass-ON/OFF ~3× is the per-iteration `out[0]` MEMORY round-trip, NOT a reduction-structure
latency benefit. Pure reduction-structure delta on SG2044 = ~1.00×.**

This decomposes the bundled pass-ON/OFF 3.0–3.3× (the R1 number above) into a GENUINE reduction-structure
component vs a CO-OCCURRING per-iteration scalar memory round-trip, using a third HAND-WRITTEN body. This is an
ANALYSIS kernel for mechanism attribution — **a footnote, NOT a third Win-C arm.** Win-C stays the compiler's
ON/OFF (the bundled 3×); the register-kept body is not compiler-emitted, so it cannot be a Win-C arm.

### The three bodies (all FIXED m1 LMUL, same e32m1 strip vl=4 @ VLEN128, same i16mf2 loads + vwmul_vv_i32m1)
| body | how obtained | reduction structure | loop-carried acc | in-loop out[0] round-trip |
|---|---|---|---|---|
| **deferred** (ON) | PASS-emitted (budget 9) | deferred vector-accumulate + ONE trailing vredsum | `vadd.vv v9,v9,v12` (vector reg) | none |
| **periter** (OFF) | PASS-emitted (budget 8 fall-through `:2548`) | per-iter vredsum every iter | `vredsum.vs` via `out[0]` MEM | YES (`vle32.v`+`vse32.v` every iter) |
| **regkept** | HAND-WRITTEN analysis kernel | per-iter vredsum every iter | `vredsum.vs v8,v11,v8` (vector reg) | NONE |

`regkept` = `periter` MINUS the round-trip: identical per-iteration `vredsum.vs` STRUCTURE, but the running
scalar accumulator is KEPT IN A REGISTER (`v8`) across iterations — seeded once from `acc[0]`, stored once
after the loop. Verified in objdump (`periter_regkept_body.o`): the inner loop `.LBB0_2` contains ONLY
`vsetvli / vle16.v ×2 / vwmul.vv / vsetvli / vredsum.vs v8,v11,v8` and `add/sub/bltu` — **NO `vle32.v` reload
of `out[0]`, NO `vse32.v` store of `out[0]` inside the loop** (the single `vse32.v` is at `.LBB0_3`, after the
loop). Contrast `periter_body.o` `.LBB0_2`, which DOES carry `vle32.v v8,(a3)` + `vse32.v v8,(a3)` every iter.
So `regkept` faithfully carries the loop-carried reduction dependency THROUGH `vredsum.vs` on the critical path
(v8→vredsum→v8) — exactly the "per-iter reduction on the critical path" the structural hypothesis was about —
while removing only the memory traffic.

### Numbers (best-of-11×16, `taskset -c 0`, all THREE timed back-to-back same-load in one interleaved rep loop; 2 runs)
| n | deferred_ns (ON) | periter_ns (OFF) | regkept_ns | bundled `periter/deferred` | **pure-structure `regkept/deferred`** | round-trip `periter/regkept` |
|---|---|---|---|---|---|---|
| 256 | 402.50 | 1223.75 | 401.25 | 3.04× | **1.00×** | 3.05× |
| 1024 | 1473.75 | 4795.00 | 1472.50 | 3.25× | **1.00×** | 3.26× |
| 4096 | 5766.25 | 19085.00 | 5763.75 | 3.31× | **1.00×** | 3.31× |
| 16384 | 23072.50 | 76502.50 | 23082.50 | 3.32× | **1.00×** | 3.31× |
| 65536 | 100205.00 | 308162.50 | 100196.25 | 3.08× | **1.00×** | 3.08× |

Run 2 reproduced this tightly: pure-structure across BOTH runs = 0.994–1.005× (`regkept` and `deferred` are
within ≤0.6%, i.e. measurement noise); bundled 3.04–3.32×; round-trip 3.04–3.32×. All three arms NUMERICALLY
EXACT vs the genuine-scalar oracle on every n INCLUDING the odd tails n=257/1000 (driver TU `-march=rv64gc`,
objdump vector-op count = 0 → oracle is genuinely scalar). The bundled 3× reproduces the R1 number, proving the
harness resolves real per-kernel cost differences (the thin noinline wrapper overhead is common-mode and
cancels in the ratios).

### Interpretation (BOTH halves are true; state both)
- **(a) The pass-ON/OFF 3× is REAL and reproduced.** By the user's definition Win-C IS the compiler's ON/OFF,
  and the compiler's OFF arm (the `:2548` fall-through) genuinely emits the per-iteration `out[0]` memory
  round-trip. So 3.0–3.3× is the legitimate Win-C pass-ON/OFF number and the R1 verdict (proceed to build the
  pass) stands.
- **(b) The PURE reduction-structure latency benefit is ≈0 on SG2044.** Once the memory round-trip is removed
  (register-kept scalar acc), a per-iteration `vredsum.vs` on the loop-carried critical path is as fast as the
  deferred `vadd.vv`-chain + one trailing reduce (pure-structure ≈ 1.00×). So the measured 3× is ENTIRELY the
  `out[0]` round-trip, NOT cross-lane-reduction latency.

**Mechanism reattribution:** the deferred structure's win comes from the compiler emitting it WITHOUT the
per-iteration memory round-trip (the round-trip is a CO-OCCURRING emit cost of today's per-iter-to-scalar
lowering, not fundamental to the structure). On this microarchitecture the cross-lane reduction itself is not
on the binding critical path. The decomposition identity `bundled ≡ pure-structure × round-trip` is ALGEBRAIC
(`periter/deferred ≡ (periter/regkept)·(regkept/deferred)`), so it is NOT independent corroboration — the
validations are the objdump (round-trip present in periter, absent in regkept, no spill) and the bundled-3×
reproduction.

### Honest framing + scope
This SUPERSEDES the "structural-only = 3.0–3.3×" language in the R1 section above (a forward-pointer is dropped
there). The R1 gate language ("≈1.0× WOULD weaken the structural claim and must be reported, not buried") is
exactly the case here, and it is reported in full: the structural component is ~1.0×, so the 3× should be
described as a **pass-ON/OFF round-trip-elimination** win, not a "pure reduction-structure" win. The
register-kept per-iter body is HAND-WRITTEN (not compiler-emitted) → it is an analysis kernel for mechanism
attribution only, a footnote, NOT a Win-C arm. SCOPE: this is "on SG2044 VLEN128 for this streaming
i16 dot-reduce, per-iter `vredsum.vs` is not on the binding critical path" — NOT "vredsum never bottlenecks."
We did not isolate the microarchitectural cause (vsetvli toggles vs load-use latency vs VLMAX=4 making the
cross-lane reduce cheap), and that is out of scope.

Artifacts: `winC-ablation/{periter_regkept_body.cpp (new), winc_driver.c (now 3-arm), winc-decomp-run1.txt,
winc-decomp-run2.txt}`. The pass-emitted `deferred_body.cpp`/`periter_body.cpp` are unchanged.
