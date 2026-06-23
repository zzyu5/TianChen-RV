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
