# Forward-pass operators — micro (OUR emit vs ggml) — FINDING

**Date:** 2026-06-25
**Board:** `ssh rvv` (Sophgo SG2044, RVV1.0, VLEN128, 64 cores, zvfh). Serial,
rvv-only (k1 untouched per discipline). `taskset -c 0`, best-of-reps **min** latency
(noise only ADDS time → lowest observed is closest to truth).
**Tooling:** existing `build/bin/tcrv-opt` (built 06-25 21:52) — **NO rebuild, no lib/
change.** Emit host-side: `tcrv-opt <test>.mlir --tcrv-rvv-lower-to-emitc |
/usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp`. Compiled on board with
`clang++-18 -O3 -march=rv64gcv_zvfh -mabi=lp64d` (`-ffp-contract=fast`; rope `=off`).
**Byte-exact correctness gate vs ggml PASSED before any perf reported, all 6.**

Fills Table 3 `micro` column (was 6/6 empty). The 6 ops are f32 forward-pass ops
(same-algorithm rewrites of ggml), so the expectation up front was **parity/TIE** for
the ops where ggml ships a hand-written RVV path, and a possible win only where ggml
has no RVV path. Verified each.

## Fresh-emit provenance
All 6 kernels re-emitted with the CURRENT tcrv-opt and confirmed **byte-identical** to
the archived (already-validated) inc15–inc20 emit (grep-diff, comments stripped) — so
the prior ssh-rvv byte-exact correctness proofs apply to exactly this code. The new
deliverable here is the micro ratio; the live byte-exact gate was still re-run fresh on
the board for every op.

## Results table

| op | LMUL (ours) | ggml competitor | gate | ratio (ggml/ours) | verdict |
|---|---|---|---|---|---|
| scale | m8 | ggml RVV `ggml_vec_scale_f32` (m8, vec.h:733) | byte-exact PASS | **1.006×** | **TIE** |
| silu | m2 | ggml RVV `ggml_vec_silu_f32` + `ggml_v_expf_m2` (vec.cpp/vec.h) | byte-exact PASS | **≈0.84×** (floor) | **LOSS** (root-caused) |
| rms_norm | m8 norm + scalar-f64 sum | ggml row body (ops.cpp:3791, **scalar**, no RVV path) | byte-exact PASS | **1.001×** | **TIE** |
| rope (norm) | m1 / scalar libm | ggml NORMAL rope (ops.cpp, **scalar** cosf/sinf) | byte-exact PASS | **0.964×** | **TIE** |
| quantize_q8_0 | m8 | ggml RVV `quantize_row_q8_0` (riscv/quants.c:32, m8) | byte-exact PASS | **0.986×** | **TIE** |
| softmax | m2 | ggml RVV `ggml_vec_soft_max_f32` + `ggml_v_expf_m2` (vec.cpp:584) | byte-exact PASS (y + f64 sum) | **0.854×** | **LOSS** (root-caused) |

**Headline: 4 TIE (scale 1.006×, rms_norm 1.001×, rope 0.964×, quantize 0.986×) +
2 LOSS (silu 0.840×, softmax 0.854×) — all 6 byte-exact. 0 win.**

## Stability / floor (min-of-reps)
- scale 1.006×, rms_norm 1.001×, quantize 0.986× — single clean run, ratios in the
  ±1-2% TIE band (same m8 path as ggml, must be parity).
- rope 0.964× — scalar-libm-dominated; both arms link the same libm; TIE band.
- **silu 0.840×** — 3 consecutive clean runs OURS≈31940 ns / GGML≈26840 ns → 0.840-0.842×.
  (One transient 0.347× was a single GGML-min noise spike paired with a load-spiked OURS;
  discarded — the stable floor is 0.840×.)
- **softmax 0.854×** — 3 consecutive clean runs 0.854-0.856×, very stable.

## The 2 LOSSes are ROOT-CAUSED (same gap, not noise, not algorithm choice)
silu and softmax both call the vectorized exp polynomial `ggml_v_expf_m2`. ggml's
polynomial has a **data-dependent short-circuit**: `if (!__riscv_vcpop_m_b16(c, vl))
return <fast path>;` — it skips the expensive slow-path overflow/underflow `vmerge`
fixup chain when **no lane** has |n|>126 (the common case for benign attention inputs).

**OUR emitted kernel has ZERO `vcpop`** (grep-confirmed: `silu_kernel.cpp` and
`soft_max_kernel.cpp` both contain 0 `vcpop`, while ggml's ref has 1) — our lowering
emits the slow-path merges **UNCONDITIONALLY** (this is even stated in the committed
test: "our emitted kernel takes the unconditional slow path"). So on benign inputs ggml
branches around the fixup and we always pay for it → the 0.84-0.85× gap.

This is **byte-exact-correct** (the unconditional path produces identical bits) and is a
**real emitter-maturity gap**, NOT an algorithm difference and NOT a tuning knob:
*we are missing the data-dependent branch elision (`vcpop`-gated fast path)* that ggml
hand-wrote into its exp polynomial. **Named emitter target: emit the `vcpop_m`-gated
fast-path short-circuit for the shared `ggml_v_expf_m2` polynomial** (would also lift any
other op routed through it). This is the same category as the block-dot maturity gaps
(correct code, missing a hand-written hardware optimization).

**Two independent confirmations the LOSS is structural, not a noise/measurement artifact
(the "is your noisy silu number real?" objection):**
1. **No measurement-order bias.** scale (1.006×) and rms_norm (1.001×) were timed in the
   SAME OURS-first / GGML-second order as silu/softmax, yet show zero second-arm
   advantage — so the silu/softmax gap is not a warmup/ordering artifact, it's structural
   (the grep-confirmed missing `vcpop` path).
2. **This is the WORST-CASE gap for these ops.** The harness inputs are benign
   (silu x∈[-50,50] → |n|≈72 < 126; softmax similar), so **no lane overflows** → ggml
   ALWAYS takes its cheap `vcpop`-gated fast path while we ALWAYS pay the slow merge.
   0.84/0.85× is therefore the *largest* gap this op can show; with overflow-forcing
   inputs both sides take the slow path and the ratio closes toward TIE. (Not re-run —
   the grep + no-order-bias evidence already satisfies "大偏差要查"; overflow-input
   re-run is the optional definitive confirmation, not needed for the claim.)

## The rms_norm TIE is the honest surprise (predicted win did NOT materialize)
ggml's rms_norm has **no RVV path** — its source is a scalar `ggml_float`(double) loop.
Naive expectation: our vectorized (m8) emit should WIN. It landed at **1.001× TIE**.
Why (verified): (1) `clang++-18 -O3 -march=rv64gcv` did **NOT** auto-vectorize ggml's
scalar ref (asm grep: 0 vector ops in the `ggml_ref` region) — the double-sum reduction
can't reassoc under fp rules, and clang left the whole row scalar. (2) Yet our emit is
still no faster, because rms_norm is **reduction-bound**: the dominant cost is the serial
`sum += (double)(x[i]*x[i])` fold, which OUR emit **also** does scalar-double (required
for byte-exactness — same crux the inc16 correctness proof pinned). The vectorized
normalize (`y[i]=x[i]*scale`) is a small tail. So vectorizing rms_norm gives no edge: the
serial f64 reduction dominates on both sides. Honest verdict: TIE, and a vectorized
rms_norm is **not** a win here — recorded against the temptation to claim "we vectorized
what ggml left scalar."

## What each ratio MEANS (framing, per project honesty discipline)
- The 4 TIEs are the **correct, expected** outcome of a same-algorithm rewrite against a
  matched-LMUL ggml RVV path (scale/quantize) or a reduction-bound/scalar-bound op
  (rms_norm/rope). They are coverage parity, **not** optimization wins and **not** Win-A
  knobs (these ops have no LMUL knob — N/A, as the table already marks).
- The 2 LOSSes are honest maturity gaps with a single named cause (no `vcpop` fast path).

## Not measured / scope
- **e2e** for all 6: still empty (out of scope here — this task is micro only). They are
  not wired into a standalone llama forward-pass driver.
- No Win-A / Win-B for any (no LMUL knob, same algorithm as ggml — N/A, unchanged).

## Artifacts
- `forward-pass-micro/bench_{scale,silu,softmax,rmsnorm,rope,quantize}.cpp` — harnesses
  (each embeds ggml's verbatim competitor + byte-exact gate + min-of-reps timing).
- `forward-pass-micro/{scale,silu,soft_max,rms_norm,rope_norm,quantize_q8_0}_kernel.cpp`
  — fresh compiler-emitted kernels (byte-identical to archived inc15–inc20 emit).
- On board: `~/fwd-micro/` (sources + binaries `b_{scale,silu,softmax,rms,rope,quant}`).
