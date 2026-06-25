# FINDING — q4_K block-dot Win-A LMUL (m1 vs mf2) e2e TRANSPORT test (2026-06-25)

> **The question (user-demanded measurement, NOT a reasoned-NULL):** does the q4_K
> block-dot Win-A LMUL micro win (m1 vs mf2; m1 narrows the loss to ggml's `_vl256`
> from 1.80×→1.38× at VLEN256, ~1.30× kernel speedup — wide-lmul-kquant-FINDING)
> TRANSPORT to llama.cpp e2e, or wash out at the memory wall? **Measure BOTH regimes**
> (decode=tg/memory-bound, prefill=pp/compute-bound) so we see WHERE compute helps.
> Prior [[kernel-wins-dont-transplant-to-e2e]]: compute wins wash at memory-bound
> decode, transport at compute-bound prefill — the HYPOTHESIS. Let data decide.

---

## TL;DR — VERDICT (scoped, honestly NOT a refutation)

On **tinyllama-1.1B-Q4_K_M @ k1 VLEN256, t4** (repack-OFF vec_dot regime), the ~1.30×
micro Win-A **transports to BOTH prefill (1.25×) AND decode (1.21×)**. Decode is NOT
flat. **BUT this does NOT refute the memory-wall reasoned-NULL** — the diagnostic is
decode-ratio≈prefill-ratio≈micro-ratio: a real memory wall would crush the decode ratio
toward 1.0×, and it did not. The decode behaves like prefill ⇒ this decode is itself
**compute-bound at k1-t4**. Mechanism: **NOT cache-residency** (the 636 MB weights ≫ any
X60 cache and stream from RAM every token — an earlier "cache-resident" guess was WRONG
and is struck); it is **thread count** — k1-t4 sits BELOW the X60 bandwidth-saturation
point, so too few cores pull memory for decode to be bandwidth-limited (a measured
thread sweep below: tg64 t1→t4 scales 2.93×, sub-linear ⇒ t4 is only PARTIALLY
saturated; q4_K's heavy per-block unpack arithmetic-intensity may also raise the
compute floor — not fully pinned). A compute win surfacing in a compute-bound decode is
**CONSISTENT with** [[kernel-wins-dont-transplant-to-e2e]], not a counterexample. **The
genuine memory-bound-decode test — an 8B model (weights ≫ cache) at HIGH thread count
(the prior wall finding used t16/64-core, not t4) — remains the pending decider**, staged
on rvv (8B Q4_K_M) but blocked this session by repeated rvv "No route to host" drops.
Two honest caveats: (1) the win exists ONLY in the **repack-DISABLED** fallback; in the
SHIPPING default (q4_K repack ON @ VLEN256) our vec_dot is **bypassed entirely** → zero
e2e effect; (2) Q4_K_M is a MIXED quant so the ratio is diluted by non-q4_K tensors (a
modest ratio is still transport, magnitude-capped).

---

## METHOD — .inc-swap of OUR emitted q4_K vec_dot into ggml (mirrors the M2 pattern)

- OUR q4_K block-dot emitted at **m1** and **mf2** (both byte-exact-correct, oracle
  PASS 2012/0 at BOTH VLENs — wide-lmul-kquant-FINDING). Kernels:
  `kq-bench/q4_K/ours-{m1,mf2}.cpp` (md5 m1 `2eee81f9` / mf2 `2aea89b4`). Swap incs
  `qk-e2e-swap/tcrv_q4k_swap_{m1,mf2}.inc` (md5 `8da97c54`/`b22e3fc0`).
  **Source-level discriminator: m1 = `vwmacc_vx_i32m4` ×4 + 2 `vslidedown` fold;
  mf2 = `vwmacc_vx_i32m2` ×8, ZERO `vslidedown`.**
- ABI shim (ggml 8-arg `ggml_vec_dot_q4_K_q8_K(n,s,bs,vx,bx,vy,by,nrc)` → emitted
  4-arg `(n,s,vx,vy)`; bs/bx/by dropped, nrc==1) `#include`d into the per-board vec_dot
  body, with a once-guard `fprintf` engagement print: **vl256 @k1** (the §9 VLEN256
  headline), vl128 @rvv. In-session A/B: build m1, bench; swap mf2, incremental rebuild
  (ggml-cpu only), bench. Same binary structure; only the `.inc` (LMUL) differs.
- block_q4_K=144B / block_q8_K=292B confirmed on board == emitted strides (×144/×292).

### CRITICAL ROUTING FINDING — ggml REPACKS q4_K at VLEN256, BYPASSING vec_dot

At first swap the engagement print fired **0×** in BOTH pp and tg on k1. Root cause
(primary source `repack.cpp:4620`): at **VLEN256** ggml registers `q4_K_16x1_q8_K`
repack tensor_traits (`case 256: if ne[1]%16==0 return &q4_K_16x1_q8_K`) → q4_K weight
tensors are **repacked to `block_q4_Kx16` at LOAD** and routed through
`ggml_gemm/gemv_q4_K_16x1_q8_K`, **completely bypassing** `ggml_vec_dot_q4_K_q8_K_vl256`
where our swap lives. (The q4_0 M2 lesson, applied to q4_K.) To exercise our vec_dot at
all, the q4_K repack MUST be disabled (`repack.cpp` q4_K `case 256: break`). With that,
the print fires 4× in **pp-only AND tg-only** → our kernel is engaged in BOTH phases.
NB: at **VLEN128 (rvv)** ggml's q4_K repack is already `case 128: break` (disabled,
TODO) → rvv's vec_dot swap engages WITHOUT this change, AND vec_dot is the SHIPPING
path there.

**Honest consequence:** the k1 A/B is the **vec_dot (per-row) regime with q4_K repack
OFF**. These absolutes are LOWER than the default repacked path (k1 default repacked
pp256≈14.5 / tg64≈7.2 vs vec-dot-repack-OFF pp256≈4.5 / tg64≈3.6). The A/B isolates the
**vec_dot LMUL knob** (m1 vs mf2) = exactly the Win-A axis; it is NOT a claim that
vec-dot beats the repack GEMM (it doesn't — the repack is ~3× faster).

---

## RESULTS

### k1 — SpacemiT X60, VLEN256 (VLENB=8 → 256-bit), taskset -c 0-3, tinyllama-1.1B-Q4_K_M

**Model:** `llama 1B Q4_K - Medium`, 636 MiB, 1.10B params. Q4_K_M = MIXED quant
(majority Q4_K + some Q6_K/Q5_K) → e2e ratio DILUTED by non-q4_K weights.

**Engagement — objdump (the load-bearing proof the two arms are genuinely different):**
| arm | once-guard | e32 LMUL configs in vl256 | vslidedown (fold) | .so |
|---|---|---|---|---|
| **m1**  | `tcrv_q4k_once_m1`  | `e32,m4` present (widened i32m4 aux32) | **1** | 924864 @14:31 |
| **mf2** | `tcrv_q4k_once_mf2` | only `e32,m2` | **0** | 924872 @14:09 |

Runtime print `TCRV Q4_K BLOCK-DOT ENGAGED (LMUL=…)` fires in BOTH pp-only and tg-only,
both arms (q4_K repack disabled).

**A/B — matched, both `-p 256 -n 64 -t 4 -r 3`, q4_K repack OFF:**

| arm (repack OFF) | pp256 (prefill) t/s | tg64 (decode) t/s |
|---|---|---|
| **m1**  | **4.45 ± 0.01** | **3.63 ± 0.00** |
| **mf2** | **3.55 ± 0.00** | **2.99 ± 0.00** |
| **m1/mf2 ratio** | **1.25×** | **1.21×** |

(m1 r3 reproduces the r1 probe 4.46/3.65 → 4.45/3.63 — stable, not noise.)

**Coherence — BOTH arms:** **m1** (the riskier arm — it carries the novel `vslidedown`
fold-back) was checked LIVE: `llama-cli -p "The capital of France is" -st` → generates
**"The capital of France is Paris."** while the engagement print fires → the
emitted-kernel-via-ABI-shim is numerically CORRECT on real weights. **mf2** correctness
rests on body byte-identity + oracle: the mf2 **kernel body** (the emitted intrinsics) is
**byte-identical to the committed archive `ours-mf2.cpp` (md5 `2aea89b4`)** — verified by
diffing the swap-inc body against the archive (the whole-file md5 differs ONLY by the
function-name rename `tcrv_emitc_q4k_kernel` + the appended ABI-shim/once-guard wrapper;
the 407-line intrinsic body is line-for-line identical) — and the archive is oracle-verified
**bit-exact 2012/0 at BOTH VLENs** (wide-lmul-kquant-FINDING) — so the "both arms coherent"
bar is met (m1 by live "Paris", mf2 by body-identity to the proven-correct archive kernel).
(The m1 swap-inc body is likewise verified line-for-line identical to `ours-m1.cpp`.)

**Decode thread-scaling (m1 build, characterizes the regime — WHY decode isn't flat):**
`-p 0 -n 64 -t {1,4} -r 2`, taskset -c 0-7:

| threads | tg64 t/s | scaling vs t1 |
|---|---|---|
| t1 | 0.96 | 1.00× |
| t4 | 2.81 | **2.93×** (of 4.0× ideal) |

> NB: this sweep's t4 tg64=2.81 is a SEPARATE run from the A/B table's t4 tg64 m1=3.63 —
> different invocation (sweep: `-c 0-7 -p 0 -n 64 -r 2`; A/B: `-c 0-3 -p 256 -n 64 -r 3`),
> so the absolutes are not directly comparable. The sweep characterizes scaling only; the
> A/B ratio is the load-bearing number.

t4 scales only 2.93× (sub-linear) ⇒ k1-t4 is **partially bandwidth-saturated already,
yet m1 STILL wins decode 1.21× there** — a modestly PRO-transport data point (the compute
knob helps even under partial memory pressure). It is NOT full saturation (the prior wall
used t16 on a 64-core box); a t8 run hung on the constrained 8-core box and is not
reported. So k1 cannot reach the fully-bandwidth-bound regime — that needs the rvv 8B@t16.

**Shipping-default caveat (q4_K repack ON, the real-world config):** with repack ON the
default pp256≈14.5 / tg64≈7.2 are produced by the repack GEMM/GEMV — our vec_dot LMUL
knob is **bypassed → ZERO e2e effect in the shipping config**. The 1.21–1.25× A/B is a
property of the **repack-OFF fallback regime only.**

### rvv — SG2044, VLEN128, 8B Q4_K_M — THE MEMORY-BOUND DECIDER, **NOT COMPLETED**

m1 swap into vl128 was **built + objdump-confirmed engaged** (emitted symbol
`tcrv_emitc_q4k_kernel` present; vl128 body shows `e32,m4` + `vslidedown` = m1) BEFORE
the board became unreachable. At VLEN128 the q4_K repack is already OFF, so vec_dot is
the SHIPPING path AND an 8B model's weights exceed cache → decode is genuinely
memory-bound. **This is the test that would actually resolve the memory-wall question.**
BLOCKED: repeated `ssh rvv` "No route to host / Connection closed" drops (transient
network) prevented the coherence + A/B bench. **PENDING** — run `-t 16` (saturates
bandwidth, the regime the prior memory-wall claim used) and `-t 1`, both pp/tg, m1 vs
mf2, on the 8B `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`.

---

## HONEST VERDICT — does the compute-side LMUL knob TRANSPORT to e2e?

**On 1.1B @ VLEN256, k1-t4 (repack-OFF vec_dot): YES, to BOTH prefill (1.25×) and decode
(1.21×) — but this is NOT a refutation of the memory-wall reasoned-NULL.** The ~1.30×
micro Win-A transports nearly un-attenuated to decode *because at k1-t4 the decode is
itself compute-bound* — diagnosed by the decode-ratio ≈ prefill-ratio ≈ micro-ratio
equality (a real memory wall would crush the decode ratio toward 1.0×, which it did NOT)
and corroborated by the thread sweep (t1→t4 scales 2.93×, sub-linear ⇒ only partial
bandwidth saturation at t4). The mechanism is **thread count, not cache** — the 636 MB
weights ≫ X60 cache and stream from RAM each token, so an earlier "cache-resident" claim
was WRONG; k1-t4 simply sits below the X60 bandwidth-saturation point (the prior wall
finding used t16 on a 64-core box). So this result is **CONSISTENT with**
[[kernel-wins-dont-transplant-to-e2e]] (compute wins surface in compute-bound phases),
not a counterexample. k1 (8 cores, 3.5 GB free) structurally cannot reach the
fully-bandwidth-bound regime.

**The reasoned-NULL is therefore NEITHER confirmed NOR refuted** by this session: the
genuine memory-bound-decode regime (8B weights ≫ cache, high thread count) was staged on
rvv but not measured (board dropped). The user's instinct — "you wrongly reasoned-NULL
the q4_K e2e without measuring" — is **partly vindicated**: the e2e *was* worth measuring
(it is NOT trivially flat — it transports fully when decode is compute-bound), AND the
decisive memory-bound test is still open, so the prior blanket "likely NULL by the memory
wall" was an unmeasured assumption. The honest state: **measured transport at 1.1B
(compute-bound decode); memory-bound decode (8B) is the pending decider.**

**What is solidly established:** (1) the routing reality — ggml repacks q4_K at VLEN256,
so the LMUL knob has zero effect in the shipping default and only acts in the repack-OFF
fallback; (2) at 1.1B repack-OFF, m1 beats mf2 1.25× prefill / 1.21× decode, both arms
objdump-distinct, coherent "Paris"; (3) the micro→e2e transport is real when decode is
compute-bound.

## BLOCKER

- **rvv 8B (the memory-bound decider): BLOCKED** by transient `ssh rvv` network drops
  ("No route to host"). m1 is built+engaged there; only the bench remains. RE-RUN when
  rvv returns: `taskset -c 2 llama-bench -m <8B-Q4_K_M> -p 256 -n {64} -t {16,1} -r 3`
  for m1, then swap mf2 `.inc`, incremental rebuild, bench — same A/B, VLEN128.
- **k1 infra walls hit + resolved (for the record):** root fs 100% full (29G; OS fills
  it) → relocated `build` to `/data` NVMe (111G) via mv+symlink, set `TMPDIR=/data/tmp`,
  and freed 18G of stale `/tmp` debug dumps (`arb_on.txt`/`ime-correct.txt`, 8.8G each).
  Builds `-j2`, RAM 3.5G free fits the 1.1B (638M) model. No `llama-quantize` / no Q4_K
  model on k1 originally → downloaded tinyllama-1.1B-Q4_K_M (host net) → scp to /data.

## ARTIFACTS

- swap incs: `kq-bench/q4_K/qk-e2e-swap/tcrv_q4k_swap_{m1,mf2,m2}.inc`
- k1 board: `/data/tinyllama-1.1b-Q4_K_M.gguf`, `/data/bench_m1_final.out` (4.45/3.63),
  `/data/bench_mf2.out` (3.55/2.99), `/data/coh_m1f.out` ("…is Paris."),
  `/data/k1_m1f_vl256.dis` (m1 objdump), `/data/k1_mf2_vl256.dis` (mf2 objdump),
  repack/quants pristine backups `/data/{repack.cpp,quants.c}.pristine`.
- the q4_K repack-disable lives in the k1 working tree only (uncommitted, per
  instruction): `repack.cpp` q4_K `case 256: break` + quants.c vl256 `.inc` swap.

---

## FINAL (matched runs) — tight-interleaved per-pair A/B (2026-06-25, session-restart re-pin)

**Method (hardened over the prior `-r 3` table):** both arms **prebuilt** as separate
`libggml-cpu.so` (m1 md5 `2c09bc73`, mf2 md5 `145db792` — confirmed DISTINCT), then
**hot-swapped by `cp` with NO rebuild between arms** → enables true tight interleave
m1,mf2,m1,mf2… So each *pair* sees near-identical load, killing run-to-run + load drift.
Per arm: `taskset -c 0-3 llama-bench -p 256 -n 64 -t 4 -r 2`, q4_K repack OFF (forced
vec_dot). **Per-run guards:** `.so` md5 re-asserted before every invocation (logged in
each row); engagement print `TCRV Q4_K BLOCK-DOT ENGAGED (LMUL=…)` confirmed firing for
BOTH arms (m1 + mf2) with the correct LMUL. objdump discriminator in the swapped `.so`:
m1 = `e32,m4` + `vslidedown` present; mf2 = only `e32,m2`, ZERO `vslidedown`.
(k1 was under sustained external load avg ~4.9–6.7 throughout — shared box, no procs killed.)

| pair | m1 pp / mf2 pp | **pp ratio** | m1 tg / mf2 tg | **tg ratio** | loads m1/mf2 |
|---|---|---|---|---|---|
| 1 | 4.47 / 3.55 | 1.259 | 3.65 / 2.99 | 1.221 | 4.86 / 6.75 |
| 2 | 4.47 / 3.56 | 1.256 | 3.67 / 3.00 | 1.223 | 6.68 / 6.40 |
| 3 | 4.47 / 3.55 | 1.259 | 3.66 / 3.00 | 1.220 | 6.49 / 6.67 |
| 4 | 4.48 / 3.56 | 1.258 | 3.66 / 3.00 | 1.220 | 6.74 / 6.64 |

- **pp256 (prefill) m1/mf2 = 1.258× (range 1.256–1.259)**
- **tg64 (decode)  m1/mf2 = 1.221× (range 1.220–1.223)**

**Robustness (why the number is trustworthy on a noisy shared box):** pair 1 ran with a
within-pair load *mismatch* (m1@4.86 vs mf2@6.75) yet produced the **same ratio** as the
load-matched pairs 2–4 — direct evidence the `taskset -c 0-3` pin makes the matched ratio
**load-invariant** (the contention lands on cores 4–7). m1 pp256 is pinned 4.47–4.48 across
all loads. So this is not "tight by luck"; the isolation holds. The earlier `-r 3` table
(1.25× / 1.21×) reproduces exactly within rounding.

**VERDICT (matched, scoped — does the compute-side LMUL Win-A transport to decode?):**
**YES in the repack-OFF vec_dot regime — the tg ratio (1.221×, non-overlapping, well above
1.1×, range 0.003) is NOT a variance artifact; the prior preliminary 1.22× HOLDS under
matched re-pin.** Decode transports the win *because* repack-OFF forces decode onto the same
per-row vec_dot as prefill, where it is compute-bound (tg ratio ≈ pp ratio ≈ micro-ratio is
the expected signature; a real memory wall would crush tg→1.0×, which it did NOT). This is
**NOT** a claim that the LMUL knob speeds the *shipping* q4_K decode — the default VLEN256
path repacks to GEMM and **bypasses this kernel entirely (zero effect there)**. Consistent
with [[kernel-wins-dont-transplant-to-e2e]] (compute wins surface in compute-bound phases);
the genuine memory-bound decider (8B ≫ cache @ high thread count) remains the rvv-pending test.
