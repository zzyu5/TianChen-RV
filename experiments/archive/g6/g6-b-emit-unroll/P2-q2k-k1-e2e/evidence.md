# G6-B Phase-2 · q2_K @ k1 · deploy ROLLED emit → e2e re-measure vs hand-brick

- **campaign**: G6-B emit-quality lever ([K-7] 5th first-emission lever) — Phase-2 PERF verdict on the
  rolled-loop emission mode built + byte-exact-sealed in Phase-1 (commit 6379bda2).
- **board**: k1 / SpacemiT X60 / VLEN256 (vlenb=32) / 8-core / Bianbu clang-18.1.8 / gov=performance 1.6GHz
- **date**: 2026-07-13 · **status**: Phase-2 · board-measured · NOT committed (main session commits)
- **role**: deploy the ROLLED q2_K repack GEMM (emit_loop_schedule="rolled") into the G5 M2-q2_K-k1-e2e
  harness, re-measure prefill/decode vs the STOCK hand-brick, and decide the pre-registered dual exit.

## ★ Compiler note (task said "gcc-15 seal" — corrected)
k1's SHIPPED / deployment compiler is **clang-18.1.8** (per [CASE-COMPILER-ASYMMETRY]: board shipped
compiler rvv=gcc-15 / **k1=clang-18**). gcc-15 is the *rvv* board's shipped compiler. This Phase-2
deployment seal is therefore **clang-18-SYMMETRIC** (both ON and OFF built with clang-18 from ONE tree
⇒ kernel-account == system-account, dual-ledger same number). The Phase-1 rvv/clang-17 A/B FORM seal is
reproduced here under k1/clang-18 (below).

## opponent identity — CONFIRMED (裁三.2 [XFER-1] discriminant key)
q2_K@k1 opponent = **STOCK hand-tuned RVV q2_K 16x1 repack** (`ggml_gemm_q2_K_16x1_q8_K`, case256 fires
at VLEN256, repack.cpp:4637), **NOT block-dot** (opposite of q5_K). Sealed from the OFF lib this run:
STOCK GEMM `vsetvli=54 vwmacc=16 spill=4`; STOCK GEVM `vsetvli=30 vwmacc=4 spill=0` (compact rolled loop).
= the q4_K Win-K1-VLEN contest (our-emit vs 真出货 hand-brick). ROLLED emit hard-contests THIS hand-brick.

## build reproducibility — 3 variants from ONE k1 tree (dedicated /data/build-k1-q2k-rolled)
ONE tree, clang-18 symmetric, recompile only repack.cpp.o + arch/riscv/repack.cpp.o + relink. The ONLY
A/B/C diff is the q2_K GEMM main-term emission schedule (GEVM identical in both ON variants).
- **OFF**         md5 `14b6add6` — 0 weft_emit q2_K symbols (STOCK hand-brick) — **== g5 baseline OFF md5 (byte-exact reproduced)**
- **ON_UNROLLED** md5 `5b036155` — full-static-unroll — **== g5 deployment ON md5 (byte-exact reproduced)** → identical to what produced the 0.873× baseline
- **ON_ROLLED**   md5 `5d2dec81` — rolled-loop main term (NEW) — banners present (gevm=1 gemm=1); all 3 .so differ
Source RESTORED byte-exact after build: GEN=3cac40aa HDR=57851439 ARCH=c3c101fd (all baseline).

## correctness — GREEN (ROLLED, k1 silicon)
1. **Silicon ZERO-MODEL** (`kquant_repack_verify_q2K`, independent scalar q2_K dequant-matmul oracle from
   ORIGINAL pre-repack blocks, isum−summs fold), ROLLED kernel, k1 clang-18 VLEN256, dual seed:
   - seed 0xC0FFEE: **INT_mismatch_total=0** all 8 shapes; NORM worst_ulp=3402 worst_norm=6.618e-07
   - seed 20260708: **INT_mismatch_total=0** all 8 shapes; NORM worst_ulp=1803 worst_norm=5.769e-07
   - INT byte-exact + NORM worst_ulp=3402(seed1) **IDENTICAL to the deployed UNROLLED** silicon verify → rolled==unrolled on silicon.
   raw: `raw/silicon_verify_ROLLED_q2K.txt`
2. **rolled==unrolled e2e** (greedy, 4-prompt corpus, temp0/top-k1): after stripping this llama-cli build's
   conversational chrome (backspace spinner, ASCII banner — the g5-documented "cannot mechanically extract
   clean greedy" caveat), R and U answer text **BYTE-IDENTICAL 4/4 prompts**; R vs S (stock) identical on
   2/4, remaining 2 differ only in trailing timing-dependent spinner chrome (not tokens). engage banner:
   `WEFT G5-q2K EMITTED GEVM ... ENGAGED ... vlen=256` (rolled kernel confirmed firing). raw: `raw/correctness.log`
   - Phase-1 already proved rolled==unrolled byte-exact directly (A/B identity 43008/43008 bytes). Since
     UNROLLED was g5-ppl-matched to stock (Δ0.36% within ±4.50), ROLLED inherits the ppl-match transitively.
   - MIRAGE ruled out: make_block straight-interleave == the emitted kernel's expected layout (byte-exact vs oracle).

## kernel-axis FORM seal — k1 clang-18 SYMMETRIC (both OUR emit; standalone .o + deployed lib agree)
| form | vsetvli | vwmacc | spill | reload | textB (standalone .o) |
|---|---:|---:|---:|---:|---:|
| UNROLLED (== shipped S6, == g5) | 86 | 2304 | 6 | 8 | 27018 |
| **ROLLED** | 82 | **384** | 53 | 57 | **12132** |

vwmacc **−83.3%** (2304→384) · textB **−55.1%** (27018→12132) = the code-volume win the lever targets,
**reproduced on k1/clang-18** (Phase-1 rvv/clang-17 was −53.6% text, vwmacc −83.3%). spill 6→53 =
register residence deliberately traded for compactness. STOCK opponent (for scale): vwmacc=16 spill=4.

## e2e phase-split (prefill pp64 / decode tg32) — 3-way paired, physical .so swap, clang-18 symmetric
Config: taskset -c 0-3, threads=4, gov=performance 1.6GHz, PP=64 TG=32 REPS=6 PASSES=2 ⇒ 12 samples/side.
Interleaved rolled/unrolled/stock per pass. Load-gated START (loadavg1<2.3, foreign_bench=0; board idle
floor ~2.05 from background daemons; G6-A M2 dead → no contention). raw: `raw/phase_split.log` + `raw/phase_analysis.txt`.

| phase | ROLLED t/s (relIQR) | UNROLLED t/s | STOCK t/s | rolled/stock | unrolled/stock (ctrl) | rolled/unrolled | verdict |
|-------|-----------:|-------------:|----------:|-------------:|----------------------:|----------------:|---------|
| PREFILL (pp64) | 3.4970 (0.26%) | 7.5362 (0.06%) | 8.6147 (0.12%) | **0.406×** | 0.875× (g5=0.873× ✓) | **0.464×** | **<parity LOSS — rolled REGRESSES 2.15×** |
| DECODE (tg32)  | 4.2002 (0.12%) | 4.1925 (0.24%) | 4.8095 (0.15%) | **0.873×** | 0.872× (g5=0.875× ✓) | 1.002× | <parity LOSS (rolled≈unrolled, both lose) |

n=12/side (2 passes × 6 reps). relIQR ≤0.26% all sides = clean uncontended 4-thread measurement, load-gated
static board (foreign_bench=0). **UNROLLED control reproduces the g5 baseline byte-exact** (0.875×/0.872×
vs g5 0.873×/0.875×) ⇒ measurement valid; ON_UNROLLED .so md5 == g5 deployment.

## VERDICT (pre-registered dual exit — 裁二.2) → **NOT-FLIP branch**
**q2_K@k1 stays 黄-对手更强 (maintained).** Rolled emit does NOT flip ≥parity; it **REGRESSES prefill 2.15×**
(rolled/unrolled 0.464×), tie in decode. [K-7] 5th lever = **mechanism/emitter-maturity milestone only**
(rolled-loop emission mode BUILT + byte-exact hard gate GREEN, Phase-1) — **NOT a perf win, NOT 成色质变.**
No K-quant flip. perf-covered unchanged.

### 反汇编再归因 — WHERE the gap remains (性能宪章规则 1: bottleneck ≠ what the lever targeted)
Objdump (k1 clang-18 deployed lib), q2_K GEMM main term:
| kernel | vwmacc | vle16 | vse16 | spill(vsNr.v) | sp/fp ld-st | property |
|---|---:|---:|---:|---:|---:|---|
| ROLLED (ours)     | 384  | 324 | 373 | 53 | 298 | compact code BUT stack-panel partials round-trip |
| UNROLLED (ours)   | 2304 | 68  | 70  | 6  | —   | register-resident BUT giant code volume |
| STOCK (hand-brick)| 16   | 2   | 0   | 4  | —   | **compact AND register-resident (both at once)** |

- **Prefill (compute-bound)**: the rolled lever traded register residence for code compactness — its inner
  runtime loop stages the per-column-per-shift i16 partials to a **stack panel** and does load-accumulate-store
  every iteration (**324 vle16 + 373 vse16 ≈ 1.8 memory ops per vwmacc**, spill 6→53, +298 sp/fp ld-st). In the
  compute/register-bound prefill this memory round-trip DOMINATES → 2.15× slower than the register-resident
  unroll. **The targeted bottleneck (code-volume / I-cache) was NOT the real prefill bottleneck** — else rolling
  would help. This empirically confirms the **"re-roll trap"** the S6 comments warn of ([GAP-P1] pattern /
  性能宪章规则 2: 直觉投影"回卷省 I-cache/vsetvli"证伪).
- **Decode (memory-bound)**: rolled≈unrolled (1.002×), both lose to stock 0.87× — kernel form washes out at the
  memory ceiling ([[kernel-wins-dont-transplant-to-e2e]]: compute-bound kernel deltas don't transplant to
  memory-bound decode).
- **The real emitter-maturity target revealed**: the emit-quality gap to the hand-brick is NOT the unroll↔roll
  axis. The stock achieves **compact (16 vwmacc, rolled) AND register-resident (spill 4, ~0 partial round-trip)
  SIMULTANEOUSLY** — a *rolled-WITHOUT-accumulator-spill* tiling that NEITHER of our current emit forms produces
  (unrolled spills code volume; rolled spills the accumulator panel). That, not this schedule knob, is the lever
  q2_K needs.

### K-quant chain re-estimation (NOT-flip → negative prior; 不自动施工)
Because the rolled lever REGRESSES the compute-bound prefill via accumulator spill, naively rolling q6_K/q3_K
giant unrolls is **NOT warranted by this evidence** (q6_K is additionally weight-reconstruction-bound + 2995
redundant-vsetivli, a different emitter immaturity; rolling would add the same stack-panel spill). Recommend:
逐格 X-0 + opponent-identity still required, but the prior for "rolled-loop = K-quant perf fix" now shifts
**negative**; the actionable target is rolled-without-spill accumulator tiling, not the schedule knob alone.

## [PERF-1] eight-gate walk (NOT-FLIP branch — 记账用)
① correctness: PASS (silicon ROLLED INT_mismatch=0 all 8 shapes dual-seed + NORM identical to unrolled + rolled==unrolled e2e byte-identical 4/4 + engage banner + MIRAGE ruled out)
② VLEN-flip/dual-board: k1/VLEN256 measured (rvv/VLEN128 = separate clang-17 FORM seal, Phase-1) → PARTIAL
③ objdump seal: PASS (ROLLED vsetvli=82 vwmacc=384 spill=53 + stack-panel 324/373 vle/vse16 sealed; STOCK vwmacc=16 spill=4)
④ micro AND e2e: e2e LOSS (prefill 0.406× / decode 0.873×) — NOT a win, rolled regresses prefill 2.15×
⑤ dual-board e2e: k1-only → PARTIAL
⑥ discipline: PASS (DVFS 1.6GHz lock, 4-core pin, load-gated static board, paired interleaved, relIQR ≤0.26%, unrolled control reproduces g5 baseline)
⑦ mechanism attribution: PASS (regression = rolled stack-panel accumulator round-trip in compute-bound prefill; decode memory-bound washout)
⑧ wording: PASS (bounded k1/VLEN256/clang-18/both-phase; our-emit rolled vs 真出货 hand-brick; NOT a beat; [K-7] = mechanism milestone not perf win)
⇒ clean NOT-FLIP e2e (correctness GREEN · rolled perf REGRESSES) · [K-7] lever = emitter-maturity milestone (Phase-1 built+byte-exact), Phase-2 PERF = null/negative.

## board restore (verified clean · double md5-cert)
- A-tree source restored byte-exact: repack.cpp=3cac40aa, repack.h=57851439, arch/riscv/repack.cpp=c3c101fd (all baseline); no leftover emitted .inc.
- dedicated build /data/build-k1-q2k-rolled: LIVE .so = OFF pristine (14b6add6) then build dir REMOVED (freed).
- g5 /data/build-k1-q2k + /data/k1build-stock (871169a0) + /tmp/g5_q2k UNTOUCHED. G6-A M2 scratch not touched. main tree no git action.
- reproduction artifacts kept: /tmp/g6b-p2-q2k/{libggml-cpu.so.OFF/ON_UNROLLED/ON_ROLLED, weft_emitted_*.inc, scripts}.

## board coordination (load-gate 错峰)
G6-A M2 already dead (no IME bench contention this window). Board idle-load floor ~2.05 (background daemons +
stale g5_q5k busy-wait pollers, negligible CPU, not mine). Load-gate: waited for loadavg1<2.3 (idle floor +
margin; strict <2.0 unreachable) with foreign_bench=0 → cleared at loadavg1=2.04, measured as the only bench
load. build/deploy ran in parallel (CPU) on dedicated dir; measure錯峰 successful (no co-measure poison).
