# cell MANIFEST — g6-b-emit-unroll / P2-q2k-k1-e2e (Phase-2 PERF verdict)

- **campaign**: G6-B emit-quality lever ([K-7] 5th first-emission lever) — Phase-2 e2e PERF verdict for the
  rolled-loop q2_K emission mode built + byte-exact-sealed in Phase-1 (commit 6379bda2).
- **status**: Phase-2 COMPLETE (2026-07-13, k1/SpacemiT-X60/VLEN256/clang-18.1.8). **NOT-FLIP branch**:
  rolled emit does NOT reach ≥parity; it REGRESSES prefill 2.15× (rolled/unrolled 0.464×; rolled/stock 0.406×),
  ties decode (1.002×). q2_K@k1 stays 黄-对手更强. [K-7] = emitter-maturity/mechanism milestone (Phase-1
  built+byte-exact) — NOT a perf win, NOT 成色质变. NOT committed (main session commits).
- **role**: deploy ROLLED q2_K repack GEMM (emit_loop_schedule="rolled") into the G5 M2-q2_K-k1-e2e harness,
  correctness-first (silicon ZERO-MODEL + e2e A==B), then 3-way paired e2e re-measure vs the STOCK hand-brick.

## durable files (this cell)
- `evidence.md`               — full writeup: opponent identity, build reproducibility (3 .so, 2 == g5 byte-exact),
                                 correctness GREEN, k1/clang-18 FORM seal, e2e numbers, NOT-FLIP verdict + 反汇编再归因, 8-gate, restore.
- `raw/build_seal.log`        — 3-variant build+seal (OFF/ON_UNROLLED/ON_ROLLED) + objdump + byte-exact source restore.
- `raw/silicon_verify_ROLLED_q2K.txt` — ROLLED silicon ZERO-MODEL verify, k1 clang-18, dual seed (INT_mismatch=0 all 8 shapes).
- `raw/correctness.log`       — e2e greedy A==B (rolled vs unrolled byte-identical 4/4; vs stock; engage banner; llama-cli chrome caveat).
- `raw/phase_split.log`       — load-gated 3-way paired phase-split (prefill/decode, 12 samples/side, relIQR ≤0.26%).
- `raw/phase_analysis.txt`    — parsed medians + ratios (rolled/stock, unrolled/stock control, rolled/unrolled).
- `board_src/g6b_p2_build_seal.sh`     — 3-variant build+seal (dedicated /data/build-k1-q2k-rolled; restores A-tree).
- `board_src/g6b_p2_phase_split.sh`    — 3-way paired phase-split driver.
- `board_src/g6b_p2_gated_measure.sh`  — load-gate (loadavg1<cap, foreign_bench=0) then exec phase-split.
- `board_src/g6b_p2_correctness.sh`    — e2e greedy A==B (rolled vs stock, rolled vs unrolled).
- `board_src/g6b_p2_analyze.py`        — 3-way ratio analyzer.

## regeneration (emitted .c deterministically regenerable — see Phase-1 REGEN_RECIPE.md)
- ROLLED gemm .inc: weft md5 `f3f5b1da` / tcrv-renamed `4e1f8089` (== Phase-1 recorded).
- UNROLLED gemm .inc: weft md5 `d9dee831` (== g5 deployed) / tcrv `2dd964fe` (== Phase-1).
- gevm .inc: weft md5 `ee76a4d3` (== g5 deployed) / tcrv `b6513f6b` (== Phase-1). All reproduced byte-exact locally.

## .so variants (board /tmp/g6b-p2-q2k, ephemeral; md5 recorded)
- OFF `14b6add6` (== g5 baseline OFF) · ON_UNROLLED `5b036155` (== g5 deployment ON) · ON_ROLLED `5d2dec81` (NEW).

## board scratch (ephemeral, untracked): k1:/tmp/g6b-p2-q2k/ ; dedicated build k1:/data/build-k1-q2k-rolled REMOVED post-run
## main tree + /data/k1build-stock (871169a0) + /data/build-k1-q2k + /tmp/g5_q2k UNTOUCHED (A-tree restore double-md5-cert). G6-A M2 scratch not touched. NO git action.

## ★ opponent identity (裁三.2 X-0)
q2_K@k1 opponent = STOCK 手调 RVV q2_K 16x1 repack (case256 fires, repack.cpp:4637), NOT block-dot (与 q5_K 相反).
Sealed this run: STOCK GEMM vwmacc=16 spill=4 (compact rolled loop). = q4_K Win-K1-VLEN 同型对局 (our-emit vs 真出货 hand-brick).

## ★ compiler note
k1 shipped/deployment compiler = clang-18 (per [CASE-COMPILER-ASYMMETRY]: rvv=gcc-15 / k1=clang-18). Task's "gcc-15 seal"
was a mis-reference; Phase-2 seal is clang-18-SYMMETRIC (kernel==system ledger).

## ★ VERDICT
q2_K@k1 e2e (rolled deployed) = **黄-对手更强 (维持)**: prefill 0.406× / decode 0.873× vs stock hand-brick; rolled
REGRESSES prefill 2.15× vs unrolled (0.464×) via stack-panel accumulator round-trip (spill 6→53, 324 vle16 + 373
vse16). correctness GREEN (silicon INT byte-exact all shapes + rolled==unrolled e2e). [K-7] 5th lever = mechanism/
emitter-maturity milestone (Phase-1 built+byte-exact) — Phase-2 PERF null/negative. perf-covered unchanged.
Real target revealed = rolled-WITHOUT-accumulator-spill tiling (stock does compact+register-resident at once).
