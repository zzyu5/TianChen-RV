# T4b M4 — DECISIVE positional: our q4_K/q5_K repack-GEMM vs ggml's OWN, REAL Q4_K_M tensor (ssh rvv)

Operational index only. Outcome: **CASE CLOSED.** On a REAL dmin!=0 Q4_K_M model tensor, fed the
SAME ggml mat-quant q8 activation, OUR compiler-emitted q4_K repack-GEMM kernel is **integer
bit-exact** (MAIN/MIN 0 mismatches) and agrees with **ggml's own generic repack GEMM** to FP
reassociation noise (~1e-4 abs / 6e-5 rel). q5_K (shared min-fold) likewise agrees with an
independent integer-exact oracle to ~1e-6. The "min-term bug" is definitively NOT in the kernel;
claims are recoverable. A-tree restored byte-exact; emitter source untouched.

## What (the decisive positional, vs M1/M2c)
M2c measured a ~5-40% NORM "min-term" error vs a hand oracle and STOPped (hypothesised latent kernel
defect). M1 ZERO-MODEL falsified that: the kernel is bit-exact for the ACTUAL dispatch (mat-quant) q8;
the "bug" was the cert-harness oracle quantizing the activation differently (row-quant / −127-max lrintf)
from what ggml really feeds the GEMM (generic mat-quant). M4 removes ALL hand oracles from the loop:
we call **ggml's OWN** `ggml_gemm_q4_K_16x1_q8_K_generic` directly, on a REAL model tensor (dmin!=0),
with the SAME `ggml_quantize_mat_q8_K_4x1` q8 fed to both — and add an in-kernel INTEGER bit-exact capture.

## Method (reversible, board; NON-destructive)
- Host, tree quiescent: re-emit CURRENT emitter's kernels
  `tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp` from
  `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir` -> `fresh_q4K.inc` md5 **90d454da**,
  and `…q5-K…` -> `fresh_q5K.inc` md5 **c209226b**. NO emitter source change (git diff HEAD empty on lib/include).
- `instrument_kernel_q4k.py` — VERBATIM M1 pure-observation instrumenter (asserts base md5 90d454da),
  captures per-(blk,row,col) integer MAIN (v47/v49/v51/v53) + MIN (v46) + d/dmin/a_d. ZERO math change.
- `m4_patch.py` — MINIMAL non-destructive: edits ONLY `arch/riscv/repack.cpp`, only APPENDS (after the
  existing tcrv emitted includes) our instrumented q4_K .inc + clean q5_K .inc + two visibility-default
  `extern "C"` wrappers (`tcrv_m4_call_q4K/q5K`). Does NOT touch the dispatch selector and does NOT modify
  `ggml_gemm_q4_K_16x1_q8_K`/`_generic` — ggml's OWN repack GEMM stays intact and callable as the opponent.
  1 tracked file edited; GEN (`repack.cpp`) untouched. Backups + scratch under `/tmp/m4_decisive`.
- `m4_recon.cpp` — gguf (no_alloc) tensor inventory + dmin!=0 census: model has 194 Q4_K (all dmin!=0),
  33 Q6_K, **0 Q5_K**. Picked q4_K = `blk.0.attn_k.weight` [K=4096,N=1024] (dmin≈9×d, strong min-term).
- `m4_probe.cpp` — driver: reads the REAL tensor bytes straight from the gguf file (no full-model load),
  repacks via the M1 repacker (`kqr_repack_q4_K`, block_q4_Kx16), mat-quants a 4-row activation via ggml's
  own `ggml_quantize_mat_q8_K_4x1` (the SAME q8 into both kernels), then compares OURS (capture ON) vs
  `ggml_gemm_q4_K_16x1_q8_K_generic` vs an independent integer-exact double oracle (on-disk W × deinterleaved
  mat-quant q8). Also calls the RVV `ggml_gemm_q4_K_16x1_q8_K` to demonstrate its VLEN128 breakage.
  q5_K: model ships no native Q5_K, so weights are REAL-MODEL-DERIVED (dequant Q6_K `blk.0.attn_v` ->
  requant Q5_K via ggml `dequantize_row_q6_K`/`quantize_row_q5_K`, dmin!=0), compared vs the integer-exact
  q5_K oracle (5-bit weight = nibble | qh<<4). ggml ships NO q5_K 16x1 gemm (only q5_Kx8) so there is no
  q5_K head-to-head — the shared fold is covered by the q4_K head-to-head + this integer-exact q5_K check.
- `m4_board_run.sh` — baseline md5 verify (GEN deb61a29 / ARCH 99131cf7) + ARCH backup -> instrument ->
  patch -> rebuild ggml-cpu -> export checks -> compile driver -> run -> objdump min-fold -> EXIT-trap
  forced restore (ARCH -> 99131cf7, rebuild pristine .so 860736 B, kernels 0/0, wrappers 0). NO git ops.

## Result (m4_decisive_result.log)
- **q4_K `blk.0.attn_k.weight` (K=4096, N=1024, nb=16): dmin!=0 = 16384/16384 (min 2.1e-4 … max 1.1e-2), MIN-TERM ACTIVE.**
  - INTEGER capture: **MAIN mismatches=0, MIN mismatches=0 => INTEGER BIT-EXACT** (cols0..7, blk0..7, rows0..3).
  - OURS vs ggml_generic: max_abs=1.07e-4, max_rel=6.31e-5. OURS vs int-exact oracle: max_abs=7.6e-6, max_rel=**2.86e-6**;
    ggml_generic vs int-exact: max_rel=6.24e-5 — i.e. OURS is CLOSER to the integer-exact fold than ggml's own generic is.
  - RVV `ggml_gemm_q4_K_16x1_q8_K`: max_abs=29.3, 4081/4096 bad — VLEN256-only (hardcodes vl=16 on f32m2/u8m2,
    clamps to 8 at VLEN128). Confirms M2c's structural note; the correct VLEN128 realization of ggml's path is `_generic`.
  - **q4_K VERDICT: CASE CLOSED.**
- **q5_K real-derived (Q6_K blk.0.attn_v -> requant Q5_K): dmin!=0 = 16384/16384, MIN-TERM ACTIVE.**
  - OURS vs int-exact oracle: max_abs=9.5e-7, max_rel=**7.75e-7**. **q5_K VERDICT: CASE CLOSED.**
- Every arithmetic term accounted: integers bit-exact (0 mismatches); residual float delta is pure FP
  reassociation (ggml_generic folds `sumi*d*a_d` per element; our kernel integer-accumulates then folds once),
  ~1000× below the M2c "5-40%" artifact floor. (max_ulp-int values are inflated by near-zero cancellation;
  operative bounds are abs/rel.)
- **A-tree RESTORED byte-exact:** ARCH 99131cf7, GEN untouched deb61a29, pristine .so 860736 B, our kernels 0/0, wrappers 0.

## Verdict — pre-registered reading satisfied
Integer bit-exact AND float bounded-ULP-with-every-term-accounted vs ggml's OWN generic (q4_K) / integer-exact
oracle (q5_K), on REAL dmin!=0 model tensors, same quantizer both sides => **CASE CLOSED**: the kernel is correct,
the "min-term bug" is a cert-harness quantizer-mismatch artifact (M1), and the narrowed q4_K/q5_K claims are recoverable.

## Durable Files
- `m4_decisive_result.log`
- m4_probe.cpp — decisive driver (real gguf tensor -> repack -> mat-quant -> ours vs ggml own generic vs int-exact oracle; q4_K + q5_K).
- m4_recon.cpp — gguf (no_alloc) tensor inventory + dmin!=0 census (194 Q4_K / 33 Q6_K / 0 Q5_K).
- instrument_kernel_q4k.py — pure-observation integer MAIN/MIN capture instrumenter (asserts base md5 90d454da).
- m4_patch.py — minimal non-destructive reversible ARCH-only patch (append include + wrappers; ggml_gemm untouched).
- m4_board_run.sh — board build+run+forced-restore driver (baseline verify, backup, instrument, patch, rebuild, run, objdump, EXIT-trap restore).
- m4_decisive_result.log — captured board run (dmin census + integer bit-exact + float head-to-head + RVV VLEN128 breakage + forced-restore proof).
- fresh_q4K.inc regenerable md5 90d454da from test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir; fresh_q5K.inc md5 c209226b from …q5-K….mlir.
