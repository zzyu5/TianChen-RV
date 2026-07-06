# SEALED — q4_0 REPACK GEMM (prefill 5.9x) + GEVM (decode) — D-board return 2026-07-06

Completes the finale interrupted at F23 (board-recovery): the "generated ~5x e2e prefill"
sentence, plus the same-session paired decode (P1). Board `ssh rvv`, VLEN128, 4 threads on
cores 8-11, DVFS locked 2.6 GHz (0% span). Supersedes the PARKED interim cell
`rvv-vlen128-q4_0-gemm-constructed-redeploy/` (F23: only ours-pass1, no stock/pass2/CI).

## Protocol (all GREEN)
- **step 0 restore-verify** (`restore_verify.txt`): found the finale CONSTRUCTED GEMM still deployed
  + monolith .inc restore pending -> restored the monolith clean baseline, symmetric full-march
  rebuild, strict preflight 4/4, greedy-consistency 3/3 A==B. Clean baseline CONFIRMED before measuring.
- **step 1 noise floor** (`noisefloor_*`): remeasure-remeasure between-pass floor on the clean
  baseline. prefill floor 2.53% / decode 2.03% (ours side, under SPEC co-tenant); stock side
  0.15% / 0.27%. Effective floor ~2.5%.
- **step 2 sealed prefill** (`constructed_*`, `objdump_fingerprint.txt`): CONSTRUCTED GEMM redeployed
  (banner + mattr-decoded objdump fingerprint confirm the be66c917 front-door kernel runs, not
  vendor/hand-op), strict preflight 4/4, paired A(constructed)/B(stock) PASSES=2 REPS=5, greedy 3/3.
- **step 3 decode + bandwidth** (`bandwidth_analysis.txt`): same paired session decode + a 4-core
  read-BW ceiling probe; achievement analysis before the washed/not-washed verdict.

## RESULT — prefill (the earned sentence)
| build            | ours t/s | stock t/s | ratio | 95% CI      | floor | verdict    |
|------------------|----------|-----------|-------|-------------|-------|------------|
| CONSTRUCTED (s2) | 31.00    | 5.24      | 5.92x | [5.91,5.93] | 0.25% | DIFFERENCE |
| monolith  (s1)   | 30.70    | 5.24      | 5.86x | [5.79,5.94] | 2.53% | DIFFERENCE |

The two agree (monolith is byte-exact to the constructed GEMM — both trace the hand op), so the
5.9x is stable and >200x the noise floor. **EARNED**: the TianChen-RV compiler front-door-constructs
the q4_0 repack GEMM from an abstract quant_contraction declaration (m_regime=prefill ->
`tcrv_rvv.typed_repack_gemm_loop_body` region, be66c917), deploys it to the A-tree prefill path,
and delivers **~5.9x e2e prefill** vs stock ggml block-dot at VLEN128 — greedy-token-consistent to
stock, byte-exact-behavior to the retired hand op. (Kernel-level byte-exact estimate was 5.045x;
e2e prefill amplifies the memory-locality win to 5.9x.)

## RESULT — decode (honest, P1)
ours (emitted GEVM) 9.02 t/s vs stock 4.72 t/s = 1.91x, CI [1.90,1.92]. Bandwidth-achievement
(`bandwidth_analysis.txt`): ours ~100% of the 5.23 GB/s 4-core read-BW ceiling (SATURATED), stock
~53% (compute/dequant-bound, below the wall). NOT the "both hug peak => washed" case: the win is
real (we convert stock's wasted dequant cycles into saturated weight streaming) but CEILING-BOUNDED
(our decode is at the memory wall; ~9 t/s is the board limit, no kernel headroom past it).

## Honest confounds
- CO-TENANT: another agent's SPEC CPU2006 401.bzip2 ran on the board throughout (3->2 bzip2 workers).
  Not interfered with. Captured in the noise floor; the paired interleaved protocol cancels
  common-mode drift so the RATIO is protected; absolute t/s is depressed vs an idle board.
- e2e correctness = greedy-token consistency + finite logits (NOT bit-exact vs ggml at e2e; both
  accumulate fp in different IEEE-legal orders). Kernel bit-exactness is a separate statement.
- objdump: board bare objdump/default llvm-objdump do NOT decode the RVV/zvfh ops; the real
  fingerprint needs `llvm-objdump --mattr=+v,+zvfh,...` (documented in objdump_fingerprint.txt).
