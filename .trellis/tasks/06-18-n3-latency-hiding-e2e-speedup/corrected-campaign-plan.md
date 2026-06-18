# CORRECTED CAMPAIGN PLAN (re-grounded in spec/journal/code, 2026-06-18)

## 0. The drift (user-corrected)
Fail-fast found ~1.56× q4_0 GEMM, but that = **M-blocking + decode-amortization = above-layer
structural** (tiling geometry; `tcrv.generic_tile` forbidden in our layer). NOT our novelty.
Re-labeling it "1.5× over naive" smuggles the dropped claim back. **Dropped.**

## 1. The corrected publishable claim (in-layer, ablation-proven)
Our contribution = the Gearbox's **automatic capability/resource-aware LOW-LEVEL codegen passes**
(enumerate legal candidates from capability+resource facts → prune by register-budget/legality →
select by measured-record else static-fallback). Claim = **these passes ARE the source of speedup**,
proven by **ablation** (passes ON vs the spec's MVP placeholder: fixed single candidate / fixed
LMUL / fixed unroll = OFF), and that it's **ONE mechanism across many RVV kernels**, not a hand-tuned
kernel. Baseline for the ablation = our codegen ON ÷ OFF (same compiler). **scalar + naive-RVV** =
spec-sanctioned external yardsticks (index.md, generation-selection-tuning.md:55). **ggml is NOT a
baseline (I9)** — only a byte-exact correctness oracle.

## 2. The ~1.5× reconciled = REGIME-SPLIT (two columns that never merge)
| Regime | vs naive RVV | vs scalar | source |
|---|---|---|---|
| **Deferred-wide contraction** (i8 byte, i16 dot-reduce) — the real N3 win | **2.1–5.4×** | 4.0–12.3× | **max-legal-LMUL** (i8mf4→i32m8 @ budget≥32) |
| ggml block-dots (q4_0/q8_0/q4_1/q5_x) — parity regime | parity (packed-i4 0.93–1.01×) | 1.5–4.3× | in-layer **measured>static** ~1.2×; max-LMUL **latent** |
Headline: deferred-wide passes = **2–5× over naive / 4–12× over scalar**, byte-exact-gated, ssh rvv.
The 1.56× GEMM = above-layer structural, NOT claimed.

## 3. OURS vs ABOVE-LAYER
- **OURS (ablatable codegen passes):** LMUL/SEW selection (max-legal-LMUL, **primary**), unroll/
  `multi_block_factor`, `strip_elision` (capability-gated), reduction/accumulator layout (derived,
  not headline — multi-acc "buys nothing"), `operand_form` (static pick).
- **ABOVE-LAYER (acknowledge, NOT our claim):** **M-block/`activation_cols`** (cache-blocking tiling
  geometry — the dropped 1.56×), weight **repacking** (Q4_0_8x8 storage = frontend/I9), GEMM tiling/
  fusion. We may cite the *selection mechanism* generalizing to a tiling key, but not its number.

## 4. Ablation table (the central publishable artifact)
Factorial pass-ON/OFF on the SAME compiler, byte-exact-gated vs `_generic` first, min-of-N on ssh rvv.
| # | ablation | yardstick | proves |
|---|---|---|---|
| A | **max-legal-LMUL ON/OFF** (deferred-wide) | naive-RVV, scalar | the selector IS the source of 2–5× — **headline** |
| B | register-budget prune binds (budget 12 vs 32) | self | resource-derived, not lookup-table |
| C | measured-pick vs static-pick (q4_1 m1/1 vs m1/4) | our own two emissions | **measured>static ~1.2×** (loss→win) |
| D | **unroll inversion** (q4_0 factor=4 vs q4_1 factor=1) | measured | a fixed heuristic can't serve both → per-kernel select necessary — **generality centerpiece** |
| E | strip_elision capability gate (Zvl128b on/off) | self | N1 capability divergence from legality prune |
| F | tune-record present/absent/stale | self | fail-closed revalidation |
Generality: ONE mechanism × 5 block-dot kernels + 2 deferred-wide families. Table = kernel × pass × {on,off}.

## 5. Honest gaps (flag, not claimed-done)
- **q5_0/q5_1 UNMEASURED** (only q4_0/q8_0/q4_1 in inc10) → measurement TODO.
- **max-LMUL LIVE on deferred-wide (2 kernels), LATENT on block-dots** → why block-dots sit at parity
  + why the live llama integration ran **~1.8× SLOWER** (i8mf4→i32m1, 4 lanes vs ggml's 16).
  **Wiring max-LMUL into block-dots = the PRIZE** (real e2e llama win), not a current claim.

## 6. KEEP vs DROP from fail-fast
- KEEP: the regime-split map; the VLEN=128 ceiling fact; deferred-wide 2-family win as genuine
  vs-scalar-AND-naive evidence; decode-amortization as *motivation* for in-body memory-form (only
  the in-body part is ours).
- DROP: "1.56× GEMM is our speedup"; any merge of deferred-wide vs-scalar numbers with block-dot
  parity numbers; superseded gate4-vs-autovec-scalar "regression" numbers (known-unfair baseline).

## 7. Next experiments (in-layer, ablation-first, general)
1. **Build the factorial pass-ON/OFF harness** → the publishable `kernel × pass × {on,off}` table on
   ssh rvv. Start rows A, C, D (strongest, have data). The paper's central artifact + meets the
   ablation requirement.
2. **Measure q5_0/q5_1** (close generality gap).
3. **Wire max-legal-LMUL into the block-dot realization owner** (the documented remaining lift) — the
   prize: block-dots beat naive in absolute terms + fix the 1.8×-slower integration → the e2e llama win.
   Ablation-gated: prove ON>OFF on ssh rvv before claiming.

## 8. The running e2e-baseline build = RECLASSIFIED
Keep it, relabel: NOT a perf claim. It's (a) the byte-exact correctness gate (vs `_generic`/ggml-real)
every ablation row passes before ranking, and (b) the naive-RVV + scalar baseline generator for the
yardstick columns.
