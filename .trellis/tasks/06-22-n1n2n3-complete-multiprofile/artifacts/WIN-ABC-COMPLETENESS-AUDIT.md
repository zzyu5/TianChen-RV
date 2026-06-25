# Win-A/B/C × {micro, e2e} — COMPLETENESS AUDIT (read-only)

**Purpose.** Map exactly what is MEASURED vs a GENUINE GAP vs justified N/A vs REASONED-NULL across the
additive Win tiers, for every kernel/family the compiler emits. This is a *re-bucketing + count + distill* of
the two capstone docs (`OPTIMIZATION-EVIDENCE-MATRIX.md`, `N1N2N3-LEDGER.md`) into the task's 4-state
taxonomy. **No numbers are re-derived; no lib/ code is touched.** Source of truth = those two docs.

## 0. The 4-state taxonomy (and the two re-bucketing traps)

Each cell {Win-A, Win-B, Win-C} × {micro, e2e} is classified as exactly one of:

- **MEASURED** — a number was produced. *A LOSS/TIE is MEASURED* (the experiment ran; the answer is "no win"). Cite the ratio + honest verdict.
- **GENUINE GAP** — not run, but **runnable here** with the existing kernel/harness/HW, and would yield a **defensible number** that fills the cell.
- **N/A** — the Win class *structurally cannot apply* (no tunable knob; no ggml algorithm to swap; monolithic body). Not work owed.
- **REASONED-NULL** — determined *without running*, justified: a compute-side win × memory-bound decode (the memory wall); a repack with a byte-identical footprint that changes no memory traffic; same-transform-no-new-locality. Not work owed.
- **BLOCKED** (sub-state, kept OFF the gap list) — could be measured *in principle* but not here: upstream-ggml missing infra (q4_1 e2e), IME large-RAM HW, or dropped RVV0.7/C920.

**Two traps where this audit's taxonomy differs from the matrix's labels (the load-bearing re-bucketing):**

1. The matrix's **"NULL/LOSS"** (experiment RAN, verdict loss/tie — IME e2e 0.86×, q4_K GEMM 0.59–0.89×, the mixed K-quant Win-B micro) → **MEASURED**, not reasoned-NULL, not a gap. *Do not demote measured losses.*
2. The matrix's **"GAP not-wired (compute-side → e2e decode NULL, memory wall)"** → **REASONED-NULL**, *not* a genuine gap. All the K-quant / IQ / FP4 / forward-pass Win-A and Win-B **e2e** cells, plus the q8_0-repack and q4_K-repack Win-B e2e, are reasoned-NULL by the memory wall. The task explicitly forbids listing these as gaps.

**Global facts (stated once, not per-cell):**
- **Win-C** is one global fact in every cell: **STRUCTURAL-NULL** (`N1N2N3-LEDGER.md` §4.4 / MATRIX §1, Win-C column). The `reduction_structure` pass toggles 3.0–3.3× ON/OFF but a register-kept decomposition shows pure-structure ≈1.00× — the 3× is a per-iter `out[0]` memory round-trip (emitter artifact), not structural latency. So **Win-C micro = MEASURED-NULL** (the decomposition IS a measurement, verdict "no structural win") and **Win-C e2e = N/A** (monolithic body, no e2e home). This holds for *every* kernel; it is not a per-kernel debt.
- **Perf cells are RATIOS only.** Correctness/oracle/maturity (IQ vluxei16, ~17-block-dot bit-exact, FP4 sweep, q4_K/q4_1 oracle PASS, §10 option-2 in-compiler algorithm selection) are NOT perf cells — they live in Distill #3 ("done but maybe not realized").
- **Emit-only families are grouped once** (per MATRIX §1f) to avoid inflating counts by triple-counting ~30 rows.

---

## 1. Per-kernel completeness map

Legend per cell: `MEASURED` (+ ratio/verdict) · `GAP` · `N/A` · `R-NULL` (reasoned-null) · `BLOCKED`.

### 1a. q4_0 repack — the cumulative-ladder kernel (fills Win-A AND Win-B); the DENSE row

| Kernel | Win-A micro | Win-A e2e | Win-B micro | Win-B e2e | Win-C micro | Win-C e2e |
|---|---|---|---|---|---|---|
| **q4_0 repack GEMV** (decode) | **MEASURED** LMUL WIDE m1 vs NARROW mf2 **2.11–2.21×** (rvv); VLEN-strip 1×16 vs 2×8 **1.48×** (K1) | **MEASURED** LMUL decode **FLAT** (≈1.05× t1 / ≈0.93× t16, memory-bound) + VLEN-strip **1.31×** e2e (K1, SEAL) | **MEASURED 1.22×** vs ggml's own `vec_dot_q4_0_q8_0` (10 RVV ops), norm=0 | **MEASURED ~2.6×** (t16 tg64, memory-locality WIN); **K1 0.74× LOSS** disclosed | MEASURED-NULL (global) | N/A (global) |
| **q4_0 repack GEMM** (prefill M=4) | **MEASURED** LMUL WIDE vs NARROW **1.27–1.38×** (rvv) | **MEASURED** prefill **1.70× t1 / 1.10× t16** (clean, both ENGAGED) | **MEASURED 6.36×** vs same ggml RVV, norm=0 | **MEASURED 5.68×** (t16 pp256, matmul-bound, micro≈e2e) | MEASURED-NULL (global) | N/A (global) |

This is the only kernel with all of {Win-A, Win-B} × {micro, e2e} MEASURED — the campaign's dense, holding result.

### 1b. Win-A-only kernels (no algorithm-swap-vs-ggml → Win-B = N/A)

| Kernel | Win-A micro | Win-A e2e | Win-B micro | Win-B e2e | Win-C |
|---|---|---|---|---|---|
| **i16 dot-reduce contraction** (the Win-A headline) | **MEASURED 2.27–3.79×** rvv / 1.8–3.6× K1 / 1.4–1.7× C920 vs tune-OFF narrow-deferred (same algo, both compiler-emitted) | **N/A by nature** — not a standalone llama hot-path; its e2e analogue *is* the q4_0 repack-LMUL row | **N/A** — no ggml algorithm to swap; baseline is tune-OFF same algorithm | **N/A** | MEASURED-NULL micro / N/A e2e (global) |
| **q4_0 full GEMM (AoS)** (M-knob building block) | **MEASURED** (gearbox M-knob; superseded e2e by repack) | **R-NULL** — superseded; the routed prefill path is the repack GEMM (already MEASURED above). No independent e2e home | **N/A** — algorithm-change home is the repack (scored §1a) | N/A | global |
| **forward-pass scale / rms_norm** | **N/A as tune** — `strip_lmul` attr togglable but **no gearbox pass selects it**; never stamped | **R-NULL** — bit-identical reimpl of the ggml fn; forward-pass elementwise, memory-bound, no compute win to transplant | **N/A** — bit-identical reimpl (same algorithm) | N/A | global |

### 1c. N1-SELECTION-divergence block-dots (Win-A *column* = N1-sel, NOT N3 tune-on/off)

These populate Win-A as **N1 capability-driven SELECTION divergence** (the winner flips across profiles), distinct from an N3 tune ON/OFF ablation. The micro is MEASURED; the e2e leg is the one genuine cluster of gaps (the SELECTION e2e seal).

| Block-dot | Win-A micro (N1-sel) | Win-A e2e | Win-B micro | Win-B e2e | Win-C |
|---|---|---|---|---|---|
| **q8_0** (load-bearing N1) | **MEASURED** LMUL-family REVERSAL m2@128 vs m1@256, ~7% both ways | **GENUINE GAP** (gap 6) — no e2e leg confirms the selected winner runs in-llama + margin survives | **MEASURED** — built/checked vs `vec_dot_q8_0_q8_0` (micro-only kernel, not headlined) | **R-NULL** — not a repack-routed hot path; block-dot Win-B e2e is the same memory wall | global |
| **q4_1** | **MEASURED** elision flip, rvv +2.9% / K1 +11.2% | **GENUINE GAP** (gap 6) | **MEASURED 2.47–2.48×** vs `vec_dot_q4_1_q8_1`, norm~2e-6 | **BLOCKED** — no ggml q8_1x4 quantizer / q4_1-repack routing (upstream, not us) | global |
| **q4_0** (block-dot) | **MEASURED** within-m1 factor flip, ~0.8% (marginal) | **GENUINE GAP** (gap 6) — reached e2e via repack, but block-dot SELECTION winner has no e2e seal | (same-algo vec_dot; Win-B home is the repack, §1a) | (see repack §1a) | global |
| **q5_0** | **MEASURED-NULL** — m1-only legal set, tie ≤1.3% | **R-NULL** — null divergence = no selection winner to seal, the e2e seal is vacuous (NOT a gap) | **MEASURED** (block-dot coverage) **0.26× LOSS** vs `vec_dot_q5_0_q8_0` | **R-NULL** (memory wall) | global |
| **q5_1** | **MEASURED-NULL** — m1-only legal set, tie ≤0.05% | **R-NULL** — null divergence, vacuous seal (NOT a gap) | **MEASURED 0.29× LOSS** vs `vec_dot_q5_1_q8_1` | **R-NULL** (memory wall) | global |

### 1d. q4_1 repack (Family-B) — algorithm-change kernel exists, e2e upstream-BLOCKED

| Kernel | Win-A micro | Win-A e2e | Win-B micro | Win-B e2e | Win-C |
|---|---|---|---|---|---|
| **q4_1 repack GEMV** | **MEASURED ~1.80×** WIDE m1 vs NARROW mf2 (rvv), byte-exact (norm 0), 3-run | **BLOCKED** — inherits the e2e block (no q8_1x4 quantizer / routing) | **N/A** — oracle PASS (norm ≤7.6e-6 vs scalar ref) is *correctness only*; **ggml ships no q4_1 repack algorithm to swap against** → Win-B perf class can't apply (same N/A criterion as i16) | **BLOCKED** — upstream e2e gap | global |
| **q4_1 repack GEMM** | **GENUINE GAP (gap 4b, QUICK)** — only GEVM arm measured; GEMM WIDE/NARROW ablation reuses the identical harness | **BLOCKED** — inherits block | **N/A** — oracle PASS (norm ≤7.634e-6, correctness; doc-drift resolved §2); no ggml q4_1 repack to compare → Win-B perf N/A | **BLOCKED** — upstream e2e gap | global |

### 1d-q8. q8_0 repack GEVM (Family-A symmetric; built + checked + merged)

| Kernel | Win-A micro | Win-A e2e | Win-B micro | Win-B e2e | Win-C |
|---|---|---|---|---|---|
| **q8_0 repack GEVM** | **MEASURED** — strip-count tune at mf2: VLEN256 gearbox selects hl=16, **1.95× faster** than hl=8 (selects faster arm); VLEN128 hl=8 forced. (mf2-vs-RVV0.7-m1 5.5× is an ISA-gen comparison, not a tune) | **GENUINE GAP (stage-2b)** — the VLEN256 strip-tune e2e `.inc`-swap is unbuilt; runnable on K1, would yield a number (this is a tune that DOES change the kernel shape, unlike the memory-wall e2e cells) | **MEASURED LOSS** ~1.3–1.7× @VLEN128 vs ggml's own `vec_dot_q8_0` (byte-exact agreement) | **R-NULL** @VLEN128 — micro LOSS + byte-identical footprint + memory-bound decode → no win branch | global |

### 1d-qk. q4_K repack GEVM+GEMM (K-quant, DOMINANT quant; emitter oracle-CORRECT)

| Kernel | Win-A micro | Win-A e2e | Win-B micro | Win-B e2e | Win-C |
|---|---|---|---|---|---|
| **q4_K repack GEVM** | **GENUINE GAP (stage-1c/2)** — strip-count tune at mf2 (like q8_0); needs the Win-A stamp wired + the VLEN256 companion. Runnable, would yield a number | **R-NULL** @VLEN128 — byte-identical 2304B footprint + memory-bound decode + micro LOSS → no win branch (contrast q4_0 which CHANGED memory traffic). **@K1 VLEN256 vs ggml's own repack = GENUINE GAP** (oracle-pending, tie-likely) | **MEASURED LOSS** ~1.5–2.1× @VLEN128 vs ggml's own `vec_dot_q4_K_q8_K_vl128` (hand-tuned VLEN128 RVV), agreement ~e-7 | **R-NULL** (memory wall) | global |
| **q4_K repack GEMM** (prefill) | (the M-knob amortization is the GEMM itself; LMUL Win-A = same stage-1c gap) | **R-NULL** — prefill compute-bound but micro LOSS → loss would show e2e, no win branch | **MEASURED LOSS 0.59–0.89×** @VLEN128 vs ggml `vec_dot_q4_K_vl128` (amortization NARROWS the GEVM gap but doesn't cross 1.0) | **R-NULL** (reasoned-loss, prefill) | global |

### 1e. IME — N2 second family (own row, not a clean A or B)

| Kernel | "Win-A" micro | "Win-A" e2e | "Win-B" micro | "Win-B" e2e | Win-C |
|---|---|---|---|---|---|
| **IME matmul** (`vmadot`) | **N/A** — no LMUL/strip/repack knob; the 4×4×8 MAC is VLEN-DERIVED (N1 capability hook), not an ablatable tune | **N/A** (no Win-A tune to take e2e) | **MEASURED** (re-baselined `0d75632d`) vs ggml's real shipped RVV: **M=1 5.66× / prefill GEMM ~11.7–12.9× — IME WINS every shape** | **MEASURED-NULL** — decode tg16 0.86× / pp32 0.98× (IME engaged); prefill apparent 1.37–1.95× is a build-codegen artifact (decode-control-isolated). Matrix unit can't help memory-bound M=1 | global |
| **IME `mma_slide`** (`vmadot1`, 5th op) | **N/A as ratio** — slide∈{1,2,3} is a window-stride knob, but slide 2/3 silicon-gated; no ON/OFF perf ablation | N/A | **N/A — no number** (no ggml RVV slide baseline exists → no manufactured speedup; discipline). Defensible = algorithm-change CAPABILITY + K1 bit-exact 16/16 (correctness, not a cell) | **R-NULL** (prefill-only; decode memory wall) | global |

Note: IME's "methodology-clean Win-A or Win-B" cell (an IME-tune ON/OFF, or vs-ggml-shipped-IME-backend) is **gap 7** — a genuine but harness-design gap (see ranked list).

### 1f. Emit-only kernel families (grouped once; no togglable Win-A unless noted)

| Family | Win-A micro | Win-A e2e | Win-B micro | Win-B e2e | Win-C |
|---|---|---|---|---|---|
| **K-quant block-dots** q6_K/q4_K/q5_K/q2_K/q3_K | **MEASURED — Win-A LMUL tune DONE** (both-board byte-exact): q4_K m1 narrows 1.80→1.38×; q6_K 2.19→1.91×; q3_K 2.10→1.83× (m1 best at BOTH VLENs; q5_K auto-covered by q4_K helper; q2_K already a gather-WIN N/A). Narrows but doesn't close (residual = ggml `_vl256` nibble-split algorithm) | **R-NULL** — Win-A is compute-side → e2e decode null by the memory wall | **MEASURED, MIXED** vs ggml's own `_vl256` (K1 VLEN256): q2_K **WIN 1.016×**, q5_K **TIE 0.998×**, q4_K 1.72× LOSS, q6_K 2.26× LOSS, q3_K 2.13× LOSS | **R-NULL** (memory wall) | global |
| **IQ-quant block-dots** iq4_xs/iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq1_s/iq1_m | **N/A** emit-only (no togglable knob) | **R-NULL** (memory wall) | **MEASURED — all LOSS 5–22×** vs ggml's own (bit-exact agreement); scalar-gather vs ggml `vluxei16`. *(5/8 now vluxei16-revectorized → gaps narrowed, e.g. iq2_xxs 6.81→1.91× — that is correctness/maturity, see Distill #3)* | **R-NULL** (memory wall) | global |
| **Ternary block-dots** tq2_0/tq1_0 | **N/A** emit-only | **R-NULL** (memory wall) | **MEASURED** tq2_0 **0.60× LOSS**, tq1_0 **0.44× LOSS** vs ggml's own | **R-NULL** | global |
| **FP4-codebook block-dots** iq4_nl/mxfp4/nvfp4/q1_0 | **N/A** — `coreLmul` read in emitter but no pass stamps it | **R-NULL** (memory wall) | **MEASURED** iq4_nl **1.32× WIN**, mxfp4 **1.21× WIN** (tiny-codebook vrgather, bit-exact vs `_vl128`); nvfp4 **N/A** (ggml ships no RVV nvfp4 kernel; vs-scalar 0.57× is not a ratio against ggml) | **R-NULL** | global |
| **Forward-pass hard-pinned** silu/softmax/quantize_row_q8_0/rope_norm | **N/A** hard-pinned (no strip_lmul knob) | **R-NULL** (elementwise, memory-bound) | **N/A** bit-identical reimpl of ggml fn (same algorithm) | **R-NULL** | global |
| **q4_0 GEMM-tile** (building block) | **N/A** — composes parent GEMM M-knob | **R-NULL** (superseded by repack) | **N/A** (building block; algorithm = shared weight decode) | **R-NULL** | global |

---

## 2. Doc-drift resolved (newer doc wins)

- **q4_1 GEMM oracle:** MATRIX §1d says **PASS** (norm ≤7.634e-6, commit `039fcd54`); LEDGER §7/§8 still says oracle **deferred/DONE**. **Take the newer MATRIX: oracle PASS = correctness DONE.** This is *correctness only* — the q4_1 GEMM Win-B *perf* vs ggml remains N/A/BLOCKED (no ggml q4_1 repack algorithm to compare against, and e2e is upstream-blocked). Updating the audit's cell accordingly (§1d above).

---

## 3. COVERAGE SUMMARY — the real state of "complete evidence"

**Counting convention (honest, no forced total).** The cell universe is the **Win-A and Win-B columns × {micro, e2e}** over the 23 kernel/family rows in §1 (Win-C is one global fact, below; correctness/maturity is excluded — that is Distill #3). Nominal ≈ 23 × 4 = **~90** perf cells (minus the 2 q4_0-block-dot Win-B cross-refs that resolve to the repack row). **The three buckets that carry the finding are small, named, and EXACT; the two large buckets (R-NULL, N/A) are reported as a justified remainder, not a forced integer** — because whether the IQ family's 8 sub-kernels count as 1 grouped row or 8 changes the integer but not the finding. The convention here groups emit-only families to **one** row each (as the matrix does), so the IQ/ternary/FP4/forward-pass e2e legs are a handful of grouped cells, not ~30.

**The exact, load-bearing buckets:**

| State | Count | Where (exact enumeration) |
|---|---|---|
| **MEASURED** (incl. honest LOSS/TIE/NULL) | **31** *(verified two ways)* | 1a q4_0 repack all 8 · 1b i16 Win-A micro + AoS Win-A micro (2) · 1c Win-A micro ×5 + Win-B micro ×4 [q8_0/q4_1/q5_0/q5_1] (9) · 1d q4_1-repack GEVM Win-A micro (1) · 1d-q8 q8_0-repack Win-A micro + Win-B micro (2) · 1d-qk q4_K-repack Win-B micro GEVM+GEMM (2) · 1e IME Win-B micro + Win-B e2e (2) · 1f K-quant Win-A micro + K-quant Win-B micro + IQ Win-B micro + ternary Win-B micro + FP4 Win-B micro (5) |
| **GENUINE GAP** (bounded, runnable here) | **7** | q4_1-repack GEMM Win-A micro (gap 4b) · block-dot SELECTION e2e seal × {q8_0, q4_1, q4_0} (gap 6, ×3) · q8_0-repack VLEN256 strip-tune e2e (stage-2b) · q4_K-repack Win-A micro stamp (stage-1c) · q4_K-repack @K1 VLEN256 Win-B oracle |
| **BLOCKED** (off the gap list) | **4** | q4_1-repack GEVM Win-A e2e + Win-B e2e (2, upstream ggml routing) · q4_1-repack GEMM Win-A e2e + Win-B e2e (2). *(IME methodology-clean / large-model e2e is NOT a grid cell — it is ranked gap #6, design+HW-blocked.)* |
| **Win-C** (GLOBAL, every cell) | one fact | **MEASURED-NULL micro / N/A e2e** — §4.4, pure-structure ≈1.00×, structural novelty NOT demonstrated |

**The remainder (~48 cells) is the justified R-NULL + N/A bulk, dominated by two patterns:**
- **REASONED-NULL (~24)** — every emit-only family's Win-A and Win-B **e2e** leg (K-quant, IQ, ternary, FP4, forward-pass), plus q8_0/q5_0/q5_1 block-dot Win-B e2e, q8_0-repack Win-B e2e, q4_K-repack Win-A/Win-B e2e @VLEN128 (GEVM+GEMM), AoS-GEMM e2e, q5_0/q5_1 Win-A e2e (vacuous seal). All are the memory wall (compute-side win × bandwidth-bound decode) or a byte-identical-footprint repack with no new traffic.
- **N/A (~24)** — every emit-only family's Win-A **micro** leg (no togglable knob: IQ, ternary, FP4, forward-pass-pinned, q4_0-tile); the i16 / q4_1-repack / forward-pass Win-B legs (no ggml algorithm to swap — bit-identical reimpl or no ggml sibling); IME Win-A (VLEN-derived, not ablatable) + IME mma_slide; nvfp4 Win-B (ggml ships no RVV nvfp4 kernel).

Split-cell note: q4_K-repack Win-A/Win-B **e2e** is R-NULL **@VLEN128** but a **GENUINE GAP @K1 VLEN256** (oracle-pending) — the K1 leg is the one in the GAP row; the @128 leg is in the R-NULL remainder.

**Honest reading:** the matrix is **dense and complete for q4_0** (all 8 A+B perf cells MEASURED, micro AND e2e, including honest NULL/LOSS) and for the **N1 q8_0 VLEN selection flip** (micro). **MEASURED (31) is the plurality** and the largest named bucket. The **genuine-gap bucket is small (7)** and dominated by two clusters: the block-dot SELECTION e2e seal and the new-repack (q8_0/q4_K) tune/VLEN256 legs. The large R-NULL/N/A remainder (~48) is **not missing evidence** — it is the campaign's central finding (compute-side wins don't transplant to memory-bound decode) and the kernels' structural nature (no knob / no ggml sibling) applied correctly cell-by-cell. The open cells are a handful of bounded measurements, not a structural void.

---

## 4. HIGHEST-VALUE GENUINE GAPS — ranked (bounded measurements that fill a real cell)

Each survives the test "would running it produce a real, defensible number?" None is a memory-wall-NULL or an N/A.

1. **Block-dot SELECTION e2e seal — q8_0 VLEN-flip (gap 6).** *Highest VALUE.* The q8_0 m2@128↔m1@256 LMUL-family reversal (~7%) is the **load-bearing N1 result** and today has **micro only**. Why this is a genuine perf gap (not a memory-wall R-NULL): the **VLEN-strip selection already transplanted to decode e2e (1.31× on K1, §4.1c)** — empirical proof that a *selection* outcome CAN survive the memory wall (it changes which kernel shape streams, not just compute). So the q8_0 LMUL-family selection e2e outcome is genuinely **unknown** and worth running: wire the selected block-dot as the in-llama kernel and measure whether the per-profile winner survives (reuse the K1 engagement+correctness pattern). The margin may compress at the bandwidth ceiling, but unlike the emit-only compute-side cells there is a measured precedent that selection survives — that is what distinguishes this from a reasoned-NULL. More plumbing than a microbench rerun, but the highest-value gap.

2. **q4_1 repack GEMM Win-A WIDE/NARROW ablation · micro (gap 4b).** *Highest TRACTABILITY.* The GEVM arm is already measured (~1.80×); the GEMM op + WIDE/NARROW gearbox arm exist and the ablation **reuses the identical `ablation_micro.cpp` harness** — same-day number on rvv. Quickest real number. (e2e inherits the upstream BLOCK → micro-only is the realistic deliverable.)

3. **q8_0 repack VLEN256 strip-tune e2e (stage-2b).** A genuine gap *distinct from the memory-wall NULLs*: the VLEN256 hl=16-vs-hl=8 strip-tune is a real RVV1.0 auto-tune (1.95× micro) that **changes the kernel shape**, runnable on K1 via `.inc`-swap. Would yield a defensible e2e number (the q4_0 VLEN-strip seal proves this regime can transplant).

4. **q4_K repack @K1 VLEN256 Win-B oracle (tie-likely).** The dominant quant's repack at the VLEN where block-as-lane fits natively. Oracle-pending; tie-likely vs ggml's own VLEN256 repack, but a measured tie at K1 is a real, defensible cell (and the only regime where q4_K's repack is not a reasoned-loss). Bounded if K1 is up.

5. **q4_K repack Win-A micro strip-stamp (stage-1c).** Wire the strip-count tune stamp into `RVVRepackStripWidthMaterialization.cpp` (the deferred q4_K Win-A knob, now scoped in §9.2). Yields a micro tune number on the dominant quant. Medium effort (emitter wiring, not a rerun).

6. **IME methodology-clean Win-A or Win-B cell (gap 7).** Needs a new harness: either an IME-emitting-pass tune ON/OFF, or an IME-vs-ggml-shipped-IME-backend comparison (to replace the current "vs competent RVV" framing). Genuine but harness-design effort, not a rerun. Lower rank = more design cost.

**NOT gaps (do not manufacture work):** every emit-only Win-A/Win-B **e2e** cell (K-quant, IQ, ternary, FP4, forward-pass) is **REASONED-NULL by the memory wall** — a compute-side win cannot transplant to memory-bound decode; building a `.inc`-swap to measure a foregone "flat" changes no cell label. The q8_0/q4_K-repack Win-B e2e are likewise reasoned-NULL (byte-identical footprint, micro LOSS, no new locality). q4_1 e2e and IME large-model e2e are **BLOCKED** (upstream-ggml / HW), not gaps fillable here.

---

## 5. DONE that the user might not realize is complete

These are real, finished deliverables that are *not perf-cell ratios* (so they don't appear as MEASURED cells) but materially complete the "complete evidence" story:

1. **K-quant Win-A LMUL tune is DONE for q4_K / q6_K / q3_K** — the deferred K-quant tune knob, both-board byte-exact, with VLEN256-discrimination control: q4_K 1.80→1.38×, q6_K 2.19→1.91×, q3_K 2.10→1.83× (m1 best at both VLENs; q5_K auto-covered by the shared q4_K helper; q2_K already a gather-WIN). These ARE filled Win-A micro cells now — not a gap.

2. **Block-dot single-kernel coverage ~17 kernels ALL emit CORRECT** (maturity) vs ggml's own shipped RVV kernel — every standard/K/IQ/ternary/FP4 block-dot bit-exact (or fp-fold-order matched). The compiler correctly emits even the hard IQ codebook/sign/grid decodes. This is the "单个kernel测试 for ALL kernels" maturity statement.

3. **FP4 family closes the quant-family sweep** — coverage now spans K / standard / IQ / ternary / **FP4**. mxfp4 bit-exact + **1.21× WIN**; nvfp4 bit-exact (incl. denormal branches). 3 micro-WINS total (q2_K, iq4_nl 1.32×, mxfp4 1.21×) — the tiny-codebook `vrgather` rule, bucket-scoped.

4. **IQ vluxei16 gather revectorization 5/8 DONE** — iq1_s/iq1_m/iq3_xxs/iq3_s/iq2_xxs revectorized from scalar gather to hardware `vluxei16` (the SAME gather ggml uses → gap-closed, e.g. iq2_xxs 6.81→1.91×, iq3_xxs 22.35→8.14×). This is correctness/maturity (parity target), NOT a Win-tier — easy to mistake for an open perf gap.

5. **q4_K + q4_1 repack emitters oracle-CORRECT** — q4_K GEVM+GEMM oracle PASS (WORST_NORM ~7e-7, 8 shapes, multiple negative controls) on the *dominant* quant; q4_1 GEMM oracle PASS (norm ≤7.6e-6, doc-drift resolved §2). Correct kernel-EXPANSION, not perf cells.

6. **§10 option-2: the in-compiler capability-aware ALGORITHM SELECTION is BUILT (lit-only) — the architectural N3 headline.** The compiler *autonomously selects, declares, and can produce* the measured-best algorithm: abstract `GgmlQuantContractionOp` → `selectContractionAlgorithm(quant, regime, VLEN)` (branch-free 3-fact AND, no string-match) → the SAME binary picks repack@VLEN128 vs block-dot@VLEN256, byte-exact, plus a validated plain→x16 packer (host memcmp==0 over 5.2M blocks). This is a true architectural N3 novelty (selection-CORRECTNESS) — easy to under-credit because it carries no new perf number (C3-C5 e2e is the remaining HW bulk).

7. **N2 rapid-family-add is mechanically PROVEN** — IME 2nd family at **0 core lines** + 9 additive wiring lines + 1918 isolated plugin lines, capability-FACT gated (not string-match), with 4 ops added (mma / tiled / mma_u / mma_su) + a 5th (mma_slide) each at 0-core, bit-exact 16/16 on real X60. The N2 *structural* claim is DONE (perf is separately N3, and e2e-NULL).

8. **Win-C is a determined REASONED-NULL, not an open gap** — the pass is built + trellis-checked (3.0–3.3× toggle) but decomposed to pure-structure ≈1.00×. "Win-C not demonstrated" is a *finished, honest* result, not unstarted work.

---

*Read-only audit. Source of truth: `OPTIMIZATION-EVIDENCE-MATRIX.md` + `N1N2N3-LEDGER.md` (this task's artifacts). The 4-state {MEASURED / GENUINE-GAP / N/A / REASONED-NULL} taxonomy, the "matrix NULL/LOSS → MEASURED" and "matrix GAP-not-wired → REASONED-NULL" re-bucketings, and the BLOCKED-off-the-gap-list rule are load-bearing — do not collapse them. No lib/ code was touched.*
