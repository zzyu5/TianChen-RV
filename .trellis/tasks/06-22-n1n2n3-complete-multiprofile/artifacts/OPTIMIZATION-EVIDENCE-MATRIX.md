# TianChen-RV — Complete Optimization Evidence Matrix

**Driving document of the 06-22 multi-profile campaign.** This is the structured target for the
"complete optimization evidence" goal: every compute kernel the compiler emits, scored on the user's
three Wins × two regimes (kernel-microbench, llama-e2e), across the three in-scope profiles
(rvv SG2044 VLEN128 / K1 X60 VLEN256 / IME on X60). Source of truth = `N1N2N3-LEDGER.md` §7 capstone;
every value here is transcribed from a committed FINDING and provenance-cited. **No scalar performance
ratio appears anywhere** (Rule 0, `N3-METHODOLOGY.md`); "byte-exact vs scalar oracle" is a *correctness*
note only. Scope cut: RVV0.7 / C920 cells are dropped and deliberately absent.

---

## 0. Definitions, cell taxonomy, and the cumulative ladder

The three Wins, exactly per the user's definitions (`N1N2N3-LEDGER.md` head):

- **Win-A** = the compiler's automatic **tuning** choice — a knob ON vs OFF, **no algorithm change**,
  **both arms compiler-emitted**, only the tuned knob (max-legal-LMUL width / VLEN-strip / selection) differs.
- **Win-B** = a generated kernel that **changes the ALGORITHM** (the q4_0 repack: 16-blocks-as-lanes
  GEMM/GEMV). Baseline = **llama.cpp's OWN shipped optimized RVV kernel** (`ggml_vec_dot_*`, real RVV ops),
  NOT scalar, NOT `_generic`, NOT a hand-naive.
- **Win-C** = an automatic **PASS that changes algorithm STRUCTURE**. **PASS BUILT; structural novelty NOT demonstrated.** The `reduction_structure` pass toggles deferred-vs-per-iteration on the i16 dot-reduce body and gives 3.0–3.3× pass-ON/OFF on rvv (real, byte-exact, I5/I7 trellis-check PASS). BUT a hand-written register-kept per-iteration analysis kernel DECOMPOSED it: the pure reduction-structure delta is **≈1.00× (NULL, measured)** — the entire 3× is the per-iter OFF arm's `out[0]` MEMORY round-trip, an emitter artifact the deferred structure happens to avoid, NOT a structural-latency win. So **Win-C-as-structural-novelty = NOT demonstrated**; the pass's honest role is as the structural ENABLER of the Win-A LMUL sweep, not a standalone Win-C (§4.4).

**Cell taxonomy (4 states — never collapsed to binary "value-or-GAP"):**

| State | Meaning | Example |
|---|---|---|
| **MEASURED** | a real ratio with provenance | q4_0 GEMM Win-B micro 6.36× |
| **NULL / LOSS** | a *filled, honest* result — the experiment ran and the answer is "no win / a loss" | IME e2e 0.86×; LMUL-in-repack decode flat; K1 repack 0.74× |
| **N/A** | the Win **class structurally cannot apply** to this kernel (no tunable knob exists, or no algorithm-swap-vs-ggml home). NOT work owed. | Win-B on i16-contraction; Win-A on hard-pinned silu |
| **GAP** | measurable in principle, just **not yet done** | q4_1 repack Win-A ablation micro |

**NULL/LOSS and N/A are NOT gaps.** Only "GAP" is a gap. This distinction is the campaign's honesty
discipline; collapsing N/A into GAP would imply work is owed where the class can't apply.

**The cumulative ladder (made explicit, per the user's request):**

- A **Win-B** kernel is *also* a Win-A kernel **iff** its algorithm-change kernel additionally carries a
  tune knob. The **q4_0 repack GEMM/GEMV are the only rows that fill Win-A AND Win-B simultaneously**: the
  repack *is* the algorithm change vs ggml (Win-B), and it *carries* the LMUL-width + VLEN-strip tune (Win-A).
- **Win-C builds on A+B** (a structural pass would re-derive the repack automatically). Since **no Win-C pass
  exists**, the Win-C column is one global fact — `NONE (§4.4)` in every cell — **not N×2 empty cells**, and
  **not** a per-kernel debt. The repack is hand-authored Win-B, correctly NOT relabeled Win-C.
- **N1-selection ≠ Win-A tune-on/off.** The block-dot rows populate the Win-A *column* but as **N1
  capability-driven SELECTION divergence** (the winning candidate flips across profiles, e.g. q8_0
  m2@VLEN128 vs m1@VLEN256). This is the N1 claim, **not** an N3 tune-ON-vs-OFF ablation. Rows are labeled
  `N1-sel` to keep them separate.

**Profile axis:** the N1 block-dot rows show the **paired rvv128/K1256 cells** (the pairing *is* the
divergence claim). Every other cell is annotated with the chip it was measured on (LMUL-in-repack = rvv;
VLEN-strip = K1; IME = K1-only) rather than triplicated into mostly-empty per-profile rows.

---

## 1. The target matrix

Cell format: `value · baseline · chip` for MEASURED; `NULL/LOSS + reason`; `N/A + why`; or `GAP + what's
missing`. All provenance roots under `.trellis/tasks/06-22-n1n2n3-complete-multiprofile/artifacts/`.

### 1a. Cumulative-ladder kernels — the q4_0 repack (fills Win-A AND Win-B)

| Kernel | Win-A kernel | Win-A e2e | Win-B kernel | Win-B e2e | Win-C kernel | Win-C e2e |
|---|---|---|---|---|---|---|
| **q4_0 repack GEMV** (decode, `repack_gemv_q4_0_q8_0`) | **MEASURED** LMUL WIDE m1 vs NARROW mf2 **2.11–2.21×** · rvv VLEN128 (⚠ AUDIT DONE: the m1 arm is HAND-PLACED via the xtheadvector stamp — genuinely faster for q4_0 [2.1× micro, **1.10× t16 prefill / NULL-flat decode** e2e] but the RVV1.0 auto-gearbox emits mf2, so auto-SELECTING m1 for q4 is an unbuilt gap [selection-vs-kernel, cf gap-6]; NOT an auto-tune win, and the 2.1× washes to ~null at e2e. See `REPACK-WINA-AXIS-RECONCILED.md`); VLEN-strip 1×16 vs 2×8 **1.48×** · K1 VLEN256 `[winA-e2e/FINDING.md; k1-vlen256/e2e-SEAL-and-caveat.md]` | LMUL decode **FLAT** (≈1.05× t1 / ≈0.93× t16, memory-BW-bound) **[NULL]**; VLEN-strip **1.31×** (K1, tg32 2.12 vs 1.62 t/s, both ENGAGED, SEAL) **[MEASURED]** | **MEASURED 1.22×** · vs ggml's OWN `ggml_vec_dot_q4_0_q8_0` (10 real RVV ops) · rvv · norm=0 byte-exact `[winB-correct-baseline/FINDING.md]` | **MEASURED ~2.6×** (t16 tg64 ~7.0 vs 2.71 t/s, memory-locality regime) · rvv. **K1 0.74× LOSS** (X60 autovec beats repack path) **[disclosed §4.2]** | NONE (§4.4) | NONE (§4.4) |
| **q4_0 repack GEMM** (prefill M=4, `repack_gemm_q4_0_q8_0`) | **MEASURED** LMUL WIDE vs NARROW **1.27–1.38×** (3 distinct `.so` md5s) · rvv `[winA-e2e/FINDING.md]` | prefill **1.70× t1** (anomalous, caveated) / **1.10× t16** (clean, pp256 19.9 vs 18.0, both ENGAGED) · rvv **[MEASURED]** | **MEASURED 6.36×** · same ggml RVV baseline · rvv · 32,740 vs 208,060 ns, norm=0 `[winB-correct-baseline/FINDING.md]` | **MEASURED 5.68×** (t16 pp256 17.9 vs 3.15 t/s, matmul-bound, micro≈e2e) · rvv | NONE (§4.4) | NONE (§4.4) |

### 1b. Win-A-only kernels (no algorithm-swap-vs-ggml → Win-B = N/A)

| Kernel | Win-A kernel | Win-A e2e | Win-B kernel | Win-B e2e | Win-C |
|---|---|---|---|---|---|
| **i16 dot-reduce contraction** (`TypedWideningDotReduce…`, the **Win-A headline**) | **MEASURED 2.27–3.79×** (rvv VLEN128) / **1.8–3.6×** (K1 VLEN256) · vs tune-OFF narrow-deferred (SAME algorithm, narrow LMUL, ALSO compiler-emitted) · byte-exact vs scalar oracle `[WIN-A-2-4x-EXPLAINED.md; commit 709bb69d]` | **N/A by nature** — not a standalone llama hot-path kernel; its e2e analogue *is* the q4_0 repack-LMUL row above. NOT a gap. | **N/A** — no ggml algorithm to swap against; baseline is tune-OFF same algorithm | **STRUCTURAL NULL** — the `reduction_structure` pass toggles ON/OFF for 3.0–3.3× (real, byte-exact, I5/I7 PASS) BUT a register-kept decomposition shows pure-structure ≈**1.00×**; the 3× is ENTIRELY the per-iter `out[0]` memory round-trip (emitter artifact), not reduction-structure latency. Win-C-as-novelty NOT demonstrated `[WIN-C-DESIGN.md decomposition]` |
| **q4_0 full GEMM (AoS)** (`q4_0_q8_0_gemm`, M-knob) | **MEASURED** (knob = activation_cols M, gearbox-selected) — superseded e2e by repack; micro is the AoS-GEMM building block | GAP — no standalone e2e row (repack GEMM is the routed prefill path) | n/a (algorithm-change home is the repack, scored above) | — | NONE (§4.4) |
| **forward-pass scale / rms_norm** (`ggml_vec_scale_f32`, `rms_norm_f32`) | **N/A as tune** — `strip_lmul` attribute-togglable (default m8) but **no gearbox PASS selects it**; never measured-stamped | GAP — not wired; no e2e row | **N/A** — bit-identical reimpl of the ggml fn (same algorithm) | N/A | NONE (§4.4) |

### 1c. N1-SELECTION-divergence block-dots (Win-A *column*, but N1-sel — NOT N3 tune-on/off)

Paired same-session rvv128/K1256, byte-exact vs ggml. The **winner flips across profiles** = the N1 claim.
Provenance: `N1N2N3-LEDGER.md` §2.1/2.2; `k1-vlen256/q8_0-paired-rvv128-k1256.log`;
`q5-q41-paired-divergence.log`.

| Block-dot kernel | Win-A·micro (N1-sel: rvv128 ↔ K1256) | Win-A·e2e | Win-B·micro | Win-B·e2e | Win-C |
|---|---|---|---|---|---|
| **q8_0** (`q8_0_q8_0_block_dot`) — *load-bearing N1* | **MEASURED** VLEN→LMUL-family **REVERSAL** m2@128 vs m1@256, **~7% both ways** (rvv m2 +7.01%; K1 m1 +6.88%; static cost-model picks wrong family on K1, ≈6.5% slower) | **GAP** (gap 6) — no e2e leg confirms the selected winner runs in-llama and the margin survives | vs `ggml_vec_dot_q8_0_q8_0` — micro not headlined (microbench-only kernel) | **GAP** — not a repack-routed hot path | NONE (§4.4) |
| **q4_1** (`q4_1_q8_1_block_dot`) | **MEASURED** elision-axis flip, rvv +2.9% / **K1 +11.2%** | **GAP** (gap 6) | **MEASURED 2.47–2.48×** vs `ggml_vec_dot_q4_1_q8_1` · rvv · norm~2e-6 | **BLOCKED** — no ggml q8_1x4 quantizer / q4_1-repack routing (upstream gap, not our compiler) | NONE (§4.4) |
| **q4_0** (`q4_0_q8_0_block_dot`) | **MEASURED** within-m1 factor flip, marginal ~0.8% | **GAP** (gap 6) — reached e2e via the repack rows, but the block-dot *selection* winner has no e2e seal | (same-algo vec_dot; the Win-B algorithm home is the repack, §1a) | (see repack §1a) | NONE (§4.4) |
| **q5_0** (`q5_0_q8_0_block_dot`) | **NULL** divergence — m1-only legal set, tie ≤1.3% **[MEASURED-NULL]** | GAP | vs `ggml_vec_dot_q5_0_q8_0` (micro-only) | GAP | NONE (§4.4) |
| **q5_1** (`q5_1_q8_1_block_dot`) | **NULL** divergence — m1-only legal set, tie ≤0.05% **[MEASURED-NULL]** | GAP | vs `ggml_vec_dot_q5_1_q8_1` (micro-only) | GAP | NONE (§4.4) |

### 1d. q4_1 repack (Family-B) — algorithm-change kernel exists, e2e upstream-blocked

| Kernel | Win-A kernel | Win-A e2e | Win-B kernel | Win-B e2e | Win-C |
|---|---|---|---|---|---|
| **q4_1 repack GEMV** (`repack_gemv_q4_1_q8_1`) | **MEASURED ~1.80×** WIDE m1 (1×16) vs NARROW mf2 (2×8) · rvv VLEN128 (⚠ AUDIT DONE: m1 arm HAND-PLACED via xtheadvector; RVV1.0 auto-emits mf2 → auto-selecting m1 is an unbuilt gap, NOT an auto-tune win. See `REPACK-WINA-AXIS-RECONCILED.md`) · byte-exact (norm 0), 3-run 1.799–1.804× `[kernel-coverage/q4_1-winA-oracle-FINDING.md; commit 039fcd54]` | GAP — inherits the e2e BLOCK | code+lit COMPLETE; **MEASURED oracle PASS** norm ≤ 7.6e-6 vs scalar q4_1 dequant-matmul ref — correctness only `[same FINDING]` | **BLOCKED** — no q8_1x4 quantizer / q4_1-repack routing; spacemit `block_q4_1x16` is a different ABI (upstream, not us) | NONE (§4.4) |
| **q4_1 repack GEMM** (`repack_gemm_q4_1_q8_1`) | **GAP (gap 4b, QUICK)** — only the GEVM arm was measured; the GEMM WIDE/NARROW ablation reuses the identical harness | GAP — inherits BLOCK | code **DONE**; **MEASURED oracle PASS** norm ≤ 7.634e-6 vs scalar q4_1 dequant-matmul (nr=16×nc∈{16,32,64,336}) — correctness only `[commit 039fcd54]` | **BLOCKED** — same upstream gap | NONE (§4.4) |

### 1d-q8. q8_0 repack GEVM (Family-A symmetric, NEW 2026-06-24) — built + trellis-checked + merged

| Kernel | Win-A·micro (axis-labeled) | Win-A·e2e | Win-B·micro | Win-B·e2e | Win-C |
|---|---|---|---|---|---|
| **q8_0 repack GEVM** (`repack_gemv_q8_0_q8_0`) | **MEASURED** — *strip-count tune at mf2* (the real RVV1.0 auto-tune): at VLEN256 the gearbox selects hl=16, **1.95× faster** than hl=8 (selects the faster arm); VLEN128 hl=8 forced (16 lanes don't fit one register). Byte-exact (norm 0), oracle PASS. SEPARATELY *mf2 vs RVV0.7-m1 form* (VLEN128) = mf2 **5.5× faster** — an ISA-generation kernel comparison (q8's i32-per-position serial chain), NOT a tune, OPPOSITE of q4 `[REPACK-WINA-AXIS-RECONCILED.md; q8_0-repack-winA-oracle-FINDING.md; commit a3442997]` | **GAP (stage-2b)** — e2e `.inc`-swap pending | **LOSS @VLEN128** ~1.3–1.7× vs ggml's own `vec_dot_q8_0` (byte-exact agreement) — SAME shape-mismatch as q4_K (VLEN256-shaped block-as-lane degraded to mf2/8-lane at VLEN128) `[q8_0-repack-winA-oracle-FINDING.md Win-B]` | **reasoned-NULL/flat @VLEN128** (micro LOSS + byte-identical footprint + memory-bound decode → no win branch; the repack's regime is VLEN256 where ggml also routes repack) | N/A |

### 1d-qk. q4_K repack GEVM (K-quant, DOMINANT quant, NEW 2026-06-24) — emitter CORRECT (oracle PASS); the campaign's hardest kernel

| Kernel | Correctness | Win-A·micro | Win-B·micro | Win-B·e2e | Win-C |
|---|---|---|---|---|---|
| **q4_K repack GEVM** (`repack_gemv_q4_K_q8_K`) | **MEASURED — CORRECT** · emitter oracle PASS **WORST_NORM 7.07e-7** (8 shapes, rvv VLEN128 mf2 re-derived path), independent pre-repack ref + 2 negative controls (min/bsums @396,000× margin, sub-block-scale @2,080,000× margin); op+verifier+emitter trellis-checked (transcription faithful vs ggml, I3/I5/I7) `[q4_K-repack-oracle-FINDING.md; 240f7221/710ad067/0d5ebc5a]` | **GAP (stage-1c/2)** — strip-count tune at mf2 (like q8_0); needs the Win-A stamp wired into `RVVRepackStripWidthMaterialization.cpp` + the VLEN256 companion | **LOSS @VLEN128** ~1.5–2.1× vs ggml's OWN `ggml_vec_dot_q4_K_q8_K_vl128` (hand-tuned VLEN128 RVV) — our VLEN256-shaped block-as-lane repack forced through mf2/8-lane strips loses; fair (agreement ~e-7, same footprint). N3 shape-mismatch gap, NOT correctness `[q4_K-repack-oracle-FINDING.md Win-B]`. @K1 VLEN256 = vs ggml's OWN repack (tie-likely, oracle-pending) — GAP | **NULL (reasoned, not measured — `.inc`-swap NOT built)** — determined without integration: our repack reads a **byte-identical 2304B footprint** to ggml + decode is **memory-bound** + micro LOSES 1.5–2.1× → the e2e can only be flat (memory wall hides compute) or a decode-loss; **no win branch at VLEN128** (contrast q4_0's 2.6× which CHANGED memory traffic — q4_K's repack does NOT). Measured e2e = optional-if-ever-cheap. **q4_K's honest contribution = correct kernel-EXPANSION of the dominant quant (oracle-verified) + the cleanest N3-Gearbox MOTIVATION** (a VLEN256-shaped algorithm mismatched to VLEN128 — exactly what capability/resource-aware shape-selection should fix), NOT a perf win | N/A |
| **q4_K repack GEMM** (`repack_gemm_q4_K_q8_K`, PREFILL) | **MEASURED — CORRECT** · oracle PASS WORST_NORM **7.05e-7** (8 shapes, ROWROT 4-row-interleave control @7.5M× margin); complete repack PAIR w/ the GEVM `[q4_K-repack-GEMM-stage1b-FINDING.md; 02c41f4d/1262b3b2]` | **LOSS @VLEN128 0.59–0.89×** vs ggml `vec_dot_q4_K_vl128` — amortization NARROWS the GEVM gap (0.47-0.66→0.59-0.89) but doesn't cross 1.0 (see ledger §8b + its CORRECTION: the cause is competitor-strength × compute-density, NOT VLEN-shape — `SHAPE-AWARE-REPACK-TUNE-DESIGN.md`) | **reasoned-LOSS** (prefill compute-bound + micro LOSS → loss shows e2e; no win branch) | N/A |

### 1e. IME — N2 second family (int8→int32 vmadot); own row, not a clean A or B

| Kernel | "Win-A" | "Win-A" e2e | "Win-B" | "Win-B" e2e | Win-C |
|---|---|---|---|---|---|
| **IME matmul** (`tcrv_ime.mma`, `vmadot`) | **N/A** — no LMUL/strip/repack knob; the 4×4×8 MAC fragment is **VLEN-DERIVED** (`deriveIMEMatmulCapability`), an N1 capability hook, not an ablatable tune | (see e2e) | **5.51× RETRACTED → being RE-BASELINED** (2026-06-25, `IME-PERF-REBASELINE-FINDING.md` in flight): the 5.51× was vs the **FORBIDDEN** hand-rolled `vwmacc`+`vredsum` baseline (NOT ggml-shipped) → INVALID as a Win-B; re-measured vs ggml's REAL RVV `mul_mat`/`vec_dot` on K1 (a tie/loss = legitimate "select RVV"). Supersedes the 5.51× whatever the result | **NULL — decode AND prefill** — orig-lib: tg16 0.86× / pp32 0.98× (lib-swap, IME ENGAGED) vs non-IME 1×16 RVV · K1 tinyllama-1B Q4_0 · matrix unit can't help memory-bound M=1 decode. **Prefill probe** (reconstructed clang IME lib, pp256–1024): apparent 1.37–1.95× is a BUILD-CODEGEN artifact, NOT the IME unit — the **decode control** (tg16 1.25× in M=1, where the matrix unit physically CAN'T help) isolates it to global clang `-fno-integrated-as`/ZVFH codegen; ratio DECAYS with M (inverse of a matmul win). Clean isolation needs a SPACEMIT=OFF same-toolchain arm. **[MEASURED-NULL]** `[n2-ime/IME-PREFILL-PROBE.md]` | NONE (§4.4) |
| **N2 zero-core-branch emission** (the *proven* N2 claim) | — | — | not a perf cell: compiler-emitted vmadot runs **bit-exact 16/16** on real X60, **zero core family-name branch** (I3 verified), capability-FACT gated on `spacemit.ime` `[n2-ime/PLUGIN-SLICE.md, FOUNDATION.md]` | — | NONE |
| **N2 rapid-add PROOF** (reviewer-facing, `329421af`) | — | — | not a perf cell: `N2-RAPID-ADD-PROOF.md` — every figure reproduced by a one-liner: founding `2eeabff9` = **0 lines in `lib/Transforms/`** (core), ~1356 IME-local, 3 wiring rows; ops 2/3/4 at **0-core each**; **grep 0** family tokens in core + **0** `contains(matmul)&&contains(spacemit)` anti-pattern; selection by the `spacemit.ime` FACT. The reuse: 6 core passes ran IME with 0 IME lines | — | NONE |
| **IME `mma_slide`** (`tcrv.ime.mma_slide`, `vmadot1`, the 5th op — NEW kernel TYPE) `f2446b75` | **Win-A = slide∈{1,2,3}** window-stride knob (slide 2/3 silicon-gated) | (compute-bound prefill regime; decode NULL) | **Win-B = GAP-not-a-number** — algorithm-change A-reuse (sliding-window/conv GEMM) vs ggml RVV, but NO ggml RVV slide baseline exists → NO manufactured speedup (discipline). Defensible = algorithm-change CAPABILITY + **K1 bit-exact 16/16** (4-way slide discriminator, objdump `smt.vmadot1`=`e620322b`). **0-core** rapid-add LIVE (263 plugin LOC) | **NULL** (prefill-only; decode memory wall) | NONE |

> **BLOCK-DOT COVERAGE MEASURED (2026-06-24, `blockdot-coverage-FINDING.md`):** 5 more emit-only block-dots
> verified CORRECT vs ggml's own RVV kernel (maturity ✓, all bit-exact / fp-fold-order) + micro-timed @VLEN128:
> q5_0 **0.26×** LOSS, q5_1 **0.29×** LOSS, **iq4_nl 1.32× WIN** (gather-heavy FP4 — our split 16-lane vrgather
> beats ggml's 32-lane), tq2_0 **0.60×** LOSS, tq1_0 **0.44×** LOSS. Same-algorithm our-emit-vs-ggml (maturity+
> micro, NOT a Win-B algorithm change). **Refined N3 pattern: GATHER-heavy kernels WIN (iq4_nl, q2_K — our
> narrower split-gather); COMPUTE-bound LOSE (ggml's wider fused m2/m4 LMUL).** IQ-quants (iq2/iq3) are also
> gather-heavy → likely wins, being measured.
> **IQ-quant coverage DONE (7/7): all bit-exact CORRECT, all LOSS 5–22×** — hypothesis REJECTED. iq4_nl won on a
> tiny 16-entry REGISTER vrgather; iq2/iq3/iq1 gather from LARGE 256–1024-entry grids where **OUR emitter does
> SCALAR per-element gather (0 vluxei/vrgather) vs ggml's hardware `vluxei16` indexed vector loads** → loses
> 5–22× (worst iq3_xxs). A CONCRETE emitter lowering target (lower IQ grid/sign gather to vluxei16+vrgather),
> not a tune. **Block-dot coverage now ~17 kernels: ALL emit CORRECT (maturity ✓); 2 micro-WINS (q2_K, iq4_nl —
> gather); rest LOSE to ggml's wide-LMUL (compute-bound) or vluxei16 (IQ-gather) — both named emitter targets.**
> **FP4 family DONE (`fp4-coverage-FINDING.md`) — coverage now spans ALL quant families (K/standard/IQ/ternary/FP4):**
> mxfp4_q8_0 **bit-exact** (rel-norm 0.0 vs ggml's REAL `_vl128`, incl. the new E8M0 denormal branch) **1.20–1.22×
> WIN** — IDENTICAL mechanism to iq4_nl (split 16-lane `vrgather` over the 16-entry codebook fills VLEN128 i8m1,
> beats ggml's 32-lane i8m2). nvfp4_q8_0 **bit-exact** but **0.57× = vector-vs-SCALAR** (ggml ships NO RVV nvfp4
> kernel → N/A, not a real loss; nvfp4's 16-elem sub-blocks force sub-VLMAX 8-lane strips). **3rd micro-WIN
> (q2_K, iq4_nl 1.32×, mxfp4 1.21×) → the rule HOLDS: register `vrgather` over a tiny ≤16-entry FP4 codebook at
> the VLEN-native 16-lane shape WINS; the win generalizes WITHIN the tiny-codebook bucket, NOT to large IQ grids
> (those are an indexed-memory `vluxei16` PARITY target, a distinct mechanism — advisor-locked, over-claim #9).**

### 1f. Emit-only kernel families (no togglable Win-A; not wired; grouped per inventory)

These carry **no `TunableScheduleOpInterface`** and **no `lookupRVVScheduleDescriptor` case** → Win-A is
genuinely **N/A (hard-pinned / emit-only)**. Win-B is per-op-vs-ggml but **micro-only / not measured**; e2e is
**GAP (not-wired, default-no per honesty discipline)**. Win-C `NONE` globally. Grouped to avoid ~30 noise rows.

| Family (ops) | Win-A | Win-B kernel | Win-B/all e2e | Win-C |
|---|---|---|---|---|
| **K-quant block-dots** q6_K/q4_K/q5_K/q2_K/q3_K (+aux32) | **N/A** emit-only — *but* q4_K Win-A knob = **emitter parametrization** (deferred, scoped): fixed-m2 emit IS why Win-B loses to ggml `_vl256` `[commit 79db10f0]` | **MEASURED, MIXED** vs ggml's OWN `_vl256` (K1 VLEN256, norm 5e-7..3e-6): q2_K **WIN 1.016×**, q5_K **TIE 0.998×**, q4_K **LOSS** (ggml 1.72×), q6_K **LOSS** (ggml 2.26×), q3_K **LOSS** (ggml 2.13×) — single-LMUL `_generic` fp-order port vs ggml's hand-tuned nibble-split `[commit d27c512a]` | **GAP** not-wired (no §7 e2e row) | NONE (§4.4) |
| **IQ-quant block-dots** iq4_xs/iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq1_s/iq1_m | **N/A** emit-only | **MATURITY: 5/8 vluxei16-DONE** (`vluxei16-iq-gather-FINDING.md`): all bit-exact CORRECT, scalar-gather LOST 5–22×, now revectorized to hardware `vluxei16` (the SAME gather ggml uses → gap-closed, NOT "beat ggml"): **iq1_s** `f27d529c` 7.4→2.3×; **iq1_m** `740c1617` 9.3→5.8×; **iq3_xxs/iq3_s** `bf3c0193` 22.35→8.14× / 5.88→5.04×; **iq2_xxs** `945afada` 6.81→**1.91×** (closest-to-parity; signs64 blocker RESOLVED via emit-const cross-group fold, NO ODS). Follow-ons iq2_xs/iq2_s (same approach); iq4_xs untouched. Residual = narrow gather-LMUL + scalar index/sign assembly (NOT the gather) | **GAP** not-wired (compute-side → e2e decode NULL, memory wall) | NONE (§4.4) |
| **Ternary block-dots** tq2_0/tq1_0 | **N/A** emit-only | vs `ggml_vec_dot_tq{2,1}_0_q8_K` — **GAP** | **GAP** not-wired | NONE (§4.4) |
| **FP4-codebook block-dots** iq4_nl/mxfp4/nvfp4/q1_0 | **N/A** — `coreLmul` read in emitter but **no pass stamps it** (attr-present-not-selected) | **MEASURED @VLEN128 (`fp4-coverage-FINDING.md`):** iq4_nl **1.32× WIN**, mxfp4 **1.21× WIN** (both bit-exact vs ggml's real `_vl128`; tiny-codebook `vrgather`); nvfp4 bit-exact but **N/A** (ggml ships no RVV nvfp4 kernel — vs-scalar 0.57×) | **GAP** not-wired (micro-only; compute-side → e2e NULL for decode by the memory wall) | NONE (§4.4) |
| **Forward-pass hard-pinned** silu / softmax / quantize_row_q8_0 / rope_norm (fixed f32m2) | **N/A** hard-pinned (no `strip_lmul` knob) | **N/A** bit-identical reimpl of the ggml fn (same algorithm) | **GAP** not-wired | NONE (§4.4) |
| **q4_0 GEMM-tile** (`q4_0_q8_0_gemm_tile`) | **N/A** — no own knob; composes the parent GEMM's M-knob | M× `ggml_vec_dot_q4_0_q8_0` (algorithm = shared weight decode) — building block, superseded e2e by repack | **GAP** (no standalone row) | NONE (§4.4) |

**Win-C column — CORRECTED (2026-06-24):** a `reduction_structure` pass was built + toggles ON/OFF for 3.0–3.3× on the i16 dot-reduce body, but a register-kept decomposition shows the PURE structural delta is ≈1.00× (NULL) — the 3× is a per-iter `out[0]` memory round-trip, an emitter artifact, NOT a structural win. **Win-C-as-structural-novelty is NOT demonstrated.** The pass is kept (correct, checked, orthogonal) as the structural ENABLER of the Win-A LMUL sweep. Other kernels: N/A-monolithic. No structural Win-C is claimed; the repack is hand-authored Win-B, correctly NOT relabeled Win-C.

---

## 2. Prioritized gap-filling plan

> **Consolidation status (2026-06-24) — folded into §1 above:**
> - ✅ **Gap 4 (GEVM) + Gap 2** CLOSED — q4_1 repack GEVM Win-A·micro **~1.80×** (byte-exact) + q4_1 GEMM
>   scalar-oracle **PASS** (norm ≤ 7.6e-6). `[q4_1-winA-oracle-FINDING.md; 039fcd54]`
> - ✅ **K-quant Win-B·micro** (all 5) vs ggml `_vl256` — honest MIXED (q2_K win / q5_K tie / q4_K·q6_K·q3_K
>   loss); the losses MOTIVATE the q4_K Win-A knob. `[kquant-winB-micro-FINDING.md; d27c512a]`
> - ⚠️ **Gap 8 (Win-C)**: pass BUILT + trellis-check PASS, but **structural novelty NULL** — decomposition shows pure-structure ≈1.00×, the 3× pass-ON/OFF is a per-iter memory round-trip (emitter artifact). Win-C-as-novelty NOT demonstrated; pass kept as the Win-A structural enabler. `WIN-C-DESIGN.md`.
> - **Quick-measurement gaps are now largely EXHAUSTED.** Remaining value = the *transplanting* kernels
>   (K-quant repack, IME prefill) and Win-C — all code-heavy, deliberate efforts, not microbench reruns.
>
> **Still open & quick:** Gap 4b (q4_1 repack GEMM Win-A WIDE/NARROW ablation — GEVM done, GEMM reuses the
> identical harness) · Gap 6 (block-dot SELECTION e2e seal, esp. the load-bearing q8_0 VLEN-flip).

Ordered by value × tractability. "Quick" = existing kernel + existing harness, just bench on rvv/K1.

### (a) QUICK measurements — existing kernels, no new HW (do first)

1. **Gap 4 — q4_1 repack Win-A (LMUL/strip) ablation · micro.** *Highest tractability.* The q4_1 repack
   GEMV/GEMM op + the VLEN→half_lanes / RVV-gen strip-width gearbox arm **already exist and emit WIDE/NARROW
   per capability**; it is the exact ablation already run for q4_0 (`winA-e2e/FINDING.md`). Reuse
   `ablation_micro.cpp`, swap the q4_1 `.inc`, run on rvv → same-day number. (e2e inherits gap 1's BLOCK, so
   **micro-only is the realistic deliverable**.)
2. **Gap 2 — q4_1 GEMM numeric oracle · micro.** The q4_1 GEVM already has a scalar-ref + in-tree
   `GgmlBlockDotQ41Q81Op` cross-check harness (`kernel-coverage/q4_1-emit/`); **mirror it for the GEMM** to
   upgrade "code-complete, oracle-deferred" → a measured Win-B micro. No new HW; a bit more work than gap 4
   (write the GEMM scalar ref + offset-verify).
3. **Gap 6 — block-dot SELECTION e2e seal.** Medium. The selection winners are already enumerated/measured;
   what's missing is **wiring the selected block-dot as the in-llama kernel** and confirming the per-profile
   margin survives (reuse the engagement+correctness SEAL pattern already proven for the VLEN-strip on K1).
   The **q8_0 VLEN-flip is the load-bearing N1 result and has micro only** → this is the highest-*value* quick-ish
   gap, just more plumbing than a microbench rerun.

### (b) NEEDS NEW kernel / infra (medium-to-large)

4. **Gaps 1 + 3 — q4_1 GEVM/GEMM e2e.** BLOCKED on three **ggml-side** pieces to author: a q8_1x4
   mat-quantizer, q4_1-repack mul_mat routing, a q4_1 GGUF model. Upstream-shaped work (mirrors the q4_0 e2e
   seal), not a measurement, **not our-compiler-side**.
5. **Gap 7 — methodology-clean IME Win-A or Win-B cell.** Needs either an IME-emitting-pass tune ablation
   (on/off) or an IME-vs-ggml-shipped-IME-backend comparison to replace the current "vs competent RVV" number.
   New harness design.

### (c) BLOCKED / out-of-scope (cannot quick-fix here)

6. **Gap 5 — IME e2e in a matmul-bound regime.** HW-blocked: only K1 has IME silicon, ~7 GB RAM caps models at
   ~1–3B (all memory-bound decode). The 5.51× kernel win **structurally cannot surface e2e** on a runnable
   model; needs larger-RAM IME hardware.
7. **Gap 8 — Win-C (automatic structural-transform pass).** Pass BUILT + trellis-check PASS, but **structural novelty NOT demonstrated** (2026-06-24): a hand-written register-kept decomposition shows the pure reduction-structure delta ≈1.00× — the 3× pass-ON/OFF is ENTIRELY the per-iter `out[0]` memory round-trip (an emitter artifact), not structural latency. The pass is kept (correct/checked/orthogonal) as the structural enabler of the Win-A LMUL sweep, not a standalone Win-C. Related N1→N3 ("winning-ALGORITHM-differs-by-profile") remains OPEN.

**Dropped (not gaps):** all RVV0.7/C920 cells (C920 Win-A, q8_0 third-profile probe, Fedora coherent-llama e2e
block, RVV0.7 repack-GEVM re-scope) — out of scope per "drop RVV0.7," deliberately absent.

**Bottom line:** the matrix is **dense for q4_0** (Win-A + Win-B, both micro AND e2e, incl. honest NULL/LOSS)
and the **N1 q8_0 VLEN selection flip**; the **cheapest real wins are the q4_1 repack Win-A ablation (micro) and
the q4_1 GEMM numeric oracle** — both reuse existing kernels/harnesses on existing hardware. Everything else is
ggml-upstream-blocked (q4_1 e2e), HW-blocked (IME large-model), or a new-pass effort (Win-C).

---

## 3. N2 — how we PROVE rapid family-add by reusing the abstraction (怎么证明快速新增/复用/发挥 N2 抽象)

**Claim:** we added the **IME** (Spacemit X60 IME1 int8→int32 matrix extension) as a SECOND hardware family by
**reusing the N2 plugin abstraction, not by hacking the core** — and the proof is mechanical, not rhetorical.

**Headline quantities:**
- **Core LOC changed ≈ 0** — *zero* lines in any core selection/materialization pass (`lib/Transforms/*`).
- **Family-local plugin LOC = 1918** — all NEW, fully isolated under `lib/Plugin/IME/`, `lib/Dialect/IME/`,
  `include/.../IME/`. The entire cost of the family lives here.
- **4 logical registration sites = 9 purely-additive lines, 0 deletions** across 7 wiring files (table rows +
  `add_subdirectory` only — never `if (family==…)`).
- **6 reused common interfaces**, modified-by-IME = 0.

**Proven THREE ways the core changed nothing (the load-bearing proof):**
1. **git numstat of the introducing commit `2eeabff9`** ("N2 PROOF: IME second family added as a plugin"): **no
   `lib/Transforms/*` file appears in the diff at all.** Only touched core-wiring = the two Builtin TABLE files
   (additive rows) + CMake = 9 added / 0 deleted.
2. **Whole-source grep for any family token**: `grep -rinE '\bime\b|spacemit|vmadot' lib/Transforms/` ⇒ **0
   hits** (the 195 case-insensitive "ime" substrings are all `runtime`/`time`/`RuntimeGuard`…). In
   `lib/Conversion/EmitC/` family tokens appear **only in comments**.
3. **The core dispatches generically over the registry**: `registry.collectVariantProposals(request, …)`
   (`VariantMaterialization.cpp:596`) and `tryConvertModuleWithRegisteredBackend(module)`
   (`EmitCLowerableMaterialization.cpp:268`). **No `if (family==…)` anywhere.**

**Capability-FACT dispatch, NOT string-matching:** the core dispatches on the **presence of a derived
capability provider**, never a family name. The discriminator `lookupProviderByID("spacemit.ime")->isAvailable()`
is **family-local** code keyed on the family's OWN id; `"spacemit.ime"` is defined **exactly once**
(`IMEExtensionPlugin.cpp:25`) — the plugin queries itself, the core names no family. That provider only exists
because `deriveIMEMatmulCapability` (line 138) derived it from validated ISA evidence (march token `xsmtvdotii`
⇒ `vmadot`, int8→int32; VLEN/SEW ⇒ 4×4×8 MAC). The immature alternative we **provably avoided** —
`if (op.getName().contains("matmul") && target.march.contains("spacemit")) emitVmadot()` — exists in no core
pass.

**Complementary-arm negative test (proves fact-gating):** `ime-mma-capability-absent-negative.mlir` — an
RVV-but-not-IME target does NOT satisfy `lookupProviderByID("spacemit.ime")`, so IME **declines** and the op
routes elsewhere; **no** `vmadot`/`spacemit` token is materialized. Dispatch is fact-gated, not name-hardcoded.

**The 6 reused interfaces (RVV uses them, IME reuses *unmodified*):**
1. `TCRVEmitCLowerableOpInterface` — declared in IME ODS exactly like RVV (`IMEOps.td:36-38`); the generic
   EmitC route keys on the interface, not the dialect name.
2. `ExtensionPluginRegistry` / `ExtensionBundleRegistry` — IME registers via the same
   `ExtensionPluginRegistrationFn` signature every family uses; core iterates it generically.
3. `BackendEmissionRegistry` — IME's `registerIMEBackendEmitter` body is **byte-for-byte** RVV's pattern
   (`static const …Driver driver; registry.registerBackend(driver);`).
4. `CapabilityDescriptor` / `lookupProviderByID` — same shared capability model RVV queries with
   `isCapabilityAvailableByID`; neither touches it.
5. `tcrv.exec` variant/dispatch/legality/boundary orchestration — IME wrote **no** orchestration; it filled in
   plugin vtable hooks called BY the generic core passes.
6. Common EmitC→C++ route — stock `mlir-translate --mlir-to-cpp` (`emitc::translateToCpp`); IME added no exporter.

**Incremental leverage (the wiring cost was paid ONCE):** a **4th op (`mma_su`/`vmadotsu`, mixed-sign) was added LIVE this session** (commits `70bb845a`+`a75adc27`, **0 core / 0 wiring**, trellis-checked, **silicon bit-exact 16/16 on X60**, operand convention confirmed — see standalone `n2-ime/N2-RAPID-ADD-SHOWCASE.md`); the 2nd IME op (tiled matmul, commit `0b8c6168`) and
3rd op (`mma_u`/`vmadotu`, commit `c7069111`) each touched **ZERO** registration / core-wiring files and added
**ZERO** core lines — pure family-local growth. **It runs on real silicon:** the compiler-emitted kernel was
cross-built with the SpacemiT GCC15.2 fork, objdump-confirmed real `smt.vmadot`, and ran **bit-exact (16/16 ==
scalar oracle)** on a real X60 pinned to the IME harts.

> **The story in one line:** family addition is fast *because nothing in the core had to learn what "IME" is* —
> 1918 isolated lines + 9 additive wiring lines + 0 core lines, dispatched by a derived capability fact and
> proven by the complementary-arm test, with the 2nd/3rd op confirming the wiring cost was paid exactly once.

---

## 4. Kernel-expansion shortlist (highest-value next kernels)

**Derived from the campaign's central honest finding** (§7.1): *a pure-compute kernel win dilutes to flat/null
in bandwidth-bound decode; a win transplants to e2e only if it changes MEMORY behavior (repack/layout) or is
prefill/matmul-bound.* So the highest-value new kernels are **repack-class** (memory-behavior change →
transplants) on **dominant real-model quants**, and the **one IME regime not HW-structurally-blocked**.

### RVV (highest value first)

1. **K-quant repack (q4_K / q6_K · GEMV + GEMM).** *The standout.* **Verified against the inventory: only
   q4_0 (GEMM+GEMV) and q4_1 (GEMV) repacks exist — there is NO K-quant repack sibling; q4_K/q6_K are
   emit-only block-dots.** q4_K/q6_K are the **dominant quants in real GGUF models**, the repack class is the
   one that *transplants to e2e* (the q4_0 repack's 5.68× prefill / ~2.6× decode are the campaign's only
   holding e2e wins), and the cell is a **total gap** (no Win-A, no Win-B-micro, no e2e). Highest
   value-density: dominant quant × transplanting class × empty cell.
2. **q5_0 / q5_1 / q8_0 repack GEMV+GEMM.** Extend the proven block-as-lane mechanism to the other tunable
   block-dots that currently have *only* a microbench/selection leg (§1c), giving them an e2e home and closing
   gap 6 for the load-bearing q8_0 VLEN-flip.
3. **q4_1 repack Win-A ablation wiring** (already-built kernel; this is gap 4 — listed here because it is the
   cheapest *new measurement* against an existing kernel, not a new kernel).

### IME (the one regime where the 5.51× could surface e2e)

1. **IME prefill / matmul-bound path** — a tiled `vmadot` GEMM driving **prefill** (M≫1) rather than M=1
   decode. The central finding says the matrix unit *can* help when matmul-bound; the current e2e NULL is
   purely an M=1-decode artifact. This is the **only IME regime not structurally HW-blocked** (gap 5 blocks the
   *large-model* path, not a prefill-bound one on a runnable model) and is the path to a non-null N2 e2e number.
2. **IME-emitting-pass tune ablation** (on/off) to give IME a methodology-clean Win-A cell (closes gap 7).

---

*Provenance root: `.trellis/tasks/06-22-n1n2n3-complete-multiprofile/artifacts/`. Source of truth:
`N1N2N3-LEDGER.md` §7. Cell taxonomy {MEASURED / NULL-LOSS / N/A / GAP} and the N1-selection-≠-Win-A-tune
distinction are load-bearing — do not collapse them.*
