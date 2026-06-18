# N3 ablation table — passes ARE the source of speedup (measured evidence + gap list)

**Read-only synthesis, 2026-06-18.** Every number traces to an archived `ssh rvv`
artifact (riscv64, clang 18.1.3, VLEN=128) by `file:line`. Nothing here was re-run.
Projections / unbacked claims are marked **GAP**. Framing = `corrected-campaign-plan.md`
sections 1–5; spec yardstick sanction = `generation-selection-tuning.md:55` ("实测赢
scalar 且赢 naive RVV") and `index.md:26`.

## 0. The two axes (do not conflate — this is what makes it an ABLATION)

The claim "the passes ARE the source" is proven by the **ABLATION axis only**:

- **ABLATION = pass ON ÷ pass OFF on the SAME compiler.** OFF = the spec's MVP
  placeholder (fixed single candidate / fixed LMUL / fixed unroll). This isolates the
  pass as the *cause*. This is the headline of every row.
- **YARDSTICK = vs genuine-scalar, vs competent-naive-RVV** (spec-sanctioned external
  references). These prove *absolute competitiveness*, NOT sourcing. Reported as
  separate columns; a yardstick number NEVER stands in for the ablation cell.
- **ggml is NOT a baseline (I9)** — only a byte-exact correctness oracle. ggml-relative
  ratios (1.012×, 0.842× …) appear ONLY as oracle context, NEVER as a pass speedup.

**Regime split is hard** (the two never share a column or get averaged):
- **Deferred-wide contraction** (i8 byte, i16 dot-reduce, u8) — the real N3 win, max-legal-LMUL.
- **ggml block-dots** (q4_0/q8_0/q4_1/q5_x) — parity-to-modest regime, measured>static + strip_elision.

---

## 1. THE ABLATION TABLE

Columns: **ABLATION (ON÷OFF, same compiler)** = the sourcing proof | **YARDSTICK
vs scalar** | **YARDSTICK vs naive-RVV** | source `file:line`.
Paths are relative to
`/home/kingdom/phdworks/TianchenRV/.trellis/tasks/archive/2026-06/`.

### REGIME 1 — deferred-wide contraction (max-legal-LMUL; the headline)

| Row | Pass / kernel | ABLATION ON÷OFF (same compiler) | vs scalar | vs naive-RVV | source |
|---|---|---|---|---|---|
| **A-i16** | max-legal-LMUL, **i16 dot-reduce, e2e autotuner** | **5.93–11.15×** (wide ÷ per-iter compiler-default = the fixed-narrow MVP placeholder) | 3.97–7.53× | 2.11–3.80× | `06-14…/artifacts/p-b8-i16-e2e/dot-reduce-i16/pb8_lamp_3way_ssh_rvv_stdout.txt:9-39`; verdict `…/p-b8-i16-e2e/PB8_VERDICT.md:43-50` |
| **A-i16-emit** | max-legal-LMUL, i16 dot-reduce, emitted-body (selection was the wall) | **5.71–11.14×** (wide ÷ per-iter default) | 3.87–7.52× | 2.22–3.78× | `…/p-b7-second-family-ceiling/dot-reduce-i16/pb7_lamp_3way_ssh_rvv_stdout.txt:9-39`; `…/p-b7-second-family-ceiling/LAMP_EMITTED_BODY.md:31-38` |
| **A-i8** | max-legal-LMUL, **i8 byte, e2e autotuner** | **GAP — needs rvv: same-compiler pass-OFF (forced-narrow i32m1) was never TIMED.** The i8 narrow path is "vestigial/unreachable at budget 32" (`pb5_verdict.txt:47-55`); only the *external* competent-naive was timed. ON÷OFF cell unfilled. | 4.11–10.82× | 3.29–5.44× | `…/p-b5-e2e-autotuner/pb5_verdict.txt:22-24`; invocation `…/p-b5-e2e-autotuner/REPRODUCE.md:8-12` |
| A-i8-emit | max-legal-LMUL, i8 byte, emitted-body | **GAP** (same as A-i8: no same-compiler narrow timed) | 4.49–10.74× (median 6.84×) | 3.16–5.40× (median 3.52×) | `…/p-b3-deferred-wide-emission/pb4_lamp_3way_emitted_body_ssh_rvv_stdout.txt:7,13,19`; verdict `…/pb4_lamp_3way_verdict.txt:11-12` |
| A-u8 | max-legal-LMUL, u8 product-reduce (hand-variant ceiling) | **8.22–14.69×** (wide ÷ per-iter default) | 4.76–9.91× | 3.71–4.99× | `…/p-b7-second-family-ceiling/u8-product-reduce/remote_run_stdout.txt:7-37` |

Per-n detail for the strongest measured ablation (A-i16, e2e autotuner, `pb8_…stdout.txt`):

| n | ABLATION wide÷per-iter | vs scalar | vs naive |
|---|---|---|---|
| 257 | 6.63× | 4.28× | 3.07× |
| 256 | 8.09× | 5.54× | 3.02× |
| 1024 | 10.17× | 6.88× | 3.56× |
| 4096 | 11.16× | 7.53× | 3.80× |
| 16384 | 6.82× | 4.83× | 2.39× |
| 65536 | 5.93× | 4.02× | 2.11× |

### REGIME 2 — ggml block-dots (measured>static; strip_elision)

Block-dot ablations are computed from **absolute ns, ggml-free** (the ggml column in
the source is correctness-oracle context only, I9). OFF = the static cost-model argmin
(the fixed-heuristic MVP); ON = the measured-fastest legal shape.

| Row | Pass / kernel | ABLATION ON÷OFF (same compiler, absolute ns) | vs scalar | vs naive-RVV | source |
|---|---|---|---|---|---|
| **C** | **measured>static**, q4_1 @ rv64gcv | **1.20×** = static m1/4/elided 1517.4ns ÷ measured m1/1/elided 1262.8ns (loss→win: the static pick is the SLOWEST legal elided shape) | **GAP** (block-dot vs-scalar not measured here) | **GAP** | `06-15…/artifacts/inc10-measurement-tuner/tuning_record.txt:61,64`; narrative `…/inc10…/RESULTS.md:77-85` |
| **D** | **unroll inversion** q4_0 vs q4_1 (one fixed factor can't serve both) | q4_1 prefers factor=1: fixing factor=4 costs **1.20×** (1517.4÷1262.8). q4_0 prefers factor=4: fixing factor=1 costs **1.17×** (m1/1/elided 1198.2 ÷ m1/4/elided 1023.1). **Opposite directions ⇒ per-kernel select necessary.** | n/a (intra-pass) | n/a | q4_0 ladder `…/inc10…/tuning_record.txt:9-19`; q4_1 ladder `…/inc10…/tuning_record.txt:60-71`; ordering `…/inc10…/RESULTS.md:97-101` |
| C-q4_0 | measured>static, q4_0 @ rv64gcv (confirms static) | **1.00×** — measured pick = static argmin (m1/4/elided); measurement CONFIRMS, doesn't overturn | GAP | GAP | `…/inc10…/tuning_record.txt:18-19`; `…/inc10…/RESULTS.md:73,87-95` |
| C-q8_0 | measured>static, q8_0 @ rv64gcv (refines factor) | **1.03×** = static m2/2 876.2ns ÷ measured m2/1 851.3ns (refines factor 2→1, near-tie cluster) | GAP | GAP | `…/inc10…/tuning_record.txt:32-45`; `…/inc10…/RESULTS.md:75,98` |
| **E** | **strip_elision capability gate** (Zvl128b on/off), q8_0 | **Capability divergence PROVEN** (stamp+emission diverge elided↔robust by the single Zvl128b boolean): cost 1005 (elided, full-V) vs 1310 (robust, zve32x). **Timing of the gate** (robust 1055ns ÷ elided 884ns ≈ **1.19×**, VLEN=128 board) measured; **the zve32x/Zvl128b-OFF branch itself is a VLEN=128 PROXY — no sub-128 silicon.** | GAP | GAP | divergence+cost `06-15…/artifacts/inc7-q8_0-breadth/RESULTS.md:96-105`; elided-vs-robust ns `…/inc7…/RESULTS.md:116-122`; proxy caveat `…/inc10…/RESULTS.md:176-181` |
| **F** | **tune-record present/absent/stale → fail-closed** | **Safety property, NO perf number** (do not manufacture one). Proven by lit: record present → measured stamp; absent → static fallback; stale/illegal → revalidate + fall back, never stamps illegal shape (I7). | n/a | n/a | `06-15…/artifacts/inc10…/RESULTS.md:103-116`; lit `test/Conversion/RVV/rvv-q4-1-q8-1-block-dot-measurement-tuner.mlir` |
| **B** | **register-budget prune binds** (budget 12/16 vs 32) | **Crossover PROVEN to bind** (resource-derived, not lookup): i8 budget≥32→wide i32m8, narrow otherwise; i16 budget≥16→wide, budget<16→pruned to narrow. **No ON÷OFF *timing* at the pruned budget** (the narrow body is the same one A measures against). | n/a (covered by A) | n/a | i16 crossover `…/p-b8-i16-e2e/PB8_VERDICT.md:25-36`; i8 `…/p-b5-e2e-autotuner/pb5_verdict.txt:33-38,56-60` |

---

## 2. "Passes ARE the source" — per-row narrative (grounded in the cited numbers)

- **A (max-legal-LMUL) — HEADLINE.** With the pass ON the compiler emits the
  deferred-wide body (load→single-widening product→loop-carried wide accumulate→ONE
  trailing reduce); with it OFF it emits the per-iteration fixed-narrow placeholder.
  Same compiler, same input: ON is **5.9–11.2× faster than its own OFF default** on the
  i16 family (`pb8_…stdout.txt:9-39`), and the same body also beats genuine-scalar
  4.0–7.5× and competent-naive-RVV 2.1–3.8×. The selector — `selectRVVDotReduce
  DeferredWideMaxLegalLMULRung` — is the only thing that changed, so it IS the source.

- **B (budget prune binds).** The wide rung fires *only* when the stamped vreg-budget
  fact admits it (i16: ≥16; i8: ≥32, reserve 8); below threshold the wide candidate is
  pruned and the compiler falls back to narrow. This is resource-derived legality, not a
  hardcoded lookup — proven by the live crossover (`PB8_VERDICT.md:25-36`).

- **C (measured>static).** A static cost model alone MIS-RANKS q4_1: it picks
  m1/4/elided, the **slowest** legal elided shape (1517.4ns); the board crowns m1/1/elided
  (1262.8ns). ON÷OFF = **1.20×**, a genuine loss→win. Measurement, not the static guess,
  is the source of that 20% (`tuning_record.txt:61,64`).

- **D (unroll inversion) — GENERALITY CENTERPIECE.** q4_1 wants factor=1; q4_0 wants
  factor=4. Forcing either fixed factor on the other costs ~1.17–1.20×. **No single
  fixed-unroll heuristic can serve both**, which is exactly why per-kernel measured
  selection (not a global constant) is necessary — the proof that this is ONE mechanism,
  not a hand-tuned kernel (`tuning_record.txt:9-19,60-71`).

- **E (strip_elision capability gate).** Flipping a single N1 capability fact (Zvl128b)
  flips the emitted shape elided↔robust — an N1 *capability* divergence distinct from the
  resource-budget prune of B. On the VLEN=128 board the elided shape is ~1.19× faster than
  the robust fallback (`inc7…RESULTS.md:116-122`). Honest caveat: the OFF (sub-128) branch
  is a proxy, not real silicon.

- **F (tune-record fail-closed).** A stale record naming a capability-pruned shape is
  revalidated and rejected; the pass falls back to the static argmin and NEVER stamps an
  illegal shape (I7). This is a correctness/safety guarantee, not a speedup — it makes the
  measured selection *trustworthy* (`inc10…RESULTS.md:103-116`).

---

## 3. GAP LIST — the minimal NEW ssh-rvv measurements to reach publishable rigor

Prioritized. Each gap states the exact emission/invocation and which cell it fills.

**P1 — the same-compiler ON/OFF cells the table currently can't fill (the core ablation):**

1. **Block-dot ablation timed as ON vs OFF directly (rows C/D), not via ggml.** For each
   of q4_0/q8_0/q4_1, emit BOTH the measured-pick shape AND the static-argmin shape from
   the SAME compiler and time them head-to-head on rvv:
   - ON: `tcrv-opt q4_1_input.mlir "--tcrv-rvv-materialize-q4-1-schedule=march=rv64gcv tune-record=tuning_record.txt"` → emits m1/1/elided.
   - OFF: same invocation **without** `tune-record=` → emits the static m1/4/elided (the MVP placeholder).
   - Lower both `--tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`, time min-of-N.
   - **Fills:** rows C, D ON÷OFF cells with a clean same-compiler delta (currently inferred
     from the ggml-relative ladder; the absolute-ns ratio is correct but should be a
     direct A/B timing run, not a back-computation).

2. **i8 deferred-wide same-compiler pass-OFF (rows A-i8, A-i8-emit).** Emit the i8 byte
   kernel with the wide rung FORCED OFF (the fixed-narrow MVP placeholder) and time it
   against the wide ON emission:
   - OFF: feed the i8 pre-realized body with a budget stamped **< 32** (e.g. sed budget=12,
     as the i16 lit already does at budget 12) through
     `--tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries
     --tcrv-rvv-lower-to-emitc` → forces the narrow i32m1 body.
   - ON: budget=32 → wide i32m8 (the body `pb5_verdict.txt` already times vs external naive).
   - **Fills:** A-i8 / A-i8-emit ABLATION cells (currently GAP — only external-naive timed).

**P2 — close the generality gap (block-dot families):**

3. **q5_0 and q5_1 ON and OFF, timed.** Both are byte-exact-only today (perf explicitly out
   of scope: `inc21-q5_0/RESULTS.md:141`, `inc22-q5_1/RESULTS.md:106`). Emit each with and
   without `tune-record=` (`--tcrv-rvv-materialize-q5-0-schedule` / `…-q5-1-schedule`), lower
   to C, time min-of-N on rvv. **Fills:** two new rows in REGIME 2 (closes the
   q4_0/q8_0/q4_1 → q5_x generality gap named in `corrected-campaign-plan.md:47`).

**P3 — make the capability/budget gates self-timing (rows B, E):**

4. **Budget-prune ON/OFF timing (row B).** Beyond proving the crossover binds, time the
   wide (budget=32) vs the pruned-narrow (budget=12) i16/i8 emissions head-to-head so B has
   its own number instead of folding into A. (Same emissions as gap #2, relabeled for B.)

5. **strip_elision real OFF branch (row E).** The elided÷robust 1.19× is on the VLEN=128
   board; the Zvl128b-OFF (zve32x) branch is a proxy. If/when sub-128 RVV silicon is
   available, time the robust shape there to replace the proxy. Until then, keep E flagged
   "VLEN=128 proxy for the OFF branch."

**P4 — the documented PRIZE (new win, not a current claim):**

6. **Wire max-legal-LMUL into the block-dot realization owner**, then ablate ON/OFF on rvv.
   Today block-dots sit at parity and the live llama integration ran ~1.8× SLOWER
   (i8mf4→i32m1, 4 lanes vs ggml's 16; `corrected-campaign-plan.md:48-50`). Emitting the
   block-dots with the wide-LMUL rung and proving ON>OFF on board is the real e2e llama win
   — explicitly **not claimed yet**.

---

## 4. HONESTY LEDGER

### Fully measured (ABLATION ON÷OFF + both yardsticks on real rvv)
- **A-i16 (e2e autotuner) and A-i16-emit:** ON÷OFF 5.9–11.2× same-compiler, AND vs-scalar
  4.0–7.5×, AND vs-naive 2.1–3.8×. The cleanest, strongest sourcing proof. The "per-iter
  compiler default" IS the literal fixed-narrow MVP placeholder, so wide÷per-iter is a true
  same-compiler ablation — and it is *larger* than vs-naive because the compiler's own
  default is worse than a competent hand-naive.
- **A-u8:** same structure (8.2–14.7× ON÷OFF), hand-variant ceiling.
- **C, D (q4_1 / q4_0 unroll inversion):** ablation ON÷OFF 1.17–1.20× from absolute ns,
  ggml-free. Loss→win and the per-kernel inversion are both real and measured.

### Partially measured (one axis solid, the other a gap)
- **A-i8 / A-i8-emit:** yardsticks (vs-scalar 4.1–10.8×, vs-naive 3.3–5.4×) fully measured
  and selector-driven e2e; but the **same-compiler ON÷OFF was never timed** (narrow path
  vestigial at budget 32) → ablation cell is GAP #2.
- **E (strip_elision):** capability divergence + elided/robust timing measured, but the
  OFF (sub-128) branch is a **VLEN=128 proxy**.
- **B (budget prune):** crossover proven to bind (live, unit-tested), but no standalone
  ON/OFF timing distinct from A.

### Gap (no measured datum)
- **q5_0 / q5_1:** byte-exact correctness only; **perf unmeasured** (gap #3).
- **Block-dot vs-scalar / vs-naive:** NOT measured in inc10. The only block-dot
  vs-scalar data that exists is the **counter-evidence**: grouped-u2 is a **regression**
  (0.50–0.61× vs scalar-C) and packed-i4 is **parity/no-win** (0.90–1.02×)
  (`06-14…/research/gearbox-current-state.md:91-92`). The campaign plan's "block-dots vs
  scalar 1.5–4.3×" does **not** trace to any pulled artifact — treated as claim-to-verify,
  not data; the corresponding cells are GAP.

### Regime split — kept intact
Deferred-wide ON/OFF (~5.9–14.7×) and block-dot ON/OFF (~1.0–1.20×) are in **separate
tables and never averaged**. They are different regimes with different mechanisms
(max-legal-LMUL live vs latent).

### Above-layer — explicitly EXCLUDED from all pass rows
- **GEMM M-block / `activation_cols`** (inc26-gemm-g3): measured M=6 @ 1.192× vs
  per-(row,col) `vec_dot` (`06-15…/artifacts/inc26-gemm-g3/tuning_record.txt:14-17`). This
  is **above-layer cache-blocking tiling geometry**, the verdict itself states it is "NOT
  capability-gated … the headline is N3 measurement-selection, not N1 divergence"
  (`…/inc26-gemm-g3/INC26_G3_VERDICT.md:198-200`). The dropped ~1.56× GEMM claim
  (`corrected-campaign-plan.md:4-6`). **Footnote only — never a pass-row cell.**
- **Weight repacking** (Q4_0_8x8 storage) = frontend / I9.
- The block-dot **selection mechanism** may be cited as generalizing to a tiling key, but
  its *number* (M=6, 1.192×) is not a pass claim.
