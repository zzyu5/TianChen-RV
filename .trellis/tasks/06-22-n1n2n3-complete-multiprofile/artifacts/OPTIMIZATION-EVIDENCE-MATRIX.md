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
- **Win-C** = an automatic **PASS that changes algorithm STRUCTURE**. **None built yet**, but a tractable
  micro candidate is designed + in-flight (`WIN-C-DESIGN.md`, §2(c) gap 8): deferred-accumulate vs
  per-iteration reduction (a real loop-carried-dependency change); e2e arm architecturally blocked.

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
| **q4_0 repack GEMV** (decode, `repack_gemv_q4_0_q8_0`) | **MEASURED** LMUL WIDE m1 vs NARROW mf2 **2.11–2.21×** · rvv VLEN128; VLEN-strip 1×16 vs 2×8 **1.48×** · K1 VLEN256 `[winA-e2e/FINDING.md; k1-vlen256/e2e-SEAL-and-caveat.md]` | LMUL decode **FLAT** (≈1.05× t1 / ≈0.93× t16, memory-BW-bound) **[NULL]**; VLEN-strip **1.31×** (K1, tg32 2.12 vs 1.62 t/s, both ENGAGED, SEAL) **[MEASURED]** | **MEASURED 1.22×** · vs ggml's OWN `ggml_vec_dot_q4_0_q8_0` (10 real RVV ops) · rvv · norm=0 byte-exact `[winB-correct-baseline/FINDING.md]` | **MEASURED ~2.6×** (t16 tg64 ~7.0 vs 2.71 t/s, memory-locality regime) · rvv. **K1 0.74× LOSS** (X60 autovec beats repack path) **[disclosed §4.2]** | NONE (§4.4) | NONE (§4.4) |
| **q4_0 repack GEMM** (prefill M=4, `repack_gemm_q4_0_q8_0`) | **MEASURED** LMUL WIDE vs NARROW **1.27–1.38×** (3 distinct `.so` md5s) · rvv `[winA-e2e/FINDING.md]` | prefill **1.70× t1** (anomalous, caveated) / **1.10× t16** (clean, pp256 19.9 vs 18.0, both ENGAGED) · rvv **[MEASURED]** | **MEASURED 6.36×** · same ggml RVV baseline · rvv · 32,740 vs 208,060 ns, norm=0 `[winB-correct-baseline/FINDING.md]` | **MEASURED 5.68×** (t16 pp256 17.9 vs 3.15 t/s, matmul-bound, micro≈e2e) · rvv | NONE (§4.4) | NONE (§4.4) |

### 1b. Win-A-only kernels (no algorithm-swap-vs-ggml → Win-B = N/A)

| Kernel | Win-A kernel | Win-A e2e | Win-B kernel | Win-B e2e | Win-C |
|---|---|---|---|---|---|
| **i16 dot-reduce contraction** (`TypedWideningDotReduce…`, the **Win-A headline**) | **MEASURED 2.27–3.79×** (rvv VLEN128) / **1.8–3.6×** (K1 VLEN256) · vs tune-OFF narrow-deferred (SAME algorithm, narrow LMUL, ALSO compiler-emitted) · byte-exact vs scalar oracle `[WIN-A-2-4x-EXPLAINED.md; commit 709bb69d]` | **N/A by nature** — not a standalone llama hot-path kernel; its e2e analogue *is* the q4_0 repack-LMUL row above. NOT a gap. | **N/A** — no ggml algorithm to swap against; baseline is tune-OFF same algorithm | N/A | NONE (§4.4) |
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
| **q4_1 repack GEMV** (`repack_gemv_q4_1_q8_1`) | **MEASURED ~1.80×** WIDE m1 (1×16) vs NARROW mf2 (2×8) · rvv VLEN128 · byte-exact (norm 0), 3-run 1.799–1.804× `[kernel-coverage/q4_1-winA-oracle-FINDING.md; commit 039fcd54]` | GAP — inherits the e2e BLOCK | code+lit COMPLETE; **MEASURED oracle PASS** norm ≤ 7.6e-6 vs scalar q4_1 dequant-matmul ref — correctness only `[same FINDING]` | **BLOCKED** — no q8_1x4 quantizer / q4_1-repack routing; spacemit `block_q4_1x16` is a different ABI (upstream, not us) | NONE (§4.4) |
| **q4_1 repack GEMM** (`repack_gemm_q4_1_q8_1`) | **GAP (gap 4b, QUICK)** — only the GEVM arm was measured; the GEMM WIDE/NARROW ablation reuses the identical harness | GAP — inherits BLOCK | code **DONE**; **MEASURED oracle PASS** norm ≤ 7.634e-6 vs scalar q4_1 dequant-matmul (nr=16×nc∈{16,32,64,336}) — correctness only `[commit 039fcd54]` | **BLOCKED** — same upstream gap | NONE (§4.4) |

### 1e. IME — N2 second family (int8→int32 vmadot); own row, not a clean A or B

| Kernel | "Win-A" | "Win-A" e2e | "Win-B" | "Win-B" e2e | Win-C |
|---|---|---|---|---|---|
| **IME matmul** (`tcrv_ime.mma`, `vmadot`) | **N/A** — no LMUL/strip/repack knob; the 4×4×8 MAC fragment is **VLEN-DERIVED** (`deriveIMEMatmulCapability`), an N1 capability hook, not an ablatable tune | (see e2e) | **N2-demonstration 5.51×** · vs a *competent RVV vector matmul* (`vwmacc`+`vredsum`) — NOT ggml-shipped, NOT scalar, NOT tune-on/off → neither a clean Win-A nor Win-B · K1 X60 256³ int8, both arms bit-exact `[n2-ime/IME-PERF-FINDING.md]` | **NULL 0.86–0.98×** (tg16 0.86× / pp32 0.98×, lib-swap A/B, IME confirmed ENGAGED) · vs non-IME RVV lib (itself Win-A-tuned 1×16) · K1, tinyllama-1B Q4_0 — real "no e2e win," matrix unit can't help memory-bound M=1 decode **[MEASURED-NULL]** | NONE (§4.4) |
| **N2 zero-core-branch emission** (the *proven* N2 claim) | — | — | not a perf cell: compiler-emitted vmadot runs **bit-exact 16/16** on real X60, **zero core family-name branch** (I3 verified), capability-FACT gated on `spacemit.ime` `[n2-ime/PLUGIN-SLICE.md, FOUNDATION.md]` | — | NONE |

### 1f. Emit-only kernel families (no togglable Win-A; not wired; grouped per inventory)

These carry **no `TunableScheduleOpInterface`** and **no `lookupRVVScheduleDescriptor` case** → Win-A is
genuinely **N/A (hard-pinned / emit-only)**. Win-B is per-op-vs-ggml but **micro-only / not measured**; e2e is
**GAP (not-wired, default-no per honesty discipline)**. Win-C `NONE` globally. Grouped to avoid ~30 noise rows.

| Family (ops) | Win-A | Win-B kernel | Win-B/all e2e | Win-C |
|---|---|---|---|---|
| **K-quant block-dots** q6_K/q4_K/q5_K/q2_K/q3_K (+aux32) | **N/A** emit-only — *but* q4_K Win-A knob = **emitter parametrization** (deferred, scoped): fixed-m2 emit IS why Win-B loses to ggml `_vl256` `[commit 79db10f0]` | **MEASURED, MIXED** vs ggml's OWN `_vl256` (K1 VLEN256, norm 5e-7..3e-6): q2_K **WIN 1.016×**, q5_K **TIE 0.998×**, q4_K **LOSS** (ggml 1.72×), q6_K **LOSS** (ggml 2.26×), q3_K **LOSS** (ggml 2.13×) — single-LMUL `_generic` fp-order port vs ggml's hand-tuned nibble-split `[commit d27c512a]` | **GAP** not-wired (no §7 e2e row) | NONE (§4.4) |
| **IQ-quant block-dots** iq4_xs/iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq1_s/iq1_m | **N/A** emit-only | vs per-op `ggml_vec_dot_iq*_q8_K` — **GAP** | **GAP** not-wired | NONE (§4.4) |
| **Ternary block-dots** tq2_0/tq1_0 | **N/A** emit-only | vs `ggml_vec_dot_tq{2,1}_0_q8_K` — **GAP** | **GAP** not-wired | NONE (§4.4) |
| **FP4-codebook block-dots** iq4_nl/mxfp4/nvfp4/q1_0 | **N/A** — `coreLmul` read in emitter but **no pass stamps it** (attr-present-not-selected) | vs per-op ggml (q1_0 not mainline → nominal) — **GAP** | **GAP** not-wired | NONE (§4.4) |
| **Forward-pass hard-pinned** silu / softmax / quantize_row_q8_0 / rope_norm (fixed f32m2) | **N/A** hard-pinned (no `strip_lmul` knob) | **N/A** bit-identical reimpl of the ggml fn (same algorithm) | **GAP** not-wired | NONE (§4.4) |
| **q4_0 GEMM-tile** (`q4_0_q8_0_gemm_tile`) | **N/A** — no own knob; composes the parent GEMM's M-knob | M× `ggml_vec_dot_q4_0_q8_0` (algorithm = shared weight decode) — building block, superseded e2e by repack | **GAP** (no standalone row) | NONE (§4.4) |

**Win-C column — single global fact:** `NONE` in every cell above because **no automatic
algorithm-STRUCTURE-changing pass exists** (§4.4). This is **one** missing capability (the strongest unclaimed
novelty), **not** per-kernel debt. The repack is hand-authored Win-B and correctly NOT relabeled Win-C.

---

## 2. Prioritized gap-filling plan

> **Consolidation status (2026-06-24) — folded into §1 above:**
> - ✅ **Gap 4 (GEVM) + Gap 2** CLOSED — q4_1 repack GEVM Win-A·micro **~1.80×** (byte-exact) + q4_1 GEMM
>   scalar-oracle **PASS** (norm ≤ 7.6e-6). `[q4_1-winA-oracle-FINDING.md; 039fcd54]`
> - ✅ **K-quant Win-B·micro** (all 5) vs ggml `_vl256` — honest MIXED (q2_K win / q5_K tie / q4_K·q6_K·q3_K
>   loss); the losses MOTIVATE the q4_K Win-A knob. `[kquant-winB-micro-FINDING.md; d27c512a]`
> - ⏳ **Gap 8 (Win-C)** designed + make-or-break IN FLIGHT — `WIN-C-DESIGN.md` (1adc631b); see (c) below.
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
7. **Gap 8 — Win-C (automatic structural-transform pass).** The single biggest conceptual gap and the strongest
   unclaimed novelty, but a **build-a-new-pass effort, not a measurement** (§4.4). Related N1→N3 target
   ("winning-ALGORITHM-differs-by-profile", §5 item 7) is OPEN.

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

**Incremental leverage (the wiring cost was paid ONCE):** the 2nd IME op (tiled matmul, commit `0b8c6168`) and
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
