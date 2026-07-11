# T3p — 模式逐条消融 (mechanism-claim → data-verdict)

> Casefile for **覆盖式铺面⑤ (测量总攻队列末项)**: systematize the scattered
> per-pattern ablation verdicts — each optimization pattern / selection pass in the
> [PAT-2] registry carries a **mechanism claim** (why it should help); T3p is the
> **paired ablation** (pass ON/OFF or structural decomposition) that lets each claim
> **earn a data verdict** ∈ {validated / refuted / structural-NULL /
> measured-negative-physical / real}. **Most verdicts already exist scattered across
> T8 / pattern-registry / casefiles — this project CODIFIES them into one T3p table**
> (+ names the one true gap). **In-tree analysis · [NG-4] mechanism C3′ evidence ·
> NOT perf-covered headline · honest** (structural-NULL / measured-negative are
> LEGITIMATE verdicts, never dressed up as wins). Behavior is only OBSERVED / codified,
> never changed — **zero `lib/` edit, zero pass/selection-logic change**.

## HEAD / provenance
- Task-pinned HEAD = `13706e79` (all rows snapshot-stamped there).
- Working-tree HEAD advanced to `a08d0689` mid-turn = **main session's concurrent commit of
  ④T4b** (the sibling item; ⑤ auto-chained after it per ROADMAP:118). This agent ran **no git**;
  files here are untracked working-tree additions for the main session to commit.

## 五模式 + winc 结构-NULL 典范 (逐条 · 机理声明 → 判读)

### ★ 命名警示 (honest, load-bearing) — "P1"/"P4" 的 canon 标签碰撞
Canon itself carries **inconsistent P-numbering**; T3p records the collisions rather than
silently picking one:
- **P1** has THREE distinct referents: (a) 科研目标v2:121 **P1 = 宽 LMUL 分组**; (b) 执行总纲v2:91
  **"P1 已机制化 (4 构造 front door)"** = the N-operand construction paths (**this task's P1**);
  (c) T8 **[GAP-P1]** = the VLEN-widen "宽 VLEN 喂饱" schedule pattern (a `q4_0-gevm-k1-vlen256`
  roofline gap). This casefile adjudicates **(b)** as the task directs, and cross-refs (a)/(c).
- **P4** = 布局/repack. 执行总纲v2:91 marks the formal **PAT-2 "P4" registry object as 未启动**,
  yet the repack MECHANISM is heavily realized under a SEPARATE registry entry (`PAT-S6`) + 11
  repack families. T3p adjudicates the **mechanism** (real, via PAT-S6) and flags the empty formal slot.

---

### P1 — N-operand route unification  → **validated-structural REAL ∧ perf-novelty RETIRED**
- **Mechanism claim**: a unified, algorithm-uncommitted N-operand contraction route (the 4 construction
  front doors) is a first-class **construction** primitive; the compiler "autonomously selects the
  measured-best algorithm" from capability facts.
- **Verdict (codify)**:
  - **Construction axis = VALIDATED-REAL**: `[B-1]` **CONFIRM** byte-exact over 3 product-reduction
    shapes (product-reduce / offset-binary N=3 / codebook-LUT N=3); `T1_C1_structural_conjunction`
    rows `C1-conj-*` = TRUE; the select-branches emit **byte-IDENTICAL** (audit attrs emitter-inert)
    ⇒ **zero perf delta by construction** — it is a *construction* enabler, not a *perf* pattern.
  - **Perf-novelty axis = RETIRED** ([[option2-path-selection-real-pass]] + `backend-maturity-triton-reframe`):
    "compiler-selects-algorithm = novelty" was **demoted to a frontend/autotuner concern** (reaching into
    ggml load layout is out-of-bounds; repack materialization is byte-redundant with ggml, `memcmp==0`).
    The **surviving** mechanism = capability-driven **DECLINE** (declining repack where it loses = matching
    what ggml already gates off = **loss-avoidance HYGIENE, NOT a 2nd novelty**).
- **Evidence ptr**: `docs/canon/TianChen-RV_执行总纲v2.md:15 (B-1 CONFIRM) / :17 (B-3 双发射) / :91 (P1 已机制化)`;
  `lib/Plugin/RVV/RVV{DequantDot,PackedI4Dot,CodebookDot,MonolithicBlockDot}SourceFrontDoor.cpp`;
  `experiments/active/result-tables/T1_C1_structural_conjunction.csv`; memory `option2-path-selection-real-pass`.
- **Ablation type**: structural-in-tree (no board — byte-exact identity, nothing to time).

### P2 — deferred-ordered fold (P2c)  → **measured-negative-physical** (codify)
- **Mechanism claim** (at landing, commit `1185729a`): PHASE-B replacing the per-block
  vector→scalar→fmaf round-trip with ONE seed-first `vfredosum.vs` ordered reduction recovers the
  +8.4% no-FMA fold cost and unlocks Win-B (**round-trip-elimination hypothesis**).
- **Verdict (codify)**: **FALSIFIED, physical.** Fair VLEN128 retest (clang-20 both sides, zfh/zvfhmin
  full-capability march, `-ffp-contract=off`, objdump 0 libcall, IQR<0.02%): deferred is **23% SLOWER
  than factory (A/C=1.232)** and **7.5% slower than per-block (A/B=1.075)**. Root cause = **vsetvli churn
  (deferred 19 vs per-block 5) > the round-trip it saves**. The apparent original **1.69× "win"
  (A/C=0.592) was 100% the fp16-libcall confound** on the factory opponent, zeroed after fair build.
  Capability + numerics are intact (**dual-board 256/256 bit-exact ULP=0** vs pinned §1 oracle) ⇒ this is
  a **layer-4 design-space asset (mechanized construction, measured-false optimization), NOT a failed brick**.
- **Evidence ptr**: T8 first-loss row `q8_0-deferred-ordered-fold-P2c` (triage=`physical`);
  `schema/pattern-registry.v1.json` `MFLAT-P2c-deferred-ordered-fold` (status=`measured-negative`);
  `experiments/archive/perf-historical/ondevice-q8_0-deferred/fair/perf_rvv_vlen128_FAIR.csv`;
  `.trellis/tasks/07-04-m-flat-nb-loop-layer/evidence/gap-log.md` GAP-1/P2c-DEFERRED-NULL.

### P2b — wide-clamp / dual-region (widen-clamp-narrow)  → **open / gap-absent (needs-build)** (honest not-done)
- **Mechanism claim**: a two-region **widen-clamp-narrow** body (SEW8/SEW32 two-region) would turn the
  clamp regression into a win (the FIRST "gap-close" pattern [K-2b]/[K-6]).
- **Verdict (honest)**: **NOT adjudicable — the winning pattern is NOT mechanized.** Only the
  **single-scope LOSING** form is built (`RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp:1088-1154`),
  which loses **0.5× vs clang-autovec** (对称-clang, `T-VALIDITY` row 39). That loss is diagnosed as
  **`missing_pattern` (缺 wide-clamp 双区能力), NOT blind misselection** (memory
  `perf-finale-metric3-selector-capability` corrected the blind-selector framing). The
  **winning dual-region body 未建**; the [K-6/GAP-1] close-loop steps 2–4 (name→close→retest→cite-gap-ID)
  are **not walked**. **No board ablation is possible today — there is nothing to toggle.** Recorded as a
  bounded work-item, honestly parked (per 硬约束 5: don't fabricate a verdict).
- **Evidence ptr**: `docs/canon/TianChen-RV_执行总纲v2.md:76 (K-2b) / :81 (K-6)`;
  `docs/canon/TianChen-RV_科研目标总纲v2.md:78 (K-2b) / :168 (C3-3③)`;
  `experiments/active/result-tables/T-VALIDITY_compiler_symmetry_ledger.md:99` (clamp 0.5× row).

### P4 — layout / repack (+ PAT-S6 output-tiling)  → **real, with honest 3-class boundary**
- **Mechanism claim**: physically repacking weights (plain→x16) + register-cliff output-tiling (S6) yields a
  memory-locality **path win** the vector opponent lacks.
- **Verdict (codify)**: **REAL path/opening exists, but with a sharply-bounded, hardware-measured transfer rule.**
  - **q4_0 e2e 5.9×** is **REAL but ROUTING white-steal, NOT kernel quality** (upstream `f3e1828` prebuilt the
    entire q4_0 repack pipeline; ours = a 1-line `case 128` route flip; kernel A/B byte-identical ⇒ win = 100%
    dispatch). **禁以 5.9× 外推 K-quant** ([[q4-0-e2e-is-routing-not-kernel]]).
  - **PAT-S6 output-tiling = MECHANIZED**, [XFER-1] **7/7 predicted==measured** 3-class boundary:
    **class-1 min-fold HOLDS ×3** (q4_K/q2_K/q5_K, spill→0/3/7, maxVreg **v30 crossed ≤32 cliff**);
    **class-2 weight-recon-bound NULL ×2** (q6_K spill **913→949 ROSE**, q3_K, **v31 NO cliff** — structural-NULL);
    **class-3 codebook already-lean NO-OP ×2** (iq4_nl 7 / iq4_xs 73, already ≤32).
  - **★vs-opponent perf WITHDRAWN** (CASE-COMPILER-ASYMMETRY): the kernel-axis vs-opponent ratios
    (q4_K 1.884× etc) were **clang-ours-vs-gcc-shipped artifacts**; symmetric-gcc remeasure = **LOSS** (e2e 0.334×).
    Surviving mechanism = the **structural register-cliff crossing** (spill→0, objdump-verified,
    compiler-internal-consistent) + **opponent-absence path**, NOT the withdrawn ratio. Kernel-axis only;
    e2e = **projection** (prefill Amdahl upper-bound ~1.59× / decode NULL).
- **Evidence ptr**: `schema/pattern-registry.v1.json` `PAT-S6-repack-gemm-output-tiling-register-cliff-XFER-1`
  (status=`mechanized`, 3 xfer_classes); T8 XFER-1 7 rows (`q4_K/q2_K/q5_K-tile-S6` HOLDS,
  `q6_K/q3_K-tile-T3` NULL, `iq4_nl/iq4_xs-frontdoor-tile` NO-OP);
  `experiments/active/visibility/T7-three-curve-G3-closure.md`; `docs/reports/2026-07-10-C3-pattern-library-evidence.md`;
  `perf-constitution-three-layers` (L1 矿脉 · CASE-COMPILER-ASYMMETRY).

### P7 — `ime ∧ shape` matrix-paradigm takeover  → **validated-kernel-micro CONTINGENT ∧ micro↛e2e ∧ SEL-2 hazard**
- **Mechanism claim**: keying the matrix paradigm on (`ime.present` ∧ matmul-shape) lets the IME range take
  over prefill GEMM for a compute-account ~2× over the vector range.
- **Verdict (codify from G4-M3 / T5b / T5c)**:
  - **Compute-isolated kernel micro (k1, compiler-symmetric gcc 13.2.0)**: matrix range **WINS ~1.9–2.1×
    (M≥4) / 1.33× (M=1 GEVM)** vs the vector paradigm's **fair best** (Cell3b), **CONTINGENT on
    `[GAP-IME-LEAF-PIPELINE]`** — the as-emitted leaf (Cell1) **LOSES ~4×** (pure per-fragment vsetvli/zero/store
    overhead); one maturity lever (register-resident batched, Cell1b) flips loss→win. **Airtight control**
    (Cell3c gives the vector path IME's exact int8 byte-traffic) comes out **SLOWER** ⇒ the advantage is
    **genuine matrix-MAC throughput** (vmadot 128 MAC/instr vs vwmacc 32), **not an int8-vs-int16 traffic artifact**.
  - **micro↛e2e (critical caveat)**: the win is **compute-isolated** (weights pre-decoded, L2-resident); the
    memory-bound decode/GEVM (M=1) regime is **parity/roofline** and does **NOT transplant**
    ([[kernel-wins-dont-transplant-to-e2e]]: IME 5.51× kernel → 0.86× decode e2e). Decode verdict = parity/open.
  - **SEL-2 selector-timing hazard**: P7 activation makes the matrix variant **silently LOSE without error**
    (ascending cost sort `RVV 1.0 < IME 20.0` ⇒ vector chosen) unless the capability-prior layer keys BEFORE/with P7
    — this is the ④T4b silent-misselection story (sibling casefile).
- **Verdict framing**: **yellow cell** with a named fixable GAP — "matrix-range advantage board-demonstrated,
  blocked on emitter maturity" — **NOT a green cell, NOT an e2e beat** ([NG-4]; opponent = SELF, not vendor).
- **Evidence ptr**: `experiments/active/g4-m3-ime-paradigm-t5b/{MANIFEST.md,T5b_ime_paradigm_lever.csv}`;
  `experiments/active/t4b-selector-ablation/MANIFEST.md` (SEL-2 four-config);
  `docs/reports/2026-07-10-IME-gating-report.md` / `2026-07-09-IME-gating-review.md`;
  memory `k1-ime-n2-hardware-candidate` / `kernel-wins-dont-transplant-to-e2e`.

### winc — reduction-structure deferred-vs-per-iter  → **structural-NULL (EXEMPLAR)** (codify)
> Task-flagged as the **同类 structural-NULL 判读典范**; a reduction-fold sibling of P2/P3, kept here as the
> canonical example of the honest structural-NULL verdict form.
- **Mechanism claim**: a `reduction_structure` pass (deferred_accumulate | per_iteration, commit `39eb9c48`)
  gives a reduction-latency advantage; it toggles **3.0–3.3× pass-ON/OFF** on the i16 dot-reduce body (rvv VLEN128).
- **Verdict (codify)**: **structural advantage NOT demonstrated (NULL).** Register-kept decomposition
  (commit `e93a3b43`): deferred(ON) vs **register-kept** per-iter = **≈1.00× (pure-structure NULL, objdump no
  `out[0]` spill in loop)**; the 3× is entirely **register-kept vs MEMORY** per-iter — the OFF arm's `out[0]`
  **memory round-trip** carries the whole 3×. So the win is an **emitter artifact** (the deferred structure
  happens to avoid a memory round-trip the per-iter emission incurs), **NOT** a reduction-structure-latency
  advantage; `vredsum.vs` is not on the binding critical path for this streaming dot. **Do NOT claim a
  structural Win-C**; the pass is KEPT as the **Win-A LMUL-sweep structural enabler**, not a standalone novelty.
- **Evidence ptr**: memory `winc-structural-null`;
  `.trellis/tasks/archive/2026-06/06-22-n1n2n3-complete-multiprofile/artifacts/WIN-C-DESIGN.md` (decomposition §)
  + `.../artifacts/winC-ablation/` (deferred/periter/periter_regkept bodies + run1/run2).

---

## T3p 表落地 (欠账表 T3p: 0 → 6 rows)
`T3p_pattern_ablation.csv` (this dir) — schema per `experiments/_templates/T3p_pattern_ablation.csv`
(`pattern_id, ablation_pair_def, applicable_board, micro_effect, micro_effect_CI, transduction,
mechanism_claim_supported, status, snapshot`). 6 rows: P1 / P2 / P2b / P4 / P7 / winc.
**Main session** may promote a copy into `experiments/active/result-tables/` and update the
ROADMAP 测量欠账表 T3p row (0→6) + `T8` cross-refs (main-session-owned; not touched here).

## 判读类型分布 (C3′ honest evidence ledger)
| verdict type | patterns | 一句话 |
|---|---|---|
| **validated-structural (REAL construction)** | P1 (construction axis) | 4 front doors byte-exact `[B-1]` CONFIRM; perf-novelty RETIRED |
| **real, honest-bounded** | P4 (PAT-S6) | 7/7 XFER-1; HOLDS ×3 / **NULL ×2** / no-op ×2; vs-opp ratio withdrawn |
| **validated-kernel-micro, contingent** | P7 | ~2× compute-account, **blocked on [GAP-IME-LEAF-PIPELINE]**; micro↛e2e; SEL-2 hazard |
| **measured-negative-physical** | P2 | deferred fold 23% slower; round-trip-elim FALSIFIED (physical) |
| **structural-NULL** | winc | pure-structure ≈1.00×; the 3× = memory round-trip artifact |
| **open / gap-absent (needs-build)** | P2b | winning dual-region body 未建; nothing to ablate |

## 需板未做 (honest) / not-done
- **P2b** — no board ablation: the winning widen-clamp-narrow dual-region body is **未 mechanized**;
  bounded work-item, not a T3p measurement gap. (Reopen when [K-2b] lands.)
- **P4 e2e transduction** — decode e2e is a **projection** (Amdahl), not measured; the kernel-axis vs-opponent
  ratios are **withdrawn** (compiler-asymmetry). Only structural cliff-cross + routing 5.9× survive as data.
- **P7 decode M=1 e2e** — parity/open; needs a with-decode (memory-bound) measurement before the selector
  trusts a matrix win at M=1. T5d vendor-path methodology control NOT run this session.

## 保行为铁门 (verification)
- **Zero `lib/` edit · zero pass/selection-logic change.** Touch set = this dir only
  (`MANIFEST.md` + `T3p_pattern_ablation.csv`). All verdicts CODIFY pre-existing board/structural evidence
  or the honest absence thereof; nothing was re-run, re-measured, or re-implemented.
- **HEAD not moved by this agent** (no git); rows snapshot-stamped `13706e79` per task pin.
