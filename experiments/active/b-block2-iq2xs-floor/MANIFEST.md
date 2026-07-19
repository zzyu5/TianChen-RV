# b-block2-iq2xs-floor — iq2_xs vec_dot@rvv floor 攻坚 (未试挂号杠杆「归约批处理」)

**Task**: `.trellis/tasks/07-19-block2-iq2xs-floor` — pull the registered reduction-batching lever
on the iq2_xs vec_dot@rvv **floor** (P2 具名-X 0.62, named floor = 16× serial vwredsum,
ISSUE-109 / ISSUE-020 / ISSUE-112). Same fusion pattern as the tq1_0 P1 leaf (8× serial
vwredsum → 1, commit 4e8d46948). clang-18 compiler-symmetric ([CASE-COMPILER-ASYMMETRY] 守),
board rvv VLEN128. Standalone mechanism proof; NO emitter change; NO git commit.

## objdump-diff (★mandated pre-attack — see `raw/rvv_fused_cold_compare.txt`)
| leaf/opp | vwredsum | vredsum | vluxei16 | vwmacc | vsetvli |
|---|---|---|---|---|---|
| OURS deployed emit (P2 floor) | 16 | 0 | 8 | 0 | 61 |
| **OPP `_vl128` (hand-tuned)** | **16** | 0 | **8** | **0** | 49 |
| OURS FUSED_A (VLEN-universal, 4-gather) | 0 | 1 | 16 | 8 | 35 |
| OURS FUSED_C (VLEN-universal, 8-gather+8 vslidedown) | 0 | 1 | 8 | 8 | 36 |
| OURS FUSED_B (VLEN128-form, 8-gather) | 0 | 1 | 8 | 8 | 32 |

**Answer**: the opponent does **NOT** batch the reductions — its 16× serial vwredsum is
*identical* in structure to our deployed emit. So the emit-vs-opp gap (0.62, same 16-vwredsum
structure, opp just has cleaner clang codegen: 49 vs 61 vsetvli) is a 脾气墙. The reduction-
batching lever is a **readable, keyable structural change the opponent itself never made** →
**公式墙**: fusing it lets us *out-structure* the opponent and cross to near-parity.

## owned FUSED leaves (PURE C-INTRINSIC · NO inline-asm · byte-exact)
Both fold the per-half integer scale (ls1/ls2) INTO each product (integer distributivity:
`bsum = Σ_h ls[h]·Σ q8·sw = Σ_all (q8·sw)·ls[halfgroup]`) → widening MAC into ONE persistent
i32 accumulator → ONE vredsum per super-block. Grid gather (vluxei16 over the 512-entry iq2xs
codebook) + signs64 fold reused verbatim from the sealed emit.

| leaf | gather | VLEN | byte-exact | cold ratio_cold_X (rvv) | verdict |
|---|---|---|---|---|---|
| `kernels/iq2_xs_fused.c`   (A) | i64m2 4-entry (**16 vluxei**) | universal | GREEN (5 seeds, ULP=0, anti-hollow 3/3) | med 0.49 (5 seeds) | **REGRESSION** — gather DOUBLED |
| `kernels/iq2_xs_fused_c.c` (C) | i64m4 8-entry (**8 vluxei**) + 8 vslidedown | **universal** | GREEN (2 seeds + 8 cold-verified, anti-hollow 3/3) | med 0.75 (8 seeds, 1/8 ≥0.8) | **具名-X (boundary)** — vslidedown overhead |
| `kernels/iq2_xs_fused_b.c` (B) | i64m4 8-entry (**8 vluxei**)  | VLEN128-form | GREEN (2 seeds + 15 cold-verified, anti-hollow 3/3) | **med 0.82 (15 seeds, 12/15 ≥0.8)** | **公式墙 PASS candidate (near-parity)** |

## byte-exact gate (`../block2-p2-iq2-vecdot/iq2_vecdot_driver.c` — shared)
ZERO-MODEL 3-way (ours == deployed-ggml == independent integer oracle) + 3-arm anti-hollow
(inject 1=ours 2=oracle 3=leaf-input, each MUST break agreement) + fp16 golden. INT-mode
bounded fill (d=1.0, q8=±1) ⇒ exact integer bsum ⇒ order-free byte-exact + valid cold.
`0.125f*sumf` fold statement order matches emit/oracle exactly. worst_ulp=0.

## board (compiler-symmetric clang-18 · VLEN128 · 224MiB flush · reps=15)
`board_run.sh` = build 3 leaves + objdump-diff + gate + 5-seed cold (稳则5). Board /tmp only;
repo side零写盘 except this data格. opp = deployed `_vl128` (tier=手调). Same-session baseline:
deployed emit med **0.62**; FUSED_B med **0.82** (closes 52% of the gap to parity).

## verdict (成色 · 诚实)
- **公式墙 ATTACK SUCCESS (mechanism proven)**: reduction-batching (16 serial vwredsum → 8
  vwmacc + 1 vredsum) is byte-exact and moves the named floor **0.62 → 0.82** (median, 12/15
  seeds ≥0.8) — a **PASS candidate at the 0.8 gate (boundary, near-parity)**. The P2 verdict
  ("16× vwredsum = the floor") is **precisified**: the floor IS crackable; the lever works.
- **成色**: cold ≤ 1.0 ⇒ near-parity **PASS-by-gate, NOT beat** (opp = 手调非便宜档; ~1.2×
  slower still). M=1 kernel-axis; [NG-4] NOT e2e, NOT a sealed Win.
- **VLEN tension (honest, corrected)**: FUSED_B is **VLEN128-form** (vget_i8m4_i8m2 32-lane
  split) — on-position vs the opponent's own `_vl128` specialization (the ggml lib ships
  _vl128/_vl256/_vl512 SEPARATELY; per-VLEN specialization IS a capability key). A **VLEN-
  universal** single kernel is realizable two ways, but neither crosses the 0.8 gate:
  FUSED_A (4-entry i64m2 gather to fit the i32-LMUL8 32-lane accumulator) regresses to **0.49**
  (gather DOUBLED); FUSED_C (8-entry gather + `vslidedown`-by-32 for the upper sub-block, no
  doubling) reaches **0.75** but the 8 vslidedown/super-block cost ~7% → below gate = 具名-X.
  So the PASS-crossing 0.82 is intrinsically VLEN128-form (matches the opponent's own strategy);
  a single VLEN-universal kernel pays a real cost that lands it at 0.75. (Earlier "structurally
  blocked" hypothesis was too strong — variant C proves universal is realizable, just slower.)
- **DEPLOYMENT (scope registered, NOT done)**: FUSED_B is a NEW structure, not the current
  `emitIQ2XSSuperBlockGridBody`. Realizing 0.82 as **地盘** needs deploying the fused structure
  into the emitter (like tq1_0 P1→deploy). This task proves the mechanism only.

## 头条 flip 纪律
具名-X(0.62) → PASS-candidate(0.82) = headline flip · needs main-session independent +/- check
(build+lit+board positive/negative re-verify) before 入账. This agent produces data+evidence ·
does NOT commit. Sealed core `lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp` UNTOUCHED.

## 谓词
`bash board_run.sh` → objdump-diff (opp vwredsum=16 vwmacc=0) + FUSED_B GREEN + med cold ≥0.8.

## 出处
task `07-19-block2-iq2xs-floor` (2026-07-19). Refs: tq1_0 P1 `../b-block2-tq10-vecdot/`
(commit 4e8d46948) · iq2 P2 `../block2-p2-iq2-vecdot/` (floor, commit f55a4b1da) ·
ISSUE-112 / ISSUE-109 / ISSUE-020.
