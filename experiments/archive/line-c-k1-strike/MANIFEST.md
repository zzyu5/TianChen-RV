# cell MANIFEST — line-c-k1-strike (Line C · K1 精准打击)

- **campaign**: 测量总攻 / **Line C · K1 精准打击** (用户 2026-07-12 裁四 · parallel with Line A[IQ/rvv] & Line B[IME/k1];
  板时与 Line B 按板批协调不抢 · harts 4-7 · 独占 `build-k1-linec/` [not needed this session — all read-only]).
- **status**: DELIVERED (1a opponent_map + 1b XFER-1 prediction + 2 q5_K [X-0] verdict). Board = `ssh k1`
  (SpacemiT X60 / VLEN256 / 8 harts / stock clang-18). **Board READ-ONLY this session** (objdump/nm/readelf/CMakeCache/
  source-read only; zero board mutation) ⇒ **restore trivially clean** (no baseline change to revert).
- **role**: k1-side opponent-identity map + q8_0 VLEN256 correctness double-exit + [XFER-1] three-lever pre-board
  prediction + q5_K e2e [X-0]/Amdahl pre-flight gate. **NO git · NO code edit · correctness-first · [NG-4] kernel-axis.**

## 三段结论 (verdict summary)
1. **1a opponent_map** — k1 stock = **clang-18** (CMakeCache proof) ⇒ kernel-axis **compiler-symmetric on k1**
   (UNLIKE rvv gcc-15; corrects t4a asymmetry assumption). No SpacemiT/IME vendor path in stock. Dispatch: **5 formats
   (q4_0/q4_K/q2_K/iq4_nl/q8_0) = riscv-16x1 tuned repack (硬对拼格)**; rest = block-dot fallback (q5_K vec_dot vectorized,
   q6_K vec_dot scalar). **★q8_0-k1 DOUBLE-EXIT = 健康 (vsetivli 16 == VLMAX@256, 硬对拼格) ∧ 破损@VLEN128
   ([GAP-Q8_0-VLEN128-KERNEL] 手写库 VLEN 不可移植 = 碎片化第二证据·C1 对手侧供弹).**
2. **1b XFER-1 prediction (T3p-X, PRE-BOARD)** — 19 formats × 3 levers (S6/col-outer/vl16). S6: 3 HOLDS / 2 NULL /
   14 NO-OP · col-outer: 0 HOLDS / 10 NULL / 9 NO-OP (col-outer predicted DEAD on k1, FU-1 1.002× null) · vl16:
   6 HOLDS / 2 NULL / 4 NO-OP / 7 LOSS (IQ-gather). Evidence-tiers marked (k1-measured / rvv-transferred / pure-pred).
3. **2 q5_K [X-0] → 声明例外** — literal Amdahl (proxy 1.92×) passes on-paper (~1.68×) but **deployed-domain
   same-input-path factor = sibling q4_K e2e 0.42× (LOSS)** ⇒ WIN upper bound < noise floor ⇒ 声明例外
   (weight-reconstruction-bound; micro↛e2e K-quant instance). **perf-covered unchanged 6/84** (no k1 new green).
   **decode 预着色 = 黄-物理墙 (roofline).** Reopen trigger = [WORK-ITEM-K1-KQUANT-E2E] (k1-clang q4_K e2e).

## 双账本 disclosure (口径 required)
k1 kernel-axis compiler = **clang-18**; system/deploy = **clang-18** ⇒ **CONVERGE on k1** (both clang-18; no
asymmetry artifact). This is the material difference from rvv (stock=gcc-15). Ratios reported are kernel-axis
prefill-GEMM (t4a reuse), now compiler-symmetric-VALID; e2e transduction is the 声明例外 axis.

## 硬纪律 compliance
- **NO git** (no commit/add/rm/mv) — all files untracked working-tree additions for main-session.
- **correctness-first** — q5_K e2e NOT attempted (deployed-domain gate not passed; correctness未过禁进 perf).
- **board reversible** — read-only probing only; no scratch created; board CLEAN (verified: no /tmp/line-c*).
- **MANIFEST one-file-per-backtick-bullet** (below). No large .inc produced (no emit this session).

## durable files (this cell)
- `opponent_map_k1.md` — 1a analysis: dispatch map + q8_0 double-exit + compiler-symmetry resolution + no-vendor-IME.
- `opponent_map_k1.csv` — 1a per-format opponent identity / as-shipped / vectorization / tier grid (24 formats).
- `T3p-X_xfer1_prediction_k1.md` — 1b analysis: count summary + evidence-tier ledger + falsification caveats.
- `T3p-X_xfer1_prediction_k1.csv` — 1b 19-format × 3-lever (S6/col-outer/vl16) pre-board prediction grid.
- `q5_K_X0_amdahl_verdict.md` — 2 gate: [X-0] 三问 + Amdahl (dual computation) + 声明例外 verdict + decode pre-color + work-item.
- `T8_double_exit_proposed.md` — proposed T8 rows (q8_0-k1 健康/破损 double-exit + opponent-tier + q5_K 声明例外) for main-session merge.
- `raw_probe_evidence.txt` — raw board-probe outputs (objdump/nm/readelf/CMakeCache/source; read-only, board clean).

## board harness (touch-set)
- `tools/e2e-harness/board/line-c-k1-strike/probe_opponent_map.sh` — reproducible READ-ONLY opponent_map + q8_0-health probe recipe.
