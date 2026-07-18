# block2-p2-iq2-vecdot — owned iq2 GRID-codebook vec_dot @rvv attack (公式墙必攻)

**Task**: `.trellis/tasks/07-19-block2-p2-iq2-vecdot` — attack the P2 公式墙 (objdump-recon verdict):
build the owned iq2 grid vec_dot leaf, prove byte-exact, board-test cold vs the deployed
hand-tuned `_vl128`. clang-18 compiler-symmetric ([CASE-COMPILER-ASYMMETRY] 守).

## Files
- `kernels/iq2_{xxs,xs}.kernel.c` — owned weft-emitted iq2 grid vec_dot leaves (vluxei16
  grid-decode over the 256/512-entry iq2 codebook + tiny vwredsum reduction). Exported
  byte-identical to the sealed CORE emit; GEN_SEAL.txt md5+recipe.
- `iq2_vecdot_driver.c` — 3-way byte-exact + 3-arm anti-hollow + cold-timing driver.
  OURS = the 2 leaves (linked .o); OPP = deployed `ggml_vec_dot_iq2_{xxs,xs}_q8_K` -> `_vl128`;
  ORACLE = pure-integer INDEPENDENT ZERO-MODEL scalar recompute (per-bit ksigns sign fold,
  NOT the emitter's expanded signs64 plane). INT-mode (weight d=fp16 1.0, q8=±1) => exact
  integer fold, order-independent byte-exact + valid cold. [NG-4] kernel-axis M=1, NOT e2e.
- `iq2_oracle_tables.h` — canonical ggml iq2xxs_grid[256]/iq2xs_grid[512]/ksigns_iq2xs[128].
- `rvv_run.sh` — board build (clang-18 `--gcc-toolchain`, symmetric with clang-18 opp lib) +
  gate + 2-seed cold. Board /tmp only; repo side零写盘 except this data格 (ISSUE-090 契约).
- `board_seal_and_run.txt` — board build_seal + run.log (VLEN128, ggml_cpu_md5, verdicts).

## Verdict (2-seed cold, clang-18 symmetric, VLEN128, K=2048 M=1 nc=512, 224MiB flush)
- **iq2_xxs**: byte-exact GREEN (3-way ALL=true, worst_ulp=0, anti-hollow 3/3 bite);
  cold ratio_cold_X 5-seed = 0.837/0.847/0.862/0.841/0.828 ALL ≥0.8 (min 0.828, IQR 1-2%) => **PASS (地盘)**.
  [headline flip 具名-X(0.697 proxy)->PASS; needs main-session independent +/- check before入账.]
- **iq2_xs**: byte-exact GREEN (3-way ALL=true, worst_ulp=0, anti-hollow 3/3 bite);
  cold ratio_cold_X 4-seed = 0.627/0.617/0.610/0.657 ALL <0.8 (~0.62) => **具名-X (边界)**. Named floor = the
  **16× serial vwredsum** per super-block (objdump self-probe OURS vwredsum=16 vs xxs=8),
  forced by iq2_xs's dual per-half scale (ls1/ls2 16-lane collapse). Registered un-pulled
  lever = reduction-batching (ISSUE-020 sum2 sign-sum vectorization / ISSUE-109 vwredsum
  floor family). NOT an architecture wall (readable, keyable, registered lever).

## Opponent caliber (成色)
Opp = ggml hand-written intrinsic `_vl128` (tier=手调, NOT the cheap `_generic`). iq2_xxs
PASS = ours within 0.84× of a hand-tuned strong opponent on its OWN readable structure
(vluxei16 grid gather双侧对称 + vwredsum), compiler-symmetric — 公式墙对位 attack, not a
temperament-wall bypass (no inline-asm, no schedule-pinning). M=1 kernel-axis; NOT e2e.
