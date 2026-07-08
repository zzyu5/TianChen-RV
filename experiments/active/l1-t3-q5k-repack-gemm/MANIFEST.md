# cell MANIFEST — l1-t3-q5k-repack-gemm

- **campaign**: repack / [KQUANT-L1] (T3-tile fmt4 — q5_K repack GEMM tiling-lever transfer test on
  rvv/VLEN128, ★[XFER-1] verify #2: does the q4_K/q2_K S6 register-cliff lever SURVIVE the q5_K qh
  5th-bit weight plane? q5_K SHARES the q4_K min fold `kquant_dmin_bsums_min` (dual d/dmin + bsums-min,
  8 sub-blocks) AND ALSO carries a qh weight plane (like q6_K's second plane, but riding the SHARED
  `weight_qh_byte_offset` slot on the MIN fold). ★register-cliff prediction: does q5_K reach ≤32 (v30,
  min-fold dominates, HOLDS) or stay v31 (qh weight-bound, NULL)? Sibling of the min-fold HOLDS cells
  `l1-tile-s6-q4k` / `l1-t3-q2k` and the weight-bound NULL cells `l1-t3-q6k` / `l1-t3-q3k`. This
  COMPLETES the K-quant tiling-transfer map (last of 5 super-blocks). Logged ALONE.)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz, gov=performance)**.
  ★ RESULT = **HOLDS (hybrid) — the min-fold register-cliff lever SURVIVES the qh plane.** (1) objdump
  at the measured clang-17.0.6 -O2 board form (campaign-canonical fixture
  `rvv-to-emitc-repack-gemm-q5-K-q8-K.mlir` + vs*r.v seal, NOT driver-export): maxVreg **v31 → v30**
  (the ≤32-vreg register cliff was REACHED — the SAME v31→v30 step q4_K S6 / q2_K T3 hit; q6_K/q3_K
  NULL stayed v31), reload **288 → 110** (−62%), vwmacc **2240 UNCHANGED** (byte-exact multiset) at
  BOTH -O2/-O3. **NUANCE: spill 155 → 105 (−32%), did NOT collapse to ~0** (contrast q2_K 619→7) —
  q5_K's qh 5th-bit weight plane leaves a residual weight-materialization floor the decode-scale/min
  panel does not stage (untiled q5_K's spill was already only 155, not the 600–900 of q2_K/q6_K,
  because untiled q5_K was already lean + already a 1.55× WIN). (2) byte-exact GREEN both ways: silicon
  oracle verify (independent scalar q5_K dequant-matmul, 8 shapes) UNTILED & TILED both WORST_NORM
  8.0090e-07 PASS (< 1e-4); control-flips QH 2.70M× / NOMIN 149291× / PERM 2.59M× / ROWROT 6.15M×
  (ref sensitive to the qh axis + min + interleave); IDENTITY-DUMP o_untiled.bin == o_tiled.bin (nr64
  32768 / nr16 8192 floats, 0 mismatch). (3) A/B tiled/untiled paired cold (nr=64 N=12 / nr=16 N=10):
  **1.417× / 1.350× — a +41.7%/+35.0% WIN** (ours cv 0.36–0.85%, per-round A/B min 1.336 > 1 →
  non-overlapping, SHAPE-ROBUST). (4) vs-opponent block-dot (clean anchor nr=64, opp cv 0.35% across 24
  samples): UNTILED parity **1.547×** (reproduces the ~1.5× vs-UNTUNED baseline) → TILED **2.193× WIN**
  (our tiled q5_K repack GEMM 2.19× FASTER than ggml_vec_dot_q5_K_q8_K block-dot); nr16 1.333× → 1.795×.
  The pre-registered gate (register cliff v30 ∧ MAC/cycle up ∧ wall T-N) PASSES ⇒ q5_K S6-scheme HOLDS.
  **[NG-4] NOT a beat**: eight [PERF-1] gates unwalked; single core; opponent = single-thread block-dot
  proxy; our correctness = construction oracle + the byte-exact oracle verify + identity gates here.
- **role**: L1-maturity T3-tile A/B evidence. Isolates the q5_K tiling increment by measuring UNTILED
  (pre-S6 front-door export `k_gemm_q5K.cpp`, 24 panel markers / 4480 src vwmacc, oracle-GREEN
  @construction) vs TILED (working-tree S6-scheme `emitRepackKQuantGemmBodyQ5K` = q4_K S6-tiled body +
  qh 5th-bit inject, 440 panel markers / 4480 src vwmacc, regenerated via read-only `build/bin/tcrv-opt`
  on the campaign-canonical front-door region fixture) as separate board binaries sharing the same
  extern-C symbol, against the opponent's REAL linked block-dot, with a silicon oracle verify + a full
  IDENTITY cmp proving the timed TILED is byte-exact to UNTILED/oracle. Harness/driver live under
  `tools/e2e-harness/board/` (`kquant_gemm_tile_q5k_t3_ab.sh` + `kquant_gemm_paired_q5k_driver.c` +
  `oracle_q5K.cpp` construction reference). Data/evidence only; the exported .c are NOT stored
  (deterministically regenerable — see the objdump seal + `experiments/active/kquant-l1-q4k-q5k-repack-
  prefill/EXPORT_RECIPE.md`; the TILED tile is the working-tree emitter delta, uncommitted). Main tree
  + build/ UNTOUCHED (no rebuild); nothing committed.

## durable files

- `tile_q5k_t3_findings.md`          — writeup: byte-exact gates (oracle verify + identity + controls),
  spill 155→105 (cliff REACHED v30, reload −62%; qh residual floor), A/B 1.417×/1.350× (nr64+nr16),
  vs-opponent 1.55×→2.19× / 1.33×→1.80×, HYBRID-HOLDS verdict, the min-fold-survives-qh transfer-boundary
  diagnosis, loadavg honesty, [NG-4] discipline.
- `tile_q5k_t3_ab_nr64.txt`          — raw board run, cold N=12, nr=64 (clean opponent anchor): preflight
  + objdump seal + hot-loop spill classification + oracle verify (both, +controls) + identity cmp + 12 A/B rounds + loadavg.
- `tile_q5k_t3_ab_nr16.txt`          — raw board A/B, cold N=10, nr=16 (shape-robust ours-side 1.35×; opponent still usable cv 1.30%).
- `tile_q5k_t3_ab.csv`               — parsed summary (nr, variant, ours/opp GMAC/s, cv, parity, A/B, vsetvli, spill, reload, maxVreg, vwmacc, textB).
- `objdump_tile_q5k_t3_seal.objdump` — UNTILED→TILED vsetvli/spill/reload/maxVreg/vwmacc/text seal at clang-17 -O2 AND -O3 + hot-loop spill classification + the min-fold-survives-qh (vs q6_K/q3_K) transfer-boundary mechanism.

## disposition (post-measurement ruling) — q5_K SHIPS TILED (S6)
Per **裁决一 [更简单者胜]** applied to a HOLDS/WIN: the S6 tile is BYTE-EXACT to the untiled body
(oracle PASS both + identity 0 mismatch + vwmacc 2240 unchanged) AND 1.35–1.42× faster AND pushes
vs-opponent 1.55×→2.19×, so the tiled emitter is the shippable form — no revert (unlike q6_K/q3_K, which
shipped construction-only). The working-tree `emitRepackKQuantGemmBodyQ5K` S6 delta is the correct end
state. The GEVM sibling is plain (never tiled; direct-connect). Construction is INTACT (the parent's
front-door q5_K: `lowerToRepackGem{v,m}KQuant` q5_K min-fold+qh branch + kQ5KDecodeFacts + schema
six-state `constructed` C_construct 39→40 + whitelist 11→10). Parent will commit (this cell
git-add/commit-free).

## note — data-only cell
Data/evidence only. The load-bearing result is a **HYBRID HOLDS (q5_K S6-scheme's min-fold lever
transfers the q4_K register-cliff step and SURVIVES the qh 5th-bit weight plane)**: peak vreg reaches
the ≤32 cliff (v31→v30), reload −62%, throughput +41.7%/+35.0%, vs-opponent 1.55×→2.19× / 1.33×→1.80× —
all **byte-exactly** (oracle PASS both + identity cmp 0 mismatch; vwmacc 2240 unchanged) — with the ONE
nuance that spill floors at 105 (−32%, not →~0) because the qh weight plane is a secondary residual the
decode/min panel does not stage. [NG-4] — L1 kernel datapoint, NOT a beat. Nothing here was committed /
git add-ed; main tree + build/ untouched (no rebuild — UNTILED is the cached pre-S6 export; TILED is
the working-tree emitter export via read-only `tcrv-opt`). Logged as the q5_K T3 HYBRID-HOLDS datapoint
— it COMPLETES the K-quant tiling-transfer map (5/5 super-blocks): the min-fold family HOLDS
(q4_K/q2_K full-collapse, q5_K min-fold-dominant-with-qh-residual) vs the weight-bound NULL (q6_K/q3_K),
ALONE.
