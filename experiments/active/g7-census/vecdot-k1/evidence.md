# G7 L1 vec_dot@k1 kernel-sym census — k1-lane (VLEN256, clang-18 symmetric)

> **Line**: G7 终编成令三 · 第一段全量 kernel 冷启动普查 · vec_dot (block-dot) k1-lane (跨板并行·rvv 忙 Batch1).
> **Board**: k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / clang-18.1.8 (Bianbu) symmetric domain.
> **Axis**: **kernel-sym micro (第二赛道)** · [NG-4] NOT e2e · NOT perf-covered 9/83 · NOT a sealed Win. 禁互推.
> **Scope**: K-quant vec_dot 5 (q2_K/q3_K/q4_K/q5_K/q6_K) + FLAT vec_dot 5 (q4_0/q4_1/q5_0/q5_1/q8_0) · 全 10 格尽测.
> **Casefile**: `experiments/active/g7-census/vecdot-k1/` — kernels/ (+GEN_SEAL.txt), vecdot_census_driver.c, k1_build.sh, k1_measure.sh, raw/{k1_run.log, k1_build_seal.txt, k1_opponent_classification.txt}.

---

## 0. Harness (net-new; only "construct" gap was the harness, not the kernels — all emit already in)

- **OURS** = weft-emitted **block-dot** vec_dot kernels, regenerated deterministically from the working-tree
  emitter on the canonical front-door fixtures:
  `weft-opt <fixture> --weft-rvv-materialize-<slug>-block-dot-source-front-door --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`
  (fixtures = `test/Target/RVV/<fmt>-*-block-dot-full-pipeline-export-e2e.mlir`). Kernel md5 sealed in `kernels/GEN_SEAL.txt`.
  Compiled clang-18 -O3 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause (== stock ggml-cpu march). Emit is board-agnostic;
  VLEN256 vsetvl adapts at runtime.
- **OPP** = ggml **as-shipped** dispatched vec_dot, linked directly from the board's stock
  `/data/k1build-stock/bin/libggml-cpu.so` (md5 871169a0…) — the exact per-column block-dot ggml runs. Genuine shipped
  kernel, zero transcription risk. **Machine-judged (objdump --disassemble per sym) — see §2. NOT the weak generic the
  prior predicted; all 10 opponents are hand-tuned native-RVV.**
- **Both sides consume IDENTICAL plain block bytes** (no repack) — a fair kernel-axis A/B, same memory footprint,
  same number of vec_dot calls. GEVM driver: output[M][nc], each cell = one vec_dot of length K. M=1 (GEVM) + M=8 便车.
- **Byte-exact gate (ZERO-MODEL vs independent stock-ggml oracle)**: INT-mode fill — super-block d = fp16 1.0,
  dmin/min = 0 (min term vanishes), small integer payloads → both sides fold an EXACT fp32 integer → require
  ours[k]==opp[k] bit-for-bit. Nonzero varied results (q6_K −10407, q3_K 5570, …) confirm the accumulation is
  exercised, not trivial 0==0. Control flow is data-independent, so this same INT-mode data doubles as valid steady-state
  timing.
- **COLD protocol**: 32 MiB flush (64× k1 L2 512KiB, NO L3) before EACH timed region; paired A/B; N=12 reps; median +
  relIQR; T-N noise-floor (nf_ns≈200). **HOT** = single-buffer warmup + best-of-8. Timing = CLOCK_MONOTONIC (k1 has NO
  cache-miss PMU). Footprint at K=2048/nc=512 ≈ 576 KiB weights > L2, plus the flush = guaranteed cold.
- **Load-gate**: idle-core picker over 0-3 (300 ms /proc/stat delta, ≥70 % idle required); pinned core (idle-pick,
  gov=performance unchanged). Post-run stray check = 0; k1 restored, governor untouched.

---

## 1. ★ Four-column table (cold_med ratio = ours GB/s ÷ opp GB/s; >1 ⇒ ours faster; parity band ±5%)

### 1.C K-quant vec_dot (ours-block-dot [KQuant aux32 int-core] vs ggml as-shipped)
| 格 | hot | cold (M=1) | cold (M=8) | 对手类 [机判·符号级] | 胜负 (cold) |
|---|:--:|:--:|:--:|---|---|
| **q2_K** | 0.678× | **0.685×** | 0.678× | **hand-tuned native-RVV inline** (376 ins / 243 rvv, csrr vlenb) | **LOSS 0.68×** (weight-recon floor vs heavy hand-tune) |
| **q3_K** | 0.538× | **0.538×** | 0.538× | **hand-tuned native-RVV inline** (239 / 116) | **LOSS 0.54×** |
| **q4_K** | 0.610× | **0.617×** | 0.606× | **runtime dispatcher → `_vl256` hand-tune** (main 9 ins, csrr→vl256) | **LOSS 0.62×** |
| **q5_K** | 1.081× | **1.085×** | 1.081× | native-RVV inline (214 / 97, **NO vl-specialization** — single VLEN-agnostic body) | **WIN 1.09×** ⚠ mechanism = opp immaturity (see §4) |
| **q6_K** | 0.547× | **0.554×** | 0.553× | **runtime dispatcher → `_vl256` hand-tune** (main 28 ins, csrr→vl256) | **LOSS 0.55×** |

### 1.B FLAT vec_dot (parity-by-adoption — ours emits ggml's own block-dot; A/B≈1.0× *by construction* — where it holds)
| 格 | hot | cold (M=1) | cold (M=8) | 对手类 [机判·符号级] | 胜负 (cold) |
|---|:--:|:--:|:--:|---|---|
| **q4_0** | 0.983× | **0.983×** | 0.983× | native-RVV inline (41 ins / 16 rvv) | **PARITY 0.98×** (parity-by-adoption ✓; tautological) |
| **q4_1** | 1.007× | **1.008×** | 1.009× | native-RVV inline (45 / 14) | **PARITY 1.01×** (✓; tautological) |
| **q5_0** | 0.385× | **0.383×** | 0.386× | native-RVV inline (80 / 32, csrr) | **LOSS 0.38×** ★ parity-by-adoption **FALSIFIED** (§4.3) |
| **q5_1** | 0.402× | **0.400×** | 0.403× | native-RVV inline (90 / 30, csrr) | **LOSS 0.40×** ★ **FALSIFIED** |
| **q8_0** | 0.984× | **0.979×** | 0.983× | native-RVV inline (34 / 10) | **PARITY 0.98×** (✓; tautological) |

**cold tally (10 格)**: **1 WIN** (q5_K) · **3 PARITY** (q4_0, q4_1, q8_0 — all parity-by-adoption/tautological) · **6 LOSS**
(q2_K, q3_K, q4_K, q6_K weight-recon; q5_0, q5_1 parity-by-adoption falsified). M=1 and M=8 agree tightly (IQR 0.1–5.7 %).

---

## 2. 对手类机判 (objdump --disassemble 逐符号·非手写类目) — raw/k1_opponent_classification.txt

- **★ 纠正 prior**: 队列表 §一.C 预判「对手 = ggml generic block-dot (弱—中)」. **实测 as-shipped k1 VLEN256 dispatched path
  = 全 10 格 hand-tuned native-RVV** — 强对手, 非 generic scalar:
  - q2_K / q3_K / q5_K = **单体 inline heavy native-RVV** (243 / 116 / 97 rvv insn; VLEN-adaptive in-body).
  - q4_K / q6_K = **runtime dispatcher** (main reads `csrr vlenb`, branches → `ggml_vec_dot_qX_K_q8_K_vl256` hand-tuned
    specialization on k1 VLEN256). This is the strongest available deployed path.
  - FLAT q4_0/q4_1/q8_0 = compact native-RVV inline; q5_0/q5_1 = larger native-RVV inline (5-bit qh handling, csrr).
- **上游更强路径**: no unused stronger path withheld — the linked main symbol IS what ggml dispatches. (Distinct axis:
  ggml also ships `ggml_gemm_q4_K_16x1/8x8_q8_K` repack-GEMM — the *format's real beat lives there / on the repack axis,
  T9 §1.1 — not on this block-dot vec_dot axis; disclosed §4.)
- **OURS objdump self-probe** (raw/k1_build_seal.txt): FLAT clean-fp16, vsetvl 4 (q4_0/q8_0) … 9 (q5_0/q5_1); K-quant
  vsetvl **38–70** + widen 32–64 (= the "2995 vsetivli weight-recon floor" signature — many re-config per super-block).

---

## 3. 数值 byte-exact 逐格 (ZERO-MODEL vs stock-ggml oracle · INT-mode exact-integer fold)

All 10 formats × {M=1, M=8} × {nc=64 verify, nc=512 census}: **int_byte_mismatch = 0, worst_ulp = 0 → BYTE-EXACT-vs-stock-ggml**.
Sample folded integers (nonzero, varied): q2_K −191/756, q3_K 5570/−2982, q4_K −4115/−3817, q5_K 4327/383, q6_K −10407/4617,
q4_0 73/−167, q5_0 663/716, q8_0 −398/−41. Hard gate PASS (all cells).

---

## 4. 净新普查发现 + 输格三出口候选 (供 L1')

1. **对手非 generic = strong native-RVV** (§2). The prior's «weak—medium generic» opponent is falsified; K-quant LOSSes
   are vs hand-tuned vl256 specializations (q4_K/q6_K) or heavy inline RVV (q2_K/q3_K), which makes them *expected* and
   *stronger-adversary* datapoints, not straw-man wins-in-disguise.
2. **★ K-quant vec_dot loss cells (q2_K/q3_K/q4_K/q6_K) — 三出口候选**:
   - **Exit A — 对手结构优势具名 (accept as GAP)**: opponent = hand-tuned native-RVV (vl256 spec / heavy inline);
     ours = generic weight-reconstruction block-dot (vsetvl 38–70, weight-recon floor). Name `[GAP-KQUANT-VECDOT-VS-NATIVE-RVV]`.
   - **Exit B — 能力键控修法**: emit a VLEN256-specialized / wide-LMUL K-quant block-dot matching ggml's hand-tune
     (SEL-1 widest-LMUL prior; the emitter-maturity wide-LMUL K-quant gap already flagged in canon-memory).
   - **Exit C — 轴转移 (the real beat)**: K-quant's genuine beat is the **repack-GEMM axis** (T9 §1.1 / Batch1
     Win-K1-VLEN q4_K), NOT block-dot vec_dot. This vec_dot cell is inherently weight-recon-bound; disposition =
     stays LOSS-on-vec_dot-axis, format-beat routed to GEMM axis. **Recommended primary disposition.**
3. **★ FLAT parity-by-adoption 部分证伪**: q4_0/q4_1/q8_0 hold (0.98–1.01×, ours emits ggml's own compact block-dot).
   BUT **q5_0/q5_1 = 0.38–0.40× LOSS** — ours does NOT emit ggml's own path here; the 5-bit qh block-dot emit is a
   **divergent, ~2.6× slower variant** (ours vsetvl=9 vs opp compact 80/90-ins native). → the "FLAT vec_dot =
   tautological parity" assumption is **format-specific, not uniform**. q5_0/q5_1 are genuine LOSSES; flag
   `[GAP-Q5x-QH-BLOCKDOT-EMIT]` for L1'.
4. **★ q5_K WIN 1.09× — disclose mechanism**: the ONE K-quant format where the ggml opponent has **no vl256
   specialization** (single VLEN-agnostic body, 214/97). Ours ≥parity here = **opponent-immaturity-driven**, not
   "ours beats a hand-tune". Legit kernel-sym ≥parity vs a *real* native-RVV opponent, but **do NOT over-claim**;
   flag for L1' scrutiny.
5. **axis hygiene**: this 1-WIN/3-PARITY/6-LOSS cold tally is **kernel-sym micro (第二赛道)** — does NOT transplant to
   perf-covered 9/83 (system/e2e) nor to certified. 禁互推.

---

## 5. kernel-sym 计数建议 (主会裁)

- **Genuine (non-tautological) ≥parity from this census = q5_K 1.085×** (1 cell, vs real native-RVV opponent) — recommend
  **+1** to kernel-sym ≥parity **IF** 主会 accepts (a) vec_dot-axis inclusion, (b) discloses opponent-immaturity mechanism (§4.4).
- **FLAT parity-by-adoption (q4_0/q4_1/q8_0) = 3 cells ≥parity but tautological** (ours emits ggml's own block-dot).
  Whether these count = 主会裁; recommend **report-as-parity, do NOT count as independent beat**.
- **q5_0/q5_1 = LOSS** (parity-by-adoption falsified) → **exclude** from any parity-by-adoption count.
- **q2_K/q3_K/q4_K/q6_K = LOSS** (< parity; weight-recon floor vs strong native opponent) → not ≥parity.
- **Do not double-count with GEMM axis**: q4_K etc. already have separate repack-GEMM kernel-sym datapoints (T9/Batch1);
  this block-dot vec_dot axis is independent.
