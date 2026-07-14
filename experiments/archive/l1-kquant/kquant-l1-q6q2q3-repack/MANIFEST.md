# cell MANIFEST — kquant-l1-q6q2q3-repack

- **campaign**: repack / [KQUANT-L1] (增补二 增补 — gemm_tile L1 mineral vein: q6_K + q2_K + q3_K
  repack GEMM+GEVM silicon numerical + opponent-absence + prefill probe on rvv/VLEN128; completes the
  K-quant family after the q4_K/q5_K sibling)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz, governor=
  performance)**. THREE findings, per format:
  ① **SILICON NUMERICAL — CONFIRMED (load-bearing).** The construction oracle was a C++ emitter-MODEL
  vs reference (never hardware); this cell compiles the ACTUAL exported `.kernel.c` (HEAD 924dc31f)
  with board clang-17 and RUNS it on rvv silicon vs an independent reference decoded from the ORIGINAL
  block. GEVM+GEMM, 8 shapes, 2 seeds: **int-mismatch=0 everywhere (BYTE-EXACT-INTEGER on silicon)** +
  bounded-norm `maxAbsErr/RMS ≈ 6–7e-7`. fp16-hardware, libcall-free. Model certificate → hardware-run.
  ② **OPPONENT-ABSENCE @VLEN128 — PROVEN (L1 structural).** Board's own ggml (llama.cpp-upstream-native
  commit f3e1828) `get_tensor_traits` returns nullptr for all three at VLEN128: q2_K = riscv branch
  `case 128 {break;}//TODO`; q6_K = NO riscv_v branch (NEON-only); q3_K = **no selector case at all +
  no q3_K repack kernel anywhere in ggml** (strongest absence). Verbatim source + LIVE `__riscv_vlenb()
  =128` dispatch probe. All fall back to block-dot `ggml_vec_dot_q{6,2,3}_K_q8_K`.
  ③ **PREFILL — LOSS, NOT a beat.** Ours repack GEMM vs opponent's REAL linked block-dot (K=2048,
  nc=512, nr∈{16,64,256}, paired N=10, cold-flush 224MiB>3×L3, best+median): **ratio ours/opp ≈
  0.18 (q6_K) / 0.21 (q2_K) / 0.176 (q3_K) — our first-construction kernels are ~5–6× SLOWER.**
  Honest cause: correctness-first, un-pipelined emit (q2_K GEMM 77 vsetvli, ~0.2–0.3 MAC/cycle, ~20×
  off peak) vs ggml's MATURE VLEN128 block-dot (q2_K/q3_K carry hand-tuned `_vl128` variants; `nm`).
  **[NG-4] honored (moot — a loss is not a beat)**; single-core; opponent=single-thread block-dot loop
  (kernel-axis proxy, not threaded mul_mat, eight [PERF-1] gates NOT walked).
- **role**: [KQUANT-L1] q6_K/q2_K/q3_K silicon evidence completing the K-quant L1 family. Answers the
  增補二 L1-critical question (working opponent repack@VLEN128? — NO for all three) AND upgrades the
  construction oracle to a hardware-run byte-exact-integer + bounded-norm certificate. OUR kernels are
  the front-door repack GEMM/GEVM exported from HEAD 924dc31f (VLEN128-correct half_lanes=8 two-strip).
- **layout**: org STAGE2 per-cell manifest. Emitter/op/verifier CODE lives outside this data cell
  (HEAD 924dc31f: RVVOps.td + RVVToEmitC*.cpp + verifiers + oracle lit). Harness/driver/probe live
  under `tools/e2e-harness/board/` (kquant_repack_verify_q{6,2,3}K.c, kquant_dispatch_probe_q6q2q3.c,
  kquant_gemm_paired_q6q2q3_driver.c). This cell holds only data/evidence + the export recipe. The
  0.8–1.8 MB exported `.kernel.c` are NOT stored (deterministically regenerable; see EXPORT_RECIPE.md);
  an objdump seal stands in for measured-binary provenance.

## durable files

- `opponent_probe.md`            — ★ L1 crux writeup: silicon numerical certificate, exact source
  gates (q2_K case-128 TODO / q6_K NEON-only / q3_K no-case), LIVE dispatch confirmation, opponent's
  real block-dot path + `_vl128` tuning, prefill paired table, [NG-4] loss discipline.
- `numeric_silicon.txt`          — raw board output of the three silicon numerical verifiers (INT
  byte-exact + NORM bounded-norm, GEVM+GEMM, 2 seeds); board/clang provenance header.
- `numeric_silicon.csv`          — parsed numerical table (fmt, mode, kernel, shape, int_mismatch,
  norm, verdict).
- `dispatch_probe_raw.txt`       — LIVE `__riscv_vlenb()=128` runtime dispatch probe output + VERBATIM
  board repack.cpp source (Q2_K/Q6_K branches), q3_K absence grep, riscv-arch 16x1 inventory, `nm`
  opponent-symbol asymmetry.
- `prefill_paired.txt`           — raw board paired A/B (ours repack GEMM vs opponent linked block-dot),
  nr sweep, method + cold-flush/N=10 protocol + [NG-4] caveats.
- `prefill_paired.csv`           — parsed prefill table (fmt, nr, ours/opp GMAC/s, ratio, opponent path).
- `kernels_objdump_seal.txt`     — objdump seal of OUR compiled q{6,2,3} GEMM/GEVM kernels (vsetvli
  counts) + fp16-hardware libcall-free check.
- `EXPORT_RECIPE.md`             — deterministic export pipeline (tcrv-opt --tcrv-rvv-lower-to-emitc |
  mlir-translate), REAL byte-layout attr table, board compile, opponent link provenance, harness map.

## note — data-only cell
Data/evidence only. The load-bearing claim is ① the SILICON byte-exact-integer + bounded-norm
numerical certificate (proven) and ② the STRUCTURAL opponent-absence (proven). ③ the prefill ratios
are a **documented LOSS ([NG-4]), NOT a beat** — our first-construction kernels are ~5–6× slower than
ggml's mature VLEN128 block-dot (marginal-cost / compiler-maturity evidence, perf axis pending).
Nothing here was committed / git add-ed.
