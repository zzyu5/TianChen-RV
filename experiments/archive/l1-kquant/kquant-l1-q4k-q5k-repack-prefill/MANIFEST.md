# cell MANIFEST — kquant-l1-q4k-q5k-repack-prefill

- **campaign**: repack / [KQUANT-L1] (增补二 — gemm_tile L1 mineral vein: q4_K + q5_K repack GEMM
  prefill path-win probe on rvv/VLEN128; sibling of the q4_0 L1 precedent)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz)**. ★ L1 opponent
  -probe CRUX: **opponent (ggml b9692/f3e1828) has NO working repack path @ VLEN128 for either q4_K or
  q5_K** — q4_K's RISC-V trait is `case 128: {break;} // TODO` → nullptr (repack.cpp:4619); q5_K has NO
  ggml_cpu_has_riscv_v() branch at all → nullptr at every VLEN (repack.cpp:4644). Confirmed at the LIVE
  __riscv_vlenb() by a runtime dispatch probe. Both fall back to block-dot ggml_vec_dot_q{4,5}_K_q8_K.
  **=> L1 STRUCTURAL path-win candidate CONFIRMED for both** (same precedent as q4_0). Prefill paired
  (ours exported repack GEMM vs opponent's REAL linked block-dot, K=2048 nc=512 nr∈{16,64,256},
  best-of-5): **q4_K ≈ PARITY 0.94–0.97×** (opponent has a hand-tuned _vl128 block-dot), **q5_K ~1.50–
  1.62× faster** (opponent q5_K block-dot is generic/untuned, no vl128 variant). ⚠ memory-correction:
  the stale "ggml q4_K repack hardcoded VLEN256" is refined — the RISC-V q4_K 16x1 kernel now EXISTS
  but is VLEN256-gated (and would be numerically wrong at VLEN128: hardcoded vfmv...(,16) > VLMAX 8).
  **[NG-4] NOT a beat**: opponent-absence proven but eight [PERF-1] gates NOT walked; single-core;
  opponent = single-thread block-dot loop (kernel-axis proxy, not threaded mul_mat); throughput timing
  (our correctness = construction oracle 039133ea bounded-norm GEMM 8/8 ~7-8e-7, not re-verified here).
- **role**: [KQUANT-L1] q4_K/q5_K prefill repack path-win evidence. Answers the増補二 L1-critical
  question — does the opponent have a WORKING repack@VLEN128? — by (1) exact source of the board's own
  ggml, (2) a live-vlenb runtime dispatch probe, (3) a prefill paired vs the opponent's REAL dispatched
  block-dot (linked from the board's own libggml-cpu.so). OUR kernels are the front-door repack GEMM
  exported from HEAD 039133ea (VLEN128-correct half_lanes=8 two-strip; oracle-GREEN at construction).
- **layout**: org STAGE2 per-cell manifest. Emitter/op/verifier CODE lives outside this data cell
  (HEAD 039133ea: RVVOps.td + RVVToEmitC*.cpp + verifiers + oracle lit). Harness/driver/probe live
  under `tools/e2e-harness/board/` (kquant_dispatch_probe.c, kquant_gemm_paired_driver.c,
  kquant_gemm_paired.sh). This cell holds only data/evidence + the export recipe. The 1.3–1.8 MB
  exported .kernel.c are NOT stored (deterministically regenerable; see EXPORT_RECIPE.md) — an
  objdump seal stands in for the measured-binary provenance.

## durable files

- `opponent_probe.md`            — ★ L1 crux writeup: exact source gates (q4_K case 128 TODO; q5_K no
  riscv branch), runtime confirmation, opponent's real block-dot path + vl128/generic tuning asymmetry,
  prefill paired table, memory-correction, [NG-4] discipline.
- `dispatch_probe_raw.txt`       — on-board runtime dispatch probe output (LIVE VLEN=128) + verbatim
  source of the trait selector + nm symbol asymmetry.
- `prefill_paired.txt`           — raw board paired A/B (ours repack GEMM vs opponent linked block-dot),
  nr sweep, method + [NG-4] caveats + reproduce/restore.
- `prefill_paired.csv`           — parsed prefill table (fmt, nr, ours/opp GMAC/s, ratio, opponent path).
- `EXPORT_RECIPE.md`             — deterministic export pipeline (tcrv-opt --tcrv-rvv-lower-to-emitc |
  mlir-translate), op layout attrs, board compile (libcall-free), opponent link provenance, harness map.
- `kernels_objdump_seal.objdump` — objdump seal of OUR compiled q4_K/q5_K kernels (e32,m2 two-strip
  half_lanes=8 = VLEN128-correct; contrast ggml's hardcoded-16).

## note — data-only cell
Data/evidence only. The prefill ratios are **L1 path-CANDIDATE datapoints ([NG-4]), NOT beats** — the
load-bearing claim is the STRUCTURAL opponent-absence (proven), not a throughput win (q4_K is parity).
Nothing here was committed / git add-ed.
