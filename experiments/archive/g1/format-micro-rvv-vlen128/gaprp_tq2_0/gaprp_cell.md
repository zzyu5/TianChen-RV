# [GAP-RP] tq2_0 register-spill elimination — BOARD A/B + vs-generic (rvv/VLEN128, 裁决二.4)

> ★M2c min-term-bug 隔离确认(2026-07-09 G3-minterm-fix 裁决一.2):本 cell kernel(tq2_0 ternary 块点积)
> grep 无 `kquant_dmin_bsums_min` / dmin·bsums-min fold —— ternary 格、无 K-quant min fold。故与 M2c
> q4_K/q5_K/q2_K repack-GEMM min-term VLEN128 bug(commit 53666846)**无共享路径、不受影响**;1.92× A/B 数值不动。

Question (same read structure as [GAP-SB]): the tq2_0 format-micro loss was triaged to a
**LMUL-over-widen regfile spill** (4 whole-reg vector spill round-trips per super-block iteration).
Commit 3186919d ("Path 3: keep m4, kill overlap, serialize the 2 chunks") eliminates the spill,
byte-exact by construction. **Does the spill elimination transduce to a silicon speedup?**

board: ssh rvv riscv64 6.12.66 VLEN128 64c (L1d 64K/core, L2 2MiB/4c, L3 64MiB), gov=performance 2.6GHz, core8.
PRE = 0ad9cf6d (has spill), POST = 3186919d (spill gone). ours exported via tools/bench/byte-exact-baseline.sh
cached detached-worktree tcrv-opt/tcrv-translate (main tree + build/ UNTOUCHED; no git stash; nothing committed).

--------------------------------------------------------------------------------
## ANSWER (honest, [NG-4]): YES — spill elimination ~DOUBLES throughput.
- **A/B (PRE/POST) ≈ 1.92× isolated** (2-way) / **2.22–2.29× (3-way)**: POST 250–298 ns/block vs PRE 554–568 ns/block,
  cold (wset 256MiB > 3×L3), N=16 rounds. The 4-spill removal is on the critical path of this latency/compute-bound
  ternary kernel, so it roughly doubles throughput — much larger than the grid-decode-bound [GAP-SB] family (1.09–1.29×).
- **byte-exact**: PRE == POST FNV identical (0xeffb9951f963a9d6) on hardware — the fix changed only schedule/live-range
  (single-live i16m4 accumulator + one merged super-block reduce); intrinsics/width/integer reduction unchanged.
- **vs-generic**: ours-POST is **3.88–3.96× faster** than the ggml `ggml_vec_dot_tq2_0_q8_K_generic` scalar ref (WIN);
  even the spilling PRE was already 1.73–1.75× faster. PRE == POST == GENERIC bit-for-bit ⇒ **ULP=0 vs generic on hardware**.

--------------------------------------------------------------------------------
## 1. objdump — the spill, PRE vs POST (gcc-15.2.0 -O3, the FAITHFUL backend)
The spill is a **gcc-15.2.0 -O3 register-allocation artifact on the emitted C** — exactly the batch2c timed path
(NOT the tcrv-translate `--tcrv-export-target-artifact` codegen, which shows 0 spills at both PRE/POST). So the A/B is
timed on gcc-15.2.0 -O3 objects (the compiler under which the loss/spill was measured). raw/tq2_0_{PRE,POST}_gcc15.objdump.txt:

| stage | insns | whole-reg spills | (sp) access | vwredsum | vwmul/vwmacc (gearbox) |
|---|---:|---:|---:|---:|---:|
| PRE (0ad9cf6d)  | 97 | **4** (vs2r.v@0x70→(a4), vs2r.v@0xa0→(sp), vl2r.v@0xc4←(a0), vl2r.v@0xe0←(sp)) | 2 | 2 | 8 |
| POST (3186919d) | 80 | **0** | 0 | 1 | 8 |

- PRE = the 病灶: i16m4 accumulator ×2 pipelined chunk + hoisted m4 seed ⇒ 2 e8m2 q8 strips (v4–v7) spill = 4 DRAM
  round-trips/super-block iteration. POST = Path 3: single-live m4 accumulator, 2-chunk reduce merged to one (vwredsum 2→1).
- **vwmul/vwmacc 8/8 UNCHANGED** PRE→POST ⇒ integer gearbox byte-exact-preserved (matches the FNV equality).

## 2. A/B (isolated PRE-vs-POST) — cache-cold N=16, median ns/block ★
- 2-way (cleanest isolation): run1 A/B=1.9420 (PRE 567.997 / POST 292.483); run2 A/B=1.8896 (562.292 / 297.576).
- 3-way (with generic interleaved): run1 A/B=2.2193 (553.889 / 249.582); run2 A/B=2.2860 (570.732 / 249.664).
- POST IQR tight (6–14 ns, ≤5%); PRE IQR 17–40 ns (3–7%, the spill adds run-to-run variance). Robustly ≈2×; the spill
  elimination is the dominant lever for this format (vs the batch2c 4.33× vs-SIMD loss). raw/board_ab.txt.

## 3. vs-generic (the defensible reference gap) — ggml scalar ref, byte-exact
factory = `ggml_vec_dot_tq2_0_q8_K_generic` from the board's pinned ggml quants.c (clang-17 -O2, same march,
-ffunction-sections + --gc-sections, objcopy-renamed). GEN/POST = 3.88–3.96× (ours FASTER = WIN); GEN/PRE = 1.73–1.75×.
PRE==POST==GENERIC FNV all 0xeffb9951f963a9d6 ⇒ ours == ggml generic bit-for-bit (ULP=0).

## 4. framing ([NG-4] — honest scope)
- A/B = isolated internal metric (spill-fix delta). vs-generic = ours beats the ggml scalar reference (WIN, byte-exact).
- vs the ggml **SIMD-dispatch** factory (batch2c: tq2_0 0.2311× = ours 4.33× slower, measured at ffcfcf80 = the SPILLING
  version): the spill-fix roughly halves that gap (~4.33/2 ≈ 2.2× slower ⇒ ~0.45×) but tq2_0 ours still LOSES to the
  hand-tuned SIMD dispatch. This is NOT a vs-SIMD-factory beat and NOT an e2e [PERF-1] eight-gate. KERNEL-micro only.
- tq1_0 was flagged the same spill class (larger blast radius) and is NOT fixed yet (commit note) — remaining.
