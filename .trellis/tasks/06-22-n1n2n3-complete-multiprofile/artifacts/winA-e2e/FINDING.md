# Win-A LMUL-WIDTH tune ablation (repack GEMV/GEMM, WIDE vs NARROW) — 2026-06-23

The Win-A "automatic tuning choice" ablation per `N3-METHODOLOGY.md`: TWO compiler-emitted repack
kernels differing ONLY in `integer_core_lmul` — WIDE (`m1`, whole-LMUL chain, ONE 16-lane strip) vs
NARROW (`mf2`, RVV1.0 fractional chain, TWO 8-lane strips). Same algorithm, same bytes read, only the
tuned knob (LMUL width + the strip count it forces). Both arms are compiler emissions of the SAME
`tcrv_rvv.repack_gemv/gemm_q4_0_q8_0` op; neither is hand-written. Measured on `ssh rvv` (Sophgo
SG2044, RVV1.0, VLEN128, clang18).

## RESULT IN ONE LINE
Microbench ablation is a CLEAN, POSITIVE Win-A: **WIDE beats NARROW 2.1× (GEMV/decode) and 1.3×
(GEMM/prefill), byte-exact (norm 0).** The e2e leg is **BLOCKED**: the WIDE `.so` deterministically
fails to route to the emitted kernel at the llama integration layer (no `ENGAGED`, ~3× slower than even
OFF, exit 0, empty stderr), while the identical-pipeline NARROW `.so` engages and runs correctly. So
the e2e WIDE/NARROW ratio could not be measured; the microbench is the decisive ablation harness here.

## How the two arms were produced (both compiler-emitted, verified)
- Source op MLIR: validated repack GEMV (`emit-repack-gemv/input-repack-gemv.mlir`) + GEMM
  (`emit-repack-gemm/input-repack-gemm.mlir`).
- NARROW: `tcrv-opt IN --tcrv-rvv-materialize-repack-strip-width=march=rv64gcv --tcrv-rvv-lower-to-emitc
  | mlir-translate --mlir-to-cpp` → `mf2`, `half_lanes=8` → i8mf2→i16m1→i32m2→f32m2,
  **vl=8, TWO strips** (4× `vle8_v_i8mf2` in GEMV).
- WIDE: same pipeline, `march=rv64gc_xtheadvector` (the pass stamps `integer_core_lmul="m1"` +
  `half_lanes=16`) → i8m1→i16m2→i32m4→f32m4, **vl=16, ONE strip** (2× `vle8_v_i8m1` in GEMV). The m1
  whole-LMUL chain is legal RVV1.0; compiled `rv64gcv_zvfh` for the VLEN128 hardware.
- LMUL spellings fully disjoint (grep-verified): WIDE = {i8m1,i16m2,i32m4,f32m4,f16m2}; NARROW =
  {i8mf2,i16m1,i32m2,f32m2,f16m1}. No leak either way.
- Binary verification (`objdump`): WIDE repack carries `vsetivli …,16,e16,m2` + `e8,m1`; NARROW carries
  `e16,m1` + `e8,mf2`. Three distinct `.so` md5s: WIDE ba8be266 / NARROW cabcd588 / OFF 1f2727b5. Both
  emitted symbols are `T`-defined in both `.so` (nm -D); symbol SETS identical (empty diff).

## Single-core microbench — the clean ablation (`ablation_micro.log`) — WIDE genuinely wins
norm(WIDE vs NARROW) = **0.000e+00** byte-exact (the 16-way-interleaved repack reads identical bytes
either LMUL — the LMUL is the *how*, not the *what*). Best-of-reps min latency:

| kernel | shape | NARROW (mf2, 2×8) | WIDE (m1, 1×16) | **ratio NARROW/WIDE** |
|---|---|---|---|---|
| **GEMV** (decode, 1 act col) | n=4096 nc=4096 | 3.30–3.60 ms | 1.56–1.63 ms | **2.12–2.21×** |
| **GEMV** (decode) | n=4096 nc=11008 | 8.88 ms | 4.20 ms | **2.11×** |
| **GEMM** (prefill, M=4 cols) | n=4096 nc=4096 | 9.45–10.0 ms | 7.26–7.42 ms | **1.27–1.38×** |
| **GEMM** (prefill) | n=4096 nc=11008 | 29.1 ms | 21.3 ms | **1.36×** |

The advisor's "per-block-reduce wall" worry does NOT apply to the repack: GEMV/GEMM accumulate
LANE-WISE via `vwmacc` (no `vredsum`/`vwredsum` anywhere), so widening LMUL halves the strip count
cleanly → **decode 2.1×**. GEMM (prefill) is a more modest 1.3× because the WIDE whole-LMUL f32m4
chain would spill the 32-vreg file, so the emitter forces a 4-pass-1-column re-decode (re-reading the
shared weight nibbles 4×) — a real cost that eats the width advantage. (This 4-pass form is exactly the
documented spill-avoidance in `RVVToEmitCBlockQuantLinear.cpp` ~line 2209.)

## llama.cpp e2e (`e2e_runs.log`) — WIDE arm BLOCKED, NARROW arm works
Swap `libggml-cpu.{WIDE,NARROW}.so` into the build, back-to-back, real 7B llama-2-chat Q4_0.

| arm | t16 pp256 | t16 tg64 | ENGAGED | note |
|---|---|---|---|---|
| **NARROW (mf2)** | **18.42 t/s** | **9.01 t/s** | GEMM+GEMV (8 lines) | emitted path engages, correct |
| **WIDE (m1)** | 3.14 t/s | 2.59 t/s | **0 (never engages)** | falls to non-emitted path |
| (isolated pp16/tg8) | NARROW 2.57/1.39 ENGAGED · WIDE 0.79/0.73 ENGAGED=0 |||

The WIDE `.so` runs ~3× SLOWER than even the OFF baseline (which is ~3.15 t/s) and prints zero
`ENGAGED` lines with a 0-byte stderr (exit 0, no fault). It deterministically survives a forced clean
rebuild (byte-identical md5). Since (a) the WIDE kernels are numerically correct (microbench norm=0 vs
NARROW), (b) both `.so` export identical symbol sets with both emitted funcs `T`-defined, and (c) the
NARROW `.so` from the identical pipeline engages perfectly — the non-engagement is an **integration /
dispatch interaction** in llama's repack mul_mat with the WIDE `m1` `.inc` (most plausibly the larger
4-pass GEMM `.inc` perturbing the repack-type routing / a runtime path selection), NOT a kernel-science
issue. Root-causing the llama dispatch is out of scope for the LMUL ablation and was time-boxed; the
clean ablation lives in the microbench.

## Honest framing (status: PARTIAL — microbench measured, e2e blocked)
- The LMUL-width tune DID transplant into the repack at the kernel level: **decode 2.1×, prefill 1.3×,
  byte-exact** — a real, clean Win-A ablation (both arms compiler-emitted, only the knob differs). It did
  NOT hit the per-block-reduce wall.
- The e2e WIDE/NARROW ratio is NOT measured (WIDE arm doesn't engage e2e — an integration block, with
  the working NARROW e2e baseline 18.4/9.0 t/s recorded for the record).
- The independently-defensible **"Win-A in llama" e2e** therefore remains the already-measured
  **VLEN-strip selection** (1×16 vs 2×8, both compiler-emitted, only strip width differs): **1.48×
  microbench / 1.31× e2e on K1 (VLEN=256)** — see `../winA-in-llama-FINDING.md`. This new microbench
  ablation STRENGTHENS the LMUL-width story (2.1× decode in isolation) but the e2e for *this* knob is
  blocked.

## Artifacts
`ablation_micro.log` (microbench), `e2e_runs.log` (e2e WIDE-blocked + NARROW baseline), kernels on rvv
under `~/winA-scratch/` (`gemv_{WIDE,NARROW}.cpp`, `gemm_{WIDE,NARROW}.cpp`, `ablation_micro.cpp`,
`libggml-cpu.{WIDE,NARROW}.so`).
