# cell MANIFEST — flat-covering-batch1 (线乙 覆盖式铺面①：FLAT 对位)

- **campaign**: 测量总攻 / 覆盖式铺面① — FLAT (q4_0-family) repack GEMM prefill fair-align vs ggml's
  real dispatched VLEN128 block-dot. Batch cells: **q4_1 / q5_0 / q5_1** (empty cells) + **q8_0** (争夺格).
  q4_0 measured as the known-good **reference** (pipeline validation).
- **status**: ACTIVE — **BOARD-MEASURED 2026-07-11**, rvv (openEuler 24.03 LTS, localhost.localdomain,
  VLEN128, 64c, core 40, governor=performance @ 2.6 GHz). HEAD 2be7f3d2 (no git; main tree + build/ untouched).
- **role**: fair repack-vs-block-dot paired A/B on the constructed FLAT repack GEMM kernels, with a REAL
  numeric ZERO-MODEL correctness gate + compiler-symmetric double-ledger + opponent-identity probe.

## board identity fingerprint
- host=localhost.localdomain / openEuler 24.03 LTS / VLEN128 / rv64gcv+zfh+zvfh / 64c / core40 perf-gov 2.6GHz
- toolchains: deployment **clang 18.1.8** (/opt/tcrv-toolchains) ; symmetric-kernel **gcc 15.2.0** (= the
  compiler that built the opponent .so) ; link/harness always gcc-15.2.0 (neutral)
- opponent = board's own `libggml-cpu.so` (md5 `d1adc634c2ca04ffc389536b30d9d9c4`, tag b9692 / commit f3e1828,
  built gcc-15 rv64gcv). Opponent symbols are the arch-dispatched `ggml_vec_dot_qX_qY` (NOT `_generic`).

## method (touch-set: tools/e2e-harness/board/flat_gemm_paired{_driver.c,.sh})
1. **Export** OUR 5 FLAT repack GEMM kernels from HEAD 2be7f3d2 (host build/bin/tcrv-opt, READ-ONLY):
   `tcrv-opt <rvv-emit-quant-contraction-{q4-0,q4-1,q5-0,q5-1,q8-0}-repack-gemm-prefill-vlen128.mlir>
    --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`
   (FRONT-DOOR: abstract quant_contraction -> in-compiler REPACK select -> typed_repack_gemm_loop_body region
   -> emitc). All 5 lower with ZERO errors. ABI = `(nr, bs, K, float* s, nc, uint8_t* vx=weights, uint8_t* vy=act)`,
   4x16 output tiling (2 strips of 8 lanes), int8xint8->i32 integer core == ggml block-dot semantics.
2. **ZERO-MODEL numeric gate (correctness BEFORE timing)**: generate random f16-scaled PLAIN weight blocks
   (block_qX) + PLAIN q8_0/q8_1 activation blocks; (a) OPPONENT ref = ggml `ggml_vec_dot_qX_qY` per (row,col);
   (b) an INDEPENDENT scalar decoder (fp64, in-driver, zero reuse of our intermediates); (c) OURS = repack the
   SAME plain blocks into the interleaved x16/x4 layout + run our kernel. Compare ours & opp each vs the scalar
   ref. **All 5 formats bit-exact (relerr 1e-5..2e-4, `relerr_ours ~= relerr_opp` = ours is exactly as accurate
   as ggml's own block-dot).** Two repack subtleties DERISKED by this gate: q4_0 needs the ggml **XOR-0x88 sign
   trick** (signed nibble via sign-extend); q5_0/q5_1 need the **transposed-qh** layout (32×u16, bit c = column c
   qh bit at position p, extracted per-lane by column index). Cross-checked by the construction byte-exact oracle
   lit (10/10 GREEN: emit + identity for all 5, run locally).
3. **Timing**: driver best-of-7 (warmup + N iters), outer reps=8 (nr=16) / reps=3 (nr=64); `taskset -c 40`;
   3× self-check IQR ~0.9% << noise-floor×1.5; median reported, IQR% logged.
4. **Compiler-symmetric double-ledger** [CASE-COMPILER-ASYMMETRY]: only OUR-kernel codegen varies (clang vs gcc);
   driver+link always gcc-15. **system/deploy account** = clang-ours vs gcc-shipped-blockdot ; **kernel account**
   = gcc-ours vs gcc-shipped-blockdot (symmetric).

## HEADLINE RESULTS (median ratio ours/opp, VLEN128, K=2048, nc=512, nr=16; see flat_gemm_paired.csv)
| fmt | ours GMAC/s | opp GMAC/s | **system(clang)** | **kernel(gcc-symmetric)** | opp vec-level | verdict |
|-----|-----|-----|-----|-----|-----|-----|
| q4_1 (tracer) | 9.08 | 1.263 | **7.19×** | **6.86×** | ~6 rvv-insn | WIN, holds both |
| q4_0 (ref)    | 8.18 | 1.231 | 6.65× | 6.76× | ~6 rvv-insn | WIN, holds both |
| q8_0 (争夺)    | 4.34 | 1.242 | 3.50× | **4.10×** | ~6 rvv-insn | WIN kernel-axis (selector caveat) |
| q5_1          | 3.92 | 2.90  | 1.33× | 1.41× | ~17 rvv-insn | modest WIN |
| q5_0          | 3.75 | 3.09  | 1.22× | 1.23× | ~17 rvv-insn | ~parity-plus |

## ★load-bearing finding: wins HOLD under compiler symmetry
Unlike the retracted K-quant S6 (1.884× clang-vs-gcc artifact -> 0.272× symmetric), the FLAT repack GEMM wins
**survive the gcc-symmetric kernel account** (q4_0 6.76× / q4_1 6.86× / q8_0 4.10× / q5_1 1.41× / q5_0 1.23×) —
if anything the gcc account is HIGHER for q4_0/q8_0/q5_1. => these are genuine **memory-locality / decode-
amortization path wins** (block-as-16-lanes streams weights + amortizes decode once-per-16-cols), compiler-
independent, NOT a clang-codegen artifact. This is the batch's most important positive result.

## opponent-absence (structural L1 path-win basis) — CONFIRMED on THIS board's ggml
`ggml/src/ggml-cpu/repack.cpp`: q4_0 (:4589) and q8_0 (:4710) have `if riscv_v: switch(vlenb*8){ case 128:
{break;}//TODO; case 256: return &qX_16x1_qY; }` => **@VLEN128 repack = nullptr => block-dot fallback** (VLEN256/k1
only). q4_1/q5_0/q5_1 have NO riscv repack branch at all. So for ALL 5 FLAT formats the block-dot we timed is
exactly ggml's real dispatched VLEN128 prefill path. Opponent vectorization (objdump): q4_0/q8_0 block-dot ~6
rvv insns (light intrinsic), q5_0 ~17 (better vectorized => higher opp throughput => our smaller margin).

## judgment (predicted three-exit; ZERO undefined yellow)
- **Correctness**: all 5 PASS (numeric ZERO-MODEL + construction lit). GREEN.
- **Kernel-axis ledger (compiler-symmetric)**: q4_0/q4_1/q8_0 = strong wins (6.76/6.86/4.10×); q5_1 modest
  (1.41×); q5_0 ~parity-plus (1.23×). All ≥ parity, correctness-gated, opponent-identity-probed.
- **Eight-gate for "beat"**: 6/8 pass (byte-exact ✓ / 对手对称 ✓ / 双账本 ✓ / 对手身份探针 ✓ / 纪律 ✓ / 措辞 ✓);
  **NOT passed: micro∧e2e** (this is MICRO/kernel-axis prefill GEMM only — no whole-model llama-bench) and
  **selector-e2e-routing** (front-door CONSTRUCTS+selects repack by lit, but real-llama routing not exercised;
  q8_0 selector historically DECLINEs). => **黄格 named [GAP-FLAT-E2E]**: needs T6 whole-model phase-split per
  format to confirm the prefill win transplants (per [kernel-wins-dont-transplant-to-e2e], prefill/memory-locality
  wins DO tend to transplant -> high-value follow-up, not a null). No cell is undefined.
- **q8_0 争夺 conclusion**: OUR constructed q8_0 repack GEMM beats the dispatched block-dot **4.10× (gcc-symmetric
  kernel account) / 3.50× (clang deploy)**, correctness bit-exact, NET-NEW construction (no prior q8_0 GEMM
  emitter). Honest caveat: (1) the path-selector historically self-DECLINEs routing q8_0 -> no automatic e2e
  engagement; (2) opp block-dot is light-intrinsic (~6 rvv insn), so wording tier = repack-vs-dispatched-intrinsic
  block-dot (not vs hand-tuned asm). Data + caveats given; no beat asserted.

## [NG-4] discipline
All numbers are L1 kernel-axis path-CANDIDATE datapoints on the kernel/compiler-symmetric ledger. NOT [PERF-1]
eight-gate beats (micro∧e2e gate open). Reported as candidates + named gap, not wins.

## data-only cell
Evidence + recipe only; the exported .kernel.c are deterministically regenerable (see method). Board scratch
/tmp/flat_export cleaned after archiving. Nothing git-added.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `EXPORT_RECIPE.md` — deterministic re-export recipe for the 5 FLAT repack GEMM kernels (HEAD 2be7f3d2).
- `flat_gemm_paired.csv` — the paired repack-vs-block-dot A/B measurement grid (q4_0/q4_1/q5_0/q5_1/q8_0).
- `opponent_probe.md` — opponent-identity probe (board `libggml-cpu.so` md5 + dispatched `ggml_vec_dot_qX_qY`).
- `raw_board_results.txt` — full board stdout (fingerprint + ZERO-MODEL correctness gate + paired timing).
