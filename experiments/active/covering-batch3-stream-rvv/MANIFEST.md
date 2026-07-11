# cell MANIFEST -- covering-batch③ (线丙 覆盖式铺面③：流式格 kernel-axis 对位)

- **campaign**: 测量总攻 / 覆盖式铺面③ — streaming-format (dequantize_row / quantize_row / forward
  elementwise) kernel-axis fair-align vs ggml's own dispatched streaming kernels. Streaming formats are
  MEMORY-BOUND (to-the-wall) operators; honest prior = physical parity for most, "to-wall speed" wins
  logged as-is. **NOT chasing perf-covered green** (streaming parity is the correct prediction).
- **status**: ACTIVE — BOARD-MEASURED 2026-07-11, rvv (openEuler 24.03, localhost.localdomain, VLEN128,
  64c, core 40, perf-gov). HEAD 11e86139 (no git; main tree + build/ untouched).
- **role**: fair OUR-front-door-streaming-kernel vs ggml-dispatched-streaming paired A/B on cold-streamed
  pools (>LLC), REAL correctness gate BEFORE timing, compiler-symmetric double-ledger, opponent-id probe.

## board identity fingerprint
- host=localhost.localdomain / openEuler 24.03 / VLEN128 / rv64gcv+zfh+zvfh / 64c / core40 perf-gov
- toolchains: deploy **clang 18.1.8** ; kernel-symmetric **gcc 15.2.0** ; driver+link ALWAYS g++-15 (neutral)
- opponent = board's own `libggml-base.so.0` (md5 `1b4580c4…`, dequant) + `libggml-cpu.so.0`
  (md5 `d1adc634…`, quant), tag b9692/f3e1828 gcc-15. Forward opponents = transcribed `ggml_vec_*`/gelu refs.

## sampling strategy (declared coverage -- full sweep too expensive)
Three morphologies sampled per family; untested same-family formats declared-covered (same律, parity类推).
- **dequant (24 fmts)**: measured **q8_0 (flat/scalar)** · **q4_K (super-block)** · **iq4_nl (codebook)**.
  Declared-covered by morphology: flat {q4_0,q4_1,q5_0,q5_1,q8_0} · super-block {q2_K,q3_K,q4_K,q5_K,q6_K,q8_K}
  · codebook/grid {iq1_s,iq1_m,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,iq4_xs} · ternary {tq1_0,tq2_0}
  · fp4 {mxfp4,nvfp4} · binary {q1_0}. (iq4_nl exposes the codebook gather-trap that all codebook fmts share.)
- **quant (3 fmts)**: measured **q8_0 · q8_K**. Declared-covered: q8_1 (same q8-family scale+int8 store).
- **forward (elementwise)**: measured **add · mul (binary vector-map)** · **gelu (unary scalar-map)**.
  Declared-covered: {scale, cpy} (trivial memory-bound maps → parity类推) · {silu} (same scalar-transcendental
  class as gelu) · {rms-norm, soft-max, rope} (reduction/rotate — separate shape, out of this batch's map class).

## method (touch-set: tools/e2e-harness/board/stream_paired_{driver.c,board.sh})
1. **Export** OUR streaming kernels from HEAD 11e86139 (host build/bin/tcrv-opt READ-ONLY):
   dequant `--tcrv-rvv-lower-to-emitc` ; forward `--tcrv-rvv-materialize-forward-elementwise-stream-front-door
   --tcrv-rvv-lower-to-emitc` ; quant `--tcrv-rvv-lower-to-emitc` ; then `mlir-translate-20 --mlir-to-cpp`.
   All lower with ZERO errors. ABIs: dequant `(size_t k, const uint8_t* x, float* y)` ;
   quant `(size_t k, const float* x, uint8_t* y)` ; forward-binary `(size_t n, const float* a,b, float* o)` ;
   forward-unary `(size_t n, const float* x, float* y)`.
2. **Correctness gate BEFORE timing**: ours vs opponent on IDENTICAL random input. dequant/quant byte-exact
   (maxrel=0 / bytes_diff=0) ; forward maxrel<5e-4 (add/mul exact; gelu 1.31e-6 vs tanh ref). ALL 8 PASS.
3. **Cache hygiene**: cold pool with in+out footprint 115–302 MiB (>LLC); each timed round sweeps the whole
   pool once (n=25.2M elems) so every block streams cold. achieved_GBs reported for roofline classification.
4. **Timing**: paired ours/opp alternation × 15 rounds, median, taskset -c 40. Self-check 3× q8_0 IQR 0.14–0.19%
   (<< noise-floor×1.5). DCE defeated by checksum sink.
5. **Compiler-symmetric double-ledger**: only OUR kernel .o codegen varies (clang deploy / gcc symmetric);
   driver+link always g++-15; opponent fixed.

## HEADLINE RESULTS (deploy=clang ledger / kernel=gcc-symmetric ledger; ratio ours/opp, >1 = ours faster)
| mode | fmt | byte-exact | deploy(clang) | kernel(gcc) | roofline | exit |
|------|-----|-----|-----|-----|-----|-----|
| dequant | q8_0   | ✓ maxrel0 | 0.98× | 0.86× | mem-bound scalar 3.7 GB/s | **parity-physical** |
| dequant | q4_K   | ✓ maxrel0 | 0.88× | 0.64× | mem-bound scale-unpack 3.0 GB/s | **named-gap [GAP-DEQ-KQUANT-UNPACK]** |
| dequant | iq4_nl | ✓ maxrel0 | 0.94× | **4.83×** | compute-bound gather 0.55 GB/s | **split**: deploy parity-physical / kernel-candidate 4.83× blocked [GAP-CLANG-GATHER-TRAP] |
| quant   | q8_0   | ✓ diff0   | 0.99× | 1.00× | mem-bound 2.8 GB/s | **parity-physical** |
| quant   | q8_K   | ✓ diff0   | 1.08× | 0.98× | mem-bound 2.5 GB/s | **parity-physical** |
| forward | add    | ✓ maxrel0 | 0.83× | 0.81× | near-wall 7.3 GB/s | **named-gap [GAP-FWD-M8-VSETVL]** |
| forward | mul    | ✓ maxrel0 | 0.76× | 0.76× | near-wall 6.4 GB/s | **named-gap [GAP-FWD-M8-VSETVL]** |
| forward | gelu   | ✓ 1.3e-6  | 0.99× | 1.00× | compute-bound tanhf 0.11 GB/s | **parity-physical** (LUT-deploy caveat) |

## judgment (three-exit; ZERO undefined)
- **parity-physical (4)**: q8_0-deq, q8_0-quant, q8_K-quant, gelu-fwd — memory/compute-bound, byte-exact,
  our streaming kernel ≈ ggml's at the wall = 黄格 full-marks (roofline-bound, NOT a GAP, NOT a loss).
- **split-ledger (1)**: iq4_nl-deq — deploy(clang) parity 0.94× (both hit the vluxei gather trap), kernel
  (gcc-symmetric) 4.83× candidate (gcc scalarizes OUR source, beats ggml's own gcc gather) — latent win
  blocked by our clang codegen. Named [GAP-CLANG-GATHER-TRAP]. NOT a beat (deploy parity; eight-gate open).
- **named-gap (3)**: q4_K-deq (0.64–0.88×, super-block scale-unpack compute), add/mul-fwd (0.76–0.83×,
  explicit-m8 map with per-iter vsetvl vs gcc-autovec). All byte-exact, roofline-close; emitter-maturity
  soft gaps 待修, NOT physical walls or algorithm defects.

## perf-covered contribution: 0 new green (as predicted)
NONE pass the eight-gate for "beat" (all kernel-axis micro, no e2e; iq4_nl kernel-candidate blocked by the
deploy gap). Streaming batch = T3 kernel-axis coverage rows + one mechanistic finding (clang gather-trap),
NOT perf-covered green — consistent with T6 finding (perf-covered moves only via the wiring battle).

## [NG-4] discipline
All numbers are kernel-axis path-CANDIDATE / parity datapoints on the double ledger. Parity is reported as
parity (full marks, not dressed as a win); the iq4_nl gcc 4.83× is reported as a compiler-symmetric-valid
candidate with the deploy-gap caveat, NOT asserted as a beat. Correctness-gated, opponent-identity-probed.

## data-only cell
Evidence + recipe only; exported .kernel.c deterministically regenerable (see EXPORT_RECIPE.md). Board
scratch /tmp/stream_batch3 cleaned after archiving. Nothing git-added.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `EXPORT_RECIPE.md` — deterministic re-export recipe for OUR streaming kernels (HEAD 11e86139).
- `opponent_probe.md` — opponent-identity probe (board `libggml-base/cpu.so` md5 + symbol dispatch).
- `raw_board_results.txt` — full board stdout (fingerprint + correctness gate + paired timing).
- `stream_paired.csv` — the paired ours/opp streaming A/B measurement grid (dequant/quant/forward).
