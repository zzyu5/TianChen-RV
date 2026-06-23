# K-quant block-dot Win-B·micro — OUR emitted kernel vs ggml's shipped RVV kernel

**Provenance**: `ssh k1` (Spacemit X60, RVV1.0, VLEN=256, 8 cores). Compiler:
clang18 (`-O3 -march=rv64gcv_zvl256b -ffp-contract=fast`), both TUs identical flags.
taskset-pinned (`taskset -c 2`). Best-of-`reps` min, timing_n super-blocks/call.

**Method (kernel-isolated microbench)**: OUR block-dot is emitted via
`build/bin/tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`
from `test/Conversion/RVV/rvv-to-emitc-q*-k-q8-k-block-dot.mlir`. The ggml baseline
is the **shipped optimized RVV kernel** `ggml_vec_dot_q*_K_q8_K_vl256` extracted from
`/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c` (the VLEN256
variant — the methodology-correct Win-B baseline on K1), NOT `_generic`/scalar.
Both fed identical random super-block bytes; numeric agreement is reported as a
**relative norm** `max |ours-ggml|/|ggml|` over several seeds×sizes (NOT byte-exact:
ours is pinned byte-exact to ggml `_generic`, whose fp32 fold order differs from
`_vl256`; the integer core is identical, only fp32 rounding order diverges — this is
expected and is exactly why the task asks for norm agreement, not byte-exact).

## Win-B·micro results

| quant | ggml variant | ours ns/call | ggml ns/call | ratio (ggml/ours) | rel-norm agreement | verdict |
|-------|--------------|-------------:|-------------:|------------------:|-------------------:|---------|
| q4_K  | `_vl256`     |      11931.6 |       6955.9 |             0.583 |          5.823e-07 | LOSS (ggml 1.72x faster) |
| q6_K  | `_vl256`     |      14228.4 |       6310.1 |             0.443 |          2.765e-06 | LOSS (ggml 2.26x faster) |
| q5_K  | inline RVV (no _vl256; LMUL=8) | 13991.5 | 13962.9 |   0.998 |          7.317e-07 | TIE (within 0.2%) |
| q2_K  | `_vl256`     |      10051.8 |      10214.1 |             1.016 |          7.494e-07 | WIN (ours 1.6% faster) |
| q3_K  | `_vl256`     |      14686.6 |       6886.1 |             0.469 |          1.214e-06 | LOSS (ggml 2.13x faster) |

ratio > 1 ⇒ OUR kernel is FASTER (Win-B win); ratio < 1 ⇒ OUR kernel is SLOWER (a loss, reported honestly).

## Remaining / status
- q4_K: DONE — agreement 5.8e-07 (finite, nonzero), ours 11931.6 ns / ggml _vl256 6955.9 ns, ratio 0.583 (LOSS). timing_n=8192, iters=2000, reps=200, taskset -c 2.
- q6_K: DONE — agreement 2.8e-06 (finite, nonzero), ours 14228.4 ns / ggml _vl256 6310.1 ns, ratio 0.443 (LOSS). Same harness params.
- q5_K: DONE — agreement 7.3e-07, ours 13991.5 ns / ggml inline-RVV 13962.9 ns, ratio 0.998 (TIE). ggml q5_K RVV path is inline under `#if defined __riscv_v` (quants.c:2081, NO _vl256 variant) and uses LMUL=8 redsums — not faster than ours on X60.
- q2_K: DONE — agreement 7.5e-07, ours 10051.8 ns / ggml _vl256 10214.1 ns, ratio 1.016 (marginal WIN, ours ~1.6% faster).
- q3_K: DONE — agreement 1.2e-06, ours 14686.6 ns / ggml _vl256 6886.1 ns, ratio 0.469 (LOSS).

## Summary (all 5 K-quant block-dots done)
- ALL 5 emit-only K-quant block-dots now have a measured Win-B·micro cell.
- Numeric agreement holds for every kernel (rel-norm ~5e-7..3e-6, all finite & nonzero) —
  ours is functionally a valid drop-in vs ggml's RVV; the residual is fp32-fold-order
  rounding (ours pinned byte-exact to ggml _generic, baseline is _vl256/inline-RVV).
- Speed verdict is honest and MIXED:
  - WIN: q2_K (1.016x — ours marginally faster).
  - TIE: q5_K (0.998x — even; ggml's RVV path uses LMUL=8 redsums, slow on X60).
  - LOSS: q4_K (0.583x), q6_K (0.443x), q3_K (0.469x) — ggml's _vl256 is 1.7-2.3x faster.
- Takeaway: OUR emit is the _generic fp-order ported to RVV with a single LMUL shape;
  ggml's _vl256 kernels are hand-tuned (e.g. q4_K splits hi/low nibble into separate
  m1 redsum streams, q6_K/q3_K use m2 widening + lane-extract scale folds). The K-quant
  emit is correct but not yet LMUL/shape-tuned for X60 — exactly the kind of gap N3
  (Gearbox capability/resource-aware tune) targets. q2_K's gather-heavy ggml path is the
  one case where OUR straightforward emit already edges ahead.

## Harness provenance / reproduce
- Bench sources: `artifacts/kernel-coverage/kq-bench/<quant>/{ours.cpp,ggml_ref.cpp,harness.cpp}`.
- ours.cpp = `build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-<q>-k-q8-k-block-dot.mlir
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`.
- ggml_ref.cpp = the `_vl256` (or inline-RVV for q5_K) body lifted verbatim from
  `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c` + a 4-arg
  ABI adapter `kern_ggml`. Both TUs compiled together on K1:
  `clang-18 -O3 -march=rv64gcv_zvl256b -ffp-contract=fast harness.cpp ours.cpp ggml_ref.cpp`,
  run `taskset -c 2 ./bench`. timing_n=8192 (=32 super-blocks/call), iters=2000, reps=200
  (best-of-min). Agreement over 8 seeds x 5 sizes (256..4096).
- The two marginal cells were variance-confirmed (3x re-run each, taskset -c 2):
  q2_K ratio = 1.017 / 1.016 / 1.016 (always >1.0 ⇒ marginal WIN is real, not noise);
  q5_K ratio = 0.998 / 0.998 / 0.998 (stable TIE). best-of-200-min gives <0.1% spread.
- Translator note: the task named `tcrv-translate`, but our `build/bin/tcrv-translate`
  does not accept `--mlir-to-cpp`; the validated template generator (`tune_block_dot.py`)
  itself uses `mlir-translate-20 --mlir-to-cpp`, so we used `/usr/bin/mlir-translate-20`
  for consistency with the reused harness. Emitted C correctness is independently
  cross-validated by the ~1e-6 numeric agreement (offsets baked from MLIR attrs in ours
  vs offsets from struct layout in ggml_ref — two independent derivations agree).
