# INC-4 — Q4_0 integer core re-anchored at LMUL m1: N3 perf on a real llama.cpp kernel

## What changed (the m1 generalization)

The ggml `ggml_vec_dot_q4_0_q8_0` block-dot kernel our compiler emits
(`tcrv_rvv.q4_0_q8_0_block_dot`) previously anchored its per-block integer core at
**i8mf4** (4 lanes/strip at VLEN=128 -> a 4-iteration inner strip loop -> **4
`vwredsum` reductions per block**). ggml's hand-written kernel anchors at **i8m1**
(16 lanes = the whole half-block in **ONE** strip -> **ONE `vwredsum` per block**).
`vwredsum` (a reduction) is the expensive op; we were doing 4x as many.

This increment generalizes the integer core off the hard mf4 pin to an LMUL
parameter and adds an **m1** anchor that matches ggml:

- Per half-block, load 16 weight bytes as ONE `vint8m1_t` (`vsetvl_e8m1(16)=16` at
  VLEN=128), `vxor.vx 0x88`, `vsll/vsra` unpack to i8m1, `vwmul_vv_i16m2` against
  the plain q8-low i8m1, `vwmacc_vv_i16m2` against q8-high i8m1, then **ONE**
  `vwredsum_vs_i16m2_i32m1` -> scalar. NO inner 4-chunk loop at VLEN=128.
- The outer block loop, the per-block dual fp16 scale, and the single
  `emitc.expression` fp32 accumulate are **UNCHANGED**.
- **VLEN-robust**: the strip loop is retained (loop step = `vsetvl_e8m1` VLMAX);
  at VLEN=128 it runs once, a VLEN<128 board re-strips correctly via the
  `sumi`-carrying seed. (This is *more* robust than ggml, which assumes VLEN>=128.)

### Files (all additive; the committed mf4 path is byte-unchanged)
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `OptionalAttr<StrAttr>:$integer_core_lmul`
  on `GgmlBlockDotQ40Q80Op` (a bounded resource/scheduling fact: the *how*, not the
  *what*; mf4 default / m1 anchor). Doc updated.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — verifier allows + bounds
  `integer_core_lmul` to "mf4"|"m1" (fail-closed I7; "m2" etc. rejected).
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `emitQ4_0Q8_0BlockDot` parameterized by the
  chosen LMUL (i8 source mf4/m1 -> i16 product mf2/m2, the `vsetvl`/`vle8`/`vwredsum`
  spellings track it). Absent attr / "mf4" emits **byte-identical** to INC-2a. The
  shared `emitOffsetBinaryDecodeProductValue` (INC-1) was already LMUL-parameterized
  and is UNTOUCHED in body, so INC-1's mf4 op stays byte-identical.
- `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot-m1.mlir` — new m1 lit test.
- `test/Dialect/RVV/q4-0-q8-0-block-dot-dataflow.mlir` — +2 cases (accept m1,
  reject m2 fail-closed).

`grep -c 'raw(' lib/Conversion/RVV/RVVToEmitC.cpp` = **0** (structured emitc only, I5).

## (2) Byte-exactness RE-CONFIRMED (yes) — ssh rvv, the gate before any perf claim

`tcrv_emitted_kernel_m1.cpp` (the UNMODIFIED compiler output) computes `*s`
**BITWISE-EQUAL** (memcmp, not tolerance) to ggml's REAL RVV kernel AND `_generic`:
- validate harness: **1000 arrays × {n=32,64,256,1024,4096}, 0 failures**, under ALL
  4 `-ffp-contract` modes (default/on/off/fast). Negative control discriminates.
- stress harness: **5859 adversarial cases, 0 failures** (5609 non-trivial
  discriminating), under `=on` and `=fast`.
- A regenerated **mf4** kernel is byte-identical to the committed INC-2a kernel
  (`diff` clean) — the additive constraint holds.

Raw board stdout: `ssh_rvv_byte_exact_stdout.txt`. LMUL-independence is why it holds:
`vwredsum` sums the same integer set, integer add is order-independent, the fp32
accumulate is unchanged.

## (3) Microbenchmark (ssh rvv, -O3 -march=rv64gcv_zfh_zvfh -mabi=lp64d -ffp-contract=fast, best-of-20)

`inc4_microbench.cpp`: one `ggml_vec_dot_q4_0_q8_0` call over n blocks, tight loop,
correctness-gated (all 3 bitwise-equal on the bench data before timing). Steady-state
numbers (ggml's transcribed reference has a warmup/scheduler outlier on the first run
— 1709–1961 ns — then settles to ~1150 ns on runs 2–5; OUR m1 is rock-stable at
~1386 ns every run, see the 5-run trace in `microbench_stdout.txt`):

| kernel | n=4096 ns/call (steady) | n=8192 ns/call | vs ggml |
|---|---|---|---|
| ggml REAL RVV (i8m1, hand-written) | ~1150 | 2289 | 1.00x |
| ours OLD mf4 | 5988 | 11955 | **4.86x–5.22x SLOWER** |
| ours NEW m1 | 1386 (stable) | 2756 | **~1.20x** |

**m1 is 4.32x faster than our OLD mf4** (perfectly stable across all runs), and
within ~20% of ggml's hand-written kernel at steady state — under scheduler noise the
two cross (m1 occasionally measures FASTER than a noisy ggml run, e.g. 0.71–0.81x),
i.e. genuine near-parity, not a strict beat. Raw + 5-run variance trace:
`microbench_stdout.txt`.

## (4) Live llama.cpp (ssh rvv, llama-2-7b-chat.Q4_0.gguf), still token-identical (yes)

Same greedy prompt, m1 override built by swapping the kernel C into the INC-3
override tree (same flags). 3 trials each, stable:

| config | generation t/s | % of stock | token output |
|---|---|---|---|
| STOCK (ggml hand-written) | 2.7 | 100% | "...France is Paris." |
| OVERRIDE-MF4 (our OLD) | 1.6 | ~59% | "...France is Paris." (identical) |
| OVERRIDE-M1 (our NEW) | 2.6 | ~96% | "...France is Paris." (identical) |

m1 delegation counter = **113341440** (our kernel runs the Q4_0 hot path).
Raw: `live_llamacpp_stdout.txt`.

## (5) Honest verdict

The m1 re-anchor takes our compiler-emitted Q4_0 kernel from **4.86x slower than
ggml (mf4)** to **~1.20x at steady state** in the microbench, and live inference from
**1.6 t/s (~59% of stock)** to **2.6 t/s (~96% of stock)** — **from 1.7x-slower to
near-parity with ggml's hand-written RVV kernel**, byte-exact (token-identical,
0-failure validate, identical perplexity expected). That is the N3 lamp on a REAL
llama.cpp kernel: the
resource-aware LMUL choice (the Gearbox principle), applied to a real kernel, measured
fair on `ssh rvv`, structured emitc only (no raw()).

We do **not yet strictly BEAT** ggml — we MATCH it within ~13–20% (micro) / ~4% (live).
At m1 our kernel is structurally identical to ggml (same intrinsics, one vwredsum/block);
the residual is the VLEN-robust per-block `vsetvl`+branch that ggml's straight-line code
lacks (clang won't elide a runtime-trip-count loop). **Next lever** (optional, VLEN-safe):
multi-block-per-iteration to amortize that branch and overlap the two independent vwredsum
latency chains. A straight-line VLEN-128 specialization would close it too but VIOLATES
the VLEN-robustness constraint, so it is off the table for this increment.

## (6) Build / lit

- Full clean lit suite: **606 discovered, 603 passed, 3 failed** — exactly the 3
  documented environmental reds (`rvv-generated-bundle-abi-e2e-self-test` + the two
  `…computed-masked-strided-input-widening-dot-reduce-add-dry-run` script tests; a
  `vlse16` strided-load pattern in an UNRELATED op). Confirmed pre-existing: with my
  source stashed the same 3 reds fail (plus my 2 new tests, which correctly fail on
  the feature-less baseline binary). Count moved 605→606 / 602→603 by my 1 new lit file.
- Both new q4_0 lit tests PASS; the committed mf4 conversion + dataflow tests PASS
  byte-identical.

## Regenerate
```
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot-m1.mlir \
  --tcrv-rvv-lower-to-emitc | /usr/bin/mlir-translate-20 --mlir-to-cpp \
  > tcrv_emitted_kernel_m1.cpp
```
