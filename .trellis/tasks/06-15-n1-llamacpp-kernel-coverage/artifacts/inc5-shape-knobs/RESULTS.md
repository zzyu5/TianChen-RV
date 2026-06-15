# INC-5 — Q4_0×Q8_0 shape knobs (multi_block_factor × strip_elision)

STEP 1 of compiler-driven tune: add the bounded SHAPE KNOBS the autotuner (a later
step) will select, set them via EXPLICIT attrs, validate each byte-exact, and
RIGOROUSLY measure the COMPILER's structured emission vs ggml. (The autotuner pass
is NOT built here — that is the next step.)

## (1) Attrs + emission added

Two bounded OptionalAttrs on `tcrv_rvv.q4_0_q8_0_block_dot` (verifier fail-closed,
I7); both are the *how* not the *what* (byte-exact for every legal combination:
the per-block fp32 fold stays in strict ascending block order, integer add is
order-independent):

- `multi_block_factor`: 1 (default) | 2 | 4 — q4_0 blocks the outer loop processes
  per iteration. factor>1 emits the main loop over `nb - nb%factor` groups +
  a `nb%factor` ROBUST single-block scalar tail.
- `strip_elision`: "robust" (default) | "elided" — inner half-block strip loop kept
  (VLEN<128-safe, sumi-carry) or elided (single vsetvl_e8m1(16) + one vwredsum per
  half-block, correct ONLY at VLEN>=128). Verifier rejects "elided" unless
  integer_core_lmul=="m1" (mf4's vsetvl_e32m1 VLMAX=4 would drop 12/16 bytes).

Emission (extend `emitQ4_0Q8_0BlockDot`, STRUCTURED — zero raw(); reuse INC-1's
`emitOffsetBinaryDecodeProductValue`): for factor>1 the `factor` integer cores are
emitted FIRST (the `factor` reductions adjacent = the latency-overlap lever), THEN
the `factor` folds in strict ascending block order. The elided core seeds vwredsum
with 0 (no carry). The `nb%factor` tail always uses the robust core.

Files:
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — the two OptionalAttrs + doc.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — allowlist + bounds (factor∈{1,2,4},
  elision∈{robust,elided}, elided⇒m1) fail-closed.
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `emitQ4_0Q8_0BlockDot` refactored into
  emitBlockCore (addr+scales+integer core: robust strip loop OR elided single-strip)
  + emitFold (the unchanged single-emitc.expression FMA accumulate) + factor-driven
  loop construction. factor=1/robust emits byte-IDENTICAL to the committed 140946cc
  / inc4 m1 (and mf4) emission.
- `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot-mb2-robust.mlir` (new lit)
- `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot-mb4-elided.mlir` (new lit)
- `test/Dialect/RVV/q4-0-q8-0-block-dot-dataflow.mlir` (+5 verifier cases: accept
  mb2-robust, accept mb4-elided, reject factor=3, reject elision="dropped", reject
  elided+mf4).

`grep -c 'raw(' lib/Conversion/RVV/RVVToEmitC.cpp` = **0** (structured emitc only, I5).
Every emitted value is an emitc node (for/call_opaque/add/mul/sub/div/rem/cast/
expression/variable/assign/load/literal/subscript); all emitc.verbatim ops are
route-source COMMENTS only.

## (2) Byte-exact — PASS for all 3 shapes (ssh rvv)

Committed recipe `-O2 -march=rv64gcv -mabi=lp64d`, `-ffp-contract` ∈ {off,on,fast}:
`*s` BITWISE-EQUAL (memcmp, not tolerance) to ggml's REAL RVV kernel AND `_generic`.
3300 checks/mode over n ∈ {32,64,96,128,160,192,224,256,1024,4096,8192} × 100 reps ×
3 shapes — **0 failures**, incl. the main+tail seams (n=96 nb3 → mb2 tail 1; n=160
nb5 / 192 nb6 / 224 nb7 → mb4 elided-main + robust-tail mixed). Negative control
non-vacuous (1-bit scale flip changes *s; bitwise check reports MISMATCH).
mb1-robust C is byte-IDENTICAL (diff) to the committed 140946cc emission.

The FULL autotuner grid {1,2,4}×{robust,elided} is byte-exact validated (not just
the 3 shipped shapes), so the next-step autotuner can select any combo safely
(`byte_exact_stdout.txt` for the 3 shipped; `grid_byte_exact_stdout.txt` for the
other 3 combos — each 3300/0 across off/on/fast):

| combo (factor × elision, m1 core) | byte-exact vs ggml real + _generic (off/on/fast) |
|---|---|
| mb1-robust (1, robust) — SHIPPED, lit | PASS (byte-identical to committed 140946cc) |
| mb2-robust (2, robust) — SHIPPED, lit | PASS |
| mb4-elided (4, elided) — SHIPPED, lit | PASS (at VLEN=128, the board) |
| f1-elided (1, elided) | PASS (grid validation) |
| f2-elided (2, elided) | PASS (grid validation) |
| f4-robust (4, robust) | PASS (grid validation) |

Honest recipe note (PROVEN by isolating builds, not inferred): under `-O3
-march=rv64gcv_zfh_zvfh` the harness shows fast-mode failures at large nb. This is an
`-O3` fp-reassociation artifact, NOT a kernel defect, and it is recipe-scoped:
- The new code is provably not the cause: mb1-robust's emitted C is byte-IDENTICAL
  to the committed 140946cc kernel, which fails the same `-O3 -ffp-contract=fast`
  gate identically.
- Isolating builds pin it to `-O3` itself reordering fp accumulation on WHICHEVER TU
  is at `-O3` (both the emitted kernel and the `static` ggml references get
  autovectorized/reassociated at `-O3`, and differently than at `-O2`):
  kernel.o@-O2 + validate-TU@-O3-fast → 206 fails (the -O3 references diverge);
  kernel.o@-O3-fast + validate-TU@-O2 → 206 fails (the -O3 kernel diverges).
  Under the committed `-O2 -march=rv64gcv` recipe BOTH sides keep the in-statement
  FMA contraction identical and everything matches (3300/0 across off/on/fast).
Byte-exactness is a property of the EMITTED kernel under the committed `-O2` recipe;
it holds across all `-ffp-contract` modes there. The `-O3` recipe is for TIMING only
(not a byte-exact gate); its single-array correctness gate is a tripwire, not the
byte-exact proof — the authoritative byte-exact evidence is the `-O2` 3300/0 harness
(`byte_exact_stdout.txt`).

## (3) RIGOROUS microbench — the COMPILER's emission, NOT a hand probe (ssh rvv)

`-march=rv64gcv_zfh_zvfh -mabi=lp64d -O3 -ffp-contract=fast`, taskset -c 3 pinned,
best-of-40 min ns/call, warmup discarded, identical data, separate-TU kernels,
correctness-gated. n=4096 (3 stable runs) + n=8192 cross-check:

| shape | ns/call (n=4096) | vs ggml | ns/call (n=8192) | vs ggml | vregs | shippable |
|---|---|---|---|---|---|---|
| **ggml** (real RVV, serial) | 1169 | 1.00x | 2317 | 1.00x | 6 | reference |
| mb1-robust (current anchor) | 1388 | 1.19x | 2757 | 1.19x | 5 | yes |
| **mb2-robust** | **1306** | **1.12x** | 2596 | 1.12x | 5 | yes (VLEN-robust) |
| **mb4-elided** | **1021** | **0.874x** | 2026 | 0.874x | 6 | VLEN>=128 only (Zvl128b) |

HONEST VERDICT on whether mb4-elided beats ggml on OUR emission: **YES** — the
compiler's structured mb4-elided emission is **0.874x ggml (~13% FASTER)**,
reproduced exactly at n=8192. The robust optimum mb2-robust is **1.12x (~6% faster
than the current mb1 anchor**, 1388→1306) but does NOT beat ggml — consistent with
the research's "no VLEN-robust shape beats ggml; the beat needs the strip-elided
form". (The research's mb2 ~1.20x→1.12x and ~11%→~6%-over-mb1 numbers came from a
hand probe with a different core/fold layout; the compiler's faithful
cores-first-then-folds emission gives the honest ~6%.) All vregs (5/5/6) are far
under the 32 budget — vreg pressure is NOT the binding prune constraint.

This is the N1 legality-divergence + N3 lever on a real llama.cpp kernel: one
capability fact (Zvl128b ⇐ rv64gcv) → mb4-elided (beats ggml ~13%) is legal for
full-V; the robust strip loop (mb1/mb2) serves zve32x/zve64x where VLEN may be <128.
The autotuner (next step) will enumerate {1,2,4}×{robust,elided}, prune elided on
Zvl128b + the vreg budget, rank by a principled cost, select, and stamp.

## (4) raw()=0 + structured proof — see (1).

## (5) lit: 608 discovered / 605 passed / 3 failed (`ninja check-tianchenrv`).
The 3 reds are the pre-existing, documented strided-`vlse16` computed-mask
dot-reduce + self-test (0 q4_0 refs; unrelated to this change). The 2 new q4_0
Conversion tests PASS; the extended dataflow verifier test PASS; existing q4_0
mf4/m1 + INC-1 + deferred-wide tests byte-identical/green.

## (6) Deviations
- `-ffp-contract=default` is unsupported by the board's clang 18.1.3 (only off/on/
  fast); validated the 3 supported modes (the committed inc4 recipe did the same).
- Byte-exact gate uses the committed `-O2 -march=rv64gcv` recipe (the only recipe
  under which fp-non-associativity is apples-to-apples between kernel and the static
  references); the microbench uses `-O3 -march=rv64gcv_zfh_zvfh` (timing, not a
  byte-exact gate). Documented in (2).
- mb4-elided emits `ptr + 0` for the chunk offset (a harmless literal-0 add the
  compiler folds; keeps the elided/robust cores sharing one strip-reduce builder).

## Regenerate
```
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot-mb2-robust.mlir \
  --tcrv-rvv-lower-to-emitc | /usr/bin/mlir-translate-20 --mlir-to-cpp > tcrv_emitted_mb2_robust.cpp
# (likewise -mb4-elided.mlir and -m1.mlir for mb1-robust)
```
Board: scp the emitted C + harnesses to ~/inc5_shapeknobs, build per the recipes above.
