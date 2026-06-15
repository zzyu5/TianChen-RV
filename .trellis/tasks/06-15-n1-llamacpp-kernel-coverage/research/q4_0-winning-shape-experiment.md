# Research: Q4_0×Q8_0 kernel-shape design space vs ggml on the board (empirical)

- **Query**: STRAND 2 — find a Q4_0×Q8_0 block-dot shape that beats ggml's `ggml_vec_dot_q4_0_q8_0` on `ssh rvv`, byte-exact, expressible as a structured (emitc.for + our ops) lowering. Map the design space (LMUL / multi-block / unroll) so the autotuner knows what to target. **Be honest if parity is the ceiling.**
- **Scope**: empirical (microbenchmark on `ssh rvv`) + internal (the emitted kernel structure we must reproduce)
- **Date**: 2026-06-15
- **Board**: `ssh rvv` — 64-core riscv64, **VLEN=128 (VLENB=16)**, rv64imafdcv+zvfh, clang 18.1.3. `e8m1`=16 lanes, `e32m1`=4 lanes.
- **Probe source**: `artifacts/inc5-tune-research/q4_0_shape_probe.c` (throwaway probes, NOT the shipped kernel). Raw numbers: `microbench_4clean_runs_n4096.txt`, `microbench_n4096_n8192.txt`, `microbench_robust_variants_n4096_n8192.txt`.

> **Headline (honest, after measuring the genuinely-VLEN-robust forms):**
> **No VLEN-robust shape beats ggml — the robust ceiling is a ~20% loss.** BUT the
> resource-best *robust* shape is **`mb2_robust` (2 blocks/iter, strip-loop + sumi-carry
> kept) = 1398 ns = 1.20x ggml**, which is **~11% FASTER than our current hand-set
> single-block m1 (1564 ns, 1.34x)**. So robust multi-block DOES help — but only up to **2
> blocks** (the robust sweet spot); 4 blocks regresses (mb4_robust 1.30x) and the
> deferred-i32-accumulator form is worse (mb4_defer 1.61x). A *clear beat* of ggml exists
> only for the **non-VLEN-robust** shapes (mb4 0.86x), which silently break at VLEN<128 and
> are forbidden by the ship constraint. **Net: the autotuner can discover a robust shape
> (`m1`, multi_block=2) ~10% better than the manual 140946cc anchor — the real N3 lamp —
> even though ggml stays ~20% ahead of any robust shape.**

---

## Methodology (fair, gated, pinned)

- One `ggml_vec_dot_q4_0_q8_0` call over `n` elements (`nb = n/32` blocks), tight loop,
  best-of-N **min** ns/call (min rejects scheduler noise), `taskset -c 3` pinned on the
  64-core board, `-march=rv64gcv_zfh_zvfh -mabi=lp64d -O3 -ffp-contract=fast`, identical
  random data, n=4096 (n=8192 cross-check), iters=100000, reps=40, several runs (ggml
  baseline settles at ~1168 ns; one warmup-outlier first run per session is discarded).
- **Correctness gate before timing**: every variant's `*s` must be **bitwise** (`memcmp` of
  the fp32 bits) equal to ggml on the bench data, or it is not a candidate. **All variants
  passed the gate** (ref = -5.22822143e+12) — including the fast-but-not-robust ones, because
  the gate runs at VLEN=128 where they happen to be correct (see the robustness caveat — the
  gate CANNOT catch the VLEN<128 defect; robustness is a code-structure property).
- ggml reference = the real hand-written RVV kernel transcribed from
  `llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:222` (i8m1, one half-block/strip, ONE
  `vwredsum`/block, serial).

### The two constraints that bound the design space
1. **Byte-exactness (fp non-associativity).** The per-block fold
   `sumf = sumf + ((float)sumi*d_x)*d_y` MUST run in strict ascending block order. Integer
   work may be reordered/batched freely (integer add is associative → same `sumi`), but
   "vectorize the scale step" is excluded (changes rounding).
2. **VLEN-robustness (HARD ship constraint).** The emitted kernel must compute correctly on a
   VLEN<128 board, by re-stripping. The mechanism is the **inner strip loop with the
   `sumi`-carry seed** (`vmv_v_x_i32m1(carried_sumi)` → `vwredsum`), as in the current emitted
   m1 kernel (`tcrv_emitted_kernel_m1.cpp:30–69`). Merely placing `vsetvl` per block (without
   the strip loop) does NOT provide robustness — at VLEN<128 such a kernel processes only the
   first `vl` lanes of each block's 16 nibble bytes and silently drops the rest.

---

## Results (n=4096, reps=40, settled ggml baseline ~1168 ns; reproduced across runs and at n=8192)

| shape | ns/call | vs ggml | distinct vregs | VLEN-robust? | shippable? |
|---|---|---|---|---|---|
| **ggml** (i8m1, serial) | **1168** | 1.00x | 6 | yes | reference |
| m1 (our current emit, 1 block, strip loop) | 1564 | 1.34x | 5 | **yes** | yes (current anchor) |
| **mb2_robust** (2 blk, strip loop + sumi-carry) | **1398** | **1.20x** | **5** | **yes** | **yes (robust optimum)** |
| mb4_robust (4 blk, strip loop + sumi-carry) | 1525 | 1.30x | 5 | **yes** | yes (regresses vs mb2) |
| mb4_defer (4 blk, i32 vector acc, 1 reduce after loop) | 1879 | 1.61x | 10 | **yes** | yes (worse) |
| m1_nl (m1, strip loop straight-lined) | 1205–1225 | 1.03–1.05x | 6 | **NO** | no |
| mb2 / mb2_vl (2 blk, NO strip loop) | 1093–1096 | 0.94x | 7 | **NO** | no |
| **mb4 / mb4_vl (4 blk, batched reduce, NO strip loop)** | **1002–1024** | **0.86–0.87x** | 11 | **NO** | no |
| mb8 (8 blk, NO strip loop) | 1009 | 0.86x | 20 | **NO** | no |

n=8192 reproduces every ratio (ggml 2341, m1 3164 = 1.35x; the robust forms scale identically).
mb2_robust is rock-stable across all three runs: 1397.8 / 1398.0 / 1400.9 ns.

> **The two rankings that matter:**
> - **Among VLEN-robust (shippable) shapes:** `mb2_robust (1.20x) < mb4_robust (1.30x) <
>   m1 (1.34x) < mb4_defer (1.61x)`. **mb2 is the robust optimum, ~11% faster than the
>   single-block m1 anchor we ship today.** None beats ggml.
> - **Non-robust shapes** (drop the strip loop) hit `mb4 0.86x` — a ~13% beat of ggml — but
>   are NOT shippable (break at VLEN<128, same defect as a straight-line VLEN-128 kernel).

### Why robust 2-block helps but 4-block / defer don't (the mechanism)
- **The strip loop costs directly AND serializes the per-block reduce** (the reduce lives
  *inside* the strip loop, carrying `sumi`). At VLEN=128 each strip loop is one iteration, and
  with **2** blocks/iter the compiler can overlap the two blocks' independent integer cores +
  reductions enough to recover ~11% (1564→1398). That is the multi-block latency-overlap lever
  working *within* the robustness constraint, at modest unroll.
- **At 4 blocks the gain reverses** (mb4_robust 1525 > mb2 1398): four single-trip strip loops
  add address-arithmetic and i-cache/scheduling pressure that outweighs further overlap — the
  reductions are still loop-bound so they don't pipeline as cleanly as the non-robust mb4's
  loop-free reductions do. So the robust sweet spot is **2**, not 4.
- **The deferred-i32-accumulator form we tried (mb4_defer) did not help** (1879, 1.61x): the
  i32m4 element-wise accumulate + the wide m4 `vredsum` cost more than freeing the reduce
  saved. (Caveat: this particular probe inlined the reduce inside the per-block helper, so the
  four reduces were not perfectly adjacent — it is not a clean disproof of "free the reduce
  then overlap" in general, only evidence that *this* deferred form regresses on this board.)
- **Precise statement (for STRAND 1):** *robust multi-block helps up to 2 blocks (1.34→1.20x)
  because the strip loop is single-trip at VLEN=128 and two independent blocks overlap; beyond
  2 it regresses. The fewest-reductions LMUL choice (m1) and a 2-block unroll are the two
  robust levers; both are autotuner-selectable. Beating ggml needs the loop-free non-robust
  form, which the ship constraint forbids.*

### Shape (c) "wider product LMUL (i16m4/i16m8) within a block" — STRUCTURALLY BLOCKED
A q4_0 block is `[2 fp16 scale bytes][16 quant bytes]` (18-byte AoS stride). At VLEN=128
`i8m1` = 16 bytes = exactly one block's quant region; a unit-stride load wider than m1 reads
into the next block's 2 scale bytes → corrupt product. Wider-LMUL-within-block needs
segmented/strided loads (`vlseg`/strided). Not pursued — multi-block already explores the same
latency-overlap goal with unit-stride loads. Report as a structural finding, not a measured loss.

---

## Expressibility (the robust optimum IS shippable and structured-emittable)

- **mb2_robust IS expressible** as `emitc.for` + our ops and IS VLEN-robust: it is the current
  m1 emission (`emitQ4_0Q8_0BlockDot`) with the outer `emitc.for` stepping by **2** blocks,
  the **two** per-block strip-loop integer cores (each keeping the `vsetvl_e8m1` strip + the
  `vmv_v_x_i32m1(sumi)`-carry `vwredsum`), and the two `sumf +=` folds in order, plus an
  `nb%2` scalar-tail block. No hand asm; the inner strip loops stay (VLEN-robust); 5 vregs.
- The ggml-beating shapes (mb4, 0.86x) are also structured-emittable but **drop the strip
  loop**, so they violate the ship constraint exactly as a straight-line VLEN-128 kernel does.
  Not a shippable target.
- Net: the **shippable** ceiling is **mb2_robust ≈ 1.20x ggml**, ~11% better than the current
  single-block m1; ggml remains ~20% ahead of any robust shape.

---

## vreg pressure (for STRAND 1's prune budget)

Distinct architectural vector registers live (from `-S` disassembly; m2 groups counted full):
robust shapes are very light — m1 5, **mb2_robust 5**, mb4_robust 5, mb4_defer 10 (the i32m4
accumulator). Non-robust: mb2 7, mb4 11, mb8 20. All ≤ 32. So the **vreg budget is NOT the
binding constraint** on this kernel — VLEN-robustness and reduction serialization are. The
robust optimum (mb2_robust) costs only 5 vregs; the prune budget has ample headroom.

---

## Honest verdict (answers to the three required questions)

1. **Is there a winning shape, and by how much?** Against **ggml**: only the **non-robust**
   mb4 wins (0.86x, ~13%), and it is **not shippable** (breaks at VLEN<128). Among
   **shippable (VLEN-robust)** shapes **none beats ggml** — but the resource-best robust shape
   **mb2_robust (1.20x) beats our current hand-set single-block m1 (1.34x) by ~11%**.
2. **Which knobs matter?** For a robust kernel: **`integer_core_lmul` (mf4→m1, fewest
   reductions)** and a **2-block unroll** (multi_block=2, the latency-overlap sweet spot).
   multi_block=4, mb4_defer, and unroll>2 regress. For a non-robust kernel multi-block scales
   further (to mb4) but that path is off the table.
3. **Can a compiler-emittable shape beat ggml?** Honestly: **not within the VLEN-robustness
   ship constraint** — ggml's straight-line, no-loop, serial-reduce code stays ~20% ahead of
   any robust shape. ggml is beatable only by adopting ggml's own non-portable VLEN≥128
   assumption (the non-robust mb4). **But the compiler CAN discover a robust shape ~10% better
   than its current manual anchor** — that is the achievable N3 result here.

## Caveats / Not found

- **Robustness is a code-structure property the gate cannot verify.** The board is VLEN=128,
  so it never re-strips; the `_vl`/`_nl`/non-robust mb* shapes pass the bitwise gate yet are
  wrong at VLEN<128. Robust/non-robust labels come from READING the loop (does it keep the
  strip loop + carry?), not from the gate. The probe's `mb2_vl`/`mb4_vl` labels say
  "VLEN-safe" but are **MISLABELED** — they drop the strip loop; treat them as non-robust.
- An earlier draft wrongly claimed mb4_vl was a shippable ~13% win (the
  `vsetvl`-placement-vs-strip-loop confusion); a second draft then over-corrected to "all
  robust multi-block is worse than m1". Both are fixed here from the raw data
  (`microbench_robust_variants_n4096_n8192.txt`): robust multi-block helps to 2 blocks
  (mb2_robust 1.20x < m1 1.34x), then regresses.
- mb4_defer's probe inlined the reduce inside the per-block helper, so its four reduces were
  not perfectly adjacent — it shows *this* deferred form regresses, not a general impossibility.
- Not re-measured end-to-end in live llama.cpp. The current m1 anchor is ~96% of stock live
  (inc4 `live_llamacpp_stdout.txt`); a robust mb2 (~11% faster kernel) would close some of that
  remaining ~4% — recommend an INC-5 live run once mb2_robust is emitted.
- Single board (VLEN=128). Re-measure on a board with different strip-loop control cost or m4
  reduce latency.
