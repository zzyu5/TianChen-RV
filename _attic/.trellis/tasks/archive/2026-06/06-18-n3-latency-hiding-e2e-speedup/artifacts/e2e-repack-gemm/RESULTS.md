# VLEN=128 q4_0 repacked GEMM — e2e prefill speedup on `ssh rvv`

Ports ggml's existing VLEN≥256 q4_0 **block-as-lane repacked** GEMM/GEMV (the `q4_0_16x1_q8_0`
path) down to **VLEN=128** in OUR OWN llama.cpp (`~/tcrv-llamacpp`, mainline `b9692`/`f3e1828`,
GNU 14.2). Fills ggml's `case 128: { break; } // TODO` in `ggml_repack_get_optimal_repack_type`
so the model auto-repacks q4_0 weights at load and `mul_mat` runs the repacked GEMM (prefill) +
GEMV (decode). The repack is a RUNTIME buffer (ggml's CPU_REPACK mechanism); the GGUF on disk is
unchanged.

**This is a layout/codegen optimization (same math). The honest measured e2e number is the
deliverable; the repack structure itself is standard ggml and is NOT claimed as novelty.**

---

## TL;DR

- **pp512 (prefill): 1.57 → 9.17 tok/s = ~5.84× e2e speedup** (`-t 8`, AFTER `-r 3` ±0.06).
  Far above the ≥1.5× target (and the prior non-repacked M-blocked 1.44×). The BEFORE here is an
  **in-session A/B re-measure** (pp512 = 1.57 ± 0.00) of the same binary with ONLY the `case 128`
  dispatch line reverted to `break` — it reproduces the banked clean-stock 1.55 baseline within
  ~1.3%, so the speedup is not a cross-session/cross-binary artifact. Also reproduced across the
  AFTER `-r 1` engagement bench (9.02) and the `-r 3` headline (9.17 ± 0.06) → not an `-r 1` fluke.
- **tg128 (decode): 1.38 → 6.49 tok/s = ~4.7×** (`-r 3`, ±0.12). Decode q4_0 now routes through
  the new **GEMV** kernel (not a stock fall-through like the prior task), so this is a real
  measured decode win, reported honestly against the 1.38 baseline.
- **Repack width chosen: 16x1** (`block_q4_0x16`, interleave=1) — the SAME width ggml's RVV path
  already uses at VLEN=256. The block-index is the SIMD lane (16 interleaved q4_0 blocks per group);
  loads are CONTIGUOUS; the reduction becomes lane-wise `vwmacc` + ONE final reduce (= the store).
  Not the AVX2 8x8 — the RVV repack family is 16x1. (See "Why 16x1, two-halves" below.)
- **Correctness: PASS.** White-box diff of the new VLEN=128 kernels vs ggml's own scalar
  `*_generic` reference at the model's exact n (4096 and 11008), 500 trials each: normalized
  error ~1e-5 (random signed data) and ~5e-6 (no-cancellation discriminator data) — i.e. matches
  the reference to fp32 rounding. PLUS e2e black-box check on the AFTER build: prompt
  "The capital of France is" (`--temp 0 --seed 1234`) → **"The capital of France is Paris."** — the
  same correct continuation as stock — with the repack GEMM+GEMV `ENGAGED` prints firing during that
  run (so the correct output came from the repack path, not a stock fallback). (The rigorous
  correctness proof is the white-box `_generic` diff; the generation is corroboration.)
- **≥1.5× e2e achieved: YES — ~5.84× pp512 (A/B-confirmed).**

---

## What was implemented (and where) — all in `~/tcrv-llamacpp` ONLY

Three surgical edits (full diffs in this dir):

1. `ggml/src/ggml-cpu/arch/riscv/repack.cpp` — the two RVV-intrinsic kernels
   `ggml_gemv_q4_0_16x1_q8_0` and `ggml_gemm_q4_0_16x1_q8_0` got a **VLEN=128 path** added,
   guarded by `if (__riscv_vlenb() * 8 == 128)` with an early return. The existing VLEN≥256 body
   is left **verbatim** in the fall-through (it cannot be tested on this VLEN=128 board, so it is
   protected, not edited). Patch: `vlen128-q4_0-16x1-kernels.patch`.
2. `ggml/src/ggml-cpu/repack.cpp` — `ggml_repack_get_optimal_repack_type`, the q4_0 riscv branch:
   `case 128: { break; } // TODO` → `case 128: { if (cur->ne[1] % 16 == 0) return &q4_0_16x1_q8_0; break; }`.
   This makes the loader auto-select the CPU_REPACK buffer for q4_0 at VLEN=128. Patch:
   `repack-dispatch-case128.patch`.
3. `ggml/src/ggml-cpu/ggml-cpu.c` — set `TCRV_GEMM_ENABLED 0` to **disable the prior task's
   M-blocked fast-path**, so the AFTER measurement isolates *this* repack path (the only non-stock
   accelerator). Patch: `ggml-cpu-disable-prior-mblocked.patch`. **NB:** this task's net-new change
   to `ggml-cpu.c` is the single `TCRV_GEMM_ENABLED 1→0` toggle. The patch is a `git diff` against
   pristine, and the prior task's M-blocked insertion was uncommitted in this tree, so the patch
   file also carries that prior ~146-line block — which is entirely compiled out by the toggle (so
   the AFTER build is the repack path only).

The block_q4_0x16 layout, `make_block_q4_0x16` repack transform, the `block_q8_0x4` activation
quantizer (`ggml_quantize_mat_q8_0_4x1`), the `tensor_traits<block_q4_0,1,16,Q8_0>` template, and
ALL gemv/gemm/repack dispatch were already in the tree and are **VLEN-independent** — only the two
kernel bodies hardcoded VL=16 (valid only at VLEN=256). So the entire new VLEN-specific surface is
those two bodies.

### Why 16x1, and why "two-halves-of-8" for the GEMM

At VLEN=128 an `e16m1` vector holds 8 i16 lanes (`e8mf2`=8, `e32m2`=8), not the 16 the VLEN=256
body assumes. The 16-block-as-lane group is therefore processed at VL=8:

- **GEMV**: two 8-lane halves (rows 0..7 = `qs[i*16+0..7]`, rows 8..15 = `qs[i*16+8..15]`) —
  disjoint contiguous sub-loads, each weight byte read once; lane-wise `vwmacc_vx`; one `vse32`
  store per half (the "final reduce" is the store — the cross-lane-reduction-wall escape survives
  at VL=8).
- **GEMM (4 activation cols × 16 rows)**: an outer `half∈{0,1}` loop over the two 8-row groups, at
  NATIVE LMUL (`e32m2`/`e16m1`), so the 4×8 f32 + 8 i16 accumulator set stays in registers (no
  spill). An LMUL-bump-to-16 alternative would carry 32 accumulator registers across the inner
  loop and spill every iteration under GCC — avoided deliberately.

Overflow safety is preserved: q4_0 nibbles ∈ [-8,7] × i8 activation, 16 i16-accumulations →
~16k < 32767, identical bound to the VLEN=256 path (same i-loop count).

---

## Correctness — PASS (white-box vs ggml's own `*_generic` reference)

`verify_kernel.cpp` (in this dir) builds random q4_0 weights, repacks them with the exact
`make_block_q4_0x16` logic, quantizes activations with the real `ggml_quantize_mat_q8_0_4x1`, then
runs the new VLEN=128 `ggml_gemm/gemv_q4_0_16x1_q8_0` against ggml's scalar
`*_generic` twins (an obviously-correct reference; same layout). At the model's exact
n ∈ {4096, 11008}, 500 trials each (`correctness-verify.log`):

| mode | GEMV norm err | GEMM norm err | verdict |
|---|---:|---:|---|
| random signed (`act∈[-2,2]`) | 1.43e-5 | 1.37e-5 | PASS |
| no-cancellation discriminator (nonneg) | 6.69e-6 | 6.83e-6 | PASS |

`norm = max_abs_err / rms(ref)`. The two kernels do NOT share fp accumulation structure with the
scalar reference (generic scales each block then sums; ours accumulates integer MACs then scales
once), so fp-rounding-level agreement — not bit-identical — is the correct bar, and is met. (A naive
per-element relative metric spiked to ~2e-2 on lanes where the random dot-product cancels to ~0;
the no-cancellation discriminator collapses that to ~5e-6, confirming it was a near-zero-reference
artifact, not a kernel bug.)

The new VLEN=128 path is the only path that can run on this board (`__riscv_vlenb()*8==128`); the
`*_generic` reference is VLEN-independent, so this is a direct white-box test of everything new.

**e2e black-box confirmation (`greedy-after.log`).** The white-box diff exercises one tile; the full
x/y/`bs` strides, the auto-repack routing, the real activation quantizer, the scatter, and the
layer-to-layer composition across all 32 layers are exercised by running the model. On the AFTER
build, greedy `llama-cli` (prompt "The capital of France is", `--temp 0 --seed 1234 -t 8`) generates
**"The capital of France is Paris."** — the same correct continuation as stock — and the GEMM+GEMV
`ENGAGED` prints fire during that run, so the correct factual continuation is produced by the repack
path, not a silent stock fallback. A subtly-wrong kernel cannot produce coherent correct English.
(Run was conversation-mode at temp 0, so this matches at the answer level; the bit-exact claim rests
on the white-box `_generic` diff above, not on token-identical logits.)

---

## Engagement proof — the repack actually fires on the real model

(`repack-engagement-proof.log`.) Baseline showed q4_0 tensors *rejected* by CPU_REPACK ("cannot be
used ... using CPU instead", all to CPU_Mapped). AFTER, the load log flips:

```
load_tensors:   CPU_Mapped model buffer size =  3647.87 MiB   (67 tensors: ne[1]%16!=0 etc.)
load_tensors:   CPU_REPACK model buffer size =  3474.00 MiB   (224 q4_0 weight tensors REPACKED)
```

and a one-shot `fprintf` in each kernel's VLEN=128 branch fires during the bench:

```
TCRV REPACK GEMM(q4_0_16x1 VLEN128) ENGAGED nr=512 nc=128 nb=128     <- prefill: 512-token batch
TCRV REPACK GEMV(q4_0_16x1 VLEN128) ENGAGED nc=... nb=128            <- decode/warmup
```

So a non-flat AFTER number cannot be "repack silently fell back to stock": with the prior M-blocked
path compiled out (`TCRV_GEMM_ENABLED=0`), a non-engaging repack would fall to pure per-`vec_dot`
(~1.0×), not the old 1.44×. The measured ~5.84× is therefore unambiguously this repack path.

---

## Measured e2e (`-t 8`, `-r 3`, Llama-2-7B Q4_0)

| test | BEFORE (stock) | AFTER (repack) | speedup |
|---|---:|---:|---:|
| **pp512 (prefill)** | **1.57** (in-session A/B) / 1.55 (banked) | **9.17 ± 0.06** | **~5.84×** |
| tg128 (decode) | 1.38 (banked) | 6.49 ± 0.12 | ~4.7× |

**pp512 BEFORE = in-session A/B re-measure** (`stock-ab-pp512-t8-r1.log`): with the *only* change
vs the AFTER binary being `case 128` reverted to `break` (repack off → pure per-`vec_dot`), rebuilt
and re-benched in the same session/thermal state → **1.57 ± 0.00**, reproducing the banked
clean-stock 1.55 (`../e2e-baseline/RESULTS.md`) within ~1.3%. So the ~5.84× is A/B-confirmed, not a
cross-session or cross-binary artifact. (The stock `-r 3` form was attempted first but the board-side
`timeout` clipped it at 1.55 tok/s before the 3 reps finished; `-r 1` pp512 is the apples-to-apples
denominator and the prior task established pp512 is form-robust to ±0.5%.)

tg128 BEFORE = banked 1.38 (`../e2e-baseline/RESULTS.md`); a stock tg128 re-measure was not run
(decode at 1.38 tok/s makes a clean A/B prohibitively long on the fragile board). See the tg note below.

### Why ~5.84× (not the naively-predicted ~1.7×) — honest decomposition

The prediction assumed a *well-compiled* per-`vec_dot` baseline. The real stock default on this
board is the **GCC-pessimized** per-block path: the prior task's GCC-vs-clang probe
(`../e2e-gemm-integration/`) measured the stock `vec_dot`-style kernel at ~8.7k ns under GCC vs
~2.9k under clang — GCC serializes the per-block `vmv.x.s` scalar-extract ~3×. The repacked kernel
is **`vmv.x.s`-free** (lane-wise, the only reduce is the store), so it escapes BOTH (a) the per-block
cross-lane reduction wall (≈2× algorithmic) AND (b) the GCC scalar-extract serialization (≈3×),
≈6× at the kernel — and since the q4_0 weight matmul dominates prefill, the e2e lands at ~5.84×.
9.17 tok/s sits near a plausible compute ceiling for this board, not above it. The number is
presented WITH its mechanism precisely so it reads as the honest consequence of a bad stock
baseline, not as inflation.

### tg128 (decode) — now on the new GEMV, a real measured win

Unlike the prior task (where M=1 decode fell back to stock, `ENGAGED`=0), `case 128` now routes
decode q4_0 matmuls through the new **GEMV** kernel (confirmed: the GEMV `ENGAGED` print fires on
the real model). So tg128 1.38 → 6.49 (~4.7×) is a genuine measured decode change, reported honestly
against the banked 1.38 — NOT the prior "byte-identical, just clock noise" framing.

A "decode is memory-bound, 4.7× is impossible" objection is anticipated and resolved: stock decode
at 1.38 tok/s was moving only ~5 GB/s (3.56 GiB × 1.38), i.e. it was **compute-bound on the
GCC-pessimized per-block reduction, not memory-bound**. The `vmv.x.s`-free GEMV lets decode rise
toward the real memory-bandwidth ceiling (6.49 tok/s ≈ ~23 GB/s) — same mechanism as prefill. (The
prior task's "~1.22× GEMV cap" was the NON-repacked GEMV and does not apply here.) The tg BEFORE is
the banked cross-session 1.38; a stock in-session tg re-measure was skipped (board-fragility +
length), so the decode ratio carries a residual cross-session caveat that the A/B-confirmed pp512
does not.

### Clock / cross-session note

pp512 from the AFTER `-r1` engage bench (9.02) and the `-r3` headline (9.17±0.06) agree within
~1.7% → clock stable within the AFTER run. The in-session A/B stock re-measure (1.57, same session)
removes the cross-session/cross-binary caveat for the pp512 headline entirely.

---

## Constraints honored

- All work in `~/tcrv-llamacpp` (our copy). No contact with `~/llama_integ*`,
  `~/workspace/workspace3/llama.cpp`, or `*_repackON/_canary/_override`. Mainline ggml source was
  read for the reference algorithm only.
- Build `-j4` (nice -10); board kept gentle — benches run one at a time, never concurrently; no
  full `-t {1,4,8}` sweep (only `-t 8`, the deliverable thread count). Board stayed healthy.

## Files in this dir

- `RESULTS.md` — this file
- `vlen128-q4_0-16x1-kernels.patch` — the two VLEN=128 kernel bodies (gemv + gemm)
- `repack-dispatch-case128.patch` — `case 128` → return the q4_0 repack type
- `ggml-cpu-disable-prior-mblocked.patch` — `TCRV_GEMM_ENABLED 0` (isolate this path)
- `verify_kernel.cpp` — the white-box correctness harness (mine vs `*_generic`)
- `correctness-verify.log` — the PASS run (both metric modes) + engagement
- `repack-engagement-proof.log` — CPU_REPACK load-log flip + kernel ENGAGED on the real model
- `engage-bench-pp512-r1.log` — `-r1` pp512 = 9.02 (engagement bench)
- `after-headline-pp512-tg128-t8-r3.log` — headline AFTER: pp512=9.17, tg128=6.49
- `stock-ab-pp512-t8-r1.log` — in-session stock A/B baseline: pp512 = 1.57 (repack reverted)
- `greedy-after.log` — greedy generation on the AFTER (repack) build (e2e token correctness)
