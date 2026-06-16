# INC-24 — q4_K K4b (full block dot) — byte-exact ssh-rvv evidence

The q4_K completion rung: the COMPLETE ggml `ggml_vec_dot_q4_K_q8_K`, emitted as
STRUCTURED emitc by our compiler, producing the fp32 `*s` output **byte-exact**
vs ggml's own `_generic` reference on real `ssh rvv`. K4a (commit `021a7ff0`)
proved the q4_K integer core (the `aux32[8]` state + the decoded 6-bit
scales/mins); K4b adds the deferred two-level fp32 fold + the q4_K **MIN term** +
the ggml ABI for a deployable drop-in. **q4_K is now COMPLETE.**

## What is proven (K4b scope)

The C **our compiler emits** for the new typed op
`tcrv_rvv.q4_k_q8_k_block_dot` reproduces the fp32 dot-product output `*s`
**byte-exact (exact IEEE-754 bit equality)** vs
`ggml_vec_dot_q4_K_q8_K_generic` (`llama.cpp/ggml/src/ggml-cpu/quants.c:645-718`)
on real `ssh rvv` (riscv64, clang 18.1.3, rv64gcv, VLEN=128), under
**`-ffp-contract=off`** (the regime where both reference and kernel are
unfused), over the named n set {256, 512, 2048, 4096, 25600, 65536} + 6 named
edge cases + 2000 random n (NB∈1..32). **0 failures (2012 positive cases).**
**MIN-term negative control 200/200 discriminating.**

## The fp32 fold WITH the MIN term — exactly matching `_generic`

Per super-block, after the K4a integer core produces `aux32[8]` + the decoded
`scales[8]`/`mins[8]`:

- **positive fold (reused from q6_K K2, byte-identical mechanism):**
  - `d = GGML_FP16_TO_FP32(x[i].d) * y[i].d` — the fp16 weight scale (byte 0,
    read via the `(float)*(const _Float16 *)` seam) times the fp32 activation
    scale (byte 0, loaded ONCE and reused for both `d` and `dmin`).
  - `sums[l] += d * aux32[l]` — 8 INDEPENDENT fp32 lane-accumulators, emitted as
    `vfcvt` (i32→f32, RNE) then a **SEPARATE** `vfmul` (by `d`) then a
    **SEPARATE** `vfadd` into `sums`. **NEVER** a fused `vfmacc`/`vfmadd`.
    `sums` (a `vfloat32m2`) is declared+zeroed **ONCE** outside the loop.
- **the MIN term (the NEW piece vs q6_K, the q4_K distinction):**
  - `int sumi = 0; for (j=0..15) sumi += (int)bsums[j] * (int)mins[j/2];` — the
    `bsums` are q8_K's int16 per-sub-block sums (@260), **sign-extended** on load
    (`const int16_t v128[j]` → `(int)`); each decoded uint6 `min` spans **TWO**
    consecutive bsums via `mins[j/2]` (j=0,1→min[0]; 2,3→min[1]; …; 14,15→min[7]
    = `scalesU8[8 + j/2]`). Integer multiply/add → order-free, a scalar loop.
  - `dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d` — the fp16 weight min scale
    (byte 2) times the SAME fp32 activation `y.d`.
  - `sumf -= dmin * sumi` — carried in a SCALAR `sumf` accumulator (init 0.0f
    ONCE outside the loop, accumulated IN-LOOP in super-block order, **distinct
    from q6_K's post-loop-only sumf**). Emitted as **ONE `emitc.expression`** so
    it renders as the single C statement `v13 = v253 - v251 * (float) v252`,
    tracking `_generic`'s `sumf -= dmin * sumi` contraction.
- **the SEQUENTIAL horizontal sum (AFTER the loop):** `sumf` already holds the
  accumulated MIN subtractions; `sums` is `vse32`-stored into a `float[8]`
  scratch, then `sumf += sums[0]; …; sumf += sums[7]` — 8 scalar `emitc.add` in
  ascending lane order. **NEVER** a vector `vfredusum` (fp add is
  non-associative). This mirrors `_generic` EXACTLY: the min subtractions land in
  `sumf` in-loop (super-block order), then the 8 lane adds run after the loop.

## How it reuses K4a + q6_K K2

- **K4a's integer core** (the 4-bit unpack into `aux8[256]`, the 6-bit
  scale/min bit-dance, the per-sub-block uint6-scaled i32 accumulation) was
  factored into a shared helper `emitQ4_KSuperBlockAux32Core` (returns the
  per-super-block `aux32` lvalue + the decoded `scalesU8` pointer; `mins =
  scalesU8 + 8`). K4a calls it (with a non-null `scaleMinOutput` so the helper
  emits the K4a-only 16-byte scale/min store IN-PLACE) then `vse32`-stores
  aux32; K4b calls it (null `scaleMinOutput` → that store elided) then folds.
  This is a **pure extraction**: K4a's emitted C is **byte-identical** to the
  committed `inc23-q4_K-k4a/tcrv_emitted_kernel.cpp` (verified by `diff` →
  IDENTICAL), and the K4a conversion + dialect lit tests stay green.
- **q6_K K2's fold mechanism** (the separate `vfcvt`/`vfmul`/`vfadd` positive
  fold + the carried 8-lane `sums` + the sequential horizontal sum) is reused
  node-for-node; K4b adds only the scalar `sumf` carry + the MIN term on top.

## raw() = 0 / STRUCTURED proof

`grep -c 'raw(' tcrv_emitted_kernel.cpp` → **0** (no actual `raw()` calls).
Every value is a typed emitc node; the `riscv_vector.h` intrinsics go through
`emitc.call_opaque` (the one sanctioned opaque seam), exactly as the
q4_0/q8_0/q4_1/q6_K/K4a siblings. The 6-bit scale/min bit-dance is scalar
`emitc.bitwise_*`, the min term is scalar `emitc.mul`/`emitc.add` + one
`emitc.expression`, the horizontal sum is 8 structured `emitc.add`. Verify:

    grep 'vfredu\|vfredo'   tcrv_emitted_kernel.cpp   # -> (none, no vector reduce)
    grep 'vfmacc\|vfmadd'   tcrv_emitted_kernel.cpp   # -> (none, no fused FMA)
    grep 'v13 = v253 - v251' tcrv_emitted_kernel.cpp  # -> the single-statement min subtract

## The `=on`/`=fast` reference-fusion artifact (honest)

Under `-ffp-contract=on`/`=fast`, the **reference** `_generic` fuses
`sums[l] += d*aux32[l]` to fma, so it diverges from our contraction-immune
kernel by 1–few ULP on ~half the random trials (913/2000). This is a
**reference-compilation artifact, not a kernel bug** — the SAME band q6_K K2 /
INC-2a documented. Attribution: the **MIN-term negative control stays 200/200
discriminating in ALL modes** (off/on/fast), so the band is solely the positive
fold; the min term (the single `emitc.expression`) is mode-stable. Our kernel
matches `_generic(=off)` exactly. Do NOT fuse the kernel to chase `=on`/`=fast`
parity (that re-breaks `=off`). See `ssh_rvv_stdout.txt`.

## Files

- `tcrv_emitted_kernel.cpp` — the UNMODIFIED kernel C our compiler emits
  (regenerate: `build/bin/tcrv-opt
  test/Conversion/RVV/rvv-to-emitc-q4-k-q8-k-block-dot.mlir
  --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`).
- `inc24_validate.cpp` — byte-exact harness (oracle = VERBATIM `_generic`, fp16
  shimmed to an exact `_Float16` cast; named n set + edges + 2000 random + the
  MIN-term negative control; bit-equality on `*s`).
- `ssh_rvv_stdout.txt` — the raw board stdout (off headline + on/fast band).

## Residuals (honest)

- **VLEN ≥ 128 pinned** (inherited from K4a/q6_K): the unpack uses
  `vsetvl_e8m2(32)` and the sub-block quarter uses `e8mf2(8)`/`e32m2`; no
  VLEN<128 re-strip. Correct on rv64gcv ⇒ Zvl128b; a zve32x/zve64x re-strip is a
  capability item.
- **The deployable 8-arg ggml ABI bridge** (a real link-overriding
  `extern "C" void ggml_vec_dot_q4_K_q8_K(int n, float*s, size_t bs, …)`, as
  inc12 built for q6_K) is a thin wrapper around the validated 4-arg kernel; not
  re-built here (the emitted kernel already carries the 4 live ABI operands
  vx/vy/s/n; ggml's trailing stride/count args are unused by the q4_K `_generic`
  math). The byte-exact kernel + the ABI mapping (`n` widened, block pointers
  cast) is the deployable core.
- **Performance not measured.** K4b is a byte-exact correctness rung.

## Reproduce on the board

    scp tcrv_emitted_kernel.cpp inc24_validate.cpp rvv:~/inc24_q4k_k4b/
    ssh rvv 'cd ~/inc24_q4k_k4b && \
      clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off \
        tcrv_emitted_kernel.cpp inc24_validate.cpp -o v_off && ./v_off'
