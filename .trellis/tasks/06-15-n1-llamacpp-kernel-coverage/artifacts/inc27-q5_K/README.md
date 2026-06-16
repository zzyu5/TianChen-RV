# INC-27 — q5_K (full block dot) — byte-exact ssh-rvv evidence

The q5_K coverage rung: the COMPLETE ggml `ggml_vec_dot_q5_K_q8_K`, emitted as
STRUCTURED emitc by our compiler, producing the fp32 `*s` output **byte-exact**
vs ggml's own `_generic` reference on real `ssh rvv`. q5_K = q4_K + a qh 5th-bit
plane. It REUSES q4_K's machinery (commit `b24884a4`, INC-24) ENTIRELY — the
6-bit scale/min bit-dance, the super-block integer core, the deferred d/dmin
fp32 fold + min term — and adds ONLY the 5th (high) weight bit from the qh
plane. **q5_K is now COMPLETE → 8 ggml dot kernels.**

## What is proven (q5_K scope)

The C **our compiler emits** for the new typed op
`tcrv_rvv.q5_k_q8_k_block_dot` reproduces the fp32 dot-product output `*s`
**byte-exact (exact IEEE-754 bit equality)** vs
`ggml_vec_dot_q5_K_q8_K_generic` (`llama.cpp/ggml/src/ggml-cpu/quants.c:720-798`)
on real `ssh rvv` (riscv64, clang 18.1.3, rv64gcv, VLEN=128), under
**`-ffp-contract=off`** (the regime where both reference and kernel are
unfused), over the named n set {256, 512, 2048, 4096, 25600, 65536} + 9 named
edge cases (incl. the qh bit-pattern + q5 all 0/31 + the min-only control) +
2000 random n (NB∈1..32). **0 failures (2015 positive cases).**
**MIN-term negative control 200/200 discriminating; qh 5th-bit negative
control 200/200 discriminating.**

## The qh 5th bit — the ONLY new piece vs q4_K

`block_q5_K` (176 bytes) = q4_K (144 bytes) + a 32-byte qh high-bit plane:

    block_q5_K: d(fp16)@0 | dmin(fp16)@2 | scales[12]@4 | qh[32]@16 | qs[128]@48

The 5-bit weight is `q5 = (nibble | (qh_bit << 4))` ∈ [0,31] **UNSIGNED**. The
qh bit→element mapping (`_generic` quants.c:756-764): `hm = x[i].qh` is a FIXED
32-byte plane reused across all 8 sub-block halves; only the tested bit `m`
varies (`m = 1`, `m <<= 1` per half). So for the 32-element half `h` (0..7),
element `l` (0..31) gets `+16` iff `(qh[l] >> h) & 1` — i.e. half `h` tests bit
`h` of `qh[l]`.

**How it is injected (STRUCTURED, in the shared q4_K core):** the qh plane is
loaded ONCE per super-block (a u8m2 `vle8` from `xb + 16`, NOT chunk-strided)
and, for each unpacked nibble in the **UINT8 domain BEFORE** the `u8->i8`
reinterpret (q5 ∈ [0,31] is non-negative so the reinterpret is exact):

    bit     = vand_vx_u8m2(vsrl_vx_u8m2(qh, h), 0x01)   // (qh[l] >> h) & 1
    contrib = vsll_vx_u8m2(bit, 0x04)                    // << 4  == 16 iff set
    nib     = vadd_vv_u8m2(nib, contrib)                 // a[l] += (qh? 16 : 0)

The emitted shifts are exactly h=0 (no vsrl), 1, 2, …, 7 for the 8 halves —
verified in `tcrv_emitted_kernel.cpp`. **8 structured `vadd_vv_u8m2` injects**
(one per half), zero raw() calls.

## How it REUSES q4_K (no re-derivation)

q4_K's K4b machinery is reused VERBATIM via a shared helper. The integer core
`emitQ4_KSuperBlockAux32Core` gained an OPTIONAL qh plane
(`Q4_KIntegerCoreContext::hasQh` / `qhOffset`), gated EXACTLY like K4a's
`scaleMinOutput`: when `hasQh` is false (q4_K) the core emits **byte-identical**
nodes (no qh load, no inject); when true (q5_K) it injects the +16. The
deferred two-level fp32 fold, the q4_K MIN term (`sumi = Σ bsums[j]*mins[j/2]`,
`sumf -= dmin*sumi` as one `emitc.expression`), and the SEQUENTIAL ascending
l=0..7 horizontal sum are reused node-for-node from K4b. Proof the reuse is
purely additive:

- `diff` the freshly-emitted q4_K kernel vs the committed
  `inc24-q4_K-k4b/tcrv_emitted_kernel.cpp` → **IDENTICAL**.
- `diff` the freshly-emitted K4a aux32 kernel vs
  `inc23-q4_K-k4a/tcrv_emitted_kernel.cpp` → **IDENTICAL**.
- `diff` the freshly-emitted q6_K kernel vs
  `inc12-q6k-k2/tcrv_emitted_kernel.cpp` → **IDENTICAL**.
- All q4_K / q4_0 / q5_0 / q6_K / forward-pass lit tests stay green.

## raw() = 0 / STRUCTURED proof

`grep -c 'raw(' tcrv_emitted_kernel.cpp` → **0** (no actual `raw()` calls).
Every value is a typed emitc node; the `riscv_vector.h` intrinsics go through
`emitc.call_opaque` (the one sanctioned opaque seam), exactly as the
q4_0/q4_1/q5_0/q6_K/q4_K siblings. The qh injection is structured
`vsrl/vand/vsll/vadd` in the u8 domain; the 6-bit scale/min bit-dance is scalar
`emitc.bitwise_*`; the min term is scalar `emitc.mul`/`emitc.add` + one
`emitc.expression`; the horizontal sum is 8 structured `emitc.add`. Verify:

    grep -c 'raw('         tcrv_emitted_kernel.cpp   # -> 0
    grep -c 'vfmacc\|vfmadd' tcrv_emitted_kernel.cpp # -> 0 (no fused FMA)
    grep -c 'vfredu\|vfredo' tcrv_emitted_kernel.cpp # -> 0 (no vector reduce)
    grep -c '= __riscv_vadd_vv_u8m2' tcrv_emitted_kernel.cpp # -> 8 (qh injects)

## The `=on`/`=fast` reference-fusion artifact (honest)

Under `-ffp-contract=on`/`=fast`, the **reference** `_generic` fuses
`sums[l] += d*aux32[l]` to fma, so it diverges from our contraction-immune
kernel by 1–few ULP on ~half the random trials (993/2000). This is a
**reference-compilation artifact, not a kernel bug** — the SAME band q4_K K4b /
q6_K K2 / INC-2a documented. Attribution: BOTH the MIN-term AND the qh 5th-bit
negative controls stay **200/200 discriminating in ALL modes** (off/on/fast),
so the band is solely the positive fold. Our kernel matches `_generic(=off)`
exactly. Do NOT fuse the kernel to chase `=on`/`=fast` parity (that re-breaks
`=off`). See `ssh_rvv_stdout.txt`.

## Files

- `tcrv_emitted_kernel.cpp` — the UNMODIFIED kernel C our compiler emits
  (regenerate: `build/bin/tcrv-opt
  test/Conversion/RVV/rvv-to-emitc-q5-k-q8-k-block-dot.mlir
  --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`).
- `inc27_validate.cpp` — byte-exact harness (oracle = VERBATIM `_generic`, fp16
  shimmed to an exact `_Float16` cast; named n set + 9 edges + 2000 random + the
  MIN-term negative control + the qh 5th-bit negative control; bit-equality on
  `*s`).
- `ssh_rvv_stdout.txt` — the raw board stdout (off headline + on/fast band).

## Residuals (honest)

- **VLEN ≥ 128 pinned** (inherited from q4_K/q6_K): the unpack uses
  `vsetvl_e8m2(32)` and the sub-block quarter uses `e8mf2(8)`/`e32m2`; no
  VLEN<128 re-strip. Correct on rv64gcv ⇒ Zvl128b; a zve32x/zve64x re-strip is a
  capability item.
- **The deployable 8-arg ggml ABI bridge** (a link-overriding
  `extern "C" void ggml_vec_dot_q5_K_q8_K(int n, float*s, size_t bs, …)`) is a
  thin wrapper around the validated 4-arg kernel; not re-built here (the emitted
  kernel already carries the 4 live ABI operands vx/vy/s/n; ggml's trailing
  stride/count args are unused by the q5_K `_generic` math).
- **Performance not measured.** This is a byte-exact correctness/coverage rung.

## Reproduce on the board

    scp tcrv_emitted_kernel.cpp inc27_validate.cpp rvv:~/inc27_q5k/
    ssh rvv 'cd ~/inc27_q5k && \
      clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off \
        tcrv_emitted_kernel.cpp inc27_validate.cpp -o v_off && ./v_off'
