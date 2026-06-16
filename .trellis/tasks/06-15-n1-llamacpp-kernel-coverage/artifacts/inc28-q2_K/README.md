# INC-28 — q2_K byte-exact compiler-emitted drop-in for ggml_vec_dot_q2_K_q8_K

The 9th ggml K-quant dot kernel. A new typed op `tcrv_rvv.q2_k_q8_k_block_dot`
lowers, fully STRUCTURED (raw()=0; I5), to a byte-exact deployable drop-in for
`ggml_vec_dot_q2_K_q8_K` — proven `*s` BITWISE-EQUAL to ggml's own `_generic`
reference on real `ssh rvv` (VLEN=128, rv64gcv, clang 18.1.3).

## What q2_K is (the 2-bit modern K-quant)

`block_q2_K` (84 bytes, ggml-common.h): `scales[16]`@0 | `qs[64]`@16 |
`d`(fp16)@80 | `dmin`(fp16)@82. `block_q8_K` (292 bytes): `d`(fp32)@0 |
`qs[256]`@4 | `bsums[16]`@260. 256 elements = 16 sub-blocks of 16.

The `_generic` math (quants.c:514-564), reproduced node-for-node:
- `summs = Σ_{j=0..15} bsums[j] * (sc[j] >> 4)`  — the min term (HIGH nibble of
  each of the 16 direct scales bytes).
- `dall = y.d * fp16(x.d)`, `dmin = y.d * fp16(x.dmin)`.
- `isum = Σ_{s=0..15} (sc[s] & 0xF) * Σ_{l in sub-block s} q8[l] * ((q2 >> shift) & 3)`
  — the positive term (LOW nibble scale, 2-bit weight unpack over shifts
  {0,2,4,6}).
- `sumf += dall*isum - dmin*summs`  — the SCALAR fp32 fold (one statement, in
  super-block order).

## What it REUSES from q4_K vs the 3 NEW pieces

REUSES the super-block scaffolding (the QK_K=256 AoS super-block loop, the q8_K
activation handling, the `(float)*(const _Float16 *)` fp16 read seam, the
structured `*s` store) and the scale+min STRUCTURE (a per-sub-block scale term
plus a `-dmin*Σ bsums*min` min term). The 3 q2_K-specific differences:

1. **2-bit weight unpack** (vs q4_K's 4-bit): for each 32-byte qs chunk (2 of
   them) and each shift in {0,2,4,6}, `aux8[128k + 32*(shift/2) + l] =
   (qs[k*32+l] >> shift) & 3` for the 32 lanes l — a u8m2(32) load + vsrl + vand
   + u8->i8 reinterpret + vse8. Sub-block s -> aux8[16s:16s+16].
2. **4-bit nibble scale/min** (vs q4_K's 6-bit utmp/kmask bit-dance): the 16
   direct `scales[16]` bytes are read straight — `sc[s] & 0xF` is the scale,
   `sc[s] >> 4` is the min — as scalar emitc.bitwise_and / _right_shift. NO
   bit-dance.
3. **SCALAR fp32 fold** (vs q4_K's 8-lane deferred vector + post-loop horizontal
   sum): q2_K's positive term is a single per-super-block int `isum`, so the
   whole fold is ONE scalar `sumf += dall*isum - dmin*summs` carried in
   super-block order — emitted as ONE emitc.expression (two products, a subtract,
   an add). NO sums vector, NO vfcvt/vfmul/vfadd lanes, NO post-loop vse32+8-add.

The per-sub-block integer dot (order-free) is vectorized: vle8 i8m1 (16 lanes) x2
-> vwmul_vv i16m2 -> vwredsum_vs into i32m1 lane 0 -> vmv_x_s. LMUL=1 (e8m1) so
the 16-element reduce sees all 16 lanes (VLMAX(e8m1) = VLEN/8 = 16 at VLEN>=128).

## Byte-exact result (ssh rvv)

`*s` BITWISE-EQUAL (IEEE-754 bit equality) to ggml `_generic` over **2013
positive cases** (named n {256,512,2048,4096,25600,65536} + 7 edges + 2000 random
NB in 1..32), **0 failures**, under **-ffp-contract=off AND on AND fast** (q2_K
is MODE-STABLE — the fold is one scalar expression with the same tree on both
sides, so unlike q4_K/q5_K there is NO on/fast band). Two negative controls,
200/200 discriminating each:
- **MIN-term control** (oracle with `dmin -> dmin+1`) — proves the `-dmin*summs`
  subtraction is load-bearing. The `min-only` edge (q2=0 -> isum=0 -> `*s =
  -dmin*summs`) directly isolates it (bit-exact).
- **2-bit-unpack control** (oracle with `>> shift` but NO `& 3`) — proves the
  2-bit weight mask is load-bearing (the q2_K-distinguishing weight decode).

See `ssh_rvv_stdout.txt` for the raw board stdout (all three modes).

## raw()=0 / additive / lit

- raw()=0; the emitted kernel is fully STRUCTURED emitc (no actual `raw()` call,
  no verbatim C control flow).
- ADDITIVE: q4_K (inc24), q5_K (inc27), q6_K (inc12) emitted kernels regenerate
  BYTE-IDENTICAL; all siblings untouched.
- Full clean rebuild green; 646 lit tests, 643 pass, exactly the 3 documented
  environmental reds (`Scripts/rvv-generated-bundle-abi-e2e-*`; pre-existing,
  unrelated to q2_K). 2 new lit tests
  (`test/Dialect/RVV/q2-k-q8-k-block-dot-dataflow.mlir` fail-closed verifier;
  `test/Conversion/RVV/rvv-to-emitc-q2-k-q8-k-block-dot.mlir` structured emission)
  pass.

## Files

- `tcrv_emitted_kernel.cpp` — the UNMODIFIED kernel C our compiler emits
  (regenerate: `build/bin/tcrv-opt
  test/Conversion/RVV/rvv-to-emitc-q2-k-q8-k-block-dot.mlir
  --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`).
- `inc28_validate.cpp` — byte-exact harness (oracle = VERBATIM `_generic`, fp16
  shimmed to an exact `_Float16` cast; named n set + 7 edges + 2000 random + the
  MIN-term negative control + the 2-bit-unpack negative control; bit-equality on
  `*s`).
- `ssh_rvv_stdout.txt` — the raw board stdout (off/on/fast all PASS).

## Residuals (honest)

- **VLEN >= 128 pinned** (inherited from q4_K/q6_K): the unpack uses
  `vsetvl_e8m2(32)` and the sub-block dot uses `e8m1(16)`/i16m2; no VLEN<128
  re-strip. Correct on rv64gcv => Zvl128b; a zve32x/zve64x re-strip is a
  capability item.
- **The deployable 8-arg ggml ABI bridge** (a link-overriding
  `extern "C" void ggml_vec_dot_q2_K_q8_K(int n, float*s, size_t bs, ...)`) is a
  thin wrapper around the validated 4-arg kernel; not re-built here (the emitted
  kernel already carries the 4 live ABI operands vx/vy/s/n; ggml's trailing
  stride/count args are unused by the q2_K `_generic` math).
- **Performance not measured.** This is a byte-exact correctness/coverage rung.

## Reproduce on the board

    scp tcrv_emitted_kernel.cpp inc28_validate.cpp rvv:~/inc28_q2k/
    ssh rvv 'cd ~/inc28_q2k && \
      for m in off on fast; do \
        clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=$m \
          tcrv_emitted_kernel.cpp inc28_validate.cpp -o v_$m && ./v_$m | tail -3; \
      done'
