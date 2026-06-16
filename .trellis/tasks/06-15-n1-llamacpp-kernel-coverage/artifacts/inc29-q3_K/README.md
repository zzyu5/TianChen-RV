# INC-29 — q3_K byte-exact compiler-emitted drop-in for ggml_vec_dot_q3_K_q8_K

The 10th ggml K-quant dot kernel, and the **LAST common K-quant**. A new typed op
`tcrv_rvv.q3_k_q8_k_block_dot` lowers, fully STRUCTURED (raw()=0; I5), to a
byte-exact deployable drop-in for `ggml_vec_dot_q3_K_q8_K` — proven `*s`
BITWISE-EQUAL to ggml's own `_generic` reference on real `ssh rvv` (VLEN=128,
rv64gcv, clang 18.1.3) under `-ffp-contract=off`.

With q3_K done, every COMMON ggml dot kernel is covered: q4_0/q8_0/q4_1/q5_0/q5_1
(legacy) + q6_K/q4_K/q5_K/q2_K/q3_K (K-quants). What remains is the **IQ/MXFP4
tail** — a different codebook/fp4 class (iq2_xxs, iq3_s, iq4_nl, mxfp4, ...) that
does NOT compose from the block-quant mechanisms in hand; it needs a codebook /
fp4 dequant family. Noted as the next rung, out of scope here.

## What q3_K is (the 3-bit modern K-quant)

`block_q3_K` (110 bytes, ggml-common.h): `hmask[32]`@0 | `qs[64]`@32 |
`scales[12]`@96 | `d`(fp16)@108. `block_q8_K` (292 bytes): `d`(fp32)@0 |
`qs[256]`@4 | `bsums[16]`@260 (bsums UNUSED by q3_K — it is symmetric, no min).
256 elements = 16 sub-blocks of 16.

The `_generic` math (quants.c:566-643), reproduced node-for-node:
- **3-bit weight, SUBTRACTIVE hmask**: `a[l] = (q3[l] >> shift) & 3` (the 2 low
  bits, q2_K-style over shifts {0,2,4,6}) then `a[l] -= (hm[l] & m ? 0 : 4)`. The
  high bit comes from the 32-byte hmask plane but is applied SUBTRACTIVELY: when
  the bit is SET the lane stays in [0,3]; when UNSET it drops by 4 into [-4,-1].
  So the lane is SIGNED in [-4,3]. The hmask plane is the SAME 32 bytes for all 8
  groups (`hm` is NEVER advanced; only the tested bit `m = 1<<p` shifts left, with
  the running bit position p = 4*chunk + shiftIdx).
- **6-bit SIGNED scales**: the 12 packed `scales[12]` bytes decode (a q3_K-OWN
  bit-dance, masks kmask1=0x03030303 / kmask2=0x0f0f0f0f — DISTINCT from q4_K's
  0x3f3f3f3f) into 16 6-bit values read as SIGNED int8; the per-sub-block scale is
  `scales[j] - 32` (signed, in [-32,31]).
- **NO min / NO dmin** (symmetric, like q6_K). The fold is q6_K's deferred
  `sums[l] += d * aux32[l]` (d = fp16(x.d) * y.d) with the signed scale folded
  into aux32; `sumf = Σ sums[l]` then `*s = sumf`.

## The three q3_K-specific tricks (vs the in-hand mechanisms)

q3_K COMPOSES q2_K + q5_K + q4_K + q6_K mechanisms; the genuinely-NEW pieces:

1. **The SUBTRACTIVE hmask high-bit injection.** q5_K injects the high bit
   ADDITIVELY (`+16` when set, lifting [0,15]→[0,31]). q3_K injects it
   SUBTRACTIVELY: emitted as `a = (low2 | (hbit<<2)) - 4` (`hbit = (hm>>p)&1`),
   which is exactly `_generic`'s `a = low2; a -= (hm&m ? 0 : 4)` — bit SET →
   `(low2|4)-4 = low2`; bit UNSET → `low2-4`. The plane is the FIXED 32 bytes (NOT
   q6_K's advancing qh); the bit position p = 4*chunk + shiftIdx runs 0..7.
2. **The SIGNED 6-bit scale dance.** q4_K's dance produces UNSIGNED 6-bit scales
   (masks 0x3f3f3f3f); q3_K's dance is a DIFFERENT shuffle (masks 0x03030303 /
   0x0f0f0f0f) read as SIGNED int8 with a `-32` bias. The 3 input words w0/w1/w2
   are captured BEFORE any output (avoiding `_generic`'s in-place
   read-before-write hazard on auxs[]) and all 4 output words computed as pure
   functions of them.
3. **The NO-min fold** (q6_K's, not q4_K/q5_K/q2_K's min-bearing fold): a single
   deferred d·Σaux32 with NO `-dmin*summs` term and NO bsums read.

## Byte-exact result (ssh rvv)

`*s` BITWISE-EQUAL (IEEE-754 bit equality) to ggml `_generic` over **2020
positive cases** (named n {256,512,2048,4096,25600,65536} + 6 edges + the 8
PER-PLANE hmask cases + 2000 random NB in 1..32), **0 failures**, under
**`-ffp-contract=off`** (the regime where both reference and kernel are
contraction-free; the headline byte-exactness target — the on-board `_vlN` path
is raw asm and does NOT define the fp32 order). Two negative controls, 200/200
discriminating each:
- **SUBTRACTIVE-hmask control** (oracle with the polarity INVERTED: `-= (hm&m ? 4
  : 0)`) — proves the subtractive direction (the q3_K high-bit decode) is
  load-bearing. The 8 per-plane cases ALSO each independently discriminate
  against the inverted oracle, proving all 8 bit-planes p=0..7 (both chunks) are
  correctly mapped — closing the "a bug that only mis-maps p≥4 passes weak tests"
  blind spot.
- **SIGNED-scale control** (oracle dropping the `-32` bias) — proves the signed
  6-bit scale (the q3_K dance's signed extraction) is load-bearing.

### -ffp-contract on/fast band (documented, NOT a kernel bug)

Under `-ffp-contract=on`/`=fast` the **reference** `_generic` fuses its scalar
`sums[l] += d*aux32[l]` to an fma, diverging from our kernel's contraction-immune
SEPARATE vfmul/vfadd — so a positive-fold band appears (1040 mismatches). This is
the SAME deferred-fold band q6_K / q5_K / q4_K K4b show; it is a
**reference-compilation artifact, not a kernel bug** (the deferred d·Σaux32 fold
is the only fp32 op affected). Both negative controls stay **200/200
discriminating in ALL modes** (off/on/fast). Our kernel matches `_generic(=off)`
exactly — the well-defined order. See `ssh_rvv_stdout.txt` for the raw board
stdout (off headline + on/fast band).

## raw()=0 / additive / lit

- raw()=0; the emitted kernel is fully STRUCTURED emitc (zero actual `raw()`
  call, zero non-comment verbatim → zero verbatim C control flow; all `tcrv_rvv.*`
  ops lowered).
- ADDITIVE: q6_K (inc12), q4_K k4b (inc24), q5_K (inc27), q2_K (inc28) emitted
  kernels regenerate BYTE-IDENTICAL; all siblings untouched. A NEW q3_K emission
  function + a NEW q3_K integer core (the q6_K/q4_K cores are byte-untouched).
- Full clean rebuild green; 648 lit tests, 645 pass, exactly the 3 documented
  environmental reds (`Scripts/rvv-generated-bundle-abi-e2e-*`; pre-existing,
  unrelated to q3_K). 2 new lit tests
  (`test/Dialect/RVV/q3-k-q8-k-block-dot-dataflow.mlir` fail-closed verifier;
  `test/Conversion/RVV/rvv-to-emitc-q3-k-q8-k-block-dot.mlir` structured emission)
  pass.

## Files

- `tcrv_emitted_kernel.cpp` — the UNMODIFIED kernel C our compiler emits
  (regenerate: `build/bin/tcrv-opt
  test/Conversion/RVV/rvv-to-emitc-q3-k-q8-k-block-dot.mlir
  --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`).
- `inc29_validate.cpp` — byte-exact harness (oracle = VERBATIM `_generic`, fp16
  shimmed to an exact `_Float16` cast; named n set + 6 edges + 8 per-plane hmask
  cases + 2000 random + the SUBTRACTIVE-hmask and SIGNED-scale negative controls;
  bit-equality on `*s`).
- `ssh_rvv_stdout.txt` — the raw board stdout (off/on/fast all reported).

## Residuals (honest)

- **VLEN >= 128 pinned** (inherited from q6_K/q4_K/q5_K/q2_K): the unpack uses
  `vsetvl_e8m2(32)` and the sub-block dot uses `e8mf2(8)`/i16m1/e32m2; no VLEN<128
  re-strip. Correct on rv64gcv => Zvl128b; a zve32x/zve64x re-strip is a
  capability item.
- **The deployable 8-arg ggml ABI bridge** (a link-overriding
  `extern "C" void ggml_vec_dot_q3_K_q8_K(int n, float*s, size_t bs, ...)`) is a
  thin wrapper around the validated 4-arg kernel; not re-built here (the emitted
  kernel already carries the 4 live ABI operands vx/vy/s/n; ggml's trailing
  stride/count args are unused by the q3_K `_generic` math).
- **Performance not measured.** This is a byte-exact correctness/coverage rung.

## Reproduce on the board

    scp tcrv_emitted_kernel.cpp inc29_validate.cpp rvv:~/inc29_q3k/
    ssh rvv 'cd ~/inc29_q3k && \
      for m in off on fast; do \
        clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=$m \
          tcrv_emitted_kernel.cpp inc29_validate.cpp -o v_$m && ./v_$m | tail -3; \
      done'
