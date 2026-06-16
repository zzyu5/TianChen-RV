# INC-11 — q6_K super-block K1 (integer core) — byte-exact ssh-rvv evidence

The first **super-block K-quant** rung: the ggml `ggml_vec_dot_q6_K_q8_K`
INTEGER CORE, emitted as STRUCTURED emitc by our compiler, byte-exact vs ggml's
own `_generic` reference on real `ssh rvv`.

## What is proven (K1 scope)

The C **our compiler emits** for the new typed op
`tcrv_rvv.q6_k_q8_k_aux32_partial` reproduces the per-super-block INTEGER state
`aux32[8]` — exactly as `ggml_vec_dot_q6_K_q8_K_generic`
(`llama.cpp/ggml/src/ggml-cpu/quants.c:800`) computes it **right before the fp32
`d`-multiply** — **byte-exact (exact i32 equality)** on real `ssh rvv` (riscv64,
clang 18.1.3, rv64imafdcv, VLEN=128), over **4013 super-blocks**: 6 named edge
cases + 4000 random single super-blocks + 7 multi-super-block runs (NB=2..8,
exercising the outer loop + the `out + ib*8` aux32 indexing). **0 failures.**

K1 isolates the genuinely-new super-block integer machinery from the fp32 fold:
- the **6-bit ql+qh unpack** (256 weights, element-ordered, biased `-32`);
- the **per-sub-block int8 scale applied in the i32 domain** (vwmacc.vx);
- the **nested sub-block structure** (16 sub-blocks of 16 = 2 halves of 8);
- the **lane-collapsed 8-lane aux32** accumulator.
The fp32 two-level fold (deferred fp16-`d`×fp32-`d` into `sums[8]` + the
sequential horizontal sum into `*s`) is **K2 — out of scope here**. K1's output
and validation target is the INTEGER `aux32[8]` state.

## The op + lowering (STRUCTURED emitc, zero raw())

`tcrv_rvv.q6_k_q8_k_aux32_partial` (`include/.../RVVOps.td`) carries the
super-block format facts as I4-mirror attrs (qk=256, sub_block=16, weight stride
210, activation stride 292, qh@128, scales@192, q8@4) and four runtime-ABI
operands (q6_K weight `vx`, q8_K activation `vy`, the `int32_t*` aux32
destination, `n`). It verifies fail-closed (I7). Its lowering
(`lib/Conversion/RVV/RVVToEmitC.cpp`, `emitQ6_KQ8_KAux32Partial`) mirrors
`_generic` line-for-line so byte-exactness is by construction:
- an outer `emitc.for` super-block loop over `nb = n/256`;
- a function-scoped `int8_t aux8[256]` scratch (`!emitc.array`), filled by the
  6-bit unpack in two 128-element chunks — each chunk a 4-strip 32-wide
  `e8m2` `vand`/`vsrl`/`vsll`/`vor`/`vreinterpret`/`vsub` reconstruct + a
  `vse8` store into `aux8` at the EXACT `_generic` element permutation
  (`a[l+0]/[l+32]/[l+64]/[l+96]`). The permutation lives ENTIRELY in the unpack;
  the dot reads `aux8` contiguously (exactly `_generic`'s `a += 8`);
- a nested `emitc.for` sub-block loop over 16 sub-blocks: a scalar `int8` scale
  load, then two 8-element halves each doing `vle8` q8 + `vle8` aux8 →
  `vwmul_vv` i8×i8→i16m1 → `vwmacc_vx` i32m2 `+= scale × i16` into the carried
  8-lane `aux32` accumulator (e32m2, VLMAX=8 at VLEN≥128);
- a `vse32_v_i32m2` store of `aux32[8]` through the output pointer at `out+ib*8`.

**raw() = 0** in the emitted kernel — every value is a typed emitc node (the
`riscv_vector.h` intrinsics go through `emitc.call_opaque`, the one sanctioned
opaque seam, exactly as the q4_0/q8_0/q4_1 siblings do). Verify:

    grep -c 'raw(' tcrv_emitted_kernel.cpp        # -> 0

## Files

- `tcrv_emitted_kernel.cpp` — the UNMODIFIED kernel C our compiler emits, rendered
  by the standard MLIR EmitC C emitter (no hand-transcription). See "Regenerate".
- `inc11_validate.cpp` — the validation harness: ggml's per-super-block aux32
  integer state as the ground truth (a line-for-line mirror of `_generic`'s
  unpack + scaled-dot, `uint8_t` ql/qh so the nibble shifts are logical) + a
  driver that feeds both the reference and the emitted kernel many random + edge
  + multi-super-block cases and asserts exact `aux32[8]` equality.
- `ssh_rvv_stdout.txt` — the raw board stdout (POSITIVE PASS + NEGATIVE control).

## Edge cases (all PASS)

- `q6-all-minus32` — every decoded weight = −32 (q6 low extreme: ql nibble 0, qh 0).
- `q6-all-plus31` — every decoded weight = +31 (q6 high extreme: ql nibble 0xF, qh 0xFF).
- `q6+31 scales-extremes q8+127` — scales alternating +127/−128, q8 +127.
- `q6-32 scale-128 q8-128` — q8 at the asymmetric int8 extreme, scale −128.
- `q6+31 scale+127 q8-128` — max-magnitude products into aux32.
- `mixed-ramp q8-extremes` — structured ql/qh/scales ramp, q8 alternating ±.

## Negative control

Perturbing the reference aux32 accumulation by `+1` makes ALL super-blocks FAIL
(4013/4013), proving the harness genuinely discriminates — it is not vacuously
passing. See `ssh_rvv_stdout.txt`.

## Residuals (honest)

- **VLEN ≥ 128 pinned.** K1 emits the byte-exact form for the validated target
  (rv64gcv ⇒ Zvl128b ⇒ VLEN ≥ 128): the unpack uses one `vsetvl_e8m2(32)` per
  32-element group and the sub-block half uses `e8mf2(8)`/`e32m2` (8 aux32
  lanes) with NO VLEN-robust re-strip loop. On a VLEN < 128 board `vsetvl`
  returns < 32 and part of `aux8` would be left unwritten — unlike the q4_0
  `robust` default which re-strips. The VLEN < 128 (zve32x/zve64x) re-strip is a
  K2/capability item, not built here.
- The op's SSA `vector<i32,"m1">` result token is vestigial (bound to a dummy `0`
  literal): the real output is the `int32_t*` aux32 store, so the token has no
  live use and is erased with the variant. The token keeps the op shape uniform
  with the q4_0/q8_0/q4_1 siblings.
- The byte-exact oracle is a line-for-line C mirror of `_generic` (the task's
  permitted "compute the identical integer partial directly from the _generic
  math" route), cross-checked against quants.c:820-847. The lowering is a
  structurally different (vectorized) implementation; agreement on 4013 diverse
  inputs with a 1-discriminating negative control gives byte-exact-vs-ggml
  transitively.

## Regenerate the emitted kernel C

    build/bin/tcrv-opt \
      test/Conversion/RVV/rvv-to-emitc-q6-k-q8-k-aux32-partial.mlir \
      --tcrv-rvv-lower-to-emitc \
      | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp

## Reproduce on the board

    scp tcrv_emitted_kernel.cpp inc11_validate.cpp rvv:~/inc11_q6k_k1/
    ssh rvv 'cd ~/inc11_q6k_k1 && \
      clang++ -O2 -march=rv64gcv -mabi=lp64d \
        tcrv_emitted_kernel.cpp inc11_validate.cpp -o inc11_validate && \
      ./inc11_validate'

## The hardest part of K1 / what is left for K2

- **Hardest in K1**: the ql/qh → element permutation (`a[l+0/32/64/96]` from the
  low/high nibbles of `ql[l]`/`ql[l+32]` with the matching 2-bit field of
  `qh[l]`). A vectorized unpack that groups it wrong is *silently* off. Kept
  byte-exact by materializing `aux8[256]` and mirroring `_generic`'s 4-write
  permutation in the unpack, so the dot reads `aux8` contiguously.
- **Left for K2** (the genuinely-hard fp32 piece, deliberately NOT built here):
  the deferred two-level fold — `d = fp16(x.d) * y.d`; `sums[l] += d * aux32[l]`
  into 8 INDEPENDENT fp32 lane-accumulators across super-blocks; then the
  **sequential** horizontal sum `sumf += sums[l]` (l=0..7) into `*s`. That fp32
  ordering (deferred multi-lane fold + ordered horizontal reduce; do NOT
  reassociate) is the byte-exactness risk a self-test passes but the board can
  fail — a K2 verify-on-board item. K2 also adds the ggml call ABI
  (`*s`/`bs`/`bx`/`by`/`nrc`) for a deployable drop-in.
