# Research: ggml_vec_dot_q4_0_q8_0 exact semantics → our IR mapping

- **Query**: Reverse-engineer EXACT semantics of `ggml_vec_dot_q4_0_q8_0` (RVV) and map to TianChen-RV IR; produce a precise gap list to a numerically-exact deployable replacement.
- **Scope**: mixed (external llama.cpp source + internal repo)
- **Date**: 2026-06-15

> **Headline correction (verified against source):** the parent task brief assumed
> we "already express the nibble unpack + widening product + reduce". We do — but for
> the **WRONG product shape and the WRONG nibble convention**. ggml's kernel is
> **i4×i8 asymmetric** with an **offset-binary (`n−8`)** nibble decode; our existing
> primitive is **i4×i4 symmetric** with a **two's-complement** nibble decode. These
> produce *different dot products from identical bytes*. The honest verdict is a
> **LARGE** gap, not small/medium. Details and per-byte proof below.

---

## (a) Exact block formats

Source: `llama.cpp/ggml/src/ggml-common.h`.

```c
#define QK4_0 32
typedef struct {
    ggml_half d;            // fp16 delta (scale)              // offset 0, 2 bytes
    uint8_t   qs[QK4_0/2];  // 16 bytes: two 4-bit nibbles each // offset 2, 16 bytes
} block_q4_0;               // sizeof == 18 bytes  (ggml-common.h:184-189)

#define QK8_0 32
typedef struct {
    ggml_half d;            // fp16 delta (scale)              // offset 0, 2 bytes
    int8_t    qs[QK8_0];    // 32 signed int8 quants            // offset 2, 32 bytes
} block_q8_0;               // sizeof == 34 bytes  (ggml-common.h:241-246)
```

Key structural facts:
- **Array-of-Structs (AoS).** `vx` / `vy` are *arrays of these structs* — fp16 scale is
  **inline at the head of every block**, not a separate flat scale buffer. Block stride
  = 18 B (q4_0) / 34 B (q8_0); the quant payload starts at byte offset 2.
- `ggml_half` = IEEE fp16. On the rvv board (`__riscv_zfhmin`) it converts via a *scalar*
  `_Float16` cast — see (c).
- **Nibble packing (q4_0).** Each `qs[j]` byte holds TWO weights:
  low nibble `qs[j] & 0x0F` = logical element `j`; high nibble `qs[j] >> 4` = logical
  element `j + 16`. The decoded signed value is **`nibble − 8`** (offset-binary): nibble
  `0 → −8`, `8 → 0`, `15 → +7`. Range = [−8, +7].
- **q8_0 quants** are plain signed `int8` in [−128, 127] (already the activation scale's
  integer residue); **NOT** packed, **NOT** re-decoded.

---

## (b) Exact math the kernel computes

Ground truth = `ggml_vec_dot_q4_0_q8_0_generic`
(`llama.cpp/ggml/src/ggml-cpu/quants.c:174-208`):

```c
const int qk = QK8_0;            // 32
const int nb = n / qk;           // number of block pairs
float sumf = 0;
for (ib = 0; ib < nb; ++ib) {
    int sumi0 = 0, sumi1 = 0;
    for (j = 0; j < qk/2; ++j) {                 // j = 0..15
        const int v0 = (x[ib].qs[j] & 0x0F) - 8; // LOW  nibble, decoded [-8,7]
        const int v1 = (x[ib].qs[j] >>   4) - 8; // HIGH nibble, decoded [-8,7]
        sumi0 += v0 * y[ib].qs[j];               // i4 × i8, low half  ↔ y[0..15]
        sumi1 += v1 * y[ib].qs[j + qk/2];        // i4 × i8, high half ↔ y[16..31]
    }
    int sumi = sumi0 + sumi1;                                 // i32
    sumf += sumi * GGML_CPU_FP16_TO_FP32(x[ib].d)             // × fp16 weight scale
                 * GGML_CPU_FP16_TO_FP32(y[ib].d);            // × fp16 activation scale
}
*s = sumf;
```

Compact form — the whole kernel is, per block pair `ib`:

```
s += d_x(ib) * d_y(ib) * Σ_{i=0..31} ( (nibble_i − 8) · q8_i )
```
where `d_x`, `d_y` are the two **per-block fp16 scales**, summed in **fp32** across all
`nb` blocks. Three multiplicative scalings: integer reduction × `d_x` × `d_y`.

Numerically critical details:
1. **Two scales per block** (`d_x * d_y`), a fp16×fp16→fp32 product, not one scale.
2. **fp32 accumulation across the block loop** (`sumf` is float).
3. **Pairing**: low nibble of byte `j` pairs with `y[j]`; high nibble with `y[j+16]`.

---

## (c) The real RVV intrinsic sequence

Source: `llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:222-275`. `vl = qk/2 = 16`.

```c
for (ib = 0; ib < nb; ++ib) {
    // load the 16 packed q4 bytes, and TWO 16-lane plain-int8 halves of q8
    vuint8m1_t tx = __riscv_vle8_v_u8m1(x[ib].qs,      vl);   // 16 packed bytes
    vint8m1_t  y0 = __riscv_vle8_v_i8m1(y[ib].qs,      vl);   // q8 lanes  0..15
    vint8m1_t  y1 = __riscv_vle8_v_i8m1(y[ib].qs + 16, vl);   // q8 lanes 16..31

    // nibble unpack of x ONLY  (offset-binary decode)
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);     // low  nibbles
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);     // high nibbles (logical srl)
    vint8m1_t  x_ai = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
    vint8m1_t  x_li = __riscv_vreinterpret_v_u8m1_i8m1(x_l);
    vint8m1_t  v0 = __riscv_vsub_vx_i8m1(x_ai, 8, vl);       // − 8 bias  → [-8,7]
    vint8m1_t  v1 = __riscv_vsub_vx_i8m1(x_li, 8, vl);       // − 8 bias  → [-8,7]

    // asymmetric widening product:  i4(decoded i8) × plain-i8  → i16
    vint16m2_t m1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);          // low  half
    vint16m2_t m2 = __riscv_vwmacc_vv_i16m2(m1, v1, y1, vl);     // + high half (fused)

    // reduce i16 → i32 lane-0
    vint32m1_t z   = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(m2, z, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

    // dual fp16 scale, scalar, fp32 accumulate
    sumf += sumi * GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
}
*s = sumf;
```

`GGML_CPU_FP16_TO_FP32` on this board = `riscv_compute_fp16_to_fp32`
(`simd-mappings.h:95-100`): a **scalar** `_Float16`→`float` cast under `__riscv_zfhmin`
(no `__riscv_zfhmin` ⇒ a 65536-entry lookup table, `simd-mappings.h:140`). **It is NOT a
vector / zvfh operation** — the scale conversion is two scalar `fcvt.s.h` per block. So a
replacement does **not** need vector-zvfh; it needs a *scalar fp16→fp32* per block.

Intrinsic inventory the real kernel uses: `vle8 (u8 + i8×2)`, `vand.vx`, `vsrl.vx`,
`vreinterpret`, `vsub.vx`, `vwmul.vv.i16`, `vwmacc.vv.i16`, `vmv.v.x.i32`,
`vwredsum.vs.i16→i32`, `vmv.x.s.i32`, scalar `fcvt.s.h` ×2, scalar fmul ×2, scalar fadd.

---

## (d) What WE already express (with file/test citations)

Our packed-i4 path lives in the typed RVV body op
`tcrv_rvv.packed_i4_nibble_unpack_product` plus `standalone_reduce` + `dequantize`.

| Concern | Our artifact | Status vs ggml |
|---|---|---|
| Packed-i4 unit-stride load | `tcrv_rvv.load … -> vector<i8,"mf4">`; `packed_storage_load = "unit-stride-vle8-i8mf4-packed-i4x2"` (test `...dequantize-f32-packed-i4.mlir:77`) | ✅ load shape matches (vle8 of packed bytes) |
| Nibble unpack + widening product | `tcrv_rvv.packed_i4_nibble_unpack_product`, lowered in `lib/Conversion/RVV/RVVToEmitC.cpp:3543-3649` to `vsll(4)/vwmul/vsra(8)/vsra(4)/vwmacc` | ⚠️ structure present but **WRONG convention + WRONG operand shape** — see gap (g1)(g2) |
| Reduce i16→i32 | `tcrv_rvv.standalone_reduce {kind="signed_widening_reduce_add"}` → `vwredsum_vs_i16mf2_i32m1` (`RVVToEmitC.cpp`; test `...packed-i4.mlir:276`) | ✅ matches ggml's `vwredsum` reduce |
| i32→f32 dequant by **one** runtime scale | `tcrv_rvv.dequantize {dequant_relation="signed-i32m1-to-f32m1-scale-f32"}`; epilogue `(float)sum * scale` (`RVVToEmitC.cpp:2071-2100`, `emitDequantEpilogue` 2609+) | ⚠️ **single** scalar scale, not `d_x*d_y`; no per-block loop — see (g3)(g4) |
| Plain i8×i8→i16 widening product | `tcrv_rvv.widening_product {product_relation="signed-i8mf4xi8mf4-to-i16mf2"}` (`RVVToEmitC.cpp:3493+`; tests `...widening-product-reduce-*.mlir`) | ✅ **reusable** for the q8 side once we have one-sided unpack |

Files:
- Op semantics/verifier: `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp:1046,1925` (only `signed-i32m1-to-f32m1-scale-f32`).
- Lowering: `lib/Conversion/RVV/RVVToEmitC.cpp`
  (`emitPackedI4NibbleUnpackProduct` 3560-3649; `emitWideningProduct` 3493+;
  `emitDequantEpilogue` 2609+; product-reduce-dequant body 1526+).
- Dialect tests: `test/Dialect/RVV/packed-i4-nibble-unpack-product-dataflow.mlir`
  (op accepts `i8 mf4 × i8 mf4 → i16 mf2`, **both** operands packed-i4).
- Conversion tests: `test/Conversion/RVV/rvv-to-emitc-widening-product-reduce-dequant-clamp-packed-i4.mlir`,
  `...-dequantize-packed-i4.mlir`.
- E2E (HW-validated, tol 1e-05): `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`
  (and `...dequant-clamp-f32-packed-i4.mlir`).

**Exact lowering of our op** (`RVVToEmitC.cpp:3614-3646`), for contrast with ggml:
```
lhs_low  = vsll_vx_i8mf4(lhs, 4)     rhs_low  = vsll_vx_i8mf4(rhs, 4)   // BOTH operands
low_p    = vwmul_vv_i16mf2(lhs_low, rhs_low)
product  = vsra_vx_i16mf2(low_p, 8)
lhs_high = vsra_vx_i8mf4(lhs, 4)     rhs_high = vsra_vx_i8mf4(rhs, 4)   // BOTH operands
product  = vwmacc_vv_i16mf2(product, lhs_high, rhs_high)
```
i.e. our op unpacks **both** sources as packed-i4 (`product_relation =
"signed-i8mf4xi8mf4-to-i16mf2"`, dialect test line 24), and decodes nibbles as
**two's-complement i4** via `vsll/vsra`.

---

## (e) Precise GAP LIST (to a numerically-exact, deployable replacement)

**(g1) Operand asymmetry: i4×i8, not i4×i4.** ggml unpacks **only x** (the q4 weights);
**y stays plain int8** loaded as two full 16-lane vectors, product = `vwmul(v0,y0)` +
`vwmacc(v1,y1)`. Our `packed_i4_nibble_unpack_product` unpacks **both** operands. We have
**no asymmetric "decoded-i4 × plain-i8 → i16" product op**. Need: either a one-sided
nibble-unpack op producing i8 lanes that feeds the *existing* signed widening-product
against a plain-i8 stream, or a dedicated `packed_i4 × i8 → i16` op.

**(g2) Nibble decode convention is incompatible (different bijection, not free swap).**
- ggml = **offset-binary** `nibble − 8`: `0→−8, 8→0, 15→+7`.
- ours = **two's-complement i4** (`vsll(4)` then arithmetic `vsra`): `0→0, 8→−8, 15→−1`.

Same *range* [−8,7], **different mapping** → fed identical bytes they yield **different
dot products**. Need a structural op for the `−8` bias (mask+sub, or XOR each byte with
`0x88` to convert ggml's encoding into our two's-complement lanes before reusing the
existing sign-extend path).

**(g3) Per-block dual fp16 scale pair.** ggml multiplies each block's i32 reduction by
`d_x * d_y` (two fp16→fp32 scalar casts then a float product). Our dequant has **one**
runtime `float scale` (`signed-i32m1-to-f32m1-scale-f32`, hard-verified at
`RVVDialect.cpp:1046,1925`). Need: a two-scale dequant (load `d_x`,`d_y` fp16 from block
heads, scalar `fcvt.s.h` ×2, multiply). **No vector zvfh required** — the scale path is
scalar (`simd-mappings.h:95-100`).

**(g4) Block-strided QK=32 iteration over AoS.** ggml loops `nb` blocks, each a *separate*
fp16-headed struct (stride 18 B / 34 B, quants at offset +2), accumulating into a single
`float sumf`. Our IR models flat SoA `runtime_abi_value` buffers (`lhs`,`rhs`,`acc`,
`scale`,`out`,`n`) with a single global reduction + one final dequant. We have **no**:
(i) AoS block stride / inner-block offset addressing, (ii) extracting the inline per-block
fp16 head, (iii) **fp32 scalar accumulation of `scale·reduction` across the block loop**.
This is the largest missing piece — it is a *block-loop outer structure*, not an op.

**(g4b) Different reduction anchor / LMUL (numerics-neutral, but a real structural gap for a
performant drop-in).** ggml anchors the whole half-block at `i8m1` (vl=16, one register) →
`i16m2` → **immediate** `vwredsum` → scalar, carrying **no i32 vector accumulator**. Our
pipeline anchors at `i32m1`, which forces the i8 source to `mf4` (only 4 lanes/chunk at
VLEN=128) and carries an `i32m1` vector accumulator across chunks. Same result, but a
different reduction structure — reinforces (g4): a different reduction *anchor*, not just a
different scale count.

**(g5) Low/high half pairing with the q8 split.** ggml pairs low nibbles with `y[0..15]`
and high nibbles with `y[16..31]` (two `vle8` of the q8 block). Our single packed source
has no notion of "the other operand is split into two halves indexed by nibble position".
Folds into (g1) but must be honored for byte-exactness.

**(g6) ABI / call signature mismatch.** Replacement must match ggml's
`void ggml_vec_dot_q4_0_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx,
const void *vy, size_t by, int nrc)` and write `*s` (a single fp32). Our emitted ABI is
`(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out,
size_t n)` (header line in the e2e test, `RVVToEmitC.cpp` ABI emission). Need the AoS
`void*` block-pointer signature + `*s` scalar store.

**Already reusable (no gap):** the i16→i32 `vwredsum` reduce; the signed i8×i8→i16
`vwmul`/`vwmacc` widening-product machinery (for the q8 side); unit-stride `vle8` packed
load; the scalar-lane-0 extract.

---

## (f) Assessment & cleanest first increment

**Gap size: LARGE** for a numerically-exact, *deployable* drop-in. The reusable core is
real (reduce + i8×i8 widening product + vle8 load), but **every concrete binding differs**:
operand asymmetry (g1), nibble convention (g2), dual fp16 scales (g3), AoS block loop with
fp32 accumulation (g4), q8 half-split pairing (g5), and the ABI (g6). Our HW-validated
packed-i4 path is *adjacent* but computes a different function (i4×i4, two's-complement,
single scale, no block loop), so it is not a swap-in.

**Cleanest first increment (correctness-first, decompose the fused op):**
1. Introduce a **standalone one-sided nibble-unpack** op: packed-i4 byte → i8 lanes in
   [−8,7] using ggml's offset-binary decode (`& 0x0F`/`>> 4` then `− 8`, **or** XOR `0x88`
   then reuse the existing sign-extend). Output: plain signed i8 mf4/m1 lanes.
2. Feed those i8 lanes + a **plain-i8** q8 stream into the **existing** signed widening
   product (`vwmul`/`vwmacc`) + `vwredsum` reduce → one block's i32 dot. This yields a
   **byte-exact single-block i4×i8 integer dot** reusing proven machinery, validated
   against `ggml_vec_dot_q4_0_q8_0_generic` for one block (n=32).

That closes (g1),(g2),(g5) and proves the integer core. The **second, larger** increment
is the AoS block loop + dual-fp16-scale fp32 accumulation + `*s` ABI ((g3),(g4),(g6)) —
the structural work that turns the single-block kernel into a deployable replacement.

## Caveats / Not Found
- Did not exhaustively confirm there is no asymmetric i4×i8 op elsewhere in the plugin
  (`RVVContractionSelectedBodyRealizationOwner.cpp` and `RVVDialectWideningOps.cpp` were
  grepped; all product relations found are symmetric `i8mf4×i8mf4` or `i16×i16`). If a
  future asymmetric op is added, (g1) shrinks.
- Performance claims out of scope here: the existing packed-i4 path is recorded as
  `same-target-packed-i4-no-win` (speedup 0.895..1.027) in the e2e fixture — i.e. even the
  i4×i4 variant is not yet a measured win, which is relevant context for (f) but is a
  separate N3 concern.
