# INC-32 — iq4_xs × q8_K byte-exact; the CODEBOOK class's SUPER-BLOCK rung

13th ggml dot kernel: `ggml_vec_dot_iq4_xs_q8_K` — the super-block variant of iq4_nl.
It is the FIRST kernel that COMBINES two already-covered structural classes: the
CODEBOOK class (the 16-entry non-linear `kvalues_iq4nl[16]` nibble lookup, opened by
iq4_nl/INC-30) and the K-quant super-block machinery (QK_K=256, the q8_K activation,
the q4_K-style 6-bit scale, opened by q4_K/q6_K). New op
`tcrv_rvv.iq4_xs_q8_k_block_dot`, STRUCTURED (raw()=0).

## (1) The op + lowering + REUSE + files

**Op** `GgmlBlockDotIQ4XSQ8KOp` (`tcrv_rvv.iq4_xs_q8_k_block_dot`):
- ABI: `(weight vx, activation vy, output *s, n, vl)`, 4 runtime ABI operands +
  the vl token → one i32 m1 vector result. `vx`/`vy` = `const uint8_t *`,
  `*s` = `float *` (fail-closed verifier, I7).
- super-block format attrs (I4 mirror): `qk=256`, `sub_block=32`,
  `weight_block_stride=136` (sizeof block_iq4_xs), `activation_block_stride=292`
  (sizeof block_q8_K), `weight_d_byte_offset=0`, `weight_scales_h_byte_offset=2`,
  `weight_scales_l_byte_offset=4`, `weight_qs_byte_offset=8`,
  `activation_d_byte_offset=0`, `activation_quant_byte_offset=4`.
- the iq4_nl codebook as a `DenseI8ArrayAttr:$codebook` of EXACTLY 16 int8 entries
  (the verifier pins size==16; the SAME `kvalues_iq4nl[16]` table iq4_nl carries).

**Lowering** `emitIQ4XSQ8KBlockDot` (a SEPARATE, additive emitter; the inner codebook
logic is COPIED from iq4_nl, NOT factored, so sibling bytes stay identical):
- outer super-block loop `for ibl in nb=n/256`; per-super-block `d4d8 = (float)x.d * y.d`
  (the fp16 weight d via the `(float)*(const _Float16 *)` seam × the fp32 q8_K d via a
  plain float load), computed ONCE.
- a FLAT unrolled loop over the 8 sub-blocks (j=0..7). Each sub-block:
  - the SIGNED 6-bit scale extracted by the q4_K-style cross-byte bit dance in its
    CLOSED form (scalar STRUCTURED emitc shift/and/or; verified byte-exact equal to
    ggml's progressive `h>>=4` over 16M sub-block extractions): see (5).
  - `d1 = d4d8 * (float)(ls - 32)` — a SEPARATE op (mirrors `_generic`'s named `d1`;
    the (ls-32) round + the d4d8 multiply stay separate from the sumi multiply so NO
    FMA fuses them under -ffp-contract).
  - the per-sub-block codebook integer dot = **iq4_nl's elided core REUSED verbatim**:
    `vsetvl_e8m1(16)` over the 16 nibble bytes at `qs+j*16`, `vand 0x0F`/`vsrl 0x04`
    → two u8 index lanes → `vrgather_vv_i8m1` through the broadcast table → the SAME
    `emitOffsetBinaryProductFromDecodedValue` chain (`vwmul` low ↔ q8[0..15],
    `vwmacc` + high ↔ q8[16..31]) → `vwredsum` (seed 0) → scalar `sumi`. iq4_nl's
    `sumi` already equals `_generic`'s `sumi1 + sumi2`.
  - `sumf = sumf + d1 * (float)sumi` — ONE `emitc.expression` (fuses the SAME FMA
    ggml does under -ffp-contract=on/default), invoked in STRICT ascending
    (super-block, sub-block) order.
- `*s = sumf` (structured subscript + assign).
- NO min term (symmetric, like q6_K). The codebook gather pins the **m1 anchor**
  (VLMAX≥16 to index all 16 entries) — a Zvl128b N1 legality fact.

**REUSE:** iq4_nl's codebook gather + `emitOffsetBinaryProductFromDecodedValue` (the
product/reduce chain, shared verbatim); q4_K/q6_K's super-block structure (the q8_K
activation, the 6-bit scale bit dance, the d4d8 = fp16(x.d)×fp32(y.d) form).

**Files:**
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `GgmlBlockDotIQ4XSQ8KOp`.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — `GgmlBlockDotIQ4XSQ8KOp::verify()`.
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `isIQ4XSQ8KBlockDotBody` + dispatch branch +
  `emitIQ4XSQ8KBlockDot`.
- `test/Conversion/RVV/rvv-to-emitc-iq4-xs-q8-k-block-dot.mlir` (lowering FileCheck).
- `test/Dialect/RVV/iq4-xs-q8-k-block-dot-dataflow.mlir` (op accept + 6 fail-closed
  negative dataflow checks).

## (2) Byte-exact vs ggml iq4_xs — 0 failures

ssh rvv (riscv64, rv64gcv+zvfh, VLEN=128, clang). The compiler-emitted kernel's `*s`
is BIT-FOR-BIT equal (memcmp of the float bits) to ggml's **VERBATIM `_generic`**
(`quants.c:1232-1276`, the `ib+=2` pair loop with progressive `h>>=4`, named d1/d2,
qs+=16, q8+=32):
- **-ffp-contract=off**: 3070 cases, **0 failures**, _generic delta 0/3070.
- **-ffp-contract=fast**: 3070 cases, **0 failures**, _generic delta 0/3070.
- n ∈ {256, 512, … 16384} (multiples of 256), 300 random reps + edge cases.
- Edge cases: scale extreme ls=−32 (scales=0) and ls=+31 (scales=0xFF…), the
  marching-nibble pattern that exercises ALL 16 codebook entries, the
  scale-extremes+all-16 combination, all crossed with q8 = +127 and q8 = ±127.

The fold shape matches `_generic` exactly (d1 separate, then one FMA-able
`sumf += d1*sumi`), so the FMA forms identically under =fast → delta=0 at both modes
(not just =off). See `ssh_rvv_byte_exact_stdout.txt`.

## (3) raw()=0 + STRUCTURED

`raw()` actual-call count in the compiler-emitted kernel: **0** (every value is an
emitc node; the only opaque pieces are the sanctioned `(float)*(const _Float16 *)` d
read and the `__riscv_*` intrinsic spellings — both structured CallOpaque). `nm` on
the board object shows ONE exported symbol
(`tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_…`) and ONE external ref
(`__extendhfsf2`, the compiler-rt fp16→fp32 builtin) — no kernel fallback, the RVV
intrinsics are inlined. See `structured_proof.txt`.

## (4) lit + reds

- 2 new lit tests PASS (the lowering FileCheck + the op-accept/6-negative dataflow).
- Full lit: **654 discovered, 651 pass, exactly the 3 documented pre-existing reds**
  (`…computed-masked-strided-input-widening-dot-reduce-add-dry-run` ×2 +
  `…abi-e2e-self-test`, all `Scripts/rvv-generated-bundle-abi-e2e-*`, unrelated to the
  dialect/conversion — documented as the same 3 reds since inc30/inc31). +2 over
  inc31's 652 = my two new tests.
- Sibling byte-identity: iq4_nl (inc30) AND q6_K (inc12) emitted bodies regenerated
  and diffed byte-for-byte UNCHANGED → the change is purely additive.
- Clean full rebuild green.

## (5) The signed 6-bit scale extraction

ggml (`quants.c:1252-1254`) extracts the scale progressively per pair of sub-blocks:
```c
ls1 = (scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
ls2 = (scales_l[ib/2] >> 4) | ((h << 2) & 0x30);
h >>= 4;   // h = scales_h, consumed 4 bits per pair
```
then `ls_j − 32` (signed). The lowering uses the equivalent CLOSED form (no mutating
`h`), per sub-block j=0..7:
```
low  = (scales_l[j/2] >> (4*(j%2))) & 0xf
high = ((scales_h >> (2*j)) & 0x3) << 4
ls   = low | high           // unsigned 6-bit, [0,63]
d1   = d4d8 * (float)(ls - 32)
```
Verified byte-exact equal to ggml's progressive form over 2M random
`scales_l[4]`+`scales_h` inputs (16M sub-block extractions, 0 mismatches), and again
re-proved on the board (the harness reference is ggml's progressive form, and the
WRONG-scale negative control — bias −31 instead of −32 — diverges, proving the
extraction is load-bearing).

## (6) Remaining IQ tail (iq2_xxs/xs/s, iq3_xxs/s, iq1_s/m)

iq4_xs was the LAST kernel reachable by the simple "16-entry per-nibble gather"
codebook mechanism. The remaining IQ kernels are a DIFFERENT, deeper codebook class —
they index large packed GRID codebooks with separate sign planes, not a 16-entry
per-nibble lookup:
- `iq2_xxs`/`iq2_xs`/`iq2_s` — 256-entry `iq2xxs_grid`/`iq2xs_grid`/`iq2s_grid`
  (each entry an 8-element 2-bit pattern packed in a uint64) + `ksigns_iq2xs`/`kmask`
  sign tables; 2-bit super-block scales.
- `iq3_xxs`/`iq3_s` — 256/512-entry `iq3xxs_grid`/`iq3s_grid` + sign planes; the
  qs+sign+scale layout differs again.
- `iq1_s`/`iq1_m` — 2048-entry `iq1_grid` ternary codebook + a per-block delta and a
  1.5-bit scheme (iq1_m adds per-sub-block scale nibbles).

These are a NEW mechanism (grid-table gather over uint64 packed entries + a sign-flip
plane + a different scale dance), NOT an extension of the 16-entry per-nibble gather —
each is its own structural increment (a niche tail; rarely the dominant weight quant).
The current 13 kernels cover every COMMON ggml dot class (legacy block-quant
q4_0..q5_1, K-quant super-block q2_K..q6_K, the 16-entry codebook iq4_nl/iq4_xs, and
fp4 mxfp4).

## Reproduce
```
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-iq4-xs-q8-k-block-dot.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > tcrv_emitted_kernel.cpp
scp tcrv_emitted_kernel.cpp inc32_validate.cpp rvv:~/inc32_iq4xs/
ssh rvv 'cd ~/inc32_iq4xs && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  -c tcrv_emitted_kernel.cpp -o k_off.o  && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  inc32_validate.cpp k_off.o  -o val_off  && ./val_off && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast -c tcrv_emitted_kernel.cpp -o k_fast.o && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast inc32_validate.cpp k_fast.o -o val_fast && ./val_fast'
```

## Files in this artifact dir
- `iq4_xs_input.mlir` — the typed-body input (the conversion test module).
- `iq4_xs_emitted.cpp` / `tcrv_emitted_kernel.cpp` — the compiler-emitted kernel C.
- `inc32_validate.cpp` — the byte-exact HW harness (verbatim `_generic` reference +
  3 negative controls).
- `ssh_rvv_byte_exact_stdout.txt` — the raw board stdout (off + fast + the nm proof).
- `structured_proof.txt` — the raw()=0 / structured-emission proof.
