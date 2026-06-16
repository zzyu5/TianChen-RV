# INC-23 — q4_K K4a: the q4_K INTEGER CORE (byte-exact vs ggml _generic)

q4_K is the most-used modern K-quant (the dominant weight quant in `*_K_M` mixes).
K4a is the INTEGER CORE ONLY, split from the fp32 fold for a reliable dispatch:

- **(1) the 6-bit scale/min bit-dance** — the `get_scale_min_k4` / `utmp[4]`/`kmask`
  cross-byte decode that splits the 12 packed `scales[12]` bytes into 8 6-bit
  scales + 8 6-bit mins. **The hardest NEW piece.** Emitted as STRUCTURED scalar
  emitc (`emitc.bitwise_and/_or/_left_shift/_right_shift`), NOT a raw string.
- **(2) the per-sub-block 4-bit dot scaled into the i32 state** — the plain 4-bit
  nibble unpack (unsigned [0,15], NO bias) + the per-sub-block UINT6-scaled
  widening dot into the 8-lane `aux32` accumulator.

**Byte-exact vs ggml `ggml_vec_dot_q4_K_q8_K_generic`'s INTEGER state** (the
`aux32[8]` accumulator + the decoded `scales`/`mins`) BEFORE the fp32 d/dmin fold.

**K4b (NOT built here)**: the fp32 two-level fold (`d=fp16(x.d)*y.d`,
`sums[l]+=d*aux32[l]`, the sequential horizontal sum), the `min` term
(`sumf -= dmin * Σ_j bsums[j]*mins[j/2]`), and the ggml ABI / `*s` store.

## What this proves

The compiler GENUINELY compiles (not string-pastes) a byte-exact, structurally-
emitted reproduction of the q4_K integer core — including the awkward 6-bit
cross-byte scale/min bit-dance the q6_K scoping doc flagged as q4_K's
distinguishing cost — as fully STRUCTURED emitc (raw()=0). This is the first
super-block K-quant *with a packed-scale decode* (q6_K's scales were a direct
int8 load); it reuses q6_K's proven super-block machinery and adds only the new
unpack + the bit-dance.

## The op + lowering

- **New op**: `tcrv_rvv.q4_k_q8_k_aux_partial` (mirrors q6_K's K1 aux32 op).
  - Operands: `vx` (const uint8_t*, block_q4_K), `vy` (const uint8_t*, block_q8_K),
    `aux32_output` (int32_t*, the 8-lane integer-state), `scalemin_output`
    (uint8_t*, the 16 decoded bytes = 8 scales then 8 mins), `n` (index), `vl`.
  - Attrs (I4 mirror): `kind="ggml_q4_k_q8_k_aux_partial"`,
    `scale_model="per-sub-block-uint6-scale-i32-domain"`, `qk=256`, `sub_block=32`,
    `weight_block_stride=144`, `activation_block_stride=292`,
    `weight_scales_byte_offset=4`, `weight_qs_byte_offset=16`,
    `activation_quant_byte_offset=4`.
  - Fail-closed verifier (I7) pins all facts + the two output C types.
- **Lowering** (`emitQ4_KQ8_KAux32Partial`, a SEPARATE core from q6_K's
  `emitQ6_KSuperBlockAux32Core` — different unpack, no qh, no -32 bias, uint
  scale; so q6_K's emitted bytes stay byte-identical):
  - (A) plain 4-bit unpack into `int8_t aux8[256]` (4 chunks of 64: low nibble
    `vand 0x0F` → a[0..31], high nibble `vsrl 4` → a[32..63], NO bias);
  - (B) the bit-dance: 3 uint32 word loads from `xb+4` → the kmask shuffle as
    scalar `emitc.bitwise_*` into `uint32_t utmp[4]` → the 16 bytes
    `[scales[0..7], mins[0..7]]` (`&utmp[0]` = scales contiguous with `&utmp[2]`
    = mins, the same type-pun `_generic` does) vse8'd to `scalemin_out + ib*16`;
  - (C) the nested 8-sub-block loop: the per-sub-block UINT8 ZERO-EXTENDED scale
    (`scales[js]`, unlike q6_K's sign-extended int8) applied in the i32 domain via
    4 quarters of 8 (`vwmul i8×i8→i16` then `vwmacc.vx i32 += scale*i16` into the
    8-lane e32m2 `aux32`), then `vse32` of `aux32[8]` to `aux32_out + ib*8`.

## How it reuses q6_K's super-block machinery

Same SHAPE as `emitQ6_KSuperBlockAux32Core` / `emitQ6_KQ8_KAux32Partial`: the
outer super-block `emitc.for` over `nb=n/256`, the `int8_t aux8[256]` scratch +
`&aux8[0]` base, per-super-block address arithmetic, the e8m2 unpack strips, the
8-lane e32m2 `aux32` accumulator reset per super-block, the nested sub-block
`emitc.for`, the `vwmul→vwmacc.vx` integer dot, the `vse32` store. q4_K diverges
ONLY in: the unpack (plain 4-bit, no qh, no -32 bias, 8×32 sub-blocks vs 16×16),
the scale extraction (the bit-dance + a uint6 zero-extended scale vs a direct
int8 sign-extended load), and the second `scalemin` output.

## Byte-exact validation (ssh rvv, -march=rv64gcv, clang 18)

`ssh_rvv_stdout.txt`:
- **POSITIVE**: `aux32[8]` + `scales[8]` + `mins[8]` EXACTLY equal (i32 / byte)
  to ggml `_generic` over **4013 super-blocks** — 6 named edge cases (q4 all
  0/15, scales packed 0x00/0xFF/0x3F/0xAA + a distinct-per-byte case exercising
  the cross-byte shuffle, q8 ±127/0) + 4000 random + 7 multi-super-block
  (NB=2..8, exercising the outer loop + the per-block `out+ib*8` / `out+ib*16`
  indexing; scales/mins differ per block). **0 failures.**
- **NEGATIVE CONTROL**: sign-extending the 6-bit scale (the q6_K bug) → **4000
  failures** — the byte-exact check discriminates the uint6 scale model (the 13
  non-diverging cases are where scales are small/zero, so sign==zero extension).

The `aux32` byte-exactness transitively validates the SCALES (a wrong scale →
wrong aux32); exposing `scales`+`mins` directly makes the MINS half of the
bit-dance a LIVE, validated computation (mins are otherwise dead until K4b's min
term) and catches a mis-unpacked scale even on the `q4 all 0` edge (where the
product is zero and aux32 alone would not).

## raw()=0 / structured proof

The emitted kernel (`tcrv_emitted_kernel.cpp`) has ZERO raw strings — every
`verbatim` line is a `// tcrv_emitc.*` step-marker COMMENT, not raw C code. The
6-bit bit-dance is structured scalar emitc:

    uint32_t v64 = v61 >> 6;     uint32_t v65 = v64 & 0x03030303;
    uint32_t v66 = v65 << 4;     uint32_t v67 = v63 >> 4;
    uint32_t v68 = v67 & 0x0f0f0f0f;  uint32_t v69 = v68 | v66;  // utmp[3]
    uint32_t v70 = v61 & 0x3f3f3f3f;                              // utmp[2]
    ...  v10[0]=utmp[0]; v10[1]=utmp[1]; v10[2]=utmp[2]; v10[3]=utmp[3];

i.e. `emitc.bitwise_right_shift/_and/_left_shift/_or` (LLVM-20 EmitC dialect,
same `EmitC_BinaryOp`+`CExpression` class as the `MulOp`/`AddOp` the kernel
already emits) — confirmed to translate cleanly through `mlir-translate
--mlir-to-cpp`. No `raw()` anywhere in the emitter.

## Lit + reds

- 2 new lit tests PASS: `test/Conversion/RVV/rvv-to-emitc-q4-k-q8-k-aux-partial.mlir`
  (structured-emission FileCheck) + `test/Dialect/RVV/q4-k-q8-k-aux-partial-dataflow.mlir`
  (9 cases: accept + 8 fail-closed rejections).
- Full lit: 637 tests, 634 pass, exactly the 3 documented environmental reds
  (`Scripts/rvv-generated-bundle-abi-e2e-*` — the ssh-bundle e2e harness dry-runs,
  not q4_K).
- **Siblings additive by construction**: the recognizer matches
  `GgmlBlockDotQ4KQ8KAux32Op` EXCLUSIVELY, no shared lowering code was edited (the
  dispatch `if` is an additive insertion that touches no existing branch; the new
  emitter is a SEPARATE core, not a generalization of `emitQ6_KSuperBlockAux32Core`).
  q6_K K1 AND K2 emitted C were byte-DIFFED (identical); the other siblings
  (q4_0/q8_0/q4_1 + deferred-wide) are pinned byte-stable by the lit suite (634 pass).

## Honest residuals

- **Word-aligned scale read.** The bit-dance reads the 12 packed `scales` bytes
  as 3 `uint32` words via `(const uint32_t*)(xb+4)`, which assumes 4-byte
  alignment of `scales@4`. This is CORRECT codegen for the validated target and
  real ggml (a 16-aligned `block_q4_K` base × the 144 stride → every `scales`
  lands at offset ≡ 4 mod 16, i.e. 4-aligned) — it passed on the board. It is an
  assumption K4b/deployment inherits; the structured fallback if a future target
  lacks unaligned scalar word loads is 4 byte-loads per word (more nodes, still
  structured, still raw()=0). Not changed now — documented.
- **VLEN≥128-pinned** (like q6_K K1): the e8m2(32) unpack + e8mf2(8) sub-block
  quarters assume VLMAX≥32/≥8 (rv64gcv ⇒ Zvl128b ⇒ VLEN≥128). A VLEN<128
  (zve32x/zve64x) re-strip is a capability item, not built here. The bit-dance is
  scalar (VLEN-independent).

## What's left for K4b

- The fp32 two-level fold: `d = fp16(x.d)*y.d`; `sums[l] += d*aux32[l]` into 8
  fp32 lane-accumulators; the SEQUENTIAL horizontal sum (do NOT reassociate — the
  fp32-ordering byte-exactness risk, as q6_K K2 + the INC-2a FMA red showed).
- The MIN term: `dmin = fp16(x.dmin)*y.d`; `sumf -= dmin * Σ_{j=0..15} bsums[j] *
  mins[j/2]` — sourced from `y[i].bsums` (the q8_K block field @260) using the
  decoded `mins` K4a already exposes. This is the reused q4_1 Family-B min pattern.
- The ggml ABI: the `void ggml_vec_dot_q4_K_q8_K(int n, float *s, ...)` 8-arg
  signature + the `*s` store + the `n%256==0`/`nrc==1` contract.

## Reproduce on the board

    build/bin/tcrv-opt \
      test/Conversion/RVV/rvv-to-emitc-q4-k-q8-k-aux-partial.mlir \
      --tcrv-rvv-lower-to-emitc \
      | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp  > tcrv_emitted_kernel.cpp

    scp tcrv_emitted_kernel.cpp inc23_validate.cpp rvv:~/inc23_q4k_k4a/
    ssh rvv 'cd ~/inc23_q4k_k4a && \
      clang++ -O2 -march=rv64gcv -mabi=lp64d \
        tcrv_emitted_kernel.cpp inc23_validate.cpp -o inc23_validate && \
      ./inc23_validate'
    # negative control: add -DNEGCTRL (must FAIL, proving the check discriminates)
