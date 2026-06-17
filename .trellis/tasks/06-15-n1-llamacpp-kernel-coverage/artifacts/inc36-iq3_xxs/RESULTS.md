# INC-36 — iq3_xxs × q8_K byte-exact; the iq3 GRID-of-4 variant (17th dot kernel)

17th ggml dot kernel: `ggml_vec_dot_iq3_xxs_q8_K` — the iq3 GRID variant of the deep IQ
tail. It REUSES the GRID-codebook + SIGN-plane mechanism established by iq2_xxs/iq2_xs/
iq2_s (a packed GRID codebook indexed by a weight byte, the `ksigns_iq2xs[128]`/`kmask`
SIGN plane, the packed-aux 4-bit integer-domain scale) with ONE load-bearing structural
delta: the IQ3 grid is **256 entries of FOUR int8** (`iq3xxs_grid[256]`, a uint32 table —
HALF the lane width of iq2_xxs's uint64 grid-of-8). New op
`tcrv_rvv.iq3_xxs_q8_k_block_dot`, STRUCTURED (raw()=0).

## (1) The op + the grid-of-4 + packed-aux scale/sign lowering + REUSE + files

**Op** `GgmlBlockDotIQ3XXSQ8KOp` (`tcrv_rvv.iq3_xxs_q8_k_block_dot`):
- ABI: `(weight vx, activation vy, output *s, n, vl)`, 4 runtime ABI operands + the vl
  token → one i32 m1 vector result. `vx`/`vy` = `const uint8_t *`, `*s` = `float *`
  (fail-closed verifier, I7).
- super-block format attrs (I4 mirror): `qk=256`, `sub_block=32`,
  `weight_block_stride=98` (sizeof block_iq3_xxs = {fp16 d; uint8 qs[96]}),
  `activation_block_stride=292` (sizeof block_q8_K), `weight_d_byte_offset=0`,
  `weight_qs_byte_offset=2` (the 64 grid index bytes `q3`),
  `weight_gas_byte_offset=66` (= qs + QK_K/4; the 32 aux bytes `gas`),
  `activation_d_byte_offset=0`, `activation_quant_byte_offset=4`.
- the two GRID-codebook tables as typed array attrs: `DenseI32ArrayAttr:$grid` of EXACTLY
  256 int32 entries (the packed uint32 `iq3xxs_grid[256]`, each = 4 int8; **i32 not i64
  because every entry < 0x80000000 (max 0x3e341c04) fits a positive int32 losslessly**)
  and `DenseI32ArrayAttr:$ksigns` of EXACTLY 128 int32 entries (the sign plane
  `ksigns_iq2xs[128]`, **i32 because ksigns values reach 255**). The verifier pins both
  sizes. `kmask` ({1<<j}) is the inline const, not a table.

**Lowering** `emitIQ3XXSQ8KBlockDot` (a SEPARATE, additive emitter; siblings byte-untouched):
- the GRID-of-4 codebook emitted ONCE as `static const uint32_t tcrv_iq3xxs_grid[256] =
  {...}` (ggml's exact uint32 hex literals with the `U` suffix), read as bytes through a
  `const int8_t *grid_i8 = (const int8_t *)tcrv_iq3xxs_grid;` cast — exactly ggml's
  `(const uint8_t *)(iq3xxs_grid + idx)` method. Each entry is **4 bytes**, so `grid_i8 +
  idx*4` addresses the 4 int8 values of entry idx (vs iq2_xxs's `idx*8`).
- the SIGN plane emitted ONCE as `static const uint8_t tcrv_iq3xxs_ksigns[128] = {...}`
  (**REUSED verbatim from iq2_xxs**); kmask as `static const uint8_t tcrv_iq3xxs_kmask[8]
  = {1,2,4,8,16,32,64,128};`, split-loaded ONCE into `kmaskLo = vle8(kmask, 4)` ({1,2,4,8}
  for grid1 lanes 0..3) and `kmaskHi = vle8(kmask+4, 4)` ({16,32,64,128} for grid2 lanes
  4..7).
- outer super-block loop `for ibl in nb=n/256`; per-super-block `d = (float)x.d * y.d`
  (fp16 weight d via `(float)*(const _Float16 *)` × fp32 q8_K d), `int32_t bsum = 0`.
- `const uint8_t *q3 = xb + 2` (64 grid index bytes), `const uint8_t *gas = xb + 66`
  (32 aux bytes) — the iq3_xxs **SEPARATE qs[96] regions** (vs iq2_xxs's interleaved aux
  pair).
- a FLAT unrolled loop over the 8 sub-blocks (ib32=0..7). Each sub-block:
  - `aux32` (ONE uint32) **reassembled from 4 little-endian byte loads** of `gas + ib32*4`
    in the **uint32_t domain** — alignment-safe (the gas region starts at block byte 66, so
    the aux words are 2-aligned, NOT 4-aligned: no `*(uint32_t*)`), and **unsigned** so the
    `>>` is logical (the signed-shift bug iq2_xxs hardware-bisected).
  - `int ls = 2*(aux32>>28)+1` (the per-sub-block 4-bit scale, [1,31]) — the iq2_xxs
    packed-aux high-nibble scale, REUSED.
  - a loop over the 4 sign groups (l=0..3). Each group: the sign byte `signs =
    tcrv_iq3xxs_ksigns[(aux32>>7*l)&127]` (the ksigns plane, REUSED), and the TWO grid
    indices `qg[2l+0]`, `qg[2l+1]` (qg = q3 + ib32*8). The group's 8 elements are decoded
    as **TWO 4-lane passes** (the grid-of-4 delta, mirroring `_generic`'s literal inner
    loop): pass A = grid1 (idx 2l+0) vs q8[0..3] with kmaskLo; pass B = grid2 (idx 2l+1)
    vs q8[4..7] with kmaskHi. Each pass: indexed `vle8(4)` over `grid_i8 + idx*4`, the
    SIGN-plane application — broadcast `signs` (`vmv_v_x_u8m1`) → `vand` kmask → `vmsne 0`
    → mask → `vmerge(grid, vneg(grid), mask)` — then `vwmul_vv_i16m2(grid_signed, q8)` →
    i16 product (each lane ≤ 62*127=7874 < 32767), and a **chained `vwredsum_vs_i16m2_i32m1`**
    seeded from the running i32 reduction (integer add is order-free, so chaining across the
    8 passes is byte-exact). q8 advances 8 per group.
  - after 4 groups, `vmv_x_s_i32m1_i32` extracts the sub-block `sumi`; `bsum += sumi * ls`
    (integer domain).
- the per-super-block fold `sumf = sumf + d * (float)bsum` is ONE `emitc.expression` (the
  SAME FMA `_generic` fuses under -ffp-contract=on/default), invoked in STRICT ascending
  super-block order.
- `*s = 0.25f * sumf` (the iq3_xxs trailing **1/4** factor — DIFFERENT from iq2_xxs's
  0.125f — a SEPARATE statement OUTSIDE the accumulate expression).

The grid lookup is an indexed `vle8(4)` over a pointer (`grid_i8 + idx*4`), the operative
vector width is **4** (one grid-of-4 entry), so there is **NO vrgather table-index legality
fact and NO m1 table anchor** (the 1024-byte table cannot broadcast). The verifier does NOT
carry a (false) m1-table rationale.

**REUSE (the task mandate "REUSE the grid+sign mechanism"):** the entire iq2_xxs group
body is reused at width 4 — the same `vle8` grid load + `vmv_v_x_u8m1`/`vand`/`vmsne`/
`vneg`/`vmerge` sign apply + `vwmul`/`vwredsum` reduce chain (the `gridOf4Pass` helper IS
iq2_xxs's group body parameterized by width 4 + a per-pass kmask half). The `ksigns_iq2xs`
sign plane, the packed-aux scale `2*(aux>>28)+1`, the `(aux>>7l)&127` sign selectors, the
q6_K-style super-block structure (q8_K activation, integer `bsum` fold, per-super-block
fp32 fold), and the fp16-d × fp32-d scale form are all REUSED verbatim.

**Files:**
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `GgmlBlockDotIQ3XXSQ8KOp`.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — `GgmlBlockDotIQ3XXSQ8KOp::verify()`.
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `isIQ3XXSQ8KBlockDotBody` + dispatch branch +
  `emitIQ3XXSQ8KBlockDot` (+ the `gridOf4Pass` reuse helper).
- `test/Conversion/RVV/rvv-to-emitc-iq3-xxs-q8-k-block-dot.mlir` (lowering FileCheck).
- `test/Dialect/RVV/iq3-xxs-q8-k-block-dot-dataflow.mlir` (op accept + 7 fail-closed
  negatives: wrong kind / weight stride / gas offset / grid size / ksigns size / scale
  model / output ctype).

## (2) Byte-exact vs ggml iq3_xxs — 0 failures

ssh rvv (riscv64, rv64gcv, VLEN=128, clang 18.1.3). The compiler-emitted kernel's `*s` is
BIT-FOR-BIT equal (memcmp of the float bits) to ggml's **VERBATIM `_generic`**
(`quants.c:999-1041`: aux32 from gas, ls = 2*(aux32>>28)+1, grid1/grid2 =
(uint8_t*)(iq3xxs_grid + q3[2l+0/1]), signs = ksigns_iq2xs[(aux32>>7l)&127], sumi +=
grid1[j]*q8[j]*(signs&kmask[j]?-1:1) + grid2[j]*q8[j+4]*(signs&kmask[j+4]?-1:1), bsum +=
sumi*ls, sumf += d*bsum, *s = 0.25f*sumf):
- **-ffp-contract=off**: 2502 cases, **0 failures**, _generic delta 0/2502.
- **-ffp-contract=fast**: 2502 cases, **0 failures**, _generic delta 0/2502.
- n ∈ {256, 512, … 16384} (multiples of 256), 300 random reps + edge cases.
- Edge cases: grid-index range (marching the 64 grid index bytes over the full [0,255]
  range + marching sign selectors over [0,127]), the scale extremes (scale nibble 0 →
  ls=1, and 15 → ls=31), q8 = +127 and q8 = ±127.

Byte-exact on the FIRST hardware run at both fp-contract modes — the uint32-domain aux
reassembly (carried over from the iq2_xxs hardware-bisected signed-shift fix) and the
two-4-lane-pass pairing of the sign group were correct by construction.

**Negative controls — the grid/signs/scale are LIVE, load-bearing:** OUR kernel (real
tables) vs a WRONG-table reference DIVERGES for all three (each at both fp-contract modes):
- **wrong-grid** (every grid byte +1): ours=-12981.77 vs wrong=-13293.42 → DIVERGES.
- **wrong-signs** (ksigns sign bit cleared): ours=-12981.77 vs wrong=-18303.68 → DIVERGES.
- **wrong-scale** (drop the `+1` in `2*(aux>>28)+1`): ours=-12981.77 vs wrong=-12015.92 →
  DIVERGES.

**COMPILER-KNOB control — the attr is the live knob → const → HW** (the stronger N1/I4
evidence): a PERTURBED-attr typed body (grid[5] low byte +1, ksigns[5] 5→200, edited in
the `$grid`/`$ksigns` DenseArray attrs) was REGENERATED by the compiler into a distinct
kernel symbol (`…_PERT_…`). On the board, with inputs that drive grid index 5 + sign
selector 5: the REAL-attr kernel = 4646.25 (= real-table ref) and the PERTURBED-attr
kernel = 19931.625 (= perturbed-table ref) — each matches the reference built with ITS OWN
table, and the two HW results DIVERGE. A hardcoded-table emitter (where the attr is dead)
could not diverge. So the grid + ksigns DenseArray attrs genuinely flow attr → emitted
const → hardware result. See `inc36_knob.cpp` + `ssh_rvv_byte_exact_stdout.txt`.

**Reference faithfulness:** the board's real `ggml_vec_dot_iq3_xxs_q8_K` falls back to
`_generic` on riscv (there is no hand-written RVV iq3_xxs kernel; the riscv quants.c has
vl128/vl256/… helpers gated on a runtime VLEN dispatch, but the externally-defined
numerically-exact contract is `_generic`), so validating against the transcribed
`_generic` validates exactly the contract the board's kernel must satisfy.

## (3) raw()=0 + STRUCTURED

`raw()` actual-call count in the compiler-emitted kernel: **0**. Every value is an emitc
node; the only opaque pieces are the sanctioned `(float)*(const _Float16 *)` d read and the
`__riscv_*` intrinsic spellings — both structured CallOpaque. The grid + ksigns + kmask
tables are structured `static const` decls (emitc VerbatimOp), the SAME channel the
GRID-codebook siblings use. `nm` on the board object shows ONE exported symbol
(`tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_…`) and ONE external ref (`__extendhfsf2`,
the compiler-rt fp16→fp32 builtin) — no kernel fallback, the RVV intrinsics inline. See
`structured_proof.txt`.

## (4) lit + reds

- 2 new lit tests PASS (the lowering FileCheck + the op-accept/7-negative dataflow).
- Full lit (`check-tianchenrv`): **662 discovered, 659 pass, exactly the 3 documented
  pre-existing reds** (`Scripts/rvv-generated-bundle-abi-e2e-*` — computed-masked-strided
  ×2 + the abi-e2e-self-test, all unrelated to the dialect/conversion, documented as the
  same 3 reds since inc30+). +2 over inc35's 660 = my two new tests.
- Sibling byte-identity (regenerated through `tcrv-opt | mlir-translate-20` and diffed
  byte-for-byte): iq2_xxs (inc33) AND iq2_s (inc35) emitted bodies **BYTE-IDENTICAL** →
  the change is purely additive. All other sibling conversion lit tests (q4_0, q6_K, q2_K,
  q3_K, q4_K, q5_K, iq4_nl, iq4_xs, mxfp4, iq2_xs, …) still pass in the full suite.
- Clean full rebuild green (all 60 targets).

## (5) The grid-of-4 + packed aux (vs iq2)

iq2_xxs established the GRID-codebook class. iq3_xxs is its iq3 GRID sibling, with ONE
load-bearing structural delta and one byte-layout delta:

| | iq2_xxs (inc33) | iq3_xxs (inc36) |
|---|---|---|
| grid table | 256 × **uint64** (8 int8/entry) | 256 × **uint32** (4 int8/entry) |
| grid attr type | `DenseI64ArrayAttr` | `DenseI32ArrayAttr` (entries < 0x80000000) |
| grid byte address | `grid_i8 + idx*8` | `grid_i8 + idx*4` |
| grid load width | `vle8(8)` (one index → 8 lanes) | `vle8(4)` (one index → 4 lanes) |
| 8-lane sign group | ONE index, 8 contiguous grid bytes | **TWO** indices (q3[2l+0], q3[2l+1]), two 4-lane passes |
| kmask split | one `vle8(kmask, 8)` | `kmaskLo`={1,2,4,8} + `kmaskHi`={16,32,64,128} |
| index/aux layout | INTERLEAVED aux pair (aux0=indices, aux1=signs+scale) in qs | **SEPARATE** regions: q3[64] (indices) @2, gas[32] (aux) @66 |
| aux per sub-block | TWO uint32 (aux0 indices + aux1 signs+scale) | ONE uint32 (signs+scale only) |
| trailing factor | `0.125f` | **`0.25f`** |
| sign plane | `ksigns_iq2xs[128]` | `ksigns_iq2xs[128]` (REUSED) |
| packed-aux scale | `2*(aux1>>28)+1` | `2*(aux32>>28)+1` (REUSED) |
| sign selector | `(aux1>>7l)&127` | `(aux32>>7l)&127` (REUSED) |

**The grid-of-4 (the new lane width):** each iq3 grid entry decodes to 4 values, not 8.
An 8-element ggml inner group `for(j<4){ grid1[j]·q8[j], grid2[j]·q8[j+4] }` is therefore
TWO grid-of-4 lookups; we realize it as two 4-lane `vle8(4)` passes (grid1 vs q8[0..3] with
kmask bits 0..3, grid2 vs q8[4..7] with kmask bits 4..7) chained into the same `vwredsum`
accumulator (order-free integer add → byte-exact). This is iq2_xxs's group body verbatim
at width 4 — the GRID + SIGN mechanism reused, the grid-of-4 + the q3/gas region split the
only deltas.

**The packed aux (REUSED from iq2_xxs):** like iq2_xxs's aux1, iq3_xxs's single per-sub-
block `aux32` carries the 4 sign selectors `(aux32>>7l)&127` (each indexing `ksigns_iq2xs`)
plus the 4-bit scale in its high nibble `2*(aux32>>28)+1`. Reassembled uint32-domain from 4
byte loads (alignment-safe) — the iq2_xxs packed-aux mechanism, unchanged.

## (6) Remaining IQ tail

iq3_xxs is the 4th GRID-codebook kernel (iq2_xxs/iq2_xs/iq2_s/iq3_xxs). The remaining IQ
kernels are siblings of this class, each its own structural increment:
- **iq3_s** — 512-entry `iq3s_grid` (each entry 4 int8, like iq3_xxs) + a qh-bit plane (a
  3rd 9th index bit) + EXPLICIT signs read from a sign-byte region (like iq2_s, NOT ksigns)
  + explicit per-sub-block `scales[]` (a two-half int4 split like iq2_xs/iq2_s) + a
  per-block sign-scale. Combines the iq3 grid-of-4 with iq2_s's qh-plane + explicit-sign +
  explicit-scales mechanisms — a direct re-composition of pieces already built.
- **iq1_s** / **iq1_m** — 2048-entry ternary `iq1_grid` (each entry packs 1.5-bit ternary
  values) + a per-block delta + a per-sub-block sign/scale scheme (iq1_m adds per-sub-block
  scale nibbles). The ternary grid + the 1.5-bit packing is the genuinely-new mechanism
  left in the tail.

These are the last (niche) IQ kernels. The 17 covered kernels span every COMMON ggml dot
class (legacy block-quant q4_0..q5_1, K-quant super-block q2_K..q6_K, the 16-entry codebook
iq4_nl/iq4_xs, fp4 mxfp4) plus the iq2 GRID-codebook trio + iq3_xxs.

## Reproduce
```
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-iq3-xxs-q8-k-block-dot.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > tcrv_emitted_kernel.cpp
scp tcrv_emitted_kernel.cpp inc36_validate.cpp rvv:~/inc36_iq3xxs/
ssh rvv 'cd ~/inc36_iq3xxs && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  -c tcrv_emitted_kernel.cpp -o k_off.o  && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  inc36_validate.cpp k_off.o  -o val_off  && ./val_off && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast -c tcrv_emitted_kernel.cpp -o k_fast.o && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast inc36_validate.cpp k_fast.o -o val_fast && ./val_fast'
```

## Files in this artifact dir
- `iq3_xxs_input.mlir` — the typed-body input (the conversion test module).
- `iq3_xxs_emitted.cpp` / `tcrv_emitted_kernel.cpp` — the compiler-emitted kernel C.
- `inc36_validate.cpp` — the byte-exact HW harness (verbatim `_generic` reference +
  3 negative controls: wrong grid + wrong signs + wrong scale).
- `iq3_xxs_input_perturbed.mlir` — the PERTURBED `$grid`/`$ksigns` typed-body (grid[5] +1,
  ksigns[5] 5→200), a distinct kernel symbol.
- `iq3_xxs_perturbed_kernel.cpp` — the compiler-regenerated perturbed kernel C.
- `inc36_knob.cpp` — the COMPILER-KNOB control (real-attr vs perturbed-attr kernel diverge
  on HW, each matching its own table → the attr is the live attr → const → HW knob).
- `ssh_rvv_byte_exact_stdout.txt` — the raw board stdout (off + fast + knob + nm proof).
- `structured_proof.txt` — the raw()=0 / structured-emission proof.
