# INC-33 — iq2_xxs × q8_K byte-exact; the GRID-codebook class (deep IQ tail opens)

14th ggml dot kernel: `ggml_vec_dot_iq2_xxs_q8_K` — the FIRST member of the deep IQ
tail, the **GRID-codebook** structural class. This is a genuinely NEW mechanism (NOT the
16-entry per-nibble gather of iq4_nl/iq4_xs): each weight byte INDEXES a 256-entry packed
**uint64 GRID codebook** (`iq2xxs_grid`, each entry = 8 int8 values), the per-element SIGN
comes from a separate **SIGN PLANE** (`ksigns_iq2xs[128]` / `kmask_iq2xs`), and the 4-bit
scale folds in the INTEGER domain. New op `tcrv_rvv.iq2_xxs_q8_k_block_dot`, STRUCTURED
(raw()=0).

## (1) The op + the grid+sign+scale lowering + files

**Op** `GgmlBlockDotIQ2XXSQ8KOp` (`tcrv_rvv.iq2_xxs_q8_k_block_dot`):
- ABI: `(weight vx, activation vy, output *s, n, vl)`, 4 runtime ABI operands + the vl
  token → one i32 m1 vector result. `vx`/`vy` = `const uint8_t *`, `*s` = `float *`
  (fail-closed verifier, I7).
- super-block format attrs (I4 mirror): `qk=256`, `sub_block=32`,
  `weight_block_stride=66` (sizeof block_iq2_xxs = {fp16 d; uint16 qs[32]}),
  `activation_block_stride=292` (sizeof block_q8_K), `weight_d_byte_offset=0`,
  `weight_qs_byte_offset=2`, `activation_d_byte_offset=0`,
  `activation_quant_byte_offset=4`.
- the two GRID-codebook tables as typed array attrs: `DenseI64ArrayAttr:$grid` of EXACTLY
  256 int64 entries (the packed uint64 `iq2xxs_grid[256]`, carried as signed-int64 bit
  patterns) and `DenseI32ArrayAttr:$ksigns` of EXACTLY 128 int32 entries (the sign plane
  `ksigns_iq2xs[128]` — **i32 because ksigns values reach 255, beyond int8**, so a
  DenseI8ArrayAttr would silently corrupt them). The verifier pins both sizes. `kmask`
  ({1<<j}) is the inline const, not a table.

**Lowering** `emitIQ2XXSQ8KBlockDot` (a SEPARATE, additive emitter; siblings byte-untouched):
- the GRID codebook emitted ONCE as a `static const int64_t tcrv_iq2xxs_grid[256] = {...}`
  decl (rendering ggml's exact uint64 hex literals with the `ULL` suffix), read as bytes
  through a `const int8_t *grid_i8 = (const int8_t *)tcrv_iq2xxs_grid;` cast — exactly
  ggml's `(const uint8_t *)(iq2xxs_grid + idx)` method.
- the SIGN plane emitted ONCE as a `static const uint8_t tcrv_iq2xxs_ksigns[128] = {...}`
  decl; kmask as `static const uint8_t tcrv_iq2xxs_kmask[8] = {1,2,4,8,16,32,64,128};`
  broadcast-loaded ONCE (`vle8_v_u8m1(kmask, 8)`).
- outer super-block loop `for ibl in nb=n/256`; per-super-block `d = (float)x.d * y.d`
  (fp16 weight d via the `(float)*(const _Float16 *)` seam × fp32 q8_K d via a plain float
  load), `int32_t bsum = 0` reset per super-block.
- a FLAT unrolled loop over the 8 sub-blocks (ib32=0..7). Each sub-block:
  - `aux1` **reassembled from 4 little-endian byte loads** (`a[4]|a[5]<<8|a[6]<<16|
    a[7]<<24`) in the **uint32_t domain** — alignment-safe (the qs stream starts at byte 2,
    so the aux words are 2-aligned, NOT 4-aligned: no `*(uint32_t*)`), and **unsigned** so
    the `>>` is logical (a signed `int` aux1 with bit 31 set would arithmetic-shift and
    corrupt the scale — see (2)).
  - `int ls = 2*(aux1>>28)+1` (the per-sub-block 4-bit scale, [1,31]).
  - a loop over the 4 groups of 8 elements. Each group: the grid index `idx = a[l]`, the
    sign byte `signs = tcrv_iq2xxs_ksigns[(aux1>>7*l)&127]`, the 8 grid int8 values loaded
    via the indexed pointer `grid_i8 + idx*8` (`vle8_v_i8m1`, 8 lanes), the q8 8 lanes
    (`vle8_v_i8m1`). The SIGN-plane application — the one genuinely-new vector construct:
    broadcast `signs` (`vmv_v_x_u8m1`) → `vand` with the kmask vector → `vmsne 0` → mask →
    `vmerge(grid, vneg(grid), mask)` (negate where the sign bit is set). Then
    `vwmul_vv_i16m2(grid_signed, q8)` → i16 product (each lane ≤ 43*127=5461 < 32767), and
    a **chained `vwredsum_vs_i16m2_i32m1`** seeded from the running i32 reduction (integer
    add is order-free, so seeding from the previous group's result across the 4 groups is
    byte-exact). After 4 groups, `vmv_x_s_i32m1_i32` extracts the sub-block `sumi`.
  - `bsum += sumi * ls` (integer domain).
- the per-super-block fold `sumf = sumf + d * (float)bsum` is ONE `emitc.expression` (the
  SAME FMA `_generic` fuses under -ffp-contract=on/default; the `(float)bsum` cast is
  explicit), invoked in STRICT ascending super-block order.
- `*s = 0.125f * sumf` (the iq2_xxs trailing 1/8 factor, a SEPARATE statement OUTSIDE the
  accumulate expression).

The grid lookup is an indexed `vle8(8)` over a pointer (`grid_i8 + idx*8`), the operative
vector width is **8** (one group), so there is **NO vrgather table-index legality fact and
NO m1 table anchor** (unlike iq4_nl/iq4_xs, whose 16-entry codebook broadcasts into a vreg
and pins m1). The 2048-byte grid cannot broadcast into a vreg — this is an honest, distinct
mechanism, and the verifier does NOT carry a (false) m1-table rationale.

**REUSE:** the q6_K-style super-block STRUCTURE — the q8_K activation, the integer `bsum`
accumulation, the per-super-block fp32 fold (`sumf += d*bsum`), the fp16-d × fp32-d scale
form. The signed widening product + `vwredsum` reduce chain.

**Files:**
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `GgmlBlockDotIQ2XXSQ8KOp`.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — `GgmlBlockDotIQ2XXSQ8KOp::verify()`.
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `isIQ2XXSQ8KBlockDotBody` + dispatch branch +
  `emitIQ2XXSQ8KBlockDot`.
- `test/Conversion/RVV/rvv-to-emitc-iq2-xxs-q8-k-block-dot.mlir` (lowering FileCheck).
- `test/Dialect/RVV/iq2-xxs-q8-k-block-dot-dataflow.mlir` (op accept + 7 fail-closed
  negative dataflow checks: wrong kind / scale model / weight stride / qs offset / grid
  size / ksigns size / output ctype).

## (2) Byte-exact vs ggml iq2_xxs — 0 failures

ssh rvv (riscv64, rv64gcv, VLEN=128, clang 18.1.3). The compiler-emitted kernel's `*s` is
BIT-FOR-BIT equal (memcmp of the float bits) to ggml's **VERBATIM `_generic`**
(`quants.c:855-895`: the memcpy aux32[2], ls = 2*(aux32[1]>>28)+1, grid =
(uint8_t*)(iq2xxs_grid+aux8[l]), signs = ksigns_iq2xs[(aux32[1]>>7*l)&127], sumi +=
grid[j]*q8[j]*(signs&kmask?-1:1), bsum += sumi*ls, sumf += d*bsum, *s = 0.125f*sumf):
- **-ffp-contract=off**: 2502 cases, **0 failures**, _generic delta 0/2502.
- **-ffp-contract=fast**: 2502 cases, **0 failures**, _generic delta 0/2502.
- n ∈ {256, 512, … 16384} (multiples of 256), 300 random reps + edge cases.
- Edge cases: grid-index range (marching grid indices over the full [0,255] grid range +
  marching sign selectors over [0,127]), the scale extremes (scale nibble 0 → ls=1, and 15
  → ls=31), q8 = +127 and q8 = ±127.

THE byte-exactness pivot found + fixed on hardware: the scale `2*(aux1>>28)+1` and the sign
selector `(aux1>>7*l)&127` must be computed in the **unsigned (uint32_t) domain**. A first
cut built `aux1` as a signed `int`; for any sub-block whose scale nibble had its top bit set
(scale ≥ 8, i.e. aux1 bit 31 set), `aux1>>28` did an arithmetic (sign-propagating) shift,
producing a negative `ls` and inverting the sub-block's contribution — diagnosed by a
scale-isolation probe on the board (scale_nib 0..7 OK, 8..15 mismatch with `sours =
-(value)`). Switching aux1 + the shifts to uint32_t (logical shift, matching ggml's uint32_t
`aux32[1]`) fixed it; all scale nibbles now match. See `ssh_rvv_byte_exact_stdout.txt`.

**Negative controls — the grid/signs/scale are LIVE, load-bearing:** OUR kernel (real
tables) vs a WRONG-table reference DIVERGES for all three (each at both fp-contract modes):
- **wrong-grid** (every grid byte +1): ours=428.73 vs wrong=451.45 → DIVERGES.
- **wrong-signs** (ksigns sign bit cleared): ours=428.73 vs wrong=-58.21 → DIVERGES.
- **wrong-scale** (drop the `+1` in `2*(aux>>28)+1`): ours=428.73 vs wrong=407.11 → DIVERGES.
These prove the kernel computes with ggml's real table *values* + scale formula.

**COMPILER-KNOB control — the attr is the live knob attr → const → HW** (the stronger
N1/I4 evidence, mirroring inc30): a PERTURBED-attr typed body (grid[5] +1, ksigns[5] 5→200,
edited in the `$grid`/`$ksigns` DenseArray attrs) was REGENERATED by the compiler into a
distinct kernel symbol. On the board, with inputs that exercise grid index 5 + sign
selector 5: the REAL-attr kernel = 49.970417 (= real-table ref) and the PERTURBED-attr
kernel = 7.962306 (= perturbed-table ref) — each matches the reference built with ITS OWN
table, and the two HW results DIVERGE. A hardcoded-table emitter (where the attr is dead)
would emit the SAME const for both inputs and could not diverge. So the grid + ksigns attrs
genuinely flow attr → emitted const → hardware result. See `inc33_knob.cpp` +
`ssh_rvv_byte_exact_stdout.txt`.

**Reference faithfulness (honest residual):** the board's real `ggml_vec_dot_iq2_xxs_q8_K`
IS `_generic` on riscv (`arch-fallback.h` maps it; there is no hand-written RVV iq2_xxs
kernel), so validating against the transcribed `_generic` validates exactly what the board
runs. The reference is a line-for-line transcription of `quants.c:855-895`; the byte-exact
match independently re-proves the transcription.

## (3) raw()=0 + STRUCTURED

`raw()` actual-call count in the compiler-emitted kernel: **0**. Every value is an emitc
node; the only opaque pieces are the sanctioned `(float)*(const _Float16 *)` d read and the
`__riscv_*` intrinsic spellings — both structured CallOpaque. The grid + ksigns + kmask
tables are structured `static const` decls (emitc VerbatimOp, raw()-clean — the same channel
the codebook siblings use). `nm` on the board object shows ONE exported symbol
(`tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_…`) and ONE external ref (`__extendhfsf2`, the
compiler-rt fp16→fp32 builtin) — no kernel fallback, the RVV intrinsics are inlined. See
`structured_proof.txt`.

## (4) lit + reds

- 2 new lit tests PASS (the lowering FileCheck + the op-accept/7-negative dataflow).
- Full lit: **656 discovered, 653 pass, exactly the 3 documented pre-existing reds**
  (`…computed-masked-strided-input-widening-dot-reduce-add-dry-run` ×2 + `…abi-e2e-self-test`,
  all `Scripts/rvv-generated-bundle-abi-e2e-*`, unrelated to the dialect/conversion —
  documented as the same 3 reds since inc30/31/32). +2 over inc32's 654 = my two new tests.
- Sibling byte-identity: iq4_nl (inc30), iq4_xs (inc32), AND q6_K (inc12) emitted bodies
  regenerated and diffed byte-for-byte UNCHANGED → the change is purely additive.
- Clean full rebuild green (all 60 targets).

## (5) The GRID codebook + the SIGN plane (the new mechanism) + how structured

The iq4_nl/iq4_xs codebook class indexes a **16-entry** per-nibble table broadcast into ONE
vreg (vrgather, m1-pinned). iq2_xxs is a fundamentally different, deeper class:
- **The GRID codebook** is a **256-entry uint64 table** (`iq2xxs_grid[256]` = 2048 bytes),
  where each entry packs **8 int8 grid values**. It cannot broadcast into a vreg, so the
  lookup is a **structured indexed load**: `grid_i8 + idx*8` then `vle8(8)` — exactly ggml's
  `(const uint8_t *)(iq2xxs_grid + idx)`. Carried as a `DenseI64ArrayAttr` (the exact uint64
  literals), emitted as a `static const int64_t[256]` decl, byte-viewed via a single
  `(const int8_t *)` cast.
- **The SIGN plane** is `ksigns_iq2xs[128]` (a 128-entry uint8 table indexed by a 7-bit
  selector) + `kmask_iq2xs[8] = {1<<j}`. The per-element sign is `signs & kmask[j] ? -1 :
  1`. Vectorized byte-exactly as: broadcast `signs` → `vand` kmask → `vmsne 0` → mask →
  `vmerge(grid, vneg(grid), mask)` (8 lanes). ksigns is carried as a `DenseI32ArrayAttr`
  (its values reach 255, beyond int8) emitted as a `static const uint8_t[128]` decl.
- **The scale** `ls = 2*(aux1>>28)+1` (a 4-bit per-sub-block scale packed in the high nibble
  of aux1) folds in the **integer domain** (`bsum += sumi*ls`), not the float domain — and
  there is a trailing super-block **0.125f** factor (`*s = 0.125f*sumf`).

All structured emitc (no actual raw() compute calls). The grid/ksigns/kmask tables are
structured const decls, NOT raw strings — and the COMPILER-KNOB control (above, §2) proves
the grid + ksigns DenseArray attrs are the LIVE knob: editing them in the typed body and
re-running the compiler flips the emitted const AND the hardware result.

## (6) Remaining IQ tail (iq2_xs/s, iq3_xxs/s, iq1_s/m)

iq2_xxs opens the GRID-codebook class. The remaining IQ kernels are siblings of this new
class (each indexes a packed grid + a sign/scale scheme), but each differs in the grid size,
the sign source, and the scale dance — each its own structural increment:
- **iq2_xs** — 512-entry `iq2xs_grid` (the index is a 9-bit value from a uint16 qs, NOT a
  byte) + per-sub-block `scales[]` nibbles + the sign bits live in the qs top bits. A direct
  sibling of iq2_xxs (same grid mechanism, different index width + scale source).
- **iq2_s** — 1024-entry `iq2s_grid` + separate `qs`/`qh`/sign byte planes + per-sub-block
  scale nibbles.
- **iq3_xxs** / **iq3_s** — 256/512-entry `iq3xxs_grid`/`iq3s_grid` (each entry 4 int8) +
  sign planes (`ksigns` reused) + a different qs/scale layout.
- **iq1_s** / **iq1_m** — 2048-entry ternary `iq1_grid` + a per-block delta + a 1.5-bit
  scheme (iq1_m adds per-sub-block scale nibbles).

These are a niche tail (rarely the dominant weight quant). The 14 covered kernels span every
COMMON ggml dot class (legacy block-quant q4_0..q5_1, K-quant super-block q2_K..q6_K, the
16-entry codebook iq4_nl/iq4_xs, fp4 mxfp4) plus the FIRST grid-codebook (iq2_xxs).

## Reproduce
```
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-iq2-xxs-q8-k-block-dot.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > tcrv_emitted_kernel.cpp
scp tcrv_emitted_kernel.cpp inc33_validate.cpp rvv:~/inc33_iq2xxs/
ssh rvv 'cd ~/inc33_iq2xxs && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  -c tcrv_emitted_kernel.cpp -o k_off.o  && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  inc33_validate.cpp k_off.o  -o val_off  && ./val_off && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast -c tcrv_emitted_kernel.cpp -o k_fast.o && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast inc33_validate.cpp k_fast.o -o val_fast && ./val_fast'
```

## Files in this artifact dir
- `iq2_xxs_input.mlir` — the typed-body input (the conversion test module).
- `iq2_xxs_emitted.cpp` / `tcrv_emitted_kernel.cpp` — the compiler-emitted kernel C.
- `inc33_validate.cpp` — the byte-exact HW harness (verbatim `_generic` reference +
  3 negative controls: wrong grid + wrong signs + wrong scale).
- `iq2_xxs_perturbed_kernel.cpp` — the compiler-regenerated kernel from a PERTURBED `$grid`
  / `$ksigns` typed-body attr (grid[5] +1, ksigns[5] 5→200), a distinct symbol.
- `inc33_knob.cpp` — the COMPILER-KNOB control (real-attr vs perturbed-attr kernel diverge
  on HW, each matching its own table → the attr is the live attr → const → HW knob).
- `ssh_rvv_byte_exact_stdout.txt` — the raw board stdout (off + fast + the nm proof + knob).
- `structured_proof.txt` — the raw()=0 / structured-emission proof.
