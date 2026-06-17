# INC-39 — iq1_m × q8_K byte-exact; the LAST classic ggml dot kernel (20th)

20th ggml dot kernel: `ggml_vec_dot_iq1_m_q8_K` — the sibling of iq1_s, the **last of the
classic q\*/iq\* family** (the 19-increment line). New op `tcrv_rvv.iq1_m_q8_k_block_dot`,
fully STRUCTURED (raw()=0; I5). It REUSES iq1_s's 2048-entry ternary grid + the per-block
DELTA bias, with three GENUINELY NEW pieces.

Byte-exact `*s` vs ggml's OWN `_generic` (`quants.c:1142-1201`) on real `ssh rvv`
(`-march=rv64gcv`, Zvl128b ⇒ VLEN≥128): **2464/2464 cases, 0 failures** at
`-ffp-contract=off` AND `fast`; the 11-bit grid index marches to **max idx = 2047 (hit
YES)**. FOUR negative controls (wrong grid + wrong scale + wrong DELTA + **wrong
iq1m_scale reconstruction**) all diverge.

## Honest coverage status (source-verified — NOT "literal 100%")

The task framed iq1_m as "the LAST -- literal 100%". Verified against THIS checkout
(`coverage_census.txt`, from `/home/kingdom/phdworks/llama.cpp` `quants.h`/`quants.c`), that
framing is **inaccurate**: there are **24** canonical `ggml_vec_dot_*` block-quant kernels;
we now cover **20/24**. iq1_m is the last of the **classic** set (q4_0/q4_1/q5_0/q5_1/q8_0
+ q2_K..q6_K + iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq1_s/iq1_m/iq4_nl/iq4_xs/mxfp4). FOUR
recently-added kernels remain UNCOVERED (real `_generic` bodies present):
`ggml_vec_dot_nvfp4_q8_0` (NVFP4), `ggml_vec_dot_q1_0_q8_0` (binary quant),
`ggml_vec_dot_tq1_0_q8_K`, `ggml_vec_dot_tq2_0_q8_K` (TriLM ternary). Those are the next
rungs; iq1_m closes the classic line, not all of ggml.

## The new mechanism (vs iq1_s) — three pieces

### (a) NO fp16 d field — the packed `iq1m_scale_t` reconstruction (the headline new piece)
`block_iq1_m` has NO `ggml_half d`. The super-block scale is RECONSTRUCTED from a packed
`iq1m_scale_t` union whose 16 fp16 bits are SPREAD across the high nibbles of the 4 uint16
`scales[]` words:
```
scale.u16 = (sc[0]>>12) | ((sc[1]>>8)&0x00f0) | ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000);
d = (float)scale.f16 * y[i].d;
```
This is a **bit reinterpret**, NOT a numeric conversion. The emitter assembles the 4 nibble
fields into a `uint16_t` lvalue, takes its address, and reads
`(float)*(const _Float16 *)(&scbits)` — byte-identical to ggml's union read on
little-endian. The per-sub-block 3-bit scales live in the SAME words' LOW bits, split into a
**two-half** scale per sub-block: `ls1 = 2*((sc[ib/2]>>(6*(ib%2)+0))&7)+1` (groups 0..1),
`ls2 = 2*((sc[ib/2]>>(6*(ib%2)+3))&7)+1` (groups 2..3).

### (b) the uint8 qh[16] plane + the index reconstruction
`qh` is `uint8 qh[16]` (TWO bytes per sub-block, vs iq1_s's uint16 qh[8]). The 11-bit grid
index for group `l`: `idx = qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)` (index-
high 3 bits from bits 0..2 of `qh[l/2]` for even `l` (shift 8), bits 4..6 for odd `l` (shift
4)). Same 2048-entry **iq1s_grid** ternary codebook (reused verbatim), byte-viewed as signed
int8 (0xff → −1) — NO sign plane / kmask.

### (c) the per-GROUP DELTA (4 independent signs ⇒ bsums UNUSABLE)
iq1_m's delta is per HALF-sub-block group of 8 elements, with FOUR independent signs:
`delta[0]=qh[0]&8`, `delta[1]=qh[0]&0x80`, `delta[2]=qh[1]&8`, `delta[3]=qh[1]&0x80`. Because
the four signs are independent, the delta CANNOT be folded through the q8 16-element `bsums`
the way iq1_s did. Instead each group computes a FRESH `lsum2 = Σ q8[0..7]` (a parallel
`vwredsum_vs_i8m1_i16m1` of the SAME q8 vector already loaded for the grid dot),
scalar-multiplied by `delta[l]`, accumulated into `sum2[l/2]`. So iq1_m reads NO bsums
region (the op carries no `activation_bsums_byte_offset`). The fold:
`sumi1 += sum1[0]*ls1 + sum1[1]*ls2`, `sumi2 += sum2[0]*ls1 + sum2[1]*ls2`, and
`sumf += d*((float)sumi1 + IQ1M_DELTA*(float)sumi2)`, `*s = sumf` (NO trailing factor).

## (1) The op + lowering + reuse + files

### New op `tcrv_rvv.iq1_m_q8_k_block_dot`
- ODS: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (`GgmlBlockDotIQ1MQ8KOp`). Four runtime
  ABI operands (vx/vy/s/n) + the vl token; the super-block-format structural attrs
  (`weight_block_stride`=56, `weight_qs_byte_offset`=0, `weight_qh_byte_offset`=32,
  `weight_scales_byte_offset`=48 — the NEW packed-scales region) PLUS the `grid` (2048 int64)
  table. **NO `weight_d_byte_offset`** (no fp16 d field), **NO `activation_bsums_byte_offset`**
  (per-group delta makes bsums unusable). Stride 56 = QK_K/8 + QK_K/16 + QK_K/32 (32+16+8),
  trusted from ggml-common.h's `static_assert` (the task prd's `scales[QK_K/64]` was WRONG).
- Verifier: `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (`GgmlBlockDotIQ1MQ8KOp::verify`).
  Fail-closed (I7): pins `kind` (`ggml_iq1_m_q8_k_block_dot`), `scale_model`
  (`packed-iq1m-scale-per-half-scale-ternary-grid-codebook-per-group-delta-int-domain`),
  qk=256, sub_block=32, **stride=56**, qs@0, **qh@32**, **scales@48**, act stride=292, act
  qs@4, grid size==2048, operand C types, i32m1 result, the policy. Rejects forbidden/unknown
  attrs (incl. a stray `activation_bsums_byte_offset`).
- Lowering: `lib/Conversion/RVV/RVVToEmitC.cpp` — `isIQ1MQ8KBlockDotBody` recognizer, a
  dispatch branch (after the iq1_s branch, before mxfp4), and `emitIQ1MQ8KBlockDot` (a
  self-contained parallel of `emitIQ1SQ8KBlockDot`, NOT a shared-helper refactor — preserves
  "siblings byte-identical / additive").

### Reuse (the task mandate "REUSE iq1_s's ternary grid + delta")
From iq1_s: the SAME 2048-entry `iq1s_grid` codebook + indexed lookup (`grid_i8 + idx*8` then
`vle8(8)`), the signed `vwmul_i16m2` widening product, the `vwredsum` i32 reduce, the
`(const int8_t *)` byte-cast grid read (NO sign plane / kmask), the q8_K super-block
structure, the per-super-block fp32 fold as ONE `emitc.expression` with `0.125f` applied
once. New: the iq1m_scale fp16 reinterpret, the uint8 qh layout, the per-half scale split,
and the per-group `Σq8` reduce (`vwredsum_vs_i8m1_i16m1`) for the delta term.

### Files (additive)
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (+ `GgmlBlockDotIQ1MQ8KOp`)
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (+ `GgmlBlockDotIQ1MQ8KOp::verify`)
- `lib/Conversion/RVV/RVVToEmitC.cpp` (+ recognizer/branch/`emitIQ1MQ8KBlockDot`)
- `test/Conversion/RVV/rvv-to-emitc-iq1-m-q8-k-block-dot.mlir` (NEW lit, full grid + CHECK)
- `test/Dialect/RVV/iq1-m-q8-k-block-dot-dataflow.mlir` (NEW lit, positive + 11 fail-closed
  rejections incl. the scales-offset, the no-d/no-bsums pins, the grid-size pin)

## (2) Byte-exact vs ggml iq1_m `_generic` (cases, incl. the 4 controls, 0 failures)

`ssh rvv`, `-march=rv64gcv -mabi=lp64d`, both `-ffp-contract=off` AND `fast`
(`ssh_rvv_byte_exact_stdout.txt`):
```
iq1_m byte-exact: 2464/2464 passed, 0 failed
grid-index coverage (edge): max idx = 2047, hit 2047 = YES
NEG-CTRL wrong-grid   (every grid byte +1)             -> DIVERGES
NEG-CTRL wrong-scale  (drop the +1 in ls1/ls2)         -> DIVERGES
NEG-CTRL wrong-DELTA  (omit the delta sumi2 term)      -> DIVERGES
NEG-CTRL wrong-RECON  (iq1m_scale wrong nibble shift)  -> DIVERGES   <- the NEW piece is load-bearing
RESULT: PASS (byte-exact vs ggml iq1_m _generic + 4 negative controls live)
```
- **2464 cases** = 300 reps × 8 sizes {256..16384} + 64 edge super-blocks; idxModes exercise
  q8=+127 and q8=±127. Edge blocks drive qs=0xff + qh index fields=7 → index 2047, scale
  fields 0 (ls=1) and 7 (ls=15), the 4 delta signs toggled.
- The **wrong-RECON** control (the iq1m_scale reconstruction reading sc[1] from the WRONG
  shift, `>>4` instead of `>>8`) diverges → proves the packed-scale reconstruction is live.
- **NOT a tautology**: the harness reference is a differential VERBATIM re-implementation of
  `_generic` (scalar nested `l`/`j` loops with the union); our emitter chains `vwredsum` per
  group in a flat 8-sub-block structure. A THIRD independent host scalar stand-in
  (`/tmp` self-check, different indexing) agreed bit-for-bit over all 2464 cases — emitter ==
  reference == stub. The 2048-entry grid is extracted PROGRAMMATICALLY from the committed
  iq1_s grid (single source of truth).

## (3) raw()=0 + structured (`structured_proof.txt`)
- **raw() actual-call count: 0**; the only non-comment verbatim in the lowered IR is the
  structured `static const uint64_t tcrv_iq1m_grid[2048]` decl.
- distinct `__riscv_*`: `vsetvl_e8m1`, `vle8_v_i8m1`, `vmv_v_x_i32m1`/`vmv_v_x_i16m1` (seeds),
  `vwmul_vv_i16m2`, `vwredsum_vs_i16m2_i32m1` (grid dot), `vwredsum_vs_i8m1_i16m1` (the NEW
  per-group Σq8), `vmv_x_s_i32m1_i32`/`vmv_x_s_i16m1_i16` (extracts). **NO sign-apply
  intrinsics** (vand/vmsne/vneg/vmerge/kmask all 0).
- exactly **1** `_Float16` reinterpret read (the iq1m_scale reconstruction); exactly **1**
  `0.125f` (IQ1M_DELTA, once per super-block fold); `*s = sumf` (no trailing factor).

## (4) lit + reds
- Two new lit tests PASS (Conversion FileCheck + Dialect `--verify-diagnostics`).
- **Full clean rebuild** (`ninja -t clean && ninja`, tablegen `.inc` regenerated from scratch),
  then `ninja check-tianchenrv`: **668 tests, 665 pass, 3 fail** — the 3 are the SAME
  pre-existing DOCUMENTED reds, unrelated to iq1_m:
  `rvv-generated-bundle-abi-e2e-{explicit,pre-realized}-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`
  and `rvv-generated-bundle-abi-e2e-self-test.test`. The count went 666→668 (both new tests
  pass; 663→665 passing).
- **Siblings byte-identical (additive)**: the iq1_s emitted kernel re-lowered from the new
  build is `md5 8295cb5e…`, byte-identical to its committed inc38 artifact. The new code is a
  self-contained branch touching no shared codepath.

## (5) The iq1m_scale reconstruction (the new piece) + how byte-exact
The packed `iq1m_scale_t` reconstruction is the genuinely-new emission: 16 fp16 bits SPREAD
across the high nibbles of 4 uint16 scales[] words, reassembled with the EXACT ggml shifts/
masks into a `uint16_t` lvalue, then read AS `_Float16` through `&scbits` — a bit reinterpret
(the union), byte-identical to ggml on little-endian. Byte-exactness holds because (i) the
reassembly mirrors ggml's `(sc0>>12)|((sc1>>8)&0xf0)|((sc2>>4)&0xf00)|(sc3&0xf000)` exactly,
(ii) the `(const _Float16 *)` read is the same bit-pun the board's clang uses for ggml's
union, (iii) the integer dot/Σq8/accumulation is order-free, and (iv) the per-super-block fp32
fold `d*((float)sumi1 + 0.125f*(float)sumi2)` is a single `emitc.expression` whose left-assoc
grouping + FMA fusion match ggml's `d*(sumi1 + IQ1M_DELTA*sumi2)` — confirmed by 2464/2464
under `-ffp-contract=fast`.

## (6) Coverage list (the full ggml dot-kernel census)
See `coverage_census.txt`. 20/24 canonical `ggml_vec_dot_*` block-quant kernels covered; iq1_m
closes the classic q\*/iq\* set. Remaining (honest): `nvfp4`, `q1_0`, `tq1_0`, `tq2_0`.

## Reproduce
```
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-iq1-m-q8-k-block-dot.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > tcrv_emitted_kernel.cpp
scp tcrv_emitted_kernel.cpp inc39_validate.cpp rvv:~/inc39_iq1m/
ssh rvv 'cd ~/inc39_iq1m && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  -c tcrv_emitted_kernel.cpp -o k_off.o  && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off  inc39_validate.cpp k_off.o  -o val_off  && ./val_off && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast -c tcrv_emitted_kernel.cpp -o k_fast.o && \
  clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=fast inc39_validate.cpp k_fast.o -o val_fast && ./val_fast'
```

## Files in this artifact dir
- `iq1_m_input.mlir` — the typed-body input (= the conversion lit test, full grid + CHECK).
- `tcrv_emitted_kernel.cpp` — the compiler-emitted kernel C (mlir-translate output).
- `inc39_validate.cpp` — the byte-exact HW harness (differential `_generic` reference with the
  iq1m_scale union + 4 negative controls).
- `ssh_rvv_byte_exact_stdout.txt` — the board output (off + fast, 2464/2464, 4 controls).
- `structured_proof.txt` — the raw()=0 + intrinsic/mechanism census.
- `coverage_census.txt` — the source-verified 20/24 ggml dot-kernel coverage table.
