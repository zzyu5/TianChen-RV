# Option-2 STAGE C1b — the ISOLATED bit-exact PACKER (FINDING, 2026-06-24)

**STATUS: COMPLETE — byte-exact PASS (memcmp==0), emitter-inert, lit PASS.**

The compiler-emitted plain `block_q4_0` → `block_q4_0x16` PACKER (ggml
`make_block_q4_0x16`, the live `blck_size_interleave==1` `^0x88` branch) is built
as a NEW typed RVV op + emitter and HOST-verified byte-identical to ggml's own
inlined oracle across the full stress matrix.

## What was built (file:lines)

- **Op def** `GgmlPackQ40ToX16Op` (`tcrv_rvv.pack_q4_0_to_q4_0x16`):
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (inserted after
  `GgmlRepackGemvQ40Q80Op`, before `GgmlRepackGemvQ41Q81Op`). A STRIPPED subset of
  the repack-GEMV op: operands `src`/`dst`/`nblocks`/`vl` (vl carried only to
  reuse the `with_vl` dispatch shape; the scalar body never uses it), attrs
  `kind, qk=32, src_block_stride=18, dst_block_stride=288,
  src_quant_byte_offset=2, dst_quant_byte_offset=32, weight_interleave=16,
  xor_mask=0x88(136)`. NO activation/half_lanes/LMUL/scale_model attrs.
- **Verifier** `GgmlPackQ40ToX16Op::verify()`:
  `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (inserted after the
  `GgmlRepackGemvQ40Q80Op::verify` at ~:1934, before `GgmlRepackGemvQ80Q80Op`).
  Fail-closed (I7): `isAllowedAttr` allowlist scoped to the 8 pack names +
  `isForbiddenDataflowParameterAttr` reject + pins `qk==32, src_stride==18,
  dst_stride==288, src_quant_off==2, dst_quant_off==32, interleave==16,
  xor==0x88`. VERIFIED fail-closed live: `dst_block_stride=289` → error
  "requires dst_block_stride == 288"; `xor_mask=119` → error "requires xor_mask
  == 0x88". Good fixture round-trips clean.
- **Dispatch + plumbing** `lib/Conversion/RVV/RVVToEmitC.cpp`: added
  `{&isPackQ4_0ToX16Body, &VariantToEmitCFunc::emitPackQ4_0ToX16}` to the `:347`
  table beside the repack-GEMV entry; `isPackQ4_0ToX16Body(WithVLOp)`
  (`isa<GgmlPackQ40ToX16Op>` single-op recognizer) mirrored on
  `isRepackGemvQ4_0Q8_0Body`.
- **Emit method** `VariantToEmitCFunc::emitPackQ4_0ToX16`:
  `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (inserted before
  `emitRepackGemvQ4_0Q8_0`). Header decls in
  `lib/Conversion/RVV/RVVToEmitCInternal.h`.
- **Artifacts** `…/artifacts/kernel-coverage/pack-q4_0-x16/`:
  `input-pack-q4_0.mlir`, `emitted.emitc.mlir`, `emitted-packer.cpp`,
  `emitted-packer.host.cpp` (riscv-ism-stripped host copy), `lower.err`/
  `translate.err` (empty), `verify_emitted_packer.cpp`, `vpack`, `pack-verify.log`.
- **Lit test** `test/Conversion/RVV/rvv-to-emitc-pack-q4-0-to-q4-0x16.mlir`
  (CHECK: pack-fn signature + structured for/subscript/bitwise_xor/assign body;
  PURE-NOT: vwmacc/vredsum/vfwmul/vle8/vsra/vsll — pure-scalar canary).

## The scalar emit body (the transform, attr-derived)

Structured EmitC (NOT a verbatim string blob): `emitc.for` ×3 nested +
`subscript`/`load`/`cast`/`bitwise_xor`/`assign`. The LLVM-20 `emitc.verbatim`
has NO operand-substitution form, so operands MUST flow as typed nodes (this is
why the body is structured, not a text blob). Emitted C (host copy):

```c
void tcrv_emitc_..._pack_q4_0_to_q4_0x16(size_t nblocks,
                                         const uint8_t* src, uint8_t* dst) {
  for (size_t b = 0; b < nblocks; b += 1) {
    size_t sbase = b*288;  size_t dbase = b*288;   // 16 src blocks*18 = 288 = dst stride
    // scales: out.d[j] = in[j].d  (2 bytes each, VERBATIM, no xor)
    for (size_t j = 0; j < 16; j += 1) {
      size_t sd = sbase + j*18;  size_t dd = dbase + j*2;
      dst[dd+0] = (uint8_t)src[sd+0];  dst[dd+1] = (uint8_t)src[sd+1];
    }
    // quants: out.qs[off*16+blk] = in[blk].qs[off] ^ 0x88  (block-major-within-byte)
    size_t dq = dbase + 32;
    for (size_t off = 0; off < 16; off += 1)
      for (size_t blk = 0; blk < 16; blk += 1)
        dst[dq + off*16 + blk] = (uint8_t)src[sbase + blk*18 + (2+off)] ^ 136;
  }
}
```

PURE byte gather + XOR — ZERO `__riscv_`/`vint`/`vfloat` on the data path. The
emitted module carries an inert `emitc.include <"riscv_vector.h">` + a dead
`__riscv_vsetvl_e32m1(nblocks)` from the `with_vl` wrapper (result unused); both
are stripped for the host build (`emitted-packer.host.cpp`) since the pure-scalar
pack uses neither. Compiles HOST `g++ -O2 -std=c++17`, NO `-march`, no
`riscv_vector.h`.

## Host memcmp gate (the verdict)

`./vpack` — EXACT byte equality vs ggml's inlined `make_block_q4_0x16`, seed
`mt19937(20260622)`, n ∈ {64,256,4096,11008,14336} × NC ∈ {16,32,64,336} ×
200 trials:

```
  NC=16   ( 1 grp)  blocks=186000     mismatch_trials=0     PASS
  NC=32   ( 2 grp)  blocks=372000     mismatch_trials=0     PASS
  NC=64   ( 4 grp)  blocks=744000     mismatch_trials=0     PASS
  NC=336  (21 grp)  blocks=3906000    mismatch_trials=0     PASS
  VERDICT: PASS (compiler-emitted PACKER byte-identical to ggml make_block_q4_0x16
           across NC=16/32/64/336, memcmp==0)
```

**5,208,000 output blocks (×288 bytes) memcmp==0** — the `i%16` vs `i/16`
interleave, the all-256-quants-and-nothing-else `^0x88`, and the verbatim 16
fp16 scale copy are all adjudicated correct by exact byte equality. Harness
gathers `src` so output block `b=g*nb+x` consumes exactly the 16 sources
`w[(g*16+i)*nb+x]` the reference's `make_block` consumes for `ref[b]`.

## Emitter inertness (before/after EQUALITY — absolutes are stale)

Captured pre-edit baselines from the clean tree, re-ran with the NEW tcrv-opt:

```
INERT  rvv-to-emitc-repack-gemv-q4-0-q8-0.mlir  (3854146c…)
INERT  rvv-to-emitc-repack-gemv-q4-1-q8-1.mlir  (353e9149…)
INERT  rvv-to-emitc-repack-gemv-q8-0-q8-0.mlir  (f331ce08…)
INERT  rvv-to-emitc-tq2-0-q8-k-block-dot.mlir   (0d206c11…)
```

All 4 byte-identical before/after — the new op + dispatch entry perturb ZERO
existing emitters. Full suite: **197/197 lit PASS** (`Conversion/RVV/` +
`Dialect/RVV/`), incl. the new pack test.

## Build discipline

Forced rebuilds via `ninja tcrv-opt`; CONFIRMED each cycle: `RVVOps.cpp.inc`
regen + `RVVToEmitCBlockQuantLinear.cpp.o` / `RVVDialectWideningOps.cpp.o` /
`RVVToEmitC.cpp.o` compile + `Linking CXX executable bin/tcrv-opt`.

## HONEST framing (carry it)

mechanism-(a): a compiler-emitted packer. **e2e-REDUNDANT** — ggml ALREADY packs
plain→x16 at model-load (`repack_q4_0_to_q4_0_16_bl`); swapping this in changes
ZERO e2e numbers. The VALUE is the ISOLATED proof that the compiler can PRODUCE
the x16 layout it DECLARES (C1's producer half), bit-exact to ggml, host-only,
memcmp-exact. It is **NEVER a kernel/perf/e2e win**. Its only live home would be
the §4.2 prefill-amortizing JIT repacker (M≫1 only).

## Blocker

None. C1b fully met: op+verifier built, host byte-exact (memcmp==0 on 5.2M
blocks), emitter-inert, 197/197 lit PASS.
