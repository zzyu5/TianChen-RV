# [X-SCALAR] family #3 — real ternary 2-bit vec_dot scalar kernel (construction note)

> Construction journal for the scalar extension family's first *real* owned
> compute kernel. Records what landed and what stays open. Status/progress is
> journal, not spec — the authority framing lives in
> `docs/canon/Weft-RV_执行总纲v2.md` §6 ([X-SCALAR] owned 内核落点).

## What landed

The `weft.scalar` family's tracer-bullet trivial compute op
(`weft_scalar.compute_skeleton`, which emitted a vacuous
`int32_t v = 7; call(v);`) is joined by a **real** ternary 2-bit vec_dot
boundary that lowers to a pure-scalar C kernel:

- **ODS op** `weft_scalar.tq2_0_q8_k_vec_dot`
  (`include/Weft/Dialect/Scalar/IR/ScalarOps.td`). Carries the tq2_0 x
  q8_K block-format facts as typed attributes: `qk`, `weight_block_stride`,
  `activation_block_stride`, `weight_d_byte_offset`,
  `activation_d_byte_offset`, `activation_quant_byte_offset`, plus
  `source_kernel` / `selected_variant` for the exported function name. Registered
  automatically via `GET_OP_LIST` (no dialect-registration edit).

- **Emitter** `ScalarTernaryQ2Q8BlockDotToEmitCFunc`
  (`lib/Plugin/Scalar/ScalarBackendEmissionDriver.cpp`) lowers the op to a
  standalone EmitC function rendered by the shared `--weft-scalar-emitc-to-cpp`
  translate route as the ggml `ggml_vec_dot_tq2_0_q8_K` contraction in plain C
  nested loops:

  ```c
  for (size_t ib = 0; ib < nb; ib += 1) {
    const uint8_t *qs = vx + ib*66;
    const int8_t  *q8 = vy + ib*292 + 4;
    int sumi = 0;
    for (size_t j = 0; j < 64; j += 32)
      for (size_t l = 0; l < 4; l += 1) {
        int shift = (int)(l * 2);
        for (size_t k = 0; k < 32; k += 1) {
          int w = (((int)qs[j + k] >> shift) & 3) - 1;   // ternary {-1,0,+1(,2)}
          sumi += (int)q8[j*4 + l*32 + k] * w;           // int8 x int8 MAC
        }
      }
    float dy = *(const float *)(vy + ib*292 + 0);
    float dx = (float)*(const _Float16 *)(vx + ib*66 + 64);
    sumf = sumf + (float)sumi * (dy * dx);
  }
  *s = sumf;
  ```

## The two locked constraints ([J-3] low-bit discipline)

1. **Pure scalar C** — FileCheck `--implicit-check-not="__riscv_"` on both the
   direct-translate and the full planning-pass pipeline. Zero vector intrinsics.
2. **NO XOR-popcount** — the low-bit weight is a 2-bit *base-I field extract*
   (`(qs >> shift) & 3`) with the ggml `-1` ternary bias, multiply-accumulated as
   int8 x int8. `--implicit-check-not="popcount"`; no codebook, no `^`.

This mirrors the vector-path form (`RVVToEmitCTernaryBinary.cpp`
`emitTQ2_0Q8_KBlockDot`) stripped of all `__riscv_` machinery — same element
pairing (`q8[j*4 + l*32 + k]` <-> `qs[j+k]` shifted by `l*2`), same scalar int32
accumulator, same single per-super-block `fp16(x.d) * f32(y.d)` scale fold.

## Anti-vacuity

Emission is operand-driven off the typed attributes: `qk=256 -> / 256`,
`weight_block_stride=66 -> ib*66`, `activation_block_stride=292 -> ib*292`,
`activation_quant_byte_offset=4 -> + 4`, `weight_d_byte_offset=64 -> + 64`.
Changing any attribute changes the emitted C (verified: stride 66->99 emits
`ib*99`).

## Byte-exact gate

`test/Target/Scalar/tq2-0-q8-k-ternary-vec-dot.mlir` locks the full structure
with FileCheck and diffs the translate output against the captured golden
`test/Target/Scalar/tq2-0-q8-k-ternary-vec-dot.golden.c` (the ggml scalar
ternary reference captured at construction). Baseline captured at construction —
no `git stash` (new op has no pre-image at the pinned base).

## fp16 scale fold — independence ([F-6])

The per-super-block scale reads `(float)*(const _Float16 *)`. At the C-source
level this is a scalar `_Float16` deref: it pulls **no** `rvv.*` intrinsic into
the closure, so `closure ∩ {rvv.*} = ∅` holds for the emitted kernel. Whether it
lowers to `zfh` hardware or a software `__extendhfsf2` libcall is a target-march
concern of the *downstream* C compiler, not of this emitter (matches §6's
"软件 fp16→fp32 / zfh" routes; the forbidden `zvfh` vector route is never used).

## Still open (honest boundary)

- **判据④ (capability-blind selection of the vector-absent variant)** — not
  wired. The op is lowered when present; there is no evidence the *scalar*
  variant is selected *because* a vector capability is absent.
- **Runtime bit-exact** — the gate is emit-golden (byte-exact C text), NOT a
  numerical run. Bit-exact vs ggml stays `pending-hardware`.
- **Scalar `zfh` fact registration** — the fp16 read is source-level portable;
  no independent `scalar.zfh` capability fact is registered yet (§6 pre-req).
- **Marginal-cost ledger (LED-2)** — this is a second owned kernel family entry,
  but the C2 cost curve is not computed here.
