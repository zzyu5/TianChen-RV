# Emit the q4_0 16x1 repack GEMM from tcrv-opt (N3 e2e win → compiler output)

Goal: make `tcrv-opt` EMIT the validated `vlen128-q4_0-16x1` repacked GEMM hot
kernel (the one that gave the 5.84× llama prefill / 4.7% decode win), so the
fast RVV C is the COMPILER's structured output, not hand-written ggml C. I5
(structured emitc, raw()=0) + I7 (fail-closed verifier) are non-negotiable.

## What was built (the GEMM hot kernel, raw()=0, byte-exact structurally)

A new typed dialect op + fail-closed verifier + STRUCTURED emitc emitter +
dispatch + lit, all consuming the ALREADY-repacked weights (the repack-at-load
buffer + mul_mat dispatch from the prior e2e-repack-gemm agent stay as-is):

- **Op** `tcrv_rvv.repack_gemm_q4_0_q8_0` (`include/.../RVVOps.td`):
  block-as-lane sibling of `tcrv_rvv.q4_0_q8_0_gemm`. Operands: repacked weight
  base (vx), repacked activation base (vy), output (s), runtime n / nr / nc /
  output-row-stride (bs), vl. Carries the 16x1 repacked block-format facts as
  typed I4-mirror attrs.
- **Verifier** (`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`): pins the 16x1
  repacked ABI fail-closed (I7) — QK=32, block_q4_0x16 stride **288**,
  block_q8_0x4 stride **136**, weight quants at **+32**, activation quants at
  **+8**, 16 weight rows / 4 activation columns per group, half-lane width **8**;
  rejects wrong kind / scale model / block fact / operand C type / operand count.
  (Struct facts confirmed against the byte-exact reference `verify_kernel.cpp`:
  `block_q4_0x16 { ggml_half d[16]; int8_t qs[256]; }` = 288B,
  `block_q8_0x4 { ggml_half d[4]; int8_t qs[128]; }` = 136B.)
- **Emitter** `emitRepackGemmQ4_0Q8_0` (`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`):
  fully STRUCTURED emitc nodes (every value a node in the IR graph; raw()=0).
  Reproduces the patch GEMM's node sequence exactly: outer activation-row-GROUP
  loop (nr/4), weight-column-GROUP loop (nc/16), two-halves loop (8-lane halves),
  4×8 f32 accumulators, the per-block lane-wise `vwmacc` product (vle8 disjoint
  sub-loads of the repacked qs, plain `vsll`/`vsra` nibble sign-extend — NO
  in-kernel vxor, the ^0x88 bias is baked into the repacked bytes — `vwmacc_vx`
  against the scalar q8 quants), the `vwadd` lo/hi combine, the
  `vfwmul`/`vfcvt`/`vfmacc` scale fold (the raw `_Float16` activation scale fed
  straight into `vfwmul_vf`, no float cast), and the `vse32` vector store. NO
  cross-lane vredsum (the per-block reduction wall is gone).
- **Dispatch + recognizer** (`lib/Conversion/RVV/RVVToEmitC.cpp` +
  `RVVToEmitCInternal.h`): first-match table entry `isRepackGemmQ4_0Q8_0Body` →
  `emitRepackGemmQ4_0Q8_0`, structural marker = the op identity (no family-name
  branch; I3).
- **Lit** `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-0-q8-0.mlir` (emitc-IR
  FileCheck of the block-as-lane structure) + dialect verifier coverage
  `test/Dialect/RVV/repack-gemm-q4-0-q8-0-dataflow.mlir` (accept + 4 fail-closed).

## Local gates (board-free) — ALL GREEN

| Gate | Result |
|---|---|
| Clean build (`ninja tcrv-opt`) | green |
| raw() grep (`grep -nE '\braw\(' lib/Conversion/RVV/*.cpp \| grep -vE '//\|no raw'`) | **0** (structured emission only, I5) |
| New lit (emitc-IR block-as-lane structure) | PASS |
| New dialect verifier lit (accept + 4 fail-closed) | PASS |
| RVV lit subset (Conversion/RVV + Dialect/RVV + Target/RVV, 358 tests) | 358/358 PASS |
| Full lit suite | 677/680; the 3 failures are a PRE-EXISTING `scripts/rvv_generated_bundle_abi_e2e.py --self-test` internal assertion (pure-Python fake-bundle gen, never invokes tcrv-opt, unrelated to GEMM; `scripts/` untouched by this work) |

## Byte-exact proof (structural + compilable C)

`tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp` (LLVM-20)
produces `emitted-repack-gemm.cpp`. The emitted C is the validated kernel:

- The intrinsic SEQUENCE matches the patch GEMM node-for-node: 4× `vfmv_v_f`
  accumulator seeds, 8× `vmv_v_x_i16m1` lo/hi seeds, the nibble loop's
  `vle8`/`vsll`/`vsra`/`vsra` decode, 8× (`*(const int8_t *)` scalar read +
  `vwmacc_vx`) lane-wise accumulate, 4× `vwadd_vv` combine, `vle16` the 8 weight
  scales, 4× (`*(const _Float16 *)` raw scale + `vfwmul_vf` + `vfcvt_f_x_v` +
  `vfmacc_vv`), 4× `vse32` store. (The vwmacc interleave is pair-wise (lo_c,hi_c)
  vs the patch's lo-block-then-hi-block — bit-identical: the 8 accumulators are
  independent and integer add is order-independent.)
- All addresses match the patch byte-for-byte: weight nibbles `bl+32+i*16+roff`,
  act lo `al+8+i*4+c`, act hi `al+8+64+i*4+c`, weight scales `bl+roff*2`, act
  scales `al+c*2`, output `(y*4+c)*bs+x*16+roff`.
- The emitted C **compiles clean** for `--target=riscv64 -march=rv64gcv_zvfh`
  (`riscv-compile.log`, exit 0) and lowers to REAL vector instructions
  (`riscv-vector-instr-histogram.txt`): 8 `vwmacc.vx`, 4 `vwadd.vv`, 4
  `vfwmul.vf`/`vfcvt.f.x.v`/`vfmacc.vv`, `vse32.v` — no vredsum wall.

The patch is already byte-exact vs `_generic` (e2e-repack-gemm RESULTS.md), so
structural fidelity to its node sequence + addresses + a clean riscv compile is
the byte-exactness evidence available board-free.

## Answer to the task questions

- **Does tcrv-opt now EMIT the repack GEMM (raw()=0, byte-exact)?** YES for the
  GEMM (prefill headline kernel). Structured emitc (raw()=0, I5), fail-closed
  verifier (I7), structurally byte-exact to the validated `vlen128-q4_0-16x1`
  GEMM, and the emitted C compiles clean to RVV vector instructions.
- **Compiler-emitted e2e number vs hand-written 5.84×?** PENDING the rvv board.
  The numeric byte-exact run (compiler-emitted kernel vs `_generic` over many n)
  and the e2e `llama-bench` swap (compiler-emitted C into ~/tcrv-llamacpp,
  keeping repack-at-load + dispatch) are board-only (I8) and not run here.
- **Honest blocker / remaining:** (1) numeric byte-exact + e2e on the rvv board
  remain — the local proof is structural + compile-clean, not a numeric run.
  (2) GEMV (`ggml_gemv_q4_0_16x1_q8_0`) is a structurally simpler sibling with a
  DIFFERENT activation ABI (plain `block_q8_0`, not `q8_0x4`) — left as follow-on
  (a separate op; hand-written GEMV can stay for the prefill-headline e2e swap).
  The prefill **5.84× is the GEMM** (this op); the decode number still rides on
  the hand-written GEMV in the swap.

## Board-swap note: the emitted ABI is NOT ggml's positional ABI (use an adapter)

The emitted symbol is `tcrv_emitc_<kernel>_<variant>` with signature
`(size_t n, float *s, const uint8_t *vx, const uint8_t *vy, size_t nr, size_t nc, size_t bs)`
— `bs` is LAST and `vx/vy/nr/nc` shift, vs ggml's
`ggml_gemm_q4_0_16x1_q8_0(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc)`.
A rename/`#define` to ggml's name would mis-wire the positional args (`bs`↔`vx`)
and silently produce garbage. The swap MUST go through a thin adapter (needed for
the name anyway), e.g.:

```c
extern "C" void tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(
    size_t, float *, const uint8_t *, const uint8_t *, size_t, size_t, size_t);
extern "C" void ggml_gemm_q4_0_16x1_q8_0(int n, float *s, size_t bs,
    const void *vx, const void *vy, int nr, int nc) {
  tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy,
      (size_t)nr, (size_t)nc, bs);
}
```

(int→size_t is safe for positive dims.) This is the FIRST place to look if the
board numeric byte-exact check fails.

## Files changed

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `GgmlRepackGemmQ40Q80Op` def
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — `::verify()` (fail-closed)
- `lib/Conversion/RVV/RVVToEmitCInternal.h` — recognizer + emitter declarations
- `lib/Conversion/RVV/RVVToEmitC.cpp` — recognizer body + dispatch table entry
- `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` — `emitRepackGemmQ4_0Q8_0`
- `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-0-q8-0.mlir` (new)
- `test/Dialect/RVV/repack-gemm-q4-0-q8-0-dataflow.mlir` (new)

## Artifacts in this dir

- `input-repack-gemm.mlir` — the typed-op MLIR input
- `emitted-repack-gemm.emitc.mlir` — `tcrv-opt --tcrv-rvv-lower-to-emitc` output
- `emitted-repack-gemm.cpp` — the compiler-emitted C (mlir-translate)
- `riscv-compile.log` — clean `rv64gcv_zvfh` compile of the emitted C
- `riscv-vector-instr-histogram.txt` — the RVV instrs the emitted C lowers to
