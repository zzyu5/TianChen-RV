# B2/B3 q4_K emit provenance (reproduce deterministically)

Host: `/home/kingdom/phdworks/TianchenRV`. Date 2026-06-29.

## Source state
- Git HEAD: `24b99893` (committed post-emitter-sweep; working tree CLEAN at emit time).
- Emit tools FORCE-CLEAN-rebuilt (per the "incremental builds unreliable" memory):
  `rm -f build/bin/tcrv-opt build/bin/tcrv-translate && ninja -C build bin/tcrv-opt bin/tcrv-translate`
  - `build/bin/tcrv-opt`       md5 `ceebd8d89d98b8c12f3c50f302ea7d4a` (rebuilt 18:03)
  - `build/bin/tcrv-translate` md5 `a3e1b570a2b60f223639b14d74e6c7a8` (rebuilt 18:03)

## Translate path decision
The project's own `tcrv-translate --tcrv-rvv-emitc-to-cpp` is NOT usable on the
post-`--tcrv-rvv-lower-to-emitc` module: it operates at the `tcrv.exec.kernel`
selection surface and errors `requires a selected path surface` (raw fixture) /
`requires at least one tcrv.exec.kernel` (post-lower module). So the emit uses the
PROVEN archived path (same as the q4_1 / q8_0 repack findings): upstream LLVM-20
`mlir-translate --mlir-to-cpp` on the lowered EmitC module.

- Translate binary: `/usr/lib/llvm-20/bin/mlir-translate`

## Exact emit commands
```
# B2 — q4_K repack GEVM (decode)
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-K-q8-K.mlir \
    --tcrv-rvv-lower-to-emitc \
  | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp \
  > tcrv_q4k_gemv_vlen128.inc

# B3 — q4_K repack GEMM (prefill)
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir \
    --tcrv-rvv-lower-to-emitc \
  | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp \
  > tcrv_q4k_gemm_vlen128.inc
```

## Emitted artifact fingerprints
- `tcrv_q4k_gemv_vlen128.inc` md5 `64c49fe69bfe8dfbc3e47fd98128e620` (7590 lines)
- `tcrv_q4k_gemm_vlen128.inc` md5 `b0b5beacac233116b6aaa481e46b8ec4` (14113 lines)
- `tcrv_q4k_gemv_shim.inc`    md5 `0bf017f4f4170c5d1b1886ec9344f35e` (hand-authored ABI shim)
- `tcrv_q4k_gemm_shim.inc`    md5 `f2de33d9de488ea4d2b4bc5005c4c035` (hand-authored ABI shim)

## Structural verification (host)
- `lit`/FileCheck on both fixtures (default + NOWALL prefixes): PASS — the
  post-sweep emitter still produces the banked block-as-lane structure
  (vwmacc lane-wise dot, **zero** `redsum`; 8-sub-block 6-bit scale/min unpack;
  super-block d/dmin fp16 strips + bsums-min fold). So the B2 byte-exact banking
  from the k1 oracle (WORST_NORM 7.07e-7) applies to this emit (structure-identical).

## Emitted symbol signatures (what the shims wrap)
```
extern "C" void tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);          // 5 args
extern "C" void tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy,
    size_t nr, size_t nc, size_t bs);                                              // 7 args, bs LAST
```
Both bodies require the `zvfh` ISA extension (fp16 super-block d/dmin loads) —
present on rvv (SG2044) and k1 (X60). The emitted file `#include`s
`<stddef.h> <stdint.h> <riscv_vector.h>` at the top (harmless duplicates when
re-included at file scope in repack.cpp; standard include guards apply).
