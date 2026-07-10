# G3-lode-flat FLAT-2 — q5_0 gemm_tile front-door promote (byte-exact cert)

**What**: promote coverage cell `{op:gemm_tile, format:q5_0, engine:rvv}` from
`dispatch-wired` (direct-emitter `emitRepackGemvQ5_0Q8_0`) to **`constructed`
(STRONG)** — the FIVE-bit family, the SECOND flat tracer through the repack front
door after q4_1.

**Front door**: `RVVLowerQuantContraction.cpp` `lowerToRepackGemvQ50` (decode) /
`lowerToRepackGemmQ50` (prefill) construct the typed
`tcrv_rvv.typed_repack_gem{v,m}_loop_body` region (fold_model
`lane_wise_vector_scale`, the q4_0 d-only fold WHOLE — NO min) out of the SHARED
q4_0 decomposed bricks:
- integer CORE `repack_lane_wise_q4_x_i8_dot` / `repack_gemm_lane_wise_q4_x_i8_dot`
  stamping `weight_nibble_unsigned` + the new `weight_qh_byte_offset` (=288) +
  `weight_offset_bias` (=16) → the q5_0 5-bit decode leaf: the RAW unsigned nibble
  peel (`vand 0x0F` / `vsrl 4`), the transposed bit-packed qh 5th bit expanded per
  lane (`vmv_v_x` splat / `vid` + lane-shift `vadd` / `vsrl_vv` / `vand 1` /
  `vsll 4` / `vncvt` → {0,16}), OR-ed into the nibble (`vor_vv_u8`), reinterpreted
  `vreinterpret_v_u8_i8`, then the offset-binary `-16` (`vsub_vx_i8` by 16). GEVM
  selects the strip lane by the COMPILE-TIME `h*half`; GEMM by the RUNTIME
  `strip_row_offset`.
- scale FOLD `repack_dual_fp16_scale_fold` / `repack_gemm_dual_fp16_scale_fold`
  d-ONLY (`vfwmul(d_x)·d_y` / `vfcvt` / `vfmacc`), NO min offset pair (the q4_0 fold
  WHOLE). The realized-body manifest = `typed_repack_gem{v,m}_loop_body` +
  `block_five_bit_qh_source` + `five_bit_offset_binary_x_i8_product` +
  `dual_fp16_scale_fold` (no min) + `typed_repack_gem{v,m}_loop_yield`.

**Arithmetic** (per block): `sumf += (d_x·d_y)·Σ_i(w_i·q8_i)`, with
`w_i = ((nibble_i) | (qh_bit_i << 4)) - 16` in [-16,15]. **NO min term** (unlike
q4_1/q5_1). Routing keyed off the abstract scale_model
`dual-fp16-per-block-d_x.d_y-five-bit` (the discriminator); the constructed loop
body carries the q4_0 fold scale_model `dual-fp16-per-block-d_x.d_y`.

## Byte-exact ZERO-MODEL host cert — `q50_repack_cert.c`

Independent per-element scalar oracle (recomputes q5_0×q8_0 from RAW PLAIN block
bytes — fp16 d + u32 qh + 16 qs — with the ggml qh-bit-index reconstruct, zero reuse
of the tested read path) vs a host-scalar transcription of BOTH front-door kernels'
exact lane-wise arithmetic on the REPACKED `block_q5_0x16` (x16) buffer: the GEVM
(plain `block_q8_0` activation) and the GEMM (interleaved `block_q8_0x4`,
RUNTIME-strip qh) tested paths.

Cert three-requirements:
1. **corpus complete** — qh bits SET (4047) and CLEAR (4145) both exercised,
   POST-OFFSET NEGATIVE weights (4145, `w < 0` after −16), full nibble [0,15] span,
   signed q8 span, `d_x/d_y != 0`; non-degenerate (not all-qh-set / all-clear).
2. **input same-origin** — both sides consume the SAME generated q8_0 activation
   bytes (GEMM x4 tile is the SAME `a_qs` re-interleaved).
3. **oracle independent** — tested reads the x16 repacked buffer + the transposed qh
   mask expansion; oracle reads the plain blocks with a naive per-element dot off the
   qh bit index. Neither shares the other's read/decode.

Result (`./q50_repack_cert`):
- **GEVM INTEGER sumi bit-exact: 0/256 mismatches** (arch-independent load-bearing
  claim; the 5-bit reconstructed-weight dot including the −16 offset + qh bit).
- **GEMM INTEGER sumi bit-exact: 0 mismatches** vs the GEVM sumi (certifies the qh
  RUNTIME-strip selection + the x4 activation addressing).
- fp output max_rel 3.73e-6 (pure reassociation).
- Falsification (non-vacuity): dropping the qh 5th bit in the tested path → 256
  integer mismatches, FAIL (the qh + −16 deltas are load-bearing).

## Emitted-C compile verification — `q50_GEVM.cpp` / `q50_GEMM.cpp`

`tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`, cross-compiled
with the spacemit riscv64 `clang -march=rv64gcv_zvfh -c` → well-formed vector object
(vle8/vand/vsrl/vor/vreinterpret/vsub/vncvt/vwmacc/vfwmul/vfmacc, plus the GEMM
`vadd_vx_u16` runtime-strip lane shift).

## Claim boundary

Kernel-axis construction coverage (byte-exact), **NOT** an e2e/perf beat [NG-4].
Host has no `qemu-riscv64` user runner, so the emitted RVV vector C is
compile-verified well-formed but NOT executed here; the numerical proof is the scalar
zero-reuse pair. A board (`ssh rvv`) run of the emitted kernel is the owed claim=full
upgrade.
