# G3-lode-flat FLAT-3 — q5_1 gemm_tile front-door promote (byte-exact cert)

**What**: promote coverage cell `{op:gemm_tile, format:q5_1, engine:rvv}` from
`dispatch-wired` (direct-emitter `emitRepackGemvQ5_1Q8_1`) to **`constructed`
(STRONG)** — the FIVE-bit-with-min family, the THIRD flat tracer through the repack
front door after q4_1/q5_0. **q5_1 COMBINES two already-constructed mechanisms**:
q5_0's 5th-bit qh gather (UNSIGNED, no −16 offset) + q4_1's single per-block MIN fold.

**Front door**: `RVVLowerQuantContraction.cpp` `lowerToRepackGemvQ51` (decode) /
`lowerToRepackGemmQ51` (prefill) construct the typed
`tcrv_rvv.typed_repack_gem{v,m}_loop_body` region (fold_model
`lane_wise_vector_scale_min`, the q4_1 dual-fp16 + single MIN fold WHOLE) out of the
SHARED q4_0 decomposed bricks:
- integer CORE `repack_lane_wise_q4_x_i8_dot` / `repack_gemm_lane_wise_q4_x_i8_dot`
  stamping `weight_nibble_unsigned` + `weight_qh_byte_offset` (=320) **with NO
  `weight_offset_bias`** → the q5_1 UNSIGNED 5-bit decode leaf: the RAW unsigned
  nibble peel (`vand 0x0F` / `vsrl 4`), the transposed bit-packed qh 5th bit expanded
  per lane (`vmv_v_x` splat / `vid` + lane-shift `vadd` / `vsrl_vv` / `vand 1` /
  `vsll 4` / `vncvt` → {0,16}), OR-ed into the nibble (`vor_vv_u8`), reinterpreted
  `vreinterpret_v_u8_i8` — and **NO `vsub` centering** (the weight is UNSIGNED [0,31];
  the asymmetric bias lives in the MIN fold, unlike q5_0's −16). The shared emitter's
  `assemble5` skips the centering `vsub` when `offsetBias == 0` (byte-identical to the
  retired q5_1 direct emitter's `assemble5Unsigned`). GEVM selects the strip lane by
  the COMPILE-TIME `h*half`; GEMM by the RUNTIME `strip_row_offset`.
- scale FOLD `repack_dual_fp16_scale_fold` / `repack_gemm_dual_fp16_scale_fold` with
  the q4_1 MIN pair (`weight_min_byte_offset` =32 / `activation_sum_byte_offset` =2
  GEVM / =8 GEMM): `vfwmul(d_x)·d_y` / `vfcvt` / `vfmacc`, then `vfwmul(m_x)·s_y` +
  `vfadd`. The realized-body manifest = `typed_repack_gem{v,m}_loop_body` +
  `block_five_bit_qh_source` + `five_bit_offset_binary_x_i8_product` +
  `dual_fp16_scale_fold` (with min) + `typed_repack_gem{v,m}_loop_yield`.

**Arithmetic** (per block): `sumf += (d_x·d_y)·Σ_i(w_i·q8_i) + m_x·s_y`, with
`w_i = (nibble_i) | (qh_bit_i << 4)` UNSIGNED in [0,31] (NO offset). The `m_x·s_y`
MIN term (like q4_1); `s_y` = `block_q8_1`'s precomputed scaled activation sum.
Routing keyed off the abstract scale_model
`dual-fp16-per-block-d_x.d_y-plus-min-five-bit` (the discriminator); the constructed
loop body carries the q4_1 fold scale_model `dual-fp16-per-block-d_x.d_y-plus-min`
(GEVM) / `…-plus-min-4col` (GEMM).

**Combines (reuse map)**: q5_1 reuses the q5_0 `expandQhBit` qh gather (transposed
qh plane, per-lane 5th-bit expand) — but UNSIGNED, no −16 — and the q4_1 min-fold
pair (`weight_min_byte_offset`/`activation_sum_byte_offset`, `m_x·s_y` via 2nd
`vfwmul` + `vfadd`). ZERO net-new decode primitives; the ONE emitter change is the
`offsetBias == 0` skip in `assemble5`. The two ODS core-brick verifiers were relaxed
from "qh + bias TOGETHER or NEITHER" to "qh may appear with OR without bias (bias
alone still rejected)".

## Byte-exact ZERO-MODEL host cert — `q51_repack_cert.c`

Independent per-element scalar oracle (recomputes q5_1×q8_1 from RAW PLAIN block
bytes — fp16 d + fp16 m + u32 qh + 16 qs — with the ggml qh-bit-index reconstruct,
UNSIGNED weight, zero reuse of the tested read path) vs a host-scalar transcription of
BOTH front-door kernels' exact lane-wise arithmetic on the REPACKED `block_q5_1x16`
(x16) buffer: the GEVM (plain `block_q8_1` activation) and the GEMM (interleaved
`block_q8_1x4`, RUNTIME-strip qh) tested paths.

Cert three-requirements:
1. **corpus complete** — qh bits SET and CLEAR both exercised, full nibble [0,15]
   span, signed q8 span, `d_x/d_y != 0`, **`m_x != 0` (min term ACTIVE)**;
   non-degenerate (not all-qh-set / all-clear).
2. **input same-origin** — both sides consume the SAME generated q8_1 activation
   bytes (quants + the precomputed `s_y`); GEMM x4 tile is the SAME `a_qs`/`a_s`
   re-interleaved.
3. **oracle independent** — tested reads the x16 repacked buffer + the transposed qh
   mask expansion; oracle reads the plain blocks with a naive per-element dot off the
   qh bit index. Neither shares the other's read/decode.

Result (`./q51_repack_cert`):
- **GEVM INTEGER sumi bit-exact: 0/256 mismatches** (arch-independent load-bearing
  claim; the UNSIGNED 5-bit reconstructed-weight dot including the qh bit).
- **GEMM INTEGER sumi bit-exact: 0 mismatches** vs the GEVM sumi (certifies the qh
  RUNTIME-strip selection + the x4 activation addressing).
- fp output max_rel ~3.0e-6 (pure reassociation; **includes the `m_x·s_y` min term**).
- Falsification (non-vacuity, `-DFALSIFY_DROP_QH` / `-DFALSIFY_DROP_MIN`):
  dropping the qh 5th bit → integer mismatches, FAIL; dropping the min term → fp
  mismatch, FAIL (both qh and min are load-bearing).

## Emitted-C compile verification — `q51_GEVM.cpp` / `q51_GEMM.cpp`

`tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`, cross-compiled
with the spacemit riscv64 `clang -march=rv64gcv_zvfh -c` → well-formed vector object
(vle8/vand/vsrl/vor/vreinterpret/vncvt/vwmacc/vfwmul/vfmacc/**vfadd** — NO `vsub`
centering — plus the GEMM `vadd_vx_u16` runtime-strip lane shift).

## Claim boundary

Kernel-axis construction coverage (byte-exact), **NOT** an e2e/perf beat [NG-4].
Host has no `qemu-riscv64` user runner, so the emitted RVV vector C is
compile-verified well-formed but NOT executed here; the numerical proof is the scalar
zero-reuse pair. A board (`ssh rvv`) run of the emitted kernel is the owed claim=full
upgrade.
