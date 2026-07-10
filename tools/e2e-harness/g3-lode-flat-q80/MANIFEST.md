# G3-lode-flat FLAT-4 收官格 — q8_0 gemm_tile front-door promote (byte-exact cert)

**What**: promote coverage cell `{op:gemm_tile, format:q8_0, engine:rvv}` from
`dispatch-wired` (direct-emitter `emitRepackGemvQ8_0Q8_0`) to **`constructed`
(STRONG)** — the FULL-int8 family, the FOURTH and LAST flat tracer through the
repack front door after q4_1/q5_0/q5_1, the SIMPLEST flat variant, COMPLETING the
flat-mainline retirement batch0.

**Front door**: `RVVLowerQuantContraction.cpp` `lowerToRepackGemvQ80` (decode) /
`lowerToRepackGemmQ80` (prefill) construct the typed
`tcrv_rvv.typed_repack_gem{v,m}_loop_body` region (fold_model
`lane_wise_vector_scale`, the q4_0 d-only fold WHOLE — NO min) out of the SHARED
decomposed bricks:
- integer CORE `repack_lane_wise_q4_x_i8_dot` / `repack_gemm_lane_wise_q4_x_i8_dot`
  stamping the NEW `weight_full_i8` selector → the q8_0 FULL-int8 decode leaf: the
  SIGNED `vle8 i8` strip load (NO nibble unpack — NO `vand`/`vsrl`/`vsll`/`vreinterpret`
  decode; the weight is a full signed int8), the per-position `vwmul` (i8×i8 → i16),
  and the `vwadd_wv` into an i32 IN-BLOCK accumulator (REPLACING q4_0's i16 `vwmacc`
  + end-of-block lo/hi `vwadd_vv` combine — full int8 products overflow i16 after 3
  terms). qk=32 positions per block, one weight per position. `weight_full_i8` is
  MUTUALLY EXCLUSIVE with the nibble decode selectors (`weight_nibble_unsigned` /
  `weight_qh_byte_offset` / `weight_offset_bias`), enforced fail-closed in both
  core-brick verifiers.
- scale FOLD `repack_dual_fp16_scale_fold` / `repack_gemm_dual_fp16_scale_fold`
  d-ONLY (`vfwmul(d_x)·d_y` / `vfcvt` / `vfmacc`), NO min offset pair (the q4_0 fold
  WHOLE).

**Arithmetic** (per block): `sumf += (d_x·d_y)·Σ_i(w_i8·q8_i8)`, w_i a FULL signed
int8. **NO nibble unpack, NO qh, NO offset, NO min** — the SIMPLEST flat family.
Routing keyed off the abstract scale_model `dual-fp16-per-block-d_x.d_y-full-i8` (the
discriminator); the constructed loop body carries the q4_0 fold scale_model
`dual-fp16-per-block-d_x.d_y`.

## Byte-exact ZERO-MODEL host cert — `q80_repack_cert.c`

Independent per-element scalar oracle (recomputes q8_0×q8_0 from RAW PLAIN block
bytes — fp16 d + 32 full int8 qs — with a naive per-element dot, zero reuse of the
tested read path) vs a host-scalar transcription of BOTH front-door kernels' exact
lane-wise arithmetic on the REPACKED `block_q8_0x16` (x16) buffer: the GEVM (plain
`block_q8_0` activation) and the GEMM (interleaved `block_q8_0x4`) tested paths.

Cert three-requirements:
1. **corpus complete** — FULL int8 weight span (positive=4071, negative=4091,
   boundary ±127 forced=61), signed q8 activation span, `d_x/d_y != 0`; non-degenerate.
2. **input same-origin** — both sides consume the SAME generated int8 weight AND q8_0
   activation bytes (GEMM x4 tile is the SAME `a_qs` re-interleaved).
3. **oracle independent** — tested reads the x16 repacked buffer with the kernel's
   exact lane-wise addressing (`byte = 32 + i*16 + c`); oracle reads the plain blocks
   with a naive per-element dot. Neither shares the other's read/decode.

Result (`./q80_repack_cert`):
- **GEVM INTEGER sumi bit-exact: 0/256 mismatches** (arch-independent load-bearing
  claim; the full-int8 dot `Σ w_i·q8_i`).
- **GEMM INTEGER sumi bit-exact: 0 mismatches** vs the GEVM sumi (certifies the x4
  interleaved activation addressing).
- fp output max_rel 3.6e-7 (pure reassociation).

## Emitted-C compile verification — `q80_GEVM.cpp` / `q80_GEMM.cpp`

`tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`, cross-compiled
with the spacemit riscv64 `clang -march=rv64gcv_zvfh -c` → well-formed vector object
(vle8_v_i8 / vwmul_vx_i16 / vwadd_wv_i32 / vfwmul / vfcvt / vfmacc / vse32; NO nibble
decode, NO vwmacc).

## Claim boundary

Kernel-axis construction coverage (byte-exact), **NOT** an e2e/perf beat [NG-4].
Host has no `qemu-riscv64` user runner, so the emitted RVV vector C is
compile-verified well-formed but NOT executed here; the numerical proof is the scalar
zero-reuse pair. A board (`ssh rvv`) run of the emitted kernel is the owed claim=full
upgrade. The GEMM path is NET-NEW construction (no q8_0 GEMM direct emitter ever
existed) validated by the INDEPENDENT oracle.
