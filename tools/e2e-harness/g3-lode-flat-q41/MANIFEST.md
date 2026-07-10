# G3-lode-flat 曳光弹 — q4_1 gemm_tile front-door promote (byte-exact cert)

**What**: promote coverage cell `{op:gemm_tile, format:q4_1, engine:rvv}` from
`dispatch-wired` (direct-emitter `emitRepackGem{v,m}Q4_1Q8_1`) to **`constructed`
(STRONG)** — the FLAT-family first tracer through the repack front door.

**Front door**: `RVVLowerQuantContraction.cpp` `lowerToRepackGemvQ41` (decode) /
`lowerToRepackGemmQ41` (prefill) construct the typed
`tcrv_rvv.typed_repack_gem{v,m}_loop_body` region (fold_model
`lane_wise_vector_scale_min`) out of the SHARED q4_0 decomposed bricks:
- integer CORE `repack_lane_wise_q4_x_i8_dot` / `repack_gemm_lane_wise_q4_x_i8_dot`
  stamping the new `weight_nibble_unsigned` UnitAttr → the q4_1 RAW unsigned nibble
  [0,15] peel (`vle8_v_u8` + `vand 0x0F`/`vsrl 0x04` + `vreinterpret_i8`), **NO**
  offset-binary `vsll/vsra -8`.
- scale FOLD `repack_dual_fp16_scale_fold` / `repack_gemm_dual_fp16_scale_fold`
  stamping the new `weight_min_byte_offset` / `activation_sum_byte_offset` I64 pair →
  the single MIN correction `acc += m_x*s_y` (`vfwmul` + `vfadd`) AFTER the shared
  dual-fp16 scale fold `acc += (d_x*d_y)*sumi` (`vfwmul` + `vfmacc`).

**Arithmetic** (per block): `sumf += (d_x·d_y)·Σ_i(nibble_i·q8_i) + m_x·s_y`, unsigned
nibble (no −8), single scalar min-fold (m_x·s_q). [PAT-S6] AlreadyLean / no-cliff →
selector reason=prior (not the q4_K 8-submin register-cliff).

## Byte-exact ZERO-MODEL host cert — `q41_repack_cert.c`

Independent per-element scalar oracle (recomputes q4_1×q8_1 from RAW PLAIN block
bytes, zero reuse of the tested read path) vs a host-scalar transcription of the
front-door kernel's exact lane-wise arithmetic on the REPACKED block_q4_1x16 (x16)
buffer.

Cert three-requirements:
1. **corpus complete** — m_x != 0 (MIN term active), d_x/d_y != 0, full unsigned
   nibble [0,15] span, signed q8 span.
2. **input same-origin** — both sides consume the SAME generated q8_1 activation.
3. **oracle independent** — tested reads the x16 repacked buffer with the kernel's
   lane addressing; oracle reads the plain blocks with a naive per-element dot.
   Neither shares the other's read/decode.

Result (`./q41_cert`):
- **INTEGER sumi bit-exact: 0/256 mismatches** (arch-independent load-bearing claim).
- fp output max_rel 1.49e-5 (pure reassociation).

## Emitted-C compile verification — `q41_GEVM.cpp` / `q41_GEMM.cpp`

`tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`, compiled with
the spacemit riscv64 `clang -march=rv64gcv_zvfh -c` → well-formed vector object code
(16 GEVM / 23 GEMM RVV instructions: vle8/vwmacc/vfmacc/vfadd/vand/vsrl/vfwmul).

## Claim boundary

Kernel-axis construction coverage (byte-exact), **NOT** an e2e/perf beat [NG-4].
Host has no `qemu-riscv64` user runner, so the emitted RVV vector C is
compile-verified well-formed but NOT executed here; the numerical proof is the scalar
zero-reuse pair. A board (`ssh rvv`) run of the emitted kernel is the owed claim=full
upgrade — declared owed, not yet in `schema/cert-lineage.v1.json` (the central registry
back-fill is the accounting line's ledger; this cert's fold_model
`lane_wise_vector_scale_min` is a new flat-min family whose controlled-vocab onboarding
is deferred to that line).
