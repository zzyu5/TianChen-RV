// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml q5_0 16x1-REPACKED GEMV (decode) hot kernel as STRUCTURED emitc IR
// (I5; ZERO raw() strings). q5_0 = q4_0 + the 5th high bit. The single typed op
// tcrv_rvv.repack_gemv_q5_0_q8_0 lowers to the BLOCK-AS-LANE single-output-column
// matmul: the 16 interleaved weight rows of a group occupy 16 vector lanes, the
// dot accumulates LANE-WISE via vwmacc (NO cross-lane vredsum wall), and at
// VLEN=128 each 16-block group is processed as TWO disjoint contiguous 8-lane
// halves. The activation is ONE plain block_q8_0 stream (stride 34, quants at +2).
// The block_q5_0x16 repack carries RAW nibbles at +32 (no ^0x88 bake) and a
// 64-byte TRANSPOSED bit-packed qh region at +288: per nibble step the lane
// decode reads a 16-bit qh mask (one per element), expands it lane-wise
// (vid/vsrl/vand/vsll/vncvt), ORs the {0,16} term into the unsigned nibble, then
// reinterprets u8->i8 and subtracts 16 (the offset-binary -16 reconstruct, the
// PROVEN block-dot path). Byte-exact vs canonical ggml q5_0 (oracle norm=0).

module {
  tcrv.exec.kernel @ggml_repack_gemv_q5_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q5_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q5_0_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q5_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q5_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q5_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 352 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_q5_0_q8_0 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q5_0_q8_0_kernel_ggml_repack_gemv_q5_0_q8_0(
// The block count nb = n / 32.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*352 (block_q5_0x16 stride 352 = 16*22).
// CHECK: literal "352"
// The two 8-lane f32 accumulators (rows 0..7, 8..15).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb; plain q8_0 activation base (stride 34).
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "34"
// The nibble-step loop over the 16 weight bytes.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// Two disjoint repacked UNSIGNED nibble sub-loads (rows 0..7, 8..15).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// The TRANSPOSED bit-packed qh: two 16-bit mask scalar reads (low element i, high
// element i+16), each expanded lane-wise to the {0,16} 5th-bit term.
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// The lane-wise qh expansion: vid/vsrl/vand/vsll over u16, narrow u16->u8.
// CHECK: call_opaque "__riscv_vid_v_u16m1"
// CHECK: call_opaque "__riscv_vsrl_vv_u16m1"
// CHECK: call_opaque "__riscv_vand_vx_u16m1"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8mf2"
// The 5-bit assemble: OR the {0,16} term into the unsigned nibble, reinterpret
// u8->i8, then subtract 16 (the offset-binary -16 reconstruct).
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// The lane-wise vwmacc accumulate against the SINGLE-column scalar q8 quants (NO
// cross-lane vredsum) -- low qs[i] and high qs[16+i], for both 8-lane halves.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// The lo/hi combine vwadd_vv_i32m2 for both halves.
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// The vector scale fold and single-column two-half store.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: NO
// vredsum / vwredsum (any SEW or signedness) appears anywhere in the emitted GEVM.
// NOWALL-NOT: redsum
