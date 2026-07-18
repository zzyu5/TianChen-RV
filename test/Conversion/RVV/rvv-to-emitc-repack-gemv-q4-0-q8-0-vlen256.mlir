// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=ONESTRIP

// The VLEN=256 fractional one-strip (mf2, numHalves==1, f32m2 accumulator) arm of
// the q4_0 16x1-repacked GEVM, now carried by the typed
// weft_rvv.typed_repack_gemv_loop_body region (the monolithic
// weft_rvv.repack_gemv_q4_0_q8_0 op is retired). At VLEN=256 an e16m1 vector holds
// 16 i16 lanes, so the 16-block-as-lane group is tiled into ONE 16-lane strip
// (half_lanes=16, numHalves==1) rather than the two disjoint 8-lane halves the
// VLEN=128 form uses -- the region carries ONE per-strip f32m2 accumulator, ONE
// integer-core brick producing ONE i32m2 sumi, ONE dual-fp16 scale-FOLD brick, and
// a yield naming the single carried-out vector. This is the region arm the front
// door does NOT auto-select (the selector keeps block-dot at the K1-VLEN256 decode
// cell), so it is authored directly here to pin the emitter's one-strip mf2 form.
//
// BYTE-SAFE only because the repack is 16-way interleaved (block_q4_0x16: 256 qs[]
// bytes = 16 blocks-as-lanes, byte i = block(i%16) offset(i/16)), so the one
// 16-lane strip reads byte-identical repacked data to the two 8-lane halves at
// VLEN=128 (the vlen128-full-body companion pins the two-halves arm). Numerical
// bit-exact-vs-ggml is pending-hardware (ssh rvv).

module {
  weft.exec.kernel @ggml_repack_gemv_q4_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 16 : i64, integer_core_lmul = "mf2", fold_model = "lane_wise_vector_scale"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">):
          %sumi = weft_rvv.repack_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %block_index : index {kind = "repack_lane_wise_q4_x_i8_dot", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">
          %an0 = weft_rvv.repack_dual_fp16_scale_fold %vx, %vy, %sumi, %acc0, %vl block %block_index : index {kind = "repack_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %an0 : !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(
// The active vl is the 16-lane e16m1 strip width (VLEN=256), NOT 8.
// CHECK: literal "16"
// The OUTER weight-column-GROUP loop over nc/16, per-group weight base (stride 288).
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "288"
// ONE 16-lane f32 accumulator (rows 0..15): a SINGLE vfmv_v_f_f32m2(0.0f, 16).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb; plain q8_0 activation stride 34.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "34"
// ONE 16-lane i16 lo + ONE 16-lane i16 hi accumulator seed (NOT four 8-lane).
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The nibble-step loop; ONE disjoint repacked sub-load (rows 0..15 at qs[i*16+0]),
// then the plain sign-extension decode (NO vxor): b_lo=vsra(vsll(b,4),4); b_hi=vsra(b,4).
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsll_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// TWO lane-wise vwmacc (low qs[i], high qs[16+i]) for the single 16-lane strip.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// ONE lo/hi combine vwadd_vv_i32m2.
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// ONE 16-lane weight-scale vle16 (b[l].d[0]) + the single raw _Float16 act scale.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// ONE 16-lane vector store vse32 (s + x*16 + 0; rows 0..15 in one strip).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// Divergence-by-construction is a COUNT, not a presence: the 1x16 strip emits
// EXACTLY ONE of each reduction the 2x8 path emits twice. A leaked second strip
// (a stray 2x8 fallback) would add a second store / accumulator / scale-load.
// The COUNT-1/NOT pairs are ordered by emission position (accumulator seed first,
// then the weight-scale load, then the output store) so each NOT scans forward
// over the remaining input.
// ONESTRIP-COUNT-1: call_opaque "__riscv_vfmv_v_f_f32m2"
// ONESTRIP-NOT: call_opaque "__riscv_vfmv_v_f_f32m2"
// ONESTRIP-COUNT-1: call_opaque "__riscv_vle16_v_f16m1"
// ONESTRIP-NOT: call_opaque "__riscv_vle16_v_f16m1"
// ONESTRIP-COUNT-1: call_opaque "__riscv_vse32_v_f32m2"
// ONESTRIP-NOT: call_opaque "__riscv_vse32_v_f32m2"
