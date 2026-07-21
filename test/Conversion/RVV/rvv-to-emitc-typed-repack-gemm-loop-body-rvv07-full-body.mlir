// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Anti-bypass / "byte-exact is contingent on the offset" divergence: rewiring the
// integer-core brick's weight_quant_byte_offset genuinely changes the emitted
// repacked-nibble addresses (SOURCED from the brick).
// RUN: weft-opt %s --weft-rvv-lower-to-emitc > %t.off32
// RUN: sed 's/weight_quant_byte_offset = 32 : i64/weight_quant_byte_offset = 40 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc > %t.off40
// RUN: not diff %t.off32 %t.off40

// M-FLAT REPACK GEMM finale M2 -- the RVV0.7.1 whole-LMUL (integer_core_lmul="m1",
// columnsPerPass==1, numHalves==1) arm of weft_rvv.typed_repack_gemm_loop_body. RVV0.7
// (XuanTie xtheadvector) has NO fractional LMUL, so the whole widening chain shifts
// up one notch (i8m1 -> i16m2 -> i32m4 -> f32m4, scale f16m2), the 16-block group is
// ONE 16-lane strip (half_lanes=16, numHalves=1), and -- because the m4 chain would
// spill all 4 f32m4 accumulators past the 32-vreg file -- ONE column is folded per
// pass (columnsPerPass=1, FOUR passes each re-decoding the shared weight nibbles).
// The region carries columnsPerPass+2 == 3 entry args (block_index + the runtime
// strip_row_offset + ONE per-column f32m4 accumulator), ONE integer-core brick
// producing ONE i32m4 sumi, ONE dual-fp16 scale-FOLD brick, and a yield naming the
// one carried-out vector. Byte-identical to the monolithic emitRepackGemmQ4_0Q8_0's
// m1 form by construction. Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv).

module {
  weft.exec.kernel @rvv_repack_gemm_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_repack_gemm attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_repack_gemm, sew = 32 : i64, source_kernel = "rvv_repack_gemm_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 16 : i64, fold_model = "lane_wise_vector_scale", integer_core_lmul = "m1", weft_rvv.loop_order = "col_outer", weft_rvv.loop_order_selection_reason = "prior", weft_rvv.tiling_variant = "plain", weft_rvv.tiling_selection_reason = "only_feasible"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m4">):
          %sumi = weft_rvv.repack_gemm_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_lane_wise_q4_x_i8_dot", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, integer_core_lmul = "m1"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m4">
          %an0 = weft_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi, %acc0, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64, integer_core_lmul = "m1"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m4">, !weft_rvv.vector<f32, "m4">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m4">
          weft_rvv.typed_repack_gemm_loop_yield %an0 : !weft_rvv.vector<f32, "m4">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_repack_gemm_kernel_rvv_repack_gemm(

// The active vl is the 16-lane whole-LMUL strip width, NOT 8.
// CHECK: literal "16"

// The whole-LMUL i8m1 sub-load + plain sign-extension decode (NO vxor).
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vsll_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// The lane-wise vwmacc i16m2 product + the lo/hi combine vwadd_vv_i32m4.
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m2"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m4"
// The f16m2 weight-scale vle16 + whole-LMUL f32m4 scale fold + store.
// CHECK: call_opaque "__riscv_vle16_v_f16m2"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// RVV0.7.1 has NO fractional LMUL -- the whole emission must be fraction-free.
// CHECK-NOT: mf2
// CHECK: return
