// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Anti-bypass / "byte-exact is contingent on the offset" divergence: rewiring the
// dual-fp16 FOLD brick's weight_scale_byte_offset to a NON-ZERO value genuinely
// changes the emitted scale address (the fold's within-block offset is SOURCED
// from the brick, not silently hardcoded to 0), so the offset!=0 emit is NO LONGER
// byte-identical to the offset-0 emit.
// RUN: weft-opt %s --weft-rvv-lower-to-emitc > %t.off0
// RUN: sed 's/weight_scale_byte_offset = 0 : i64/weight_scale_byte_offset = 4 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc > %t.off4
// RUN: not diff %t.off0 %t.off4

// M-FLAT REPACK loop-scaffold -- the FULL-BODY RVV0.7 whole-LMUL arm of
// weft_rvv.typed_repack_gemv_loop_body, the SOLE representation of the q4_0
// 16x1-repacked GEVM (the monolithic emitRepackGemvQ4_0Q8_0 is retired; the
// region-vs-monolith byte-exactness was EMPIRICALLY proven at Phase B before
// retirement). The pre-ratification RVV0.7.1 generation (XuanTie xtheadvector) has
// NO fractional LMUL, so integer_core_lmul="m1" shifts the whole widening chain up
// one notch (i8m1 -> i16m2 -> i32m4 -> f32m4, f16 scale m2) as ONE 16-lane strip
// (half_lanes=16, numHalves==1). The region carries TWO entry args (block_index +
// ONE f32m4 accumulator), ONE integer-core brick producing ONE i32m4 sumi, ONE
// dual-fp16 scale-FOLD brick, and a yield naming the carried-out f32m4 vector. The
// lowering derives l8/l16/l32 = m1/m2/m4 from the loop op's integer_core_lmul and
// drives the CORE + FOLD from the SHARED leaves; the CHECK below pins the full
// whole-LMUL node sequence. Numerical bit-exact-vs-ggml is pending-hardware.

module {
  weft.exec.kernel @rvv_repack_gemv_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_repack_gemv attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 16 : i64, integer_core_lmul = "m1", fold_model = "lane_wise_vector_scale"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m4">):
          // ONE i32m4 sumi from the integer-core brick, THEN ONE dual-fp16 scale
          // FOLD brick -- the whole-LMUL one-strip form (i8m1 -> i16m2 -> i32m4 ->
          // f32m4).
          %sumi = weft_rvv.repack_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %block_index : index {kind = "repack_lane_wise_q4_x_i8_dot", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, integer_core_lmul = "m1"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m4">
          %an0 = weft_rvv.repack_dual_fp16_scale_fold %vx, %vy, %sumi, %acc0, %vl block %block_index : index {kind = "repack_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64, integer_core_lmul = "m1"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m4">, !weft_rvv.vector<f32, "m4">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m4">
          weft_rvv.typed_repack_gemv_loop_yield %an0 : !weft_rvv.vector<f32, "m4">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_repack_gemv_kernel_rvv_repack_gemv(

// The active vl is the 16-lane whole-LMUL strip width.
// CHECK: literal "16"

// ONE 16-lane f32m4 accumulator (rows 0..15): a SINGLE vfmv_v_f_f32m4(0.0f).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m4"

// ===== The region integer CORE, whole-LMUL m1 (byte-exact to the monolith) =====
// ONE 16-lane i16m2 lo + ONE 16-lane i16m2 hi accumulator seed.
// CHECK: call_opaque "__riscv_vmv_v_x_i16m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m2"
// ONE disjoint whole-LMUL i8m1 sub-load; the plain sign-extension decode on i8m1.
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vsll_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// TWO lane-wise vwmacc into i16m2 and ONE lo/hi vwadd combine into i32m4.
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m2"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m4"
// The RVV1.0 fractional (mf2 / f32m2) spellings must NOT appear on this arm.
// CHECK-NOT: __riscv_vle8_v_i8mf2
// CHECK-NOT: __riscv_vfmacc_vv_f32m2

// ===== The region dual-fp16 scale FOLD, whole-LMUL m1 (byte-exact to the monolith):
// f16m2 weight scale -> _Float16 act scale -> f32m4 vfwmul / vfcvt / vfmacc. =====
// CHECK: call_opaque "__riscv_vle16_v_f16m2"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m4"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m4"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m4"

// The single 16-lane whole-LMUL vector store vse32_v_f32m4 (NO horizontal reduction).
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// CHECK: return
