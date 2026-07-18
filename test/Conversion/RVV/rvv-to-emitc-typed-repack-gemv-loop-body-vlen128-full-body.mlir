// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Anti-bypass / "byte-exact is contingent on the offset" divergence: rewiring the
// per-strip dual-fp16 FOLD bricks' weight_scale_byte_offset to a NON-ZERO value
// genuinely changes the emitted scale addresses (the fold's within-block offset is
// SOURCED from the brick, not silently hardcoded to 0), so the offset!=0 emit is
// NO LONGER byte-identical to the offset-0 emit.
// RUN: weft-opt %s --weft-rvv-lower-to-emitc > %t.off0
// RUN: sed 's/weight_scale_byte_offset = 0 : i64/weight_scale_byte_offset = 4 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc > %t.off4
// RUN: not diff %t.off0 %t.off4

// M-FLAT REPACK loop-scaffold -- the FULL-BODY numHalves==2 (VLEN=128
// two-8-lane-halves) arm of weft_rvv.typed_repack_gemv_loop_body, the SOLE
// representation of the q4_0 16x1-repacked GEVM (the monolithic
// emitRepackGemvQ4_0Q8_0 is retired; the region-vs-monolith byte-exactness was
// EMPIRICALLY proven at Phase B before retirement). The region carries
// numHalves+1 == 3 entry args (block_index + TWO per-strip f32m2 accumulators),
// ONE integer-core brick producing TWO per-strip i32m2 sumi (a VARIADIC result
// group), TWO dual-fp16 scale-FOLD bricks (one per strip), and a yield naming BOTH
// carried-out vectors. The lowering GATES the emit on the core brick's
// block_index-tied anti-bypass, on EACH fold brick's block_index + base +
// sumi(result h) + acc(region arg 1+h) dataflow tie, and on the yield naming both
// folds' acc_next, then drives the CORE + FOLD from the SHARED
// emitRepackQ4LaneWiseIntegerCore / emitRepackDualFp16ScaleFold leaves. The
// half_lanes=8 two-strip body's CHECK below pins the full node sequence.
// Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv).

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
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_repack_gemv, sew = 32 : i64, source_kernel = "rvv_repack_gemv_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "lane_wise_vector_scale"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // TWO per-strip i32m2 sumi from ONE integer-core brick (variadic results:
          // strip 0 = rows 0..7, strip 1 = rows 8..15), THEN TWO dual-fp16 scale
          // FOLD bricks (one per strip, each folding its sumi + its carried acc).
          %sumi:2 = weft_rvv.repack_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %block_index : index {kind = "repack_lane_wise_q4_x_i8_dot", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          %an0 = weft_rvv.repack_dual_fp16_scale_fold %vx, %vy, %sumi#0, %acc0, %vl block %block_index : index {kind = "repack_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          %an1 = weft_rvv.repack_dual_fp16_scale_fold %vx, %vy, %sumi#1, %acc1, %vl block %block_index : index {kind = "repack_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %an0, %an1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_repack_gemv_kernel_rvv_repack_gemv(

// The active vl is the 8-lane half-strip width (VLEN=128), NOT 16.
// CHECK: %[[VL:.*]] = literal "8" : !emitc.opaque<"size_t">

// nb = n / QK; nc_groups = nc / weight_interleave.
// CHECK: div
// CHECK: div

// The OUTER weight-column-GROUP loop over nc/16; per-group weight base (stride 288).
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "288"

// TWO 8-lane f32m2 strip accumulators (rows 0..7 then rows 8..15), each seeded per
// group with vfmv_v_f_f32m2(0.0f) -- the numHalves==2 multi-accumulator contrast.
// CHECK: %[[SUMF0:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: %[[SUMF1:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"

// The inner contraction-BLOCK loop over nb; plain q8_0 activation stride 34.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "34"

// ===== The region integer CORE, numHalves==2 (byte-exact to the monolith) =====
// FOUR 8-lane i16m1 lo/hi accumulator seeds (a_lo, a_hi, b_lo, b_hi).
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// TWO disjoint repacked i8mf2 sub-loads per nibble step (rows 0..7 at qs[i*16+0],
// rows 8..15 at qs[i*16+8]) -- the two-8-lane-halves load pair.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// FOUR lane-wise vwmacc (two strips x lo/hi) and TWO lo/hi vwadd combines (two sumi).
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// NO offset-binary xor (the repacked nibbles carry the ^0x88 bias already).
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8

// ===== The region dual-fp16 scale FOLD, one per strip (byte-exact to the monolith).
// TWO vle16 weight scales, ONE shared _Float16 act scale, TWO vfwmul / vfcvt /
// vfmacc, each folding its strip's sumi into its strip's accumulator. =====
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"

// TWO per-strip lane-wise vector stores (NO horizontal reduction): rows 0..7 to
// s + x*16 + 0 and rows 8..15 to s + x*16 + 8.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
