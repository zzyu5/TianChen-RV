// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// Anti-bypass / "byte-exact is contingent on the offset" divergence: rewiring the
// integer-core brick's weight_quant_byte_offset to a NON-32 value genuinely changes
// the emitted repacked-nibble addresses (the core's within-block offset is SOURCED
// from the brick, not silently hardcoded), so the offset!=32 emit is NO LONGER
// byte-identical to the offset-32 emit.
// RUN: weft-opt %s --weft-rvv-lower-to-emitc > %t.off32
// RUN: sed 's/weight_quant_byte_offset = 32 : i64/weight_quant_byte_offset = 40 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc > %t.off40
// RUN: not diff %t.off32 %t.off40

// M-FLAT REPACK GEMM finale M2 -- the FULL-BODY VLEN=128 (columnsPerPass==4,
// numHalves==2) arm of weft_rvv.typed_repack_gemm_loop_body, the region-carrying
// typed sibling of the monolithic emitRepackGemmQ4_0Q8_0 (the hand-authored
// weft_rvv.repack_gemm_q4_0_q8_0 op). The region carries columnsPerPass+2 == 6 entry
// args (block_index + the runtime strip_row_offset + FOUR per-column f32m2
// accumulators), ONE integer-core brick producing FOUR per-column i32m2 sumi (a
// VARIADIC result group over the block_q8_0x4 interleaved columns of ONE runtime
// strip), FOUR dual-fp16 scale-FOLD bricks (one per column), and a yield naming all
// four carried-out vectors. The lowering GATES the emit on the core brick's
// block_index + strip_row_offset anti-bypass, on EACH fold brick's block_index +
// strip + base + sumi(result c) + acc(region arg 2+c) dataflow tie, and on the yield
// naming the folds' acc_next, then wraps the region's inner block loop in the SAME
// four outer loops selected by the complete schedule plan (column-group / row-group /
// runtime strip /
// column pass) and drives the CORE + FOLD from the SHARED
// emitRepackGemmQ4LaneWiseIntegerCore / emitRepackGemmDualFp16ScaleFold leaves --
// byte-identical to emitRepackGemmQ4_0Q8_0's kernel body by construction.
// Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv).

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
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "lane_wise_vector_scale", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // FOUR per-column i32m2 sumi from ONE integer-core brick (variadic
          // results: one per interleaved activation column of ONE runtime strip),
          // THEN FOUR dual-fp16 scale FOLD bricks (one per column, each folding its
          // sumi + its carried acc).
          %sumi:4 = weft_rvv.repack_gemm_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_lane_wise_q4_x_i8_dot", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          %an0 = weft_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#0, %acc0, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          %an1 = weft_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#1, %acc1, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          %an2 = weft_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#2, %acc2, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          %an3 = weft_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#3, %acc3, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %an0, %an1, %an2, %an3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_repack_gemm_kernel_rvv_repack_gemm(

// The active vl is the 8-lane half-strip width (VLEN=128), NOT 16.
// CHECK: %[[VL:.*]] = literal "8" : !emitc.opaque<"size_t">

// nb = n / QK; nr_groups = nr / 4; nc_groups = nc / 16.
// CHECK: div
// CHECK: div
// CHECK: div

// The layout prior selects the OUTER weight-column-GROUP loop over nc/16;
// per-group weight base (stride 288) is hoisted above the row sweep.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT: verbatim "{{.*}}callee=weight_group_base"
// CHECK: literal "288"
// The activation-row-GROUP loop over nr/4 is inside it; per-group activation
// base uses the block_q8_0x4 stride 136.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT: verbatim "{{.*}}callee=act_group_base"
// CHECK: literal "136"
// The RUNTIME strip loop over the numHalves == 2 disjoint 8-lane strips.
// CHECK: for %[[H:.*]] = %{{.*}} to %{{.*}} step

// The 4x8 f32m2 accumulator set: four vfmv_v_f_f32m2(0.0f, 8) seeds (columnsPerPass
// == 4 columns folded in ONE pass).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"

// The inner contraction-BLOCK loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step

// ===== The region integer CORE, columnsPerPass==4 (byte-exact to the monolith).
// The 4x{lo,hi} i16m1 lane accumulator seeds. =====
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The nibble-step loop over the 16 weight bytes; ONE disjoint repacked i8mf2
// sub-load per step (one runtime strip), plain sign-extension decode (NO vxor).
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsll_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// The lane-wise vwmacc accumulate against the scalar q8 quants (NO cross-lane
// vredsum) and the lo/hi vwadd_vv_i32m2 combine.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// NO offset-binary xor (the repacked nibbles carry the ^0x88 bias already).
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8

// ===== The region dual-fp16 scale FOLD, one per column (byte-exact to the monolith).
// vle16 the per-strip weight scales, per-column _Float16 act scale, vfwmul / vfcvt /
// vfmacc. =====
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"

// The 4x8 vector store vse32_v_f32m2 (NO per-block scalar *s store).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
