// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// [loop-order REALIZE 铺面] col_outer byte-exact fixture for the FLAT-family prefill-GEMM
// leaf (the emitTypedRepackGemmLoopBody flat fall-through, fold_model
// "lane_wise_vector_scale": q4_0/q4_1/q5_0/q8_0). The loop-body op is stamped
// weft_rvv.loop_order = "col_outer" with selection_reason "measured" -- the ONLY
// combination the flat path honors (siblingColGroupOuter); an unmeasured layout-prior
// stamp keeps the M1-committed row_outer default (pinned byte-identical by the existing
// row_outer fixture). Under the MEASURED col_outer stamp the flat path REALIZES the
// loop-interchange: the weight-column-GROUP loop is hoisted OUTER and the
// activation-row-GROUP loop sweeps INSIDE it. PURE loop-interchange -- the per-(y,x)
// tile is emitted by the SAME emitTile lambda as the row_outer arm, so every
// K-accumulation and fold is byte-identical (only the two enclosing ForOp headers swap).

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
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "lane_wise_vector_scale", weft_rvv.loop_order = "col_outer", weft_rvv.loop_order_selection_reason = "measured"} {
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

// [hollow-gate fix] The bare stride literals below are NOT discriminating on their own:
// both nests emit the same multiset of literals, so a forward CHECK scan matched the
// row_outer output too. The UNIQUE discriminator is the group-base callee marker
// pinned by CHECK-NEXT to each group loop header: col_outer => the WEIGHT group base
// is computed in the OUTER loop, act group base INSIDE. Negative control: flipping
// weft_rvv.loop_order to "row_outer" swaps the two markers and turns this RED.
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK: emitc.func @
// col_outer: the weight-column-GROUP loop is OUTER; its per-group weight base
// (weight_block_stride 288) is HOISTED above the row sweep.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK-NEXT: verbatim "{{.*}}callee=weight_group_base"
// CHECK: literal "288"
// the activation-row-GROUP loop sweeps INSIDE it (activation_block_stride
// 136), then the SAME hot core follows -- byte-identical to the row_outer default.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK-NEXT: verbatim "{{.*}}callee=act_group_base"
// CHECK: literal "136"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: return
