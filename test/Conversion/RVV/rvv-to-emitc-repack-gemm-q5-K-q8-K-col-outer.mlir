// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// [loop-order REALIZE 铺面] col_outer byte-exact fixture for the q5-K prefill-GEMM leaf
// (emitRepackKQuantGemmBodyQ5K). The loop-body op is stamped weft_rvv.loop_order = "col_outer" with
// selection_reason "measured" -- the ONLY combination the sibling emitter honors
// (an unmeasured layout-prior stamp keeps the M1-committed row_outer default, which
// the existing row_outer fixture pins byte-identical). Under the MEASURED col_outer
// stamp the emitter REALIZES the loop-interchange: the weight-column-GROUP loop is
// hoisted OUTER and the activation-row-GROUP loop sweeps INSIDE it. This is a PURE
// loop-interchange -- the per-(y,x) output tile is emitted by the SAME emitTile
// lambda as the row_outer arm, so every K-accumulation and end-of-block fold is
// byte-identical (only the two enclosing ForOp headers swap; loop-interchange
// preserves the accumulation order per (m,c), same invariant proven for q4_K).

module {
  weft.exec.kernel @ggml_repack_gemm_q5_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q5_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q5_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q5_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "kquant_dmin_bsums_min", weft_rvv.loop_order = "col_outer", weft_rvv.loop_order_selection_reason = "measured"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied q5_K GEMM integer-core BRICK:
          // per-block lane-wise q5_K dot across the 4 interleaved columns -> the
          // columnsPerPass (4) per-column i32 sumi. The typed emitter re-emits the whole
          // byte-exact S6-tiled q5_K GEMM body from this brick's identity; the yield
          // passes through the carried-in per-column accs.
          %sumi:4 = weft_rvv.repack_gemm_kquant_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_kquant_core", decode_model = "q5_K", weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door fully lowers to emitc (no typed op left behind).
// [hollow-gate fix] The bare stride literals below are NOT discriminating on their own:
// both nests emit the same multiset of literals, so a forward CHECK scan matched the
// row_outer output too. The UNIQUE discriminator is the group-base callee marker
// pinned by CHECK-NEXT to each group loop header: col_outer => the WEIGHT group base
// is computed in the OUTER loop, act group base INSIDE. Negative control: flipping
// weft_rvv.loop_order to "row_outer" swaps the two markers and turns this RED.
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK: emitc.func @
// col_outer: the weight-column-GROUP loop is OUTER; its per-group weight base
// (weight_block_stride 2816) is HOISTED above the row sweep.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK-NEXT: verbatim "{{.*}}callee=weight_group_base"
// CHECK: literal "2816"
// the activation-row-GROUP loop sweeps INSIDE it (activation_block_stride
// 1168), then the SAME hot core (contraction-block loop + per-column store)
// follows -- byte-identical to the row_outer default.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK-NEXT: verbatim "{{.*}}callee=act_group_base"
// CHECK: literal "1168"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: return
