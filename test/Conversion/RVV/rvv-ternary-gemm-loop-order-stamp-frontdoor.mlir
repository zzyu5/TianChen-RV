// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s

// [C1 loop-order 全格消费] FRONT-DOOR STAMP regression guard for the TERNARY GEMM leaf.
//
// The loop-order schedule axis and the SP4 output-tiling axis are ORTHOGONAL, but they
// used to share one gate: the loop_order stamp sat INSIDE stampTilingSelection AFTER its
// `classifyTilingBottleneckShape(foldModel)` fail-safe early return. "ternary_single_fp16_scale"
// has no output-tiling axis => classifies to std::nullopt => the early return fired and
// the loop-order stamp was silently skipped. The ternary GEMM builder ALSO never called
// the stamp helper at all (the one typed repack-GEMM construction point that stamped
// NEITHER axis). Net effect: the ternary leaves reached an emitter that DOES read
// weft_rvv.loop_order with the attr ABSENT, defaulting to row_outer by accident rather
// than by selection.
//
// This pins the DECOUPLING: the ternary prefill GEMM now carries a first-class,
// capability-keyed loop-order selection. EMISSION-NEUTRAL by construction -- tq2_0's
// repacked weight panel stride (1056) is < the q8_K activation panel stride (1168), so
// repackColGroupOuterForLayout returns false and the layout prior resolves to row_outer:
// the SAME nest that already shipped. Verified byte-exact on the emitted C for all 18
// repack-GEMM formats across the fix.
//
// NEGATIVE CONTROL (this fixture is not hollow): re-coupling the two axes -- i.e. moving
// the loop_order stamp back below the tiling classifier's early return, or dropping the
// ternary builder's stampScheduleSelections call -- removes these attrs and turns the
// three CHECK lines below RED.

// The ternary GEMM leaf is stamped with a first-class loop-order SELECTION...
// CHECK: weft_rvv.typed_repack_gemm_loop_body
// ...the layout prior resolves to the already-shipped row_outer nest (weight 1056 < act 1168):
// CHECK-SAME: weft_rvv.loop_order = "row_outer"
// ...on an HONEST cold-start reason (no ternary board seed exists):
// CHECK-SAME: weft_rvv.loop_order_selection_reason = "prior"
// ...carrying the full [D-4] attribution record over the SAME declared-instance hash:
// CHECK-SAME: weft_rvv.loop_order_selection_record = "{{.*}}\22chosen\22:\22row_outer\22{{.*}}\22kernel\22:\22tq2_0\22{{.*}}\22reason\22:\22prior\22

module {
  weft.exec.kernel @ggml_repack_gemm_tq2_0_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_tq2_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_tq2_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_tq2_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT ternary PREFILL request: PLAIN block_tq2_0 (stride 66) x PLAIN
        // block_q8_K (stride 292), qk 256, scale_model = the base ternary tq2_0 WHAT,
        // m_regime = prefill, block_dot_compute_heavy = true (routes REPACK => GEMM).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "tq2_0", scale_model = "superblock-d.fp16-single-scale-2bit-ternary-nomin", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
