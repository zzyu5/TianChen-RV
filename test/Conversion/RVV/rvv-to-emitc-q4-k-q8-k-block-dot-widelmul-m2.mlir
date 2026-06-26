// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// L1 widening-chain DISCRIMINATING fixture (the {mf2,m1}-only vs {mf2,m1,m2}
// divergence bug seam). The q4_K K4b integer core is the ONE in-tree path that
// is BOTH verifier-legal at integer_core_lmul = "m2" AND routed through the
// single-source detail::deriveWideningChain helper. Stamped at "m2", the
// i8 -> i16 -> i32 widening chain must climb TWO full rungs:
//   l8  = m2  (the i8m2 strip / sub-block)
//   l16 = m4  (the vwmul i16 product)
//   l32 = m8  (the vwmacc i32 deferred accumulator + its vmv_v_x seed)
//
// The OLD inline derivation `l16 = (coreLmul == "m1") ? "m2" : "m1"` (and its
// l32 sibling) silently collapses the "m2" base into the "m1"/else branch,
// emitting i16m1 / i32m2 instead. deriveWideningChain("m2") returns
// {l8=m2, l16=m4, l32=m8, stripWidth=32, foldGroups=4}, so the wide callees
// below appear and the buggy-shape callees do NOT. This fixture FAILS against
// the old `?"m2":"m1"` logic and PASSES with deriveWideningChain.
//
// (No in-tree q4_K fixture stamps "m2", so the byte-exact golden set is mf2/m1
// shaped and unaffected; this file only exercises the m2 rung for the unit
// check.)

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_K_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q4_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_k_q8_k_block_dot", integer_core_lmul = "m2", scale_model = "per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_dmin_byte_offset = 2 : i64, weight_scales_byte_offset = 4 : i64, weight_qs_byte_offset = 16 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q4_K_q8_K_kernel_ggml_vec_dot_q4_K_q8_K(
// The m2 integer-core widening chain: the i32m8 aux32 seed, the vwmul i16m4
// product, the vwmacc i32m8 deferred accumulate. deriveWideningChain("m2")
// gives l16 = m4 and l32 = m8.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m8"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m4"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m8"
// The OLD buggy `?"m2":"m1"` derivation would emit i16m1 / i32m2 here instead;
// assert those buggy-shape integer-MAC callees never appear.
// CHECK-NOT: call_opaque "__riscv_vwmul_vv_i16m1"
// CHECK-NOT: call_opaque "__riscv_vwmacc_vx_i32m2"
