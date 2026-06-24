// OPTION-2 M1 -- KERNEL B (DIRECT repack-op emit, the EMIT-IDENTITY byte-diff target).
//
// This is the C1 fixture's EXACT ABI wrapper (@ggml_vec_dot_q4_0_q8_0_kernel /
// @ggml_vec_dot_q4_0_q8_0, the SAME 8 runtime_abi_values n,s,bs,vx,bx,vy,by,nrc)
// with the abstract tcrv_rvv.quant_contraction op REPLACED by the DIRECT
// tcrv_rvv.repack_gemv_q4_0_q8_0 op the C1 bridge builds on the RVV1.0 VLEN128
// path -- carrying the SAME x16 facts the bridge stamps (weight_block_stride=288,
// weight_interleave=16, weight_quant_byte_offset=32, half_lanes=8 => mf2, NO
// integer_core_lmul; activation_block_stride=34, activation_quant_byte_offset=2)
// and the SAME column_count operand (%bs, operand 5, mirroring A's nc wiring) but
// WITHOUT the 4 emitter-INERT audit attrs (tcrv_rvv.contraction_algorithm /
// path_materialization / path_selection_reason / weight_layout_contract).
//
// diff (this --tcrv-rvv-lower-to-emitc | mlir-translate) vs Kernel A (the
// stage-b-selection abstract op auto-lowered at march=rv64gcv then
// --tcrv-rvv-lower-to-emitc | mlir-translate) must be BYTE-IDENTICAL -- the
// EMIT-IDENTITY gate. Same wrapper => same function name/signature; the only thing
// that could differ is audit-attr leakage, which the gate proves absent.
//
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %bx = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = tcrv_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q4_0_q8_0 %vx, %vy, %s, %n, %bs, %vl {kind = "ggml_repack_gemv_q4_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
