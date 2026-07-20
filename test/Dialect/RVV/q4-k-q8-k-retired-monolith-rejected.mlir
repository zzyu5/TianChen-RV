// RUN: not weft-opt %s 2>&1 | FileCheck %s
//
// A7 Batch 2 mutation guard: the former whole-kernel weft_rvv.q4_k_q8_k_block_dot surface must
// stay unregistered. Source-format recognition remains public, but construction
// must materialize the typed super-block body instead.
// CHECK: custom op 'weft_rvv.q4_k_q8_k_block_dot' is unknown

module {
  weft.exec.kernel @q4_k_block_dot_accepts_ggml_abi {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "q4_k_block_dot_accepts_ggml_abi", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.q4_k_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_k_q8_k_block_dot", scale_model = "per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 0 : i64, weight_dmin_byte_offset = 2 : i64, weight_scales_byte_offset = 4 : i64, weight_qs_byte_offset = 16 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64, activation_bsums_byte_offset = 260 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
