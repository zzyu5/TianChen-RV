// JE1 pinned-flip fixture (extracted verbatim from
// test/Dialect/RVV/repack-gemv-q8-0-q8-0-dataflow.mlir section 1).
// A single monolithic weft_rvv.repack_gemv_q8_0_q8_0 op with half_lanes = 8
// authored. Drive `--weft-rvv-materialize-repack-strip-width=march=...` and the
// Schedule pass's deriveRepackHalfLanes (RVVRepackStripWidthMaterialization.cpp
// :176-177, the q8_0 GEVM arm) re-stamps half_lanes from the -march-derived VLEN:
//   march=rv64gcv          => deriveMinimumVLEN=128 => half_lanes 8
//   march=rv64gcv_zvl256b  => deriveMinimumVLEN=256 => half_lanes 16
module {
  weft.exec.kernel @repack_gemv_q80_accepts_16x1_abi {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "repack_gemv_q80_accepts_16x1_abi", status = "selected-lowering-boundary"} {
        %g = weft_rvv.repack_gemv_q8_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q8_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 544 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}
