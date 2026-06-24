// option-2 stage-C1b: the ISOLATED bit-exact PACKER (plain block_q4_0 ->
// block_q4_0x16, ggml make_block_q4_0x16) as one bounded typed RVV dataflow
// step. PURE scalar byte gather + ^0x88 -- the materialization-capability proof
// (the compiler PRODUCES the x16 layout it DECLARES). e2e-REDUNDANT (ggml packs
// at model-load); NEVER a kernel/perf/e2e win. Lowered with
// `tcrv-opt --tcrv-rvv-lower-to-emitc`; emitted symbol
// tcrv_emitc_ggml_pack_q4_0_to_q4_0x16_kernel_ggml_pack_q4_0_to_q4_0x16.
module {
  tcrv.exec.kernel @ggml_pack_q4_0_to_q4_0x16_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_pack_q4_0_to_q4_0x16 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %nblocks = tcrv_rvv.runtime_abi_value {c_name = "nblocks", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nblocks", role = "runtime-element-count"} : index
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-plain-src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "uint8_t *", ownership = "target-export-abi-owned", purpose = "q4x16-dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %nblocks {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_pack_q4_0_to_q4_0x16, sew = 32 : i64, source_kernel = "ggml_pack_q4_0_to_q4_0x16_kernel", status = "selected-lowering-boundary"} {
        %p = tcrv_rvv.pack_q4_0_to_q4_0x16 %src, %dst, %nblocks, %vl {kind = "ggml_pack_q4_0_to_q4_0x16", qk = 32 : i64, src_block_stride = 18 : i64, dst_block_stride = 288 : i64, src_quant_byte_offset = 2 : i64, dst_quant_byte_offset = 32 : i64, weight_interleave = 16 : i64, xor_mask = 136 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
