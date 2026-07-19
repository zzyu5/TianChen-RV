// RUN: not weft-opt %s --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADCOMBO
// RUN: sed 's/carrier_kind = "bare_int8", //; s/min_byte_offset = 2 : i64, //' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=NOCARRIER
//
// F1 NEGATIVE CONTROL (phase-1). The nibble-family descriptor legality gate is REAL
// (fail-closed, I7), so the falsifier's "descriptor row only" claim is not a rubber
// stamp. Two illegal descriptor shapes each fail verify CLOSED:
//   (1 badcombo)  the bare_int8 carrier (q8_0, a bare signed-int8 scale) ILLEGALLY
//                 carrying a 4-bit nibble min fact (min_byte_offset).
//   (2 nocarrier) a flat nibble-family decode leaf with the carrier_kind selector
//                 stripped (the carrier is never omitted or value_or self-supplied).

module {
  weft.exec.kernel @dequant_bad_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_bad attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_bad, sew = 32 : i64, source_kernel = "dequant_bad_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "q8_0", kind = "typed_dequantize_row_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64} {
        ^bb0(%block_index: index):
          weft_rvv.dequantize_row_decode_core %x, %y, %block_index {carrier_kind = "bare_int8", decode_model = "q8_0", min_byte_offset = 2 : i64, qk = 32 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 34 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_dequantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// BADCOMBO: must NOT carry a 4-bit nibble decode fact
// NOCARRIER: flat nibble-family dequant leaf and requires the carrier_kind
