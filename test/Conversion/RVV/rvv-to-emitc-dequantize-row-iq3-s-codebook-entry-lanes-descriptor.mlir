// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/codebook_entry_lanes = 4 : i64/codebook_entry_lanes = 7 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MUTATED
// RUN: sed 's/, codebook_entry_lanes = 4 : i64//' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=NOENTRY

// JUDGMENT experiment for the entryLanes -> codebook_entry_lanes DESCRIPTOR migration
// (律2: the codebook grid ENTRY byte-width is g-axis grid geometry -- a per-format
// value -- and must live in a descriptor, READ by the mechanism body, NOT baked into
// the body as a `const int64_t entryLanes = N`). The iq3_s owned narrow-per-entry grid
// dequant body (emitDequantizeRowIQ3SVectorBody) reconstructs each grid-of-4 uint32
// entry over its 4 CONTIGUOUS grid bytes; that lane count (4) is now stamped by the
// dequant-stream front door onto the dequantize_row_decode_core brick as the
// codebook_entry_lanes descriptor and READ at emit -- it drives the grid-entry pointer
// stride (idx * entryLanes) AND the narrow-pipeline vl. This test is falsify-proof:
//   (1) normal value 4 -> the grid-entry stride + vl literal is "4" (byte-exact golden);
//   (2) mutating the descriptor 4 -> 7 CHANGES the emitted C (the same operands become
//       "7") -- a byte-diff the mutation cannot fake away, proving the emit CONSUMES the
//       descriptor rather than baking the constant;
//   (3) with the descriptor ABSENT the op FAILS CLOSED at verify time with a named
//       diagnostic -- it is NEVER value_or self-supplied to the g-axis (anti-decoration).

module {
  weft.exec.kernel @dequant_iq3_s_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @dequant_iq3_s attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = weft_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %k {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq3_s, sew = 32 : i64, source_kernel = "dequant_iq3_s_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "iq3_s", kind = "typed_dequantize_row_loop_body", qk = 256 : i64, weight_block_stride = 110 : i64} {
        ^bb0(%block_index: index):
          weft_rvv.dequantize_row_decode_core %x, %y, %block_index {decode_model = "iq3_s", dequant_mechanism = "grid-lookup", grid_decode_leaf = "iq3-s", qk = 256 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 110 : i64, codebook_entry_lanes = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_dequantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// NORMAL (codebook_entry_lanes = 4): the grid-entry pointer (cast to const int8_t *) is
// loaded by a narrow unit-stride vle8 whose vl is the entry-lane descriptor value "4".
// CHECK: emitc.func @weft_emitc_dequant_iq3_s_kernel_dequant_iq3_s(
// CHECK: %[[GP:.*]] = cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
// CHECK-NEXT: %[[VL:.*]] = literal "4" : !emitc.opaque<"size_t">
// CHECK-NEXT: verbatim{{.*}}callee=__riscv_vle8_v_i8mf4
// CHECK-NEXT: call_opaque "__riscv_vle8_v_i8mf4"(%[[GP]], %[[VL]])

// MUTATED (descriptor 4 -> 7): the SAME grid-entry load's vl becomes "7" -- the emit
// really READS the descriptor (falsify-proof byte-diff), it did not bake a 4.
// MUTATED: %[[GP2:.*]] = cast %{{.*}} : !emitc.ptr<!emitc.opaque<"const uint8_t">> to !emitc.ptr<!emitc.opaque<"const int8_t">>
// MUTATED-NEXT: %[[VL2:.*]] = literal "7" : !emitc.opaque<"size_t">
// MUTATED-NEXT: verbatim{{.*}}callee=__riscv_vle8_v_i8mf4
// MUTATED-NEXT: call_opaque "__riscv_vle8_v_i8mf4"(%[[GP2]], %[[VL2]])

// NOENTRY (descriptor ABSENT): fail closed at verify time with a named diagnostic --
// the g-axis geometry is never value_or self-supplied (anti-decoration).
// NOENTRY: selected grid leaf requires construction-owned codebook_entry_lanes geometry
