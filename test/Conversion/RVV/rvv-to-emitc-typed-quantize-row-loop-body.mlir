// RUN: tcrv-opt %s | FileCheck %s
// RUN: sed 's/kind = "typed_quantize_row_loop_body"/kind = "plain_quant_loop"/' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/"q8_0"/"iq4_nl"/g' %s | not tcrv-opt 2>&1 | FileCheck %s --check-prefix=BADENCODE

// G3 line-B quantize FRONT DOOR -- the streaming CONSTRUCTED f32->QUANT sibling of
// the dequantize_row front door (the MIRROR direction: dequant reads block_qX and
// writes an f32 row; quantize reads an f32 row and writes the block_qX byte buffer).
// tcrv_rvv.typed_quantize_row_loop_body carries the ggml quantize_row encoder's
// `nb = k / QK` block loop as ONE region-carrying op whose SOLE entry argument is the
// block_index induction variable (unlike the block-dot loop op there is NO
// loop-carried accumulator: each block's amax reduction is block-local and the encoder
// STORES each block's scale + int8 quants straight through the output byte cursor, so
// the region is terminated by the VOID tcrv_rvv.typed_quantize_row_loop_yield). The
// per-block encode is the separate typed brick tcrv_rvv.quantize_row_encode_core
// (encode_model "q8_0"). This lit locks the typed-region VERIFIER contract (I7
// fail-closed on the loop kind and the encode_model leaf); the full construct-from-
// abstract -> byte-exact C lowering is proven in the per-format
// rvv-to-emitc-ggml-quantize-row-q8-{0,1,k}.mlir lits (route_source_op provenance
// CHECK). The block_q8_0 encode is byte-exact to ggml's EXACT RVV method
// (riscv/quants.c: vfncvt/vncvt round-to-nearest-EVEN + native (_Float16) cast).

module {
  tcrv.exec.kernel @quant_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @quant_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "uint8_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @quant_q8_0, sew = 32 : i64, source_kernel = "quant_q8_0_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_quantize_row_loop_body %x, %vy, %n attributes {encode_model = "q8_0", kind = "typed_quantize_row_loop_body", qk = 32 : i64, block_stride = 34 : i64} {
        ^bb0(%block_index: index):
          // The per-block encode leaf (encode_model "q8_0"): the whole per-block
          // amax/scale/narrow is emitter-inlined by the brick lowering.
          tcrv_rvv.quantize_row_encode_core %x, %vy, %block_index {encode_model = "q8_0", qk = 32 : i64, block_stride = 34 : i64, scale_byte_offset = 0 : i64, quant_byte_offset = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
          tcrv_rvv.typed_quantize_row_loop_yield
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// The valid typed region round-trips (parses + verifies + prints back): the loop op,
// the per-block encode brick, and the VOID yield.
// CHECK: tcrv_rvv.typed_quantize_row_loop_body
// CHECK: tcrv_rvv.quantize_row_encode_core
// CHECK: tcrv_rvv.typed_quantize_row_loop_yield

// The bounded surface is fail-closed on the loop kind and the encode_model leaf (I7).
// BADKIND: currently supports only kind "typed_quantize_row_loop_body"
// BADENCODE: is not a CONSTRUCTED quantize_row encode
