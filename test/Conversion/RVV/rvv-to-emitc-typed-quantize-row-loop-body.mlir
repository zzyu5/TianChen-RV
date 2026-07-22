// RUN: weft-opt %s | FileCheck %s
// RUN: sed 's/kind = "typed_quantize_row_loop_body"/kind = "plain_quant_loop"/' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/encode_model = "q8_0"/encode_model = "renamed-provenance"/g' %s | weft-opt | FileCheck %s --check-prefix=PROVENANCE
// RUN: sed 's/quantize_leaf = #weft_rvv<quantize_row_leaf q8_0>, //g' %s | not weft-opt 2>&1 | FileCheck %s --check-prefix=MISSINGLEAF

// G3 line-B quantize FRONT DOOR -- the streaming CONSTRUCTED f32->QUANT sibling of
// the dequantize_row front door (the MIRROR direction: dequant reads block_qX and
// writes an f32 row; quantize reads an f32 row and writes the block_qX byte buffer).
// weft_rvv.typed_quantize_row_loop_body carries the ggml quantize_row encoder's
// `nb = k / QK` block loop as ONE region-carrying op whose SOLE entry argument is the
// block_index induction variable (unlike the block-dot loop op there is NO
// loop-carried accumulator: each block's amax reduction is block-local and the encoder
// STORES each block's scale + int8 quants straight through the output byte cursor, so
// the region is terminated by the VOID weft_rvv.typed_quantize_row_loop_yield). The
// per-block encode is the separate typed brick weft_rvv.quantize_row_encode_core
// (typed quantize_leaf q8_0; encode_model is provenance only). This lit locks the
// typed-region verifier contract: the leaf is required and parent/core agree; the
// full construct-from-
// abstract -> byte-exact C lowering is proven in the per-format
// rvv-to-emitc-ggml-quantize-row-q8-{0,1,k}.mlir lits (route_source_op provenance
// CHECK). The block_q8_0 encode is byte-exact to ggml's EXACT RVV method
// (riscv/quants.c: vfncvt/vncvt round-to-nearest-EVEN + native (_Float16) cast).

module {
  weft.exec.kernel @quant_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @quant_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "uint8_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_quantize_row_loop_body %x, %vy, %n attributes {quantize_leaf = #weft_rvv<quantize_row_leaf q8_0>, encode_model = "q8_0", kind = "typed_quantize_row_loop_body", qk = 32 : i64, block_stride = 34 : i64} {
        ^bb0(%block_index: index):
          // The per-block typed q8_0 leaf: the whole per-block
          // amax/scale/narrow is emitter-inlined by the brick lowering.
          weft_rvv.quantize_row_encode_core %x, %vy, %block_index {quantize_leaf = #weft_rvv<quantize_row_leaf q8_0>, encode_model = "q8_0", qk = 32 : i64, block_stride = 34 : i64, scale_byte_offset = 0 : i64, quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
          weft_rvv.typed_quantize_row_loop_yield
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// The valid typed region round-trips (parses + verifies + prints back): the loop op,
// the per-block encode brick, and the VOID yield.
// CHECK: weft_rvv.typed_quantize_row_loop_body
// CHECK: weft_rvv.quantize_row_encode_core
// CHECK: weft_rvv.typed_quantize_row_loop_yield

// The bounded surface is fail-closed on loop kind and missing typed leaf. Provenance
// text can change without changing compute.
// BADKIND: currently supports only kind "typed_quantize_row_loop_body"
// PROVENANCE: quantize_leaf = #weft_rvv<quantize_row_leaf q8_0>
// PROVENANCE: encode_model = "renamed-provenance"
// MISSINGLEAF: requires attribute 'quantize_leaf'
