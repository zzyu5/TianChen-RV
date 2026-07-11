// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_dequantize_row_loop_body"/kind = "plain_dequant_loop"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/"q8_0"/"bf16"/g' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADDECODE

// G3 line-B dequant FRONT DOOR -- the streaming CONSTRUCTED sibling of the flat
// block-dot loop scaffold. tcrv_rvv.typed_dequantize_row_loop_body carries the ggml
// dequantize_row decode's `nb = k / QK` block loop as ONE region-carrying op whose
// SOLE entry argument is the block_index induction variable (unlike the block-dot
// loop op there is NO loop-carried accumulator: the decode STORES each f32 straight
// through the output pointer, so the region is terminated by the VOID
// tcrv_rvv.typed_dequantize_row_loop_yield). The per-block decode is the separate
// typed brick tcrv_rvv.dequantize_row_decode_core (decode_model "q8_0"): read the
// fp16 block scale via the (float)*(const _Float16 *) seam, then the bare signed-int8
// scale y[j] = qs[j] * d over all 32 block lanes (the load sign-extends).
//
// This is the FAMILY-HEAD of the 23-format dequantize_row spectrum: the shared
// front-door construction path is proven ONCE here (abstract dequantize_row ->
// typed region -> byte-exact C); subsequent formats swap only the decode leaf. The
// emit is BYTE-EXACT to the retired dispatch-wired q8_0 monolith (emitGgmlDequantizeRow
// bareInt8 branch) via the SHARED body emitter emitDequantizeRowQ8_0BodyShared,
// modulo only the source-op provenance token. Byte-exact-vs-ggml-reference
// dequantize_row_q8_0 (a scalar AoS block loop; no reduction). Numerical
// bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested here.

module {
  tcrv.exec.kernel @dequant_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q8_0, sew = 32 : i64, source_kernel = "dequant_q8_0_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "q8_0", kind = "typed_dequantize_row_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64} {
        ^bb0(%block_index: index):
          // The per-block decode leaf (decode_model "q8_0"): the whole per-block
          // scalar decode is emitter-inlined by the brick lowering (the streaming
          // analog of q1_0/nvfp4's flat single-core-brick emit).
          tcrv_rvv.dequantize_row_decode_core %x, %y, %block_index {decode_model = "q8_0", qk = 32 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 34 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
          tcrv_rvv.typed_dequantize_row_loop_yield
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_q8_0_kernel_dequant_q8_0(
// The AoS block count nb = k / 32 + the block loop.
// CHECK: %[[NB:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: for %{{.*}} = %{{.*}} to %[[NB]] step %{{.*}}  : !emitc.opaque<"size_t"> {
// The fp16 block scale seam (fcvt.s.h).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The signed-int8 read (sign-extending) + the bare block scale y[j] = qs[j] * d.
// CHECK: !emitc.opaque<"const int8_t">
// CHECK: %{{.*}} = mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: return

// The bounded surface is fail-closed on the loop kind and the decode_model leaf (I7).
// BADKIND: currently supports only kind "typed_dequantize_row_loop_body"
// BADDECODE: is not a CONSTRUCTED dequantize_row decode
