// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/dual-fp16-per-block-d_x.d_y/five-bit-offset-binary/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADMODEL
// RUN: sed 's/dual_fp16_per_block_scale_product/plain_i8_product/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/scale_model = "dual-fp16-per-block-d_x.d_y"/scale_model = "dual-fp16-per-block-d_x.d_y", lhs_block_stride = 34 : i64/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADSTRIDEONLY

// M-FLAT milestone brick 1/5 -- the per-block dual-fp16 SCALE reconstruction
// primitive (dismantling wall 1 of the four-walls flat-block-dot analysis:
// "no scalar-fp16-load / fp16->f32-convert / scalar-mul typed op"). The typed
// weft_rvv.block_fp16_scale_product op reads two per-block fp16 scale headers
// (d_x, d_y), converts each fp16 -> f32, and multiplies into one f32 scalar
// `f32(d_x) * f32(d_y)` (ggml's q8_0 scale order: the scales are multiplied
// FIRST). Its lowering is BYTE-IDENTICAL to the two scalar fp16->fp32 reads +
// float multiply the monolithic block-dot emitters produce inline for the
// `d_x * d_y` fold, because both share the exact kFp16ScaleReadCallee
// call_opaque callee and the float emitc.mul spelling. This is an
// emit-consistency ("CORE == emission-plans") lit, NOT a numerical oracle:
// f32 fully covers the fp16 domain, so the reconstruction is byte-exact by
// construction; bit-exact-vs-ggml is pending-hardware (ssh rvv).

module {
  weft.exec.kernel @rvv_block_fp16_scale_product_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_block_fp16_scale_product attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "block-fp16-scale-product:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "block-fp16-scale-product:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "block-fp16-scale-product:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
        %scale = weft_rvv.block_fp16_scale_product %lhs, %rhs {kind = "dual_fp16_per_block_scale_product", scale_model = "dual-fp16-per-block-d_x.d_y"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_block_fp16_scale_product_kernel_rvv_block_fp16_scale_product(%[[VX:.*]]: !emitc.ptr<!emitc.opaque<"const int8_t">>, %[[VY:.*]]: !emitc.ptr<!emitc.opaque<"const int8_t">>, %{{.*}}: !emitc.opaque<"size_t">)
// The typed op lowers to the SAME two scalar fp16->fp32 reads the monolithic
// block-dot emitters produce inline (the kFp16ScaleReadCallee call_opaque).
// CHECK: %[[DX:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[VX]]) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"float">
// CHECK: %[[DY:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[VY]]) : (!emitc.ptr<!emitc.opaque<"const int8_t">>) -> !emitc.opaque<"float">
// The scale product d_x * d_y (ggml's q8_0 order): one scalar float multiply.
// CHECK: mul %[[DX]], %[[DY]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: return

// The bounded surface is fail-closed on the scale model and kind facts (I7).
// BADMODEL: currently supports only scale_model "dual-fp16-per-block-d_x.d_y"
// BADKIND: currently supports only kind "dual_fp16_per_block_scale_product"
// The loop-capable extension is additive: in the single-block form (block_index
// absent) the per-block stride attrs are meaningless and rejected fail-closed.
// BADSTRIDEONLY: lhs_block_stride / rhs_block_stride are only valid with a present block_index
