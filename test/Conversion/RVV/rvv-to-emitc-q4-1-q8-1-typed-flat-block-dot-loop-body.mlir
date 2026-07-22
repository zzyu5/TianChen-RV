// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/fold_model = "scale_plus_min"/fold_model = "unsupported_fold"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADFOLD
// Anti-bypass: the MIN brick's byte offset is READ to place the m_x read address,
// so mutating lhs_min_byte_offset 2 -> 8 must MOVE the emitted m_x read to `xb+8`
// (a gate-only build reading a fixed descriptor offset would stay at 2). The
// literal "8" does not appear anywhere in the byte-exact baseline emit.
// RUN: sed 's/lhs_min_byte_offset = 2 : i64/lhs_min_byte_offset = 8 : i64/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MINOFF

// The FULL q4_1 (ggml Q4_1 x Q8_1) typed flat block-dot loop body -- the Family-B
// sibling of the q8_0/q4_0 typed loop. It exercises the region-driven OP-BY-OP
// emit of the two Family-B deltas over q4_0: (1) the UNSIGNED-nibble packed-i4 x
// i8 integer core (u8 weight load + vand 0x0F / vsrl 0x04 / reinterpret /
// vwmul+vwmacc, NO offset-binary xor bias), and (2) the per-block MIN correction
// brick (m_x*s_y) folded through the ScalePlusMin tree `sumf + ((d_x*d_y)*sumi +
// m_x*s_y)`. Every region op is lowered from ITS OWN operands (the loads from
// their block_stride/quant_byte_offset, the product from its weight/low/high
// operands, the min brick from its lhs_min/rhs_sum bases + byte offsets), so the
// emit provably tracks the region content -- byte-identical to the monolithic
// q4_1 mbf1/elided instance. Numerical bit-exact-vs-ggml is pending-hardware.

module {
  weft.exec.kernel @ggml_vec_dot_q4_1_q8_1_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_q4_1_q8_1_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %2 = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %3 = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %4 = weft_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %5 = weft_rvv.setvl %0 {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %5 attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q4_1_q8_1_block_dot, sew = 8 : i64, source_kernel = "ggml_vec_dot_q4_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_flat_block_dot_loop_body %2, %3, %1, %0 attributes {activation_block_stride = 36 : i64, fold_model = "scale_plus_min", fold_structure = "per-block", integer_core_lmul = "m1", kind = "typed_flat_block_dot_loop_body", multi_block_factor = 1 : i64, numerics_tier = "strict", qk = 32 : i64, strip_elision = "elided", weight_block_stride = 20 : i64} {
        ^bb0(%arg0: index, %arg1: f32):
          %6 = weft_rvv.block_fp16_scale_product %2, %3 block %arg0 : index {kind = "dual_fp16_per_block_scale_product", lhs_block_stride = 20 : i64, rhs_block_stride = 36 : i64, scale_model = "dual-fp16-per-block-d_x.d_y"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
          %7 = weft_rvv.block_fp16_min_product %2, %3 block %arg0 : index {kind = "dual_fp16_per_block_min_product", lhs_block_stride = 20 : i64, lhs_min_byte_offset = 2 : i64, rhs_block_stride = 36 : i64, rhs_sum_byte_offset = 2 : i64, scale_model = "dual-fp16-per-block-m_x.s_y"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value -> f32
          %8 = weft_rvv.load %2, %5 block %arg0 : index {block_stride = 20 : i64, quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui8, "m1">
          %9 = weft_rvv.load %3, %5 block %arg0 : index {block_stride = 36 : i64, quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
          %10 = weft_rvv.load %3, %5 block %arg0 : index {block_stride = 36 : i64, quant_byte_offset = 20 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
          %11 = weft_rvv.unsigned_nibble_x_i8_product %8, %9, %10, %5 {kind = "unsigned_nibble_x_i8_product", product_relation = "unsigned-nibble-i4m1-x-i8m1x2-to-i16m2"} : !weft_rvv.vector<ui8, "m1">, !weft_rvv.vector<i8, "m1">, !weft_rvv.vector<i8, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m2">
          %12 = weft_rvv.standalone_reduce %11, %4, %5 {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "m2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
          %13 = weft_rvv.typed_vector_lane0_to_scalar_extract %12, %5 {extract_relation = "i32m1-lane0-to-scalar-i32", kind = "vector_lane0_to_scalar_i32_extract"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> i32
          %14 = weft_rvv.block_computed_scale_dequant %13, %6 min %7 : f32 {dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32", kind = "computed_scale_sumi_dequant"} : i32, f32 -> f32
          %15 = weft_rvv.cross_block_f32_accumulate %arg1, %14 {accumulate_order = "strict-ascending-block-carried", kind = "cross_block_f32_scalar_accumulate"} : f32, f32 -> f32
          weft_rvv.typed_flat_block_dot_loop_yield %15 : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_vec_dot_q4_1_q8_1_kernel_rvv_q4_1_q8_1_block_dot(

// The outer block loop + per-block base address arithmetic: xb = vx + ib*20 (the
// q4_1 weight AoS block stride) and yb = vy + ib*36 (the q8_1 activation stride)
// -- the family-defining block strides, propagated from the region load ops'
// block_stride operands into the emitted base mul/add (migrated from the monolith
// emit fixture, strengthened to pin the stride literals).
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// CHECK: %[[STRIDEX:.*]] = literal "20" : !emitc.opaque<"size_t">
// CHECK: mul %[[IB]], %[[STRIDEX]]
// CHECK: add %arg2, %{{.*}}
// CHECK: %[[STRIDEY:.*]] = literal "36" : !emitc.opaque<"size_t">
// CHECK: mul %[[IB]], %[[STRIDEY]]
// CHECK: add %arg3, %{{.*}}

// item4 (fcvt.s.h reschedule): brick 1's d_x/d_y and the MIN brick's m_x/s_y fp16
// reads are DEFERRED to after the integer core (right before the fold -- ggml
// factory placement); their CHECKs now sit just before the fold expression below.
// Values are byte-identical; only the scalar fp16->f32 conversions' position moved.

// The UNSIGNED integer core: u8 weight load + unsigned-nibble decode (vand 0x0F /
// vsrl 0x04 / reinterpret to i8), then the asymmetric vwmul + vwmacc product.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: %[[LOWMASK:.*]] = literal "0x0F" : !emitc.opaque<"int">
// CHECK: call_opaque "__riscv_vand_vx_u8m1"(%{{.*}}, %[[LOWMASK]]
// CHECK: %[[HISH:.*]] = literal "0x04" : !emitc.opaque<"int">
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"(%{{.*}}, %[[HISH]]
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// Per-block reduce into the i32 scalar: the seed broadcast + vwredsum + lane-0
// extract (the reduce-seed vmv_v_x migrated from the monolith emit fixture).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"

// item4: the deferred brick 1 (d_x,d_y) + MIN brick (m_x,s_y) fp16 reads -- now
// emitted AFTER the integer core, right before the fold expression. m_x at xb+2,
// s_y at yb+2 (the min/sum byte offsets travel WITH their own fcvt reads).
// CHECK: %[[DX:.*]] = call_opaque "(float)*(const _Float16 *)"
// CHECK: %[[DY:.*]] = call_opaque "(float)*(const _Float16 *)"
// CHECK: %[[MO:.*]] = literal "2" : !emitc.opaque<"size_t">
// CHECK: %[[MXA:.*]] = add %{{.*}}, %[[MO]] : (!emitc.ptr<{{.*}}>, !emitc.opaque<"size_t">)
// CHECK: %[[MX:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[MXA]])
// CHECK: %[[SO:.*]] = literal "2" : !emitc.opaque<"size_t">
// CHECK: %[[SYA:.*]] = add %{{.*}}, %[[SO]] : (!emitc.ptr<{{.*}}>, !emitc.opaque<"size_t">)
// CHECK: %[[SY:.*]] = call_opaque "(float)*(const _Float16 *)"(%[[SYA]])

// The ScalePlusMin fold, fused into ONE emitc.expression: (float)sumi;
// scaleProduct = d_x*d_y; scaleTerm = scaleProduct*sumi; minTerm = m_x*s_y;
// blockTerm = scaleTerm + minTerm; sumf + blockTerm. The m_x*s_y MUL is
// recomputed INSIDE the fused expression (the min brick result is gate-only).
// CHECK: expression
// CHECK: %[[SUMIF:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[SCALEP:.*]] = mul %[[DX]], %[[DY]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[SCALET:.*]] = mul %[[SCALEP]], %[[SUMIF]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[MINT:.*]] = mul %[[MX]], %[[SY]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[BLOCKT:.*]] = add %[[SCALET]], %[[MINT]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: add %{{.*}}, %[[BLOCKT]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">

// Fail-closed on the fold_model (I7).
// BADFOLD: currently supports only fold_model

// Anti-bypass: the mutated min byte offset propagates into the emitted m_x read
// address (literal "8" is absent from the byte-exact baseline).
// MINOFF: literal "8" : !emitc.opaque<"size_t">
