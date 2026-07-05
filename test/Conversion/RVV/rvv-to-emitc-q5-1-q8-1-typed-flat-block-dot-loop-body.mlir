// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/fold_model = "scale_plus_min"/fold_model = "unsupported_fold"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADFOLD
// Anti-bypass #1 (qh): the qh brick's byte offset is READ to place the qh 16-bit
// read address, so mutating qh_byte_offset 4 -> 9 must MOVE the emitted qh low read
// to `xb+9` and the high read to `xb+11` (a gate-only build reading a fixed
// descriptor offset would stay at 4/6). The literals "9"/"11" do not appear in the
// byte-exact baseline emit.
// RUN: sed 's/qh_byte_offset = 4 : i64/qh_byte_offset = 9 : i64/' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=QHOFF
// Anti-bypass #2 (min): the MIN brick's byte offset is READ to place the m_x read
// address, so mutating lhs_min_byte_offset 2 -> 8 must MOVE the emitted m_x read to
// `xb+8` (a gate-only build reading a fixed descriptor offset would stay at 2). The
// literal "8" does not appear in the byte-exact baseline emit.
// RUN: sed 's/lhs_min_byte_offset = 2 : i64/lhs_min_byte_offset = 8 : i64/' %s | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MINOFF

// The FULL q5_1 (ggml Q5_1 x Q8_1) typed flat block-dot loop body -- the cohort
// LAST cell, the UNION of q5_0's five-bit integer core and q4_1's MIN term. It
// exercises the region-driven OP-BY-OP emit that COMPOSES: (1) the FIVE-BIT
// offset-binary integer core (u8 weight load + vand 0x0F / vsrl 0x04 nibble decode
// + per-lane qh 5th-bit merge + vreinterpret + vwmul/vwmacc) with applyOffsetBias
// FALSE -- so NO `vsub 16` (q5_1's bias lives in the per-block MIN scale, NOT the
// offset-binary `-16`), (2) the per-block qh SOURCE brick (32-bit qh field re-read
// as two aligned u16 halves off the SHARED weight base at the brick's OWN
// qh_byte_offset), (3) the per-block MIN brick (m_x*s_y off the SHARED bases at the
// brick's OWN byte offsets, read in the q4_1 slot RIGHT AFTER dX/dY and BEFORE the
// qh halves), and (4) the ScalePlusMin fold `sumf + ((d_x*d_y)*sumi + m_x*s_y)`.
// Every region op is lowered from ITS OWN operands, so the emit provably tracks the
// region content -- byte-identical to the monolithic q5_1 mbf1/elided m1 instance.
// Numerical bit-exact-vs-ggml is pending-hardware.

module {
  tcrv.exec.kernel @ggml_vec_dot_q5_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_q5_1_q8_1_block_dot attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %1 = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %2 = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %3 = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %4 = tcrv_rvv.runtime_abi_value {c_name = "zero_seed", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "loop-body:reduce-seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %5 = tcrv_rvv.setvl %0 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %5 attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q5_1_q8_1_block_dot, sew = 8 : i64, source_kernel = "ggml_vec_dot_q5_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_flat_block_dot_loop_body %2, %3, %1, %0 attributes {activation_block_stride = 36 : i64, fold_model = "scale_plus_min", integer_core_lmul = "m1", kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, strip_elision = "elided", weight_block_stride = 24 : i64} {
        ^bb0(%arg0: index, %arg1: f32):
          %6 = tcrv_rvv.block_fp16_scale_product %2, %3 block %arg0 : index {kind = "dual_fp16_per_block_scale_product", lhs_block_stride = 24 : i64, rhs_block_stride = 36 : i64, scale_model = "dual-fp16-per-block-d_x.d_y"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value -> f32
          %7 = tcrv_rvv.block_fp16_min_product %2, %3 block %arg0 : index {kind = "dual_fp16_per_block_min_product", lhs_block_stride = 24 : i64, lhs_min_byte_offset = 2 : i64, rhs_block_stride = 36 : i64, rhs_sum_byte_offset = 2 : i64, scale_model = "dual-fp16-per-block-m_x.s_y"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value -> f32
          %8 = tcrv_rvv.block_five_bit_qh_source %2 block %arg0 : index {block_stride = 24 : i64, kind = "block_five_bit_qh_source", qh_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value -> i32
          %9 = tcrv_rvv.load %2, %5 block %arg0 : index {block_stride = 24 : i64, quant_byte_offset = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "m1">
          %10 = tcrv_rvv.load %3, %5 block %arg0 : index {block_stride = 36 : i64, quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
          %11 = tcrv_rvv.load %3, %5 block %arg0 : index {block_stride = 36 : i64, quant_byte_offset = 20 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
          %12 = tcrv_rvv.five_bit_offset_binary_x_i8_product %9, %8, %10, %11, %5 {kind = "five_bit_offset_binary_x_i8_product", product_relation = "five-bit-offset-binary-i4m1-x-i8m1x2-to-i16m2"} : !tcrv_rvv.vector<ui8, "m1">, i32, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
          %13 = tcrv_rvv.standalone_reduce %12, %4, %5 {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %14 = tcrv_rvv.typed_vector_lane0_to_scalar_extract %13, %5 {extract_relation = "i32m1-lane0-to-scalar-i32", kind = "vector_lane0_to_scalar_i32_extract"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> i32
          %15 = tcrv_rvv.block_computed_scale_dequant %14, %6 min %7 : f32 {dequant_relation = "scalar-i32-sumi-to-f32-computed-scale-f32", kind = "computed_scale_sumi_dequant"} : i32, f32 -> f32
          %16 = tcrv_rvv.cross_block_f32_accumulate %arg1, %15 {accumulate_order = "strict-ascending-block-carried", kind = "cross_block_f32_scalar_accumulate"} : f32, f32 -> f32
          tcrv_rvv.typed_flat_block_dot_loop_yield %16 : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_rvv_q5_1_q8_1_block_dot(

// The outer block loop + per-block base address arithmetic: xb = vx + ib*24 (the
// q5_1 weight AoS block stride) and yb = vy + ib*36 (the q8_1 activation stride)
// -- the family-defining block strides, propagated from the region load ops'
// block_stride operands into the emitted base mul/add (migrated from the monolith
// emit fixture, strengthened to pin the stride literals).
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// CHECK: %[[STRIDEX:.*]] = literal "24" : !emitc.opaque<"size_t">
// CHECK: mul %[[IB]], %[[STRIDEX]]
// CHECK: add %arg2, %{{.*}}
// CHECK: %[[STRIDEY:.*]] = literal "36" : !emitc.opaque<"size_t">
// CHECK: mul %[[IB]], %[[STRIDEY]]
// CHECK: add %arg3, %{{.*}}

// item4 (fcvt.s.h reschedule): brick 1's d_x/d_y and the MIN brick's m_x/s_y fp16
// reads are DEFERRED to after the integer core (right before the fold -- ggml
// factory placement); their CHECKs now sit just before the fold expression below.
// Values are byte-identical; only the scalar fp16->f32 conversions' position moved.

// The qh field's TWO aligned 16-bit halves, read BEFORE the (now-deferred) m_x/s_y
// and BEFORE the integer core. qh low at xb+4, qh high at xb+6, each a raw
// (uint16_t)*(const uint16_t *) read to a uint32_t -- NOT an fp16 fcvt read.
// CHECK: %[[QLO:.*]] = literal "4" : !emitc.opaque<"size_t">
// CHECK: %[[QLOA:.*]] = add %{{.*}}, %[[QLO]] : (!emitc.ptr<{{.*}}>, !emitc.opaque<"size_t">)
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"(%[[QLOA]]){{.*}}-> !emitc.opaque<"uint32_t">
// CHECK: %[[QHI:.*]] = literal "6" : !emitc.opaque<"size_t">
// CHECK: %[[QHIA:.*]] = add %{{.*}}, %[[QHI]] : (!emitc.ptr<{{.*}}>, !emitc.opaque<"size_t">)
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"(%[[QHIA]]){{.*}}-> !emitc.opaque<"uint32_t">

// The FIVE-BIT integer core: u8 weight load (at qs@8) + two i8 q8 halves (at qs@4
// and qs@20), unsigned-nibble decode (vand 0x0F / vsrl 0x04), the per-lane qh
// 5th-bit merge off the broadcast qh halves, vreinterpret to i8, then the
// asymmetric vwmul + vwmacc product.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: %[[LOWMASK:.*]] = literal "0x0F" : !emitc.opaque<"int">
// CHECK: call_opaque "__riscv_vand_vx_u8m1"(%{{.*}}, %[[LOWMASK]]
// CHECK: %[[HISH:.*]] = literal "0x04" : !emitc.opaque<"int">
// CHECK: call_opaque "__riscv_vsrl_vx_u8m1"(%{{.*}}, %[[HISH]]
// The per-element qh 5th-bit injection: the vid+c shift vector / vsrl_vv / vand 1
// / vsll 4 / vncvt narrowing -- the FULL injection chain migrated from the monolith
// -m1 emit fixture (previously only vid/vncvt were pinned on the typed path).
// CHECK: call_opaque "__riscv_vid_v_u16m2"
// CHECK: call_opaque "__riscv_vadd_vx_u16m2"
// CHECK: call_opaque "__riscv_vmv_v_x_u16m2"
// CHECK: call_opaque "__riscv_vsrl_vv_u16m2"
// CHECK: call_opaque "__riscv_vand_vx_u16m2"
// CHECK: call_opaque "__riscv_vsll_vx_u16m2"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8m1"
// The two vor merges (low + high nibble halves) + the value-identity reinterpret
// on BOTH halves.
// CHECK: call_opaque "__riscv_vor_vv_u8m1"
// CHECK: call_opaque "__riscv_vor_vv_u8m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"
// The q5_1 UNSIGNED path DROPS the offset-binary bias entirely: applyOffsetBias
// FALSE emits NO `vsub 16` (the bias is folded into the per-block MIN scale).
// CHECK-NOT: call_opaque "__riscv_vsub_vx_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// Per-block reduce: the seed broadcast + vwredsum + lane-0 extract (the reduce-
// seed vmv_v_x migrated from the monolith emit fixture).
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
// blockTerm = scaleTerm + minTerm; sumf + blockTerm. The m_x*s_y MUL is recomputed
// INSIDE the fused expression (the min brick result is gate-only).
// CHECK: expression
// CHECK: %[[SUMIF:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[SCALEP:.*]] = mul %[[DX]], %[[DY]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[SCALET:.*]] = mul %[[SCALEP]], %[[SUMIF]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[MINT:.*]] = mul %[[MX]], %[[SY]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: %[[BLOCKT:.*]] = add %[[SCALET]], %[[MINT]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">
// CHECK: add %{{.*}}, %[[BLOCKT]] : (!emitc.opaque<"float">, !emitc.opaque<"float">) -> !emitc.opaque<"float">

// Fail-closed on the fold_model (I7).
// BADFOLD: currently supports only fold_model

// Anti-bypass #1: the mutated qh byte offset propagates into the emitted qh read
// addresses (literals "9" and "11" are absent from the byte-exact baseline emit).
// QHOFF: literal "9" : !emitc.opaque<"size_t">
// QHOFF: literal "11" : !emitc.opaque<"size_t">

// Anti-bypass #2: the mutated min byte offset propagates into the emitted m_x read
// address (literal "8" is absent from the byte-exact baseline).
// MINOFF: literal "8" : !emitc.opaque<"size_t">
