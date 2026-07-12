// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The FIVE-BIT (nibble+qh) offset-binary packed x plain-i8 product-reduce integer
// core (the integer core of ggml Q5_0 x Q8_0). The typed body loads the UNSIGNED
// packed weight (ui8/m1, each byte two 4-bit nibbles) and TWO plain-int8 activation
// halves (i8/m1, the q8 low/high halves), names the 32-bit qh field as an
// operand-flow SOURCE brick (weft_rvv.block_five_bit_qh_source, gate-only i32),
// chains the five-bit product op into the existing signed widening reduction
// (i16/m2 -> i32/m1 via vwredsum), and carries the scalar accumulator through the
// output cell.
//
// The five_bit_offset_binary_x_i8_product op lowers to the q5_0 decode chain: the
// weight byte is split into the two UNSIGNED nibble lanes (vand.vx 0x0F low,
// vsrl.vx 0x04 high), each lane is MERGED with its per-element qh 5th bit (re-read
// from the qh SOURCE brick's own qh_base + qh_byte_offset as two aligned 16-bit
// halves, then the vid+c shift / vsrl_vv / vand 1 / vsll 4 / vncvt narrowing lane),
// vor'd into the nibble, reinterpreted u8->i8 and offset-binary biased (vsub 16),
// then fed the SAME asymmetric widening product the siblings use (vwmul against the
// PLAIN low activation, vwmacc against the PLAIN high activation). Structure-level
// CHECKs.

module {
  weft.exec.kernel @rvv_five_bit_q5_0_integer_core_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_five_bit_q5_0_integer_core attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %w = weft_rvv.runtime_abi_value {c_name = "w", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qlo = weft_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-low", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qhi = weft_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-high", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_five_bit_q5_0_integer_core, sew = 8 : i64, source_kernel = "rvv_five_bit_q5_0_integer_core_kernel", status = "selected-lowering-boundary"} {
        %w_vec = weft_rvv.load %w, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui8, "m1">
        %qlo_vec = weft_rvv.load %qlo, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
        %qhi_vec = weft_rvv.load %qhi, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
        %qh = weft_rvv.block_five_bit_qh_source %w {kind = "block_five_bit_qh_source", qh_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value -> i32
        %product = weft_rvv.five_bit_offset_binary_x_i8_product %w_vec, %qh, %qlo_vec, %qhi_vec, %vl {kind = "five_bit_offset_binary_x_i8_product", product_relation = "five-bit-offset-binary-i4m1-x-i8m1x2-to-i16m2"} : !weft_rvv.vector<ui8, "m1">, i32, !weft_rvv.vector<i8, "m1">, !weft_rvv.vector<i8, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m2">
        %reduced = weft_rvv.standalone_reduce %product, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "m2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %reduced, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_five_bit_q5_0_integer_core_kernel_rvv_five_bit_q5_0_integer_core(
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// Pre-loop i32 seed: out[0] = acc[0].
// CHECK: %[[ACCSCALAR:.*]] = load
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"(%[[ACCSCALAR]],
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg4,
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e8m1"
// The UNSIGNED packed weight load + the two plain-i8 q8 activation halves.
// CHECK: %[[W:.*]] = call_opaque "__riscv_vle8_v_u8m1"
// CHECK: %[[QLO:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// CHECK: %[[QHI:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// The 5th-bit qh field re-read from the qh SOURCE brick's own base+offset as TWO
// aligned 16-bit halves (w+2 / w+4) -- the operand/source-driven anti-bypass.
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// The 5-BIT decode chain (m1 core / m2 wide): the unsigned nibble unpack (vand
// 0x0F / vsrl 0x04), then the per-element 5th-bit injection from the broadcast qh
// halves -- vid+c shift / vsrl_vv / vand 1 / vsll 4 / vncvt narrowing -- the vor
// into the nibble, the reinterpret + vsub 16 offset-binary bias, then the SAME
// signed vwmul/vwmacc product against the plain q8 halves.
// CHECK: %[[XLO:.*]] = call_opaque "__riscv_vand_vx_u8m1"(%[[W]],
// CHECK: %[[XHI:.*]] = call_opaque "__riscv_vsrl_vx_u8m1"(%[[W]],
// CHECK: call_opaque "__riscv_vid_v_u16m2"
// CHECK: call_opaque "__riscv_vadd_vx_u16m2"
// CHECK: call_opaque "__riscv_vmv_v_x_u16m2"
// CHECK: call_opaque "__riscv_vsrl_vv_u16m2"
// CHECK: call_opaque "__riscv_vand_vx_u16m2"
// CHECK: call_opaque "__riscv_vsll_vx_u16m2"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8m1"
// CHECK: call_opaque "__riscv_vor_vv_u8m1"
// CHECK: call_opaque "__riscv_vor_vv_u8m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"
// CHECK: call_opaque "__riscv_vsub_vx_i8m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"
// CHECK: call_opaque "__riscv_vsub_vx_i8m1"
// CHECK-NOT: call_opaque "__riscv_vrgather_vv_i8m1"
// CHECK: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: %[[PAIR:.*]] = call_opaque "__riscv_vwmacc_vv_i16m2"
// In-loop i32 running seed + the signed widening reduce (vwredsum), i16m2 -> i32m1.
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%[[PAIR]], %[[SEED]], %[[BODYVL]])
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg4, %[[RED]],
// CHECK: return
