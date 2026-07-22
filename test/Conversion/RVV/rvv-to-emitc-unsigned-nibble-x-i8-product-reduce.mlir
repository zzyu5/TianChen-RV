// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The UNSIGNED-nibble packed-i4 x plain-i8 product-reduce integer core (the
// integer core of ggml Q4_1 x Q8_1). The typed body loads the UNSIGNED packed-i4
// weight (ui8/m1, each byte two 4-bit UNSIGNED quants in [0,15]) and TWO plain-int8
// activation halves (i8/m1, the q8 low/high halves), chains the unsigned-nibble
// product op into the existing signed widening reduction (i16/m2 -> i32/m1 via
// vwredsum), and carries the scalar accumulator through the output cell.
//
// The unsigned_nibble product op lowers to the q4_1 decode chain DISTINCT from
// BOTH the q4_0 offset-binary xor/sll/sra decode AND the codebook gather: the
// weight byte is split into the two UNSIGNED nibble lanes (vand.vx 0x0F low,
// vsrl.vx 0x04 high), each lane is reinterpreted u8->i8 (value-identity for
// [0,15]), then the SAME asymmetric widening product the offset-binary sibling
// uses (vwmul against the PLAIN low activation, vwmacc against the PLAIN high
// activation) -- NO vxor.vx(0x88), NO vrgather codebook table. Structure-level
// CHECKs.

module {
  weft.exec.kernel @rvv_unsigned_nibble_q8_1_integer_core_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_unsigned_nibble_q8_1_integer_core attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %w = weft_rvv.runtime_abi_value {c_name = "w", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qlo = weft_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-low", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qhi = weft_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-high", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
        %w_vec = weft_rvv.load %w, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<ui8, "m1">
        %qlo_vec = weft_rvv.load %qlo, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
        %qhi_vec = weft_rvv.load %qhi, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m1">
        %product = weft_rvv.unsigned_nibble_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "unsigned_nibble_x_i8_product", product_relation = "unsigned-nibble-i4m1-x-i8m1x2-to-i16m2"} : !weft_rvv.vector<ui8, "m1">, !weft_rvv.vector<i8, "m1">, !weft_rvv.vector<i8, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m2">
        %reduced = weft_rvv.standalone_reduce %product, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "m2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %reduced, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_unsigned_nibble_q8_1_integer_core_kernel_rvv_unsigned_nibble_q8_1_integer_core(
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// Pre-loop i32 seed: out[0] = acc[0].
// CHECK: %[[ACCSCALAR:.*]] = load
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"(%[[ACCSCALAR]],
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg4,
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e8m1"
// The UNSIGNED packed-i4 weight load + the two plain-i8 q8 activation halves.
// CHECK: %[[W:.*]] = call_opaque "__riscv_vle8_v_u8m1"
// CHECK: %[[QLO:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// CHECK: %[[QHI:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// The unsigned nibble split: low = vand 0x0F, high = vsrl 0x04 (on the u8 lane).
// CHECK: %[[XLO:.*]] = call_opaque "__riscv_vand_vx_u8m1"(%[[W]],
// CHECK: %[[XHI:.*]] = call_opaque "__riscv_vsrl_vx_u8m1"(%[[W]],
// Reinterpret each unsigned nibble lane to signed i8 (value-identity for [0,15]);
// NO offset-binary xor-0x88 bias, NO codebook gather table.
// CHECK: %[[V0:.*]] = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%[[XLO]])
// CHECK: %[[V1:.*]] = call_opaque "__riscv_vreinterpret_v_u8m1_i8m1"(%[[XHI]])
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8m1"
// CHECK-NOT: call_opaque "__riscv_vrgather_vv_i8m1"
// Asymmetric widening product: decoded i8 weight x PLAIN i8 activation halves.
// CHECK: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i16m2"(%[[V0]], %[[QLO]], %[[BODYVL]]) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
// CHECK: %[[PAIR:.*]] = call_opaque "__riscv_vwmacc_vv_i16m2"(%[[PROD]], %[[V1]], %[[QHI]], %[[BODYVL]]) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
// In-loop i32 running seed + the signed widening reduce (vwredsum), i16m2 -> i32m1.
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%[[PAIR]], %[[SEED]], %[[BODYVL]]) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg4, %[[RED]],
// CHECK: return
