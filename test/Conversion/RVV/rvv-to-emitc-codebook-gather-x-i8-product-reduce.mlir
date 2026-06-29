// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// Track B G2 换心 — the CODEBOOK (vrgather) packed-i4 x plain-i8 product-reduce
// integer core (the integer core of ggml IQ4_NL / FP4 codebook x Q8_0). The typed
// body broadcast-loads the 16-entry non-linear int8 codebook table
// (tcrv_rvv.codebook_table_broadcast -> i8/m1 values vreg), loads the UNSIGNED
// packed-i4 weight (ui8/m1, each byte two 4-bit table indices) and TWO plain-int8
// activation halves (i8/m1, the q8 low/high halves), chains the codebook-gather
// product op into the existing signed widening reduction (i16/m2 -> i32/m1 via
// vwredsum), and carries the scalar accumulator through the output cell.
//
// The codebook_gather product op lowers to the codebook-class chain DISTINCT from
// the q4_0 offset-binary xor/sll/sra arithmetic decode: the weight byte is split
// into the two UNSIGNED nibble index lanes (vand.vx 0x0F low, vsrl.vx 0x04 high),
// each index is GATHERED through the broadcast table (vrgather_vv_i8m1(values,
// idx)) into a signed-i8 weight lane, then the SAME asymmetric widening product
// the offset-binary sibling uses (vwmul against the PLAIN low activation, vwmacc
// against the PLAIN high activation) -- NO vxor.vx(0x88), NO vsll/vsra. Structure-
// level CHECKs; byte identity to ggml's RVV codebook method (the same vrgather the
// iq4_nl board kernel uses) is pinned by the iq4_nl block-dot lit's decode chain.

module {
  tcrv.exec.kernel @rvv_codebook_q8_0_integer_core_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_codebook_q8_0_integer_core attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %w = tcrv_rvv.runtime_abi_value {c_name = "w", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qlo = tcrv_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-low", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qhi = tcrv_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-high", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_codebook_q8_0_integer_core, sew = 32 : i64, source_kernel = "rvv_codebook_q8_0_integer_core_kernel", status = "selected-lowering-boundary"} {
        %table = tcrv_rvv.codebook_table_broadcast {codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>, table_symbol = "tcrv_iq4_nl_kvalues"} : !tcrv_rvv.vector<i8, "m1">
        %w_vec = tcrv_rvv.load %w, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<ui8, "m1">
        %qlo_vec = tcrv_rvv.load %qlo, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
        %qhi_vec = tcrv_rvv.load %qhi, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
        %product = tcrv_rvv.codebook_gather_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %table, %vl {kind = "signed_codebook_gather_x_i8_product", product_relation = "codebook-gather-i8-x-i8x2-to-i16"} : !tcrv_rvv.vector<ui8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
        %reduced = tcrv_rvv.standalone_reduce %product, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_codebook_q8_0_integer_core_kernel_rvv_codebook_q8_0_integer_core(
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// Pre-loop i32 seed: out[0] = acc[0].
// CHECK: %[[ACCSCALAR:.*]] = load
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"(%[[ACCSCALAR]],
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg4,
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// The 16-entry non-linear codebook as a structured static const decl + broadcast.
// CHECK: verbatim "static const int8_t tcrv_iq4_nl_kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};"
// CHECK: %[[VALUES:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// The UNSIGNED packed-i4 weight load + the two plain-i8 q8 activation halves.
// CHECK: %[[W:.*]] = call_opaque "__riscv_vle8_v_u8m1"
// CHECK: %[[QLO:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// CHECK: %[[QHI:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// The codebook nibble split: low index = vand 0x0F, high index = vsrl 0x04 (u8).
// CHECK: %[[IDXLO:.*]] = call_opaque "__riscv_vand_vx_u8m1"(%[[W]],
// CHECK: %[[IDXHI:.*]] = call_opaque "__riscv_vsrl_vx_u8m1"(%[[W]],
// The codebook GATHER: each index lane maps through the broadcast table (NOT a
// q4_0 xor 0x88 / sll / sra arithmetic decode).
// CHECK: %[[V0:.*]] = call_opaque "__riscv_vrgather_vv_i8m1"(%[[VALUES]], %[[IDXLO]],
// CHECK: %[[V1:.*]] = call_opaque "__riscv_vrgather_vv_i8m1"(%[[VALUES]], %[[IDXHI]],
// The q4_0 offset-binary decode chain must be ABSENT (this is a codebook gather).
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8m1"
// Asymmetric widening product: gathered i8 weight x PLAIN i8 activation halves.
// CHECK: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i16m2"(%[[V0]], %[[QLO]], %[[BODYVL]]) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
// CHECK: %[[PAIR:.*]] = call_opaque "__riscv_vwmacc_vv_i16m2"(%[[PROD]], %[[V1]], %[[QHI]], %[[BODYVL]]) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
// In-loop i32 running seed + the signed widening reduce (vwredsum), i16m2 -> i32m1.
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%[[PAIR]], %[[SEED]], %[[BODYVL]]) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg4, %[[RED]],
// CHECK: return
