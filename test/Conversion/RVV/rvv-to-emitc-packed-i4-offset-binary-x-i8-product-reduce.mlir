// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// The m1 flat-cohort rung: SAME offset-binary decode + asymmetric product
// STRUCTURE at the SEW8 byte-anchor strip scope (i4m1 weight x i8m1 low/high
// activation -> i16m2), sed-derived from the narrow mf4/mf2 integer-core anchor.
// The emitter is fully type-driven, so the ONLY difference is the width tokens
// (i8mf4->i8m1, i16mf2->i16m2, e32m1->e8m1, vwredsum i16mf2->i16m2) -- byte-exact
// structure. This exercises the packed_i4_offset_binary_x_i8_product verifier's
// m1 rung ("offset-binary-i4m1-x-i8m1x2-to-i16m2") end to end (parse+verify+lower).
// RUN: sed -e 's/sew = 32/sew = 8/g; s/"mf4"/"m1"/g; s/i16, "mf2"/i16, "m2"/g; s/offset-binary-i4mf4-x-i8mf4x2-to-i16mf2/offset-binary-i4m1-x-i8m1x2-to-i16m2/g' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=M1

// Stage 4 换心 — ASYMMETRIC offset-binary packed-i4 x plain-i8 product-reduce
// (the integer core of ggml Q4_0 x Q8_0). The typed body loads the packed-i4
// weight (i8/mf4, each byte two offset-binary nibbles) and TWO plain-int8
// activation halves (i8/mf4, the q8 low/high halves), chains the asymmetric
// product op into the existing signed widening reduction (i16/mf2 -> i32/m1 via
// vwredsum), and carries the scalar accumulator through the output cell.
//
// The product op lowers to the ONE-SIDED offset-binary chain: vxor.vx(0x88) the
// weight to convert each offset-binary nibble into a two's-complement lane, then
// vsll(4)/vsra(4) -> low-nibble i8 lane and vsra(4) -> high-nibble i8 lane, then
// vwmul against the PLAIN low activation + vwmacc against the PLAIN high
// activation (the activations are NOT shifted, and there is NO vsra(product,8)
// rescale -- that distinguishes it from the SYMMETRIC two's-complement
// packed_i4_nibble_unpack_product). Structure-level CHECKs; byte identity to the
// ggml integer partial is pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc1-integer-core/.

module {
  weft.exec.kernel @rvv_q4_0_q8_0_integer_core_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_q4_0_q8_0_integer_core attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %w = weft_rvv.runtime_abi_value {c_name = "w", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qlo = weft_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-low", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qhi = weft_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-high", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q4_0_q8_0_integer_core, sew = 32 : i64, source_kernel = "rvv_q4_0_q8_0_integer_core_kernel", status = "selected-lowering-boundary"} {
        %w_vec = weft_rvv.load %w, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %qlo_vec = weft_rvv.load %qlo, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %qhi_vec = weft_rvv.load %qhi, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %product = weft_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %reduced = weft_rvv.standalone_reduce %product, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %reduced, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_q4_0_q8_0_integer_core_kernel_rvv_q4_0_q8_0_integer_core(
// CHECK: call_opaque "__riscv_vsetvl_e32m1"
// Pre-loop i32 seed: out[0] = acc[0].
// CHECK: %[[ACCSCALAR:.*]] = load
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"(%[[ACCSCALAR]],
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg4,
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"
// The three i8/mf4 source loads: packed-i4 weight + plain q8 low + plain q8 high.
// CHECK: %[[W:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// CHECK: %[[QLO:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// CHECK: %[[QHI:.*]] = call_opaque "__riscv_vle8_v_i8mf4"
// Offset-binary -> two's-complement: xor 0x88 the WEIGHT only.
// CHECK: %[[WXOR:.*]] = call_opaque "__riscv_vxor_vx_i8mf4"(%[[W]],
// Low nibble: shift into the high nibble then arithmetic-shift back (sign-extend).
// CHECK: %[[WLOSH:.*]] = call_opaque "__riscv_vsll_vx_i8mf4"(%[[WXOR]],
// CHECK: %[[V0:.*]] = call_opaque "__riscv_vsra_vx_i8mf4"(%[[WLOSH]],
// High nibble: arithmetic-shift sign-extends it in place.
// CHECK: %[[V1:.*]] = call_opaque "__riscv_vsra_vx_i8mf4"(%[[WXOR]],
// Asymmetric widening product: decoded i8 weight x PLAIN i8 activation halves.
// CHECK: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i16mf2"(%[[V0]], %[[QLO]], %[[BODYVL]]) : (!emitc.opaque<"vint8mf4_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16mf2_t">
// CHECK: %[[PAIR:.*]] = call_opaque "__riscv_vwmacc_vv_i16mf2"(%[[PROD]], %[[V1]], %[[QHI]], %[[BODYVL]]) : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"vint8mf4_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16mf2_t">
// In-loop i32 running seed + the signed widening reduce (vwredsum).
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: %[[RED:.*]] = call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"(%[[PAIR]], %[[SEED]], %[[BODYVL]]) : (!emitc.opaque<"vint16mf2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%arg4, %[[RED]],
// CHECK: return

// ===================== m1 flat-cohort rung (SEW8/m1 byte anchor -> i16m2) =====
// The SAME chain at the byte-anchor strip scope: e8m1 setvl, i8m1 loads/decode,
// i16m2 asymmetric product, the byte-anchor i16m2->i32m1 vwredsum. Proves the
// emitter emits the m1 rung purely from the operand/result types (no mf4/SEW32
// assumption) -- the "emit supports m1" evidence for the verifier's m1 rung.
// M1-LABEL: emitc.func @weft_emitc_rvv_q4_0_q8_0_integer_core_kernel_rvv_q4_0_q8_0_integer_core(
// M1: call_opaque "__riscv_vsetvl_e8m1"
// M1: %[[ACCSCALAR:.*]] = load
// M1: call_opaque "__riscv_vmv_v_x_i32m1"(%[[ACCSCALAR]],
// M1: call_opaque "__riscv_vse32_v_i32m1"(%arg4,
// M1: for %{{.*}} = %{{.*}} to %{{.*}} step
// M1: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e8m1"
// The three i8/m1 source loads: packed-i4 weight + plain q8 low + plain q8 high.
// M1: %[[W:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// M1: %[[QLO:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// M1: %[[QHI:.*]] = call_opaque "__riscv_vle8_v_i8m1"
// Offset-binary -> two's-complement: xor 0x88 the WEIGHT only.
// M1: %[[WXOR:.*]] = call_opaque "__riscv_vxor_vx_i8m1"(%[[W]],
// Low nibble sign-extend, then high nibble sign-extend in place.
// M1: %[[WLOSH:.*]] = call_opaque "__riscv_vsll_vx_i8m1"(%[[WXOR]],
// M1: %[[V0:.*]] = call_opaque "__riscv_vsra_vx_i8m1"(%[[WLOSH]],
// M1: %[[V1:.*]] = call_opaque "__riscv_vsra_vx_i8m1"(%[[WXOR]],
// Asymmetric widening product at the m1 rung: i8m1 x i8m1 -> i16m2.
// M1: %[[PROD:.*]] = call_opaque "__riscv_vwmul_vv_i16m2"(%[[V0]], %[[QLO]], %[[BODYVL]]) : (!emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
// M1: %[[PAIR:.*]] = call_opaque "__riscv_vwmacc_vv_i16m2"(%[[PROD]], %[[V1]], %[[QHI]], %[[BODYVL]]) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"vint8m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint16m2_t">
// The byte-anchor i16m2 -> i32m1 widening reduce (vwredsum flips with the anchor).
// M1: %[[SEED:.*]] = call_opaque "__riscv_vmv_v_x_i32m1"
// M1: %[[RED:.*]] = call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"(%[[PAIR]], %[[SEED]], %[[BODYVL]]) : (!emitc.opaque<"vint16m2_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">
// M1: call_opaque "__riscv_vse32_v_i32m1"(%arg4, %[[RED]],
// M1: return
