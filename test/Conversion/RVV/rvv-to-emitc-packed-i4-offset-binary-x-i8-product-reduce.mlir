// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

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
  tcrv.exec.kernel @rvv_q4_0_q8_0_integer_core_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_q4_0_q8_0_integer_core attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %w = tcrv_rvv.runtime_abi_value {c_name = "w", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qlo = tcrv_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-low", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qhi = tcrv_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q8-high", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_q4_0_q8_0_integer_core, sew = 32 : i64, source_kernel = "rvv_q4_0_q8_0_integer_core_kernel", status = "selected-lowering-boundary"} {
        %w_vec = tcrv_rvv.load %w, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %qlo_vec = tcrv_rvv.load %qlo, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %qhi_vec = tcrv_rvv.load %qhi, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %product = tcrv_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %reduced = tcrv_rvv.standalone_reduce %product, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_q4_0_q8_0_integer_core_kernel_rvv_q4_0_q8_0_integer_core(
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
