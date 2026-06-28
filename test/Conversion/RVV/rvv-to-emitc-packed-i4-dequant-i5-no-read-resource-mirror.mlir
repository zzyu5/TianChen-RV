// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// I5 no-read guard (Stage 3 single-scope packed-i4 flip, risk #2): the realization
// re-homes the gearbox resource facts onto the lone tcrv_rvv.with_vl as
// REPORTING-only metadata. This fixture stamps ADVERSARIAL / contradictory values
// on every such mirror attr (operand_form claims unpacked-byte, unpack_intent
// claims none-direct, realized_unroll_factor=99, low_precision_resource.* set to
// garbage) and asserts the RVV->emitc conversion STILL emits the identical correct
// packed-i4 nibble-unpack widening-product chain. The conversion derives compute
// from the TYPED OPS only (tcrv_rvv.packed_i4_nibble_unpack_product carries the
// nibble-unpack structure; the fixed vsll/vsra/vwmul/vsra/vwmacc chain is its
// lowering) -- it reads ZERO resource-mirror strings (RVVToEmitC.cpp:2564). If any
// conversion read pulled compute from these strings, the adversarial values would
// corrupt the emission and this test would diverge.

module {
  tcrv.exec.kernel @i5_no_read_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @i5_no_read attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      // Adversarial / contradictory resource-mirror attrs on with_vl: a string
      // machine that READ these to shape compute would mis-emit. The conversion
      // ignores them all.
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], selected_path_role = "dispatch case", selected_variant = @i5_no_read, sew = 32 : i64, source_kernel = "i5_no_read_kernel", status = "selected-lowering-boundary", unroll_factor = 1 : i64, tcrv_rvv.low_precision_resource.operand_form = "unpacked-byte-elements", tcrv_rvv.low_precision_resource.unpack_intent = "none-direct-widening-product", tcrv_rvv.low_precision_resource.realized_unroll_factor = 99 : i64, tcrv_rvv.low_precision_resource.realized_vsetvl_region_count = 7 : i64, tcrv_rvv.low_precision_resource.remediation_statement_strategy = "ADVERSARIAL-string-machine-bait", tcrv_rvv.low_precision_resource.packed_unpack_plan = "ADVERSARIAL-wrong-unpack"} {
        %l0 = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %r0 = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %p0 = tcrv_rvv.packed_i4_nibble_unpack_product %l0, %r0, %vl {kind = "signed_packed_i4_nibble_unpack_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %red0 = tcrv_rvv.standalone_reduce %p0, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %deq = tcrv_rvv.dequantize %red0, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %out, %deq, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// The conversion derives the nibble-unpack compute from the TYPED OP, NOT the
// adversarial with_vl resource-mirror strings: the i4 sign-extend/unpack chain is
// emitted correctly regardless.
// CHECK-LABEL: emitc.func @tcrv_emitc_i5_no_read_kernel_i5_no_read
// CHECK: %[[ACC:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vint32m1_t">>
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// The fixed signed-i4 nibble-unpack widening-product intrinsic chain, unperturbed by
// the adversarial operand_form="unpacked-byte-elements" / unpack_intent="none" attrs.
// CHECK: call_opaque "__riscv_vsll_vx_i8mf4"
// CHECK: call_opaque "__riscv_vsll_vx_i8mf4"
// CHECK: call_opaque "__riscv_vwmul_vv_i16mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i16mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf4"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf4"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16mf2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// Single runtime-VL chunk loop despite realized_unroll_factor=99 / region_count=7:
// the loop is driven by the structural unroll_factor op-attr + the typed slices, not
// the adversarial realized_* mirrors. NO unroll: exactly one widening reduce.
// CHECK-NOT: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
