// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv segment2
// DEINTERLEAVE body (Stage 3 换心) via a real MLIR DialectConversion. The
// deinterleave family segment-loads one interleaved source into a TUPLE
// (vint32m1x2_t), extracts the two fields with vget, and stores each unit-stride:
//   tcrv_rvv.segment2_load ->
//       __riscv_vlseg2e32_v_i32m1x2(src + i*2, vl)  (interleaved tuple load)
//   tcrv_rvv.move (copy) x2 -> __riscv_vget_v_i32m1x2_i32m1(tuple, 0 / 1)
//   tcrv_rvv.store x2       -> __riscv_vse32_v_i32m1 (out0, out1)
// The move ops sourced from the segment2_load field results emit the field
// extracts. Asserts STRUCTURE; byte-equivalence to the legacy segment2
// string-plan oracle is pinned by the e2e diff.

module {
  tcrv.exec.kernel @seg2_deinterleave_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_seg2_deinterleave attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_seg2_deinterleave, sew = 32 : i64, source_kernel = "seg2_deinterleave_kernel", status = "selected-lowering-boundary"} {
        %field0, %field1 = tcrv_rvv.segment2_load %src, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">
        %field0_copy = tcrv_rvv.move %field0, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %field1_copy = tcrv_rvv.move %field1, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out0, %field0_copy, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
        tcrv_rvv.store %out1, %field1_copy, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_seg2_deinterleave_kernel_rvv_seg2_deinterleave(

// interleaved tuple load, then two field extracts.
// CHECK: %[[TUPLE:.*]] = call_opaque "__riscv_vlseg2e32_v_i32m1x2"(%{{.*}}, %{{.*}}) : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1x2_t">
// CHECK: %[[FX0:.*]] = call_opaque "__riscv_vget_v_i32m1x2_i32m1"(%[[TUPLE]], %{{.*}}) : (!emitc.opaque<"vint32m1x2_t">, index) -> !emitc.opaque<"vint32m1_t">
// CHECK: %[[FX1:.*]] = call_opaque "__riscv_vget_v_i32m1x2_i32m1"(%[[TUPLE]], %{{.*}}) : (!emitc.opaque<"vint32m1x2_t">, index) -> !emitc.opaque<"vint32m1_t">

// two unit-stride field stores.
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[FX0]], %{{.*}})
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[FX1]], %{{.*}})
// CHECK: return
