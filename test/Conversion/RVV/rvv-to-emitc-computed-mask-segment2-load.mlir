// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// COMPUTED-MASK segment2 LOAD body (Stage 3 换心) via a real MLIR
// DialectConversion. A compare in the same VL scope produces the predicate, then
// the masked segment2 load merges the interleaved source over the old
// destinations (passthrough tuple) and extracts the two fields:
//   tcrv_rvv.load x4 (cmp_lhs, cmp_rhs, old0, old1) -> __riscv_vle32_v_i32m1
//   tcrv_rvv.compare -> __riscv_vmslt_vv_i32m1_b32 (compare-produced mask)
//   tcrv_rvv.masked_segment2_load ->
//       __riscv_vcreate_v_i32m1x2(old0, old1)              (passthrough pack)
//       __riscv_vlseg2e32_v_i32m1x2_tumu(mask, pass, src+i*2, vl)
//       __riscv_vget_v_i32m1x2_i32m1(tuple, 0 / 1)         (field extracts)
//   tcrv_rvv.store x2 -> __riscv_vse32_v_i32m1
// The masked load consumes the COMPARE mask. Asserts STRUCTURE.

module {
  tcrv.exec.kernel @cmseg2_load_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_cmseg2_load attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_cmseg2_load, sew = 32 : i64, source_kernel = "cmseg2_load_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %old0 = tcrv_rvv.load %out0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %old1 = tcrv_rvv.load %out1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %field0, %field1 = tcrv_rvv.masked_segment2_load %src, %mask, %old0, %old1, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", inactive_lane_policy = "preserve-passthrough-on-false-lanes", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out0, %field0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
        tcrv_rvv.store %out1, %field1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_cmseg2_load_kernel_rvv_cmseg2_load(

// compare produces the predicate mask.
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmslt_vv_i32m1_b32"(%{{.*}}, %{{.*}}, %{{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">

// pack passthroughs, masked tuple load (_tumu), then two field extracts.
// CHECK: %[[PASS:.*]] = call_opaque "__riscv_vcreate_v_i32m1x2"(%{{.*}}, %{{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"vint32m1x2_t">
// CHECK: %[[TUPLE:.*]] = call_opaque "__riscv_vlseg2e32_v_i32m1x2_tumu"(%[[MASK]], %[[PASS]], %{{.*}}, %{{.*}}) : (!emitc.opaque<"vbool32_t">, !emitc.opaque<"vint32m1x2_t">, !emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1x2_t">
// CHECK: %[[FX0:.*]] = call_opaque "__riscv_vget_v_i32m1x2_i32m1"(%[[TUPLE]], %{{.*}})
// CHECK: %[[FX1:.*]] = call_opaque "__riscv_vget_v_i32m1x2_i32m1"(%[[TUPLE]], %{{.*}})
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[FX0]], %{{.*}})
// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[FX1]], %{{.*}})
// CHECK: return
