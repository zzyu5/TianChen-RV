// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv
// COMPUTED-MASK segment2 LOAD body (Stage 3 换心) via a real MLIR
// DialectConversion. A compare in the same VL scope produces the predicate, then
// the masked segment2 load merges the interleaved source over the old
// destinations (passthrough tuple) and extracts the two fields:
//   weft_rvv.load x4 (cmp_lhs, cmp_rhs, old0, old1) -> __riscv_vle32_v_i32m1
//   weft_rvv.compare -> __riscv_vmslt_vv_i32m1_b32 (compare-produced mask)
//   weft_rvv.masked_segment2_load ->
//       __riscv_vcreate_v_i32m1x2(old0, old1)              (passthrough pack)
//       __riscv_vlseg2e32_v_i32m1x2_tumu(mask, pass, src+i*2, vl)
//       __riscv_vget_v_i32m1x2_i32m1(tuple, 0 / 1)         (field extracts)
//   weft_rvv.store x2 -> __riscv_vse32_v_i32m1
// The masked load consumes the COMPARE mask. Asserts STRUCTURE.

module {
  weft.exec.kernel @cmseg2_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_cmseg2_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %out0 = weft_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-output-buffer"} : !weft_rvv.runtime_abi_value
      %out1 = weft_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs_vec = weft_rvv.load %cmp_lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_vec = weft_rvv.load %cmp_rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %old0 = weft_rvv.load %out0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %old1 = weft_rvv.load %out1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %mask = weft_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "slt"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %field0, %field1 = weft_rvv.masked_segment2_load %src, %mask, %old0, %old1, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", inactive_lane_policy = "preserve-passthrough-on-false-lanes", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !weft_rvv.runtime_abi_value, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out0, %field0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
        weft_rvv.store %out1, %field1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_cmseg2_load_kernel_rvv_cmseg2_load(

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
