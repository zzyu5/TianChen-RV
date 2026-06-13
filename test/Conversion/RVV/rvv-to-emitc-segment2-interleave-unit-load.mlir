// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv segment2
// INTERLEAVE body (Stage 3 换心) via a real MLIR DialectConversion. The
// interleave family packs two unit-stride field loads into one segment2 TUPLE
// (vint32m1x2_t) and stores it to the interleaved destination:
//   tcrv_rvv.load x2     -> __riscv_vle32_v_i32m1 (field0, field1)
//   tcrv_rvv.segment2_store ->
//       __riscv_vcreate_v_i32m1x2(field0, field1)  (tuple pack)
//       __riscv_vsseg2e32_v_i32m1x2(dst + i*2, tuple, vl)  (interleaved store)
// The interleaved destination base advances by i*2 (the two fields are adjacent
// per element). Asserts STRUCTURE; byte-equivalence to the legacy segment2
// string-plan oracle is pinned by the e2e diff (all 7 segment2 fixtures are
// byte-identical).

module {
  tcrv.exec.kernel @seg2_interleave_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_seg2_interleave attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src0 = tcrv_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src1 = tcrv_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-interleaved-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_seg2_interleave, sew = 32 : i64, source_kernel = "seg2_interleave_kernel", status = "selected-lowering-boundary"} {
        %field0 = tcrv_rvv.load %src0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %field1 = tcrv_rvv.load %src1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.segment2_store %dst, %field0, %field1, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", segment_count = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_seg2_interleave_kernel_rvv_seg2_interleave(

// two unit-stride field loads.
// CHECK: %[[F0:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[F1:.*]] = call_opaque "__riscv_vle32_v_i32m1"

// pack the two fields into a segment2 tuple, then interleaved store.
// CHECK: %[[TUPLE:.*]] = call_opaque "__riscv_vcreate_v_i32m1x2"(%[[F0]], %[[F1]]) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">) -> !emitc.opaque<"vint32m1x2_t">
// CHECK: call_opaque "__riscv_vsseg2e32_v_i32m1x2"(%{{.*}}, %[[TUPLE]], %{{.*}}) : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"vint32m1x2_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
