// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv strided
// elementwise add beachhead body via a real MLIR DialectConversion.
// weft_rvv.strided_load lowers to __riscv_vlse32_v_i32m1 with a scaled element
// pointer (base + induction * stride) and a (ptrdiff_t) byte stride
// (stride * 4); weft_rvv.strided_store lowers to __riscv_vsse32_v_i32m1 with
// the same scaled-pointer / byte-stride computation. Byte-equivalence to the
// legacy materializer C is pinned by the e2e diff; this test asserts STRUCTURE.

module {
  weft.exec.kernel @explicit_selected_body_strided_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_strided_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:n", role = "runtime-element-count"} : index
      %lhs_stride = weft_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = weft_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:rhs-stride", role = "rhs-input-stride"} : index
      %out_stride = weft_rvv.runtime_abi_value {c_name = "out_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-strided-add:out-stride", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %a = weft_rvv.strided_load %lhs, %lhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %b = weft_rvv.strided_load %rhs, %rhs_stride, %vl : !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %a, %b, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.strided_store %out, %sum, %out_stride, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, index, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_explicit_selected_body_strided_add_kernel_explicit_selected_body_rvv_strided_add(
// CHECK: %[[I:[a-zA-Z0-9_]+]] = {{.*}}to{{.*}}step
// CHECK: %[[BODYVL:.*]] = call_opaque "__riscv_vsetvl_e32m1"

// lhs strided load: scaled element pointer + (ptrdiff_t) byte stride.
// CHECK: %[[LOFF:.*]] = mul %[[I]], %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[LPTR:.*]] = add %{{.*}}, %[[LOFF]] : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"const int32_t">>
// CHECK: %[[LSC:.*]] = cast %{{.*}} : !emitc.opaque<"size_t"> to !emitc.opaque<"ptrdiff_t">
// CHECK: %[[L4:.*]] = literal "4" : !emitc.opaque<"size_t">
// CHECK: %[[L4C:.*]] = cast %[[L4]] : !emitc.opaque<"size_t"> to !emitc.opaque<"ptrdiff_t">
// CHECK: %[[LBS:.*]] = mul %[[LSC]], %[[L4C]] : (!emitc.opaque<"ptrdiff_t">, !emitc.opaque<"ptrdiff_t">) -> !emitc.opaque<"ptrdiff_t">
// CHECK: %[[LHSVEC:.*]] = call_opaque "__riscv_vlse32_v_i32m1"(%[[LPTR]], %[[LBS]], %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"ptrdiff_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// rhs strided load.
// CHECK: %[[RHSVEC:.*]] = call_opaque "__riscv_vlse32_v_i32m1"

// add.
// CHECK: %[[SUM:.*]] = call_opaque "__riscv_vadd_vv_i32m1"(%[[LHSVEC]], %[[RHSVEC]], %[[BODYVL]])

// strided store: scaled pointer + byte stride, then vsse with the result.
// CHECK: %[[SPTR:.*]] = add %{{.*}} : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"size_t">) -> !emitc.ptr<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vsse32_v_i32m1"(%[[SPTR]], %{{.*}}, %[[SUM]], %[[BODYVL]]) : (!emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"ptrdiff_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
