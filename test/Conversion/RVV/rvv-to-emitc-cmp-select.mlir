// RUN: weft-opt %s --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp | FileCheck %s

// Stage 3 换心 — compare-select family conversion. The emitc built by
// --weft-rvv-lower-to-emitc (real DialectConversion over the typed
// weft_rvv.compare/select dataflow), rendered through the same upstream
// translateToCpp the legacy export path uses, must be byte-equivalent to the
// hardware-validated legacy C for cmp_select. The compare lowers to vmseq
// (eq predicate) producing a vbool32_t mask; the select lowers to
// vmerge_vvm(false_vec, true_vec, mask, vl) — false before true is the legacy
// merge order. Every CHECK line below is the exact corresponding legacy line.
// Any drift is a conversion bug, not a fixture to edit.

module {
  weft.exec.kernel @explicit_selected_body_cmp_select_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_i32_cmp_select attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_i32_cmp_select, sew = 32 : i64, source_kernel = "explicit_selected_body_cmp_select_kernel", status = "selected-lowering-boundary"} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %b = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %mask = weft_rvv.compare %a, %b, %vl {kind = "eq"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %selected = weft_rvv.select %mask, %a, %b, %vl : !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %selected, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: #include <stddef.h>
// CHECK-NEXT: #include <stdint.h>
// CHECK-NEXT: #include <riscv_vector.h>
// CHECK-NEXT: extern "C" void weft_emitc_explicit_selected_body_cmp_select_kernel_explicit_selected_body_rvv_i32_cmp_select(const int32_t* v1, const int32_t* v2, int32_t* v3, size_t v4) {
// CHECK-NEXT: // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK-NEXT: size_t v5 = __riscv_vsetvl_e32m1(v4);
// CHECK-NEXT: for (size_t v6 = 0; v6 < v4; v6 += v5) {
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK-NEXT: size_t v7 = v4 - v6;
// CHECK-NEXT: size_t v8 = __riscv_vsetvl_e32m1(v7);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// CHECK-NEXT: const int32_t* v9 = v1 + v6;
// CHECK-NEXT: vint32m1_t v10 = __riscv_vle32_v_i32m1(v9, v8);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// CHECK-NEXT: const int32_t* v11 = v2 + v6;
// CHECK-NEXT: vint32m1_t v12 = __riscv_vle32_v_i32m1(v11, v8);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.compare role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmseq_vv_i32m1_b32
// CHECK-NEXT: vbool32_t v13 = __riscv_vmseq_vv_i32m1_b32(v10, v12, v8);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.select role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_i32m1
// CHECK-NEXT: vint32m1_t v14 = __riscv_vmerge_vvm_i32m1(v12, v10, v13, v8);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
// CHECK-NEXT: int32_t* v15 = v3 + v6;
// CHECK-NEXT: __riscv_vse32_v_i32m1(v15, v14, v8);
// CHECK-NEXT: }
// CHECK-NEXT: return;
// CHECK-NEXT: }
