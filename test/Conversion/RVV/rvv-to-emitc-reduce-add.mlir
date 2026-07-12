// RUN: weft-opt %s --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp | FileCheck %s

// Stage 3 换心 — reduction (reduce_add) family conversion. The emitc built by
// --weft-rvv-lower-to-emitc (real DialectConversion over the typed
// weft_rvv.reduce dataflow), rendered through the same upstream translateToCpp
// the legacy export path uses, must be byte-equivalent to the
// hardware-validated legacy C for reduce_add. The reduce lowers to
// __riscv_vredsum_vs_i32m1_i32m1(input, accumulator, vl) producing the lane-0
// reduction; the rhs-loaded vector is the accumulator seed; the result is
// stored back to the output chunk base with a VL=1 store (lane 0). Every CHECK
// line below is the exact corresponding legacy line. Any drift is a conversion
// bug, not a fixture to edit.

module {
  weft.exec.kernel @explicit_selected_body_reduce_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_reduce_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:input", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:accumulator", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_reduce_add, sew = 32 : i64, source_kernel = "explicit_selected_body_reduce_add_kernel", status = "selected-lowering-boundary"} {
        %input = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %acc = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %reduced = weft_rvv.reduce %input, %acc, %vl {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", kind = "add", result_layout = "store-reduction-lane0-to-output-chunk-base"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %reduced, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: #include <stddef.h>
// CHECK-NEXT: #include <stdint.h>
// CHECK-NEXT: #include <riscv_vector.h>
// CHECK-NEXT: extern "C" void weft_emitc_explicit_selected_body_reduce_add_kernel_explicit_selected_body_rvv_reduce_add(const int32_t* v1, const int32_t* v2, int32_t* v3, size_t v4) {
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
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.reduce role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vredsum_vs_i32m1_i32m1
// CHECK-NEXT: vint32m1_t v13 = __riscv_vredsum_vs_i32m1_i32m1(v10, v12, v8);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
// CHECK-NEXT: int32_t* v14 = v3 + v6;
// CHECK-NEXT: __riscv_vse32_v_i32m1(v14, v13, 1);
// CHECK-NEXT: }
// CHECK-NEXT: return;
// CHECK-NEXT: }
