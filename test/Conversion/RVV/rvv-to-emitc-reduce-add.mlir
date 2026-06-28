// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp | FileCheck %s

// Stage 3 换心 — reduction (reduce_add) family conversion. The emitc built by
// --tcrv-rvv-lower-to-emitc (real DialectConversion over the typed
// tcrv_rvv.reduce dataflow), rendered through the same upstream translateToCpp
// the legacy export path uses, must be byte-equivalent to the
// hardware-validated legacy C for reduce_add. The reduce lowers to
// __riscv_vredsum_vs_i32m1_i32m1(input, accumulator, vl) producing the lane-0
// reduction; the rhs-loaded vector is the accumulator seed; the result is
// stored back to the output chunk base with a VL=1 store (lane 0). Every CHECK
// line below is the exact corresponding legacy line. Any drift is a conversion
// bug, not a fixture to edit.

module {
  tcrv.exec.kernel @explicit_selected_body_reduce_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_reduce_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:input", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:accumulator", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-reduce:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_reduce_add, sew = 32 : i64, source_kernel = "explicit_selected_body_reduce_add_kernel", status = "selected-lowering-boundary"} {
        %input = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %acc = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %reduced = tcrv_rvv.reduce %input, %acc, %vl {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", kind = "add", result_layout = "store-reduction-lane0-to-output-chunk-base"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: #include <stddef.h>
// CHECK-NEXT: #include <stdint.h>
// CHECK-NEXT: #include <riscv_vector.h>
// CHECK-NEXT: extern "C" void tcrv_emitc_explicit_selected_body_reduce_add_kernel_explicit_selected_body_rvv_reduce_add(const int32_t* v1, const int32_t* v2, int32_t* v3, size_t v4) {
// CHECK-NEXT: // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK-NEXT: size_t v5 = __riscv_vsetvl_e32m1(v4);
// CHECK-NEXT: for (size_t v6 = 0; v6 < v4; v6 += v5) {
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK-NEXT: size_t v7 = v4 - v6;
// CHECK-NEXT: size_t v8 = __riscv_vsetvl_e32m1(v7);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// CHECK-NEXT: const int32_t* v9 = v1 + v6;
// CHECK-NEXT: vint32m1_t v10 = __riscv_vle32_v_i32m1(v9, v8);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// CHECK-NEXT: const int32_t* v11 = v2 + v6;
// CHECK-NEXT: vint32m1_t v12 = __riscv_vle32_v_i32m1(v11, v8);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.reduce role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vredsum_vs_i32m1_i32m1
// CHECK-NEXT: vint32m1_t v13 = __riscv_vredsum_vs_i32m1_i32m1(v10, v12, v8);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
// CHECK-NEXT: int32_t* v14 = v3 + v6;
// CHECK-NEXT: __riscv_vse32_v_i32m1(v14, v13, 1);
// CHECK-NEXT: }
// CHECK-NEXT: return;
// CHECK-NEXT: }
