// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp | FileCheck %s

// Stage 3 换心 — f32 compare-select (clamp) family conversion. The clamp body
// loads an f32 vector, splats the two f32 scalar bounds (vfmv_v_f), compares
// with the f-prefixed vmflt, and merges with vmerge_vvm_f32m1 — the float
// counterparts of the integer compare-select intrinsics over the f32/m1 grid.
// The emitc built by --tcrv-rvv-lower-to-emitc, rendered through the same
// upstream translateToCpp the legacy export path uses, must be byte-equivalent
// to the hardware-validated legacy C for f32_clamp_select. Every CHECK line is
// the exact corresponding legacy line; any drift is a conversion bug.

module {
  tcrv.exec.kernel @pre_realized_f32_clamp_select_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @pre_realized_rvv_f32_clamp_select attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:input", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %1 = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:lower", role = "lower-bound-scalar-value"} : f32
      %2 = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:upper", role = "upper-bound-scalar-value"} : f32
      %3 = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %4 = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:n", role = "runtime-element-count"} : index
      %5 = tcrv_rvv.setvl %4 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %5 attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @pre_realized_rvv_f32_clamp_select, sew = 32 : i64, source_kernel = "pre_realized_f32_clamp_select_kernel", status = "selected-lowering-boundary"} {
        %6 = tcrv_rvv.load %0, %5 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        %7 = tcrv_rvv.splat %1, %5 : f32, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        %8 = tcrv_rvv.splat %2, %5 : f32, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        %9 = tcrv_rvv.compare %6, %7, %5 {kind = "slt"} : !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<f32, "m1">
        %10 = tcrv_rvv.select %9, %7, %6, %5 : !tcrv_rvv.mask<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        %11 = tcrv_rvv.compare %8, %10, %5 {kind = "slt"} : !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<f32, "m1">
        %12 = tcrv_rvv.select %11, %8, %10, %5 : !tcrv_rvv.mask<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %3, %12, %5 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: #include <stddef.h>
// CHECK-NEXT: #include <stdint.h>
// CHECK-NEXT: #include <riscv_vector.h>
// CHECK-NEXT: extern "C" void tcrv_emitc_pre_realized_f32_clamp_select_kernel_pre_realized_rvv_f32_clamp_select(const float* v1, float v2, float v3, float* v4, size_t v5) {
// CHECK-NEXT: // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK-NEXT: size_t v6 = __riscv_vsetvl_e32m1(v5);
// CHECK-NEXT: for (size_t v7 = 0; v7 < v5; v7 += v6) {
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK-NEXT: size_t v8 = v5 - v7;
// CHECK-NEXT: size_t v9 = __riscv_vsetvl_e32m1(v8);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m1
// CHECK-NEXT: const float* v10 = v1 + v7;
// CHECK-NEXT: vfloat32m1_t v11 = __riscv_vle32_v_f32m1(v10, v9);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
// CHECK-NEXT: vfloat32m1_t v12 = __riscv_vfmv_v_f_f32m1(v2, v9);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.splat role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m1
// CHECK-NEXT: vfloat32m1_t v13 = __riscv_vfmv_v_f_f32m1(v3, v9);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
// CHECK-NEXT: vbool32_t v14 = __riscv_vmflt_vv_f32m1_b32(v11, v12, v9);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
// CHECK-NEXT: vfloat32m1_t v15 = __riscv_vmerge_vvm_f32m1(v11, v12, v14, v9);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.compare role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmflt_vv_f32m1_b32
// CHECK-NEXT: vbool32_t v16 = __riscv_vmflt_vv_f32m1_b32(v13, v15, v9);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.select role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m1
// CHECK-NEXT: vfloat32m1_t v17 = __riscv_vmerge_vvm_f32m1(v15, v13, v16, v9);
// CHECK-NEXT: // tcrv_emitc.source_op=tcrv_rvv.store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
// CHECK-NEXT: float* v18 = v4 + v7;
// CHECK-NEXT: __riscv_vse32_v_f32m1(v18, v17, v9);
// CHECK-NEXT: }
// CHECK-NEXT: return;
// CHECK-NEXT: }
