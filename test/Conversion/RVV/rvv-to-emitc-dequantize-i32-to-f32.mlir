// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules --weft-materialize-emission-plans --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp | FileCheck %s

// Stage 3 换心 — STANDALONE i32->f32 runtime-scale dequant family conversion.
// The standalone dequant body is the SIMPLEST dequant shape: a single with_vl
// scope whose only compute is load -> dequantize -> store (no product / reduce /
// accumulator / clamp). The Gearbox schedule pass marks it u2; the conversion
// reproduces the legacy string materializer's Gearbox-unrolled two-slice runtime-
// avl setvl loop: full setvl(n), loop step vlmax*2, each slice recomputing the
// remaining AVL fresh from (n - i) and the second slice subtracting the first
// slice's runtime VL, pointer offset (base + i [+ vl0]), no separate scalar tail
// loop. The dequant chain reuses the shared vfcvt_f_x_v + vfmul_vf core. The
// emitc built by --weft-rvv-lower-to-emitc, rendered through the same upstream
// translateToCpp the legacy export path uses, must be byte-equivalent to the
// hardware-validated legacy C for dequantize-i32-to-f32. Every CHECK line is the
// exact corresponding legacy line; any drift is a conversion bug.

module {
  weft.exec.kernel @explicit_selected_body_dequantize_i32_to_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_dequantize_i32_to_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_dequantize_i32_to_f32, sew = 32 : i64, source_kernel = "explicit_selected_body_dequantize_i32_to_f32_kernel", status = "selected-lowering-boundary"} {
        %source = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        weft_rvv.store %out, %result, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_dequantize_i32_to_f32 {origin = "rvv-plugin", policy = "explicit-selected-body-dequantize-i32-to-f32-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-dequantize-i32-to-f32-fallback-envelope"}
    }
  }
}

// CHECK: #include <stddef.h>
// CHECK-NEXT: #include <stdint.h>
// CHECK-NEXT: #include <riscv_vector.h>
// CHECK-NEXT: extern "C" void weft_emitc_explicit_selected_body_dequantize_i32_to_f32_kernel_explicit_selected_body_rvv_dequantize_i32_to_f32(const int32_t* v1, float v2, float* v3, size_t v4) {
// CHECK-NEXT: // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK-NEXT: size_t v5 = __riscv_vsetvl_e32m1(v4);
// CHECK-NEXT: size_t v6 = v5 * 2;
// CHECK-NEXT: for (size_t v7 = 0; v7 < v4; v7 += v6) {
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK-NEXT: size_t v8 = v4 - v7;
// CHECK-NEXT: size_t v9 = __riscv_vsetvl_e32m1(v8);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// CHECK-NEXT: const int32_t* v10 = v1 + v7;
// CHECK-NEXT: vint32m1_t v11 = __riscv_vle32_v_i32m1(v10, v9);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.dequantize role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
// CHECK-NEXT: vfloat32m1_t v12 = __riscv_vfcvt_f_x_v_f32m1(v11, v9);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.dequantize role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
// CHECK-NEXT: vfloat32m1_t v13 = __riscv_vfmul_vf_f32m1(v12, v2, v9);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
// CHECK-NEXT: float* v14 = v3 + v7;
// CHECK-NEXT: __riscv_vse32_v_f32m1(v14, v13, v9);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// CHECK-NEXT: size_t v15 = v4 - v7;
// CHECK-NEXT: size_t v16 = v15 - v9;
// CHECK-NEXT: size_t v17 = __riscv_vsetvl_e32m1(v16);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.load role=load op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// CHECK-NEXT: const int32_t* v18 = v1 + v7;
// CHECK-NEXT: const int32_t* v19 = v18 + v9;
// CHECK-NEXT: vint32m1_t v20 = __riscv_vle32_v_i32m1(v19, v17);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.dequantize role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfcvt_f_x_v_f32m1
// CHECK-NEXT: vfloat32m1_t v21 = __riscv_vfcvt_f_x_v_f32m1(v20, v17);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.dequantize role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m1
// CHECK-NEXT: vfloat32m1_t v22 = __riscv_vfmul_vf_f32m1(v21, v2, v17);
// CHECK-NEXT: // weft_emitc.source_op=weft_rvv.store role=store op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m1
// CHECK-NEXT: float* v23 = v3 + v7;
// CHECK-NEXT: float* v24 = v23 + v9;
// CHECK-NEXT: __riscv_vse32_v_f32m1(v24, v22, v17);
// CHECK-NEXT: }
// CHECK-NEXT: return;
// CHECK-NEXT: }
