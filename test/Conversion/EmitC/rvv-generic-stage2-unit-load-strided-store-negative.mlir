// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s

// Stage 3 换心 re-target: the legacy string route rejected a unit-stride load +
// byte-strided store body ("cannot mix strided memory ops with unit-stride
// load/store") as a memory-form-mixing scope-limit. With the string-plan owner
// retired, the real RVV->emitc DialectConversion lowers the actual typed
// dataflow: a contiguous load feeding a strided scatter store is a well-formed,
// useful kernel, so it MATERIALIZES (vle32 load + vsse32 strided store) rather
// than hitting the deleted legacy check.
module {
  weft.exec.kernel @rvv_generic_unit_load_strided_store_reject_incomplete_body {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_generic_unit_load_strided_store_missing_move attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_generic_unit_load_strided_store_missing_move, sew = 32 : i64, source_kernel = "rvv_generic_unit_load_strided_store_reject_incomplete_body", status = "selected-lowering-boundary"} {
        %loaded = weft_rvv.load %src, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.strided_store %dst, %loaded, %dst_stride_bytes, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, index, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: emitc.func @weft_emitc_rvv_generic_unit_load_strided_store_reject_incomplete_body_rvv_generic_unit_load_strided_store_missing_move
// CHECK: callee=__riscv_vle32_v_i32m1
// CHECK: callee=__riscv_vsse32_v_i32m1
