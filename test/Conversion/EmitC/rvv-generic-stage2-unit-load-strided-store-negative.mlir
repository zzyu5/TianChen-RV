// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | FileCheck %s

// Stage 3 换心 re-target: the legacy string route rejected a unit-stride load +
// byte-strided store body ("cannot mix strided memory ops with unit-stride
// load/store") as a memory-form-mixing scope-limit. With the string-plan owner
// retired, the real RVV->emitc DialectConversion lowers the actual typed
// dataflow: a contiguous load feeding a strided scatter store is a well-formed,
// useful kernel, so it MATERIALIZES (vle32 load + vsse32 strided store) rather
// than hitting the deleted legacy check.
module {
  tcrv.exec.kernel @rvv_generic_unit_load_strided_store_reject_incomplete_body {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_generic_unit_load_strided_store_missing_move attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %dst_stride_bytes = tcrv_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_generic_unit_load_strided_store_missing_move, sew = 32 : i64, source_kernel = "rvv_generic_unit_load_strided_store_reject_incomplete_body", status = "selected-lowering-boundary"} {
        %loaded = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.strided_store %dst, %loaded, %dst_stride_bytes, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, index, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_rvv_generic_unit_load_strided_store_reject_incomplete_body_rvv_generic_unit_load_strided_store_missing_move
// CHECK: callee=__riscv_vle32_v_i32m1
// CHECK: callee=__riscv_vsse32_v_i32m1
