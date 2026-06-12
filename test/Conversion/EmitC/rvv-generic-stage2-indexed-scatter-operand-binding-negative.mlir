// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | FileCheck %s

// Stage 3 换心 re-target: this section used to assert the legacy string-route
// runtime-ABI-name contract — "RVV selected-body runtime ABI contract invalid;
// expected src, index, dst, n for the bounded int32_t indexed-scatter route" —
// a c_name convention check. With the base-memory string-plan owner retired, the
// real RVV->emitc DialectConversion lowers the actual typed dataflow: it binds
// the unit-stride source load, the index_load, and the indexed (scatter) store
// by SSA Value and ABI ROLE (lhs-input-buffer is the source, output-buffer is
// the scatter destination), not by the ABI c_name spelling. The c_names here are
// deliberately swapped (the source buffer is spelled "dst", the destination
// "src"), but the typed body is well-formed: the source is loaded, byte-scaled
// indices drive __riscv_vsoxei32_v_i32m1 into the output-role buffer. The
// rendered parameters are positional, so the swap is cosmetic and the scatter
// binds to the correct (output-role) buffer. The body MATERIALIZES, byte-identical
// to the well-named indexed-scatter route, rather than hitting the deleted legacy
// c_name contract check (I5: executable facts come from typed body, not ABI
// strings).

module {
  tcrv.exec.kernel @rvv_indexed_scatter_reject_source_destination_name_swap {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_indexed_scatter_swapped_names attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_indexed_scatter_swapped_names, sew = 32 : i64, source_kernel = "rvv_indexed_scatter_reject_source_destination_name_swap", status = "selected-lowering-boundary"} {
        %loaded = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %indices = tcrv_rvv.index_load %index, %vl {index_eew = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.index_vector<i32, "m1">
        %moved = tcrv_rvv.move %loaded, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.indexed_store %dst, %indices, %moved, %vl {index_eew = 32 : i64, index_uniqueness = "unique", offset_unit = "element"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.index_vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_rvv_indexed_scatter_reject_source_destination_name_swap_rvv_indexed_scatter_swapped_names
// CHECK: callee=__riscv_vle32_v_i32m1
// CHECK: callee=__riscv_vle32_v_u32m1
// CHECK: callee=__riscv_vmul_vx_u32m1
// CHECK: callee=__riscv_vsoxei32_v_i32m1
