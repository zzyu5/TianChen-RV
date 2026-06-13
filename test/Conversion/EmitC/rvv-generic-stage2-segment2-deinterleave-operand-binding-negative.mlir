// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | FileCheck %s

// Stage 3 换心 re-target: this section used to assert the legacy string-route
// runtime-ABI-name contract — "RVV selected-body runtime ABI contract invalid;
// expected src, out0, out1, n for the bounded int32_t segment2 deinterleave
// route" — a c_name convention check. With the Segment2 string-plan owner
// retired, the real RVV->emitc DialectConversion lowers the actual typed
// dataflow: it binds the interleaved source segment2_load, the two field
// move/vget extracts, and the two unit-stride field stores by SSA Value and ABI
// ROLE (lhs-input-buffer is the interleaved source; the two
// segment-field{0,1}-output-buffer roles are the deinterleaved destinations),
// not by the ABI c_name spelling. The c_names here are deliberately swapped (the
// interleaved source buffer is spelled "out0", the first destination "src"), but
// the typed body is well-formed: the source is segment-loaded into one tuple,
// the two fields are extracted with __riscv_vget_v_i32m1x2_i32m1, and each lands
// in its output-role buffer with a unit-stride store. The rendered parameters
// are positional, so the swap is cosmetic and the body MATERIALIZES,
// byte-identical to the well-named segment2 deinterleave route, rather than
// hitting the deleted legacy c_name contract check (I5: executable facts come
// from the typed body, not ABI strings).

module {
  tcrv.exec.kernel @rvv_segment2_deinterleave_reject_source_output_name_swap {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_segment2_deinterleave_swapped_names attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_segment2_deinterleave_swapped_names, sew = 32 : i64, source_kernel = "rvv_segment2_deinterleave_reject_source_output_name_swap", status = "selected-lowering-boundary"} {
        %field0, %field1 = tcrv_rvv.segment2_load %src, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">
        %moved0 = tcrv_rvv.move %field0, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %moved1 = tcrv_rvv.move %field1, %vl {kind = "copy"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out0, %moved0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
        tcrv_rvv.store %out1, %moved1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_rvv_segment2_deinterleave_reject_source_output_name_swap_rvv_segment2_deinterleave_swapped_names
// CHECK: callee=__riscv_vlseg2e32_v_i32m1x2
// CHECK: callee=__riscv_vget_v_i32m1x2_i32m1
// CHECK: callee=__riscv_vget_v_i32m1x2_i32m1
// CHECK: callee=__riscv_vse32_v_i32m1
// CHECK: callee=__riscv_vse32_v_i32m1
