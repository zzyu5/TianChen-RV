// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-rvv-materialize-gearbox-schedules

module {
  weft.exec.kernel @rvv_gearbox_stale_schedule_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_gearbox_stale_schedule attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-stale:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-stale:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-stale:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-stale:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      // expected-error@+1 {{RVV Gearbox pass found stale schedule fact 'weft_rvv.gearbox.schedule_id': expected 'rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1' but found 'route-string-derived-gear'}}
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_gearbox_stale_schedule, sew = 32 : i64, source_kernel = "rvv_gearbox_stale_schedule_kernel", status = "selected-lowering-boundary", weft_rvv.gearbox.schedule_id = "route-string-derived-gear"} {
        %source = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        weft_rvv.store %out, %result, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_gearbox_stale_selected_candidate_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_gearbox_stale_selected_candidate attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-stale-selected:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-stale-selected:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-stale-selected:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-stale-selected:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      // expected-error@+1 {{RVV Gearbox pass found stale schedule fact 'weft_rvv.gearbox.selected_candidate': expected 'rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1' but found 'artifact-name-derived-gear'}}
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_gearbox_stale_selected_candidate, sew = 32 : i64, source_kernel = "rvv_gearbox_stale_selected_candidate_kernel", status = "selected-lowering-boundary", weft_rvv.gearbox.candidate_set = "rvv-gearbox-candidate-set.v1[rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1,rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1]", weft_rvv.gearbox.selected_candidate = "artifact-name-derived-gear"} {
        %source = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        weft_rvv.store %out, %result, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_gearbox_bad_avl_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_gearbox_bad_avl attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-bad-avl:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-bad-avl:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-bad-avl:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %len = weft_rvv.runtime_abi_value {c_name = "len", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "rvv-gearbox-bad-avl:len", role = "runtime-element-count"} : index
      // expected-error@+1 {{RVV Gearbox schedule derivation for runtime AVL requires runtime ABI role 'runtime-element-count' named 'n'}}
      %vl = weft_rvv.setvl %len {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_gearbox_bad_avl, sew = 32 : i64, source_kernel = "rvv_gearbox_bad_avl_kernel", status = "selected-lowering-boundary"} {
        %source = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        weft_rvv.store %out, %result, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}
