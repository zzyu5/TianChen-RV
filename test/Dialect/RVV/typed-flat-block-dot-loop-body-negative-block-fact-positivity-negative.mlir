// RUN: tcrv-opt %s --split-input-file --verify-diagnostics

// M-FLAT verifier fail-closed (positivity gate on the externally-defined ggml
// block facts qk / weight_block_stride / activation_block_stride). These i64
// attrs are read through ODS uint64_t accessors (getQk() etc.) that
// ZERO-EXTEND the value, so a NEGATIVE i64 reinterprets as a huge positive
// count and used to fail-OPEN the `<= 0` guard (F-5 fuzz: qk=0 rejected but
// qk=-32 slipped through + fully lowered). The gate now reads the SIGNED attr
// view (getXAttr().getInt()), fail-CLOSING every non-positive spelling. These
// cases lock qk=-32 / weight_block_stride<0 / activation_block_stride<0 as
// rejected (zero is separately covered by the F-5 zero_* rows).

// CASE neg qk: NEGATIVE block element count
module {
  tcrv.exec.kernel @rvv_typed_flat_block_dot_loop_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_typed_flat_block_dot_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %term = tcrv_rvv.runtime_abi_value {c_name = "stub_term", c_type = "float", ownership = "target-export-abi-owned", purpose = "loop-body:stub-term", role = "lower-bound-scalar-value"} : f32
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_typed_flat_block_dot_loop_body, sew = 8 : i64, source_kernel = "rvv_typed_flat_block_dot_loop_body_kernel", status = "selected-lowering-boundary"} {
        // expected-error@+1 {{requires qk > 0 (the QK block element count); got -32}}
        tcrv_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = -32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "elided"} {
        ^bb0(%block_index: index, %acc: f32):
          %acc_next = tcrv_rvv.cross_block_f32_accumulate %acc, %term {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          tcrv_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// CASE neg weight_block_stride: NEGATIVE AoS weight stride
module {
  tcrv.exec.kernel @rvv_typed_flat_block_dot_loop_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_typed_flat_block_dot_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %term = tcrv_rvv.runtime_abi_value {c_name = "stub_term", c_type = "float", ownership = "target-export-abi-owned", purpose = "loop-body:stub-term", role = "lower-bound-scalar-value"} : f32
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_typed_flat_block_dot_loop_body, sew = 8 : i64, source_kernel = "rvv_typed_flat_block_dot_loop_body_kernel", status = "selected-lowering-boundary"} {
        // expected-error@+1 {{requires weight_block_stride > 0 (the AoS weight block stride); got -34}}
        tcrv_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = -34 : i64, activation_block_stride = 34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "elided"} {
        ^bb0(%block_index: index, %acc: f32):
          %acc_next = tcrv_rvv.cross_block_f32_accumulate %acc, %term {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          tcrv_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}

// -----

// CASE neg activation_block_stride: NEGATIVE AoS activation stride
module {
  tcrv.exec.kernel @rvv_typed_flat_block_dot_loop_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_typed_flat_block_dot_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %term = tcrv_rvv.runtime_abi_value {c_name = "stub_term", c_type = "float", ownership = "target-export-abi-owned", purpose = "loop-body:stub-term", role = "lower-bound-scalar-value"} : f32
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_typed_flat_block_dot_loop_body, sew = 8 : i64, source_kernel = "rvv_typed_flat_block_dot_loop_body_kernel", status = "selected-lowering-boundary"} {
        // expected-error@+1 {{requires activation_block_stride > 0 (the AoS activation block stride); got -34}}
        tcrv_rvv.typed_flat_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_flat_block_dot_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = -34 : i64, fold_model = "sumi_times_scales", integer_core_lmul = "m2", strip_elision = "elided"} {
        ^bb0(%block_index: index, %acc: f32):
          %acc_next = tcrv_rvv.cross_block_f32_accumulate %acc, %term {kind = "cross_block_f32_scalar_accumulate", accumulate_order = "strict-ascending-block-carried"} : f32, f32 -> f32
          tcrv_rvv.typed_flat_block_dot_loop_yield %acc_next : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
