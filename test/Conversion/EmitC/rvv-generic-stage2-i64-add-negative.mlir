// RUN: weft-opt %s --split-input-file --verify-diagnostics --weft-materialize-emitc-lowerable-routes

// Stage 3 换心 note: three former sections asserted legacy SCOPE-limit rejections
// that the conversion now genuinely + correctly covers — i64/m1 and i64/m2
// elementwise (extended by commit 4de23a4e) and a single-load `x+x` body (the
// legacy "exactly two loads" structural count). Those bodies now MATERIALIZE
// through the real RVV->emitc DialectConversion (i64 positive coverage:
// rvv-to-emitc-i64-add.mlir + rvv-generic-stage2-i64-add-materialization.mlir),
// so they are no longer negatives and were removed.
//
// The sections retained below are GENUINE rejections that survive the换心: the
// runtime-ABI c_type must agree with the loaded vector element (the conversion
// declines a const int32_t* buffer feeding an i64 load, so the legacy ABI
// validator still fires); an undisturbed tail policy is outside the converted
// slice (the conversion declines it rather than mislower as agnostic, so the
// legacy profile validator still fires); plus op-verifier invariants (element
// width vs with_vl SEW, the required capability `requires` array, and an AVL not
// defined by an explicit runtime_abi_value).

// expected-error@+1 {{no registered backend emission driver fully legalizes the selected variant @rvv_i64_wrong_abi body to EmitC}}
module {
  weft.exec.kernel @rvv_i64_wrong_abi_rejected {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_i64_wrong_abi attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i64, "m1">, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// -----

// expected-error@+1 {{no registered backend emission driver fully legalizes the selected variant @rvv_i64_policy body to EmitC}}
module {
  weft.exec.kernel @rvv_i64_policy_rejected {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_i64_policy attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = undisturbed, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = undisturbed, mask = agnostic>, sew = 64 : i64} {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i64, "m1">, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_i64_element_sew_mismatch_rejected {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_i64_element_sew_mismatch attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        // expected-error@+1 {{requires result element width 32 to agree with enclosing weft_rvv.with_vl SEW64 metadata}}
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_i64_missing_capability_rejected {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    // expected-error@+1 {{requires structured array attribute 'requires' containing capability symbol references}}
    weft.exec.variant @rvv_i64_missing_capability attributes { origin = "rvv-plugin" } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i64, "m1">, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// -----

// expected-error@+1 {{no registered backend emission driver fully legalizes the selected variant @rvv_i64_missing_avl_runtime body to EmitC}}
module {
  weft.exec.kernel @rvv_i64_missing_avl_runtime_rejected {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_i64_missing_avl_runtime attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i64, "m1">, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}
