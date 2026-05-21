// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_masked_store_dataflow_valid
  tcrv.exec.kernel @rvv_masked_store_dataflow_valid {
    %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} {
      // CHECK: tcrv_rvv.load
      %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // CHECK: tcrv_rvv.mask_load
      %predicate = tcrv_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
      // CHECK: tcrv_rvv.masked_store
      tcrv_rvv.masked_store %dst, %predicate, %payload, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", memory_form = "masked-unit-store"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_masked_store_reject_inactive_policy {
    %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} {
      %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      %predicate = tcrv_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
      // expected-error@+1 {{requires inactive_lane_policy "preserve-output-on-false-lanes" because false mask lanes must preserve the preinitialized output buffer}}
      tcrv_rvv.masked_store %dst, %predicate, %payload, %vl {inactive_lane_policy = "preserve-old-destination", memory_form = "masked-unit-store"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_masked_store_reject_fake_mask {
    %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %fake_mask = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.mask<i32, "m1">
    %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} {
      %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // expected-error@+1 {{requires mask operand to be produced by tcrv_rvv.mask_load or tcrv_rvv.compare inside the selected RVV typed body}}
      tcrv_rvv.masked_store %dst, %fake_mask, %payload, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", memory_form = "masked-unit-store"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_masked_store_reject_output_role {
    %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} {
      %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      %predicate = tcrv_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
      // expected-error@+1 {{requires masked store output buffer operand to bind runtime ABI role 'output-buffer'}}
      tcrv_rvv.masked_store %dst, %predicate, %payload, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", memory_form = "masked-unit-store"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_masked_store_reject_mask_lmul_mismatch {
    %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} {
      %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // expected-error@+1 {{requires result type '!tcrv_rvv.mask<i32, "m2">' to agree with enclosing tcrv_rvv.with_vl LMUL metadata 'm1'}}
      %predicate = tcrv_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m2">
      tcrv_rvv.masked_store %dst, %predicate, %payload, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", memory_form = "masked-unit-store"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m2">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}
