// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The weft_rvv.packed_i4_nibble_unpack_product op makes the signed packed-i4
// nibble-unpack widening-product STRUCTURE first-class in the typed RVV body:
// two i8 LMUL mf4 packed source vectors (each i8 packing two signed 4-bit
// nibbles) reduce to one i16 LMUL mf2 product. The Stage-3 conversion lowers it
// to the fixed vsll/vsra/vwmul/vsra/vwmacc intrinsic chain; the candidate
// selection (which packing/unpack the chain implements) is owned by the
// realization layer, not re-derived in the conversion from mirror strings.

// CHECK-LABEL: weft.exec.kernel @packed_i4_nibble_unpack_product_accepts_i8mf4_to_i16mf2
module {
  weft.exec.kernel @packed_i4_nibble_unpack_product_accepts_i8mf4_to_i16mf2 {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      // CHECK: weft_rvv.with_vl {{.*}}unroll_factor = 2
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "packed_i4_nibble_unpack_product_accepts_i8mf4_to_i16mf2", status = "selected-lowering-boundary", unroll_factor = 2 : i64} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %rhs_vec = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        // CHECK: weft_rvv.packed_i4_nibble_unpack_product
        %product = weft_rvv.packed_i4_nibble_unpack_product %lhs_vec, %rhs_vec, %vl {kind = "signed_packed_i4_nibble_unpack_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @packed_i4_nibble_unpack_product_rejects_i16_source {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 16 : i64, source_kernel = "packed_i4_nibble_unpack_product_rejects_i16_source", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %rhs_vec = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        // expected-error @+1 {{requires lhs and rhs source vectors to have type !weft_rvv.vector<i8, "mf4">}}
        %product = weft_rvv.packed_i4_nibble_unpack_product %lhs_vec, %rhs_vec, %vl {kind = "signed_packed_i4_nibble_unpack_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @packed_i4_nibble_unpack_product_rejects_unknown_kind {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "packed_i4_nibble_unpack_product_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %rhs_vec = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        // expected-error @+1 {{currently supports only kind "signed_packed_i4_nibble_unpack_product"}}
        %product = weft_rvv.packed_i4_nibble_unpack_product %lhs_vec, %rhs_vec, %vl {kind = "unsigned_packed_i4", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @with_vl_rejects_non_positive_unroll_factor {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      // expected-error @+1 {{requires optional 'unroll_factor' to be a positive structural main-loop unroll count}}
      weft_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 16 : i64, source_kernel = "with_vl_rejects_non_positive_unroll_factor", status = "selected-lowering-boundary", unroll_factor = 0 : i64} {
      } : !weft_rvv.vl
    }
  }
}
