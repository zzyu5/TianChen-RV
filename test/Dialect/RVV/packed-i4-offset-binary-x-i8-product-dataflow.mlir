// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The weft_rvv.packed_i4_offset_binary_x_i8_product op makes the ASYMMETRIC
// offset-binary packed-i4 x plain-i8 widening-product STRUCTURE first-class in
// the typed RVV body -- the integer core of ggml Q4_0 x Q8_0. ONLY the packed-i4
// weight (i8 mf4, each byte two offset-binary nibbles) is nibble-decoded; the two
// plain int8 activation operands (i8 mf4, the q8 low/high halves paired with the
// low/high nibbles) stay plain. The Stage-4 conversion lowers it to the fixed
// vxor(0x88)/vsll/vsra/vwmul/vwmacc one-sided chain; this is the SYMMETRIC
// weft_rvv.packed_i4_nibble_unpack_product's asymmetric sibling, defined ALONGSIDE
// it (the symmetric op is byte-untouched).

// CHECK-LABEL: weft.exec.kernel @packed_i4_offset_binary_x_i8_product_accepts_i8mf4x3_to_i16mf2
module {
  weft.exec.kernel @packed_i4_offset_binary_x_i8_product_accepts_i8mf4x3_to_i16mf2 {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %w = weft_rvv.runtime_abi_value {c_name = "w", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qlo = weft_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-low", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qhi = weft_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-high", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "packed_i4_offset_binary_x_i8_product_accepts_i8mf4x3_to_i16mf2", status = "selected-lowering-boundary"} {
        %w_vec = weft_rvv.load %w, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %qlo_vec = weft_rvv.load %qlo, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %qhi_vec = weft_rvv.load %qhi, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        // CHECK: weft_rvv.packed_i4_offset_binary_x_i8_product
        %product = weft_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @packed_i4_offset_binary_x_i8_product_rejects_i16_weight {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %w = weft_rvv.runtime_abi_value {c_name = "w", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qlo = weft_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "activation-low", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qhi = weft_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "activation-high", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 16 : i64, source_kernel = "packed_i4_offset_binary_x_i8_product_rejects_i16_weight", status = "selected-lowering-boundary"} {
        %w_vec = weft_rvv.load %w, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %qlo_vec = weft_rvv.load %qlo, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        %qhi_vec = weft_rvv.load %qhi, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
        // expected-error @+1 {{requires the packed-i4 weight source vector to have type !weft_rvv.vector<i8, "mf4">}}
        %product = weft_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @packed_i4_offset_binary_x_i8_product_rejects_unknown_kind {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %w = weft_rvv.runtime_abi_value {c_name = "w", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qlo = weft_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-low", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qhi = weft_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-high", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "packed_i4_offset_binary_x_i8_product_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        %w_vec = weft_rvv.load %w, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %qlo_vec = weft_rvv.load %qlo, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %qhi_vec = weft_rvv.load %qhi, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        // expected-error @+1 {{currently supports only kind "signed_packed_i4_offset_binary_x_i8_product"}}
        %product = weft_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary", product_relation = "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @packed_i4_offset_binary_x_i8_product_rejects_unknown_relation {
    weft.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %w = weft_rvv.runtime_abi_value {c_name = "w", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qlo = weft_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-low", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %qhi = weft_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-high", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "packed_i4_offset_binary_x_i8_product_rejects_unknown_relation", status = "selected-lowering-boundary"} {
        %w_vec = weft_rvv.load %w, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %qlo_vec = weft_rvv.load %qlo, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        %qhi_vec = weft_rvv.load %qhi, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "mf4">
        // expected-error @+1 {{requires product_relation "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"}}
        %product = weft_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vector<i8, "mf4">, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      } : !weft_rvv.vl
    }
  }
}
