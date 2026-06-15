// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tcrv_rvv.packed_i4_offset_binary_x_i8_product op makes the ASYMMETRIC
// offset-binary packed-i4 x plain-i8 widening-product STRUCTURE first-class in
// the typed RVV body -- the integer core of ggml Q4_0 x Q8_0. ONLY the packed-i4
// weight (i8 mf4, each byte two offset-binary nibbles) is nibble-decoded; the two
// plain int8 activation operands (i8 mf4, the q8 low/high halves paired with the
// low/high nibbles) stay plain. The Stage-4 conversion lowers it to the fixed
// vxor(0x88)/vsll/vsra/vwmul/vwmacc one-sided chain; this is the SYMMETRIC
// tcrv_rvv.packed_i4_nibble_unpack_product's asymmetric sibling, defined ALONGSIDE
// it (the symmetric op is byte-untouched).

// CHECK-LABEL: tcrv.exec.kernel @packed_i4_offset_binary_x_i8_product_accepts_i8mf4x3_to_i16mf2
module {
  tcrv.exec.kernel @packed_i4_offset_binary_x_i8_product_accepts_i8mf4x3_to_i16mf2 {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %w = tcrv_rvv.runtime_abi_value {c_name = "w", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qlo = tcrv_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-low", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qhi = tcrv_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-high", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "packed_i4_offset_binary_x_i8_product_accepts_i8mf4x3_to_i16mf2", status = "selected-lowering-boundary"} {
        %w_vec = tcrv_rvv.load %w, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %qlo_vec = tcrv_rvv.load %qlo, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %qhi_vec = tcrv_rvv.load %qhi, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        // CHECK: tcrv_rvv.packed_i4_offset_binary_x_i8_product
        %product = tcrv_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @packed_i4_offset_binary_x_i8_product_rejects_i16_weight {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %w = tcrv_rvv.runtime_abi_value {c_name = "w", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qlo = tcrv_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "activation-low", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qhi = tcrv_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "activation-high", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "mf2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 16 : i64, source_kernel = "packed_i4_offset_binary_x_i8_product_rejects_i16_weight", status = "selected-lowering-boundary"} {
        %w_vec = tcrv_rvv.load %w, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %qlo_vec = tcrv_rvv.load %qlo, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %qhi_vec = tcrv_rvv.load %qhi, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        // expected-error @+1 {{requires the packed-i4 weight source vector to have type !tcrv_rvv.vector<i8, "mf4">}}
        %product = tcrv_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @packed_i4_offset_binary_x_i8_product_rejects_unknown_kind {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %w = tcrv_rvv.runtime_abi_value {c_name = "w", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qlo = tcrv_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-low", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qhi = tcrv_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-high", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "packed_i4_offset_binary_x_i8_product_rejects_unknown_kind", status = "selected-lowering-boundary"} {
        %w_vec = tcrv_rvv.load %w, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %qlo_vec = tcrv_rvv.load %qlo, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %qhi_vec = tcrv_rvv.load %qhi, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        // expected-error @+1 {{currently supports only kind "signed_packed_i4_offset_binary_x_i8_product"}}
        %product = tcrv_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary", product_relation = "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @packed_i4_offset_binary_x_i8_product_rejects_unknown_relation {
    tcrv.exec.variant @rvv attributes {origin = "rvv-plugin", requires = []} {
      %w = tcrv_rvv.runtime_abi_value {c_name = "w", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qlo = tcrv_rvv.runtime_abi_value {c_name = "qlo", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-low", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %qhi = tcrv_rvv.runtime_abi_value {c_name = "qhi", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "activation-high", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv, sew = 32 : i64, source_kernel = "packed_i4_offset_binary_x_i8_product_rejects_unknown_relation", status = "selected-lowering-boundary"} {
        %w_vec = tcrv_rvv.load %w, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %qlo_vec = tcrv_rvv.load %qlo, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        %qhi_vec = tcrv_rvv.load %qhi, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "mf4">
        // expected-error @+1 {{requires product_relation "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2"}}
        %product = tcrv_rvv.packed_i4_offset_binary_x_i8_product %w_vec, %qlo_vec, %qhi_vec, %vl {kind = "signed_packed_i4_offset_binary_x_i8_product", product_relation = "signed-i8mf4xi8mf4-to-i16mf2"} : !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vector<i8, "mf4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
      } : !tcrv_rvv.vl
    }
  }
}
