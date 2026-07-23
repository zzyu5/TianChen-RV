// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: weft.exec.kernel @valid_anchor
// CHECK-SAME: problem = @mac
weft.exec.kernel @valid_anchor attributes {problem = @mac} {
  // CHECK: weft.exec.int8_mac_problem @mac
  // CHECK-SAME: k = 8
  // CHECK-SAME: lhs_signedness = #weft<integer_signedness signed>
  // CHECK-SAME: m = 4
  // CHECK-SAME: n = 4
  // CHECK-SAME: rhs_signedness = #weft<integer_signedness signed>
  weft.exec.int8_mac_problem @mac {
    lhs_signedness = #weft<integer_signedness signed>,
    rhs_signedness = #weft<integer_signedness signed>,
    m = 4 : i64,
    n = 4 : i64,
    k = 8 : i64
  }
}

// -----

// expected-error@+1 {{problem references unknown direct canonical problem @missing}}
weft.exec.kernel @dangling_anchor attributes {problem = @missing} {
}

// -----

// expected-error@+1 {{problem @cap resolves to a direct symbol that is not a canonical operator problem}}
weft.exec.kernel @foreign_anchor attributes {problem = @cap} {
  weft.exec.capability @cap {
    id = "spacemit.ime",
    kind = "isa-matrix-vector-backed",
    status = "available"
  }
}

// -----

weft.exec.kernel @invalid_geometry attributes {problem = @mac} {
  // expected-error@+1 {{requires positive logical MAC geometry m/n/k; got 4x0x8}}
  weft.exec.int8_mac_problem @mac {
    lhs_signedness = #weft<integer_signedness signed>,
    rhs_signedness = #weft<integer_signedness signed>,
    m = 4 : i64,
    n = 0 : i64,
    k = 8 : i64
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @bounded_problem_schemas
// CHECK: weft.exec.int8_sliding_mac_problem @slide
// CHECK: weft.exec.block_q4_0_contraction_problem @q40
// CHECK: weft.exec.block_q8_0_contraction_problem @q80
// CHECK: weft.exec.block_q4_K_contraction_problem @q4k
// CHECK: weft.exec.ternary_q2_q8_block_dot_problem @tq2
weft.exec.kernel @bounded_problem_schemas attributes {problem = @q4k} {
  weft.exec.int8_sliding_mac_problem @slide {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64, slide = 1 : i64}
  weft.exec.block_q4_0_contraction_problem @q40 {activation_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 256 : i64, qk = 32 : i64, weight_block_stride = 18 : i64, weight_scale_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
  weft.exec.block_q8_0_contraction_problem @q80 {activation_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 256 : i64, qk = 32 : i64, weight_block_stride = 34 : i64, weight_scale_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
  weft.exec.block_q4_K_contraction_problem @q4k {activation_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 256 : i64, qk = 256 : i64, weight_block_stride = 144 : i64, weight_scale_byte_offset = 4 : i64, weight_quant_byte_offset = 16 : i64, subblock_length = 32 : i64, num_subblocks = 8 : i64, scale_bits = 6 : i64, scale_table_bytes = 12 : i64}
  weft.exec.ternary_q2_q8_block_dot_problem @tq2 {qk = 256 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, weight_d_byte_offset = 64 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64}
}

// -----

weft.exec.kernel @invalid_slide attributes {problem = @slide} {
  // expected-error@+1 {{requires bounded source slide geometry in {1,2,3}; got 4}}
  weft.exec.int8_sliding_mac_problem @slide {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64, slide = 4 : i64}
}

// -----

weft.exec.kernel @invalid_q4k attributes {problem = @q4k} {
  // expected-error@+1 {{requires canonical q4_K layout}}
  weft.exec.block_q4_K_contraction_problem @q4k {activation_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 256 : i64, qk = 256 : i64, weight_block_stride = 144 : i64, weight_scale_byte_offset = 4 : i64, weight_quant_byte_offset = 16 : i64, subblock_length = 32 : i64, num_subblocks = 8 : i64, scale_bits = 5 : i64, scale_table_bytes = 12 : i64}
}
