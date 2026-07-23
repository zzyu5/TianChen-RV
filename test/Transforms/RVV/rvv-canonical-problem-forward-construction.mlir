// RUN: weft-opt %s --split-input-file --weft-execution-planning-pipeline | FileCheck %s

// Exact-P forward-construction witness. None of these modules contains a source
// adapter or a prebuilt variant/body; the RVV owner must derive the candidate,
// pass legality/selection, and construct the typed body from P + bound c_o.

// CHECK-LABEL: weft.exec.kernel @vector_exact_pointer
// CHECK: weft.exec.i32_vector_binary_problem @ignored_add
// CHECK: weft.exec.i32_vector_binary_problem @chosen_mul
// CHECK: weft.exec.variant @rvv_vector_mul
// CHECK: weft_rvv.binary
// CHECK-SAME: kind = "mul"
module {
  weft.exec.target @rvv_profile {
    capability_providers = [@rvv_cap],
    construction_domain = "riscv-execution",
    id = "test.rvv.profile",
    status = "available",
    target_kind = "profile"
  }
  weft.exec.capability @rvv_cap {
    architecture = "riscv64",
    id = "rvv",
    kind = "isa-vector",
    minimum_vlen = 128 : i64,
    rvv_version = "1.0",
    status = "available",
    supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8",
    supported_sew = "8,16,32,64",
    vreg_count = 32 : i64
  }
  weft.exec.kernel @vector_exact_pointer attributes {
    problem = @chosen_mul,
    target = @rvv_profile
  } {
    weft.exec.i32_vector_binary_problem @ignored_add {
      kind = "add",
      source_vector_lanes = 4 : i64
    }
    weft.exec.i32_vector_binary_problem @chosen_mul {
      kind = "mul",
      source_vector_lanes = 8 : i64
    }
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @compare_scalar
// CHECK: weft.exec.variant @rvv_vector_runtime_scalar_cmp_select_sle
// CHECK: weft_rvv.splat
// CHECK: weft_rvv.compare
// CHECK-SAME: kind = "sle"
// CHECK: weft_rvv.select
module {
  weft.exec.target @rvv_profile {
    capability_providers = [@rvv_cap], construction_domain = "riscv-execution",
    id = "test.rvv.profile", status = "available", target_kind = "profile"
  }
  weft.exec.capability @rvv_cap {
    architecture = "riscv64", id = "rvv", kind = "isa-vector",
    minimum_vlen = 128 : i64, rvv_version = "1.0", status = "available",
    supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8",
    supported_sew = "8,16,32,64", vreg_count = 32 : i64
  }
  weft.exec.kernel @compare_scalar attributes {
    problem = @compare_problem, target = @rvv_profile
  } {
    weft.exec.i32_vector_compare_select_problem @compare_problem {
      predicate = "sle", rhs_form = "runtime-scalar",
      source_vector_lanes = 16 : i64
    }
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @widening_reduce
// CHECK: weft.exec.variant @rvv_widening_dot_reduce_i8
// CHECK: weft_rvv.widening_product
// CHECK: weft_rvv.standalone_reduce
module {
  weft.exec.target @rvv_profile {
    capability_providers = [@rvv_cap], construction_domain = "riscv-execution",
    id = "test.rvv.profile", status = "available", target_kind = "profile"
  }
  weft.exec.capability @rvv_cap {
    architecture = "riscv64", id = "rvv", kind = "isa-vector",
    minimum_vlen = 128 : i64, rvv_version = "1.0", status = "available",
    supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8",
    supported_sew = "8,16,32,64", vreg_count = 32 : i64
  }
  weft.exec.kernel @widening_reduce attributes {
    problem = @reduce_problem, target = @rvv_profile
  } {
    weft.exec.i8_widening_dot_reduce_problem @reduce_problem {
      block_length = 32 : i64, dequantize_to_f32 = false
    }
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @widening_dequant
// CHECK: weft.exec.variant @rvv_widening_dot_reduce_dequantize_i8
// CHECK: weft_rvv.widening_product
// CHECK: weft_rvv.standalone_reduce
// CHECK: weft_rvv.dequantize
module {
  weft.exec.target @rvv_profile {
    capability_providers = [@rvv_cap], construction_domain = "riscv-execution",
    id = "test.rvv.profile", status = "available", target_kind = "profile"
  }
  weft.exec.capability @rvv_cap {
    architecture = "riscv64", id = "rvv", kind = "isa-vector",
    minimum_vlen = 256 : i64, rvv_version = "1.0", status = "available",
    supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8",
    supported_sew = "8,16,32,64", vreg_count = 32 : i64
  }
  weft.exec.kernel @widening_dequant attributes {
    problem = @dequant_problem, target = @rvv_profile
  } {
    weft.exec.i8_widening_dot_reduce_problem @dequant_problem {
      block_length = 32 : i64, dequantize_to_f32 = true
    }
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @packed_i4
// CHECK: weft.exec.variant @rvv_packed_i4_offset_binary_dot_i8
// CHECK: role = "rhs-secondary-input-buffer"
// CHECK: weft_rvv.packed_i4_offset_binary_x_i8_product
module {
  weft.exec.target @rvv_profile {
    capability_providers = [@rvv_cap], construction_domain = "riscv-execution",
    id = "test.rvv.profile", status = "available", target_kind = "profile"
  }
  weft.exec.capability @rvv_cap {
    architecture = "riscv64", id = "rvv", kind = "isa-vector",
    minimum_vlen = 128 : i64, rvv_version = "1.0", status = "available",
    supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8",
    supported_sew = "8,16,32,64", vreg_count = 32 : i64
  }
  weft.exec.kernel @packed_i4 attributes {
    problem = @packed_problem, target = @rvv_profile
  } {
    weft.exec.packed_i4_q8_dot_problem @packed_problem {
      block_length = 32 : i64
    }
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @codebook_i4
// CHECK: weft.exec.variant @rvv_codebook_gather_dot_i8
// CHECK: weft_rvv.codebook_table_broadcast
// CHECK: weft_rvv.codebook_gather_x_i8_product
module {
  weft.exec.target @rvv_profile {
    capability_providers = [@rvv_cap], construction_domain = "riscv-execution",
    id = "test.rvv.profile", status = "available", target_kind = "profile"
  }
  weft.exec.capability @rvv_cap {
    architecture = "riscv64", id = "rvv", kind = "isa-vector",
    minimum_vlen = 128 : i64, rvv_version = "1.0", status = "available",
    supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8",
    supported_sew = "8,16,32,64", vreg_count = 32 : i64
  }
  weft.exec.kernel @codebook_i4 attributes {
    problem = @codebook_problem, target = @rvv_profile
  } {
    weft.exec.codebook_i4_q8_dot_problem @codebook_problem {
      block_length = 16 : i64,
      codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>,
      table_symbol = "weft_iq4_nl_kvalues"
    }
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @quantized_block
// CHECK: weft.exec.variant @rvv_q4_0_q8_0_block_dot
// CHECK: weft_rvv.typed_flat_block_dot_loop_body
// CHECK-SAME: weft_rvv.flat_body_family = "shared"
// CHECK-SAME: weft_rvv.flat_decode_primitive = "offset-binary-nibble"
module {
  weft.exec.target @rvv_profile {
    capability_providers = [@rvv_cap], construction_domain = "riscv-execution",
    id = "test.rvv.profile", status = "available", target_kind = "profile"
  }
  weft.exec.capability @rvv_cap {
    architecture = "riscv64", id = "rvv", kind = "isa-vector",
    minimum_vlen = 128 : i64, rvv_version = "1.0", status = "available",
    supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8",
    supported_sew = "8,16,32,64", vreg_count = 32 : i64
  }
  weft.exec.kernel @quantized_block attributes {
    problem = @q4_0_problem, target = @rvv_profile
  } {
    weft.exec.quantized_block_dot_problem @q4_0_problem {
      activation_block_stride = 34 : i64,
      activation_encoding = "q8_0",
      qk = 32 : i64,
      topology = "dual-fp16-per-block-d_x.d_y",
      weight_block_stride = 18 : i64,
      weight_encoding = "q4_0"
    }
  }
}
