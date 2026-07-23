// RUN: not weft-opt %s --weft-execution-planning-pipeline 2>&1 | FileCheck %s

// A near-miss P must not fall back to the first row with the same encoding
// strings. Exact geometry is part of g and the owner fails closed before body
// construction.
// CHECK: error: RVV canonical-problem construction rejected: has no exact RVV block-dot formula

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
  weft.exec.kernel @wrong_q4_geometry attributes {
    problem = @wrong_problem, target = @rvv_profile
  } {
    weft.exec.quantized_block_dot_problem @wrong_problem {
      activation_block_stride = 34 : i64,
      activation_encoding = "q8_0",
      qk = 64 : i64,
      topology = "dual-fp16-per-block-d_x.d_y",
      weight_block_stride = 18 : i64,
      weight_encoding = "q4_0"
    }
  }
}
