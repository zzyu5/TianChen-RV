// RUN: not tcrv-opt %s --split-input-file --tcrv-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s

// N1 COVERAGE INCREMENT THAT CARRIES THE DIVERGENCE AXIS: the new f64 (SEW=64,
// double-precision) elementwise coverage rung is itself capability-gated on
// supported_sew. The SAME f64 kernel body is ACCEPTED on a full-V profile
// (supported_sew includes 64) and REJECTED fail-closed on an embedded zve32x
// profile (supported_sew = 8,16,32). So the broadened coverage is not just a
// raw op-count increment -- it lands a feature whose legality DIVERGES across
// two real profiles, which is the spec's N1 evidence form.

// --- Profile A: full-V (rv64gcv), supported_sew includes 64 -> f64 ACCEPTED. ---
module {
  tcrv.exec.kernel @f64_full_v {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      supported_sew = "8,16,32,64",
      supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"
    }
    tcrv.exec.variant @f64_body attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const double *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const double *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "double *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f64, "m1">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f64, "m1">
        %sum = tcrv_rvv.binary %a, %b, %vl {kind = "add"} : !tcrv_rvv.vector<f64, "m1">, !tcrv_rvv.vector<f64, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f64, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f64, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// Profile A (full-V) ACCEPTS the f64/SEW64 body -> the double-precision EmitC.
// CHECK-DAG: emitc.func @tcrv_emitc_f64_full_v_f64_body
// CHECK-DAG: callee=__riscv_vfadd_vv_f64m1

// -----

// --- Profile B: zve32x, supported_sew = 8,16,32 (NO 64) -> f64 REJECTED. ---
module {
  tcrv.exec.kernel @f64_zve32x {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available",
      architecture = "riscv64",
      isa_vector_hints = "rv64imac_zve32x_zvl128b",
      supported_sew = "8,16,32",
      supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"
    }
    tcrv.exec.variant @f64_body attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const double *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const double *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "double *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f64, "m1">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f64, "m1">
        %sum = tcrv_rvv.binary %a, %b, %vl {kind = "add"} : !tcrv_rvv.vector<f64, "m1">, !tcrv_rvv.vector<f64, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f64, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f64, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// Profile B (zve32x) REJECTS the IDENTICAL f64/SEW64 body fail-closed.
// CHECK-DAG: error: TianChen-RV EmitC lowerable materialization failed: no registered backend emission driver fully legalizes the selected variant @f64_body body to EmitC
// CHECK-NOT: emitc.func @tcrv_emitc_f64_zve32x_f64_body
