// RUN: not tcrv-opt %s --split-input-file --tcrv-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s

// N1 EVIDENCE: the SAME kernel body (an i64 / SEW=64 elementwise add) is routed
// to DIFFERENT legality outcomes on TWO real RVV capability profiles, driven
// purely by the capability query (supported_sew). The two providers are
// structurally identical and differ ONLY in their supported_sew allow-list --
// the allow-list the C++ probe->capability authority derives from each real
// profile's ISA (-march): full-V (rv64gcv) advertises supported_sew up to 64,
// the embedded zve32x profile advertises 8,16,32 only. So:
//   * profile A (full-V, supported_sew includes 64)  -> body ACCEPTED, lowered;
//   * profile B (zve32x, supported_sew excludes 64)  -> body REJECTED (fail-
//     closed, no EmitC emitted).
// This instantiates the spec's N1 bar: "同一 kernel 在多个真实 profile 上，被
// capability 查询导向不同的合法性/选择/dispatch 结果."

// --- Profile A: full-V (rv64gcv). supported_sew includes 64 -> ACCEPTED. ---
module {
  tcrv.exec.kernel @diverge_full_v {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      supported_sew = "8,16,32,64",
      supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"
    }
    tcrv.exec.variant @diverge_body attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
        %sum = tcrv_rvv.binary %a, %b, %vl {kind = "add"} : !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// Profile A (full-V) ACCEPTS: the i64/SEW64 body is fully lowered to EmitC.
// (CHECK-DAG so the accept stdout and the reject stderr can interleave in any
// order under the merged 2>&1 stream.)
// CHECK-DAG: emitc.func @tcrv_emitc_diverge_full_v_diverge_body
// CHECK-DAG: callee=__riscv_vadd_vv_i64m1

// -----

// --- Profile B: zve32x. supported_sew = 8,16,32 (NO 64) -> REJECTED. ---
module {
  tcrv.exec.kernel @diverge_zve32x {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available",
      architecture = "riscv64",
      isa_vector_hints = "rv64imac_zve32x_zvl128b",
      supported_sew = "8,16,32",
      supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"
    }
    tcrv.exec.variant @diverge_body attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
        %sum = tcrv_rvv.binary %a, %b, %vl {kind = "add"} : !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// Profile B (zve32x) REJECTS the IDENTICAL body fail-closed: the capability
// supported_sew excludes the SEW=64 typed body, so no backend emission driver
// legalizes it -- no emitc.func for this kernel is emitted.
// CHECK-DAG: error: TianChen-RV EmitC lowerable materialization failed: no registered backend emission driver fully legalizes the selected variant @diverge_body body to EmitC
// CHECK-NOT: emitc.func @tcrv_emitc_diverge_zve32x_diverge_body
