// RUN: not weft-opt %s --split-input-file --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s

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
  weft.exec.kernel @diverge_full_v {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      supported_sew = "8,16,32,64",
      supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"
    }
    weft.exec.variant @diverge_body attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %b = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %sum = weft_rvv.binary %a, %b, %vl {kind = "add"} : !weft_rvv.vector<i64, "m1">, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        weft_rvv.store %out, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// Profile A (full-V) ACCEPTS: the i64/SEW64 body is fully lowered to EmitC.
// (CHECK-DAG so the accept stdout and the reject stderr can interleave in any
// order under the merged 2>&1 stream.)
// CHECK-DAG: emitc.func @weft_emitc_diverge_full_v_diverge_body
// CHECK-DAG: callee=__riscv_vadd_vv_i64m1

// -----

// --- Profile B: zve32x. supported_sew = 8,16,32 (NO 64) -> REJECTED. ---
module {
  weft.exec.kernel @diverge_zve32x {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available",
      architecture = "riscv64",
      isa_vector_hints = "rv64imac_zve32x_zvl128b",
      supported_sew = "8,16,32",
      supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"
    }
    weft.exec.variant @diverge_body attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %b = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        %sum = weft_rvv.binary %a, %b, %vl {kind = "add"} : !weft_rvv.vector<i64, "m1">, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
        weft_rvv.store %out, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i64, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// Profile B (zve32x) REJECTS the IDENTICAL body fail-closed: the capability
// supported_sew excludes the SEW=64 typed body, so no backend emission driver
// legalizes it -- no emitc.func for this kernel is emitted.
// CHECK-DAG: error: Weft-RV EmitC lowerable materialization failed: no registered backend emission driver fully legalizes the selected variant @diverge_body body to EmitC
// CHECK-NOT: emitc.func @weft_emitc_diverge_zve32x_diverge_body
