// N1 ISA-GENERATION EVIDENCE (the deepest N1 divergence axis). The SAME kernel
// body -- an i32 / SEW=32 elementwise add that REQUIRES the ratified RVV1.0
// tail/mask-agnostic (ta/ma) vector policy -- is routed to DIFFERENT legality
// outcomes on two RVV profiles that share EVERY other capability axis (VLEN,
// SEW, LMUL) and differ ONLY in the RVV ISA GENERATION. The generation is the
// `rvv_version` capability fact the plugin-local C++ authority derives from each
// profile's -march and stamps onto the bare @rvv provider; the EmitC legality
// gate then queries it:
//   * Profile A (--march=rv64gcv,             RVV1.0): the agnostic-policy body
//     is ACCEPTED and lowered to __riscv_vadd_vv_i32m1.
//   * Profile B (--march=rv64gc_xtheadvector, RVV0.7 / C920): the IDENTICAL body
//     is REJECTED fail-closed -- RVV0.7 LACKS the ratified ta/ma policy, so the
//     version capability gates the agnostic-policy body out (no EmitC emitted).
//
// The two RUN lines differ ONLY in the -march value; the input IR is identical
// and carries no version attr. This is the ISA-generation instance of the spec's
// N1 bar ("同一 kernel 在多个真实 profile 上，被 capability 查询导向不同的合法性
// /选择/dispatch 结果"), the axis K1's VLEN divergence cannot reach: rv64gcv and
// rv64gc_xtheadvector are NOT binary compatible (a 1.0 binary SIGILLs on the
// C920) yet share VLEN=128 / SEW 8..64 / all LMUL -- only the generation differs.
//
// HARDWARE GROUNDING: the C920 runs 0.7.1 `th.v*` under rv64gc_xtheadvector and
// SIGILLs on a 1.0 rv64gcv binary (proven on the real silicon). This lit gates
// at the CAPABILITY-MODEL level (the version fact + the ta/ma legality
// divergence); full RVV0.7 EMISSION (the 0.7 `th.v*` spelling / 0.7 vsetvli) is
// deferred and out of scope here.

// First, the pass-only RUN lines prove CAUSATION (not correlation): the derived
// rvv_version that lands on the bare @rvv provider differs purely by the -march,
// while supported_sew is IDENTICAL (both full vector units) -- so the legality
// divergence below is the ISA-generation axis, not a SEW/LMUL axis.

// RUN: tcrv-opt %s \
// RUN:   --tcrv-rvv-materialize-probed-capability-axes=march=rv64gcv \
// RUN: | FileCheck %s --check-prefix=STAMP-RVV10

// RUN: tcrv-opt %s \
// RUN:   --tcrv-rvv-materialize-probed-capability-axes=march=rv64gc_xtheadvector \
// RUN: | FileCheck %s --check-prefix=STAMP-RVV07

// Then the full pipeline proves the legality DIVERGENCE driven by the version.

// RUN: tcrv-opt %s \
// RUN:   --tcrv-rvv-materialize-probed-capability-axes=march=rv64gcv \
// RUN:   --tcrv-materialize-emitc-lowerable-routes \
// RUN: | FileCheck %s --check-prefix=RVV10

// RUN: not tcrv-opt %s \
// RUN:   --tcrv-rvv-materialize-probed-capability-axes=march=rv64gc_xtheadvector \
// RUN:   --tcrv-materialize-emitc-lowerable-routes 2>&1 \
// RUN: | FileCheck %s --check-prefix=RVV07

module {
  tcrv.exec.kernel @diverge_rvv_version {
    // Bare RVV capability provider: identity only, NO supported_sew /
    // supported_lmul / rvv_version. The version (and the support axes) are
    // materialized live by the march below.
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available",
      architecture = "riscv64"
    }
    tcrv.exec.variant @diverge_body attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %a, %b, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// Causation: the pass stamps the march-derived ISA generation onto the bare @rvv
// provider. rv64gcv -> rvv_version 1.0; rv64gc_xtheadvector -> rvv_version 0.7.
// supported_sew is the SAME for both (8,16,32,64) -- the only divergence is the
// generation, so the legality split below is purely the ISA-generation axis.
// STAMP-RVV10: tcrv.exec.capability @rvv
// STAMP-RVV10-SAME: rvv_version = "1.0"
// STAMP-RVV10-SAME: supported_sew = "8,16,32,64"
// STAMP-RVV07: tcrv.exec.capability @rvv
// STAMP-RVV07-SAME: rvv_version = "0.7"
// STAMP-RVV07-SAME: supported_sew = "8,16,32,64"

// Profile A (RVV1.0 rv64gcv): the live-materialized rvv_version is 1.0, which has
// the ratified ta/ma policy, so the agnostic-policy body is fully lowered to
// EmitC.
// RVV10: emitc.func @tcrv_emitc_diverge_rvv_version_diverge_body
// RVV10: callee=__riscv_vadd_vv_i32m1

// Profile B (RVV0.7 rv64gc_xtheadvector): the live-materialized rvv_version is
// 0.7, which LACKS the ratified ta/ma policy the typed body requires, so the
// version capability gates it out fail-closed -- no backend emission driver
// legalizes it, no emitc.func is emitted.
// RVV07: error: TianChen-RV EmitC lowerable materialization failed: no registered backend emission driver fully legalizes the selected variant @diverge_body body to EmitC
// RVV07-NOT: emitc.func @tcrv_emitc_diverge_rvv_version_diverge_body
