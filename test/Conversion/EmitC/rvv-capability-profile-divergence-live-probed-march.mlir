// LIVE N1 EVIDENCE: the SAME kernel body (an i64 / SEW=64 elementwise add) is
// routed to DIFFERENT legality outcomes purely by the SELECTED RVV -march, with
// NO hand-authored supported_sew / supported_lmul fixture attributes. The
// in-kernel capability provider declares only its identity (id="rvv",
// kind="isa-vector"); the divergence axes are MATERIALIZED onto it by the
// plugin-local capability authority from the live profile selection (-march),
// then queried by the EmitC legality gate. This closes the probe->gate seam:
// the live profile, not a fixture attr, drives the in-IR divergence.
//
//   * Profile A (--march=rv64gcv): the authority derives supported_sew up to 64
//     (full-V), stamps it on @rvv -> the SEW=64 body is ACCEPTED and lowered to
//     __riscv_vadd_vv_i64m1.
//   * Profile B (--march=rv64gc_zve32x): the authority derives supported_sew =
//     8,16,32 (the embedded 32-bit-element tier, no 64), stamps it on @rvv ->
//     the IDENTICAL SEW=64 body is REJECTED fail-closed (no EmitC emitted).
//
// The two RUN lines differ ONLY in the -march value; the input IR is identical
// and carries no axis attrs. This instantiates the spec's N1 bar ("同一 kernel
// 在多个真实 profile 上，被 capability 查询导向不同的合法性/选择/dispatch 结果")
// with the axes flowing LIVE from the profile selection.

// First, the pass-only RUN lines prove CAUSATION (not just correlation): the
// derived supported_sew allow-list that lands on the bare @rvv provider differs
// purely by the -march. rv64gcv -> includes 64; rv64gc_zve32x -> 8,16,32 only.

// RUN: tcrv-opt %s \
// RUN:   --tcrv-rvv-materialize-probed-capability-axes=march=rv64gcv \
// RUN: | FileCheck %s --check-prefix=STAMP-FULLV

// RUN: tcrv-opt %s \
// RUN:   --tcrv-rvv-materialize-probed-capability-axes=march=rv64gc_zve32x \
// RUN: | FileCheck %s --check-prefix=STAMP-ZVE32X

// Then the full pipeline proves the legality DIVERGENCE driven by those axes.

// RUN: tcrv-opt %s \
// RUN:   --tcrv-rvv-materialize-probed-capability-axes=march=rv64gcv \
// RUN:   --tcrv-materialize-emitc-lowerable-routes \
// RUN: | FileCheck %s --check-prefix=FULLV

// RUN: not tcrv-opt %s \
// RUN:   --tcrv-rvv-materialize-probed-capability-axes=march=rv64gc_zve32x \
// RUN:   --tcrv-materialize-emitc-lowerable-routes 2>&1 \
// RUN: | FileCheck %s --check-prefix=ZVE32X

module {
  tcrv.exec.kernel @diverge_live_probed {
    // Bare RVV capability provider: identity only, NO supported_sew /
    // supported_lmul. The axes are materialized live by the march below.
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available",
      architecture = "riscv64"
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

// Causation: the pass stamps the march-derived allow-list onto the bare @rvv
// provider. Full-V advertises SEW up to 64; zve32x stops at 32.
// STAMP-FULLV: tcrv.exec.capability @rvv
// STAMP-FULLV-SAME: supported_sew = "8,16,32,64"
// STAMP-ZVE32X: tcrv.exec.capability @rvv
// STAMP-ZVE32X-SAME: supported_sew = "8,16,32"

// Profile A (full-V rv64gcv): the live-materialized supported_sew includes 64,
// so the i64/SEW64 body is fully lowered to EmitC.
// FULLV: emitc.func @tcrv_emitc_diverge_live_probed_diverge_body
// FULLV: callee=__riscv_vadd_vv_i64m1

// Profile B (zve32x): the live-materialized supported_sew = 8,16,32 excludes the
// SEW=64 typed body, so the capability gates it out fail-closed -- no backend
// emission driver legalizes it, no emitc.func is emitted.
// ZVE32X: error: TianChen-RV EmitC lowerable materialization failed: no registered backend emission driver fully legalizes the selected variant @diverge_body body to EmitC
// ZVE32X-NOT: emitc.func @tcrv_emitc_diverge_live_probed_diverge_body
