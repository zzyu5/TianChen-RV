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

// RUN: weft-opt %s \
// RUN:   --weft-rvv-materialize-probed-capability-axes=march=rv64gcv \
// RUN: | FileCheck %s --check-prefix=STAMP-FULLV

// RUN: weft-opt %s \
// RUN:   --weft-rvv-materialize-probed-capability-axes=march=rv64gc_zve32x \
// RUN: | FileCheck %s --check-prefix=STAMP-ZVE32X

// Then the full pipeline proves the legality DIVERGENCE driven by those axes.

// RUN: weft-opt %s \
// RUN:   --weft-rvv-materialize-probed-capability-axes=march=rv64gcv \
// RUN:   --weft-materialize-emitc-lowerable-routes \
// RUN: | FileCheck %s --check-prefix=FULLV

// RUN: not weft-opt %s \
// RUN:   --weft-rvv-materialize-probed-capability-axes=march=rv64gc_zve32x \
// RUN:   --weft-materialize-emitc-lowerable-routes 2>&1 \
// RUN: | FileCheck %s --check-prefix=ZVE32X

module {
  weft.exec.kernel @diverge_live_probed {
    // Bare RVV capability provider: identity only, NO supported_sew /
    // supported_lmul. The axes are materialized live by the march below.
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available",
      architecture = "riscv64"
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

// Causation: the pass stamps the march-derived allow-list onto the bare @rvv
// provider. Full-V advertises SEW up to 64; zve32x stops at 32.
// STAMP-FULLV: weft.exec.capability @rvv
// STAMP-FULLV-SAME: supported_sew = "8,16,32,64"
// STAMP-ZVE32X: weft.exec.capability @rvv
// STAMP-ZVE32X-SAME: supported_sew = "8,16,32"

// Profile A (full-V rv64gcv): the live-materialized supported_sew includes 64,
// so the i64/SEW64 body is fully lowered to EmitC.
// FULLV: emitc.func @weft_emitc_diverge_live_probed_diverge_body
// FULLV: callee=__riscv_vadd_vv_i64m1

// Profile B (zve32x): the live-materialized supported_sew = 8,16,32 excludes the
// SEW=64 typed body, so the capability gates it out fail-closed -- no backend
// emission driver legalizes it, no emitc.func is emitted.
// ZVE32X: error: Weft-RV EmitC lowerable materialization failed: no registered backend emission driver fully legalizes the selected variant @diverge_body body to EmitC
// ZVE32X-NOT: emitc.func @weft_emitc_diverge_live_probed_diverge_body
