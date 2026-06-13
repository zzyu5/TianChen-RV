// RUN: not tcrv-opt %s --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s

// Stage 3 换心 — adversarial guard for the signed widening CONVERSION family. A
// widening_convert carrying a malformed conversion kind must NOT mislower into a
// vwcvt intrinsic: the op verifier fail-closes the body before the conversion
// can take it over, and the conversion's own emitWideningConvert guard refuses
// any kind other than the two bounded signed widening kinds
// ("sign_extend_widen_vf2", "widen_i32_to_i64"). Either gate refusing keeps a
// malformed body off the conversion path (no __riscv_vwcvt_x_x_v emitted, no
// emitc.func materialized).

module {
  tcrv.exec.kernel @rvv_bad_widen_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_bad_widen attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "bad-widen:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "bad-widen:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bad-widen:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_bad_widen, sew = 64 : i64, source_kernel = "rvv_bad_widen_kernel", status = "selected-lowering-boundary"} {
        %source = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %widened = tcrv_rvv.widening_convert %source, %vl {kind = "bogus_widen_kind"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m2">
        tcrv_rvv.store %out, %widened, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i64, "m2">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: error
// CHECK-NOT: __riscv_vwcvt_x_x_v
// CHECK-NOT: emitc.func
