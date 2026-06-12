// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | FileCheck %s

// Stage 3 换心 re-target: the legacy string route rejected a selected body with
// no tcrv_rvv.store ("requires exactly one tcrv_rvv.store op") as a structural
// scope-limit. With the string-plan owner retired, the real RVV->emitc
// DialectConversion is the authority: it faithfully lowers the typed dataflow
// that is present. A load/load/binary body with no store is structurally valid
// IR (the computed vector is simply unused), so the conversion materializes the
// load + compute it carries; there is no malformed type or operand to reject.
// The body therefore now MATERIALIZES through the conversion rather than failing
// the deleted legacy structural check. (A meaningful store-bearing body is
// pinned by rvv-first-slice-materialization.mlir.)
module {
  tcrv.exec.kernel @rvv_missing_store_rejected {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_missing_store attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_rvv_missing_store_rejected_rvv_missing_store
// CHECK: callee=__riscv_vle32_v_i32m1
// CHECK: callee=__riscv_vadd_vv_i32m1
