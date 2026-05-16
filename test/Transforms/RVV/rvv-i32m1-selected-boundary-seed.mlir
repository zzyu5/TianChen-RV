// RUN: tcrv-opt %s --tcrv-rvv-materialize-i32m1-selected-boundary-seed | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="func.func"
// RUN: tcrv-opt %s --tcrv-rvv-materialize-i32m1-selected-boundary-seed --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-i32m1-selected-boundary-seed --tcrv-materialize-emission-plans --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC

module {
  func.func @seed(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) attributes {tcrv_rvv.lowering_seed = "i32m1_add"} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }
}

// BOUNDARY-LABEL: tcrv.exec.kernel @seed_kernel
// BOUNDARY: tcrv.exec.capability @rvv
// BOUNDARY-SAME: id = "rvv"
// BOUNDARY-SAME: kind = "isa-vector"
// BOUNDARY-LABEL: tcrv.exec.variant @seed_rvv_i32_add
// BOUNDARY-SAME: origin = "rvv-plugin"
// BOUNDARY-SAME: requires = [@rvv]
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "lhs"
// BOUNDARY-SAME: c_type = "const int32_t *"
// BOUNDARY-SAME: role = "lhs-input-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "rhs"
// BOUNDARY-SAME: role = "rhs-input-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "out"
// BOUNDARY-SAME: role = "output-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "n"
// BOUNDARY-SAME: c_type = "size_t"
// BOUNDARY-SAME: role = "runtime-element-count"
// BOUNDARY: tcrv_rvv.setvl
// BOUNDARY-SAME: lmul = "m1"
// BOUNDARY-SAME: sew = 32 : i64
// BOUNDARY: tcrv_rvv.with_vl
// BOUNDARY-SAME: lmul = "m1"
// BOUNDARY-SAME: sew = 32 : i64
// BOUNDARY: tcrv_rvv.i32_load
// BOUNDARY: tcrv_rvv.i32_load
// BOUNDARY: tcrv_rvv.i32_add
// BOUNDARY: tcrv_rvv.i32_store
// BOUNDARY: tcrv.exec.diagnostic
// BOUNDARY-SAME: reason = "variant-selected"
// BOUNDARY-SAME: selection_kind = "fallback-only"
// BOUNDARY-SAME: target = @seed_rvv_i32_add

// PLAN: tcrv.exec.diagnostic {artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: lowering_pipeline = "tcrv-rvv-i32m1-add-riscv-elf-object"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_parameters = [{c_name = "lhs"
// PLAN-SAME: c_name = "rhs"
// PLAN-SAME: c_name = "out"
// PLAN-SAME: c_name = "n"
// PLAN-SAME: runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @seed_rvv_i32_add

// EMITC: emitc.include <"riscv_vector.h">
// EMITC: emitc.func @tcrv_emitc_seed_kernel_seed_rvv_i32_add
// EMITC: tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
// EMITC: tcrv_emitc.source_op=tcrv_rvv.i32_load role=load op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_i32m1
// EMITC: tcrv_emitc.source_op=tcrv_rvv.i32_add role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m1
// EMITC: tcrv_emitc.source_op=tcrv_rvv.i32_store role=store op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_i32m1
