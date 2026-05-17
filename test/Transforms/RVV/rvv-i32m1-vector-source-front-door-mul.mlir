// RUN: tcrv-opt %s --tcrv-rvv-materialize-i32m1-vector-source-front-door | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="func.func" --implicit-check-not="tcrv_rvv.i32_add" --implicit-check-not="tcrv_rvv.i32_sub"
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN

module {
  func.func @vector_source_mul(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %product = arith.muli %a, %b : vector<4xi32>
      vector.store %product, %out[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }
}

// BOUNDARY-LABEL: tcrv.exec.kernel @vector_source_mul_kernel
// BOUNDARY-LABEL: tcrv.exec.variant @vector_source_mul_rvv_i32_mul
// BOUNDARY-SAME: origin = "rvv-plugin"
// BOUNDARY-SAME: requires = [@rvv]
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "lhs"
// BOUNDARY-SAME: role = "lhs-input-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "rhs"
// BOUNDARY-SAME: role = "rhs-input-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "out"
// BOUNDARY-SAME: role = "output-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "n"
// BOUNDARY-SAME: role = "runtime-element-count"
// BOUNDARY: tcrv_rvv.setvl
// BOUNDARY: tcrv_rvv.with_vl
// BOUNDARY: tcrv_rvv.i32_load
// BOUNDARY: tcrv_rvv.i32_load
// BOUNDARY: tcrv_rvv.i32_mul
// BOUNDARY: tcrv_rvv.i32_store
// BOUNDARY: tcrv.exec.case @vector_source_mul_rvv_i32_mul
// BOUNDARY-SAME: origin = "rvv-plugin"

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_metadata = [{key = "rvv_emitc_lowerable_route", value = "rvv-i32m1-mul-emitc-route"}
// PLAN-SAME: {key = "rvv_arithmetic_op", value = "mul"}
// PLAN-SAME: {key = "tcrv_rvv.bounded_slice", value = "multi-vl-i32m1-arithmetic"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_pipeline = "rvv-i32m1-arithmetic-emitc-route-family"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: runtime_abi = "rvv-i32m1-mul-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_name = "rvv-i32m1-mul-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @vector_source_mul_rvv_i32_mul
