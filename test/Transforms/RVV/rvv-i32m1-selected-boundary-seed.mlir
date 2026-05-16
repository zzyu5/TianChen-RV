// RUN: tcrv-opt %s --tcrv-rvv-materialize-i32m1-selected-boundary-seed | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="func.func"
// RUN: tcrv-opt %s --tcrv-rvv-materialize-i32m1-selected-boundary-seed --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-source-seed-artifact-front-door-pipeline | FileCheck %s --check-prefix=PLAN
// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-rvv-materialize-i32m1-selected-boundary-seed 2>&1 | FileCheck %s --check-prefix=NO-BUILTIN

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
// BOUNDARY: tcrv.exec.capability @scalar_fallback
// BOUNDARY-SAME: id = "scalar.fallback"
// BOUNDARY-SAME: kind = "fallback"
// BOUNDARY-LABEL: tcrv.exec.variant @seed_rvv_i32_add
// BOUNDARY-SAME: origin = "rvv-plugin"
// BOUNDARY-SAME: requires = [@rvv]
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "lhs"
// BOUNDARY-SAME: c_type = "const int32_t *"
// BOUNDARY-SAME: purpose = "source-arg-0:lhs"
// BOUNDARY-SAME: role = "lhs-input-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "rhs"
// BOUNDARY-SAME: purpose = "source-arg-1:rhs"
// BOUNDARY-SAME: role = "rhs-input-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "out"
// BOUNDARY-SAME: purpose = "source-arg-2:out"
// BOUNDARY-SAME: role = "output-buffer"
// BOUNDARY: = tcrv_rvv.runtime_abi_value
// BOUNDARY-SAME: c_name = "n"
// BOUNDARY-SAME: c_type = "size_t"
// BOUNDARY-SAME: purpose = "source-arg-3:n"
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
// BOUNDARY: tcrv.exec.variant @seed_scalar_fallback
// BOUNDARY-SAME: fallback_role = "conservative"
// BOUNDARY-SAME: origin = "scalar-plugin"
// BOUNDARY-SAME: requires = [@scalar_fallback]
// BOUNDARY: tcrv.exec.dispatch
// BOUNDARY: tcrv.exec.case @seed_rvv_i32_add
// BOUNDARY-SAME: origin = "rvv-plugin"
// BOUNDARY-SAME: policy = "source-seed-selected-rvv-case"
// BOUNDARY: tcrv.exec.fallback @seed_scalar_fallback
// BOUNDARY-SAME: fallback_role = "conservative"
// BOUNDARY-SAME: origin = "scalar-plugin"

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: artifact_metadata = [{key = "rvv_emitc_lowerable_route", value = "rvv-i32m1-add-emitc-route"}, {key = "rvv_arithmetic_op", value = "add"}]
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: lowering_pipeline = "rvv-i32m1-arithmetic-emitc-route-family"
// PLAN-SAME: message = "RVV selected i32m1 arithmetic route materializes
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: required_capabilities = [@rvv]
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_parameters = [{c_name = "lhs", c_type = "const int32_t *"
// PLAN-SAME: runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @seed_rvv_i32_add
// PLAN: tcrv.exec.diagnostic {artifact_kind = "unsupported-emission-diagnostic"
// PLAN-SAME: emission_kind = "scalar-fallback-unsupported-emission"
// PLAN-SAME: origin = "scalar-plugin"
// PLAN-SAME: role = "dispatch fallback"
// PLAN-SAME: status = "unsupported"
// PLAN-SAME: target = @seed_scalar_fallback

// NO-BUILTIN: Unknown command line argument
// NO-BUILTIN-SAME: tcrv-rvv-materialize-i32m1-selected-boundary-seed
