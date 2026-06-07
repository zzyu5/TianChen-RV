// RUN: tcrv-opt %s --tcrv-rvv-materialize-vector-add-source-front-door | FileCheck %s --check-prefix=MATERIALIZED --implicit-check-not="tcrv_rvv.i32_"
// RUN: tcrv-opt %s --tcrv-rvv-materialize-vector-add-source-front-door --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: tcrv-opt %s --tcrv-rvv-materialize-vector-add-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: tcrv-opt %s --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PIPELINE --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"

module attributes {
  tcrv_rvv.source_front_door = "bounded_vector_source",
  tcrv_rvv.source_kernel = "rvv_vector_add_from_vector_source"
} {
  func.func @source_vector_add(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %sum = arith.addi %a, %b : vector<4xi32>
    vector.transfer_write %sum, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// MATERIALIZED-LABEL: tcrv.exec.kernel @rvv_vector_add_from_vector_source
// MATERIALIZED: tcrv.exec.capability @rvv
// MATERIALIZED-SAME: id = "rvv"
// MATERIALIZED-SAME: kind = "isa-vector"
// MATERIALIZED: tcrv.exec.variant @rvv_vector_add
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "lhs"{{.*}}role = "lhs-input-buffer"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "rhs"{{.*}}role = "rhs-input-buffer"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "out"{{.*}}role = "output-buffer"
// MATERIALIZED: %[[N:.*]] = tcrv_rvv.runtime_abi_value {{.*}}c_name = "n"{{.*}}role = "runtime-element-count"{{.*}} : index
// MATERIALIZED: %[[VL:.*]] = tcrv_rvv.setvl %[[N]]
// MATERIALIZED-SAME: lmul = "m1"
// MATERIALIZED-SAME: sew = 32
// MATERIALIZED: tcrv_rvv.with_vl %[[VL]]
// MATERIALIZED-SAME: required_capabilities = [@rvv]
// MATERIALIZED-SAME: rvv_construction_protocol = "extension-family-construction-protocol.v1"
// MATERIALIZED-SAME: rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family"
// MATERIALIZED-SAME: selected_variant = @rvv_vector_add
// MATERIALIZED-SAME: source_kernel = "rvv_vector_add_from_vector_source"
// MATERIALIZED: tcrv_rvv.load
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv_rvv.load
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv_rvv.binary
// MATERIALIZED-SAME: kind = "add"
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv_rvv.store
// MATERIALIZED: tcrv.exec.variant @rvv_vector_add_scalar_fallback
// MATERIALIZED-SAME: fallback_role = "conservative"
// MATERIALIZED: tcrv.exec.dispatch
// MATERIALIZED: tcrv.exec.case @rvv_vector_add
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED: tcrv.exec.fallback @rvv_vector_add_scalar_fallback
// MATERIALIZED-SAME: origin = "scalar-plugin"

// PLAN: {key = "rvv_selected_body_operation", value = "add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_vector_add

// HEADER: tianchenrv.rvv.selected_variant: @rvv_vector_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-binary-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: void tcrv_emitc_rvv_vector_add_from_vector_source_rvv_vector_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// PIPELINE: {key = "rvv_selected_body_operation", value = "add"}
// PIPELINE-SAME: reason = "emission_plan"
// PIPELINE-SAME: target = @rvv_vector_add
