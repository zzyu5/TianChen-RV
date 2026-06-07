// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door | FileCheck %s --check-prefix=MATERIALIZED --implicit-check-not="tcrv_rvv.i32_"
// RUN: tcrv-opt %S/../../Support/RVV/rvv-vector-runtime-scalar-cmp-select-source-front-door-sle.mlir.inc --tcrv-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: tcrv-opt %S/../../Support/RVV/rvv-vector-runtime-scalar-cmp-select-source-front-door-sle.mlir.inc --tcrv-source-artifact-front-door-pipeline | FileCheck %s --check-prefix=PIPELINE --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: tcrv-opt %S/../../Support/RVV/rvv-vector-runtime-scalar-cmp-select-source-front-door-sle.mlir.inc --tcrv-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-SLE --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"

module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  func.func @source_vector_runtime_scalar_cmp_select_eq(%lhs: memref<?xi32>, %rhs_scalar: i32, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.splat %rhs_scalar : vector<4xi32>
    %true_vec = vector.transfer_read %true_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %false_vec = vector.transfer_read %false_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi eq, %lhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %mask, %true_vec, %false_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  func.func @source_vector_runtime_scalar_cmp_select_slt(%lhs: memref<?xi32>, %rhs_scalar: i32, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.splat %rhs_scalar : vector<4xi32>
    %true_vec = vector.transfer_read %true_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %false_vec = vector.transfer_read %false_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi slt, %lhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %mask, %true_vec, %false_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  func.func @source_vector_runtime_scalar_cmp_select_sle(%lhs: memref<?xi32>, %rhs_scalar: i32, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.splat %rhs_scalar : vector<4xi32>
    %true_vec = vector.transfer_read %true_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %false_vec = vector.transfer_read %false_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi sle, %lhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %mask, %true_vec, %false_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// MATERIALIZED-LABEL: tcrv.exec.kernel @rvv_vector_runtime_scalar_cmp_select_eq_from_vector_source
// MATERIALIZED: tcrv.exec.capability @rvv
// MATERIALIZED-SAME: id = "rvv"
// MATERIALIZED-SAME: kind = "isa-vector"
// MATERIALIZED: tcrv.exec.variant @rvv_vector_runtime_scalar_cmp_select_eq
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "lhs"{{.*}}purpose = "rvv-vector-runtime-scalar-cmp-select-source-front-door:lhs"{{.*}}role = "lhs-input-buffer"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "rhs_scalar"{{.*}}purpose = "rvv-vector-runtime-scalar-cmp-select-source-front-door:rhs_scalar"{{.*}}role = "rhs-scalar-value"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "true_value"{{.*}}role = "true-value-input-buffer"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "false_value"{{.*}}role = "false-value-input-buffer"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "out"{{.*}}role = "output-buffer"
// MATERIALIZED: %[[EQ_N:.*]] = tcrv_rvv.runtime_abi_value {{.*}}c_name = "n"{{.*}}role = "runtime-element-count"{{.*}} : index
// MATERIALIZED: %[[EQ_VL:.*]] = tcrv_rvv.setvl %[[EQ_N]]
// MATERIALIZED-SAME: lmul = "m1"
// MATERIALIZED-SAME: sew = 32
// MATERIALIZED: tcrv_rvv.with_vl %[[EQ_VL]]
// MATERIALIZED-SAME: required_capabilities = [@rvv]
// MATERIALIZED-SAME: rvv_construction_protocol = "extension-family-construction-protocol.v1"
// MATERIALIZED-SAME: rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family"
// MATERIALIZED-SAME: selected_variant = @rvv_vector_runtime_scalar_cmp_select_eq
// MATERIALIZED-SAME: source_kernel = "rvv_vector_runtime_scalar_cmp_select_eq_from_vector_source"
// MATERIALIZED: tcrv_rvv.load
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv_rvv.splat
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv_rvv.compare
// MATERIALIZED-SAME: kind = "eq"
// MATERIALIZED-SAME: -> !tcrv_rvv.mask<i32, "m1">
// MATERIALIZED: tcrv_rvv.select
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv_rvv.store
// MATERIALIZED: tcrv.exec.variant @rvv_vector_runtime_scalar_cmp_select_eq_scalar_fallback
// MATERIALIZED-SAME: fallback_role = "conservative"
// MATERIALIZED: tcrv.exec.case @rvv_vector_runtime_scalar_cmp_select_eq
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-SAME: policy = "rvv-vector-runtime-scalar-cmp-select-source-front-door-case"
// MATERIALIZED: tcrv.exec.fallback @rvv_vector_runtime_scalar_cmp_select_eq_scalar_fallback
// MATERIALIZED-SAME: origin = "scalar-plugin"

// MATERIALIZED-LABEL: tcrv.exec.kernel @rvv_vector_runtime_scalar_cmp_select_slt_from_vector_source
// MATERIALIZED: tcrv.exec.variant @rvv_vector_runtime_scalar_cmp_select_slt
// MATERIALIZED: tcrv_rvv.compare
// MATERIALIZED-SAME: kind = "slt"
// MATERIALIZED: tcrv.exec.case @rvv_vector_runtime_scalar_cmp_select_slt
// MATERIALIZED-SAME: policy = "rvv-vector-runtime-scalar-cmp-select-source-front-door-case"

// MATERIALIZED-LABEL: tcrv.exec.kernel @rvv_vector_runtime_scalar_cmp_select_sle_from_vector_source
// MATERIALIZED: tcrv.exec.variant @rvv_vector_runtime_scalar_cmp_select_sle
// MATERIALIZED: tcrv_rvv.compare
// MATERIALIZED-SAME: kind = "sle"
// MATERIALIZED: tcrv.exec.case @rvv_vector_runtime_scalar_cmp_select_sle
// MATERIALIZED-SAME: policy = "rvv-vector-runtime-scalar-cmp-select-source-front-door-case"

// PLAN: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_select"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.select"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs_scalar,true_value,false_value,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_select_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-select-plan-validated"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_vector_runtime_scalar_cmp_select_sle

// PIPELINE: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_select"}
// PIPELINE-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PIPELINE-SAME: target = @rvv_vector_runtime_scalar_cmp_select_sle

// HEADER-SLE: tianchenrv.rvv.selected_variant: @rvv_vector_runtime_scalar_cmp_select_sle
// HEADER-SLE: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-select-callable-c-abi.v1
// HEADER-SLE: tianchenrv.rvv.runtime_abi_order: lhs,rhs_scalar,true_value,false_value,out,n
// HEADER-SLE: tianchenrv.rvv.compare_predicate_kind: sle
// HEADER-SLE: tianchenrv.rvv.computed_mask_select_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER-SLE: void tcrv_emitc_rvv_vector_runtime_scalar_cmp_select_sle_from_vector_source_rvv_vector_runtime_scalar_cmp_select_sle(const int32_t *lhs, int32_t rhs_scalar, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);
