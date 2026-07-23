// RUN: weft-opt %s --split-input-file --weft-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door | FileCheck %s --check-prefix=SOURCE --implicit-check-not="weft.exec.variant" --implicit-check-not="weft_rvv."
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-runtime-scalar-cmp-select-source-front-door-sle.mlir.inc --weft-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door --weft-execution-planning-pipeline | FileCheck %s --check-prefix=PLAN --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: not weft-opt %S/../../Support/RVV/rvv-vector-runtime-scalar-cmp-select-source-front-door-sle.mlir.inc --weft-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPELINE-FAIL --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="rvv_selected_body_operation" --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-runtime-scalar-cmp-select-source-front-door-sle.mlir.inc --weft-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door --weft-execution-planning-pipeline | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-SLE --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"

module attributes {weft_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
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

module attributes {weft_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
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

module attributes {weft_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
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

// SOURCE: weft.exec.target @rvv_vector_runtime_scalar_cmp_select_eq_from_vector_source_target_profile
// SOURCE: weft.exec.capability @rvv_vector_runtime_scalar_cmp_select_eq_from_vector_source_rvv_capability
// SOURCE-SAME: id = "rvv"
// SOURCE-SAME: kind = "isa-vector"
// SOURCE-LABEL: weft.exec.kernel @rvv_vector_runtime_scalar_cmp_select_eq_from_vector_source
// SOURCE-SAME: problem = @canonical_problem
// SOURCE: weft.exec.i32_vector_compare_select_problem @canonical_problem
// SOURCE-SAME: predicate = "eq"
// SOURCE-SAME: rhs_form = "runtime-scalar"
// SOURCE-LABEL: weft.exec.kernel @rvv_vector_runtime_scalar_cmp_select_slt_from_vector_source
// SOURCE: weft.exec.i32_vector_compare_select_problem @canonical_problem
// SOURCE-SAME: predicate = "slt"
// SOURCE-SAME: rhs_form = "runtime-scalar"
// SOURCE-LABEL: weft.exec.kernel @rvv_vector_runtime_scalar_cmp_select_sle_from_vector_source
// SOURCE: weft.exec.i32_vector_compare_select_problem @canonical_problem
// SOURCE-SAME: predicate = "sle"
// SOURCE-SAME: rhs_form = "runtime-scalar"

// PLAN: weft.exec.diagnostic {{.*}}emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_vector_runtime_scalar_cmp_select_sle

// PIPELINE-FAIL: Weft-RV execution plan coherence check failed for kernel <missing>
// PIPELINE-FAIL-SAME: requires at least one weft.exec.kernel

// HEADER-SLE: weft.rvv.selected_variant: @rvv_vector_runtime_scalar_cmp_select_sle
// HEADER-SLE: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER-SLE: void weft_emitc_rvv_vector_runtime_scalar_cmp_select_sle_from_vector_source_rvv_vector_runtime_scalar_cmp_select_sle(const int32_t *lhs, int32_t rhs_scalar, const int32_t *true_value, const int32_t *false_value, int32_t *out, size_t n);
