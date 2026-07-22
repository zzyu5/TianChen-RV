// RUN: weft-opt %s --split-input-file --weft-rvv-materialize-vector-compare-select-source-front-door | FileCheck %s --check-prefix=MATERIALIZED --implicit-check-not="weft_rvv.i32_"
// RUN: weft-opt %s --split-input-file --weft-rvv-materialize-vector-compare-select-source-front-door --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: not weft-opt %s --split-input-file --weft-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPELINE-FAIL --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="rvv_selected_body_operation" --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-compare-select-source-front-door-eq.mlir.inc --weft-rvv-materialize-vector-compare-select-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-EQ --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-compare-select-source-front-door-sle.mlir.inc --weft-rvv-materialize-vector-compare-select-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-SLE --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"

module attributes {weft_rvv.source_front_door = "bounded_vector_compare_select_source"} {
  func.func @source_vector_cmp_select_eq(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi eq, %a, %b : vector<4xi32>
    %selected = arith.select %mask, %a, %b : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {weft_rvv.source_front_door = "bounded_vector_compare_select_source"} {
  func.func @source_vector_cmp_select_slt(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi slt, %a, %b : vector<4xi32>
    %selected = arith.select %mask, %a, %b : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {weft_rvv.source_front_door = "bounded_vector_compare_select_source"} {
  func.func @source_vector_cmp_select_sle(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi sle, %a, %b : vector<4xi32>
    %selected = arith.select %mask, %a, %b : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// MATERIALIZED-LABEL: weft.exec.kernel @rvv_vector_cmp_select_eq_from_vector_source
// MATERIALIZED: weft.exec.capability @rvv
// MATERIALIZED-SAME: id = "rvv"
// MATERIALIZED-SAME: kind = "isa-vector"
// MATERIALIZED: weft.exec.variant @rvv_vector_cmp_select_eq
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-DAG: weft_rvv.runtime_abi_value {{.*}}c_name = "lhs"{{.*}}role = "lhs-input-buffer"
// MATERIALIZED-DAG: weft_rvv.runtime_abi_value {{.*}}c_name = "rhs"{{.*}}role = "rhs-input-buffer"
// MATERIALIZED-DAG: weft_rvv.runtime_abi_value {{.*}}c_name = "out"{{.*}}role = "output-buffer"
// MATERIALIZED: %[[EQ_N:.*]] = weft_rvv.runtime_abi_value {{.*}}c_name = "n"{{.*}}role = "runtime-element-count"{{.*}} : index
// MATERIALIZED: %[[EQ_VL:.*]] = weft_rvv.setvl %[[EQ_N]]
// MATERIALIZED-SAME: lmul = "m1"
// MATERIALIZED-SAME: sew = 32
// MATERIALIZED: weft_rvv.with_vl %[[EQ_VL]]
// MATERIALIZED-SAME: required_capabilities = [@rvv]
// MATERIALIZED-SAME: rvv_construction_protocol = "extension-family-construction-protocol.v1"
// MATERIALIZED-SAME: rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family"
// MATERIALIZED-SAME: selected_variant = @rvv_vector_cmp_select_eq
// MATERIALIZED-SAME: source_kernel = "rvv_vector_cmp_select_eq_from_vector_source"
// MATERIALIZED: weft_rvv.load
// MATERIALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// MATERIALIZED: weft_rvv.load
// MATERIALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// MATERIALIZED: weft_rvv.compare
// MATERIALIZED-SAME: kind = "eq"
// MATERIALIZED-SAME: -> !weft_rvv.mask<i32, "m1">
// MATERIALIZED: weft_rvv.select
// MATERIALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// MATERIALIZED: weft_rvv.store
// MATERIALIZED: weft.exec.variant @rvv_vector_cmp_select_eq_scalar_fallback
// MATERIALIZED-SAME: fallback_role = "conservative"
// MATERIALIZED: weft.exec.case @rvv_vector_cmp_select_eq
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-SAME: policy = "rvv-vector-compare-select-source-front-door-case"
// MATERIALIZED: weft.exec.fallback @rvv_vector_cmp_select_eq_scalar_fallback
// MATERIALIZED-SAME: origin = "scalar-plugin"

// MATERIALIZED-LABEL: weft.exec.kernel @rvv_vector_cmp_select_slt_from_vector_source
// MATERIALIZED: weft.exec.variant @rvv_vector_cmp_select_slt
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED: weft_rvv.with_vl
// MATERIALIZED-SAME: selected_variant = @rvv_vector_cmp_select_slt
// MATERIALIZED-SAME: source_kernel = "rvv_vector_cmp_select_slt_from_vector_source"
// MATERIALIZED: weft_rvv.compare
// MATERIALIZED-SAME: kind = "slt"
// MATERIALIZED: weft.exec.case @rvv_vector_cmp_select_slt
// MATERIALIZED-SAME: policy = "rvv-vector-compare-select-source-front-door-case"

// MATERIALIZED-LABEL: weft.exec.kernel @rvv_vector_cmp_select_sle_from_vector_source
// MATERIALIZED: weft.exec.variant @rvv_vector_cmp_select_sle
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED: weft_rvv.with_vl
// MATERIALIZED-SAME: selected_variant = @rvv_vector_cmp_select_sle
// MATERIALIZED-SAME: source_kernel = "rvv_vector_cmp_select_sle_from_vector_source"
// MATERIALIZED: weft_rvv.compare
// MATERIALIZED-SAME: kind = "sle"
// MATERIALIZED: weft.exec.case @rvv_vector_cmp_select_sle
// MATERIALIZED-SAME: policy = "rvv-vector-compare-select-source-front-door-case"

// PLAN: weft.exec.diagnostic {{.*}}runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"{{.*}}target = @rvv_vector_cmp_select_eq
// PLAN: weft.exec.diagnostic {{.*}}runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"{{.*}}target = @rvv_vector_cmp_select_slt
// PLAN: weft.exec.diagnostic {{.*}}runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"{{.*}}target = @rvv_vector_cmp_select_sle

// PIPELINE-FAIL: Weft-RV execution plan coherence check failed for kernel <missing>
// PIPELINE-FAIL-SAME: requires at least one weft.exec.kernel

// HEADER-EQ: weft.rvv.selected_variant: @rvv_vector_cmp_select_eq
// HEADER-EQ: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER-EQ: void weft_emitc_rvv_vector_cmp_select_eq_from_vector_source_rvv_vector_cmp_select_eq(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// HEADER-SLE: weft.rvv.selected_variant: @rvv_vector_cmp_select_sle
// HEADER-SLE: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER-SLE: void weft_emitc_rvv_vector_cmp_select_sle_from_vector_source_rvv_vector_cmp_select_sle(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
