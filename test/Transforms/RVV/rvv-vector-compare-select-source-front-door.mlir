// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-materialize-vector-compare-select-source-front-door | FileCheck %s --check-prefix=MATERIALIZED --implicit-check-not="tcrv_rvv.i32_"
// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-materialize-vector-compare-select-source-front-door --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: not tcrv-opt %s --split-input-file --tcrv-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPELINE-FAIL --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="rvv_selected_body_operation" --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""
// RUN: tcrv-opt %S/../../Support/RVV/rvv-vector-compare-select-source-front-door-eq.mlir.inc --tcrv-rvv-materialize-vector-compare-select-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-EQ --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: tcrv-opt %S/../../Support/RVV/rvv-vector-compare-select-source-front-door-sle.mlir.inc --tcrv-rvv-materialize-vector-compare-select-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-SLE --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"

module attributes {tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"} {
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

module attributes {tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"} {
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

module attributes {tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"} {
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

// MATERIALIZED-LABEL: tcrv.exec.kernel @rvv_vector_cmp_select_eq_from_vector_source
// MATERIALIZED: tcrv.exec.capability @rvv
// MATERIALIZED-SAME: id = "rvv"
// MATERIALIZED-SAME: kind = "isa-vector"
// MATERIALIZED: tcrv.exec.variant @rvv_vector_cmp_select_eq
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "lhs"{{.*}}role = "lhs-input-buffer"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "rhs"{{.*}}role = "rhs-input-buffer"
// MATERIALIZED-DAG: tcrv_rvv.runtime_abi_value {{.*}}c_name = "out"{{.*}}role = "output-buffer"
// MATERIALIZED: %[[EQ_N:.*]] = tcrv_rvv.runtime_abi_value {{.*}}c_name = "n"{{.*}}role = "runtime-element-count"{{.*}} : index
// MATERIALIZED: %[[EQ_VL:.*]] = tcrv_rvv.setvl %[[EQ_N]]
// MATERIALIZED-SAME: lmul = "m1"
// MATERIALIZED-SAME: sew = 32
// MATERIALIZED: tcrv_rvv.with_vl %[[EQ_VL]]
// MATERIALIZED-SAME: required_capabilities = [@rvv]
// MATERIALIZED-SAME: rvv_construction_protocol = "extension-family-construction-protocol.v1"
// MATERIALIZED-SAME: rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family"
// MATERIALIZED-SAME: selected_variant = @rvv_vector_cmp_select_eq
// MATERIALIZED-SAME: source_kernel = "rvv_vector_cmp_select_eq_from_vector_source"
// MATERIALIZED: tcrv_rvv.load
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv_rvv.load
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv_rvv.compare
// MATERIALIZED-SAME: kind = "eq"
// MATERIALIZED-SAME: -> !tcrv_rvv.mask<i32, "m1">
// MATERIALIZED: tcrv_rvv.select
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv_rvv.store
// MATERIALIZED: tcrv.exec.variant @rvv_vector_cmp_select_eq_scalar_fallback
// MATERIALIZED-SAME: fallback_role = "conservative"
// MATERIALIZED: tcrv.exec.case @rvv_vector_cmp_select_eq
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-SAME: policy = "rvv-vector-compare-select-source-front-door-case"
// MATERIALIZED: tcrv.exec.fallback @rvv_vector_cmp_select_eq_scalar_fallback
// MATERIALIZED-SAME: origin = "scalar-plugin"

// MATERIALIZED-LABEL: tcrv.exec.kernel @rvv_vector_cmp_select_slt_from_vector_source
// MATERIALIZED: tcrv.exec.variant @rvv_vector_cmp_select_slt
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED: tcrv_rvv.with_vl
// MATERIALIZED-SAME: selected_variant = @rvv_vector_cmp_select_slt
// MATERIALIZED-SAME: source_kernel = "rvv_vector_cmp_select_slt_from_vector_source"
// MATERIALIZED: tcrv_rvv.compare
// MATERIALIZED-SAME: kind = "slt"
// MATERIALIZED: tcrv.exec.case @rvv_vector_cmp_select_slt
// MATERIALIZED-SAME: policy = "rvv-vector-compare-select-source-front-door-case"

// MATERIALIZED-LABEL: tcrv.exec.kernel @rvv_vector_cmp_select_sle_from_vector_source
// MATERIALIZED: tcrv.exec.variant @rvv_vector_cmp_select_sle
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED: tcrv_rvv.with_vl
// MATERIALIZED-SAME: selected_variant = @rvv_vector_cmp_select_sle
// MATERIALIZED-SAME: source_kernel = "rvv_vector_cmp_select_sle_from_vector_source"
// MATERIALIZED: tcrv_rvv.compare
// MATERIALIZED-SAME: kind = "sle"
// MATERIALIZED: tcrv.exec.case @rvv_vector_cmp_select_sle
// MATERIALIZED-SAME: policy = "rvv-vector-compare-select-source-front-door-case"

// PLAN: {key = "rvv_selected_body_operation", value = "cmp_select"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.select"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: runtime_abi_name = "rvv-generic-cmp-select-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_vector_cmp_select_eq
// PLAN: {key = "tcrv_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: target = @rvv_vector_cmp_select_slt
// PLAN: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: target = @rvv_vector_cmp_select_sle

// PIPELINE-FAIL: TianChen-RV execution plan coherence check failed for kernel <missing>
// PIPELINE-FAIL-SAME: requires at least one tcrv.exec.kernel

// HEADER-EQ: tianchenrv.rvv.selected_variant: @rvv_vector_cmp_select_eq
// HEADER-EQ: tianchenrv.rvv.runtime_abi_name: rvv-generic-cmp-select-callable-c-abi.v1
// HEADER-EQ: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-EQ: tianchenrv.rvv.compare_predicate_kind: eq
// HEADER-EQ: tianchenrv.rvv.select_layout: select-lhs-when-mask-else-rhs
// HEADER-EQ: void tcrv_emitc_rvv_vector_cmp_select_eq_from_vector_source_rvv_vector_cmp_select_eq(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// HEADER-SLE: tianchenrv.rvv.selected_variant: @rvv_vector_cmp_select_sle
// HEADER-SLE: tianchenrv.rvv.runtime_abi_name: rvv-generic-cmp-select-callable-c-abi.v1
// HEADER-SLE: tianchenrv.rvv.compare_predicate_kind: sle
// HEADER-SLE: void tcrv_emitc_rvv_vector_cmp_select_sle_from_vector_source_rvv_vector_cmp_select_sle(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
