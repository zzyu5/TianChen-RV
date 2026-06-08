// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-materialize-vector-binary-source-front-door | FileCheck %s --check-prefix=MATERIALIZED --implicit-check-not="tcrv_rvv.i32_"
// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-materialize-vector-binary-source-front-door --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: not tcrv-opt %s --split-input-file --tcrv-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPELINE-FAIL --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="rvv_selected_body_operation" --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""
// RUN: tcrv-opt %S/../../Support/RVV/rvv-vector-binary-source-front-door-add.mlir.inc --tcrv-rvv-materialize-vector-binary-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-ADD --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: tcrv-opt %S/../../Support/RVV/rvv-vector-binary-source-front-door-sub.mlir.inc --tcrv-rvv-materialize-vector-binary-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-SUB --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: tcrv-opt %S/../../Support/RVV/rvv-vector-binary-source-front-door-mul.mlir.inc --tcrv-rvv-materialize-vector-binary-source-front-door --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-MUL --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"

module attributes {tcrv_rvv.source_front_door = "bounded_vector_source"} {
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

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_source"} {
  func.func @source_vector_sub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %diff = arith.subi %a, %b : vector<4xi32>
    vector.transfer_write %diff, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_source"} {
  func.func @source_vector_mul(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %product = arith.muli %a, %b : vector<4xi32>
    vector.transfer_write %product, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
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
// MATERIALIZED: %[[ADD_N:.*]] = tcrv_rvv.runtime_abi_value {{.*}}c_name = "n"{{.*}}role = "runtime-element-count"{{.*}} : index
// MATERIALIZED: %[[ADD_VL:.*]] = tcrv_rvv.setvl %[[ADD_N]]
// MATERIALIZED-SAME: lmul = "m1"
// MATERIALIZED-SAME: sew = 32
// MATERIALIZED: tcrv_rvv.with_vl %[[ADD_VL]]
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
// MATERIALIZED: tcrv.exec.case @rvv_vector_add
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-SAME: policy = "rvv-vector-binary-source-front-door-case"
// MATERIALIZED: tcrv.exec.fallback @rvv_vector_add_scalar_fallback
// MATERIALIZED-SAME: origin = "scalar-plugin"

// MATERIALIZED-LABEL: tcrv.exec.kernel @rvv_vector_sub_from_vector_source
// MATERIALIZED: tcrv.exec.variant @rvv_vector_sub
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED: tcrv_rvv.with_vl
// MATERIALIZED-SAME: selected_variant = @rvv_vector_sub
// MATERIALIZED-SAME: source_kernel = "rvv_vector_sub_from_vector_source"
// MATERIALIZED: tcrv_rvv.binary
// MATERIALIZED-SAME: kind = "sub"
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv.exec.variant @rvv_vector_sub_scalar_fallback
// MATERIALIZED-SAME: fallback_role = "conservative"
// MATERIALIZED: tcrv.exec.case @rvv_vector_sub
// MATERIALIZED-SAME: policy = "rvv-vector-binary-source-front-door-case"
// MATERIALIZED: tcrv.exec.fallback @rvv_vector_sub_scalar_fallback

// MATERIALIZED-LABEL: tcrv.exec.kernel @rvv_vector_mul_from_vector_source
// MATERIALIZED: tcrv.exec.variant @rvv_vector_mul
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED: tcrv_rvv.with_vl
// MATERIALIZED-SAME: selected_variant = @rvv_vector_mul
// MATERIALIZED-SAME: source_kernel = "rvv_vector_mul_from_vector_source"
// MATERIALIZED: tcrv_rvv.binary
// MATERIALIZED-SAME: kind = "mul"
// MATERIALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// MATERIALIZED: tcrv.exec.variant @rvv_vector_mul_scalar_fallback
// MATERIALIZED-SAME: fallback_role = "conservative"
// MATERIALIZED: tcrv.exec.case @rvv_vector_mul
// MATERIALIZED-SAME: policy = "rvv-vector-binary-source-front-door-case"
// MATERIALIZED: tcrv.exec.fallback @rvv_vector_mul_scalar_fallback

// PLAN: {key = "rvv_selected_body_operation", value = "add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_vector_add
// PLAN: {key = "rvv_selected_body_operation", value = "sub"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: runtime_abi_name = "rvv-generic-binary-sub-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_vector_sub
// PLAN: {key = "rvv_selected_body_operation", value = "mul"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: runtime_abi_name = "rvv-generic-binary-mul-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_vector_mul

// PIPELINE-FAIL: TianChen-RV execution plan coherence check failed for kernel <missing>
// PIPELINE-FAIL-SAME: requires at least one tcrv.exec.kernel

// HEADER-ADD: tianchenrv.rvv.selected_variant: @rvv_vector_add
// HEADER-ADD: tianchenrv.rvv.runtime_abi_name: rvv-generic-binary-add-callable-c-abi.v1
// HEADER-ADD: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-ADD: void tcrv_emitc_rvv_vector_add_from_vector_source_rvv_vector_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// HEADER-SUB: tianchenrv.rvv.selected_variant: @rvv_vector_sub
// HEADER-SUB: tianchenrv.rvv.runtime_abi_name: rvv-generic-binary-sub-callable-c-abi.v1
// HEADER-SUB: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-SUB: void tcrv_emitc_rvv_vector_sub_from_vector_source_rvv_vector_sub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// HEADER-MUL: tianchenrv.rvv.selected_variant: @rvv_vector_mul
// HEADER-MUL: tianchenrv.rvv.runtime_abi_name: rvv-generic-binary-mul-callable-c-abi.v1
// HEADER-MUL: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER-MUL: void tcrv_emitc_rvv_vector_mul_from_vector_source_rvv_vector_mul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
