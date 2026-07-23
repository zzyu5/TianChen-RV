// RUN: weft-opt %s --split-input-file --weft-rvv-materialize-vector-binary-source-front-door | FileCheck %s --check-prefix=SOURCE --implicit-check-not="weft.exec.variant" --implicit-check-not="weft_rvv."
// RUN: weft-opt %s --split-input-file --weft-rvv-materialize-vector-binary-source-front-door --weft-execution-planning-pipeline | FileCheck %s --check-prefix=PLAN --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: not weft-opt %s --split-input-file --weft-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPELINE-FAIL --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="rvv_selected_body_operation" --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-binary-source-front-door-add.mlir.inc --weft-rvv-materialize-vector-binary-source-front-door --weft-execution-planning-pipeline | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-ADD --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-binary-source-front-door-sub.mlir.inc --weft-rvv-materialize-vector-binary-source-front-door --weft-execution-planning-pipeline | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-SUB --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-binary-source-front-door-mul.mlir.inc --weft-rvv-materialize-vector-binary-source-front-door --weft-execution-planning-pipeline | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-MUL --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"

module attributes {weft_rvv.source_front_door = "bounded_vector_source"} {
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

module attributes {weft_rvv.source_front_door = "bounded_vector_source"} {
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

module attributes {weft_rvv.source_front_door = "bounded_vector_source"} {
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

// SOURCE: weft.exec.target @rvv_vector_add_from_vector_source_target_profile
// SOURCE: weft.exec.capability @rvv_vector_add_from_vector_source_rvv_capability
// SOURCE-SAME: id = "rvv"
// SOURCE-SAME: kind = "isa-vector"
// SOURCE-LABEL: weft.exec.kernel @rvv_vector_add_from_vector_source
// SOURCE-SAME: problem = @canonical_problem
// SOURCE: weft.exec.i32_vector_binary_problem @canonical_problem
// SOURCE-SAME: kind = "add"
// SOURCE-LABEL: weft.exec.kernel @rvv_vector_sub_from_vector_source
// SOURCE: weft.exec.i32_vector_binary_problem @canonical_problem
// SOURCE-SAME: kind = "sub"
// SOURCE-LABEL: weft.exec.kernel @rvv_vector_mul_from_vector_source
// SOURCE: weft.exec.i32_vector_binary_problem @canonical_problem
// SOURCE-SAME: kind = "mul"

// PLAN: weft.exec.diagnostic {{.*}}runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"{{.*}}target = @rvv_vector_add
// PLAN: weft.exec.diagnostic {{.*}}runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"{{.*}}target = @rvv_vector_sub
// PLAN: weft.exec.diagnostic {{.*}}runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"{{.*}}target = @rvv_vector_mul

// PIPELINE-FAIL: Weft-RV execution plan coherence check failed for kernel <missing>
// PIPELINE-FAIL-SAME: requires at least one weft.exec.kernel

// HEADER-ADD: weft.rvv.selected_variant: @rvv_vector_add
// HEADER-ADD: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER-ADD: void weft_emitc_rvv_vector_add_from_vector_source_rvv_vector_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// HEADER-SUB: weft.rvv.selected_variant: @rvv_vector_sub
// HEADER-SUB: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER-SUB: void weft_emitc_rvv_vector_sub_from_vector_source_rvv_vector_sub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

// HEADER-MUL: weft.rvv.selected_variant: @rvv_vector_mul
// HEADER-MUL: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER-MUL: void weft_emitc_rvv_vector_mul_from_vector_source_rvv_vector_mul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
