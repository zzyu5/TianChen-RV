// RUN: weft-opt %s --split-input-file --weft-rvv-materialize-vector-binary-source-front-door | FileCheck %s --check-prefix=MATERIALIZED --implicit-check-not="weft_rvv.i32_"
// RUN: weft-opt %s --split-input-file --weft-rvv-materialize-vector-binary-source-front-door --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: not weft-opt %s --split-input-file --weft-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPELINE-FAIL --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="rvv_selected_body_operation" --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-binary-source-front-door-add.mlir.inc --weft-rvv-materialize-vector-binary-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-ADD --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-binary-source-front-door-sub.mlir.inc --weft-rvv-materialize-vector-binary-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-SUB --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"
// RUN: weft-opt %S/../../Support/RVV/rvv-vector-binary-source-front-door-mul.mlir.inc --weft-rvv-materialize-vector-binary-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER-MUL --implicit-check-not="rvv-i32m1" --implicit-check-not="descriptor" --implicit-check-not="source-export"

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

// MATERIALIZED-LABEL: weft.exec.kernel @rvv_vector_add_from_vector_source
// MATERIALIZED: weft.exec.capability @rvv
// MATERIALIZED-SAME: id = "rvv"
// MATERIALIZED-SAME: kind = "isa-vector"
// MATERIALIZED: weft.exec.variant @rvv_vector_add
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-DAG: weft_rvv.runtime_abi_value {{.*}}c_name = "lhs"{{.*}}role = "lhs-input-buffer"
// MATERIALIZED-DAG: weft_rvv.runtime_abi_value {{.*}}c_name = "rhs"{{.*}}role = "rhs-input-buffer"
// MATERIALIZED-DAG: weft_rvv.runtime_abi_value {{.*}}c_name = "out"{{.*}}role = "output-buffer"
// MATERIALIZED: %[[ADD_N:.*]] = weft_rvv.runtime_abi_value {{.*}}c_name = "n"{{.*}}role = "runtime-element-count"{{.*}} : index
// MATERIALIZED: %[[ADD_VL:.*]] = weft_rvv.setvl %[[ADD_N]]
// MATERIALIZED-SAME: lmul = "m1"
// MATERIALIZED-SAME: sew = 32
// MATERIALIZED: weft_rvv.with_vl %[[ADD_VL]]
// MATERIALIZED: weft_rvv.load
// MATERIALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// MATERIALIZED: weft_rvv.load
// MATERIALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// MATERIALIZED: weft_rvv.binary
// MATERIALIZED-SAME: kind = "add"
// MATERIALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// MATERIALIZED: weft_rvv.store
// The conservative fallback variant is authored by dispatching to the
// fallback-owning plugin (scalar plugin) via the shared materialization path,
// so it carries that plugin's origin/role/requires/policy verbatim -- the RVV
// front door no longer hand-mints scalar's identity.
// MATERIALIZED: weft.exec.variant @rvv_vector_add_scalar_fallback
// MATERIALIZED-SAME: fallback_role = "conservative"
// MATERIALIZED-SAME: origin = "scalar-plugin"
// MATERIALIZED-SAME: policy = "portable_scalar_fallback_first_slice"
// MATERIALIZED-SAME: requires = [@scalar_fallback]
// MATERIALIZED: weft.exec.case @rvv_vector_add
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED-SAME: policy = "rvv-vector-binary-source-front-door-case"
// MATERIALIZED: weft.exec.fallback @rvv_vector_add_scalar_fallback
// MATERIALIZED-SAME: origin = "scalar-plugin"

// MATERIALIZED-LABEL: weft.exec.kernel @rvv_vector_sub_from_vector_source
// MATERIALIZED: weft.exec.variant @rvv_vector_sub
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED: weft_rvv.with_vl
// MATERIALIZED: weft_rvv.binary
// MATERIALIZED-SAME: kind = "sub"
// MATERIALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// MATERIALIZED: weft.exec.variant @rvv_vector_sub_scalar_fallback
// MATERIALIZED-SAME: fallback_role = "conservative"
// MATERIALIZED: weft.exec.case @rvv_vector_sub
// MATERIALIZED-SAME: policy = "rvv-vector-binary-source-front-door-case"
// MATERIALIZED: weft.exec.fallback @rvv_vector_sub_scalar_fallback

// MATERIALIZED-LABEL: weft.exec.kernel @rvv_vector_mul_from_vector_source
// MATERIALIZED: weft.exec.variant @rvv_vector_mul
// MATERIALIZED-SAME: origin = "rvv-plugin"
// MATERIALIZED: weft_rvv.with_vl
// MATERIALIZED: weft_rvv.binary
// MATERIALIZED-SAME: kind = "mul"
// MATERIALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// MATERIALIZED: weft.exec.variant @rvv_vector_mul_scalar_fallback
// MATERIALIZED-SAME: fallback_role = "conservative"
// MATERIALIZED: weft.exec.case @rvv_vector_mul
// MATERIALIZED-SAME: policy = "rvv-vector-binary-source-front-door-case"
// MATERIALIZED: weft.exec.fallback @rvv_vector_mul_scalar_fallback

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
