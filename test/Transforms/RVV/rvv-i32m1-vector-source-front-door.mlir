// RUN: not tcrv-opt %s --tcrv-rvv-materialize-i32m1-vector-source-front-door 2>&1 | FileCheck %s --check-prefix=FAIL --implicit-check-not="tcrv.exec.kernel" --implicit-check-not="tcrv_rvv.i32_"
// RUN: not tcrv-opt %s --tcrv-rvv-materialize-i32m1-vector-source-front-door --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=FAIL --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""
// RUN: not tcrv-opt %s --tcrv-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPE-FAIL --implicit-check-not="rvv-i32m1-add-emitc-route" --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""
// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-rvv-materialize-i32m1-vector-source-front-door 2>&1 | FileCheck %s --check-prefix=NO-BUILTIN

module {
  func.func @vector_source(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %remaining = arith.subi %n, %i : index
      %mask = vector.create_mask %remaining : vector<4xi1>
      %a = vector.transfer_read %lhs[%i], %pad, %mask : memref<?xi32>, vector<4xi32>
      %b = vector.transfer_read %rhs[%i], %pad, %mask : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.transfer_write %sum, %out[%i], %mask : vector<4xi32>, memref<?xi32>
    }
    return
  }
}

// FAIL: bounded RVV i32m1 vector-source front door failed
// FAIL-SAME: RVV Stage1 source-front-door materialization is disabled
// FAIL-SAME: explicit selected generic tcrv_rvv.load/tcrv_rvv.binary/tcrv_rvv.store body

// NO-BUILTIN: Unknown command line argument
// NO-BUILTIN-SAME: tcrv-rvv-materialize-i32m1-vector-source-front-door

// PIPE-FAIL: TianChen-RV execution plan coherence check failed for kernel <missing>
// PIPE-FAIL-SAME: requires at least one tcrv.exec.kernel
