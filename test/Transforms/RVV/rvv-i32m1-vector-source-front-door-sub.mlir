// RUN: not tcrv-opt %s --tcrv-rvv-fail-closed-legacy-vector-source-front-door 2>&1 | FileCheck %s --check-prefix=FAIL --implicit-check-not="tcrv.exec.kernel" --implicit-check-not="tcrv_rvv.i32_"
// RUN: not tcrv-opt %s --tcrv-rvv-fail-closed-legacy-vector-source-front-door --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=FAIL --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""

module {
  func.func @vector_source_sub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %remaining = arith.subi %n, %i : index
      %mask = vector.create_mask %remaining : vector<4xi1>
      %a = vector.transfer_read %lhs[%i], %pad, %mask : memref<?xi32>, vector<4xi32>
      %b = vector.transfer_read %rhs[%i], %pad, %mask : memref<?xi32>, vector<4xi32>
      %diff = arith.subi %a, %b : vector<4xi32>
      vector.transfer_write %diff, %out[%i], %mask : vector<4xi32>, memref<?xi32>
    }
    return
  }
}

// FAIL: legacy RVV vector-source front door failed
// FAIL-SAME: RVV Stage1 source-front-door materialization is disabled
// FAIL-SAME: explicit selected generic tcrv_rvv.load/tcrv_rvv.binary/tcrv_rvv.store body
