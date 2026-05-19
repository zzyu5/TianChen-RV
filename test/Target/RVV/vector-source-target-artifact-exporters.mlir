// RUN: not tcrv-opt %s --tcrv-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=PIPE-FAIL --implicit-check-not="rvv-generic-binary-add-emitc-route" --implicit-check-not="artifact_kind = \"riscv-elf-relocatable-object\""
// RUN: not tcrv-opt %s --tcrv-source-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=NO-OBJECT --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"

// Source-only RVV MLIR is no longer a default production artifact front door.
// The explicit RVV materializer pass may still be used as a non-default
// typed-body seed, but the default source-artifact pipeline must fail before
// producing a selected RVV artifact candidate.

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

// PIPE-FAIL: TianChen-RV execution plan coherence check failed for kernel <missing>
// PIPE-FAIL-SAME: requires at least one tcrv.exec.kernel

// NO-OBJECT: TianChen-RV execution plan coherence check failed for kernel <missing>
// NO-OBJECT-SAME: requires at least one tcrv.exec.kernel
