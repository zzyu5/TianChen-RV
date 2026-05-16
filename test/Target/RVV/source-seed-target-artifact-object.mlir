// REQUIRES: tianchenrv-local-rvv-object-clang
// RUN: rm -f %t.o
// RUN: tcrv-opt %S/../../Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir --tcrv-source-seed-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable
