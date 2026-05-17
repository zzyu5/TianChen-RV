// REQUIRES: tianchenrv-local-riscv-object-clang
// RUN: rm -f %t.o
// RUN: tcrv-opt %S/../../Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z82tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice"

// This file carries no standalone input. The positive route intentionally
// starts from the TensorExtLite source-front-door fixture, materializes the
// plugin-owned EmitC module, emits C++ through MLIR EmitC, and packages that
// source into a relocatable object without claiming TensorExtLite runtime
// correctness or performance.

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice
