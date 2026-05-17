// REQUIRES: tianchenrv-local-riscv-object-clang
// RUN: rm -f %t.o
// RUN: tcrv-translate --tcrv-export-target-artifact %S/tensorext-lite-target-artifact-header.mlir > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z82tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice"

// This lower-level exporter test intentionally starts from the materialized
// TensorExtLite target fixture in this directory. It does not exercise the
// source artifact front-door workflow.

// OBJECT: Format: elf64-littleriscv
// OBJECT: Arch: riscv64
// OBJECT: Type: Relocatable

// SYMBOL: Name: tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice
