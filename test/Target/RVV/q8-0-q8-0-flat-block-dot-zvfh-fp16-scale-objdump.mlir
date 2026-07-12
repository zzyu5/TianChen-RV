// REGRESSION GUARD for the block-dot family Zvfh object-packager fix. The
// monolithic block-dot / repack families fold per-block fp16 scales; before this
// fix the flat / super-block block-dot object packaged under the baseline
// -march=rv64gcv, which has no half-precision support, so each block's scalar
// fp16 scale read ((float)*(const _Float16 *)) DEGRADED into a __extendhfsf2
// softfloat libcall. That libcall (a) left an undefined symbol that blocked
// format-micro linking, and (b) one-sidedly polluted the perf ratio vs the ggml
// factory baseline (which uses a hardware fcvt.s.h). The block-dot families now
// package under -march=rv64gcv_zvfh (which implies Zfhmin), so the two per-block
// fp16 scale reads become HARDWARE fcvt.s.h and the __extendhfsf2 libcall is
// gone. This objdump-asserts BOTH halves so a regression back to rv64gcv is
// caught (no existing block-dot export test disassembles the object).
//
// q8_0 is a FLAT block-dot with two per-block fp16 scale reads (d_x, d_y). clang
// for a RISC-V RVV relocatable object is required to package the artifact.
// REQUIRES: weft-local-rvv-object-clang

// RUN: rm -f %t.o
// RUN: weft-opt %s --weft-rvv-materialize-q8-0-q8-0-block-dot-source-front-door --weft-materialize-emission-plans | weft-translate --weft-export-target-artifact > %t.o

// The two per-block fp16 scale reads are HARDWARE half->single conversions.
// RUN: llvm-objdump -d %t.o | FileCheck %s --check-prefix=FCVT
// FCVT: fcvt.s.h

// NO __extendhfsf2 softfloat libcall (neither an emitted call/reloc nor an
// undefined symbol reference): the -dr disassembly and the symbol table are both
// clean of it.
// RUN: llvm-objdump -dr %t.o | FileCheck %s --check-prefix=NOSOFT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=NOSOFT
// NOSOFT-NOT: __extendhfsf2

module attributes {weft_rvv.source_front_door = "ggml_q8_0_q8_0_block_dot_source",
                   weft_rvv.source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel"} {
  func.func @source_q8_0_q8_0_block_dot(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}
