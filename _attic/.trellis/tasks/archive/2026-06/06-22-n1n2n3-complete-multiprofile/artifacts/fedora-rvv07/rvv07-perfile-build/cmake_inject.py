#!/usr/bin/env python3
# Insert the TianChen-RV RVV0.7 per-file-march mitigation block into the riscv64
# branch of ggml/src/ggml-cpu/CMakeLists.txt, right after the base ARCH_FLAGS
# line. Base build stays scalar rv64gc (so ggml's 383-fractional-LMUL quants.c
# #if-outs to scalar fallback); only the 3 vector TUs (our q4_0 repack path +
# the repack-type selection TU + the riscv-v runtime gate) get
# -march=rv64gc_xtheadvector -D__riscv_zvfh so OUR compiler-emitted RVV0.7 q4_0
# kernel is real RVV0.7 vector and gets selected+engaged.
import sys
path = sys.argv[1]
s = open(path).read()
anchor = '            list(APPEND ARCH_FLAGS "-march=${MARCH_STR}" -mabi=lp64d)\n'
assert anchor in s, "anchor not found"

block = anchor + '''
            # === TianChen-RV RVV0.7 (xtheadvector / C920) per-file-march mitigation ===
            # ggml's hand RVV reference (arch/riscv/quants.c) assumes RVV1.0
            # fractional LMUL (383 mf2/mf4 intrinsics) that RVV0.7.1 lacks; under a
            # whole-tree xtheadvector march it will not compile. Those fractional
            # intrinsics are all inside `#if defined(__riscv_v_intrinsic)`, so the
            # base ARCH_FLAGS above MUST stay scalar rv64gc (no vector token): the
            # RVV blocks `#if` out -> scalar fallback -> clean compile. Only the
            # TUs carrying OUR compiler-emitted fraction-free RVV0.7 q4_0 repack
            # path (+ the repack-type selection / riscv-v runtime gate) are
            # overridden to the vector march so the q4_0 decode hot path is real
            # RVV0.7 (genuine th.v* 0.7.1 ops) and gets selected+engaged. xtheadvector
            # provides fp16-vector but does not predefine the ratified `__riscv_zvfh`
            # macro the ggml selection guards key on, so we force-define it (changes
            # visibility, not codegen legality: the fp16-vector ops are proven
            # bit-exact on the C920).
            if (GGML_TCRV_RVV07_PERFILE)
                message(STATUS "TianChen-RV: RVV0.7 per-file-march mitigation ENABLED (base=rv64gc scalar, q4_0 repack path=rv64gc_xtheadvector)")
                set(TCRV_RVV07_FLAGS "-march=rv64gc_xtheadvector -mabi=lp64d -D__riscv_zvfh")
                set_source_files_properties(
                    ggml-cpu/arch/riscv/repack.cpp
                    ggml-cpu/repack.cpp
                    ggml-cpu/ggml-cpu.c
                    PROPERTIES COMPILE_FLAGS "${TCRV_RVV07_FLAGS}")
            endif()
            # === end TianChen-RV mitigation ===
'''
s = s.replace(anchor, block, 1)
open(path, "w").write(s)
print("injected per-file-march block")
