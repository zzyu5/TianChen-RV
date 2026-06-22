#!/usr/bin/env python3
# Surgical RVV0.7 mitigation for arch/riscv/repack.cpp:
# - q4_0 16x1 GEMV/GEMM: keep the compiler-EMITTED RVV0.7 dispatch (returns on
#   VLEN=128/C920 -> our fraction-free .inc kernel engages), then a scalar
#   _generic tail fallback. The fractional-LMUL hand bodies (dead on the C920)
#   are removed.
# - every other 16x1 / 8x8 repack function (q4_K/iq4_nl/q8_0/q2_K + q4_0_8x8):
#   replace the fractional-LMUL hand body with a scalar _generic tail call.
# Net: NO fractional-LMUL symbol survives in any compiled body, but the q4_0
# decode hot path stays REAL RVV0.7 vector (our emitted kernel). Honest scalar
# fallback for the non-q4_0 ops, exactly per the task's mitigation.
import re, sys

src = open(sys.argv[1]).read()

def find_body(text, sig_start):
    # sig_start = index of 'void ggml_...'; find the opening '{' then balance.
    i = text.index('{', sig_start)
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i, j  # indices of '{' and matching '}'
        j += 1
    raise RuntimeError("unbalanced")

def replace_func_body(text, fname, new_body):
    m = re.search(r'\nvoid ' + re.escape(fname) + r'\(', text)
    if not m:
        raise RuntimeError("func not found: " + fname)
    open_i, close_i = find_body(text, m.start())
    return text[:open_i+1] + new_body + text[close_i:]

# Common args forwarded to the _generic fallbacks (same ABI).
ARGS = "n, s, bs, vx, vy, nr, nc"

# q4_0 GEMV: emitted dispatch (engages our RVV0.7 kernel) + generic tail.
q40_gemv_body = '''
    // TianChen-RV RVV0.7/C920 mitigation: the COMPILER-EMITTED RVV0.7 q4_0 GEMV
    // kernel (fraction-free i8m1->i16m2->i32m4->f32m4, from tcrv-opt) engages on
    // the VLEN=128 C920; the fractional-LMUL hand fallback is removed (RVV0.7 has
    // no mf2/mf4). Non-VLEN128 falls through to the scalar _generic reference.
#if defined __riscv_v_intrinsic
    if (__riscv_vlenb() * 8 == 128) {
        { static volatile int announced_egemv = 0; if (!announced_egemv) { announced_egemv = 1;
            fprintf(stderr, "TCRV EMITTED GEMV(q4_0_16x1 VLEN128 compiler-emitted RVV0.7) ENGAGED nc=%d nb=%d\\n", nc, n / QK8_0); } }
        tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_ggml_repack_gemv_q4_0_q8_0(
            (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
        return;
    }
#endif
    ggml_gemv_q4_0_16x1_q8_0_generic(''' + ARGS + ''');
'''

# q4_0 GEMM: emitted dispatch + generic tail.
q40_gemm_body = '''
    // TianChen-RV RVV0.7/C920 mitigation: COMPILER-EMITTED RVV0.7 q4_0 GEMM kernel.
#if defined __riscv_v_intrinsic
    if (__riscv_vlenb() * 8 == 128) {
        { static volatile int announced_egemm = 0; if (!announced_egemm) { announced_egemm = 1;
            fprintf(stderr, "TCRV EMITTED GEMM(q4_0_16x1 VLEN128 compiler-emitted RVV0.7) ENGAGED nr=%d nc=%d nb=%d\\n", nr, nc, n / QK8_0); } }
        tcrv_emitc_ggml_repack_gemm_q4_0_q8_0_kernel_ggml_repack_gemm_q4_0_q8_0(
            (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nr, (size_t)nc, bs);
        return;
    }
#endif
    ggml_gemm_q4_0_16x1_q8_0_generic(''' + ARGS + ''');
'''

def generic_body(gen_name):
    return "\n    " + gen_name + "(" + ARGS + ");\n"

# Map: function -> replacement body (scalar _generic, except q4_0 16x1).
repl = {
    "ggml_gemv_q4_0_8x8_q8_0":   generic_body("ggml_gemv_q4_0_8x8_q8_0_generic"),
    "ggml_gemv_q4_0_16x1_q8_0":  q40_gemv_body,
    "ggml_gemv_q4_K_16x1_q8_K":  generic_body("ggml_gemv_q4_K_16x1_q8_K_generic"),
    "ggml_gemv_iq4_nl_16x1_q8_0":generic_body("ggml_gemv_iq4_nl_16x1_q8_0_generic"),
    "ggml_gemv_q8_0_16x1_q8_0":  generic_body("ggml_gemv_q8_0_16x1_q8_0_generic"),
    "ggml_gemv_q2_K_16x1_q8_K":  generic_body("ggml_gemv_q2_K_16x1_q8_K_generic"),
    "ggml_gemm_q4_0_8x8_q8_0":   generic_body("ggml_gemm_q4_0_8x8_q8_0_generic"),
    "ggml_gemm_q4_0_16x1_q8_0":  q40_gemm_body,
    "ggml_gemm_q4_K_16x1_q8_K":  generic_body("ggml_gemm_q4_K_16x1_q8_K_generic"),
    "ggml_gemm_iq4_nl_16x1_q8_0":generic_body("ggml_gemm_iq4_nl_16x1_q8_0_generic"),
    "ggml_gemm_q8_0_16x1_q8_0":  generic_body("ggml_gemm_q8_0_16x1_q8_0_generic"),
    "ggml_gemm_q2_K_16x1_q8_K":  generic_body("ggml_gemm_q2_K_16x1_q8_K_generic"),
}

for fname, body in repl.items():
    src = replace_func_body(src, fname, body)

open(sys.argv[1], "w").write(src)
print("transformed OK")
