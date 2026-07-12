#!/usr/bin/env python3
# [G5 q2_K @ k1] OUR-emitted VLA q2_K repack (S6-tiled GEMM + plain GEVM) REPLACES the k1
# STOCK hand-tuned RVV q2_K 16x1 repack, for an our-emit-vs-stock-hand-brick e2e A/B.
#
# ★ KEY DIFFERENCE vs q5_K@k1 (which was net-new-vs-block-dot): the k1 tree ALREADY ships a
# competent hand-tuned RVV q2_K 1,16 repack that FIRES at case256 (VLEN256) -> the OFF opponent
# is the STOCK hand-tuned repack, NOT block-dot. This mirrors q4_K@k1 (Win-K1-VLEN: our-emit vs
# 真出货 hand-brick, green 1.085x), NOT q5_K (net-new). So we swap TWO things in the ON variant:
#   (1) make_block_q2_Kx16 (GEN repack.cpp): stock "Sequential-Parallel" scale permutation ->
#       OUR STRAIGHT 16-way interleave (scales[s*16+c]=in[c].scales[s]; qs unchanged, already
#       straight in stock). Our emitted kernel expects the straight-interleave scale layout
#       (proven byte-exact by kquant_repack_verify_q2K: INT_mismatch=0 all shapes).
#   (2) arch ggml_gemm/gemv_q2_K_16x1_q8_K bodies -> call OUR emitted VLA kernel (+ engage banner).
# The struct block_q2_Kx16 (stride 1344, byte-identical layout), the repack template, the
# gemv/gemm<block_q2_K,1,16,Q8_K> templates, the q2_K_16x1_q8_K trait, and the case256 dispatch
# ALL already exist in stock -> UNCHANGED. OFF = stock make + stock kernel (self-consistent).
# ON = our make + our kernel (self-consistent). Both correct; the ONLY A/B diff is q2_K compute.
# Edits 2 tracked files (GEN repack.cpp + arch/riscv/repack.cpp). repack.h UNCHANGED. Reversible. NO git.
import hashlib, sys, re
GEN  = "/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/repack.cpp"
HDR  = "/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/repack.h"
ARCH = "/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/arch/riscv/repack.cpp"
BASE_GEN  = "3cac40aa55aece1f69e3d08d4e7e9ae2"
BASE_HDR  = "57851439e7c6f5e35aca7986e148e42b"
BASE_ARCH = "c3c101fdcc07cf803c4b70551c94343a"
def md5(p): return hashlib.md5(open(p,"rb").read()).hexdigest()

g0,h0,a0 = md5(GEN),md5(HDR),md5(ARCH)
if g0!=BASE_GEN or h0!=BASE_HDR or a0!=BASE_ARCH:
    print(f"*** A-tree NOT at baseline: GEN={g0} HDR={h0} ARCH={a0} -- ABORT"); sys.exit(10)

# ---- brace-balanced body replacer: replace from `sig` line through its matching close brace ----
def replace_fn_body(text, sig, new_block, label):
    idx = text.find(sig)
    assert idx != -1, f"[{label}] signature NOT found"
    assert text.count(sig) == 1, f"[{label}] signature count = {text.count(sig)} (expected 1)"
    # find first '{' at/after the signature
    b = text.find("{", idx)
    assert b != -1, f"[{label}] no opening brace"
    depth = 0; i = b; n = len(text)
    in_line_comment=False; in_block_comment=False; in_str=None
    while i < n:
        c = text[i]; c2 = text[i:i+2]
        if in_line_comment:
            if c == "\n": in_line_comment=False
        elif in_block_comment:
            if c2 == "*/": in_block_comment=False; i+=1
        elif in_str is not None:
            if c == "\\": i+=1
            elif c == in_str: in_str=None
        else:
            if c2 == "//": in_line_comment=True; i+=1
            elif c2 == "/*": in_block_comment=True; i+=1
            elif c == '"' or c == "'": in_str=c
            elif c == "{": depth+=1
            elif c == "}":
                depth-=1
                if depth == 0:
                    end = i+1
                    return text[:idx] + new_block + text[end:]
        i+=1
    raise AssertionError(f"[{label}] unbalanced braces")

# ============================ GEN repack.cpp: make_block_q2_Kx16 ============================
g = open(GEN).read()
make_sig = "static block_q2_Kx16 make_block_q2_Kx16(const block_q2_K * in, unsigned int blck_size_interleave) {"
our_make = r'''static block_q2_Kx16 make_block_q2_Kx16(const block_q2_K * in, unsigned int blck_size_interleave) {
    // [TCRV-G5 q2_K@k1] STRAIGHT 16-way interleave (matches OUR compiler-emitted kernel's decode
    // leaf; proven byte-exact by kquant_repack_verify_q2K INT_mismatch=0). REPLACES the stock
    // "Sequential-Parallel" scale permutation. d@0 dmin@32 scales@64 qs@320, stride 1344.
    block_q2_Kx16 out;
    constexpr int N_COLS = 16;
    GGML_ASSERT(blck_size_interleave == 1);
    // 1. Super-scales d / super-mins dmin (verbatim: byte-identical to stock)
    for (int c = 0; c < N_COLS; c++) {
        out.d[c]    = in[c].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[c] = in[c].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }
    // 2. qs 2-bit weights: straight 16-way interleave  out.qs[i*16+c] = in[c].qs[i]  (64 bytes/col)
    for (int c = 0; c < N_COLS; c++) {
        for (int i = 0; i < 64; i++) {
            out.qs[i * N_COLS + c] = in[c].qs[i];
        }
    }
    // 3. scales: STRAIGHT 16-way interleave  out.scales[s*16+c] = in[c].scales[s]  (16 packed 4-bit
    //    scale/min sub-blocks x 16 cols = 256 bytes).  (stock used even-low/odd-low/... permutation)
    for (int c = 0; c < N_COLS; c++) {
        for (int s = 0; s < 16; s++) {
            out.scales[s * N_COLS + c] = in[c].scales[s];
        }
    }
    return out;
}'''
g = replace_fn_body(g, make_sig, our_make, "make_block_q2_Kx16")
open(GEN,"w").write(g)

# ============================ ARCH arch/riscv/repack.cpp ============================
a = open(ARCH).read()

# (1) include emitted .inc (next to the existing q4_0 tcrv_emitted_* includes)
inc_anchor = '#include "tcrv_emitted_repack_gemv.inc"\n'
assert a.count(inc_anchor) == 1, "arch inc anchor"
a = a.replace(inc_anchor,
    inc_anchor +
    '// TianChen-RV [G5 q2_K@k1] net-new: compiler-emitted VLA q2_K S6-tiled GEMM + plain GEVM\n'
    '// replacing the stock hand-tuned RVV q2_K 16x1 repack. VLEN256 (k1) carrier. Reversible.\n'
    '#include "weft_emitted_gemm_q2_K.inc"\n'
    '#include "weft_emitted_gevm_q2_K.inc"\n', 1)

# (2) replace GEVM body -> our emitted kernel call (GEVM ABI n,s,vx,vy,nc)
gevm_sig = "void ggml_gemv_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {"
gevm_new = r'''void ggml_gemv_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    (void)bs; (void)nr;
    assert(n % QK_K == 0);
    assert(nc % 16 == 0);
    // [TCRV-G5 q2_K@k1] OUR EMITTED VLA GEVM replaces the stock hand-tuned repack (proven
    // bit-exact-integer vs oracle). GEVM ABI map (n,s,vx,vy,nc).
    static volatile int announced_egevm_q2k = 0; if (!announced_egevm_q2k) { announced_egevm_q2k = 1;
        fprintf(stderr, "WEFT G5-q2K EMITTED GEVM(q2_K_16x1 compiler-emitted VLA) ENGAGED n=%d nc=%d vlen=%d\n", n, nc, (int)(__riscv_vlenb()*8)); }
    weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
}'''
a = replace_fn_body(a, gevm_sig, gevm_new, "ggml_gemv_q2_K_16x1_q8_K")

# (3) replace GEMM body -> our emitted kernel call (GEMM ABI n,s,vx,vy,nr,nc,bs, store stride bs)
gemm_sig = "void ggml_gemm_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {"
gemm_new = r'''void ggml_gemm_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % 16 == 0);
    // [TCRV-G5 q2_K@k1] OUR EMITTED VLA S6-tiled GEMM replaces the stock hand-tuned repack.
    // GEMM ABI map (n,s,vx,vy,nr,nc,bs) -- store row stride = bs.
    static volatile int announced_egemm_q2k = 0; if (!announced_egemm_q2k) { announced_egemm_q2k = 1;
        fprintf(stderr, "WEFT G5-q2K EMITTED GEMM(q2_K_16x1 compiler-emitted VLA S6-tiled) ENGAGED n=%d nr=%d nc=%d vlen=%d\n", n, nr, nc, (int)(__riscv_vlenb()*8)); }
    weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nr, (size_t)nc, (size_t)bs);
}'''
a = replace_fn_body(a, gemm_sig, gemm_new, "ggml_gemm_q2_K_16x1_q8_K")
open(ARCH,"w").write(a)

print("PATCH OK (G5 q2_K @ k1: our-emit replaces stock hand-tuned repack)")
print("GEN  md5:", md5(GEN),  " straight_make:", "STRAIGHT 16-way interleave" in g)
print("ARCH md5:", md5(ARCH), " inc_gemm:", "weft_emitted_gemm_q2_K.inc" in a, " inc_gevm:", "weft_emitted_gevm_q2_K.inc" in a,
      " gevm_call:", "weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel" in a, " gemm_call:", "weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel" in a)
print("HDR  md5:", md5(HDR), "(unchanged, expect", BASE_HDR+")")
