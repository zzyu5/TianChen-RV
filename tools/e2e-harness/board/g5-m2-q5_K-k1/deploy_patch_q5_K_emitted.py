#!/usr/bin/env python3
# [G5-M2 q5_K @ k1] NET-NEW riscv 1,16 q5_K repack scaffold + deploy OUR compiler-EMITTED
# VLA repack kernels (gemm md5 ba30ba54 / gevm md5 c445b89e) into the k1 ggml A-tree
# (SpacemiT X60, VLEN256, stock clang-18).
#
# WHY net-new (vs q4_K which is a stock as-shipped route on k1): the k1 tree DOES ship a
# q4_K/q2_K 1,16 riscv repack (case256 -> q{4,2}_K_16x1_q8_K), but q5_K's dispatch has ONLY
# NEON sub-branches (q5_K_8x8/8x4 under has_neon&&matmul_int8/dotprod) -> on the riscv board
# get_tensor_traits returns nullptr at EVERY VLEN => stock ggml_vec_dot_q5_K_q8_K block-dot.
# So the <block_q5_K,1,16,GGML_TYPE_Q8_K> riscv trait our emitted kernel targets is ABSENT
# => this is a NET-NEW dispatch (our kernel), NOT a stock-repack toggle (unlike [WORK-ITEM]).
#
# q5_K = q4_K nibble + a qh 5th-bit plane + DUAL d/dmin (SHARES q4_K's kquant_dmin_bsums_min
# fold WHOLE). The block_q5_Kx16 interleaved weight layout (stride 2816) is DERIVED from the
# golden emitted kernel (kquant_repacker.h + RVVLowerQuantContraction.cpp kQ5KDecodeFacts):
#     d[16] @0 | dmin[16] @32 | scales[192] @64 (ggml CUSTOM 6-bit split, byte-identical to
#     q4_K) | qh[512] @256 (5th-bit plane, STRAIGHT 16-way interleave) | qs[2048] @768 (nibble,
#     STRAIGHT 16-way interleave).  make_block_q5_Kx16 == make_block_q4_Kx16 body + qh plane
#     + qs pushed AFTER qh (@768). Proven bit-exact vs oracle (kernel-axis t4a, k1 VLEN256
#     ULP0 int / 8.0e-07 norm).
#
# The emitted kernel is VLA-valid at VLEN>=128 (uses avl=8 f32m2 chunks). On k1 (VLEN256) the
# arch body intercepts and calls it directly. Edits 3 tracked files (repack.h + GEN repack.cpp
# + arch/riscv/repack.cpp). Baseline = k1 shared-source baseline. Reversible. NO git.
import hashlib, sys
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

def ins_after(text, anchor, addition, label):
    c = text.count(anchor)
    assert c == 1, f"[{label}] anchor count = {c} (expected 1)"
    return text.replace(anchor, anchor + addition)

# ============================ HEADER (repack.h) ============================
h = open(HDR).read()

# piece #1: net-new interleaved q5_K weight block (stride 2816). d[16]/dmin[16]/scales[192]
# byte-identical to block_q4_Kx16; ADD qh[512] 5th-bit plane @256; qs[2048] @768.
h = ins_after(h,
    'static_assert(sizeof(block_q4_Kx16) == sizeof(ggml_half) * 32 + K_SCALE_SIZE * 16 + QK_K * 8, "wrong q4_K block size/padding");\n',
    "\n// TianChen-RV [G5-M2] net-new interleaved q5_K weight block (== q4_K header + a qh 5th-bit\n"
    "// plane @256 + qs pushed to @768). d@0 dmin@32 scales@64 qh@256 qs@768 => 2816.\n"
    "struct block_q5_Kx16 {\n"
    "    ggml_half d[16];      // super-block scale for quantized scales\n"
    "    ggml_half dmin[16];   // super-block scale for quantized mins\n"
    "    uint8_t scales[192];  // scales and mins, quantized with 6 bits (byte-identical to q4_K)\n"
    "    uint8_t qh[512];      // 5th-bit high plane (32 qh bytes x 16 cols, straight interleave)\n"
    "    uint8_t qs[2048];     // 4-bit quants\n"
    "};\n"
    "static_assert(sizeof(block_q5_Kx16) == sizeof(ggml_half) * 32 + K_SCALE_SIZE * 16 + (QK_K / 8) * 16 + QK_K * 8, \"wrong q5_K block size/padding\");\n",
    "hdr-struct")

# piece #2: non-generic arch-body decls, after the q4_K 16x1 gemm decl (line ~185)
h = ins_after(h,
    "void ggml_gemm_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);\n",
    "void ggml_gemv_q5_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n"
    "void ggml_gemm_q5_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n",
    "hdr-decl-arch")
open(HDR,"w").write(h)

# ============================ GEN (repack.cpp) ============================
g = open(GEN).read()

# pieces #3+#4: make_block_q5_Kx16 + repack_q5_K_to_q5_K_16_bl (before make_block_q5_Kx8)
make_repack = r'''// ============ TianChen-RV [G5-M2] q5_K net-new 1,16 interleaver (correctness-critical) ============
// make_block_q5_Kx16: block_q5_Kx16 stride 2816. d/dmin/scale-region byte-IDENTICAL to
// make_block_q4_Kx16 (the shared K-quant 6-bit scale split); ADD the qh 5th-bit plane (straight
// 16-way interleave, verbatim byte copy -- preserves all 8 planes, no bit-transpose) and push the
// nibble plane to out.qs (@768). Proven bit-exact vs oracle (kernel-axis t4a, k1 VLEN256).
static block_q5_Kx16 make_block_q5_Kx16(block_q5_K * in, unsigned int blck_size_interleave) {
    block_q5_Kx16 out;
    for (int i = 0; i < 16; i++) {
        out.d[i]    = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }
    for (int i = 0; i < 16; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    GGML_ASSERT(blck_size_interleave == 1);

    // nibble plane: straight 16-way interleave  out.qs[p*16+c] = in[c].qs[p]  (128 qs bytes)
    const int end = QK_K * 8 / blck_size_interleave;   // 2048
    for (int i = 0; i < end; ++i) {
        int src_id = i % 16;
        int src_offset = i / 16;
        out.qs[i] = in[src_id].qs[src_offset];
    }
    // qh 5th-bit plane: straight 16-way interleave  out.qh[m*16+c] = in[c].qh[m]  (32 qh bytes)
    const int endh = (QK_K / 8) * 16 / blck_size_interleave;   // 512
    for (int i = 0; i < endh; ++i) {
        int src_id = i % 16;
        int src_offset = i / 16;
        out.qh[i] = in[src_id].qh[src_offset];
    }

    // scale/min 6-bit region -- byte-IDENTICAL to make_block_q4_Kx16
    uint8_t s[128], m[128];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            s[i * 16 + j] = in[j].scales[i] & 63;
            m[i * 16 + j] = in[j].scales[i + 4] & 63;
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
            m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
        }
    }
    for (int i = 0; i < 128; i++) {
        out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
    }
    for (int i = 0; i < 64; i++) {
        out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
    }

    return out;
}

static int repack_q5_K_to_q5_K_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    constexpr int nrows_interleaved = 16;

    block_q5_Kx16 * dst = (block_q5_Kx16*)t->data;
    const block_q5_K * src = (const block_q5_K*) data;
    block_q5_K dst_tmp[16];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q5_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q5_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

'''
anchor_make = "static block_q5_Kx8 make_block_q5_Kx8(block_q5_K * in, unsigned int blck_size_interleave) {\n"
assert g.count(anchor_make) == 1, "gen make_block_q5_Kx8 anchor"
g = g.replace(anchor_make, make_repack + anchor_make, 1)

# piece #5: repack<block_q5_K,1,16> template (after repack<block_q4_K,1,16>)
g = ins_after(g,
    "template <> int repack<block_q4_K, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q4_K_to_q4_K_16_bl(t, 1, data, data_size);\n}\n",
    "\ntemplate <> int repack<block_q5_K, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q5_K_to_q5_K_16_bl(t, 1, data, data_size);\n}\n",
    "gen-repack-tmpl")

# piece #6a: gemv<block_q5_K,1,16,Q8_K> (after q4_K gemv template)
g = ins_after(g,
    "template <> void gemv<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemv<block_q5_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q5_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemv-tmpl")

# piece #6b: gemm<block_q5_K,1,16,Q8_K> (after q4_K gemm template)
g = ins_after(g,
    "template <> void gemm<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemm<block_q5_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q5_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemm-tmpl")

# piece #7: trait registration (net-new 1,16 riscv variant; coexists with aarch64 8x8/8x4)
g = ins_after(g,
    "    static const ggml::cpu::repack::tensor_traits<block_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;\n",
    "    static const ggml::cpu::repack::tensor_traits<block_q5_K, 1, 16, GGML_TYPE_Q8_K> q5_K_16x1_q8_K; /* TCRV-G5-M2 */\n",
    "gen-trait")

# piece #8: dispatch -- insert a riscv branch INTO the existing q5_K else-if block (its current
# routes are NEON-only, so on riscv it falls through to nullptr => block-dot).  Mirror q4_K's
# riscv switch (case256 = our net-new emitted-kernel route; other VLENs left as TODO).
q5_anchor = ("    } else if (cur->type == GGML_TYPE_Q5_K) {\n"
             "        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {\n"
             "            if (cur->ne[1] % 8 == 0) {\n"
             "                return &q5_K_8x8_q8_K;\n"
             "            }\n"
             "        }\n"
             "        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {\n"
             "            if (cur->ne[1] % 8 == 0) {\n"
             "                return &q5_K_8x4_q8_K;\n"
             "            }\n"
             "        }\n")
q5_riscv = (
    "        /* TCRV-G5-M2: net-new q5_K riscv 1,16 repack route (case256=ON, emitted VLA carrier) */\n"
    "        if (ggml_cpu_has_riscv_v()) {\n"
    "            #if defined __riscv_zvfh\n"
    "            switch (__riscv_vlenb() * 8) {\n"
    "                case 128:  { break; } // TODO\n"
    "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q5_K_16x1_q8_K; } break; }\n"
    "                case 512:  { break; } // TODO\n"
    "                case 1024: { break; } // TODO\n"
    "                default:   { return nullptr; }\n"
    "            }\n"
    "            #endif\n"
    "        }\n")
assert g.count(q5_anchor) == 1, f"gen q5_K dispatch anchor count = {g.count(q5_anchor)}"
g = g.replace(q5_anchor, q5_anchor + q5_riscv, 1)
open(GEN,"w").write(g)

# ============================ ARCH (arch/riscv/repack.cpp) ============================
a = open(ARCH).read()

# piece #9: include emitted .inc (next to q4_0 ones)
a = ins_after(a,
    '#include "tcrv_emitted_repack_gemv.inc"\n',
    '// TianChen-RV [G5-M2] net-new q5_K: compiler-emitted VLA GEMM (md5 ba30ba54) + GEVM\n'
    '// (md5 c445b89e). VLEN256 (k1) carrier. Reversible.\n'
    '#include "tcrv_emitted_gemm_q5_K.inc"\n'
    '#include "tcrv_emitted_gevm_q5_K.inc"\n',
    "arch-inc")

# pieces #10+#11: net-new q5_K gemv/gemm bodies (before q4_K gemv body). Call OUR emitted VLA
# kernel directly (valid at VLEN>=128; k1 is VLEN256). GEMM ABI (n,s,vx,vy,nr,nc,bs); GEVM
# ABI (n,s,vx,vy,nc) -- verified against the exported kernel entry signatures.
q5_arch = r'''void ggml_gemv_q5_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined __riscv_v_intrinsic
    // [TCRV-G5-M2] net-new q5_K: our EMITTED VLA GEVM IS the kernel (proven bit-exact vs oracle,
    // k1 VLEN256 ULP0). GEVM ABI map (n,s,vx,vy,nc).
    static volatile int announced_egevm_q5k = 0; if (!announced_egevm_q5k) { announced_egevm_q5k = 1;
        fprintf(stderr, "TCRV G5-M2 EMITTED GEVM(q5_K_16x1 compiler-emitted VLA ba30ba54-sib) ENGAGED n=%d nc=%d nb=%d vlen=%d\n", n, nc, nb, (int)(__riscv_vlenb()*8)); }
    tcrv_emitc_ggml_repack_gemv_q5_K_q8_K_kernel_ggml_repack_gemv_q5_K_q8_K(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
    return;
#else
    GGML_ABORT("q5_K 1,16 repack requires __riscv_v_intrinsic");
#endif
}

void ggml_gemm_q5_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined __riscv_v_intrinsic
    // [TCRV-G5-M2] net-new q5_K: our EMITTED VLA GEMM (md5 ba30ba54). GEMM ABI map
    // (n,s,vx,vy,nr,nc,bs) -- store row stride = bs (v7).
    static volatile int announced_egemm_q5k = 0; if (!announced_egemm_q5k) { announced_egemm_q5k = 1;
        fprintf(stderr, "TCRV G5-M2 EMITTED GEMM(q5_K_16x1 compiler-emitted VLA ba30ba54) ENGAGED n=%d nr=%d nc=%d vlen=%d\n", n, nr, nc, (int)(__riscv_vlenb()*8)); }
    tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(
        (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nr, (size_t)nc, (size_t)bs);
    return;
#else
    GGML_ABORT("q5_K 1,16 repack requires __riscv_v_intrinsic");
#endif
}

'''
anchor_q4gevm = "void ggml_gemv_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n"
assert a.count(anchor_q4gevm) == 1, "arch q4_K gevm anchor"
a = a.replace(anchor_q4gevm, q5_arch + anchor_q4gevm, 1)
open(ARCH,"w").write(a)

print("PATCH OK (G5-M2 q5_K @ k1 net-new riscv 1,16 scaffold + emitted VLA kernel deploy)")
print("HDR  md5:", md5(HDR),  " struct:", "block_q5_Kx16" in h, " decls:", h.count("ggml_gemv_q5_K_16x1_q8_K"))
print("GEN  md5:", md5(GEN),  " make:", "make_block_q5_Kx16" in g,
      " repack_tmpl:", "repack<block_q5_K, 1, 16>" in g,
      " gemv_tmpl:", "gemv<block_q5_K, 1, 16" in g, " gemm_tmpl:", "gemm<block_q5_K, 1, 16" in g,
      " trait:", "q5_K_16x1_q8_K;" in g, " dispatch_riscv:", "net-new q5_K riscv 1,16" in g)
print("ARCH md5:", md5(ARCH), " inc_gemm:", "tcrv_emitted_gemm_q5_K.inc" in a, " inc_gevm:", "tcrv_emitted_gevm_q5_K.inc" in a,
      " gevm_body:", "void ggml_gemv_q5_K_16x1_q8_K(int n" in a, " gemm_body:", "void ggml_gemm_q5_K_16x1_q8_K(int n" in a,
      " gevm_call:", "tcrv_emitc_ggml_repack_gemv_q5_K_q8_K_kernel" in a, " gemm_call:", "tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel" in a)
