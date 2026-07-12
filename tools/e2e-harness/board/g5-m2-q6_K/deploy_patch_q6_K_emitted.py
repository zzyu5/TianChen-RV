#!/usr/bin/env python3
# [G5-M2 q6_K] NET-NEW upstream riscv scaffold + EMITTED vl=8 kernel deploy (L-wiring 2).
# q6_K is a K-quant NO-MIN super-block (16 signed int8 per-16 scales + single fp16
# super-block d; 6-bit weight = ql low-nibble | (qh 2-bit high plane)<<4, then -32).
#
# Upstream state (recon, board baseline WinB-q4_0-ON):
#   * ggml HAS an aarch64-only q6_K repack: block_q6_Kx8 + <block_q6_K,4,8>/<8,8> traits
#     + gemv/gemm 8x4/8x8 bodies, but the dispatch routes them ONLY under
#     ggml_cpu_has_neon()&&(matmul_int8||dotprod) -> on the rvv board get_tensor_traits
#     returns nullptr -> stock block-dot.  ARCH riscv has ZERO q6_K.  The <block_q6_K,1,16,
#     GGML_TYPE_Q8_K> riscv-style trait our emitted kernel targets is ABSENT => NET-NEW.
#   * q8_K activation is FULLY present upstream (block_q8_Kx4 stride 1168 +
#     ggml_quantize_mat_t<1/4/8,Q8_K> + from_float=quantize_row_q8_K), shared with q4_K's
#     <block_q4_K,1,16,Q8_K> path => NO activation sub-scaffold needed (unlike q5_1's q8_1).
#
# make_block_q6_Kx16 (stride 3360) byte layout is the GROUND TRUTH the emitted kernel reads
# (from the conversion-fixture op attrs + kquant_repack_verify_q6K.c pack_w):
#     d[16] fp16 @0 | scales[256] i8 @32 | qh[1024] @288 | ql[2048] @1312
#     interleave straight: scales[s*16+c]  qh[i*16+c]  ql[i*16+c]  (col c 0..15)
# Proven bit-exact vs independent oracle in ut_q6_K_verify.cpp (board UT, build-gate).
#
# Edits 3 tracked files. Baseline = WinB-q4_0-ON tree. Reversible. NO git.
import hashlib, sys
GEN  = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/repack.cpp"
HDR  = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/repack.h"
ARCH = "/home/ubuntu/tcrv-llamacpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp"
BASE_GEN  = "deb61a29dd079440ffdc8996b5bd2fa1"
BASE_HDR  = "57851439e7c6f5e35aca7986e148e42b"
BASE_ARCH = "99131cf791e30348b588423b2388e0b8"
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

# piece #1: net-new interleaved q6_K weight block (stride 3360). NO-MIN K-quant, so NO
# dmin; 16 signed int8 sub-block scales; 6-bit weight split into a qh 2-bit plane + ql
# low-nibble plane. Field order d@0 scales@32 qh@288 ql@1312 matches emitted kernel.
# block_q8_Kx4 activation already exists upstream (shared with q4_K) => not redeclared.
h = ins_after(h,
    "using block_q8_0x16 = block<8, 16>;\n",
    "\n// TianChen-RV [G5-M2] net-new interleaved q6_K weight block (NO-MIN K-quant, 6-bit\n"
    "// ql|qh<<4 -32, 16 signed int8 per-16 scales). d@0 scales@32 qh@288 ql@1312 => 3360.\n"
    "struct block_q6_Kx16 { ggml_half d[16]; int8_t scales[256]; uint8_t qh[1024]; uint8_t ql[2048]; };\n"
    "static_assert(sizeof(block_q6_Kx16) == 3360, \"wrong block_q6_Kx16 size/padding\");\n",
    "hdr-struct")

# piece #2a: non-generic decls (arch bodies), after q8_0 gemm decl
h = ins_after(h,
    "void ggml_gemm_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);\n",
    "void ggml_gemv_q6_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n"
    "void ggml_gemm_q6_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n",
    "hdr-decl-arch")

# piece #2b: generic decls, after q8_0 generic gemm decl
h = ins_after(h,
    "void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);\n",
    "void ggml_gemv_q6_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n"
    "void ggml_gemm_q6_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n",
    "hdr-decl-generic")
open(HDR,"w").write(h)

# ============================ GEN (repack.cpp) ============================
g = open(GEN).read()

# pieces #3+#4: make_block_q6_Kx16 + repack_q6_K_to_q6_K_16_bl (static, before q8_0 make)
make_repack = r'''// ============ TianChen-RV [G5-M2] q6_K net-new interleaver (correctness-critical) ============
// make_block_q6_Kx16: block_q6_Kx16 stride 3360. NO union on block_q6_K (plain struct
// {ql[128], qh[64], scales[16], d}). Straight 16-way interleave, no bit transpose, no
// scale-min dance (q6_K scales are direct signed int8). Proven bit-exact vs oracle (UT).
static block_q6_Kx16 make_block_q6_Kx16(block_q6_K * in, unsigned int blck_size_interleave) {
    block_q6_Kx16 out;
    GGML_ASSERT(blck_size_interleave == 1);
    for (int c = 0; c < 16; c++) {
        out.d[c] = in[c].d;                                   // fp16 super-block d @0
        for (int s = 0; s < 16; s++) out.scales[s * 16 + c] = in[c].scales[s]; // i8 scales @32
        for (int i = 0; i < 64;  i++) out.qh[i * 16 + c]    = in[c].qh[i];     // qh plane @288
        for (int i = 0; i < 128; i++) out.ql[i * 16 + c]    = in[c].ql[i];     // ql plane @1312
    }
    return out;
}

static int repack_q6_K_to_q6_K_16_bl(struct ggml_tensor * t, int interleave_block,
                                     const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    constexpr int nrows_interleaved = 16;

    block_q6_Kx16 *    dst = (block_q6_Kx16 *) t->data;
    const block_q6_K * src = (const block_q6_K *) data;
    block_q6_K         dst_tmp[16];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q6_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q6_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

'''
anchor_make = "static block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned int blck_size_interleave) {\n"
assert g.count(anchor_make) == 1, "gen make anchor"
g = g.replace(anchor_make, make_repack + anchor_make, 1)

# piece #7: scalar generic fallbacks (inside extern "C" block), before q2_K gemm generic.
# NO-MIN q6_K decode over the interleaved layout (col j): 6-bit = (ql&0xF|((qh>>sh)&3)<<4)-32,
# per-16 signed int8 scale, single fp16 super-block d * fp32 activation d. Mirrors the
# canonical vec_dot_q6_K_q8_K (kquant_repack_verify_q6K.c ref_isum_block). NOT board-exercised
# at VLEN128 (emitted kernel intercepts); present for correctness + non-128 fallback.
generics = r'''void ggml_gemv_q6_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK_K;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);
    UNUSED(bs);
    UNUSED(nr);

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q6_Kx16 * b_ptr = (const block_q6_Kx16 *) vx + (x * nb);
        float sumf[16];
        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0f;
        for (int l = 0; l < nb; l++) {
            const float a_d = a_ptr[l].d;
            for (int j = 0; j < ncols_interleaved; j++) {
                int64_t isum = 0;
                for (int half = 0; half < 2; half++) {
                    const int ql_base = half * 64, qh_base = half * 32, sb = half * 8;
                    for (int k = 0; k < 32; k++) {
                        const int is = k / 16;
                        const uint8_t qlL = b_ptr[l].ql[(ql_base + k)      * 16 + j];
                        const uint8_t qlH = b_ptr[l].ql[(ql_base + k + 32) * 16 + j];
                        const uint8_t qh  = b_ptr[l].qh[(qh_base + k)      * 16 + j];
                        const int q1 = ((qlL & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
                        const int q2 = ((qlH & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
                        const int q3 = ((qlL >> 4)  | (((qh >> 4) & 3) << 4)) - 32;
                        const int q4 = ((qlH >> 4)  | (((qh >> 6) & 3) << 4)) - 32;
                        const int base = half * 128 + k;
                        isum += (int64_t) b_ptr[l].scales[(sb + is + 0) * 16 + j] * q1 * a_ptr[l].qs[base + 0];
                        isum += (int64_t) b_ptr[l].scales[(sb + is + 2) * 16 + j] * q2 * a_ptr[l].qs[base + 32];
                        isum += (int64_t) b_ptr[l].scales[(sb + is + 4) * 16 + j] * q3 * a_ptr[l].qs[base + 64];
                        isum += (int64_t) b_ptr[l].scales[(sb + is + 6) * 16 + j] * q4 * a_ptr[l].qs[base + 96];
                    }
                }
                sumf[j] += (float) isum * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_d;
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemm_q6_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK_K;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q6_Kx16 * b_ptr = (const block_q6_Kx16 *) vx + (x * nb);
            float sumf[4][16];
            for (int m = 0; m < 4; m++)
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0f;
            for (int l = 0; l < nb; l++) {
                for (int m = 0; m < 4; m++) {
                    const float a_d = a_ptr[l].d[m];
                    for (int j = 0; j < ncols_interleaved; j++) {
                        int64_t isum = 0;
                        for (int half = 0; half < 2; half++) {
                            const int ql_base = half * 64, qh_base = half * 32, sb = half * 8;
                            for (int k = 0; k < 32; k++) {
                                const int is = k / 16;
                                const uint8_t qlL = b_ptr[l].ql[(ql_base + k)      * 16 + j];
                                const uint8_t qlH = b_ptr[l].ql[(ql_base + k + 32) * 16 + j];
                                const uint8_t qh  = b_ptr[l].qh[(qh_base + k)      * 16 + j];
                                const int q1 = ((qlL & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
                                const int q2 = ((qlH & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
                                const int q3 = ((qlL >> 4)  | (((qh >> 4) & 3) << 4)) - 32;
                                const int q4 = ((qlH >> 4)  | (((qh >> 6) & 3) << 4)) - 32;
                                const int base = half * 128 + k;
                                isum += (int64_t) b_ptr[l].scales[(sb + is + 0) * 16 + j] * q1 * a_ptr[l].qs[(base +  0) * 4 + m];
                                isum += (int64_t) b_ptr[l].scales[(sb + is + 2) * 16 + j] * q2 * a_ptr[l].qs[(base + 32) * 4 + m];
                                isum += (int64_t) b_ptr[l].scales[(sb + is + 4) * 16 + j] * q3 * a_ptr[l].qs[(base + 64) * 4 + m];
                                isum += (int64_t) b_ptr[l].scales[(sb + is + 6) * 16 + j] * q4 * a_ptr[l].qs[(base + 96) * 4 + m];
                            }
                        }
                        sumf[m][j] += (float) isum * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_d;
                    }
                }
            }
            for (int m = 0; m < 4; m++)
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
        }
    }
}

'''
anchor_q2kgen = "void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n"
assert g.count(anchor_q2kgen) == 1, "gen q2_K generic anchor"
g = g.replace(anchor_q2kgen, generics + anchor_q2kgen, 1)

# piece #5: repack<block_q6_K,1,16> template
g = ins_after(g,
    "template <> int repack<block_q8_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q8_0_to_q8_0_16_bl(t, 1, data, data_size);\n}\n",
    "\ntemplate <> int repack<block_q6_K, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q6_K_to_q6_K_16_bl(t, 1, data, data_size);\n}\n",
    "gen-repack-tmpl")

# piece #6a: gemv<block_q6_K,1,16,Q8_K>
g = ins_after(g,
    "template <> void gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemv<block_q6_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q6_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemv-tmpl")

# piece #6b: gemm<block_q6_K,1,16,Q8_K>
g = ins_after(g,
    "template <> void gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemm<block_q6_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q6_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemm-tmpl")

# piece #9: trait registration (net-new 1,16 riscv variant; coexists with aarch64 8x4/8x8)
g = ins_after(g,
    "    static const ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;\n",
    "    static const ggml::cpu::repack::tensor_traits<block_q6_K, 1, 16, GGML_TYPE_Q8_K> q6_K_16x1_q8_K; /* TCRV-G5-M2 */\n",
    "gen-trait")

# piece #8: dispatch -- insert a riscv branch INTO the existing q6_K else-if block (its
# current routes are NEON-only, so on rvv it falls through to nullptr => block-dot).
q6_anchor = "    } else if (cur->type == GGML_TYPE_Q6_K) {\n"
q6_riscv = (
    "        /* TCRV-G5-M2: net-new q6_K riscv 1,16 repack route (case128/256=ON, emitted vl=8 carrier) */\n"
    "        if (ggml_cpu_has_riscv_v()) {\n"
    "            #if defined __riscv_zvfh\n"
    "            switch (__riscv_vlenb() * 8) {\n"
    "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q6_K_16x1_q8_K; } break; }\n"
    "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q6_K_16x1_q8_K; } break; }\n"
    "                case 512:  { break; }\n"
    "                case 1024: { break; }\n"
    "                default:   { break; }\n"
    "            }\n"
    "            #endif\n"
    "        }\n")
assert g.count(q6_anchor) == 1, f"gen q6_K dispatch anchor count = {g.count(q6_anchor)}"
g = g.replace(q6_anchor, q6_anchor + q6_riscv, 1)
open(GEN,"w").write(g)

# ============================ ARCH (arch/riscv/repack.cpp) ============================
a = open(ARCH).read()

# piece #12: include emitted .inc (next to q4_0 ones)
a = ins_after(a,
    '#include "tcrv_emitted_repack_gemv.inc"\n',
    '// TianChen-RV [G5-M2] net-new q6_K: compiler-emitted vl=8 GEMM + GEVM (VLEN128).\n'
    '#include "tcrv_emitted_gemm_q6_K.inc"\n'
    '#include "tcrv_emitted_gevm_q6_K.inc"\n',
    "arch-inc")

# pieces #10+#11: net-new q6_K gemv/gemm bodies (before q8_0 gemv body)
q6_arch = r'''void ggml_gemv_q6_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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
    // [TCRV-G5-M2] VLEN128 correctness-carrier: q6_K has NO upstream riscv repack body;
    // our EMITTED vl=8 GEVM IS the kernel (proven bit-exact vs oracle UT). Map (n,s,vx,vy,nc).
    if (__riscv_vlenb() * 8 == 128) {
        static volatile int announced_egevm_q6k = 0; if (!announced_egevm_q6k) { announced_egevm_q6k = 1;
            fprintf(stderr, "TCRV G5-M2 EMITTED GEVM(q6_K_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nc=%d nb=%d\n", n, nc, nb); }
        tcrv_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K(
            (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
        return;
    }
#endif
    ggml_gemv_q6_K_16x1_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q6_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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
    // [TCRV-G5-M2] VLEN128 correctness-carrier: EMITTED vl=8 GEMM (bit-exact vs oracle UT).
    // current-HEAD sig reorder: (nr, bs, n, s, vx, vy, nc).
    if (__riscv_vlenb() * 8 == 128) {
        static volatile int announced_egemm_q6k = 0; if (!announced_egemm_q6k) { announced_egemm_q6k = 1;
            fprintf(stderr, "TCRV G5-M2 EMITTED GEMM(q6_K_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nr=%d nc=%d\n", n, nr, nc); }
        tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(
            (size_t)nr, bs, (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy, (size_t)nc);
        return;
    }
#endif
    ggml_gemm_q6_K_16x1_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

'''
anchor_q8gevm = "void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n"
assert a.count(anchor_q8gevm) == 1, "arch q8 gevm anchor"
a = a.replace(anchor_q8gevm, q6_arch + anchor_q8gevm, 1)
open(ARCH,"w").write(a)

print("PATCH OK (G5-M2 q6_K net-new riscv scaffold + reused q8_K activation + emitted vl=8 deploy)")
print("HDR  md5:", md5(HDR),  " struct_w:", "block_q6_Kx16" in h, " decls:", h.count("ggml_gemv_q6_K_16x1_q8_K"))
print("GEN  md5:", md5(GEN),  " make:", "make_block_q6_Kx16" in g,
      " repack_tmpl:", "repack<block_q6_K, 1, 16>" in g,
      " gemv_tmpl:", "gemv<block_q6_K, 1, 16" in g, " gemm_tmpl:", "gemm<block_q6_K, 1, 16" in g,
      " trait:", "q6_K_16x1_q8_K;" in g, " dispatch_riscv:", "net-new q6_K riscv 1,16" in g,
      " generics:", g.count("q6_K_16x1_q8_K_generic("))
print("ARCH md5:", md5(ARCH), " inc_gemm:", "tcrv_emitted_gemm_q6_K.inc" in a, " inc_gevm:", "tcrv_emitted_gevm_q6_K.inc" in a,
      " gevm_body:", "void ggml_gemv_q6_K_16x1_q8_K(int n" in a, " gemm_body:", "void ggml_gemm_q6_K_16x1_q8_K(int n" in a,
      " gevm_call:", "tcrv_emitc_ggml_repack_gemv_q6_K_q8_K_kernel" in a, " gemm_call:", "tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel" in a)
