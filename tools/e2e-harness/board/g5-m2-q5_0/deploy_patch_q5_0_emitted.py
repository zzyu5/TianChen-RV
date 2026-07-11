#!/usr/bin/env python3
# [G5-M2 q5_0] NET-NEW upstream scaffold + EMITTED vl=8 kernel deploy (L-接线②).
# Unlike q8_0/q4_K (upstream trait+arch body existed => flip gate + intercept),
# q5_0 has ZERO upstream riscv repack material (block<K,N> cannot express the 5th
# bit qh). This patch BUILDS the full 12-piece scaffold across 3 files:
#   header repack.h : struct block_q5_0x16 {d[16]; qs[256]; qh[64]}=352B + decls
#   GEN  repack.cpp : make_block_q5_0x16 (transposed-qh interleaver, UT bit-exact),
#                     repack fn, repack<> + gemv<> + gemm<> specializations,
#                     scalar generic fallbacks, trait registration, dispatch case128
#   ARCH riscv/repack.cpp : #include emitted .inc + net-new gemv/gemm bodies that
#                     intercept VLEN128 with our EMITTED vl=8 kernels + return.
# make_block_q5_0x16 field order places qh @288 (NOT recipe #1's {d;qh;qs}) to match
# the emitted kernel's ground-truth offsets (352/288/320) -- MIRAGE trap defused,
# proven bit-exact vs independent oracle in ut_q5_0_interleaver.cpp / ut_q5_0_gemm.cpp.
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

# piece #1: struct (field order d,qs,qh => qh@288, stride 352)
h = ins_after(h,
    "using block_q8_0x16 = block<8, 16>;\n",
    "\n// TianChen-RV [G5-M2] net-new interleaved q5_0 weight block (5th-bit qh cannot\n"
    "// use block<K,N>). Field order places qh @288 to match emitted kernel (stride 352).\n"
    "struct block_q5_0x16 { ggml_half d[16]; uint8_t qs[256]; uint8_t qh[64]; };\n"
    "static_assert(sizeof(block_q5_0x16) == 352, \"wrong block_q5_0x16 size/padding\");\n",
    "hdr-struct")

# piece #2a: non-generic decls (arch bodies), after q8_0 gemm decl
h = ins_after(h,
    "void ggml_gemm_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);\n",
    "void ggml_gemv_q5_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n"
    "void ggml_gemm_q5_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n",
    "hdr-decl-arch")

# piece #2b: generic decls, after q8_0 generic gemm decl
h = ins_after(h,
    "void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);\n",
    "void ggml_gemv_q5_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n"
    "void ggml_gemm_q5_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n",
    "hdr-decl-generic")
open(HDR,"w").write(h)

# ============================ GEN (repack.cpp) ============================
g = open(GEN).read()

# pieces #3+#4: make_block_q5_0x16 + repack_q5_0_to_q5_0_16_bl (static, before q8_0 make)
make_repack = r'''// ================= TianChen-RV [G5-M2] q5_0 net-new interleaver (correctness-critical) =================
// make_block_q5_0x16: transpose original per-column qh (uint32, bit e = 5th bit of
// elem e) into qh_lo[k]/qh_hi[k] (u16 @ out.qh+0 / +32) where bit c = 5th bit of
// elem k / k+16 of column c. qs interleave=1. Proven bit-exact vs oracle (UT GREEN).
static block_q5_0x16 make_block_q5_0x16(block_q5_0 * in, unsigned int blck_size_interleave) {
    block_q5_0x16 out;
    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }
    const int end = (QK5_0 / 2) * 16 / blck_size_interleave;   // 256
    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id     = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;
            out.qs[dst_offset] = in[src_id].qs[src_offset];   // no xor for q5_0
        }
    } else {
        GGML_ASSERT(false);
    }
    uint32_t qh_col[16];
    for (int c = 0; c < 16; c++) {
        memcpy(&qh_col[c], in[c].qh, sizeof(uint32_t));
    }
    for (int k = 0; k < 16; k++) {
        uint16_t lo = 0, hi = 0;
        for (int c = 0; c < 16; c++) {
            lo |= (uint16_t)(((qh_col[c] >> k)        & 1u) << c);
            hi |= (uint16_t)(((qh_col[c] >> (k + 16)) & 1u) << c);
        }
        memcpy(out.qh +  0 + k * 2, &lo, 2);
        memcpy(out.qh + 32 + k * 2, &hi, 2);
    }
    return out;
}

static int repack_q5_0_to_q5_0_16_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_0);
    constexpr int nrows_interleaved = 16;

    block_q5_0x16 *     dst = (block_q5_0x16 *) t->data;
    const block_q5_0 * src = (const block_q5_0 *) data;
    block_q5_0         dst_tmp[16];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK5_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q5_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q5_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

'''
g = ins_after(g,
    "static block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned int blck_size_interleave) {\n",
    "", "gen-noop")  # placeholder to keep count check style; real insert below
# real insert: prepend make_repack before q8_0 make
anchor_make = "static block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned int blck_size_interleave) {\n"
assert g.count(anchor_make) == 1, "gen make anchor"
g = g.replace(anchor_make, make_repack + anchor_make, 1)

# piece #7: scalar generic fallbacks (inside extern "C" block), before q2_K gemm generic
generics = r'''void ggml_gemv_q5_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK5_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);
    UNUSED(bs);
    UNUSED(nr);

    float sumf[16];
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q5_0x16 * b_ptr = (const block_q5_0x16 *) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0f;
        for (int l = 0; l < nb; l++) {
            for (int j = 0; j < ncols_interleaved; j++) {
                int sumi = 0;
                for (int k = 0; k < qk / 2; k++) {
                    const uint8_t byte = b_ptr[l].qs[k * ncols_interleaved + j];
                    uint16_t qhl, qhh;
                    memcpy(&qhl, b_ptr[l].qh +  0 + k * 2, 2);
                    memcpy(&qhh, b_ptr[l].qh + 32 + k * 2, 2);
                    const int xh_lo = ((qhl >> j) & 1) << 4;
                    const int xh_hi = ((qhh >> j) & 1) << 4;
                    const int v_lo = ((byte & 0x0F) | xh_lo) - 16;
                    const int v_hi = ((byte >>   4) | xh_hi) - 16;
                    sumi += v_lo * a_ptr[l].qs[k] + v_hi * a_ptr[l].qs[k + 16];
                }
                sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemm_q5_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK5_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][16];
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q5_0x16 * b_ptr = (const block_q5_0x16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++)
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0f;
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < qk / 2; k++) {
                    uint16_t qhl, qhh;
                    memcpy(&qhl, b_ptr[l].qh +  0 + k * 2, 2);
                    memcpy(&qhh, b_ptr[l].qh + 32 + k * 2, 2);
                    for (int j = 0; j < ncols_interleaved; j++) {
                        const uint8_t byte = b_ptr[l].qs[k * ncols_interleaved + j];
                        const int xh_lo = ((qhl >> j) & 1) << 4;
                        const int xh_hi = ((qhh >> j) & 1) << 4;
                        const int v_lo = ((byte & 0x0F) | xh_lo) - 16;
                        const int v_hi = ((byte >>   4) | xh_hi) - 16;
                        for (int m = 0; m < 4; m++) {
                            const int a_lo = a_ptr[l].qs[k * 4 + m];
                            const int a_hi = a_ptr[l].qs[(k + 16) * 4 + m];
                            sumf[m][j] += (v_lo * a_lo + v_hi * a_hi)
                                * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
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
g = ins_after(g,
    "void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n",
    "", "gen-generic-noop")
anchor_q2kgen = "void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n"
assert g.count(anchor_q2kgen) == 1, "gen q2_K generic anchor"
g = g.replace(anchor_q2kgen, generics + anchor_q2kgen, 1)

# piece #5: repack<block_q5_0,1,16> template (namespace ggml::cpu::repack)
g = ins_after(g,
    "template <> int repack<block_q8_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q8_0_to_q8_0_16_bl(t, 1, data, data_size);\n}\n",
    "\ntemplate <> int repack<block_q5_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q5_0_to_q5_0_16_bl(t, 1, data, data_size);\n}\n",
    "gen-repack-tmpl")

# piece #6a: gemv<block_q5_0,1,16,Q8_0>
g = ins_after(g,
    "template <> void gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemv<block_q5_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q5_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemv-tmpl")

# piece #6b: gemm<block_q5_0,1,16,Q8_0>
g = ins_after(g,
    "template <> void gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemm<block_q5_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q5_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemm-tmpl")

# piece #9: trait registration
g = ins_after(g,
    "    static const ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;\n",
    "    static const ggml::cpu::repack::tensor_traits<block_q5_0, 1, 16, GGML_TYPE_Q8_0> q5_0_16x1_q8_0; /* TCRV-G5-M2 */\n",
    "gen-trait")

# piece #8: dispatch case (append else-if after q8_0 block)
q8_block = (
    "        if (ggml_cpu_has_riscv_v()) {\n"
    "            #if defined __riscv_zvfh\n"
    "            switch (__riscv_vlenb() * 8) {\n"
    "                case 128:  { break; } // TODO\n"
    "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; }\n"
    "                case 512:  { break; } // TODO\n"
    "                case 1024: { break; } // TODO\n"
    "                default:   { return nullptr; }\n"
    "            }\n"
    "            #endif\n"
    "        }\n"
    "    }\n")
q5_addon = (
    " else if (cur->type == GGML_TYPE_Q5_0) {\n"
    "        /* TCRV-G5-M2: net-new q5_0 riscv repack route (case128=ON, emitted vl=8 carrier) */\n"
    "        if (ggml_cpu_has_riscv_v()) {\n"
    "            #if defined __riscv_zvfh\n"
    "            switch (__riscv_vlenb() * 8) {\n"
    "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q5_0_16x1_q8_0; } break; }\n"
    "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q5_0_16x1_q8_0; } break; }\n"
    "                case 512:  { break; } // TODO\n"
    "                case 1024: { break; } // TODO\n"
    "                default:   { return nullptr; }\n"
    "            }\n"
    "            #endif\n"
    "        }\n"
    "    }\n")
assert g.count(q8_block) == 1, f"gen dispatch q8_0 block count = {g.count(q8_block)}"
# insert q5 addon right before the final '}' that closes the else-if chain: append after q8 block's closing '    }\n'
g = g.replace(q8_block, q8_block[:-6] + "    }" + q5_addon[q5_addon.index(" else"):], 1)
open(GEN,"w").write(g)

# ============================ ARCH (arch/riscv/repack.cpp) ============================
a = open(ARCH).read()

# piece #12: include emitted .inc (next to q4_0 ones)
a = ins_after(a,
    '#include "tcrv_emitted_repack_gemv.inc"\n',
    '// TianChen-RV [G5-M2] net-new q5_0: compiler-emitted vl=8 GEMM + GEVM (VLEN128).\n'
    '#include "tcrv_emitted_gemm_q5_0.inc"\n'
    '#include "tcrv_emitted_gevm_q5_0.inc"\n',
    "arch-inc")

# pieces #10+#11: net-new q5_0 gemv/gemm bodies (before q8_0 gemv body)
q5_arch = r'''void ggml_gemv_q5_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK5_0;
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
    // [TCRV-G5-M2] VLEN128 correctness-carrier: q5_0 has NO upstream repack body;
    // our EMITTED vl=8 GEVM IS the kernel (proven bit-exact vs oracle UT).
    if (__riscv_vlenb() * 8 == 128) {
        static volatile int announced_egevm_q50 = 0; if (!announced_egevm_q50) { announced_egevm_q50 = 1;
            fprintf(stderr, "TCRV G5-M2 EMITTED GEVM(q5_0_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nc=%d\n", n, nc); }
        tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
            (size_t)n, s, (size_t)nc, (const uint8_t *)vx, (size_t)0, (const uint8_t *)vy, (size_t)0, (int32_t)0);
        return;
    }
#endif
    ggml_gemv_q5_0_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q5_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK5_0;
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
    if (__riscv_vlenb() * 8 == 128) {
        static volatile int announced_egemm_q50 = 0; if (!announced_egemm_q50) { announced_egemm_q50 = 1;
            fprintf(stderr, "TCRV G5-M2 EMITTED GEMM(q5_0_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nr=%d nc=%d\n", n, nr, nc); }
        tcrv_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0(
            (size_t)nr, bs, (size_t)n, s, (size_t)nc, (const uint8_t *)vx, (const uint8_t *)vy);
        return;
    }
#endif
    ggml_gemm_q5_0_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

'''
a = ins_after(a,
    "void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n",
    "", "arch-noop")
anchor_q8gevm = "void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n"
assert a.count(anchor_q8gevm) == 1, "arch q8 gevm anchor"
a = a.replace(anchor_q8gevm, q5_arch + anchor_q8gevm, 1)
open(ARCH,"w").write(a)

print("PATCH OK (G5-M2 q5_0 net-new 12-piece scaffold + emitted vl=8 deploy)")
print("HDR  md5:", md5(HDR),  " struct:", "block_q5_0x16" in h, " decls:", h.count("ggml_gemv_q5_0_16x1_q8_0"))
print("GEN  md5:", md5(GEN),  " make:", "make_block_q5_0x16" in g, " repack_tmpl:", "repack<block_q5_0, 1, 16>" in g,
      " gemv_tmpl:", "gemv<block_q5_0, 1, 16" in g, " gemm_tmpl:", "gemm<block_q5_0, 1, 16" in g,
      " trait:", "q5_0_16x1_q8_0;" in g, " dispatch:", "GGML_TYPE_Q5_0" in g, " generics:", g.count("q5_0_16x1_q8_0_generic("))
print("ARCH md5:", md5(ARCH), " inc_gemm:", "tcrv_emitted_gemm_q5_0.inc" in a, " inc_gevm:", "tcrv_emitted_gevm_q5_0.inc" in a,
      " gevm_body:", "void ggml_gemv_q5_0_16x1_q8_0(int n" in a, " gemm_body:", "void ggml_gemm_q5_0_16x1_q8_0(int n" in a,
      " gevm_call:", "tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel" in a, " gemm_call:", "tcrv_emitc_ggml_gemm_q5_0_q8_0_kernel" in a)
