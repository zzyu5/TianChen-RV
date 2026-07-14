#!/usr/bin/env python3
# [G7-L3 q5_0@k1] NET-NEW upstream scaffold + EMITTED vl=16 VLEN256 kernel deploy.
# Ports the rvv G5-M2 q5_0 net-new 12-piece scaffold (deploy_patch_q5_0_emitted.py)
# to the k1 tree (/home/bianbu/tcrv-k1-llama) with VLEN256 deltas:
#   * dispatch case256 intercept (k1 stock has ZERO q5_0 repack -> block-dot only).
#   * arch bodies gated on __riscv_vlenb()*8 == 256, calling the VLEN256-native vl=16
#     emitted kernels (weft_emitc_..., AVL=16 seal).
# make_block_q5_0x16 places qh @288 (field order d;qs;qh, stride 352) to match the
# emitted kernel's ground-truth offsets -- transposed-qh interleaver is the MIRAGE
# trap, proven bit-exact vs oracle in ut_q5_0_interleaver_k1.cpp / ut_q5_0_gemm_k1.cpp.
# q5_0 activation = STOCK q8_0 (block_q8_0x4, ggml_quantize_mat_q8_0_4x1) -- NO new
# mat-quant symbol. Edits 3 tracked files. Reversible. NO git.
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
    print(f"*** k1 tree NOT at baseline: GEN={g0} HDR={h0} ARCH={a0} -- ABORT"); sys.exit(10)

def ins_after(text, anchor, addition, label):
    c = text.count(anchor)
    assert c == 1, f"[{label}] anchor count = {c} (expected 1)"
    return text.replace(anchor, anchor + addition)

# ============================ HEADER (repack.h) ============================
h = open(HDR).read()
h = ins_after(h,
    "using block_q8_0x16 = block<8, 16>;\n",
    "\n// TianChen-RV [G7-L3] net-new interleaved q5_0 weight block (5th-bit qh cannot\n"
    "// use block<K,N>). Field order places qh @288 to match emitted kernel (stride 352).\n"
    "struct block_q5_0x16 { ggml_half d[16]; uint8_t qs[256]; uint8_t qh[64]; };\n"
    "static_assert(sizeof(block_q5_0x16) == 352, \"wrong block_q5_0x16 size/padding\");\n",
    "hdr-struct")
h = ins_after(h,
    "void ggml_gemm_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);\n",
    "void ggml_gemv_q5_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G7-L3 */\n"
    "void ggml_gemm_q5_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G7-L3 */\n",
    "hdr-decl-arch")
h = ins_after(h,
    "void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);\n",
    "void ggml_gemv_q5_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G7-L3 */\n"
    "void ggml_gemm_q5_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G7-L3 */\n",
    "hdr-decl-generic")
open(HDR,"w").write(h)

# ============================ GEN (repack.cpp) ============================
g = open(GEN).read()
make_repack = r'''// ================= TianChen-RV [G7-L3] q5_0 net-new interleaver (correctness-critical) =================
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
anchor_make = "static block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned int blck_size_interleave) {\n"
assert g.count(anchor_make) == 1, "gen make anchor"
g = g.replace(anchor_make, make_repack + anchor_make, 1)

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
anchor_q2kgen = "void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n"
assert g.count(anchor_q2kgen) == 1, "gen q2_K generic anchor"
g = g.replace(anchor_q2kgen, generics + anchor_q2kgen, 1)

g = ins_after(g,
    "template <> int repack<block_q8_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q8_0_to_q8_0_16_bl(t, 1, data, data_size);\n}\n",
    "\ntemplate <> int repack<block_q5_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q5_0_to_q5_0_16_bl(t, 1, data, data_size);\n}\n",
    "gen-repack-tmpl")
g = ins_after(g,
    "template <> void gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemv<block_q5_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q5_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemv-tmpl")
g = ins_after(g,
    "template <> void gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemm<block_q5_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q5_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemm-tmpl")
g = ins_after(g,
    "    static const ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;\n",
    "    static const ggml::cpu::repack::tensor_traits<block_q5_0, 1, 16, GGML_TYPE_Q8_0> q5_0_16x1_q8_0; /* TCRV-G7-L3 */\n",
    "gen-trait")

# dispatch: append q5_0 else-if after the q8_0 riscv block
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
    "    }")
q5_addon = (
    " else if (cur->type == GGML_TYPE_Q5_0) {\n"
    "        /* TCRV-G7-L3: net-new q5_0 riscv repack route (VLEN256 emitted vl=16 carrier) */\n"
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
    "    }")
assert g.count(q8_block) == 1, f"gen dispatch q8_0 block count = {g.count(q8_block)}"
g = g.replace(q8_block, q8_block + q5_addon, 1)
open(GEN,"w").write(g)

# ============================ ARCH (arch/riscv/repack.cpp) ============================
a = open(ARCH).read()
a = ins_after(a,
    '#include "tcrv_emitted_repack_gemv.inc"\n',
    '// TianChen-RV [G7-L3] net-new q5_0: compiler-emitted vl=16 GEMM + GEVM (VLEN256).\n'
    '#include "weft_emitted_gemm_q5_0.inc"\n'
    '#include "weft_emitted_gevm_q5_0.inc"\n',
    "arch-inc")

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
    // [TCRV-G7-L3] VLEN256 correctness-carrier: q5_0 has NO stock repack body;
    // our EMITTED vl=16 one-strip GEVM IS the kernel (proven byte-exact vs oracle UT).
    if (__riscv_vlenb() * 8 == 256) {
        static volatile int announced_egevm_q50 = 0; if (!announced_egevm_q50) { announced_egevm_q50 = 1;
            fprintf(stderr, "TCRV G7-L3 EMITTED GEVM(q5_0_16x1 VLEN256 compiler-emitted vl=16) ENGAGED n=%d nc=%d\n", n, nc); }
        weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
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
    // [TCRV-G7-L3] VLEN256 correctness-carrier: EMITTED vl=16 GEMM (byte-exact vs oracle UT).
    if (__riscv_vlenb() * 8 == 256) {
        static volatile int announced_egemm_q50 = 0; if (!announced_egemm_q50) { announced_egemm_q50 = 1;
            fprintf(stderr, "TCRV G7-L3 EMITTED GEMM(q5_0_16x1 VLEN256 compiler-emitted vl=16) ENGAGED n=%d nr=%d nc=%d\n", n, nr, nc); }
        weft_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0(
            (size_t)nr, bs, (size_t)n, s, (size_t)nc, (const uint8_t *)vx, (const uint8_t *)vy);
        return;
    }
#endif
    ggml_gemm_q5_0_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

'''
anchor_q8gevm = "void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n"
assert a.count(anchor_q8gevm) == 1, "arch q8 gevm anchor"
a = a.replace(anchor_q8gevm, q5_arch + anchor_q8gevm, 1)
open(ARCH,"w").write(a)

print("PATCH OK (G7-L3 q5_0@k1 net-new scaffold + emitted vl=16 VLEN256 deploy)")
print("HDR  md5:", md5(HDR),  " struct:", "block_q5_0x16" in h, " decls:", h.count("ggml_gemv_q5_0_16x1_q8_0"))
print("GEN  md5:", md5(GEN),  " make:", "make_block_q5_0x16" in g, " repack_tmpl:", "repack<block_q5_0, 1, 16>" in g,
      " trait:", "q5_0_16x1_q8_0;" in g, " dispatch:", "GGML_TYPE_Q5_0" in g, " generics:", g.count("q5_0_16x1_q8_0_generic("))
print("ARCH md5:", md5(ARCH), " inc_gemm:", "weft_emitted_gemm_q5_0.inc" in a, " inc_gevm:", "weft_emitted_gevm_q5_0.inc" in a,
      " gemm_call:", "weft_emitc_ggml_gemm_q5_0_q8_0_kernel" in a, " gevm_call:", "weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel" in a)
