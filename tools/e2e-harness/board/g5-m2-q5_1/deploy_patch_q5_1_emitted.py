#!/usr/bin/env python3
# [G5-M2 q5_1] NET-NEW upstream scaffold + EMITTED vl=8 kernel deploy (L-wiring 2).
# Reuses the q5_0 net-new scaffold recipe (12-piece) with the q5_1 deltas:
#   * q5_1 is UNSIGNED 5-bit (nibble | qh_bit<<4), NO -16 offset-binary bias.
#   * q5_1 carries a per-column MIN (m) => weight block gains an m[16] field @32
#     and the fold adds  + m_w[j]*s_a  (s_a = q8_1 activation running sum).
#   * activation param type = GGML_TYPE_Q8_1 (NOT Q8_0): needs block_q8_1x4 (d[4]@0,
#     s[4]@8, qs[128]@16 = 144B) AND a net-new ggml_quantize_mat_t<1,Q8_1> mat-
#     quantizer (upstream only has Q8_0/Q8_K). The GEVM path uses the STOCK
#     from_float = quantize_row_q8_1 (already registered), no new symbol needed there.
# make_block_q5_1x16 places d[16]@0, m[16]@32, qs[256]@64, qh[64]@320 (stride 384)
# to match the emitted kernel's ground-truth offsets (proven bit-exact vs oracle in
# ut_q5_1_interleaver.cpp / ut_q5_1_gemm.cpp).  Edits 3 tracked files. Baseline =
# WinB-q4_0-ON tree. Reversible. NO git.
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

# piece #1: net-new interleaved q5_1 weight block (d[16]@0, m[16]@32, qs[256]@64,
# qh[64]@320 => stride 384) + block_q8_1x4 activation (d[4]@0, s[4]@8, qs[128]@16 =>
# stride 144). Neither can use block<K,N> (that template has neither m/s nor qh).
h = ins_after(h,
    "using block_q8_0x16 = block<8, 16>;\n",
    "\n// TianChen-RV [G5-M2] net-new interleaved q5_1 weight block (UNSIGNED 5-bit +\n"
    "// per-column min; 5th-bit qh cannot use block<K,N>). Field order: d@0 m@32\n"
    "// nibbles@64 qh@320 to match emitted kernel (stride 384).\n"
    "struct block_q5_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[256]; uint8_t qh[64]; };\n"
    "static_assert(sizeof(block_q5_1x16) == 384, \"wrong block_q5_1x16 size/padding\");\n"
    "// net-new q8_1 x4 activation block (q8_1 needs the running-sum s field, so it also\n"
    "// cannot use block<K,N>). d@0 s@8 qs@16 => stride 144.\n"
    "struct block_q8_1x4 { ggml_half d[4]; ggml_half s[4]; int8_t qs[128]; };\n"
    "static_assert(sizeof(block_q8_1x4) == 144, \"wrong block_q8_1x4 size/padding\");\n",
    "hdr-struct")

# piece #2a: non-generic decls (arch bodies), after q8_0 gemm decl
h = ins_after(h,
    "void ggml_gemm_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);\n",
    "void ggml_gemv_q5_1_16x1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n"
    "void ggml_gemm_q5_1_16x1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n",
    "hdr-decl-arch")

# piece #2b: generic decls, after q8_0 generic gemm decl
h = ins_after(h,
    "void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);\n",
    "void ggml_gemv_q5_1_16x1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n"
    "void ggml_gemm_q5_1_16x1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc); /* TCRV-G5-M2 */\n",
    "hdr-decl-generic")
open(HDR,"w").write(h)

# ============================ GEN (repack.cpp) ============================
g = open(GEN).read()

# piece #Q8_1-MAT: net-new q8_1 activation mat-quantizer specialization (upstream has
# only Q8_0/Q8_K). Inlined body (no extra symbol => keeps 3-file touch-set, no
# arch-fallback.h edit). Byte-matches STOCK quantize_row_q8_1_ref: d=amax/127,
# roundf, s = sum*d (original float d), interleave=1 => qs[e*4+m]. Inserted right
# after the <1,Q8_0> specialization (same #if __riscv_zvfh region).
q81_mat = r'''
template <> void ggml_quantize_mat_t<1, GGML_TYPE_Q8_1>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    // TCRV-G5-M2 net-new q8_1 activation mat-quantizer (block_q8_1x4). Mirrors the
    // q8_0 4x1 packer + the q8_1 running-sum s field, byte-exact vs quantize_row_q8_1.
    assert(nrow == 4);
    UNUSED(nrow);
    const int64_t k = n_per_row;
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;
    block_q8_1x4 * GGML_RESTRICT y = (block_q8_1x4 *) vy;
    const int blck_size_interleave = 1;
    float srcv[4][QK8_1];
    float id[4];
    float drow[4];
    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f;
            for (int j = 0; j < QK8_1; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_1 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }
            const float d = amax / ((1 << 7) - 1);
            drow[row_iter] = d;
            id[row_iter]   = d ? 1.0f / d : 0.0f;
            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }
        int sum[4] = { 0, 0, 0, 0 };
        for (int j = 0; j < QK8_1 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            float x0 = srcv[src_id][src_offset] * id[src_id];
            int8_t q = (int8_t) roundf(x0);
            y[i].qs[j] = q;
            sum[src_id] += q;
        }
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            y[i].s[row_iter] = GGML_CPU_FP32_TO_FP16(sum[row_iter] * drow[row_iter]);
        }
    }
}
'''
g = ins_after(g,
    "template <> void ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {\n"
    "    assert(nrow == 4);\n"
    "    UNUSED(nrow);\n"
    "    ggml_quantize_mat_q8_0_4x1(x, vy, n_per_row);\n"
    "}\n",
    q81_mat, "gen-q81-mat")

# pieces #3+#4: make_block_q5_1x16 + repack_q5_1_to_q5_1_16_bl (static, before q8_0 make)
make_repack = r'''// ============ TianChen-RV [G5-M2] q5_1 net-new interleaver (correctness-critical) ============
// make_block_q5_1x16: d[16]@0, m[16]@32 (per-column delta+min), qs[256]@64 nibbles
// interleave=1, qh[64]@320 transposed (qh_lo[k]@0/qh_hi[k]@32, bit c = 5th bit of
// elem k / k+16 of column c). q5_1 = q5_0's qh transpose (unsigned/no -16) + q4_1's
// min. d/m read via byte offsets (block_q5_1 uses a C++ AGGR union). Proven bit-exact
// vs oracle (UT GREEN).
static block_q5_1x16 make_block_q5_1x16(block_q5_1 * in, unsigned int blck_size_interleave) {
    block_q5_1x16 out;
    for (int i = 0; i < 16; i++) {
        const uint8_t * raw = (const uint8_t *) &in[i];
        memcpy(&out.d[i], raw + 0, sizeof(ggml_half));   // block_q5_1 d @0
        memcpy(&out.m[i], raw + 2, sizeof(ggml_half));   // block_q5_1 m @2
    }
    const int end = (QK5_1 / 2) * 16 / blck_size_interleave;   // 256
    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id     = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;
            out.qs[dst_offset] = in[src_id].qs[src_offset];   // no xor for q5_1
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

static int repack_q5_1_to_q5_1_16_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_1);
    constexpr int nrows_interleaved = 16;

    block_q5_1x16 *     dst = (block_q5_1x16 *) t->data;
    const block_q5_1 * src = (const block_q5_1 *) data;
    block_q5_1         dst_tmp[16];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK5_1;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q5_1));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q5_1x16(dst_tmp, interleave_block);
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
# UNSIGNED 5-bit decode (no -16) + q4_1 min fold. d/s read via byte offsets (block_q8_1
# AGGR union); qs plain array direct.
generics = r'''void ggml_gemv_q5_1_16x1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK5_1;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);
    UNUSED(bs);
    UNUSED(nr);

    float sumf[16];
    const block_q8_1 * a_ptr = (const block_q8_1 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q5_1x16 * b_ptr = (const block_q5_1x16 *) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0f;
        for (int l = 0; l < nb; l++) {
            const uint8_t * araw = (const uint8_t *) &a_ptr[l];
            ggml_half ad_h, as_h;
            memcpy(&ad_h, araw + 0, sizeof(ggml_half));   // q8_1 d @0
            memcpy(&as_h, araw + 2, sizeof(ggml_half));   // q8_1 s @2
            const float a_d = GGML_CPU_FP16_TO_FP32(ad_h);
            const float a_s = GGML_CPU_FP16_TO_FP32(as_h);
            for (int j = 0; j < ncols_interleaved; j++) {
                int sumi = 0;
                for (int k = 0; k < qk / 2; k++) {
                    const uint8_t byte = b_ptr[l].qs[k * ncols_interleaved + j];
                    uint16_t qhl, qhh;
                    memcpy(&qhl, b_ptr[l].qh +  0 + k * 2, 2);
                    memcpy(&qhh, b_ptr[l].qh + 32 + k * 2, 2);
                    const int xh_lo = ((qhl >> j) & 1) << 4;
                    const int xh_hi = ((qhh >> j) & 1) << 4;
                    const int v_lo = ((byte & 0x0F) | xh_lo);   // UNSIGNED, no -16
                    const int v_hi = ((byte >>   4) | xh_hi);
                    sumi += v_lo * a_ptr[l].qs[k] + v_hi * a_ptr[l].qs[k + 16];
                }
                sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_d
                         + GGML_CPU_FP16_TO_FP32(b_ptr[l].m[j]) * a_s;
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemm_q5_1_16x1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK5_1;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][16];
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_1x4 * a_ptr = (const block_q8_1x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q5_1x16 * b_ptr = (const block_q5_1x16 *) vx + (x * nb);
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
                        const int v_lo = ((byte & 0x0F) | xh_lo);   // UNSIGNED
                        const int v_hi = ((byte >>   4) | xh_hi);
                        for (int m = 0; m < 4; m++) {
                            const int a_lo = a_ptr[l].qs[k * 4 + m];
                            const int a_hi = a_ptr[l].qs[(k + 16) * 4 + m];
                            sumf[m][j] += (v_lo * a_lo + v_hi * a_hi)
                                * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
                // q4_1 min term: + m_w[j] * s_a[m], once per block per (row,col)
                for (int m = 0; m < 4; m++)
                    for (int j = 0; j < ncols_interleaved; j++)
                        sumf[m][j] += GGML_CPU_FP16_TO_FP32(b_ptr[l].m[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].s[m]);
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

# piece #5: repack<block_q5_1,1,16> template
g = ins_after(g,
    "template <> int repack<block_q8_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q8_0_to_q8_0_16_bl(t, 1, data, data_size);\n}\n",
    "\ntemplate <> int repack<block_q5_1, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {\n"
    "    return repack_q5_1_to_q5_1_16_bl(t, 1, data, data_size);\n}\n",
    "gen-repack-tmpl")

# piece #6a: gemv<block_q5_1,1,16,Q8_1>
g = ins_after(g,
    "template <> void gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemv<block_q5_1, 1, 16, GGML_TYPE_Q8_1>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemv_q5_1_16x1_q8_1(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemv-tmpl")

# piece #6b: gemm<block_q5_1,1,16,Q8_1>
g = ins_after(g,
    "template <> void gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);\n}\n",
    "\ntemplate <> void gemm<block_q5_1, 1, 16, GGML_TYPE_Q8_1>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {\n"
    "    ggml_gemm_q5_1_16x1_q8_1(n, s, bs, vx, vy, nr, nc);\n}\n",
    "gen-gemm-tmpl")

# piece #9: trait registration
g = ins_after(g,
    "    static const ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;\n",
    "    static const ggml::cpu::repack::tensor_traits<block_q5_1, 1, 16, GGML_TYPE_Q8_1> q5_1_16x1_q8_1; /* TCRV-G5-M2 */\n",
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
    " else if (cur->type == GGML_TYPE_Q5_1) {\n"
    "        /* TCRV-G5-M2: net-new q5_1 riscv repack route (case128=ON, emitted vl=8 carrier) */\n"
    "        if (ggml_cpu_has_riscv_v()) {\n"
    "            #if defined __riscv_zvfh\n"
    "            switch (__riscv_vlenb() * 8) {\n"
    "                case 128:  { if (cur->ne[1] % 16 == 0) { return &q5_1_16x1_q8_1; } break; }\n"
    "                case 256:  { if (cur->ne[1] % 16 == 0) { return &q5_1_16x1_q8_1; } break; }\n"
    "                case 512:  { break; } // TODO\n"
    "                case 1024: { break; } // TODO\n"
    "                default:   { return nullptr; }\n"
    "            }\n"
    "            #endif\n"
    "        }\n"
    "    }\n")
assert g.count(q8_block) == 1, f"gen dispatch q8_0 block count = {g.count(q8_block)}"
g = g.replace(q8_block, q8_block[:-6] + "    }" + q5_addon[q5_addon.index(" else"):], 1)
open(GEN,"w").write(g)

# ============================ ARCH (arch/riscv/repack.cpp) ============================
a = open(ARCH).read()

# piece #12: include emitted .inc (next to q4_0 ones)
a = ins_after(a,
    '#include "tcrv_emitted_repack_gemv.inc"\n',
    '// TianChen-RV [G5-M2] net-new q5_1: compiler-emitted vl=8 GEMM + GEVM (VLEN128).\n'
    '#include "tcrv_emitted_gemm_q5_1.inc"\n'
    '#include "tcrv_emitted_gevm_q5_1.inc"\n',
    "arch-inc")

# pieces #10+#11: net-new q5_1 gemv/gemm bodies (before q8_0 gemv body)
q5_arch = r'''void ggml_gemv_q5_1_16x1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK5_1;
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
    // [TCRV-G5-M2] VLEN128 correctness-carrier: q5_1 has NO upstream repack body;
    // our EMITTED vl=8 GEVM IS the kernel (proven bit-exact vs oracle UT).
    if (__riscv_vlenb() * 8 == 128) {
        static volatile int announced_egevm_q51 = 0; if (!announced_egevm_q51) { announced_egevm_q51 = 1;
            fprintf(stderr, "TCRV G5-M2 EMITTED GEVM(q5_1_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nc=%d\n", n, nc); }
        tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
            (size_t)n, s, (size_t)nc, (const uint8_t *)vx, (size_t)0, (const uint8_t *)vy, (size_t)0, (int32_t)0);
        return;
    }
#endif
    ggml_gemv_q5_1_16x1_q8_1_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q5_1_16x1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK5_1;
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
        static volatile int announced_egemm_q51 = 0; if (!announced_egemm_q51) { announced_egemm_q51 = 1;
            fprintf(stderr, "TCRV G5-M2 EMITTED GEMM(q5_1_16x1 VLEN128 compiler-emitted vl=8) ENGAGED n=%d nr=%d nc=%d\n", n, nr, nc); }
        tcrv_emitc_ggml_gemm_q5_1_q8_1_kernel_ggml_gemm_q5_1_q8_1(
            (size_t)nr, bs, (size_t)n, s, (size_t)nc, (const uint8_t *)vx, (const uint8_t *)vy);
        return;
    }
#endif
    ggml_gemm_q5_1_16x1_q8_1_generic(n, s, bs, vx, vy, nr, nc);
}

'''
anchor_q8gevm = "void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {\n"
assert a.count(anchor_q8gevm) == 1, "arch q8 gevm anchor"
a = a.replace(anchor_q8gevm, q5_arch + anchor_q8gevm, 1)
open(ARCH,"w").write(a)

print("PATCH OK (G5-M2 q5_1 net-new scaffold + q8_1 mat-quant + emitted vl=8 deploy)")
print("HDR  md5:", md5(HDR),  " struct_w:", "block_q5_1x16" in h, " struct_a:", "block_q8_1x4" in h, " decls:", h.count("ggml_gemv_q5_1_16x1_q8_1"))
print("GEN  md5:", md5(GEN),  " q81mat:", "ggml_quantize_mat_t<1, GGML_TYPE_Q8_1>" in g, " make:", "make_block_q5_1x16" in g,
      " repack_tmpl:", "repack<block_q5_1, 1, 16>" in g,
      " gemv_tmpl:", "gemv<block_q5_1, 1, 16" in g, " gemm_tmpl:", "gemm<block_q5_1, 1, 16" in g,
      " trait:", "q5_1_16x1_q8_1;" in g, " dispatch:", "GGML_TYPE_Q5_1" in g, " generics:", g.count("q5_1_16x1_q8_1_generic("))
print("ARCH md5:", md5(ARCH), " inc_gemm:", "tcrv_emitted_gemm_q5_1.inc" in a, " inc_gevm:", "tcrv_emitted_gevm_q5_1.inc" in a,
      " gevm_body:", "void ggml_gemv_q5_1_16x1_q8_1(int n" in a, " gemm_body:", "void ggml_gemm_q5_1_16x1_q8_1(int n" in a,
      " gevm_call:", "tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel" in a, " gemm_call:", "tcrv_emitc_ggml_gemm_q5_1_q8_1_kernel" in a)
