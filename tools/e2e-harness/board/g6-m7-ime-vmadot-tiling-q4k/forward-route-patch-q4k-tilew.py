#!/usr/bin/env python3
# G6-A M7 (vmadot-tiling / array-utilization) patch for vendor spacemit ime.cpp -- q4_K super-block IME bridge.
#
# q4_K is a SUPER-BLOCK format (256-elem super-block = 8 sub-blocks of 32; per-sub-block 6-bit scale
# AND 6-bit min; two-level fold Cf += (d*ad)*Sc - (dmin*ad)*Sm). The vmadot LEAF is format-agnostic
# (byte-identical 0xe210312b), so the M7 WIDE tiling (reuse A across NJW col-tiles) applies at the
# leaf exactly as for q4_0/q8_0. The q4_K-specific work is: the DEREF cache pre-dequantizes B into
# fragment-major int8 (+ per-column scale/min/d/dmin tables), the asum (min-term) partial is
# precomputed once from the packed activation, and the epilogue folds the TWO int32 accumulators
# (Sc scale-term, Sm min-term) 4-wide across the 4 output columns.
#
# BYTE-EXACT by construction for the WIDE tiling: vmadot_mac_kloop_w2 produces sumi[0..15] (col-tile
# nj) and sumi[16..31] (col-tile nj+1) each BIT-IDENTICAL to a separate width-1 vmadot (same A, same
# pre-dequantized B, same kf order); each col-tile then folds its OWN sumi into its OWN Sc/Sm with the
# SAME per-column scales -> onw2 == onjout by construction (the epilogue code is shared; only the
# vmadot SCHEDULE differs). The int32 accumulation order is untouched.
#
# Env gates (additive; default OFF -> byte-identical to the sealed q4_K bridge):
#   TCRV_IME_Q4K_BRIDGE    : route q4_K prefill mul_mat through the tcrv IME kernel
#   TCRV_IME_Q4K_CACHE     : [M1] load-once weight-dequant cache (== DEREF for q4_K)
#   TCRV_IME_Q4K_THREADS   : [M2] column-tile multithread the matmul
#   TCRV_IME_Q4K_DEREF     : [M3-a] pre-dequantized B + scale/min tables
#   TCRV_IME_Q4K_DEREF_EPI : [M3-b] register-accumulated epilogue (requires DEREF)
#   TCRV_IME_Q4K_PARSETUP  : [M4]  parallelize the ith==0 activation quant + asum + lighten allocs
#   TCRV_IME_Q4K_EPIVEC    : [M5]  4-wide (across-columns) vectorized two-level epilogue (bit-exact)
#   TCRV_IME_Q4K_NJOUTER   : [M6]  nj-outer/mi-inner tile order for B-feed locality (bit-exact)
#   TCRV_IME_Q4K_TILEW     : [M7]  WIDE vmadot output tile: reuse A across NJW=2 col-tiles (bit-exact)
#   TCRV_IME_Q4K_MMPROF_NOEPI : [prof] vmadot only (skip fold)                      -- timing-only
#   TCRV_IME_Q4K_MMPROF_L1    : [prof] vmadot from fixed L1 scratch (compute only)  -- timing-only
#   TCRV_IME_Q4K_PROF      : per-stage ns split (dequant/quant/asum/alloc/copyback/matmul)
import sys

F = "ggml/src/ggml-cpu/spacemit/ime.cpp"
s = open(F).read()

inc_anchor = '#include <cstdio>  // for GGML_ASSERT\n'
assert inc_anchor in s, "cstdio include anchor not found"
add_inc = ('#include <cstdlib>       // [TCRV-Q4K] std::getenv/atexit\n'
           '#include <cstring>       // [TCRV-Q4K] memcpy/memset\n'
           '#include <ctime>         // [TCRV-Q4K-M1] clock_gettime prof\n'
           '#include <vector>        // [TCRV-Q4K-M2] shared work vectors\n'
           '#include <unordered_map> // [TCRV-Q4K-M1] weight dequant cache\n')
if "[TCRV-Q4K] std::getenv" not in s:
    s = s.replace(inc_anchor, inc_anchor + add_inc, 1)

cls_anchor = "// Impl By IME1\n"
assert s.count(cls_anchor) == 1, "expected exactly one '// Impl By IME1' anchor"

block = r'''// ==================== [TCRV-IME-Q4K-BRIDGE] super-block routing + M1 dequant-cache + M2 mt + M3 deref + M4 parsetup + M5/M6/M7 ====================
// q4_K two-level fold with pre-dequantized B + WIDE vmadot tiling (format-agnostic leaf). real vmadot 0xe210312b.
extern "C" void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);

namespace tcrv_q4k {
static inline void vmadot_mac_kloop(const int8_t * A, const int8_t * B, long kt, int32_t * frag) {
    __asm__ volatile(
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "vmv.v.i v2, 0\n\t"
        "vmv.v.i v3, 0\n\t"
        "mv t2, %[kt]\n\t"
        "mv t3, %[pa]\n\t"
        "mv t4, %[pb]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t"
        "vle8.v v1, (t4)\n\t"
        "vmadot v2, v0, v1\n\t"
        "addi t3, t3, 32\n\t"
        "addi t4, t4, 32\n\t"
        "addi t2, t2, -1\n\t"
        "bnez t2, 1b\n\t"
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "vse32.v v2, (%[pf])\n\t"
        "addi t5, %[pf], 32\n\t"
        "vse32.v v3, (t5)\n\t"
        :
        : [pa] "r"(A), [pb] "r"(B), [kt] "r"(kt), [pf] "r"(frag)
        : "t0", "t2", "t3", "t4", "t5", "v0", "v1", "v2", "v3", "memory");
}
// [M7] WIDE vmadot tiling NJW=2 (byte-identical to sealed q4_0 wide leaf).
static inline void vmadot_mac_kloop_w2(const int8_t * A, const int8_t * B0, long bstride, long kt, int32_t * frag) {
    const int8_t * B1 = B0 + bstride;
    __asm__ volatile(
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "vmv.v.i v2, 0\n\t"
        "vmv.v.i v3, 0\n\t"
        "vmv.v.i v4, 0\n\t"
        "vmv.v.i v5, 0\n\t"
        "mv t2, %[kt]\n\t"
        "mv t3, %[pa]\n\t"
        "mv t4, %[pb0]\n\t"
        "mv t6, %[pb1]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t"
        "vle8.v v1, (t4)\n\t"
        "vle8.v v6, (t6)\n\t"
        "vmadot v2, v0, v1\n\t"
        "vmadot v4, v0, v6\n\t"
        "addi t3, t3, 32\n\t"
        "addi t4, t4, 32\n\t"
        "addi t6, t6, 32\n\t"
        "addi t2, t2, -1\n\t"
        "bnez t2, 1b\n\t"
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "vse32.v v2, (%[pf])\n\t"
        "addi t5, %[pf], 32\n\t"
        "vse32.v v3, (t5)\n\t"
        "addi t5, %[pf], 64\n\t"
        "vse32.v v4, (t5)\n\t"
        "addi t5, %[pf], 96\n\t"
        "vse32.v v5, (t5)\n\t"
        :
        : [pa] "r"(A), [pb0] "r"(B0), [pb1] "r"(B1), [kt] "r"(kt), [pf] "r"(frag)
        : "t0", "t2", "t3", "t4", "t5", "t6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "memory");
}
static inline unsigned short load_fp16(const uint8_t * p) {
    return (unsigned short) ((unsigned) p[0] | ((unsigned) p[1] << 8));
}
static inline void dequant_fragment(const uint8_t * blk, int b, int kf, int8_t * out8) {
    const uint8_t * qs = blk + 16;
    for (int kl = 0; kl < 8; ++kl) {
        int     pl   = kf * 8 + kl;
        uint8_t byte = qs[(b / 2) * 32 + pl];
        out8[kl]     = (int8_t) ((b & 1) ? (byte >> 4) : (byte & 0x0F));
    }
}
static inline void get_scale_min(int j, const uint8_t * q, uint8_t * sc, uint8_t * m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
// [M3-a] pre-dequantize the immutable q4_K weight ONCE: fragment-major int8 B + per-(col,sb,b) scale
// and min + per-(col,sb) d/dmin. Tile-major so a col-tile's 4 columns are contiguous (vector-friendly).
//   Bdec_all[(nj*kt + gf)*32 + nl*8 + kl], gf = sb*32 + b*4 + kf   (int8, N*K bytes)
//   sc_all/mm_all[((nj*nsb + sb)*8 + b)*4 + nl]                    (uint8, N*nsb*8 bytes)
//   d_all/dmin_all[(nj*nsb + sb)*4 + nl]                           (float, N*nsb)
static void repack_dequant_weight(const uint8_t * wq, int8_t * Bdec_all, uint8_t * sc_all, uint8_t * mm_all,
                                  float * d_all, float * dmin_all, long N, long K) {
    const long nsb = K / 256, kt = K / 8, q4kb = 144;
    for (long n = 0; n < N; ++n) {
        long nj = n / 4, nl = n % 4;
        for (long sb = 0; sb < nsb; ++sb) {
            const uint8_t * blk = wq + (size_t) (n * nsb + sb) * q4kb;
            d_all[(nj * nsb + sb) * 4 + nl]    = ggml_fp16_to_fp32(load_fp16(blk + 0));
            dmin_all[(nj * nsb + sb) * 4 + nl] = ggml_fp16_to_fp32(load_fp16(blk + 2));
            for (int b = 0; b < 8; ++b) {
                uint8_t sc, mm;
                get_scale_min(b, blk + 4, &sc, &mm);
                sc_all[((nj * nsb + sb) * 8 + b) * 4 + nl] = sc;
                mm_all[((nj * nsb + sb) * 8 + b) * 4 + nl] = mm;
                for (int kf = 0; kf < 4; ++kf) {
                    int8_t out8[8];
                    dequant_fragment(blk, b, kf, out8);
                    long     gf  = sb * 32 + b * 4 + kf;
                    int8_t * dst = Bdec_all + (nj * kt + gf) * 32 + nl * 8;
                    for (int kl = 0; kl < 8; ++kl) dst[kl] = out8[kl];
                }
            }
        }
    }
}
// activation: ggml f32 (M x K) -> q8_K (quantize_row_q8_K_ref) -> Apack + dA.
static void quant_pack_act(const float * X, int8_t * Apack, float * dA, long M, long K, uint8_t * scratch) {
    const long nsb = K / 256, q8kb = (long) sizeof(block_q8_K);
    for (long m = 0; m < M; ++m) {
        quantize_row_q8_K_ref(X + m * K, scratch, K);
        for (long sb = 0; sb < nsb; ++sb) {
            const block_q8_K * blk = (const block_q8_K *) (scratch + (size_t) sb * q8kb);
            dA[m * nsb + sb]       = blk->d;
            const int8_t * qs      = blk->qs;
            for (long kk = 0; kk < 256; ++kk) {
                long k = sb * 256 + kk, mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
                Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] = qs[kk];
            }
        }
    }
}
static void quant_pack_act_range(const float * X, int8_t * Apack, float * dA, long M, long K,
                                 uint8_t * scratch, long m_start, long m_end) {
    const long nsb = K / 256, q8kb = (long) sizeof(block_q8_K);
    (void) M;
    for (long m = m_start; m < m_end; ++m) {
        quantize_row_q8_K_ref(X + m * K, scratch, K);
        for (long sb = 0; sb < nsb; ++sb) {
            const block_q8_K * blk = (const block_q8_K *) (scratch + (size_t) sb * q8kb);
            dA[m * nsb + sb]       = blk->d;
            const int8_t * qs      = blk->qs;
            for (long kk = 0; kk < 256; ++kk) {
                long k = sb * 256 + kk, mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
                Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] = qs[kk];
            }
        }
    }
}
// asum[m, sb, b] = sum over the 32 activation quants of sub-block (sb,b) for row m (min-term partial).
// Depends only on the packed activation; precomputed once. asum_all[m*(nsb*8) + sb*8 + b].
static void compute_asum(const int8_t * Apack, int32_t * asum_all, long M, long K,
                         long m_start, long m_end) {
    const long nsb = K / 256;
    for (long m = m_start; m < m_end; ++m) {
        long mi = m / 4, ml = m % 4;
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long sb = 0; sb < nsb; ++sb)
            for (int b = 0; b < 8; ++b) {
                long    gf0 = sb * 32 + b * 4;
                int32_t sacc = 0;
                for (int kf = 0; kf < 4; ++kf) {
                    const int8_t * Aframe = Arow + (gf0 + kf) * 32;
                    for (int kl = 0; kl < 8; ++kl) sacc += (int32_t) Aframe[ml * 8 + kl];
                }
                asum_all[m * (nsb * 8) + sb * 8 + b] = sacc;
            }
    }
}
// [M6] width-1 nj-outer/mi-inner vectorized two-level epilogue (onjout). Each output column
// accumulates over sub-blocks in the SAME order; the two int32 accumulators Sc (scale) and Sm (min)
// are folded 4-wide across the 4 columns of a row. Cf[m,n] = sum_sb (d*ad)*Sc - (dmin*ad)*Sm.
static void matmul_deref_epi_vec_njouter(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                         const uint8_t * sc_all, const uint8_t * mm_all,
                                         const float * d_all, const float * dmin_all, const int32_t * asum_all,
                                         float * Cf, long M, long N, long K, long nj_start, long nj_end) {
    const long mt = M / 4, nsb = K / 256, kt = K / 8, fpb = 4;
    const size_t vl = __riscv_vsetvl_e32m1(4);
    for (long nj = nj_start; nj < nj_end; ++nj) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            vfloat32m1_t cf0 = __riscv_vfmv_v_f_f32m1(0.0f, vl), cf1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t cf2 = __riscv_vfmv_v_f_f32m1(0.0f, vl), cf3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (long sb = 0; sb < nsb; ++sb) {
                vint32m1_t Sc0 = __riscv_vmv_v_x_i32m1(0, vl), Sc1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Sc2 = __riscv_vmv_v_x_i32m1(0, vl), Sc3 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Sm0 = __riscv_vmv_v_x_i32m1(0, vl), Sm1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Sm2 = __riscv_vmv_v_x_i32m1(0, vl), Sm3 = __riscv_vmv_v_x_i32m1(0, vl);
                for (int b = 0; b < 8; ++b) {
                    long           gf0 = sb * 32 + b * 4;
                    int32_t        sumi[16];
                    vmadot_mac_kloop(Arow + gf0 * 32, Bcol + gf0 * 32, fpb, sumi);
                    const uint8_t * scp = sc_all + ((long) (nj * nsb + sb) * 8 + b) * 4;
                    const uint8_t * mmp = mm_all + ((long) (nj * nsb + sb) * 8 + b) * 4;
                    int32_t scbuf[4] = { scp[0], scp[1], scp[2], scp[3] };
                    int32_t mmbuf[4] = { mmp[0], mmp[1], mmp[2], mmp[3] };
                    vint32m1_t scb = __riscv_vle32_v_i32m1(scbuf, vl);
                    vint32m1_t mmb = __riscv_vle32_v_i32m1(mmbuf, vl);
                    vint32m1_t su0 = __riscv_vle32_v_i32m1(sumi + 0,  vl);
                    vint32m1_t su1 = __riscv_vle32_v_i32m1(sumi + 4,  vl);
                    vint32m1_t su2 = __riscv_vle32_v_i32m1(sumi + 8,  vl);
                    vint32m1_t su3 = __riscv_vle32_v_i32m1(sumi + 12, vl);
                    Sc0 = __riscv_vadd_vv_i32m1(Sc0, __riscv_vmul_vv_i32m1(scb, su0, vl), vl);
                    Sc1 = __riscv_vadd_vv_i32m1(Sc1, __riscv_vmul_vv_i32m1(scb, su1, vl), vl);
                    Sc2 = __riscv_vadd_vv_i32m1(Sc2, __riscv_vmul_vv_i32m1(scb, su2, vl), vl);
                    Sc3 = __riscv_vadd_vv_i32m1(Sc3, __riscv_vmul_vv_i32m1(scb, su3, vl), vl);
                    int32_t as0 = asum_all[(mi * 4 + 0) * (nsb * 8) + sb * 8 + b];
                    int32_t as1 = asum_all[(mi * 4 + 1) * (nsb * 8) + sb * 8 + b];
                    int32_t as2 = asum_all[(mi * 4 + 2) * (nsb * 8) + sb * 8 + b];
                    int32_t as3 = asum_all[(mi * 4 + 3) * (nsb * 8) + sb * 8 + b];
                    Sm0 = __riscv_vadd_vv_i32m1(Sm0, __riscv_vmul_vx_i32m1(mmb, as0, vl), vl);
                    Sm1 = __riscv_vadd_vv_i32m1(Sm1, __riscv_vmul_vx_i32m1(mmb, as1, vl), vl);
                    Sm2 = __riscv_vadd_vv_i32m1(Sm2, __riscv_vmul_vx_i32m1(mmb, as2, vl), vl);
                    Sm3 = __riscv_vadd_vv_i32m1(Sm3, __riscv_vmul_vx_i32m1(mmb, as3, vl), vl);
                }
                vfloat32m1_t dvec  = __riscv_vle32_v_f32m1(d_all + (long) (nj * nsb + sb) * 4, vl);
                vfloat32m1_t dmvec = __riscv_vle32_v_f32m1(dmin_all + (long) (nj * nsb + sb) * 4, vl);
                float ad0 = dA[(mi * 4 + 0) * nsb + sb], ad1 = dA[(mi * 4 + 1) * nsb + sb];
                float ad2 = dA[(mi * 4 + 2) * nsb + sb], ad3 = dA[(mi * 4 + 3) * nsb + sb];
                #define Q4KFOLD(cf, ad, Sc, Sm) do { \
                    vfloat32m1_t p_ = __riscv_vfmul_vf_f32m1(dvec, (ad), vl); \
                    vfloat32m1_t q_ = __riscv_vfmul_vf_f32m1(dmvec, (ad), vl); \
                    vfloat32m1_t sf_ = __riscv_vfcvt_f_x_v_f32m1((Sc), vl); \
                    vfloat32m1_t mf_ = __riscv_vfcvt_f_x_v_f32m1((Sm), vl); \
                    vfloat32m1_t t1_ = __riscv_vfmul_vv_f32m1(p_, sf_, vl); \
                    vfloat32m1_t t2_ = __riscv_vfmul_vv_f32m1(q_, mf_, vl); \
                    (cf) = __riscv_vfadd_vv_f32m1((cf), __riscv_vfsub_vv_f32m1(t1_, t2_, vl), vl); \
                } while (0)
                Q4KFOLD(cf0, ad0, Sc0, Sm0);
                Q4KFOLD(cf1, ad1, Sc1, Sm1);
                Q4KFOLD(cf2, ad2, Sc2, Sm2);
                Q4KFOLD(cf3, ad3, Sc3, Sm3);
                #undef Q4KFOLD
            }
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + nj * 4, cf0, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + nj * 4, cf1, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + nj * 4, cf2, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + nj * 4, cf3, vl);
        }
    }
}
// [M7] WIDE NJW=2 two-level epilogue. vmadot_mac_kloop_w2 gives sumi[0..15] (nj) and sumi[16..31]
// (nj+1), each bit-identical to width-1; each col-tile folds its OWN sumi into its OWN Sc/Sm with its
// OWN scales -> Cf BIT-FOR-BIT the width-1 njouter output. asum is column-independent (shared). Odd
// tail col-tile -> width-1 body verbatim.
static void matmul_deref_epi_vec_njouter_w2(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                            const uint8_t * sc_all, const uint8_t * mm_all,
                                            const float * d_all, const float * dmin_all, const int32_t * asum_all,
                                            float * Cf, long M, long N, long K, long nj_start, long nj_end) {
    const long mt = M / 4, nsb = K / 256, kt = K / 8, fpb = 4;
    const size_t vl = __riscv_vsetvl_e32m1(4);
    const long bstride = kt * 32;
    long nj = nj_start;
    for (; nj + 2 <= nj_end; nj += 2) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl), a1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t a2 = __riscv_vfmv_v_f_f32m1(0.0f, vl), a3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t b0 = __riscv_vfmv_v_f_f32m1(0.0f, vl), b1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t b2 = __riscv_vfmv_v_f_f32m1(0.0f, vl), b3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (long sb = 0; sb < nsb; ++sb) {
                vint32m1_t Ac0 = __riscv_vmv_v_x_i32m1(0, vl), Ac1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Ac2 = __riscv_vmv_v_x_i32m1(0, vl), Ac3 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Am0 = __riscv_vmv_v_x_i32m1(0, vl), Am1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Am2 = __riscv_vmv_v_x_i32m1(0, vl), Am3 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Bc0 = __riscv_vmv_v_x_i32m1(0, vl), Bc1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Bc2 = __riscv_vmv_v_x_i32m1(0, vl), Bc3 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Bm0 = __riscv_vmv_v_x_i32m1(0, vl), Bm1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t Bm2 = __riscv_vmv_v_x_i32m1(0, vl), Bm3 = __riscv_vmv_v_x_i32m1(0, vl);
                for (int b = 0; b < 8; ++b) {
                    long    gf0 = sb * 32 + b * 4;
                    int32_t sumi[32];
                    vmadot_mac_kloop_w2(Arow + gf0 * 32, Bcol + gf0 * 32, bstride, fpb, sumi);
                    const uint8_t * scpA = sc_all + ((long) (nj * nsb + sb) * 8 + b) * 4;
                    const uint8_t * mmpA = mm_all + ((long) (nj * nsb + sb) * 8 + b) * 4;
                    const uint8_t * scpB = sc_all + ((long) ((nj + 1) * nsb + sb) * 8 + b) * 4;
                    const uint8_t * mmpB = mm_all + ((long) ((nj + 1) * nsb + sb) * 8 + b) * 4;
                    int32_t scaA[4] = { scpA[0], scpA[1], scpA[2], scpA[3] };
                    int32_t mmaA[4] = { mmpA[0], mmpA[1], mmpA[2], mmpA[3] };
                    int32_t scaB[4] = { scpB[0], scpB[1], scpB[2], scpB[3] };
                    int32_t mmaB[4] = { mmpB[0], mmpB[1], mmpB[2], mmpB[3] };
                    vint32m1_t scbA = __riscv_vle32_v_i32m1(scaA, vl), mmbA = __riscv_vle32_v_i32m1(mmaA, vl);
                    vint32m1_t scbB = __riscv_vle32_v_i32m1(scaB, vl), mmbB = __riscv_vle32_v_i32m1(mmaB, vl);
                    vint32m1_t su0 = __riscv_vle32_v_i32m1(sumi + 0,  vl);
                    vint32m1_t su1 = __riscv_vle32_v_i32m1(sumi + 4,  vl);
                    vint32m1_t su2 = __riscv_vle32_v_i32m1(sumi + 8,  vl);
                    vint32m1_t su3 = __riscv_vle32_v_i32m1(sumi + 12, vl);
                    vint32m1_t sv0 = __riscv_vle32_v_i32m1(sumi + 16, vl);
                    vint32m1_t sv1 = __riscv_vle32_v_i32m1(sumi + 20, vl);
                    vint32m1_t sv2 = __riscv_vle32_v_i32m1(sumi + 24, vl);
                    vint32m1_t sv3 = __riscv_vle32_v_i32m1(sumi + 28, vl);
                    int32_t as0 = asum_all[(mi * 4 + 0) * (nsb * 8) + sb * 8 + b];
                    int32_t as1 = asum_all[(mi * 4 + 1) * (nsb * 8) + sb * 8 + b];
                    int32_t as2 = asum_all[(mi * 4 + 2) * (nsb * 8) + sb * 8 + b];
                    int32_t as3 = asum_all[(mi * 4 + 3) * (nsb * 8) + sb * 8 + b];
                    Ac0 = __riscv_vadd_vv_i32m1(Ac0, __riscv_vmul_vv_i32m1(scbA, su0, vl), vl);
                    Ac1 = __riscv_vadd_vv_i32m1(Ac1, __riscv_vmul_vv_i32m1(scbA, su1, vl), vl);
                    Ac2 = __riscv_vadd_vv_i32m1(Ac2, __riscv_vmul_vv_i32m1(scbA, su2, vl), vl);
                    Ac3 = __riscv_vadd_vv_i32m1(Ac3, __riscv_vmul_vv_i32m1(scbA, su3, vl), vl);
                    Am0 = __riscv_vadd_vv_i32m1(Am0, __riscv_vmul_vx_i32m1(mmbA, as0, vl), vl);
                    Am1 = __riscv_vadd_vv_i32m1(Am1, __riscv_vmul_vx_i32m1(mmbA, as1, vl), vl);
                    Am2 = __riscv_vadd_vv_i32m1(Am2, __riscv_vmul_vx_i32m1(mmbA, as2, vl), vl);
                    Am3 = __riscv_vadd_vv_i32m1(Am3, __riscv_vmul_vx_i32m1(mmbA, as3, vl), vl);
                    Bc0 = __riscv_vadd_vv_i32m1(Bc0, __riscv_vmul_vv_i32m1(scbB, sv0, vl), vl);
                    Bc1 = __riscv_vadd_vv_i32m1(Bc1, __riscv_vmul_vv_i32m1(scbB, sv1, vl), vl);
                    Bc2 = __riscv_vadd_vv_i32m1(Bc2, __riscv_vmul_vv_i32m1(scbB, sv2, vl), vl);
                    Bc3 = __riscv_vadd_vv_i32m1(Bc3, __riscv_vmul_vv_i32m1(scbB, sv3, vl), vl);
                    Bm0 = __riscv_vadd_vv_i32m1(Bm0, __riscv_vmul_vx_i32m1(mmbB, as0, vl), vl);
                    Bm1 = __riscv_vadd_vv_i32m1(Bm1, __riscv_vmul_vx_i32m1(mmbB, as1, vl), vl);
                    Bm2 = __riscv_vadd_vv_i32m1(Bm2, __riscv_vmul_vx_i32m1(mmbB, as2, vl), vl);
                    Bm3 = __riscv_vadd_vv_i32m1(Bm3, __riscv_vmul_vx_i32m1(mmbB, as3, vl), vl);
                }
                vfloat32m1_t dvA  = __riscv_vle32_v_f32m1(d_all + (long) (nj * nsb + sb) * 4, vl);
                vfloat32m1_t dmvA = __riscv_vle32_v_f32m1(dmin_all + (long) (nj * nsb + sb) * 4, vl);
                vfloat32m1_t dvB  = __riscv_vle32_v_f32m1(d_all + (long) ((nj + 1) * nsb + sb) * 4, vl);
                vfloat32m1_t dmvB = __riscv_vle32_v_f32m1(dmin_all + (long) ((nj + 1) * nsb + sb) * 4, vl);
                float ad0 = dA[(mi * 4 + 0) * nsb + sb], ad1 = dA[(mi * 4 + 1) * nsb + sb];
                float ad2 = dA[(mi * 4 + 2) * nsb + sb], ad3 = dA[(mi * 4 + 3) * nsb + sb];
                #define Q4KFOLD2(cf, dv, dmv, ad, Sc, Sm) do { \
                    vfloat32m1_t p_ = __riscv_vfmul_vf_f32m1((dv), (ad), vl); \
                    vfloat32m1_t q_ = __riscv_vfmul_vf_f32m1((dmv), (ad), vl); \
                    vfloat32m1_t sf_ = __riscv_vfcvt_f_x_v_f32m1((Sc), vl); \
                    vfloat32m1_t mf_ = __riscv_vfcvt_f_x_v_f32m1((Sm), vl); \
                    vfloat32m1_t t1_ = __riscv_vfmul_vv_f32m1(p_, sf_, vl); \
                    vfloat32m1_t t2_ = __riscv_vfmul_vv_f32m1(q_, mf_, vl); \
                    (cf) = __riscv_vfadd_vv_f32m1((cf), __riscv_vfsub_vv_f32m1(t1_, t2_, vl), vl); \
                } while (0)
                Q4KFOLD2(a0, dvA, dmvA, ad0, Ac0, Am0);
                Q4KFOLD2(a1, dvA, dmvA, ad1, Ac1, Am1);
                Q4KFOLD2(a2, dvA, dmvA, ad2, Ac2, Am2);
                Q4KFOLD2(a3, dvA, dmvA, ad3, Ac3, Am3);
                Q4KFOLD2(b0, dvB, dmvB, ad0, Bc0, Bm0);
                Q4KFOLD2(b1, dvB, dmvB, ad1, Bc1, Bm1);
                Q4KFOLD2(b2, dvB, dmvB, ad2, Bc2, Bm2);
                Q4KFOLD2(b3, dvB, dmvB, ad3, Bc3, Bm3);
                #undef Q4KFOLD2
            }
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + nj * 4, a0, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + nj * 4, a1, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + nj * 4, a2, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + nj * 4, a3, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + (nj + 1) * 4, b0, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + (nj + 1) * 4, b1, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + (nj + 1) * 4, b2, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + (nj + 1) * 4, b3, vl);
        }
    }
    if (nj < nj_end)
        matmul_deref_epi_vec_njouter(Apack, dA, Bdec_all, sc_all, mm_all, d_all, dmin_all, asum_all,
                                     Cf, M, N, K, nj, nj_end);
}
// ---- timing-only probes (PROF phase; format-agnostic vmadot leaf, same call count as real) --------
static void matmul_noepi_njouter(const int8_t * Apack, const int8_t * Bdec_all, long M, long N, long K,
                                 long nj_start, long nj_end) {
    const long mt = M / 4, nsb = K / 256, kt = K / 8, fpb = 4;
    (void) N;
    for (long nj = nj_start; nj < nj_end; ++nj) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            for (long sb = 0; sb < nsb; ++sb)
                for (int b = 0; b < 8; ++b) {
                    long gf0 = sb * 32 + b * 4; int32_t sumi[16];
                    vmadot_mac_kloop(Arow + gf0 * 32, Bcol + gf0 * 32, fpb, sumi);
                }
        }
    }
}
static void matmul_noepi_njouter_w2(const int8_t * Apack, const int8_t * Bdec_all, long M, long N, long K,
                                    long nj_start, long nj_end) {
    const long mt = M / 4, nsb = K / 256, kt = K / 8, fpb = 4, bstride = (K / 8) * 32;
    (void) N;
    long nj = nj_start;
    for (; nj + 2 <= nj_end; nj += 2) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            for (long sb = 0; sb < nsb; ++sb)
                for (int b = 0; b < 8; ++b) {
                    long gf0 = sb * 32 + b * 4; int32_t sumi[32];
                    vmadot_mac_kloop_w2(Arow + gf0 * 32, Bcol + gf0 * 32, bstride, fpb, sumi);
                }
        }
    }
    for (; nj < nj_end; ++nj) matmul_noepi_njouter(Apack, Bdec_all, M, N, K, nj, nj + 1);
}
static void matmul_noepi_l1(long M, long K, long nj_start, long nj_end) {
    const long mt = M / 4, nsb = K / 256, fpb = 4;
    alignas(64) int8_t sA[128]; alignas(64) int8_t sB[128];
    for (int i = 0; i < 128; ++i) { sA[i] = (int8_t) (i & 7); sB[i] = (int8_t) ((i * 3) & 7); }
    for (long nj = nj_start; nj < nj_end; ++nj)
        for (long mi = 0; mi < mt; ++mi)
            for (long sb = 0; sb < nsb; ++sb)
                for (int b = 0; b < 8; ++b) { int32_t sumi[16]; vmadot_mac_kloop(sA, sB, fpb, sumi); }
}
static void matmul_noepi_l1_w2(long M, long K, long nj_start, long nj_end) {
    const long mt = M / 4, nsb = K / 256, fpb = 4;
    alignas(64) int8_t sA[128]; alignas(64) int8_t sB[256];
    for (int i = 0; i < 128; ++i) sA[i] = (int8_t) (i & 7);
    for (int i = 0; i < 256; ++i) sB[i] = (int8_t) ((i * 3) & 7);
    for (long nj = nj_start; nj + 2 <= nj_end; nj += 2)
        for (long mi = 0; mi < mt; ++mi)
            for (long sb = 0; sb < nsb; ++sb)
                for (int b = 0; b < 8; ++b) { int32_t sumi[32]; vmadot_mac_kloop_w2(sA, sB, 128, fpb, sumi); }
}

// ---------------- dequant cache + shared MT store + [PROF] accounting ------------------------------
struct PackedWDec {
    std::vector<int8_t> Bdec; std::vector<uint8_t> sc; std::vector<uint8_t> mm;
    std::vector<float> d; std::vector<float> dmin; long N; long K;
};
static std::unordered_map<const void *, PackedWDec> g_wcache_dec;
struct MTWork { const int8_t * Apack; const float * dA; const int8_t * Bdec;
                const uint8_t * sc; const uint8_t * mm; const float * d; const float * dmin;
                const int32_t * asum; float * Cfp; const float * X; long Mp; long Ml; long Nl; long Kl; };
static MTWork              g_mtwork = {};
static std::vector<int8_t>  g_Apack;
static std::vector<float>   g_dA;
static std::vector<int32_t> g_asum;
static std::vector<float>   g_Cfp;
static uint64_t g_cyc_dequant = 0, g_cyc_quant = 0, g_cyc_asum = 0, g_cyc_matmul = 0, g_cyc_alloc = 0, g_cyc_copyback = 0;
static uint64_t g_n_calls = 0, g_n_dequant_runs = 0, g_n_cache_hits = 0;
static bool     g_env_read = false, g_cache = false, g_threads = false, g_prof = false, g_prof_reg = false;
static bool     g_deref = false, g_deref_epi = false, g_parsetup = false;
static bool     g_epivec = false, g_mmprof_noepi = false, g_njouter = false, g_mmprof_l1 = false;
static int      g_tilew = 0;
static inline uint64_t nowns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}
// dispatch the deref matmul family for a column-tile [njs,nje]. q4_K deployed form is always
// deref+epivec+njouter; tilew selects the WIDE (NJW=2) leaf. prof variants take priority.
static inline void run_deref_epi_tile(const MTWork & w, long Mp, long Nl, long Kl, long njs, long nje) {
    if (g_mmprof_l1) {
        if (g_tilew == 2) matmul_noepi_l1_w2(Mp, Kl, njs, nje);
        else              matmul_noepi_l1(Mp, Kl, njs, nje);
    } else if (g_mmprof_noepi) {
        if (g_tilew == 2) matmul_noepi_njouter_w2(w.Apack, w.Bdec, Mp, Nl, Kl, njs, nje);
        else              matmul_noepi_njouter(w.Apack, w.Bdec, Mp, Nl, Kl, njs, nje);
    } else {
        if (g_tilew == 2) matmul_deref_epi_vec_njouter_w2(w.Apack, w.dA, w.Bdec, w.sc, w.mm, w.d, w.dmin,
                                                          w.asum, w.Cfp, Mp, Nl, Kl, njs, nje);
        else              matmul_deref_epi_vec_njouter(w.Apack, w.dA, w.Bdec, w.sc, w.mm, w.d, w.dmin,
                                                       w.asum, w.Cfp, Mp, Nl, Kl, njs, nje);
    }
}
static void prof_dump() {
    uint64_t tot = g_cyc_dequant + g_cyc_quant + g_cyc_asum + g_cyc_matmul + 1;
    fprintf(stderr,
            "[TCRV-Q4K-PROF] cache=%d threads=%d deref=%d deref_epi=%d parsetup=%d epivec=%d "
            "njouter=%d tilew=%d mmprof_noepi=%d mmprof_l1=%d calls=%llu dequant_runs=%llu cache_hits=%llu | "
            "ns_dequant=%llu ns_quant=%llu ns_asum=%llu ns_alloc=%llu ns_copyback=%llu ns_matmul=%llu | "
            "dequant_share=%.4f quant_share=%.4f asum_share=%.4f matmul_share=%.4f\n",
            (int) g_cache, (int) g_threads, (int) g_deref, (int) g_deref_epi, (int) g_parsetup,
            (int) g_epivec, (int) g_njouter, g_tilew, (int) g_mmprof_noepi, (int) g_mmprof_l1,
            (unsigned long long) g_n_calls, (unsigned long long) g_n_dequant_runs,
            (unsigned long long) g_n_cache_hits, (unsigned long long) g_cyc_dequant,
            (unsigned long long) g_cyc_quant, (unsigned long long) g_cyc_asum,
            (unsigned long long) g_cyc_alloc, (unsigned long long) g_cyc_copyback,
            (unsigned long long) g_cyc_matmul,
            (double) g_cyc_dequant / (double) tot, (double) g_cyc_quant / (double) tot,
            (double) g_cyc_asum / (double) tot, (double) g_cyc_matmul / (double) tot);
}
static inline void env_read_once() {
    if (g_env_read) return;
    g_env_read = true;
    g_cache        = std::getenv("TCRV_IME_Q4K_CACHE")        != nullptr;
    g_threads      = std::getenv("TCRV_IME_Q4K_THREADS")      != nullptr;
    g_deref        = std::getenv("TCRV_IME_Q4K_DEREF")        != nullptr;
    g_deref_epi    = std::getenv("TCRV_IME_Q4K_DEREF_EPI")    != nullptr;
    g_parsetup     = std::getenv("TCRV_IME_Q4K_PARSETUP")     != nullptr;
    g_epivec       = std::getenv("TCRV_IME_Q4K_EPIVEC")       != nullptr;
    g_njouter      = std::getenv("TCRV_IME_Q4K_NJOUTER")      != nullptr;
    g_mmprof_noepi = std::getenv("TCRV_IME_Q4K_MMPROF_NOEPI") != nullptr;
    g_mmprof_l1    = std::getenv("TCRV_IME_Q4K_MMPROF_L1")    != nullptr;
    { const char * tw = std::getenv("TCRV_IME_Q4K_TILEW"); g_tilew = tw ? atoi(tw) : 0;
      if (g_tilew != 2) g_tilew = 0; }   // q4_K WIDE = NJW=2 only (W4 register pressure too high)
    g_prof         = std::getenv("TCRV_IME_Q4K_PROF")         != nullptr;
    if (g_prof && !g_prof_reg) { g_prof_reg = true; atexit(prof_dump); }
}
}  // namespace tcrv_q4k

class tcrv_q4_K_tensor_traits : public tensor_traits_base {
    bool work_size(int /*n_threads*/, const ggml_tensor * /*op*/, size_t & /*size*/) override { return false; }

    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        memcpy(t->data, data, data_size);  // NATIVE passthrough (144B/block preserved)
        return 0;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;
        if (src0->type != GGML_TYPE_Q4_K || src1->type != GGML_TYPE_F32) {
            return false;
        }
        const int64_t K = src0->ne[0];
        const int64_t N = src0->ne[1];
        const int64_t M = src1->ne[1];
        if (M <= 1 || N % 4 != 0 || K % 256 != 0) {
            return false;
        }
        if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
            return false;
        }
        if (src1->nb[0] != sizeof(float) || src1->nb[1] != (size_t) K * sizeof(float)) {
            return false;
        }
        if (dst->nb[0] != sizeof(float) || dst->nb[1] != (size_t) N * sizeof(float)) {
            return false;
        }
        if (src0->nb[1] != (size_t) (K / QK_K) * sizeof(block_q4_K)) {
            return false;
        }
        const int  ith = params->ith;
        const int  nth = params->nth;
        const long Ml = (long) M, Nl = (long) N, Kl = (long) K;
        const long Mp = (Ml + 3) / 4 * 4;
        const long nsb = Kl / 256;
        float *    Cf = (float *) dst->data;

        if (ith == 0) {
            tcrv_q4k::env_read_once();
            tcrv_q4k::g_n_calls++;
            memset(Cf, 0, (size_t) Ml * Nl * sizeof(float));

            const int8_t *  Bdec_p = nullptr; const uint8_t * sc_p = nullptr; const uint8_t * mm_p = nullptr;
            const float *   d_p = nullptr; const float * dmin_p = nullptr;
            tcrv_q4k::PackedWDec * pw_local = nullptr;
            static tcrv_q4k::PackedWDec s_local;   // per-call scratch when cache off
            // q4_K: DEREF==CACHE (both = load-once dequant). Baseline = per-call dequant into s_local.
            if (tcrv_q4k::g_cache || tcrv_q4k::g_deref) {
                auto it = tcrv_q4k::g_wcache_dec.find(src0->data);
                if (it == tcrv_q4k::g_wcache_dec.end() || it->second.N != Nl || it->second.K != Kl) {
                    tcrv_q4k::PackedWDec pw; pw.N = Nl; pw.K = Kl;
                    pw.Bdec.assign((size_t) (Nl / 4) * (Kl / 8) * 32, 0);
                    pw.sc.assign((size_t) Nl * nsb * 8, 0);
                    pw.mm.assign((size_t) Nl * nsb * 8, 0);
                    pw.d.assign((size_t) Nl * nsb, 0.0f);
                    pw.dmin.assign((size_t) Nl * nsb, 0.0f);
                    uint64_t t0 = tcrv_q4k::nowns();
                    tcrv_q4k::repack_dequant_weight((const uint8_t *) src0->data, pw.Bdec.data(), pw.sc.data(),
                                                    pw.mm.data(), pw.d.data(), pw.dmin.data(), Nl, Kl);
                    tcrv_q4k::g_cyc_dequant += tcrv_q4k::nowns() - t0;
                    tcrv_q4k::g_n_dequant_runs++;
                    it = tcrv_q4k::g_wcache_dec.emplace(src0->data, std::move(pw)).first;
                } else {
                    tcrv_q4k::g_n_cache_hits++;
                }
                pw_local = &it->second;
            } else {
                s_local.N = Nl; s_local.K = Kl;
                s_local.Bdec.assign((size_t) (Nl / 4) * (Kl / 8) * 32, 0);
                s_local.sc.assign((size_t) Nl * nsb * 8, 0);
                s_local.mm.assign((size_t) Nl * nsb * 8, 0);
                s_local.d.assign((size_t) Nl * nsb, 0.0f);
                s_local.dmin.assign((size_t) Nl * nsb, 0.0f);
                uint64_t t0 = tcrv_q4k::nowns();
                tcrv_q4k::repack_dequant_weight((const uint8_t *) src0->data, s_local.Bdec.data(), s_local.sc.data(),
                                                s_local.mm.data(), s_local.d.data(), s_local.dmin.data(), Nl, Kl);
                tcrv_q4k::g_cyc_dequant += tcrv_q4k::nowns() - t0;
                tcrv_q4k::g_n_dequant_runs++;
                pw_local = &s_local;
            }
            Bdec_p = pw_local->Bdec.data(); sc_p = pw_local->sc.data(); mm_p = pw_local->mm.data();
            d_p = pw_local->d.data(); dmin_p = pw_local->dmin.data();

            const bool epi_full = tcrv_q4k::g_deref_epi;
            uint64_t ta = tcrv_q4k::nowns();
            tcrv_q4k::g_Apack.assign((size_t) Mp * Kl, 0);
            tcrv_q4k::g_dA.assign((size_t) Mp * nsb, 0.0f);
            tcrv_q4k::g_asum.assign((size_t) Mp * nsb * 8, 0);
            if (tcrv_q4k::g_parsetup && epi_full) tcrv_q4k::g_Cfp.resize((size_t) Mp * Nl);
            else                                  tcrv_q4k::g_Cfp.assign((size_t) Mp * Nl, 0.0f);
            tcrv_q4k::g_cyc_alloc += tcrv_q4k::nowns() - ta;

            const bool par_quant = tcrv_q4k::g_parsetup && tcrv_q4k::g_threads;
            if (!par_quant) {
                std::vector<uint8_t> scratch((size_t) nsb * sizeof(block_q8_K));
                uint64_t t1 = tcrv_q4k::nowns();
                tcrv_q4k::quant_pack_act((const float *) src1->data, tcrv_q4k::g_Apack.data(),
                                         tcrv_q4k::g_dA.data(), Ml, Kl, scratch.data());
                tcrv_q4k::g_cyc_quant += tcrv_q4k::nowns() - t1;
                uint64_t t1a = tcrv_q4k::nowns();
                tcrv_q4k::compute_asum(tcrv_q4k::g_Apack.data(), tcrv_q4k::g_asum.data(), Mp, Kl, 0, Ml);
                tcrv_q4k::g_cyc_asum += tcrv_q4k::nowns() - t1a;
            }

            tcrv_q4k::g_mtwork.Apack = tcrv_q4k::g_Apack.data();
            tcrv_q4k::g_mtwork.dA    = tcrv_q4k::g_dA.data();
            tcrv_q4k::g_mtwork.Bdec  = Bdec_p;
            tcrv_q4k::g_mtwork.sc    = sc_p;
            tcrv_q4k::g_mtwork.mm    = mm_p;
            tcrv_q4k::g_mtwork.d     = d_p;
            tcrv_q4k::g_mtwork.dmin  = dmin_p;
            tcrv_q4k::g_mtwork.asum  = tcrv_q4k::g_asum.data();
            tcrv_q4k::g_mtwork.Cfp   = tcrv_q4k::g_Cfp.data();
            tcrv_q4k::g_mtwork.X     = (const float *) src1->data;
            tcrv_q4k::g_mtwork.Mp    = Mp;
            tcrv_q4k::g_mtwork.Ml    = Ml;
            tcrv_q4k::g_mtwork.Nl    = Nl;
            tcrv_q4k::g_mtwork.Kl    = Kl;

            if (!tcrv_q4k::g_threads) {
                const long nt = Nl / 4;
                uint64_t t2 = tcrv_q4k::nowns();
                tcrv_q4k::run_deref_epi_tile(tcrv_q4k::g_mtwork, Mp, Nl, Kl, 0, nt);
                tcrv_q4k::g_cyc_matmul += tcrv_q4k::nowns() - t2;
                memcpy(Cf, tcrv_q4k::g_Cfp.data(), (size_t) Ml * Nl * sizeof(float));
            }

            static bool announced = false;
            if (!announced) {
                announced = true;
                fprintf(stderr,
                        "[TCRV-IME-Q4K-BRIDGE] routed real q4_K PREFILL mul_mat through tcrv IME kernel: "
                        "M=%ld N=%ld K=%ld (real vmadot 0xe210312b two-level fold) cache=%d threads=%d deref=%d deref_epi=%d parsetup=%d epivec=%d njouter=%d tilew=%d nth=%d\n",
                        Ml, Nl, Kl, (int) tcrv_q4k::g_cache, (int) tcrv_q4k::g_threads,
                        (int) tcrv_q4k::g_deref, (int) tcrv_q4k::g_deref_epi, (int) tcrv_q4k::g_parsetup,
                        (int) tcrv_q4k::g_epivec, (int) tcrv_q4k::g_njouter, tcrv_q4k::g_tilew, nth);
            }
        }
        ggml_barrier(params->threadpool);  // barrier #1

        if (tcrv_q4k::g_parsetup && tcrv_q4k::g_threads) {
            const long Mlq = tcrv_q4k::g_mtwork.Ml;
            const long Klq = tcrv_q4k::g_mtwork.Kl;
            const long Mpq = tcrv_q4k::g_mtwork.Mp;
            long       per = (Mlq + nth - 1) / nth;
            per = (per + 3) & ~3L;
            long ms = (long) ith * per;
            long me = ms + per;
            if (me > Mlq) me = Mlq;
            uint64_t tq = (ith == 0) ? tcrv_q4k::nowns() : 0;
            if (ms < me) {
                std::vector<uint8_t> scratch((size_t) (Klq / 256) * sizeof(block_q8_K));
                tcrv_q4k::quant_pack_act_range(tcrv_q4k::g_mtwork.X, tcrv_q4k::g_Apack.data(),
                                               tcrv_q4k::g_dA.data(), Mlq, Klq, scratch.data(), ms, me);
                tcrv_q4k::compute_asum(tcrv_q4k::g_Apack.data(), tcrv_q4k::g_asum.data(), Mpq, Klq, ms, me);
            }
            ggml_barrier(params->threadpool);  // barrier #1b
            if (ith == 0) tcrv_q4k::g_cyc_quant += tcrv_q4k::nowns() - tq;
        }

        if (tcrv_q4k::g_threads) {
            const long nt  = tcrv_q4k::g_mtwork.Nl / 4;
            long       per = (nt + nth - 1) / nth;
            per = (per + 3) & ~3L;
            long njs = (long) ith * per;
            long nje = njs + per;
            if (nje > nt) nje = nt;
            uint64_t t2 = (ith == 0) ? tcrv_q4k::nowns() : 0;
            if (njs < nje) {
                tcrv_q4k::run_deref_epi_tile(tcrv_q4k::g_mtwork, tcrv_q4k::g_mtwork.Mp,
                                             tcrv_q4k::g_mtwork.Nl, tcrv_q4k::g_mtwork.Kl, njs, nje);
            }
            ggml_barrier(params->threadpool);  // barrier #2
            if (ith == 0) {
                tcrv_q4k::g_cyc_matmul += tcrv_q4k::nowns() - t2;
                uint64_t tc = tcrv_q4k::nowns();
                memcpy(Cf, tcrv_q4k::g_mtwork.Cfp, (size_t) Ml * Nl * sizeof(float));
                tcrv_q4k::g_cyc_copyback += tcrv_q4k::nowns() - tc;
            }
        }
        ggml_barrier(params->threadpool);  // barrier #3
        return true;
    }
};

static tcrv_q4_K_tensor_traits tcrv_q4_K_bridge;

'''
s = s.replace(cls_anchor, block + cls_anchor, 1)

# --- env-gated selection in get_optimal_repack_type Q4_K case --------------
sel_anchor = ("        case GGML_TYPE_Q4_K:\n"
              "            {\n"
              "#if defined(RISCV64_SPACEMIT_IME2)\n")
assert s.count(sel_anchor) == 1, "expected exactly one Q4_K get_optimal anchor"
sel_new = ("        case GGML_TYPE_Q4_K:\n"
           "            {\n"
           "                if (std::getenv(\"TCRV_IME_Q4K_BRIDGE\") && cur->ne[1] % 4 == 0 && cur->ne[0] % 256 == 0) {\n"
           "                    return &ggml::cpu::riscv64_spacemit::tcrv_q4_K_bridge;\n"
           "                }\n"
           "#if defined(RISCV64_SPACEMIT_IME2)\n")
s = s.replace(sel_anchor, sel_new, 1)

open(F, "w").write(s)

t = open(F).read()
assert "class tcrv_q4_K_tensor_traits" in t, "trait class missing after write"
assert "static tcrv_q4_K_tensor_traits tcrv_q4_K_bridge;" in t, "static instance missing"
assert "return &ggml::cpu::riscv64_spacemit::tcrv_q4_K_bridge;" in t, "selection missing"
assert "repack_dequant_weight" in t, "dequant populate missing"
assert "g_wcache_dec" in t, "deref cache missing"
assert "compute_asum" in t, "asum precompute missing"
assert "matmul_deref_epi_vec_njouter" in t, "onjout matmul missing"
assert "matmul_deref_epi_vec_njouter_w2" in t, "[M7] wide NJW=2 matmul missing"
assert "vmadot_mac_kloop_w2" in t, "[M7] wide vmadot leaf missing"
assert "TCRV_IME_Q4K_TILEW" in t and "g_tilew" in t, "[M7] tilew gate/flag missing"
assert "TCRV_IME_Q4K_EPIVEC" in t and "TCRV_IME_Q4K_NJOUTER" in t, "[M5/M6] gates missing"
assert "TCRV_IME_Q4K_MMPROF_L1" in t and "TCRV_IME_Q4K_MMPROF_NOEPI" in t, "prof gates missing"
assert ("barrier #1" in t and "barrier #2" in t and "barrier #3" in t), "barriers missing"
assert "TCRV-Q4K-PROF" in t, "[PROF] dump missing"
assert t.count("[TCRV-IME-Q4K-BRIDGE]") >= 2, "markers missing"
print("patched OK; bridge_markers=%d" % t.count("TCRV-IME-Q4K-BRIDGE") +
      "; dequant_cache=1; asum=1; two_level_epi_vec=1; njouter=1; tilew_w2=1; "
      "mmprof_noepi=1; mmprof_l1=1; trait=1; selection=1")
