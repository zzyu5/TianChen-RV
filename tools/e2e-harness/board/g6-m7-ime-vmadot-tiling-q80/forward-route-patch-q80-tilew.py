#!/usr/bin/env python3
# G6-A M7 (vmadot-tiling / array-utilization) patch for vendor spacemit ime.cpp -- q8_0 IME bridge.
#
# q8_0 is INT8-DIRECT: the native weight is already int8 (34B block = fp16 scale + 32 int8), so the
# "repack" is a plain int8 GATHER into fragment-major layout -- IDENTICAL in content/layout to what
# q4_0's DEREF cache produces after nibble->int8 dequant. Therefore the ENTIRE int8-reading matmul
# family (deref / deref_epi / deref_epi_vec / _njouter / _njouter_w2 / _njouter_w4 + all timing
# probes) is BYTE-IDENTICAL to the sealed q4_0 M7 family, and the q8_0 activation-quant path is the
# SAME quantize_row_q8_0_ref used by q4_0. The ONLY format-specific code is the int8 weight gather.
# For q8_0 there is NO nibble path: DEREF==CACHE (both hold the same int8 gather); the M3 "deref"
# gate is a no-op relayout distinction and both feed the identical int8 B to vmadot.
#
# [M7 optimization] TCRV_IME_Q80_TILEW=2|4 (requires EPIVEC + NJOUTER): WIDEN the vmadot output tile
#   to reuse the A fragment IN-REGISTER across NJW=2|4 adjacent column-tiles. ONE vle8 of the 4x8 A
#   fragment feeds NJW independent vmadot chains -> the array sees NJW in-flight MACs (latency hidden,
#   not serialized on one accumulator) and A-load + per-block vsetvli/clear/store is amortized NJW-fold.
#   Each 4x4 int32 sub-tile still accumulates its kt fragments in the SAME kf order into its OWN
#   accumulator -> int32 BIT-IDENTICAL to NJW separate vmadot_mac_kloop calls (byte-exact by
#   construction; only the loop/reuse SCHEDULE changes, the 0xe210312b vmadot MAC + int32 accumulation
#   order are untouched). nj-outer B-locality preserved; odd tail -> M6 single-tile body (also bit-exact).
#
# Env gates (additive; default OFF -> byte-identical to the sealed q8_0 bridge):
#   TCRV_IME_Q80_BRIDGE    : route q8_0 prefill mul_mat through the tcrv IME kernel
#   TCRV_IME_Q80_CACHE     : [M1] load-once int8 weight-gather cache
#   TCRV_IME_Q80_THREADS   : [M2] column-tile multithread the matmul
#   TCRV_IME_Q80_DEREF     : [M3-a] read the pre-gathered int8 weight (for q8_0 == CACHE)
#   TCRV_IME_Q80_DEREF_EPI : [M3-b] register-accumulated epilogue (requires DEREF)
#   TCRV_IME_Q80_PARSETUP  : [M4]  parallelize the ith==0 activation quant + lighten allocs
#   TCRV_IME_Q80_EPIVEC    : [M5]  4-wide (across-columns) vectorized epilogue fold (bit-exact)
#   TCRV_IME_Q80_NJOUTER   : [M6]  nj-outer/mi-inner tile order for B-feed locality (bit-exact)
#   TCRV_IME_Q80_TILEW     : [M7]  WIDE vmadot output tile: reuse A across NJW=2|4 col-tiles (bit-exact)
#   TCRV_IME_Q80_MMPROF_NOEPI : [prof] vmadot only (skip fold)                      -- timing-only
#   TCRV_IME_Q80_MMPROF_L1    : [prof] vmadot from fixed L1 scratch (compute only)  -- timing-only
#   TCRV_IME_Q80_PROF      : per-stage ns split (gather/quant/alloc/copyback/matmul)
import sys

F = "ggml/src/ggml-cpu/spacemit/ime.cpp"
s = open(F).read()

# --- 1. includes -----------------------------------------------------------
inc_anchor = '#include <cstdio>  // for GGML_ASSERT\n'
assert inc_anchor in s, "cstdio include anchor not found"
add_inc = ('#include <cstdlib>       // [TCRV-Q80] std::getenv/atexit\n'
           '#include <cstring>       // [TCRV-Q80] memcpy/memset\n'
           '#include <ctime>         // [TCRV-Q80-M1] clock_gettime prof\n'
           '#include <vector>        // [TCRV-Q80-M2] shared work vectors\n'
           '#include <unordered_map> // [TCRV-Q80-M1] weight gather cache\n')
if "[TCRV-Q80] std::getenv" not in s:
    s = s.replace(inc_anchor, inc_anchor + add_inc, 1)

# --- 2. helpers + trait class + static instance (inside namespace) ---------
cls_anchor = "// Impl By IME1\n"
assert s.count(cls_anchor) == 1, "expected exactly one '// Impl By IME1' anchor"

block = r'''// ==================== [TCRV-IME-Q80-BRIDGE] int8-direct routing + M1 gather-cache + M2 mt + M3 deref + M4 parsetup + M5/M6/M7 ====================
// q8_0 weight is int8-direct: repack = int8 GATHER into fragment-major (== q4_0's dequantized B).
// The int8-reading matmul family below is byte-identical to the sealed q4_0 M7 kernels; only the
// weight gather is format-specific. real vmadot 0xe210312b. All gates default OFF -> sealed bridge.
extern "C" void quantize_row_q8_0_ref(const float * x, void * y, int64_t k);

namespace tcrv_q80 {
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
// [M7] WIDE vmadot tiling NJW=4 (byte-identical to sealed q4_0 wide leaf).
static inline void vmadot_mac_kloop_w4(const int8_t * A, const int8_t * B0, long bstride, long kt, int32_t * frag) {
    const int8_t * B1 = B0 + bstride, * B2 = B0 + 2 * bstride, * B3 = B0 + 3 * bstride;
    __asm__ volatile(
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "vmv.v.i v2, 0\n\t"  "vmv.v.i v3, 0\n\t"
        "vmv.v.i v4, 0\n\t"  "vmv.v.i v5, 0\n\t"
        "vmv.v.i v10, 0\n\t" "vmv.v.i v11, 0\n\t"
        "vmv.v.i v12, 0\n\t" "vmv.v.i v13, 0\n\t"
        "mv t2, %[kt]\n\t"
        "mv t1, %[pa]\n\t"
        "mv t3, %[pb0]\n\t"
        "mv t4, %[pb1]\n\t"
        "mv t5, %[pb2]\n\t"
        "mv t6, %[pb3]\n\t"
        "1:\n\t"
        "vle8.v v0, (t1)\n\t"
        "vle8.v v1, (t3)\n\t"
        "vle8.v v6, (t4)\n\t"
        "vle8.v v7, (t5)\n\t"
        "vle8.v v8, (t6)\n\t"
        "vmadot v2, v0, v1\n\t"
        "vmadot v4, v0, v6\n\t"
        "vmadot v10, v0, v7\n\t"
        "vmadot v12, v0, v8\n\t"
        "addi t1, t1, 32\n\t"
        "addi t3, t3, 32\n\t"
        "addi t4, t4, 32\n\t"
        "addi t5, t5, 32\n\t"
        "addi t6, t6, 32\n\t"
        "addi t2, t2, -1\n\t"
        "bnez t2, 1b\n\t"
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "mv t1, %[pf]\n\t"
        "vse32.v v2, (t1)\n\t"  "addi t1, t1, 32\n\t"
        "vse32.v v3, (t1)\n\t"  "addi t1, t1, 32\n\t"
        "vse32.v v4, (t1)\n\t"  "addi t1, t1, 32\n\t"
        "vse32.v v5, (t1)\n\t"  "addi t1, t1, 32\n\t"
        "vse32.v v10, (t1)\n\t" "addi t1, t1, 32\n\t"
        "vse32.v v11, (t1)\n\t" "addi t1, t1, 32\n\t"
        "vse32.v v12, (t1)\n\t" "addi t1, t1, 32\n\t"
        "vse32.v v13, (t1)\n\t"
        :
        : [pa] "r"(A), [pb0] "r"(B0), [pb1] "r"(B1), [pb2] "r"(B2), [pb3] "r"(B3), [kt] "r"(kt), [pf] "r"(frag)
        : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "v0", "v1", "v2", "v3", "v4", "v5",
          "v6", "v7", "v8", "v10", "v11", "v12", "v13", "memory");
}
// q8_0 INT8-DIRECT weight gather: ggml NATIVE q8_0 (row-major N x nb, 34B/block) -> fragment-major
// int8 Bpack + per-(col,block) dW. Output layout (nj*kt + kf)*32 + nl*8 + kl is IDENTICAL to q4_0's
// dequantized B, so the int8-reading matmul family consumes it verbatim.
static void repack_int8_weight(const uint8_t * wq, int8_t * Bpack, float * dW, long N, long K) {
    const long nb = K / 32, kt = K / 8;
    memset(Bpack, 0, (size_t) (N / 4) * kt * 32);
    for (long n = 0; n < N; ++n)
        for (long b = 0; b < nb; ++b) {
            const uint8_t * src = wq + (n * nb + b) * 34;
            uint16_t        dh  = (uint16_t) (src[0] | (src[1] << 8));
            dW[n * nb + b]      = ggml_fp16_to_fp32(dh);
            const int8_t * qs   = (const int8_t *) (src + 2);
            for (long kk = 0; kk < 32; ++kk) {
                long k = b * 32 + kk, nj = n / 4, nl = n % 4, kf = k / 8, kl = k % 8;
                Bpack[(nj * kt + kf) * 32 + nl * 8 + kl] = qs[kk];
            }
        }
}
// activation: ggml f32 (M x K) -> q8_0 (quantize_row_q8_0_ref) -> Apack + dA (byte-identical to q4_0).
static void quant_pack_act(const float * X, int8_t * Apack, float * dA, long M, long K, uint8_t * scratch) {
    const long nb = K / 32;
    for (long m = 0; m < M; ++m) {
        quantize_row_q8_0_ref(X + m * K, scratch, K);
        for (long b = 0; b < nb; ++b) {
            const uint8_t * blk = scratch + b * 34;
            uint16_t        dh  = (uint16_t) (blk[0] | (blk[1] << 8));
            dA[m * nb + b]      = ggml_fp16_to_fp32(dh);
            const int8_t * qs   = (const int8_t *) (blk + 2);
            for (long kk = 0; kk < 32; ++kk) {
                long k = b * 32 + kk, mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
                Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] = qs[kk];
            }
        }
    }
}
static void quant_pack_act_range(const float * X, int8_t * Apack, float * dA, long M, long K,
                                 uint8_t * scratch, long m_start, long m_end) {
    const long nb = K / 32;
    (void) M;
    for (long m = m_start; m < m_end; ++m) {
        quantize_row_q8_0_ref(X + m * K, scratch, K);
        for (long b = 0; b < nb; ++b) {
            const uint8_t * blk = scratch + b * 34;
            uint16_t        dh  = (uint16_t) (blk[0] | (blk[1] << 8));
            dA[m * nb + b]      = ggml_fp16_to_fp32(dh);
            const int8_t * qs   = (const int8_t *) (blk + 2);
            for (long kk = 0; kk < 32; ++kk) {
                long k = b * 32 + kk, mi = m / 4, ml = m % 4, kf = k / 8, kl = k % 8;
                Apack[mi * 4 * K + kf * 32 + ml * 8 + kl] = qs[kk];
            }
        }
    }
}
// ---- int8-reading matmul family (BYTE-IDENTICAL to sealed q4_0 M7 kernels) -----------------------
// [M3-a] full-range int8 matmul (single-hart baseline / non-epi path).
static void matmul_f32_deref(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                             const float * dW, float * Cf, long M, long N, long K) {
    const long mt = M / 4, nt = N / 4, nb = K / 32, kt = K / 8, fpb = 4;
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = 0; nj < nt; ++nj) {
            const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag);
                for (long r = 0; r < 4; ++r)
                    for (long c = 0; c < 4; ++c) {
                        long m = mi * 4 + r, n = nj * 4 + c;
                        Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float) frag[r * 4 + c];
                    }
            }
        }
    }
}
static void matmul_f32_range_deref(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                   const float * dW, float * Cf, long M, long N, long K,
                                   long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = nj_start; nj < nj_end; ++nj) {
            const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag);
                for (long r = 0; r < 4; ++r)
                    for (long c = 0; c < 4; ++c) {
                        long m = mi * 4 + r, n = nj * 4 + c;
                        Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float) frag[r * 4 + c];
                    }
            }
        }
    }
}
static void matmul_f32_range_deref_epi(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                       const float * dW, float * Cf, long M, long N, long K,
                                       long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = nj_start; nj < nj_end; ++nj) {
            const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
            float acc[16];
            for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag);
                for (long r = 0; r < 4; ++r)
                    for (long c = 0; c < 4; ++c) {
                        long m = mi * 4 + r, n = nj * 4 + c;
                        acc[r * 4 + c] += dA[m * nb + b] * dW[n * nb + b] * (float) frag[r * 4 + c];
                    }
            }
            for (long r = 0; r < 4; ++r)
                for (long c = 0; c < 4; ++c) {
                    long m = mi * 4 + r, n = nj * 4 + c;
                    Cf[m * N + n] = acc[r * 4 + c];
                }
        }
    }
}
static void matmul_f32_range_deref_epi_vec(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                           const float * dW, float * Cf, long M, long N, long K,
                                           long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    const size_t   vl     = __riscv_vsetvl_e32m1(4);
    const ptrdiff_t dwstr = (ptrdiff_t) nb * (ptrdiff_t) sizeof(float);
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = nj_start; nj < nj_end; ++nj) {
            const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
            const float *  dWc  = dW + (long) (nj * 4) * nb;
            vfloat32m1_t acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag);
                vfloat32m1_t vdw = __riscv_vlse32_v_f32m1(dWc + b, dwstr, vl);
                vfloat32m1_t vf0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 0,  vl), vl);
                vfloat32m1_t vf1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 4,  vl), vl);
                vfloat32m1_t vf2 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 8,  vl), vl);
                vfloat32m1_t vf3 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 12, vl), vl);
                vfloat32m1_t p0  = __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 0) * nb + b], vl);
                vfloat32m1_t p1  = __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 1) * nb + b], vl);
                vfloat32m1_t p2  = __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 2) * nb + b], vl);
                vfloat32m1_t p3  = __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 3) * nb + b], vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, p0, vf0, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, p1, vf1, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, p2, vf2, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, p3, vf3, vl);
            }
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + nj * 4, acc0, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + nj * 4, acc1, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + nj * 4, acc2, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + nj * 4, acc3, vl);
        }
    }
}
static void matmul_f32_range_deref_epi_vec_njouter(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                                   const float * dW, float * Cf, long M, long N, long K,
                                                   long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    const size_t   vl     = __riscv_vsetvl_e32m1(4);
    const ptrdiff_t dwstr = (ptrdiff_t) nb * (ptrdiff_t) sizeof(float);
    for (long nj = nj_start; nj < nj_end; ++nj) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        const float *  dWc  = dW + (long) (nj * 4) * nb;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            vfloat32m1_t acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag);
                vfloat32m1_t vdw = __riscv_vlse32_v_f32m1(dWc + b, dwstr, vl);
                vfloat32m1_t vf0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 0,  vl), vl);
                vfloat32m1_t vf1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 4,  vl), vl);
                vfloat32m1_t vf2 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 8,  vl), vl);
                vfloat32m1_t vf3 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 12, vl), vl);
                vfloat32m1_t p0  = __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 0) * nb + b], vl);
                vfloat32m1_t p1  = __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 1) * nb + b], vl);
                vfloat32m1_t p2  = __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 2) * nb + b], vl);
                vfloat32m1_t p3  = __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 3) * nb + b], vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, p0, vf0, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, p1, vf1, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, p2, vf2, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, p3, vf3, vl);
            }
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + nj * 4, acc0, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + nj * 4, acc1, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + nj * 4, acc2, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + nj * 4, acc3, vl);
        }
    }
}
// [M7] WIDE nj-outer vec matmul NJW=2 (reuse A across 2 col-tiles; bit-identical to width-1 njouter).
static void matmul_f32_range_deref_epi_vec_njouter_w2(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                                      const float * dW, float * Cf, long M, long N, long K,
                                                      long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    const size_t   vl      = __riscv_vsetvl_e32m1(4);
    const ptrdiff_t dwstr  = (ptrdiff_t) nb * (ptrdiff_t) sizeof(float);
    const long     bstride = kt * 32;
    long nj = nj_start;
    for (; nj + 2 <= nj_end; nj += 2) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        const float *  dWc0  = dW + (long) (nj * 4) * nb;
        const float *  dWc1  = dW + (long) ((nj + 1) * 4) * nb;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl), a1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t a2 = __riscv_vfmv_v_f_f32m1(0.0f, vl), a3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t b0 = __riscv_vfmv_v_f_f32m1(0.0f, vl), b1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t b2 = __riscv_vfmv_v_f_f32m1(0.0f, vl), b3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (long b = 0; b < nb; ++b) {
                int32_t frag[32];
                vmadot_mac_kloop_w2(Arow + b * fpb * 32, Bcol + b * fpb * 32, bstride, fpb, frag);
                vfloat32m1_t vdw0 = __riscv_vlse32_v_f32m1(dWc0 + b, dwstr, vl);
                vfloat32m1_t vdw1 = __riscv_vlse32_v_f32m1(dWc1 + b, dwstr, vl);
                vfloat32m1_t f0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 0,  vl), vl);
                vfloat32m1_t f1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 4,  vl), vl);
                vfloat32m1_t f2 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 8,  vl), vl);
                vfloat32m1_t f3 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 12, vl), vl);
                vfloat32m1_t g0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 16, vl), vl);
                vfloat32m1_t g1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 20, vl), vl);
                vfloat32m1_t g2 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 24, vl), vl);
                vfloat32m1_t g3 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 28, vl), vl);
                a0 = __riscv_vfmacc_vv_f32m1(a0, __riscv_vfmul_vf_f32m1(vdw0, dA[(mi * 4 + 0) * nb + b], vl), f0, vl);
                a1 = __riscv_vfmacc_vv_f32m1(a1, __riscv_vfmul_vf_f32m1(vdw0, dA[(mi * 4 + 1) * nb + b], vl), f1, vl);
                a2 = __riscv_vfmacc_vv_f32m1(a2, __riscv_vfmul_vf_f32m1(vdw0, dA[(mi * 4 + 2) * nb + b], vl), f2, vl);
                a3 = __riscv_vfmacc_vv_f32m1(a3, __riscv_vfmul_vf_f32m1(vdw0, dA[(mi * 4 + 3) * nb + b], vl), f3, vl);
                b0 = __riscv_vfmacc_vv_f32m1(b0, __riscv_vfmul_vf_f32m1(vdw1, dA[(mi * 4 + 0) * nb + b], vl), g0, vl);
                b1 = __riscv_vfmacc_vv_f32m1(b1, __riscv_vfmul_vf_f32m1(vdw1, dA[(mi * 4 + 1) * nb + b], vl), g1, vl);
                b2 = __riscv_vfmacc_vv_f32m1(b2, __riscv_vfmul_vf_f32m1(vdw1, dA[(mi * 4 + 2) * nb + b], vl), g2, vl);
                b3 = __riscv_vfmacc_vv_f32m1(b3, __riscv_vfmul_vf_f32m1(vdw1, dA[(mi * 4 + 3) * nb + b], vl), g3, vl);
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
    for (; nj < nj_end; ++nj) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        const float *  dWc  = dW + (long) (nj * 4) * nb;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            vfloat32m1_t acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl), acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl), acc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag);
                vfloat32m1_t vdw = __riscv_vlse32_v_f32m1(dWc + b, dwstr, vl);
                vfloat32m1_t vf0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 0,  vl), vl);
                vfloat32m1_t vf1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 4,  vl), vl);
                vfloat32m1_t vf2 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 8,  vl), vl);
                vfloat32m1_t vf3 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + 12, vl), vl);
                acc0 = __riscv_vfmacc_vv_f32m1(acc0, __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 0) * nb + b], vl), vf0, vl);
                acc1 = __riscv_vfmacc_vv_f32m1(acc1, __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 1) * nb + b], vl), vf1, vl);
                acc2 = __riscv_vfmacc_vv_f32m1(acc2, __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 2) * nb + b], vl), vf2, vl);
                acc3 = __riscv_vfmacc_vv_f32m1(acc3, __riscv_vfmul_vf_f32m1(vdw, dA[(mi * 4 + 3) * nb + b], vl), vf3, vl);
            }
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + nj * 4, acc0, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + nj * 4, acc1, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + nj * 4, acc2, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + nj * 4, acc3, vl);
        }
    }
}
// [M7] WIDE nj-outer vec matmul NJW=4 (tail -> w2).
static void matmul_f32_range_deref_epi_vec_njouter_w4(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                                      const float * dW, float * Cf, long M, long N, long K,
                                                      long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    const size_t   vl      = __riscv_vsetvl_e32m1(4);
    const ptrdiff_t dwstr  = (ptrdiff_t) nb * (ptrdiff_t) sizeof(float);
    const long     bstride = kt * 32;
    long nj = nj_start;
    for (; nj + 4 <= nj_end; nj += 4) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        const float *  dWc[4] = { dW + (long) ((nj + 0) * 4) * nb, dW + (long) ((nj + 1) * 4) * nb,
                                  dW + (long) ((nj + 2) * 4) * nb, dW + (long) ((nj + 3) * 4) * nb };
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            vfloat32m1_t t00 = __riscv_vfmv_v_f_f32m1(0.0f, vl), t01 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t t02 = __riscv_vfmv_v_f_f32m1(0.0f, vl), t03 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t t10 = __riscv_vfmv_v_f_f32m1(0.0f, vl), t11 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t t12 = __riscv_vfmv_v_f_f32m1(0.0f, vl), t13 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t t20 = __riscv_vfmv_v_f_f32m1(0.0f, vl), t21 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t t22 = __riscv_vfmv_v_f_f32m1(0.0f, vl), t23 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t t30 = __riscv_vfmv_v_f_f32m1(0.0f, vl), t31 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t t32 = __riscv_vfmv_v_f_f32m1(0.0f, vl), t33 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (long b = 0; b < nb; ++b) {
                int32_t frag[64];
                vmadot_mac_kloop_w4(Arow + b * fpb * 32, Bcol + b * fpb * 32, bstride, fpb, frag);
                vfloat32m1_t w0 = __riscv_vlse32_v_f32m1(dWc[0] + b, dwstr, vl);
                vfloat32m1_t w1 = __riscv_vlse32_v_f32m1(dWc[1] + b, dwstr, vl);
                vfloat32m1_t w2 = __riscv_vlse32_v_f32m1(dWc[2] + b, dwstr, vl);
                vfloat32m1_t w3 = __riscv_vlse32_v_f32m1(dWc[3] + b, dwstr, vl);
                float da0 = dA[(mi * 4 + 0) * nb + b], da1 = dA[(mi * 4 + 1) * nb + b];
                float da2 = dA[(mi * 4 + 2) * nb + b], da3 = dA[(mi * 4 + 3) * nb + b];
                #define FX(off) __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(frag + (off), vl), vl)
                t00 = __riscv_vfmacc_vv_f32m1(t00, __riscv_vfmul_vf_f32m1(w0, da0, vl), FX(0),  vl);
                t01 = __riscv_vfmacc_vv_f32m1(t01, __riscv_vfmul_vf_f32m1(w0, da1, vl), FX(4),  vl);
                t02 = __riscv_vfmacc_vv_f32m1(t02, __riscv_vfmul_vf_f32m1(w0, da2, vl), FX(8),  vl);
                t03 = __riscv_vfmacc_vv_f32m1(t03, __riscv_vfmul_vf_f32m1(w0, da3, vl), FX(12), vl);
                t10 = __riscv_vfmacc_vv_f32m1(t10, __riscv_vfmul_vf_f32m1(w1, da0, vl), FX(16), vl);
                t11 = __riscv_vfmacc_vv_f32m1(t11, __riscv_vfmul_vf_f32m1(w1, da1, vl), FX(20), vl);
                t12 = __riscv_vfmacc_vv_f32m1(t12, __riscv_vfmul_vf_f32m1(w1, da2, vl), FX(24), vl);
                t13 = __riscv_vfmacc_vv_f32m1(t13, __riscv_vfmul_vf_f32m1(w1, da3, vl), FX(28), vl);
                t20 = __riscv_vfmacc_vv_f32m1(t20, __riscv_vfmul_vf_f32m1(w2, da0, vl), FX(32), vl);
                t21 = __riscv_vfmacc_vv_f32m1(t21, __riscv_vfmul_vf_f32m1(w2, da1, vl), FX(36), vl);
                t22 = __riscv_vfmacc_vv_f32m1(t22, __riscv_vfmul_vf_f32m1(w2, da2, vl), FX(40), vl);
                t23 = __riscv_vfmacc_vv_f32m1(t23, __riscv_vfmul_vf_f32m1(w2, da3, vl), FX(44), vl);
                t30 = __riscv_vfmacc_vv_f32m1(t30, __riscv_vfmul_vf_f32m1(w3, da0, vl), FX(48), vl);
                t31 = __riscv_vfmacc_vv_f32m1(t31, __riscv_vfmul_vf_f32m1(w3, da1, vl), FX(52), vl);
                t32 = __riscv_vfmacc_vv_f32m1(t32, __riscv_vfmul_vf_f32m1(w3, da2, vl), FX(56), vl);
                t33 = __riscv_vfmacc_vv_f32m1(t33, __riscv_vfmul_vf_f32m1(w3, da3, vl), FX(60), vl);
                #undef FX
            }
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + (nj + 0) * 4, t00, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + (nj + 0) * 4, t01, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + (nj + 0) * 4, t02, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + (nj + 0) * 4, t03, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + (nj + 1) * 4, t10, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + (nj + 1) * 4, t11, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + (nj + 1) * 4, t12, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + (nj + 1) * 4, t13, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + (nj + 2) * 4, t20, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + (nj + 2) * 4, t21, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + (nj + 2) * 4, t22, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + (nj + 2) * 4, t23, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 0) * N + (nj + 3) * 4, t30, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 1) * N + (nj + 3) * 4, t31, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 2) * N + (nj + 3) * 4, t32, vl);
            __riscv_vse32_v_f32m1(Cf + (long) (mi * 4 + 3) * N + (nj + 3) * 4, t33, vl);
        }
    }
    if (nj < nj_end)
        matmul_f32_range_deref_epi_vec_njouter_w2(Apack, dA, Bdec_all, dW, Cf, M, N, K, nj, nj_end);
}
// ---- timing-only probes (PROF phase; not on correctness/e2e path) --------------------------------
static void matmul_noepi_njouter(const int8_t * Apack, const float * /*dA*/, const int8_t * Bdec_all,
                                 const float * /*dW*/, float * /*Cf*/, long M, long N, long K,
                                 long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    (void) N;
    for (long nj = nj_start; nj < nj_end; ++nj) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            for (long b = 0; b < nb; ++b) { int32_t frag[16]; vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag); }
        }
    }
}
static void matmul_noepi_njouter_w2(const int8_t * Apack, const float * /*dA*/, const int8_t * Bdec_all,
                                    const float * /*dW*/, float * /*Cf*/, long M, long N, long K,
                                    long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    const long bstride = kt * 32;
    (void) N;
    long nj = nj_start;
    for (; nj + 2 <= nj_end; nj += 2) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            for (long b = 0; b < nb; ++b) { int32_t frag[32]; vmadot_mac_kloop_w2(Arow + b * fpb * 32, Bcol + b * fpb * 32, bstride, fpb, frag); }
        }
    }
    for (; nj < nj_end; ++nj) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            for (long b = 0; b < nb; ++b) { int32_t frag[16]; vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag); }
        }
    }
}
static void matmul_noepi_l1(const int8_t * /*Apack*/, const float * /*dA*/, const int8_t * /*Bdec_all*/,
                            const float * /*dW*/, float * /*Cf*/, long M, long N, long K,
                            long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, fpb = 4;
    (void) N;
    alignas(64) int8_t sA[128]; alignas(64) int8_t sB[128];
    for (int i = 0; i < 128; ++i) { sA[i] = (int8_t) (i & 7); sB[i] = (int8_t) ((i * 3) & 7); }
    for (long nj = nj_start; nj < nj_end; ++nj)
        for (long mi = 0; mi < mt; ++mi)
            for (long b = 0; b < nb; ++b) { int32_t frag[16]; vmadot_mac_kloop(sA, sB, fpb, frag); }
}
static void matmul_noepi_l1_w2(const int8_t * /*Apack*/, const float * /*dA*/, const int8_t * /*Bdec_all*/,
                               const float * /*dW*/, float * /*Cf*/, long M, long N, long K,
                               long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, fpb = 4;
    (void) N;
    alignas(64) int8_t sA[128]; alignas(64) int8_t sB[256];
    for (int i = 0; i < 128; ++i) sA[i] = (int8_t) (i & 7);
    for (int i = 0; i < 256; ++i) sB[i] = (int8_t) ((i * 3) & 7);
    for (long nj = nj_start; nj + 2 <= nj_end; nj += 2)
        for (long mi = 0; mi < mt; ++mi)
            for (long b = 0; b < nb; ++b) { int32_t frag[32]; vmadot_mac_kloop_w2(sA, sB, 128, fpb, frag); }
}

// ---------------- int8 gather cache + shared MT store + [PROF] accounting --------------------------
struct PackedWDec { std::vector<int8_t> Bdec; std::vector<float> dW; long N; long K; };
static std::unordered_map<const void *, PackedWDec> g_wcache_dec;
struct MTWork { const int8_t * Apack; const float * dA; const int8_t * Bdec; const float * dW;
                float * Cfp; const float * X; long Mp; long Ml; long Nl; long Kl; };
static MTWork              g_mtwork = {};
static std::vector<int8_t> g_Apack;
static std::vector<float>  g_dA;
static std::vector<float>  g_Cfp;
static uint64_t g_cyc_gather = 0, g_cyc_quant = 0, g_cyc_matmul = 0, g_cyc_alloc = 0, g_cyc_copyback = 0;
static uint64_t g_n_calls = 0, g_n_gather_runs = 0, g_n_cache_hits = 0;
static bool     g_env_read = false, g_cache = false, g_threads = false, g_prof = false, g_prof_reg = false;
static bool     g_deref = false, g_deref_epi = false, g_parsetup = false;
static bool     g_epivec = false, g_mmprof_noepi = false, g_njouter = false, g_mmprof_l1 = false;
static int      g_tilew = 0;
static inline uint64_t nowns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}
static inline void run_deref_epi_tile(const MTWork & w, long Mp, long Nl, long Kl, long njs, long nje) {
    if (g_mmprof_l1) {
        if (g_njouter && g_tilew == 2)      matmul_noepi_l1_w2(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
        else                                matmul_noepi_l1(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
    } else if (g_mmprof_noepi) {
        if (g_njouter && g_tilew == 2)      matmul_noepi_njouter_w2(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
        else                                matmul_noepi_njouter(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
    } else if (g_epivec) {
        if (g_njouter && g_tilew == 2)      matmul_f32_range_deref_epi_vec_njouter_w2(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
        else if (g_njouter && g_tilew == 4) matmul_f32_range_deref_epi_vec_njouter_w4(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
        else if (g_njouter)                 matmul_f32_range_deref_epi_vec_njouter(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
        else                                matmul_f32_range_deref_epi_vec(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
    } else {
        matmul_f32_range_deref_epi(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
    }
}
static void prof_dump() {
    uint64_t tot = g_cyc_gather + g_cyc_quant + g_cyc_matmul + 1;
    fprintf(stderr,
            "[TCRV-Q80-PROF] cache=%d threads=%d deref=%d deref_epi=%d parsetup=%d epivec=%d "
            "njouter=%d tilew=%d mmprof_noepi=%d mmprof_l1=%d calls=%llu gather_runs=%llu cache_hits=%llu | "
            "ns_gather=%llu ns_quant=%llu ns_alloc=%llu ns_copyback=%llu ns_matmul=%llu | "
            "gather_share=%.4f quant_share=%.4f matmul_share=%.4f\n",
            (int) g_cache, (int) g_threads, (int) g_deref, (int) g_deref_epi, (int) g_parsetup,
            (int) g_epivec, (int) g_njouter, g_tilew, (int) g_mmprof_noepi, (int) g_mmprof_l1,
            (unsigned long long) g_n_calls, (unsigned long long) g_n_gather_runs,
            (unsigned long long) g_n_cache_hits, (unsigned long long) g_cyc_gather,
            (unsigned long long) g_cyc_quant, (unsigned long long) g_cyc_alloc,
            (unsigned long long) g_cyc_copyback, (unsigned long long) g_cyc_matmul,
            (double) g_cyc_gather / (double) tot, (double) g_cyc_quant / (double) tot,
            (double) g_cyc_matmul / (double) tot);
}
static inline void env_read_once() {
    if (g_env_read) return;
    g_env_read = true;
    g_cache        = std::getenv("TCRV_IME_Q80_CACHE")        != nullptr;
    g_threads      = std::getenv("TCRV_IME_Q80_THREADS")      != nullptr;
    g_deref        = std::getenv("TCRV_IME_Q80_DEREF")        != nullptr;
    g_deref_epi    = std::getenv("TCRV_IME_Q80_DEREF_EPI")    != nullptr;
    g_parsetup     = std::getenv("TCRV_IME_Q80_PARSETUP")     != nullptr;
    g_epivec       = std::getenv("TCRV_IME_Q80_EPIVEC")       != nullptr;
    g_njouter      = std::getenv("TCRV_IME_Q80_NJOUTER")      != nullptr;
    g_mmprof_noepi = std::getenv("TCRV_IME_Q80_MMPROF_NOEPI") != nullptr;
    g_mmprof_l1    = std::getenv("TCRV_IME_Q80_MMPROF_L1")    != nullptr;
    { const char * tw = std::getenv("TCRV_IME_Q80_TILEW"); g_tilew = tw ? atoi(tw) : 0;
      if (g_tilew != 2 && g_tilew != 4) g_tilew = 0; }
    g_prof         = std::getenv("TCRV_IME_Q80_PROF")         != nullptr;
    if (g_prof && !g_prof_reg) { g_prof_reg = true; atexit(prof_dump); }
}
}  // namespace tcrv_q80

class tcrv_q8_0_tensor_traits : public tensor_traits_base {
    bool work_size(int /*n_threads*/, const ggml_tensor * /*op*/, size_t & /*size*/) override { return false; }

    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        memcpy(t->data, data, data_size);  // NATIVE passthrough (34B/block preserved)
        return 0;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;
        if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32) {
            return false;
        }
        const int64_t K = src0->ne[0];
        const int64_t N = src0->ne[1];
        const int64_t M = src1->ne[1];
        if (M <= 1 || N % 4 != 0 || K % 32 != 0) {
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
        if (src0->nb[1] != (size_t) (K / QK8_0) * sizeof(block_q8_0)) {
            return false;
        }
        const int  ith = params->ith;
        const int  nth = params->nth;
        const long Ml = (long) M, Nl = (long) N, Kl = (long) K;
        const long Mp = (Ml + 3) / 4 * 4;
        float *    Cf = (float *) dst->data;

        if (ith == 0) {
            tcrv_q80::env_read_once();
            tcrv_q80::g_n_calls++;
            memset(Cf, 0, (size_t) Ml * Nl * sizeof(float));

            const int8_t * Bdec_p = nullptr;
            const float *  dW_p   = nullptr;
            std::vector<int8_t> Bdec_local;
            std::vector<float>  dW_local;
            // q8_0 int8-direct: DEREF==CACHE (both = load-once int8 gather). Baseline = per-call gather.
            if (tcrv_q80::g_cache || tcrv_q80::g_deref) {
                auto it = tcrv_q80::g_wcache_dec.find(src0->data);
                if (it == tcrv_q80::g_wcache_dec.end() || it->second.N != Nl || it->second.K != Kl) {
                    tcrv_q80::PackedWDec pw; pw.N = Nl; pw.K = Kl;
                    pw.Bdec.assign((size_t) (Nl / 4) * (Kl / 8) * 32, 0);
                    pw.dW.assign((size_t) Nl * (Kl / 32), 0.0f);
                    uint64_t t0 = tcrv_q80::nowns();
                    tcrv_q80::repack_int8_weight((const uint8_t *) src0->data, pw.Bdec.data(), pw.dW.data(), Nl, Kl);
                    tcrv_q80::g_cyc_gather += tcrv_q80::nowns() - t0;
                    tcrv_q80::g_n_gather_runs++;
                    it = tcrv_q80::g_wcache_dec.emplace(src0->data, std::move(pw)).first;
                } else {
                    tcrv_q80::g_n_cache_hits++;
                }
                Bdec_p = it->second.Bdec.data();
                dW_p   = it->second.dW.data();
            } else {
                Bdec_local.assign((size_t) (Nl / 4) * (Kl / 8) * 32, 0);
                dW_local.assign((size_t) Nl * (Kl / 32), 0.0f);
                uint64_t t0 = tcrv_q80::nowns();
                tcrv_q80::repack_int8_weight((const uint8_t *) src0->data, Bdec_local.data(), dW_local.data(), Nl, Kl);
                tcrv_q80::g_cyc_gather += tcrv_q80::nowns() - t0;
                tcrv_q80::g_n_gather_runs++;
                Bdec_p = Bdec_local.data();
                dW_p   = dW_local.data();
            }

            const bool epi_full = tcrv_q80::g_deref_epi;   // deref implied (int8-direct)
            uint64_t ta = tcrv_q80::nowns();
            tcrv_q80::g_Apack.assign((size_t) Mp * Kl, 0);
            tcrv_q80::g_dA.assign((size_t) Mp * (Kl / 32), 0.0f);
            if (tcrv_q80::g_parsetup && epi_full) tcrv_q80::g_Cfp.resize((size_t) Mp * Nl);
            else                                  tcrv_q80::g_Cfp.assign((size_t) Mp * Nl, 0.0f);
            tcrv_q80::g_cyc_alloc += tcrv_q80::nowns() - ta;

            const bool par_quant = tcrv_q80::g_parsetup && tcrv_q80::g_threads;
            if (!par_quant) {
                std::vector<uint8_t> scratch((size_t) (Kl / 32) * 34);
                uint64_t t1 = tcrv_q80::nowns();
                tcrv_q80::quant_pack_act((const float *) src1->data, tcrv_q80::g_Apack.data(),
                                         tcrv_q80::g_dA.data(), Ml, Kl, scratch.data());
                tcrv_q80::g_cyc_quant += tcrv_q80::nowns() - t1;
            }

            tcrv_q80::g_mtwork.Apack = tcrv_q80::g_Apack.data();
            tcrv_q80::g_mtwork.dA    = tcrv_q80::g_dA.data();
            tcrv_q80::g_mtwork.Bdec  = Bdec_p;
            tcrv_q80::g_mtwork.dW    = dW_p;
            tcrv_q80::g_mtwork.Cfp   = tcrv_q80::g_Cfp.data();
            tcrv_q80::g_mtwork.X     = (const float *) src1->data;
            tcrv_q80::g_mtwork.Mp    = Mp;
            tcrv_q80::g_mtwork.Ml    = Ml;
            tcrv_q80::g_mtwork.Nl    = Nl;
            tcrv_q80::g_mtwork.Kl    = Kl;

            if (!tcrv_q80::g_threads) {
                const long nt = Nl / 4;
                uint64_t t2 = tcrv_q80::nowns();
                if (tcrv_q80::g_deref_epi) {
                    tcrv_q80::run_deref_epi_tile(tcrv_q80::g_mtwork, Mp, Nl, Kl, 0, nt);
                } else {
                    tcrv_q80::matmul_f32_deref(tcrv_q80::g_mtwork.Apack, tcrv_q80::g_mtwork.dA,
                                               tcrv_q80::g_mtwork.Bdec, tcrv_q80::g_mtwork.dW,
                                               tcrv_q80::g_mtwork.Cfp, Mp, Nl, Kl);
                }
                tcrv_q80::g_cyc_matmul += tcrv_q80::nowns() - t2;
                memcpy(Cf, tcrv_q80::g_Cfp.data(), (size_t) Ml * Nl * sizeof(float));
            }

            static bool announced = false;
            if (!announced) {
                announced = true;
                fprintf(stderr,
                        "[TCRV-IME-Q80-BRIDGE] routed real q8_0 PREFILL mul_mat through tcrv IME kernel: "
                        "M=%ld N=%ld K=%ld (real vmadot 0xe210312b int8-direct) cache=%d threads=%d deref=%d deref_epi=%d parsetup=%d epivec=%d njouter=%d tilew=%d nth=%d\n",
                        Ml, Nl, Kl, (int) tcrv_q80::g_cache, (int) tcrv_q80::g_threads,
                        (int) tcrv_q80::g_deref, (int) tcrv_q80::g_deref_epi, (int) tcrv_q80::g_parsetup,
                        (int) tcrv_q80::g_epivec, (int) tcrv_q80::g_njouter, tcrv_q80::g_tilew, nth);
            }
        }
        ggml_barrier(params->threadpool);  // barrier #1

        if (tcrv_q80::g_parsetup && tcrv_q80::g_threads) {
            const long Mlq = tcrv_q80::g_mtwork.Ml;
            const long Klq = tcrv_q80::g_mtwork.Kl;
            long       per = (Mlq + nth - 1) / nth;
            per = (per + 3) & ~3L;
            long ms = (long) ith * per;
            long me = ms + per;
            if (me > Mlq) me = Mlq;
            uint64_t tq = (ith == 0) ? tcrv_q80::nowns() : 0;
            if (ms < me) {
                std::vector<uint8_t> scratch((size_t) (Klq / 32) * 34);
                tcrv_q80::quant_pack_act_range(tcrv_q80::g_mtwork.X, tcrv_q80::g_Apack.data(),
                                               tcrv_q80::g_dA.data(), Mlq, Klq, scratch.data(), ms, me);
            }
            ggml_barrier(params->threadpool);  // barrier #1b
            if (ith == 0) tcrv_q80::g_cyc_quant += tcrv_q80::nowns() - tq;
        }

        if (tcrv_q80::g_threads) {
            const long nt  = tcrv_q80::g_mtwork.Nl / 4;
            long       per = (nt + nth - 1) / nth;
            per = (per + 3) & ~3L;
            long njs = (long) ith * per;
            long nje = njs + per;
            if (nje > nt) nje = nt;
            uint64_t t2 = (ith == 0) ? tcrv_q80::nowns() : 0;
            if (njs < nje) {
                if (tcrv_q80::g_deref_epi) {
                    tcrv_q80::run_deref_epi_tile(tcrv_q80::g_mtwork, tcrv_q80::g_mtwork.Mp,
                                                 tcrv_q80::g_mtwork.Nl, tcrv_q80::g_mtwork.Kl, njs, nje);
                } else {
                    tcrv_q80::matmul_f32_range_deref(tcrv_q80::g_mtwork.Apack, tcrv_q80::g_mtwork.dA,
                                                     tcrv_q80::g_mtwork.Bdec, tcrv_q80::g_mtwork.dW,
                                                     tcrv_q80::g_mtwork.Cfp, tcrv_q80::g_mtwork.Mp,
                                                     tcrv_q80::g_mtwork.Nl, tcrv_q80::g_mtwork.Kl, njs, nje);
                }
            }
            ggml_barrier(params->threadpool);  // barrier #2
            if (ith == 0) {
                tcrv_q80::g_cyc_matmul += tcrv_q80::nowns() - t2;
                uint64_t tc = tcrv_q80::nowns();
                memcpy(Cf, tcrv_q80::g_mtwork.Cfp, (size_t) Ml * Nl * sizeof(float));
                tcrv_q80::g_cyc_copyback += tcrv_q80::nowns() - tc;
            }
        }
        ggml_barrier(params->threadpool);  // barrier #3
        return true;
    }
};

static tcrv_q8_0_tensor_traits tcrv_q8_0_bridge;

'''
s = s.replace(cls_anchor, block + cls_anchor, 1)

# --- 3. env-gated selection in get_optimal_repack_type Q8_0 case -----------
sel_anchor = ("        case GGML_TYPE_Q8_0:\n"
              "            {\n"
              "#if defined(RISCV64_SPACEMIT_IME2)\n")
assert s.count(sel_anchor) == 1, "expected exactly one Q8_0 get_optimal anchor"
sel_new = ("        case GGML_TYPE_Q8_0:\n"
           "            {\n"
           "                if (std::getenv(\"TCRV_IME_Q80_BRIDGE\") && cur->ne[1] % 4 == 0 && cur->ne[0] % 32 == 0) {\n"
           "                    return &ggml::cpu::riscv64_spacemit::tcrv_q8_0_bridge;\n"
           "                }\n"
           "#if defined(RISCV64_SPACEMIT_IME2)\n")
s = s.replace(sel_anchor, sel_new, 1)

open(F, "w").write(s)

# --- post-write verification ----------------------------------------------
t = open(F).read()
assert "class tcrv_q8_0_tensor_traits" in t, "trait class missing after write"
assert "static tcrv_q8_0_tensor_traits tcrv_q8_0_bridge;" in t, "static instance missing"
assert "return &ggml::cpu::riscv64_spacemit::tcrv_q8_0_bridge;" in t, "selection missing"
assert "repack_int8_weight" in t, "int8 gather missing"
assert "g_wcache_dec" in t, "deref cache missing"
assert "matmul_f32_range_deref_epi_vec_njouter_w2" in t, "[M7] wide nj-outer NJW=2 missing"
assert "matmul_f32_range_deref_epi_vec_njouter_w4" in t, "[M7] wide nj-outer NJW=4 missing"
assert "vmadot_mac_kloop_w2" in t and "vmadot_mac_kloop_w4" in t, "[M7] wide vmadot leaf missing"
assert "TCRV_IME_Q80_TILEW" in t and "g_tilew" in t, "[M7] tilew gate/flag missing"
assert "TCRV_IME_Q80_EPIVEC" in t and "TCRV_IME_Q80_NJOUTER" in t, "[M5/M6] gates missing"
assert "TCRV_IME_Q80_MMPROF_L1" in t and "TCRV_IME_Q80_MMPROF_NOEPI" in t, "prof gates missing"
assert ("barrier #1" in t and "barrier #2" in t and "barrier #3" in t), "barriers missing"
assert "TCRV-Q80-PROF" in t, "[PROF] dump missing"
assert t.count("[TCRV-IME-Q80-BRIDGE]") >= 2, "markers missing"
print("patched OK; bridge_markers=%d" % t.count("TCRV-IME-Q80-BRIDGE") +
      "; int8_gather=1; deref_cache=1; epivec=1; njouter=1; tilew=1; wide_w2=1; wide_w4=1; "
      "mmprof_noepi=1; mmprof_l1=1; trait=1; selection=1")
