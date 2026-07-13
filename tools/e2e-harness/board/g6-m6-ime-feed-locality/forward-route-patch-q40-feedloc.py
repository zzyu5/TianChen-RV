#!/usr/bin/env python3
# G6-A M6 (B-feed locality) patch for vendor spacemit ime.cpp -- q4_0 IME bridge.
#
# SUPERSET of the M5 (matmul-internal) patch. Everything M1..M5 is BYTE-IDENTICAL. M5's profile
# put matmul-internal at vmadot+feed 68.5% (5.763s) / scalar-epilogue 31.5% and vectorized the
# epilogue (matmul 1.172x, e2e 1.142x). The now-dominant residual is vmadot+feed. M6 attacks the
# FEED half (bit-exact, no emitter): the pre-dequantized weight B (this hart's column slice ~1MB
# > L2) is re-streamed from DRAM once per mi-tile under the mi-outer loop; interchanging to
# nj-OUTER streams each nj B-tile (4*K int8 = 8KB, L1) ONCE and reuses it across all mi. It also
# adds a compute-isolation probe (fixed L1 scratch) so the feed share of the 5.763s is measured
# BEFORE committing (if feed is small -> compute-bound -> loop-interchange ROI is limited and the
# residual points at [b] vmadot array-utilization = emitter vmadot-tiling, coordinated separately).
#
#   [M5 matmul-internal profiling] (retained):
#     TCRV_IME_Q40_MMPROF_NOEPI   : run vmadot but SKIP the fold  -> ns_matmul ~= vmadot + B/A feed
#     TCRV_IME_Q40_MMPROF_NOMADOT : SKIP vmadot, run the fold     -> ns_matmul ~= scalar epilogue
#
#   [M6 optimization] TCRV_IME_Q40_NJOUTER (requires EPIVEC): interchange the deref+epi vec matmul
#     to nj-OUTER / mi-INNER. ONLY the tile-visitation ORDER changes -- each output tile (mi,nj)
#     still accumulates over b in the SAME order into its own register acc, single writer per
#     Cf[m,n] -> BIT-FOR-BIT the M5 (mi-outer vec) output. Cuts this hart's B DRAM traffic ~mt-fold.
#   [M6 profiling]:
#     TCRV_IME_Q40_MMPROF_L1      : same vmadot CALL COUNT as noepi, A/B from a FIXED L1 scratch
#                                    -> ns_matmul ~= vmadot COMPUTE only. feed = noepi - noepi_l1.
#     (NOEPI honors NJOUTER too -> noepi_njouter measures vmadot+feed under B-locality.)
#
#   [M5 matmul-internal decomposition context] M4 collapsed the serial setup floor; the parallel
#   matmul (~8.4s) is 95.9% of the ith==0 critical path. M5 profiling variants (below) decompose
#   ns_matmul into vmadot+feed vs scalar-epilogue:
#
#   [X-0 profiling] two timing-only matmul variants (gated, run ONLY in the PROF phase, never
#   on the correctness/e2e path) decompose ns_matmul into vmadot+feed vs scalar-epilogue:
#     TCRV_IME_Q40_MMPROF_NOEPI   : run vmadot (volatile asm streams A/B, writes frag) but SKIP
#                                    the scalar fold      -> ns_matmul ~= vmadot compute + B/A feed
#     TCRV_IME_Q40_MMPROF_NOMADOT : SKIP vmadot, run the scalar fold on a fixed frag
#                                    -> ns_matmul ~= scalar epilogue (dA/dW feed + fmul/fmadd + Cf)
#     epilogue-marginal = full - NOEPI ; vmadot+feed = NOEPI.
#
#   [M5 optimization] TCRV_IME_Q40_EPIVEC (requires DEREF + DEREF_EPI): vectorize the scalar
#   4x4 epilogue fold 4-WIDE ACROSS THE 4 INDEPENDENT COLUMNS of each output row. Each output
#   element Cf[m,n] accumulates over blocks b in the SAME left-to-right order as the scalar
#   register epilogue; only the 4 columns of a row are folded in parallel SIMD lanes (columns
#   are independent -> no cross-column reassociation -> bit-exact). The scalar epilogue compiles
#   (clang-18 -O3) to:  p = fmul(dA,dW) ; f = fcvt.s.w(frag) ; acc = fmadd(p, f, acc)  -- i.e. a
#   plain multiply of the two scales THEN a *fused* multiply-add of (p*frag + acc). The vector
#   form mirrors this EXACTLY per lane:
#       vp   = vfmul_vf (vdw, dA[m,b])      // p = dA * dW[c]          (commutative, bit-exact)
#       vf   = vfcvt_f_x(frag[r*4 .. +3])   // f = (float)frag[c]      (RNE, same as (float) cast)
#       vacc = vfmacc_vv(vacc, vp, vf)      // acc = fma(p, f, acc)    (single-rounding, == fmadd.s)
#   so every lane executes the identical fmul + fcvt + fma sequence in the identical b-order ->
#   the vectorized output is bit-for-bit the scalar output. vl=4 (one row); 4 row-accumulators
#   kept in registers across the b loop; dW[n,b] loaded once per b via a strided load (stride
#   = nb floats). VLEN-agnostic (vl=4 <= any VLEN>=128 at e32m1).
#
# All M5 additions are orchestration + a bit-exact SIMD reshape of an existing scalar fold. NO
# vmadot kernel byte changes (objdump vmadot count unchanged vs M4 = 36; the vec/prof matmuls
# each inline one vmadot copy so the count rises by the number of new matmul variants that call
# it -- see the run script's expectation line). All gates default OFF -> byte-identical to the M4
# path (-> M3 -> M2 -> M1 -> sealed bridge). EPIVEC without DEREF+DEREF_EPI is a no-op.
# Fully reversible (driver restores src/.o/.so; md5 double-proof).
#
# Env gates (additive over M1/M2/M3/M4):
#   TCRV_IME_Q40_BRIDGE     : route q4_0 prefill mul_mat through the tcrv IME kernel
#   TCRV_IME_Q40_CACHE      : [M1] cache the repacked (nibble) weight (M2 path only)
#   TCRV_IME_Q40_THREADS    : [M2] column-tile multithread the matmul
#   TCRV_IME_Q40_DEREF      : [M3-a] cache the DEQUANTIZED int8 weight; hot matmul reads it
#   TCRV_IME_Q40_DEREF_EPI  : [M3-b] register-accumulated epilogue (requires DEREF)
#   TCRV_IME_Q40_PARSETUP   : [M4]  parallelize the ith==0 activation quant + lighten allocs
#   TCRV_IME_Q40_EPIVEC     : [M5]  4-wide (across-columns) vectorized epilogue fold (bit-exact)
#   TCRV_IME_Q40_NJOUTER    : [M6]  nj-outer/mi-inner tile order for B-feed locality (bit-exact)
#   TCRV_IME_Q40_MMPROF_NOEPI   : [M5-prof] vmadot only (skip fold)   -- timing-only
#   TCRV_IME_Q40_MMPROF_NOMADOT : [M5-prof] fold only (skip vmadot)   -- timing-only
#   TCRV_IME_Q40_MMPROF_L1      : [M6-prof] vmadot from fixed L1 scratch (compute only) -- timing-only
#   TCRV_IME_Q40_PROF       : per-stage ns split (repack/dequant-populate/quant/alloc/copyback/matmul)
import sys

F = "ggml/src/ggml-cpu/spacemit/ime.cpp"
s = open(F).read()

# --- 1. includes -----------------------------------------------------------
inc_anchor = '#include <cstdio>  // for GGML_ASSERT\n'
assert inc_anchor in s, "cstdio include anchor not found"
add_inc = ('#include <cstdlib>       // [TCRV-Q40] std::getenv/atexit\n'
           '#include <cstring>       // [TCRV-Q40] memcpy/memset\n'
           '#include <ctime>         // [TCRV-Q40-M1] clock_gettime prof\n'
           '#include <vector>        // [TCRV-Q40-M2] shared work vectors\n'
           '#include <unordered_map> // [TCRV-Q40-M1] weight repack cache\n')
if "[TCRV-Q40] std::getenv" not in s:
    s = s.replace(inc_anchor, inc_anchor + add_inc, 1)

# --- 2. helpers + trait class + static instance (inside namespace) ---------
cls_anchor = "// Impl By IME1\n"
assert s.count(cls_anchor) == 1, "expected exactly one '// Impl By IME1' anchor"

block = r'''// ==================== [TCRV-IME-Q40-BRIDGE] session-3 forward routing + [M1] repack cache + [M2] multithread + [M3] de-reference-form + [M4] parallel-setup + [M5] matmul-internal (vec epilogue + prof) ====================
// Env-gated (TCRV_IME_Q40_BRIDGE) parallel tcrv tensor_traits. repack() is a NATIVE
// passthrough so any op we do not intercept falls through to the correct ggml default
// on native bytes. compute_forward routes real q4_0 PREFILL mul_mat traffic through our
// fragment-major scale-fold IME kernel (real vmadot 0xe210312b). The kernel primitives
// below are byte-identical to the sealed session-2 board UT (g5m3_bridge_ut.c).
// [M1] adds a SIDE weight-repack (nibble) cache + per-stage profiling (orchestration only).
// [M2] adds column-tile multithreading of the matmul (orchestration only; the inner
//      vmadot/dequant kernel body is unchanged and each C element has a single writer).
// [M3] de-reference-form (orchestration only; kernel bytes + arithmetic unchanged):
//      (a) cache the DEQUANTIZED int8 weight once and feed vmadot pre-dequantized bytes;
//      (b) accumulate the 4x4 output tile in a register-resident local, assign Cf once.
// [M4] parallel-setup (orchestration only; kernel bytes + arithmetic unchanged):
//      (a) slice the ith==0 activation quant across harts; (b) drop redundant g_Cfp zero-fill.
// [M5] matmul-internal (orchestration + bit-exact SIMD reshape of the M3-b scalar fold):
//      (vec) matmul_f32_range_deref_epi_vec folds the 4x4 tile 4-wide across the 4 independent
//            output columns using vfmul_vf(dA,dW) + vfcvt.f.x(frag) + vfmacc.vv (fused), the
//            EXACT per-lane sequence the scalar epilogue compiles to -> bit-identical output.
//      (prof) two timing-only variants decompose ns_matmul (vmadot+feed vs scalar epilogue).
extern "C" void quantize_row_q8_0_ref(const float * x, void * y, int64_t k);

namespace tcrv_q40 {
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
static inline void dequant_fragment(const uint8_t * blk, int8_t * out) {
    const uint8_t * qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        out[j]      = (int8_t) ((int) (qs[j] & 0x0F) - 8);
        out[j + 16] = (int8_t) ((int) (qs[j] >> 4) - 8);
    }
}
// bridge #4: ggml NATIVE q4_0 (row-major N x nb 18B) -> fragment-major Bnib + per-(col,block) dW.
static void repack_weight(const uint8_t * wq, uint8_t * Bnib, float * dW, long N, long K) {
    const long nb = K / 32, kt = K / 8, q40b = 18;
    memset(Bnib, 0, (size_t) (N / 4) * kt * q40b);
    for (long n = 0; n < N; ++n)
        for (long b = 0; b < nb; ++b) {
            const uint8_t * src = wq + (n * nb + b) * 18;
            uint16_t        dh  = (uint16_t) (src[0] | (src[1] << 8));
            dW[n * nb + b]      = ggml_fp16_to_fp32(dh);
            const uint8_t * qs_src = src + 2;
            for (long kk = 0; kk < 32; ++kk) {
                long      k = b * 32 + kk, nj = n / 4, nl = n % 4, kf = k / 8, kl = k % 8;
                uint8_t * blk = Bnib + (nj * kt + kf) * q40b;
                blk[0]        = src[0];
                blk[1]        = src[1];
                uint8_t * qs  = blk + 2;
                long      idx = nl * 8 + kl;
                int       hi   = kk >= 16;
                uint8_t   byte = qs_src[kk % 16];
                int       v4   = hi ? (byte >> 4) : (byte & 0x0F);
                if (idx < 16)
                    qs[idx] = (uint8_t) ((qs[idx] & 0xF0) | (v4 & 0x0F));
                else
                    qs[idx - 16] = (uint8_t) ((qs[idx - 16] & 0x0F) | ((v4 & 0x0F) << 4));
            }
        }
}
// bridge #3: ggml f32 activation (M x K) -> q8_0 (quantize_row_q8_0_ref) -> Apack + dA.
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
// [M4-a] row-range slice of quant_pack_act (loop body BYTE-IDENTICAL; only m bounds change).
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
// scale-fold f32 GEMM (bridge #1 + #2): C[m,n] = sum_b dA*dW*(int32 vmadot partial).
static void matmul_f32(const int8_t * Apack, const float * dA, const uint8_t * Bnib, const float * dW,
                       float * Cf, long M, long N, long K) {
    const long mt = M / 4, nt = N / 4, nb = K / 32, kt = K / 8, q40b = 18, fpb = 4;
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = 0; nj < nt; ++nj) {
            const uint8_t * Bcol = Bnib + (long) nj * kt * q40b;
            for (long b = 0; b < nb; ++b) {
                int8_t Bdec[128];
                for (long f = 0; f < fpb; ++f)
                    dequant_fragment(Bcol + (b * fpb + f) * q40b, Bdec + f * 32);
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bdec, fpb, frag);
                for (long r = 0; r < 4; ++r)
                    for (long c = 0; c < 4; ++c) {
                        long m = mi * 4 + r, n = nj * 4 + c;
                        Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float) frag[r * 4 + c];
                    }
            }
        }
    }
}
// [M2] column-tile slice of matmul_f32 (byte-identical body; only nj bounds change).
static void matmul_f32_range(const int8_t * Apack, const float * dA, const uint8_t * Bnib, const float * dW,
                             float * Cf, long M, long N, long K, long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, q40b = 18, fpb = 4;
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = nj_start; nj < nj_end; ++nj) {
            const uint8_t * Bcol = Bnib + (long) nj * kt * q40b;
            for (long b = 0; b < nb; ++b) {
                int8_t Bdec[128];
                for (long f = 0; f < fpb; ++f)
                    dequant_fragment(Bcol + (b * fpb + f) * q40b, Bdec + f * 32);
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bdec, fpb, frag);
                for (long r = 0; r < 4; ++r)
                    for (long c = 0; c < 4; ++c) {
                        long m = mi * 4 + r, n = nj * 4 + c;
                        Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float) frag[r * 4 + c];
                    }
            }
        }
    }
}

// ---------------- [M3-a] one-time dequantized-B populate (compose VERBATIM kernels) ------------------
static void repack_dequant_weight(const uint8_t * wq, int8_t * Bdec_all, float * dW, long N, long K,
                                  uint8_t * nib_tmp) {
    const long nt = N / 4, kt = K / 8, q40b = 18;
    repack_weight(wq, nib_tmp, dW, N, K);
    for (long nj = 0; nj < nt; ++nj)
        for (long kf = 0; kf < kt; ++kf)
            dequant_fragment(nib_tmp + (nj * kt + kf) * q40b, Bdec_all + (nj * kt + kf) * 32);
}
// [M3-a] column-tile matmul reading the PRE-DEQUANTIZED int8 B (no per-call dequant).
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
// [M3-a + M3-b] pre-dequantized B + register-accumulated (scalar) epilogue.
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
// [M5] pre-dequantized B + register-accumulated epilogue folded 4-WIDE ACROSS THE 4 INDEPENDENT
// OUTPUT COLUMNS of each row. For each output row r (fixed m = mi*4+r), the four column
// accumulators acc[r*4+0..3] are held in one vl=4 vector register across the whole b loop, and
// each block b does the identical per-lane sequence the scalar epilogue emits:
//     vp   = vfmul_vf(vdw, dA[m,b])   // dA[m,b] * dW[n,b]   (fmul; commutative -> bit-exact)
//     vf   = vfcvt_f_x(frag[r*4..])   // (float)frag[c]      (RNE)
//     vacc = vfmacc_vv(vacc, vp, vf)  // fma(vp, vf, vacc)   == scalar fmadd.s
// Columns are independent across b, so each lane's accumulation order (b=0,1,...,nb-1) is
// IDENTICAL to the scalar per-column order -> the result is bit-for-bit the scalar output.
// dW[n,b] for the 4 columns n=nj*4+0..3 is one strided load per b (stride = nb floats).
static void matmul_f32_range_deref_epi_vec(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                           const float * dW, float * Cf, long M, long N, long K,
                                           long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    const size_t   vl     = __riscv_vsetvl_e32m1(4);           // vl = 4 : one output row (4 columns)
    const ptrdiff_t dwstr = (ptrdiff_t) nb * (ptrdiff_t) sizeof(float);  // dW column stride (bytes)
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = nj_start; nj < nj_end; ++nj) {
            const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
            const float *  dWc  = dW + (long) (nj * 4) * nb;  // dW[(nj*4)*nb], columns strided by nb
            vfloat32m1_t acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t acc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag);
                vfloat32m1_t vdw = __riscv_vlse32_v_f32m1(dWc + b, dwstr, vl);   // dW[n0..3, b]
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
// [M6] B-FEED LOCALITY: identical to matmul_f32_range_deref_epi_vec but with the mi/nj loops
// INTERCHANGED to nj-OUTER, mi-INNER. Rationale: the pre-dequantized weight B is N*K int8 (this
// hart's column slice = (nje-njs)*4*K int8, ~1MB for N=2048/4-hart, > L2), while the activation
// pack A is Mp*K int8 (~40KB, L1-resident). mi-outer re-streams this hart's whole B from DRAM
// once per mi-tile (mt times); nj-outer streams each nj-tile's B (4*K int8 = 8KB, L1) ONCE and
// reuses it across all mi, trading cheap L1 A re-reads for ~mt-fold less B DRAM traffic.
// ONLY the tile-visitation ORDER changes: each output tile (mi,nj) still accumulates over b in
// the SAME order (b=0..nb-1) into its own register-resident acc, and each Cf[m,n] has a single
// writer -> the result is BIT-FOR-BIT the mi-outer vec output (and thus the M3/M4/M5 scalar).
static void matmul_f32_range_deref_epi_vec_njouter(const int8_t * Apack, const float * dA, const int8_t * Bdec_all,
                                                   const float * dW, float * Cf, long M, long N, long K,
                                                   long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    const size_t   vl     = __riscv_vsetvl_e32m1(4);
    const ptrdiff_t dwstr = (ptrdiff_t) nb * (ptrdiff_t) sizeof(float);
    for (long nj = nj_start; nj < nj_end; ++nj) {              // NJ OUTER (B column-tile streamed once)
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        const float *  dWc  = dW + (long) (nj * 4) * nb;
        for (long mi = 0; mi < mt; ++mi) {                     // MI INNER (A re-read, L1-resident)
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
// [M5-prof] TIMING-ONLY: run vmadot (volatile asm streams A/B + writes frag) but SKIP the fold.
// Isolates vmadot compute + B/A feed. Not numerically valid (Cf left as-is) -> PROF phase only.
static void matmul_f32_range_deref_epi_noepi(const int8_t * Apack, const float * /*dA*/,
                                             const int8_t * Bdec_all, const float * /*dW*/,
                                             float * /*Cf*/, long M, long N, long K,
                                             long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    (void) N;
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = nj_start; nj < nj_end; ++nj) {
            const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag);  // volatile: not DCE'd
            }
        }
    }
}
// [M6-prof] TIMING-ONLY: noepi with nj-OUTER loop order (measures vmadot+feed under B-locality).
static void matmul_f32_range_deref_epi_noepi_njouter(const int8_t * Apack, const float * /*dA*/,
                                                     const int8_t * Bdec_all, const float * /*dW*/,
                                                     float * /*Cf*/, long M, long N, long K,
                                                     long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, kt = K / 8, fpb = 4;
    (void) N;
    for (long nj = nj_start; nj < nj_end; ++nj) {
        const int8_t * Bcol = Bdec_all + (long) nj * kt * 32;
        for (long mi = 0; mi < mt; ++mi) {
            const int8_t * Arow = Apack + (long) mi * 4 * K;
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(Arow + b * fpb * 32, Bcol + b * fpb * 32, fpb, frag);  // volatile: not DCE'd
            }
        }
    }
}
// [M6-prof] TIMING-ONLY: same vmadot CALL COUNT as noepi, but every call reads A/B from a FIXED
// 128-byte L1-resident scratch (no strided streaming). Isolates vmadot COMPUTE (+ L1 latency)
// from the DRAM/L2 FEED: feed_share = noepi(real strided B) - noepi_l1(this). PROF phase only.
static void matmul_f32_range_deref_epi_noepi_l1(const int8_t * /*Apack*/, const float * /*dA*/,
                                                const int8_t * /*Bdec_all*/, const float * /*dW*/,
                                                float * /*Cf*/, long M, long N, long K,
                                                long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32, fpb = 4;
    (void) N;
    alignas(64) int8_t sA[128]; alignas(64) int8_t sB[128];
    for (int i = 0; i < 128; ++i) { sA[i] = (int8_t) (i & 7); sB[i] = (int8_t) ((i * 3) & 7); }
    for (long nj = nj_start; nj < nj_end; ++nj) {
        for (long mi = 0; mi < mt; ++mi) {
            for (long b = 0; b < nb; ++b) {
                int32_t frag[16];
                vmadot_mac_kloop(sA, sB, fpb, frag);   // fixed L1 scratch -> compute only, no streaming
            }
        }
    }
}
// [M5-prof] TIMING-ONLY: SKIP vmadot, run the scalar fold on a FIXED frag. Isolates the scalar
// epilogue (dA/dW feed + fmul/fmadd + Cf writes), no B streaming. PROF phase only.
static void matmul_f32_range_deref_epi_nomadot(const int8_t * Apack, const float * dA,
                                               const int8_t * /*Bdec_all*/, const float * dW,
                                               float * Cf, long M, long N, long K,
                                               long nj_start, long nj_end) {
    const long mt = M / 4, nb = K / 32;
    (void) Apack;
    int32_t frag[16];
    for (int i = 0; i < 16; ++i) frag[i] = 1;   // fixed non-zero pattern
    for (long mi = 0; mi < mt; ++mi) {
        for (long nj = nj_start; nj < nj_end; ++nj) {
            float acc[16];
            for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
            for (long b = 0; b < nb; ++b) {
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

// ---------------- [M1] weight-repack cache + [PROF] Amdahl accounting (orchestration only) ---------
struct PackedW { std::vector<uint8_t> Bnib; std::vector<float> dW; long N; long K; };
static std::unordered_map<const void *, PackedW> g_wcache;   // keyed by src0->data (immutable weight)
// ---------------- [M3-a] dequantized-B cache (keyed by src0->data; replaces nibble cache when DEREF) ---
struct PackedWDec { std::vector<int8_t> Bdec; std::vector<float> dW; long N; long K; };
static std::unordered_map<const void *, PackedWDec> g_wcache_dec;
// ---------------- [M2] shared multithread work store (published by ith==0 before barrier #1) --------
struct MTWork { const int8_t * Apack; const float * dA; const uint8_t * Bnib; const int8_t * Bdec;
                const float * dW; float * Cfp; const float * X; long Mp; long Ml; long Nl; long Kl; };
static MTWork              g_mtwork = {};
static std::vector<int8_t> g_Apack;   // shared activation pack (reused across calls)
static std::vector<float>  g_dA;      // shared activation scales
static std::vector<float>  g_Cfp;     // shared padded output accumulator
static uint64_t g_cyc_repack = 0, g_cyc_dequant = 0, g_cyc_quant = 0, g_cyc_matmul = 0;
static uint64_t g_cyc_alloc = 0, g_cyc_copyback = 0;   // [M4] serial-setup attribution
static uint64_t g_n_calls = 0, g_n_repack_runs = 0, g_n_cache_hits = 0, g_n_dequant_runs = 0;
static bool     g_env_read = false, g_cache = false, g_threads = false, g_prof = false, g_prof_reg = false;
static bool     g_deref = false, g_deref_epi = false, g_parsetup = false;
static bool     g_epivec = false, g_mmprof_noepi = false, g_mmprof_nomadot = false;   // [M5]
static bool     g_njouter = false, g_mmprof_l1 = false;   // [M6] B-feed locality + compute-isolation probe
static inline uint64_t nowns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}
// [M5/M6] dispatch the deref+epi matmul family for a column-tile [nj_start,nj_end): timing-prof
// variants take priority (PROF phase only), then [M6] nj-outer / [M5] mi-outer vec epilogue,
// then M3 scalar. g_njouter selects the B-feed-locality (nj-outer) tile order.
static inline void run_deref_epi_tile(const MTWork & w, long Mp, long Nl, long Kl, long njs, long nje) {
    if (g_mmprof_l1) {
        matmul_f32_range_deref_epi_noepi_l1(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
    } else if (g_mmprof_noepi) {
        if (g_njouter) matmul_f32_range_deref_epi_noepi_njouter(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
        else           matmul_f32_range_deref_epi_noepi(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
    } else if (g_mmprof_nomadot) {
        matmul_f32_range_deref_epi_nomadot(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
    } else if (g_epivec) {
        if (g_njouter) matmul_f32_range_deref_epi_vec_njouter(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
        else           matmul_f32_range_deref_epi_vec(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
    } else {
        matmul_f32_range_deref_epi(w.Apack, w.dA, w.Bdec, w.dW, w.Cfp, Mp, Nl, Kl, njs, nje);
    }
}
static void prof_dump() {
    uint64_t tot = g_cyc_repack + g_cyc_dequant + g_cyc_quant + g_cyc_matmul + 1;
    fprintf(stderr,
            "[TCRV-Q40-PROF] cache=%d threads=%d deref=%d deref_epi=%d parsetup=%d epivec=%d "
            "njouter=%d mmprof_noepi=%d mmprof_nomadot=%d mmprof_l1=%d calls=%llu repack_runs=%llu "
            "dequant_runs=%llu cache_hits=%llu | "
            "ns_repack=%llu ns_dequant=%llu ns_quant=%llu ns_alloc=%llu ns_copyback=%llu ns_matmul=%llu | "
            "repack_share=%.4f dequant_share=%.4f quant_share=%.4f matmul_share=%.4f\n",
            (int) g_cache, (int) g_threads, (int) g_deref, (int) g_deref_epi, (int) g_parsetup,
            (int) g_epivec, (int) g_njouter, (int) g_mmprof_noepi, (int) g_mmprof_nomadot, (int) g_mmprof_l1,
            (unsigned long long) g_n_calls, (unsigned long long) g_n_repack_runs,
            (unsigned long long) g_n_dequant_runs, (unsigned long long) g_n_cache_hits,
            (unsigned long long) g_cyc_repack, (unsigned long long) g_cyc_dequant,
            (unsigned long long) g_cyc_quant, (unsigned long long) g_cyc_alloc,
            (unsigned long long) g_cyc_copyback, (unsigned long long) g_cyc_matmul,
            (double) g_cyc_repack / (double) tot, (double) g_cyc_dequant / (double) tot,
            (double) g_cyc_quant / (double) tot, (double) g_cyc_matmul / (double) tot);
}
static inline void env_read_once() {
    if (g_env_read) return;
    g_env_read = true;
    g_cache          = std::getenv("TCRV_IME_Q40_CACHE")          != nullptr;
    g_threads        = std::getenv("TCRV_IME_Q40_THREADS")        != nullptr;
    g_deref          = std::getenv("TCRV_IME_Q40_DEREF")          != nullptr;
    g_deref_epi      = std::getenv("TCRV_IME_Q40_DEREF_EPI")      != nullptr;
    g_parsetup       = std::getenv("TCRV_IME_Q40_PARSETUP")       != nullptr;
    g_epivec         = std::getenv("TCRV_IME_Q40_EPIVEC")         != nullptr;
    g_njouter        = std::getenv("TCRV_IME_Q40_NJOUTER")        != nullptr;
    g_mmprof_noepi   = std::getenv("TCRV_IME_Q40_MMPROF_NOEPI")   != nullptr;
    g_mmprof_nomadot = std::getenv("TCRV_IME_Q40_MMPROF_NOMADOT") != nullptr;
    g_mmprof_l1      = std::getenv("TCRV_IME_Q40_MMPROF_L1")      != nullptr;
    g_prof           = std::getenv("TCRV_IME_Q40_PROF")           != nullptr;
    if (g_prof && !g_prof_reg) { g_prof_reg = true; atexit(prof_dump); }
}
}  // namespace tcrv_q40

class tcrv_q4_0_tensor_traits : public tensor_traits_base {
    bool work_size(int /*n_threads*/, const ggml_tensor * /*op*/, size_t & /*size*/) override { return false; }

    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        memcpy(t->data, data, data_size);  // NATIVE passthrough
        return 0;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;
        if (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32) {
            return false;
        }
        const int64_t K = src0->ne[0];
        const int64_t N = src0->ne[1];
        const int64_t M = src1->ne[1];
        // route only the regime our fragment-major kernel covers; everything else ->
        // ggml default on native bytes (correct via passthrough repack).
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
        if (src0->nb[1] != (size_t) (K / QK4_0) * sizeof(block_q4_0)) {
            return false;
        }
        const int  ith = params->ith;
        const int  nth = params->nth;
        const long Ml = (long) M, Nl = (long) N, Kl = (long) K;
        const long Mp = (Ml + 3) / 4 * 4;
        float *    Cf = (float *) dst->data;

        // ---------------- ith==0: env + serial setup (repack/dequant cache, activation quant) ----------
        if (ith == 0) {
            tcrv_q40::env_read_once();
            tcrv_q40::g_n_calls++;
            memset(Cf, 0, (size_t) Ml * Nl * sizeof(float));

            const uint8_t * Bnib_p = nullptr;
            const int8_t *  Bdec_p = nullptr;
            const float *   dW_p   = nullptr;
            std::vector<uint8_t> Bnib_local;
            std::vector<float>   dW_local;

            if (tcrv_q40::g_deref) {
                // ---- [M3-a] DEQUANTIZED-B cache (load-once): dequant the immutable weight ONCE ----
                auto it = tcrv_q40::g_wcache_dec.find(src0->data);
                if (it == tcrv_q40::g_wcache_dec.end() || it->second.N != Nl || it->second.K != Kl) {
                    tcrv_q40::PackedWDec pw;
                    pw.N = Nl; pw.K = Kl;
                    pw.Bdec.assign((size_t) Nl * Kl, 0);                 // fragment-major int8 (N*K bytes)
                    pw.dW.assign((size_t) Nl * (Kl / 32), 0.0f);
                    std::vector<uint8_t> nib_tmp((size_t) (Nl / 4) * (Kl / 8) * 18, 0);  // transient
                    uint64_t t0 = tcrv_q40::nowns();
                    tcrv_q40::repack_dequant_weight((const uint8_t *) src0->data, pw.Bdec.data(),
                                                    pw.dW.data(), Nl, Kl, nib_tmp.data());
                    tcrv_q40::g_cyc_dequant += tcrv_q40::nowns() - t0;
                    tcrv_q40::g_n_dequant_runs++;
                    it = tcrv_q40::g_wcache_dec.emplace(src0->data, std::move(pw)).first;
                } else {
                    tcrv_q40::g_n_cache_hits++;
                }
                Bdec_p = it->second.Bdec.data();
                dW_p   = it->second.dW.data();
            } else if (tcrv_q40::g_cache) {
                // ---- [M1] weight repack: cached (load-once, first-use memoization) ----
                auto it = tcrv_q40::g_wcache.find(src0->data);
                if (it == tcrv_q40::g_wcache.end() || it->second.N != Nl || it->second.K != Kl) {
                    tcrv_q40::PackedW pw;
                    pw.N = Nl; pw.K = Kl;
                    pw.Bnib.assign((size_t) (Nl / 4) * (Kl / 8) * 18, 0);
                    pw.dW.assign((size_t) Nl * (Kl / 32), 0.0f);
                    uint64_t t0 = tcrv_q40::nowns();
                    tcrv_q40::repack_weight((const uint8_t *) src0->data, pw.Bnib.data(), pw.dW.data(), Nl, Kl);
                    tcrv_q40::g_cyc_repack += tcrv_q40::nowns() - t0;
                    tcrv_q40::g_n_repack_runs++;
                    it = tcrv_q40::g_wcache.emplace(src0->data, std::move(pw)).first;
                } else {
                    tcrv_q40::g_n_cache_hits++;
                }
                Bnib_p = it->second.Bnib.data();
                dW_p   = it->second.dW.data();
            } else {
                // ---- per-call repack (onnc baseline) ----
                Bnib_local.assign((size_t) (Nl / 4) * (Kl / 8) * 18, 0);
                dW_local.assign((size_t) Nl * (Kl / 32), 0.0f);
                uint64_t t0 = tcrv_q40::nowns();
                tcrv_q40::repack_weight((const uint8_t *) src0->data, Bnib_local.data(), dW_local.data(), Nl, Kl);
                tcrv_q40::g_cyc_repack += tcrv_q40::nowns() - t0;
                tcrv_q40::g_n_repack_runs++;
                Bnib_p = Bnib_local.data();
                dW_p   = dW_local.data();
            }

            // ---- allocate SHARED activation + output buffers (timed as serial-setup alloc) ----
            const bool epi_full = tcrv_q40::g_deref && tcrv_q40::g_deref_epi;
            uint64_t ta = tcrv_q40::nowns();
            tcrv_q40::g_Apack.assign((size_t) Mp * Kl, 0);
            tcrv_q40::g_dA.assign((size_t) Mp * (Kl / 32), 0.0f);
            if (tcrv_q40::g_parsetup && epi_full) tcrv_q40::g_Cfp.resize((size_t) Mp * Nl);
            else                                  tcrv_q40::g_Cfp.assign((size_t) Mp * Nl, 0.0f);
            tcrv_q40::g_cyc_alloc += tcrv_q40::nowns() - ta;

            // ---- activation quant/pack into SHARED store ----
            const bool par_quant = tcrv_q40::g_parsetup && tcrv_q40::g_threads;
            if (!par_quant) {
                std::vector<uint8_t> scratch((size_t) (Kl / 32) * 34);
                uint64_t t1 = tcrv_q40::nowns();
                tcrv_q40::quant_pack_act((const float *) src1->data, tcrv_q40::g_Apack.data(),
                                         tcrv_q40::g_dA.data(), Ml, Kl, scratch.data());
                tcrv_q40::g_cyc_quant += tcrv_q40::nowns() - t1;
            }

            // ---- publish work descriptor for the (parallel) matmul + (parallel) quant ----
            tcrv_q40::g_mtwork.Apack = tcrv_q40::g_Apack.data();
            tcrv_q40::g_mtwork.dA    = tcrv_q40::g_dA.data();
            tcrv_q40::g_mtwork.Bnib  = Bnib_p;
            tcrv_q40::g_mtwork.Bdec  = Bdec_p;
            tcrv_q40::g_mtwork.dW    = dW_p;
            tcrv_q40::g_mtwork.Cfp   = tcrv_q40::g_Cfp.data();
            tcrv_q40::g_mtwork.X     = (const float *) src1->data;
            tcrv_q40::g_mtwork.Mp    = Mp;
            tcrv_q40::g_mtwork.Ml    = Ml;
            tcrv_q40::g_mtwork.Nl    = Nl;
            tcrv_q40::g_mtwork.Kl    = Kl;

            if (!tcrv_q40::g_threads) {
                // ---- single-hart full matmul (byte-identical output across all gate combos) ----
                const long nt = Nl / 4;
                uint64_t t2 = tcrv_q40::nowns();
                if (tcrv_q40::g_deref && tcrv_q40::g_deref_epi) {
                    tcrv_q40::run_deref_epi_tile(tcrv_q40::g_mtwork, Mp, Nl, Kl, 0, nt);
                } else if (tcrv_q40::g_deref) {
                    tcrv_q40::matmul_f32_range_deref(tcrv_q40::g_mtwork.Apack, tcrv_q40::g_mtwork.dA,
                                                     tcrv_q40::g_mtwork.Bdec, tcrv_q40::g_mtwork.dW,
                                                     tcrv_q40::g_mtwork.Cfp, Mp, Nl, Kl, 0, nt);
                } else {
                    tcrv_q40::matmul_f32(tcrv_q40::g_mtwork.Apack, tcrv_q40::g_mtwork.dA,
                                         tcrv_q40::g_mtwork.Bnib, tcrv_q40::g_mtwork.dW,
                                         tcrv_q40::g_mtwork.Cfp, Mp, Nl, Kl);
                }
                tcrv_q40::g_cyc_matmul += tcrv_q40::nowns() - t2;
                memcpy(Cf, tcrv_q40::g_Cfp.data(), (size_t) Ml * Nl * sizeof(float));
            }

            static bool announced = false;
            if (!announced) {
                announced = true;
                fprintf(stderr,
                        "[TCRV-IME-Q40-BRIDGE] routed real q4_0 PREFILL mul_mat through tcrv IME kernel: "
                        "M=%ld N=%ld K=%ld (real vmadot 0xe210312b) cache=%d threads=%d deref=%d deref_epi=%d parsetup=%d epivec=%d njouter=%d nth=%d\n",
                        Ml, Nl, Kl, (int) tcrv_q40::g_cache, (int) tcrv_q40::g_threads,
                        (int) tcrv_q40::g_deref, (int) tcrv_q40::g_deref_epi, (int) tcrv_q40::g_parsetup,
                        (int) tcrv_q40::g_epivec, (int) tcrv_q40::g_njouter, nth);
            }
        }
        ggml_barrier(params->threadpool);  // barrier #1: publish env + packed work to all harts

        // ---------------- [M4-a] all harts: parallel activation quant over DISJOINT row tiles --------
        if (tcrv_q40::g_parsetup && tcrv_q40::g_threads) {
            const long Mlq = tcrv_q40::g_mtwork.Ml;
            const long Klq = tcrv_q40::g_mtwork.Kl;
            long       per = (Mlq + nth - 1) / nth;
            per = (per + 3) & ~3L;                 // mi-tile (4-row) align -> whole Apack tiles per hart
            long ms = (long) ith * per;
            long me = ms + per;
            if (me > Mlq) me = Mlq;
            uint64_t tq = (ith == 0) ? tcrv_q40::nowns() : 0;
            if (ms < me) {
                std::vector<uint8_t> scratch((size_t) (Klq / 32) * 34);   // per-hart transient
                tcrv_q40::quant_pack_act_range(tcrv_q40::g_mtwork.X, tcrv_q40::g_Apack.data(),
                                               tcrv_q40::g_dA.data(), Mlq, Klq, scratch.data(), ms, me);
            }
            ggml_barrier(params->threadpool);      // barrier #1b: all activation quant complete
            if (ith == 0) tcrv_q40::g_cyc_quant += tcrv_q40::nowns() - tq;
        }

        // ---------------- [M2] all harts: parallel matmul over DISJOINT column tiles ----------------
        if (tcrv_q40::g_threads) {
            const long nt  = tcrv_q40::g_mtwork.Nl / 4;
            long       per = (nt + nth - 1) / nth;
            per = (per + 3) & ~3L;                // cache-line align (4 nj-tiles = 64B) -> no false sharing
            long njs = (long) ith * per;
            long nje = njs + per;
            if (nje > nt) nje = nt;
            uint64_t t2 = (ith == 0) ? tcrv_q40::nowns() : 0;
            if (njs < nje) {
                if (tcrv_q40::g_deref && tcrv_q40::g_deref_epi) {
                    tcrv_q40::run_deref_epi_tile(tcrv_q40::g_mtwork, tcrv_q40::g_mtwork.Mp,
                                                 tcrv_q40::g_mtwork.Nl, tcrv_q40::g_mtwork.Kl, njs, nje);
                } else if (tcrv_q40::g_deref) {
                    tcrv_q40::matmul_f32_range_deref(tcrv_q40::g_mtwork.Apack, tcrv_q40::g_mtwork.dA,
                                                     tcrv_q40::g_mtwork.Bdec, tcrv_q40::g_mtwork.dW,
                                                     tcrv_q40::g_mtwork.Cfp, tcrv_q40::g_mtwork.Mp,
                                                     tcrv_q40::g_mtwork.Nl, tcrv_q40::g_mtwork.Kl, njs, nje);
                } else {
                    tcrv_q40::matmul_f32_range(tcrv_q40::g_mtwork.Apack, tcrv_q40::g_mtwork.dA,
                                               tcrv_q40::g_mtwork.Bnib, tcrv_q40::g_mtwork.dW,
                                               tcrv_q40::g_mtwork.Cfp, tcrv_q40::g_mtwork.Mp,
                                               tcrv_q40::g_mtwork.Nl, tcrv_q40::g_mtwork.Kl, njs, nje);
                }
            }
            ggml_barrier(params->threadpool);  // barrier #2: all column slices complete
            if (ith == 0) {
                tcrv_q40::g_cyc_matmul += tcrv_q40::nowns() - t2;  // critical-path wall time of parallel matmul
                uint64_t tc = tcrv_q40::nowns();
                memcpy(Cf, tcrv_q40::g_mtwork.Cfp, (size_t) Ml * Nl * sizeof(float));
                tcrv_q40::g_cyc_copyback += tcrv_q40::nowns() - tc;
            }
        }
        ggml_barrier(params->threadpool);  // barrier #3: final (matches sealed bridge)
        return true;
    }
};

static tcrv_q4_0_tensor_traits tcrv_q4_0_bridge;

'''
s = s.replace(cls_anchor, block + cls_anchor, 1)

# --- 3. env-gated selection in get_optimal_repack_type Q4_0 case -----------
sel_anchor = ("        case GGML_TYPE_Q4_0:\n"
              "            {\n"
              "#if defined(RISCV64_SPACEMIT_IME2)\n")
assert s.count(sel_anchor) == 1, "expected exactly one Q4_0 get_optimal anchor"
sel_new = ("        case GGML_TYPE_Q4_0:\n"
           "            {\n"
           "                if (std::getenv(\"TCRV_IME_Q40_BRIDGE\") && cur->ne[1] % 16 == 0 && cur->ne[0] % 32 == 0) {\n"
           "                    return &ggml::cpu::riscv64_spacemit::tcrv_q4_0_bridge;\n"
           "                }\n"
           "#if defined(RISCV64_SPACEMIT_IME2)\n")
s = s.replace(sel_anchor, sel_new, 1)

open(F, "w").write(s)

# --- post-write verification ----------------------------------------------
t = open(F).read()
assert "class tcrv_q4_0_tensor_traits" in t, "trait class missing after write"
assert "static tcrv_q4_0_tensor_traits tcrv_q4_0_bridge;" in t, "static instance missing"
assert "return &ggml::cpu::riscv64_spacemit::tcrv_q4_0_bridge;" in t, "selection missing"
assert "g_wcache_dec" in t, "[M3] dequant cache missing"
assert "repack_dequant_weight" in t, "[M3-a] dequant-populate missing"
assert "matmul_f32_range_deref" in t, "[M3-a] deref matmul missing"
assert "matmul_f32_range_deref_epi" in t, "[M3-b] deref+epi matmul missing"
assert "quant_pack_act_range" in t, "[M4-a] parallel quant range missing"
assert "TCRV_IME_Q40_PARSETUP" in t, "[M4] parsetup gate missing"
assert "barrier #1b: all activation quant complete" in t, "[M4] barrier #1b missing"
assert "matmul_f32_range_deref_epi_vec" in t, "[M5] vectorized epilogue missing"
assert "matmul_f32_range_deref_epi_vec_njouter" in t, "[M6] nj-outer vec epilogue missing"
assert "matmul_f32_range_deref_epi_noepi" in t, "[M5-prof] noepi missing"
assert "matmul_f32_range_deref_epi_noepi_njouter" in t, "[M6-prof] noepi nj-outer missing"
assert "matmul_f32_range_deref_epi_noepi_l1" in t, "[M6-prof] noepi L1 compute-isolation missing"
assert "matmul_f32_range_deref_epi_nomadot" in t, "[M5-prof] nomadot missing"
assert "run_deref_epi_tile" in t, "[M5] deref-epi dispatcher missing"
assert "__riscv_vfmacc_vv_f32m1" in t, "[M5] vfmacc (fused) missing"
assert "__riscv_vlse32_v_f32m1" in t, "[M5] strided dW load missing"
assert "TCRV_IME_Q40_EPIVEC" in t, "[M5] epivec gate missing"
assert "TCRV_IME_Q40_NJOUTER" in t, "[M6] njouter gate missing"
assert "TCRV_IME_Q40_MMPROF_NOEPI" in t and "TCRV_IME_Q40_MMPROF_NOMADOT" in t, "[M5] mmprof gates missing"
assert "TCRV_IME_Q40_MMPROF_L1" in t, "[M6] mmprof_l1 gate missing"
assert "g_epivec" in t and "g_mmprof_noepi" in t and "g_mmprof_nomadot" in t, "[M5] flags missing"
assert "g_njouter" in t and "g_mmprof_l1" in t, "[M6] flags missing"
assert "TCRV_IME_Q40_DEREF" in t and "TCRV_IME_Q40_DEREF_EPI" in t, "[M3] deref gates missing"
assert "matmul_f32_range" in t, "[M2] parallel matmul missing"
assert "TCRV_IME_Q40_THREADS" in t, "[M2] threads gate missing"
assert ("barrier #1: publish" in t and "barrier #2: all column slices" in t
        and "barrier #3: final" in t), "expected the labeled barriers in the bridge block"
assert "TCRV-Q40-PROF" in t, "[PROF] dump missing"
assert t.count("[TCRV-IME-Q40-BRIDGE]") >= 2, "markers missing"
print("patched OK; bridge_markers=%d" % t.count("TCRV-IME-Q40-BRIDGE") +
      "; epivec=1; njouter=1; mmprof_noepi=1; mmprof_nomadot=1; mmprof_l1=1; vfmacc=1; strided_dW=1; range_matmuls=9; trait=1; selection=1")
