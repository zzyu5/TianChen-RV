#!/usr/bin/env python3
# G6-A M4 (parallel-serial-setup) patch for vendor spacemit ime.cpp -- q4_0 IME bridge.
#
# SUPERSET of the M3 (de-reference-form) patch, which was a superset of M2 (multithread)
# / M1 (repack-cache). The vmadot GEMM kernel primitives (vmadot_mac_kloop /
# dequant_fragment / repack_weight / quant_pack_act / matmul_f32 / matmul_f32_range /
# repack_dequant_weight / matmul_f32_range_deref{,_epi}) are BYTE-IDENTICAL to the M3
# patch (emitter-verbatim seal intact). M4 attacks the M3 profile's single largest
# remaining residual: after the matmul critical path collapsed 11.49x (cached-dequant B +
# register epilogue), the SERIAL ith==0 SETUP floor surfaced (activation quant ~10% of the
# onderef kernel while 3 harts idle-wait at barrier #1, plus per-call allocs + copyback).
# M4 keeps the vmadot int32 anchor and the exact scalar f32 arithmetic untouched and
# parallelizes / lightens that serial setup:
#
#   [M4-a] PARALLEL ACTIVATION QUANT (TCRV_IME_Q40_PARSETUP, requires THREADS): the M3
#          activation quant (quant_pack_act) runs ENTIRELY on ith==0 before barrier #1
#          (0.95s serial; 3 harts idle). M4 slices the row loop across harts -- each hart
#          quantizes a disjoint mi-tile-aligned row range [m_start, m_end) via the
#          VERBATIM-bodied quant_pack_act_range, writing to DISJOINT dA/Apack slots (row m
#          -> deterministic positions; no cross-hart accumulation) -> the union of ranges
#          == the full serial quant, BYTE-IDENTICAL. A new barrier #1b syncs all quant
#          before any hart's matmul reads Apack. quantize_row_q8_0_ref is a pure function
#          of one activation row => same quantized bytes regardless of which hart runs it.
#
#   [M4-b] LIGHTENED SETUP ALLOC (folded under PARSETUP): the M3 register epilogue ASSIGNS
#          (=) every one of the Mp*Nl output accumulator elements, so the g_Cfp zero-fill
#          is redundant when deref_epi -> resize (no memset) instead of assign(0,...).
#          Padded (discarded) Apack/dA rows likewise need no zeroing. Copyback memcpy is
#          timed for attribution. These are numerically inert (valid rows fully written;
#          padded rows are computed then dropped by the Ml-row copyback).
#
# All are pure orchestration reorganizations (parallelize the quant; drop redundant memset)
# -- no kernel byte changes, no arithmetic changes, no summation-order changes. M4 output ==
# M3 output bit-for-bit (hard gate). All gates default OFF -> byte-identical to the M3 path
# (-> M2 -> M1 -> sealed bridge). PARSETUP without THREADS is a no-op (serial quant kept).
# Fully reversible (driver restores src/.o/.so; md5 double-proof).
#
# Env gates (additive over M1/M2/M3):
#   TCRV_IME_Q40_BRIDGE     : route q4_0 prefill mul_mat through the tcrv IME kernel
#   TCRV_IME_Q40_CACHE      : [M1] cache the repacked (nibble) weight (M2 path only)
#   TCRV_IME_Q40_THREADS    : [M2] column-tile multithread the matmul
#   TCRV_IME_Q40_DEREF      : [M3-a] cache the DEQUANTIZED int8 weight; hot matmul reads it
#   TCRV_IME_Q40_DEREF_EPI  : [M3-b] register-accumulated epilogue (requires DEREF)
#   TCRV_IME_Q40_PARSETUP   : [M4]  parallelize the ith==0 activation quant + lighten allocs
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

block = r'''// ==================== [TCRV-IME-Q40-BRIDGE] session-3 forward routing + [M1] repack cache + [M2] multithread + [M3] de-reference-form + [M4] parallel-setup ====================
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
//      (a) slice the ith==0 activation quant across harts (mi-tile-aligned disjoint row
//          ranges via the verbatim-bodied quant_pack_act_range; barrier #1b syncs before
//          matmul reads Apack); (b) drop the redundant g_Cfp zero-fill (register epilogue
//          assigns every element). Same quantized bytes + same vmadot bytes + same scalar
//          f32 expression + same summation order => bit-identical to M3.
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
// [M4-a] row-range slice of quant_pack_act: the loop body is BYTE-IDENTICAL to
// quant_pack_act, only the m bounds change to [m_start, m_end). Each row m writes to
// deterministic dA[m*nb+b] / Apack[(m/4)*4*K + kf*32 + (m%4)*8 + kl] slots (no cross-row
// state), so the union of disjoint ranges over harts == the full serial quant, byte-exact.
// scratch is a PER-HART transient (no sharing). Ranges are mi-tile (4-row) aligned so each
// hart owns whole 4*K Apack tiles -> no false sharing.
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
// [M2] column-tile slice of matmul_f32: identical arithmetic restricted to nj in
// [nj_start, nj_end). The inner (mi, nj, b, r, c) loop body is byte-identical to
// matmul_f32; only the nj bounds change. Each C[m,n] with n in [nj_start*4, nj_end*4)
// is fully accumulated over b in the SAME order as single-thread => bit-exact.
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
// [M3-a + M3-b] pre-dequantized B + register-accumulated epilogue.
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

// ---------------- [M1] weight-repack cache + [PROF] Amdahl accounting (orchestration only) ---------
struct PackedW { std::vector<uint8_t> Bnib; std::vector<float> dW; long N; long K; };
static std::unordered_map<const void *, PackedW> g_wcache;   // keyed by src0->data (immutable weight)
// ---------------- [M3-a] dequantized-B cache (keyed by src0->data; replaces nibble cache when DEREF) ---
struct PackedWDec { std::vector<int8_t> Bdec; std::vector<float> dW; long N; long K; };
static std::unordered_map<const void *, PackedWDec> g_wcache_dec;
// ---------------- [M2] shared multithread work store (published by ith==0 before barrier #1) --------
// [M4] adds X (raw activation ptr) + Ml (unpadded rows) so all harts can quant their row range.
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
static inline uint64_t nowns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}
static void prof_dump() {
    uint64_t tot = g_cyc_repack + g_cyc_dequant + g_cyc_quant + g_cyc_matmul + 1;
    fprintf(stderr,
            "[TCRV-Q40-PROF] cache=%d threads=%d deref=%d deref_epi=%d parsetup=%d calls=%llu repack_runs=%llu "
            "dequant_runs=%llu cache_hits=%llu | "
            "ns_repack=%llu ns_dequant=%llu ns_quant=%llu ns_alloc=%llu ns_copyback=%llu ns_matmul=%llu | "
            "repack_share=%.4f dequant_share=%.4f quant_share=%.4f matmul_share=%.4f\n",
            (int) g_cache, (int) g_threads, (int) g_deref, (int) g_deref_epi, (int) g_parsetup,
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
    g_cache     = std::getenv("TCRV_IME_Q40_CACHE")      != nullptr;
    g_threads   = std::getenv("TCRV_IME_Q40_THREADS")    != nullptr;
    g_deref     = std::getenv("TCRV_IME_Q40_DEREF")      != nullptr;
    g_deref_epi = std::getenv("TCRV_IME_Q40_DEREF_EPI")  != nullptr;
    g_parsetup  = std::getenv("TCRV_IME_Q40_PARSETUP")   != nullptr;
    g_prof      = std::getenv("TCRV_IME_Q40_PROF")       != nullptr;
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
            // [M4-b] when the register epilogue is active it ASSIGNS every Mp*Nl output element, so
            //        the g_Cfp zero-fill is redundant -> resize (no memset). g_Apack/g_dA padded rows
            //        (>= Ml) are computed then dropped by the Ml-row copyback, so their content is
            //        don't-care; kept assign(0) here (cheap, defensive, exactly M3 for those bytes).
            const bool epi_full = tcrv_q40::g_deref && tcrv_q40::g_deref_epi;
            uint64_t ta = tcrv_q40::nowns();
            tcrv_q40::g_Apack.assign((size_t) Mp * Kl, 0);
            tcrv_q40::g_dA.assign((size_t) Mp * (Kl / 32), 0.0f);
            if (tcrv_q40::g_parsetup && epi_full) tcrv_q40::g_Cfp.resize((size_t) Mp * Nl);
            else                                  tcrv_q40::g_Cfp.assign((size_t) Mp * Nl, 0.0f);
            tcrv_q40::g_cyc_alloc += tcrv_q40::nowns() - ta;

            // ---- activation quant/pack into SHARED store ----
            // [M4-a] when PARSETUP+THREADS: DEFER to the parallel row-range quant after barrier #1
            //        (all harts). Otherwise (M3 path or single-hart): do the full serial quant here.
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
                    tcrv_q40::matmul_f32_range_deref_epi(tcrv_q40::g_mtwork.Apack, tcrv_q40::g_mtwork.dA,
                                                         tcrv_q40::g_mtwork.Bdec, tcrv_q40::g_mtwork.dW,
                                                         tcrv_q40::g_mtwork.Cfp, Mp, Nl, Kl, 0, nt);
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
                        "M=%ld N=%ld K=%ld (real vmadot 0xe210312b) cache=%d threads=%d deref=%d deref_epi=%d parsetup=%d nth=%d\n",
                        Ml, Nl, Kl, (int) tcrv_q40::g_cache, (int) tcrv_q40::g_threads,
                        (int) tcrv_q40::g_deref, (int) tcrv_q40::g_deref_epi, (int) tcrv_q40::g_parsetup, nth);
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
                    tcrv_q40::matmul_f32_range_deref_epi(tcrv_q40::g_mtwork.Apack, tcrv_q40::g_mtwork.dA,
                                                         tcrv_q40::g_mtwork.Bdec, tcrv_q40::g_mtwork.dW,
                                                         tcrv_q40::g_mtwork.Cfp, tcrv_q40::g_mtwork.Mp,
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
assert "g_cyc_alloc" in t and "g_cyc_copyback" in t, "[M4] serial-setup counters missing"
assert "TCRV_IME_Q40_DEREF" in t and "TCRV_IME_Q40_DEREF_EPI" in t, "[M3] deref gates missing"
assert "matmul_f32_range" in t, "[M2] parallel matmul missing"
assert "TCRV_IME_Q40_THREADS" in t, "[M2] threads gate missing"
assert ("barrier #1: publish" in t and "barrier #2: all column slices" in t
        and "barrier #3: final" in t), "expected the labeled barriers in the bridge block"
assert "TCRV-Q40-PROF" in t, "[PROF] dump missing"
assert t.count("[TCRV-IME-Q40-BRIDGE]") >= 2, "markers missing"
print("patched OK; bridge_markers=%d" % t.count("TCRV-IME-Q40-BRIDGE") +
      "; parsetup=1; par_quant_range=1; barrier1b=1; deref=1; deref_epi=1; dequant_cache=1; range_matmuls=3; trait=1; selection=1")
