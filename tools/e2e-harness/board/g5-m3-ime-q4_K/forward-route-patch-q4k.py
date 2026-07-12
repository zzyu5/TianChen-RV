#!/usr/bin/env python3
# G5-M3 FORWARD TRAFFIC ROUTING patch for the vendor spacemit ime.cpp -- the
# SUPER-BLOCK K-quant sibling of the q4_0 forward-route patch.
#
# Registers an env-gated (TCRV_IME_Q4K_BRIDGE) parallel tcrv tensor_traits for
# q4_K that routes REAL q4_K PREFILL mul_mat traffic through our super-block IME
# kernel (real vmadot 0xe210312b, two-level 6-bit scale/min fold). Its repack() is
# a NATIVE passthrough (the native 144B block_q4_K bytes are preserved verbatim, so
# every op we do not intercept -- get_rows, decode M==1, odd shapes -- falls through
# to the correct ggml default on native bytes -> full-model correctness). The
# spacemit Q4_K buffer alloc (160B/block via remap_block_nbytes) is >= native 144B,
# so the passthrough write fits. The #4 weight block-gather, #3 q8_K activation
# quant/pack and the two-level fold GEMM are byte-identical to the board integration
# UT (g5m3_q4k_bridge_ut.c), which is A==B vs REAL ggml dequantize_row_q4_K /
# quantize_row_q8_K / ggml_vec_dot_q4_K_q8_K.
#
# Env OFF -> get_optimal_repack_type returns the vendor trait unchanged (stock
# behaviour, byte-identical). Env ON -> returns our tcrv trait for q4_K.
#
# Fully reversible: restore = clean source -> clean .o -> overwrite .so with the
# ORIG binary -> md5 zero-change (driver run-forward-route-q4k.sh).
import sys

F = "ggml/src/ggml-cpu/spacemit/ime.cpp"
s = open(F).read()

# --- 1. includes -----------------------------------------------------------
inc_anchor = '#include <cstdio>  // for GGML_ASSERT\n'
assert inc_anchor in s, "cstdio include anchor not found"
add_inc = ('#include <cstdlib>  // [TCRV-Q4K] std::getenv\n'
           '#include <cstring>  // [TCRV-Q4K] memcpy/memset\n')
if "[TCRV-Q4K] std::getenv" not in s:
    s = s.replace(inc_anchor, inc_anchor + add_inc, 1)

# --- 2. helpers + trait class + static instance (inside namespace) ---------
cls_anchor = "// Impl By IME1\n"
assert s.count(cls_anchor) == 1, "expected exactly one '// Impl By IME1' anchor"

block = r'''// ==================== [TCRV-IME-Q4K-BRIDGE] forward traffic routing (super-block) ====================
// Env-gated (TCRV_IME_Q4K_BRIDGE) parallel tcrv tensor_traits for q4_K. repack() is a
// NATIVE passthrough (144B block_q4_K preserved verbatim) so any op we do not intercept
// falls through to the correct ggml default on native bytes. compute_forward routes real
// q4_K PREFILL mul_mat traffic through our super-block IME kernel (real vmadot 0xe210312b,
// two-level 6-bit scale/min fold + per-(row,super-block) q8_K activation-scale fold). The
// bridges below are byte-identical to the board integration UT (g5m3_q4k_bridge_ut.c).
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
static inline unsigned short load_fp16(const uint8_t * p) {
    return (unsigned short) ((unsigned) p[0] | ((unsigned) p[1] << 8));
}
// q4_K RAW-nibble decode (unsigned [0,15]; emitter/seal-verbatim).
static inline void dequant_fragment(const uint8_t * blk, int b, int kf, int8_t * out8) {
    const uint8_t * qs = blk + 16;
    for (int kl = 0; kl < 8; ++kl) {
        int     pl   = kf * 8 + kl;
        uint8_t byte = qs[(b / 2) * 32 + pl];
        out8[kl]     = (int8_t) ((b & 1) ? (byte >> 4) : (byte & 0x0F));
    }
}
// canonical ggml get_scale_min_k4 (6-bit sc/m unpack; emitter/seal-verbatim).
static inline void get_scale_min(int j, const uint8_t * q, uint8_t * sc, uint8_t * m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
// bridge #4: ggml NATIVE q4_K (row-major N x nsb 144B) -> (nj,sb,nl) block-gather.
static void repack_weight(const uint8_t * wq, uint8_t * Bq4k, long N, long K) {
    const long nsb = K / 256, q4kb = 144;
    for (long n = 0; n < N; ++n)
        for (long sb = 0; sb < nsb; ++sb) {
            const uint8_t * src = wq + (size_t) (n * nsb + sb) * q4kb;
            long            nj = n / 4, nl = n % 4;
            uint8_t *       dst = Bq4k + (size_t) (((nj * nsb + sb) * 4) + nl) * q4kb;
            memcpy(dst, src, q4kb);
        }
}
// bridge #3: ggml f32 activation (M x K) -> q8_K (quantize_row_q8_K_ref) -> fragment-major
// Apack (raw int8) + per-(row,super-block) dA = q8_K.d.
static void quant_pack_act(const float * X, int8_t * Apack, float * dA, long M, long K, uint8_t * scratch) {
    const long nsb = K / 256;
    const long q8kb = (long) sizeof(block_q8_K);
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
// two-level scale/min-fold f32 GEMM: int32 core (S_scale = Sum sc_b*sumi_b, S_min = Sum
// m_b*asum_b) via real vmadot; epilogue folds q8_K activation scale y.d per super-block:
// C += (d_w*y.d)*S_scale - (dmin_w*y.d)*S_min  (== ggml vec_dot_q4_K_q8_K).
static void matmul_f32(const int8_t * Apack, const float * dA, const uint8_t * Bq4k, float * Cf,
                       long M, long N, long K) {
    const long mt = M / 4, nt = N / 4, nsb = K / 256, q4kb = 144;
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = 0; nj < nt; ++nj) {
            for (long sb = 0; sb < nsb; ++sb) {
                const uint8_t * blk[4];
                uint8_t         sc[8][4], mm[8][4];
                for (int nl = 0; nl < 4; ++nl) {
                    blk[nl] = Bq4k + (size_t) ((((nj * nsb) + sb) * 4) + nl) * q4kb;
                    for (int b = 0; b < 8; ++b)
                        get_scale_min(b, blk[nl] + 4, &sc[b][nl], &mm[b][nl]);
                }
                int32_t Sc[16], Sm[16];
                for (int r = 0; r < 16; ++r) { Sc[r] = 0; Sm[r] = 0; }
                for (int b = 0; b < 8; ++b) {
                    int32_t sumi[16];
                    int32_t asum[4] = {0, 0, 0, 0};
                    int8_t  Bdec[128];
                    for (int kf = 0; kf < 4; ++kf) {
                        long           gf     = sb * 32 + b * 4 + kf;
                        const int8_t * Aframe = Arow + gf * 32;
                        for (int nl = 0; nl < 4; ++nl)
                            dequant_fragment(blk[nl], b, kf, Bdec + kf * 32 + nl * 8);
                        for (int ml = 0; ml < 4; ++ml)
                            for (int kl = 0; kl < 8; ++kl)
                                asum[ml] += (int32_t) Aframe[ml * 8 + kl];
                    }
                    const int8_t * Ablk = Arow + (long) (sb * 32 + b * 4) * 32;
                    vmadot_mac_kloop(Ablk, Bdec, 4, sumi);
                    for (int ml = 0; ml < 4; ++ml)
                        for (int nl = 0; nl < 4; ++nl) {
                            Sc[ml * 4 + nl] += (int32_t) sc[b][nl] * sumi[ml * 4 + nl];
                            Sm[ml * 4 + nl] += (int32_t) mm[b][nl] * asum[ml];
                        }
                }
                for (int ml = 0; ml < 4; ++ml)
                    for (int nl = 0; nl < 4; ++nl) {
                        long  mo = mi * 4 + ml, no = nj * 4 + nl;
                        float dw    = ggml_fp16_to_fp32(load_fp16(blk[nl] + 0));
                        float dminw = ggml_fp16_to_fp32(load_fp16(blk[nl] + 2));
                        float ad    = dA[mo * nsb + sb];
                        Cf[mo * N + no] += (dw * ad) * (float) Sc[ml * 4 + nl] -
                                           (dminw * ad) * (float) Sm[ml * 4 + nl];
                    }
            }
        }
    }
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
        // route only the regime our super-block kernel covers; everything else ->
        // ggml default on native bytes (correct via passthrough repack).
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
        const int ith = params->ith;
        if (ith == 0) {
            const long Ml = (long) M, Nl = (long) N, Kl = (long) K;
            const long Mp = (Ml + 3) / 4 * 4;
            const long nsb = Kl / 256;
            float *    Cf = (float *) dst->data;
            memset(Cf, 0, (size_t) Ml * Nl * sizeof(float));
            std::vector<uint8_t> Bq4k((size_t) Nl * nsb * 144);
            tcrv_q4k::repack_weight((const uint8_t *) src0->data, Bq4k.data(), Nl, Kl);
            std::vector<int8_t>  Apack((size_t) Mp * Kl, 0);
            std::vector<float>   dA((size_t) Mp * nsb, 0.0f);
            std::vector<uint8_t> scratch((size_t) nsb * sizeof(block_q8_K));
            tcrv_q4k::quant_pack_act((const float *) src1->data, Apack.data(), dA.data(), Ml, Kl, scratch.data());
            std::vector<float> Cfp((size_t) Mp * Nl, 0.0f);
            tcrv_q4k::matmul_f32(Apack.data(), dA.data(), Bq4k.data(), Cfp.data(), Mp, Nl, Kl);
            memcpy(Cf, Cfp.data(), (size_t) Ml * Nl * sizeof(float));
            static bool announced = false;
            if (!announced) {
                announced = true;
                fprintf(stderr,
                        "[TCRV-IME-Q4K-BRIDGE] routed real q4_K PREFILL mul_mat through tcrv IME kernel: "
                        "M=%ld N=%ld K=%ld (real vmadot 0xe210312b two-level fold)\n",
                        Ml, Nl, Kl);
            }
        }
        ggml_barrier(params->threadpool);
        return true;
    }
};

static tcrv_q4_K_tensor_traits tcrv_q4_K_bridge;

'''
s = s.replace(cls_anchor, block + cls_anchor, 1)

# --- 3. env-gated selection in get_optimal_repack_type Q4_K case -----------
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

# --- post-write verification ----------------------------------------------
t = open(F).read()
assert "class tcrv_q4_K_tensor_traits" in t, "trait class missing after write"
assert "static tcrv_q4_K_tensor_traits tcrv_q4_K_bridge;" in t, "static instance missing"
assert "return &ggml::cpu::riscv64_spacemit::tcrv_q4_K_bridge;" in t, "selection missing"
assert t.count("[TCRV-IME-Q4K-BRIDGE]") >= 2, "markers missing"
print("patched OK; markers=%d; trait=1; selection=1" % t.count("TCRV-IME-Q4K-BRIDGE"))
