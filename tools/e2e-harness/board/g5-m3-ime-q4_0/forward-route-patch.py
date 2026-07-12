#!/usr/bin/env python3
# G5-M3 session-3 FORWARD TRAFFIC ROUTING patch for the vendor spacemit ime.cpp.
#
# Registers an env-gated (TCRV_IME_Q40_BRIDGE) parallel tcrv tensor_traits for
# q4_0 that routes REAL q4_0 PREFILL mul_mat traffic through our fragment-major
# scale-fold IME kernel (real vmadot 0xe210312b). Its repack() is a NATIVE
# passthrough, so every op we do not intercept (get_rows, decode M==1, odd shapes)
# falls through to the correct ggml default on native bytes -> full-model
# correctness. The #4 weight repack, #3 activation quant/pack and the scale-fold
# GEMM are byte-identical to the session-2 board UT (g5m3_bridge_ut.c).
#
# Env OFF -> get_optimal_repack_type returns the vendor trait unchanged (stock
# behaviour, byte-identical). Env ON -> returns our tcrv trait for q4_0.
#
# Fully reversible: restore = clean source -> clean .o -> overwrite .so with the
# ORIG binary -> md5 zero-change (driver run-forward-route.sh).
import sys

F = "ggml/src/ggml-cpu/spacemit/ime.cpp"
s = open(F).read()

# --- 1. includes -----------------------------------------------------------
inc_anchor = '#include <cstdio>  // for GGML_ASSERT\n'
assert inc_anchor in s, "cstdio include anchor not found"
add_inc = ('#include <cstdlib>  // [TCRV-Q40] std::getenv\n'
           '#include <cstring>  // [TCRV-Q40] memcpy/memset\n')
if "[TCRV-Q40] std::getenv" not in s:
    s = s.replace(inc_anchor, inc_anchor + add_inc, 1)

# --- 2. helpers + trait class + static instance (inside namespace) ---------
cls_anchor = "// Impl By IME1\n"
assert s.count(cls_anchor) == 1, "expected exactly one '// Impl By IME1' anchor"

block = r'''// ==================== [TCRV-IME-Q40-BRIDGE] session-3 forward traffic routing ====================
// Env-gated (TCRV_IME_Q40_BRIDGE) parallel tcrv tensor_traits. repack() is a NATIVE
// passthrough so any op we do not intercept falls through to the correct ggml default
// on native bytes. compute_forward routes real q4_0 PREFILL mul_mat traffic through our
// fragment-major scale-fold IME kernel (real vmadot 0xe210312b). The three bridges below
// are byte-identical to the session-2 board UT (g5m3_bridge_ut.c).
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
        const int ith = params->ith;
        if (ith == 0) {
            const long Ml = (long) M, Nl = (long) N, Kl = (long) K;
            const long Mp = (Ml + 3) / 4 * 4;
            float *    Cf = (float *) dst->data;
            memset(Cf, 0, (size_t) Ml * Nl * sizeof(float));
            std::vector<uint8_t> Bnib((size_t) (Nl / 4) * (Kl / 8) * 18);
            std::vector<float>   dW((size_t) Nl * (Kl / 32));
            tcrv_q40::repack_weight((const uint8_t *) src0->data, Bnib.data(), dW.data(), Nl, Kl);
            std::vector<int8_t>  Apack((size_t) Mp * Kl, 0);
            std::vector<float>   dA((size_t) Mp * (Kl / 32), 0.0f);
            std::vector<uint8_t> scratch((size_t) (Kl / 32) * 34);
            tcrv_q40::quant_pack_act((const float *) src1->data, Apack.data(), dA.data(), Ml, Kl, scratch.data());
            std::vector<float> Cfp((size_t) Mp * Nl, 0.0f);
            tcrv_q40::matmul_f32(Apack.data(), dA.data(), Bnib.data(), dW.data(), Cfp.data(), Mp, Nl, Kl);
            memcpy(Cf, Cfp.data(), (size_t) Ml * Nl * sizeof(float));
            static bool announced = false;
            if (!announced) {
                announced = true;
                fprintf(stderr,
                        "[TCRV-IME-Q40-BRIDGE] routed real q4_0 PREFILL mul_mat through tcrv IME kernel: "
                        "M=%ld N=%ld K=%ld (real vmadot 0xe210312b)\n",
                        Ml, Nl, Kl);
            }
        }
        ggml_barrier(params->threadpool);
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
assert t.count("[TCRV-IME-Q40-BRIDGE]") >= 2, "markers missing"
print("patched OK; markers=%d" % t.count("TCRV-IME-Q40-BRIDGE") +
      "; trait=1; selection=1")
