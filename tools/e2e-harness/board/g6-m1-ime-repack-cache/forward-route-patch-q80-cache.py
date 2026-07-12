#!/usr/bin/env python3
# G6-A M1 (repack caching) patch for vendor spacemit ime.cpp -- q8_0 flat int8-direct IME bridge.
# Superset of the sealed forward-route-patch-q8.py: vmadot kernel functions BYTE-IDENTICAL.
# ONLY addition = orchestration in compute_forward: [M1] env-gated (TCRV_IME_Q80_CACHE)
# weight int8-gather CACHE (immutable q8_0 weight int8-gathered ONCE, keyed by src0->data
# +N+K, reused) + [PROF] (TCRV_IME_Q80_PROF) per-stage ns dumped at exit. t->data stays
# NATIVE (passthrough) -> fallback correctness; cache = side store of the same gather
# (numerically neutral). Gates default OFF -> identical to sealed bridge.
# NOTE: q8_0 repack is a straight int8 gather (no nibble unpack, no dequant in matmul) =
# the cheapest repack of the triple, so M1 upside here is the smallest; profiling quantifies.
import sys
F = "ggml/src/ggml-cpu/spacemit/ime.cpp"
s = open(F).read()

inc_anchor = '#include <cstdio>  // for GGML_ASSERT\n'
assert inc_anchor in s, "cstdio include anchor not found"
add_inc = ('#include <cstdlib>       // [TCRV-Q80] std::getenv/atexit\n'
           '#include <cstring>       // [TCRV-Q80] memcpy/memset\n'
           '#include <ctime>         // [TCRV-Q80-M1] clock_gettime prof\n'
           '#include <unordered_map> // [TCRV-Q80-M1] weight gather cache\n')
if "[TCRV-Q80] std::getenv" not in s:
    s = s.replace(inc_anchor, inc_anchor + add_inc, 1)

cls_anchor = "// Impl By IME1\n"
assert s.count(cls_anchor) == 1, "expected exactly one '// Impl By IME1' anchor"

block = r'''// ==================== [TCRV-IME-Q80-BRIDGE] flat int8-direct routing + [M1] gather cache ====================
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
static void repack_weight(const uint8_t * wq, int8_t * Bpack, float * dW, long N, long K) {
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
static void matmul_f32(const int8_t * Apack, const float * dA, const int8_t * Bpack, const float * dW,
                       float * Cf, long M, long N, long K) {
    const long mt = M / 4, nt = N / 4, nb = K / 32, kt = K / 8, fpb = 4;
    for (long mi = 0; mi < mt; ++mi) {
        const int8_t * Arow = Apack + (long) mi * 4 * K;
        for (long nj = 0; nj < nt; ++nj) {
            const int8_t * Bcol = Bpack + (long) nj * kt * 32;
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

// ---------------- [M1] weight gather cache + [PROF] Amdahl accounting (orchestration only) ---------
struct PackedW { std::vector<int8_t> Bpack; std::vector<float> dW; long N; long K; };
static std::unordered_map<const void *, PackedW> g_wcache;
static uint64_t g_cyc_repack = 0, g_cyc_quant = 0, g_cyc_matmul = 0;
static uint64_t g_n_calls = 0, g_n_repack_runs = 0, g_n_cache_hits = 0;
static bool     g_env_read = false, g_cache = false, g_prof = false, g_prof_reg = false;
static inline uint64_t nowns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}
static void prof_dump() {
    uint64_t tot = g_cyc_repack + g_cyc_quant + g_cyc_matmul + 1;
    fprintf(stderr,
            "[TCRV-Q80-PROF] cache=%d calls=%llu repack_runs=%llu cache_hits=%llu | "
            "ns_repack=%llu ns_quant=%llu ns_matmul=%llu | "
            "repack_share=%.4f quant_share=%.4f matmul_share=%.4f\n",
            (int) g_cache, (unsigned long long) g_n_calls, (unsigned long long) g_n_repack_runs,
            (unsigned long long) g_n_cache_hits, (unsigned long long) g_cyc_repack,
            (unsigned long long) g_cyc_quant, (unsigned long long) g_cyc_matmul,
            (double) g_cyc_repack / (double) tot, (double) g_cyc_quant / (double) tot,
            (double) g_cyc_matmul / (double) tot);
}
static inline void env_read_once() {
    if (g_env_read) return;
    g_env_read = true;
    g_cache = std::getenv("TCRV_IME_Q80_CACHE") != nullptr;
    g_prof  = std::getenv("TCRV_IME_Q80_PROF")  != nullptr;
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
        const int ith = params->ith;
        if (ith == 0) {
            tcrv_q80::env_read_once();
            tcrv_q80::g_n_calls++;
            const long Ml = (long) M, Nl = (long) N, Kl = (long) K;
            const long Mp = (Ml + 3) / 4 * 4;
            float *    Cf = (float *) dst->data;
            memset(Cf, 0, (size_t) Ml * Nl * sizeof(float));

            // ---- [M1] weight int8-gather: cached (load-once) or per-call ----
            const int8_t * Bpack_p;
            const float *  dW_p;
            std::vector<int8_t> Bpack_local;
            std::vector<float>  dW_local;
            if (tcrv_q80::g_cache) {
                auto it = tcrv_q80::g_wcache.find(src0->data);
                if (it == tcrv_q80::g_wcache.end() || it->second.N != Nl || it->second.K != Kl) {
                    tcrv_q80::PackedW pw; pw.N = Nl; pw.K = Kl;
                    pw.Bpack.assign((size_t) (Nl / 4) * (Kl / 8) * 32, 0);
                    pw.dW.assign((size_t) Nl * (Kl / 32), 0.0f);
                    uint64_t t0 = tcrv_q80::nowns();
                    tcrv_q80::repack_weight((const uint8_t *) src0->data, pw.Bpack.data(), pw.dW.data(), Nl, Kl);
                    tcrv_q80::g_cyc_repack += tcrv_q80::nowns() - t0;
                    tcrv_q80::g_n_repack_runs++;
                    it = tcrv_q80::g_wcache.emplace(src0->data, std::move(pw)).first;
                } else {
                    tcrv_q80::g_n_cache_hits++;
                }
                Bpack_p = it->second.Bpack.data();
                dW_p    = it->second.dW.data();
            } else {
                Bpack_local.assign((size_t) (Nl / 4) * (Kl / 8) * 32, 0);
                dW_local.assign((size_t) Nl * (Kl / 32), 0.0f);
                uint64_t t0 = tcrv_q80::nowns();
                tcrv_q80::repack_weight((const uint8_t *) src0->data, Bpack_local.data(), dW_local.data(), Nl, Kl);
                tcrv_q80::g_cyc_repack += tcrv_q80::nowns() - t0;
                tcrv_q80::g_n_repack_runs++;
                Bpack_p = Bpack_local.data();
                dW_p    = dW_local.data();
            }

            std::vector<int8_t>  Apack((size_t) Mp * Kl, 0);
            std::vector<float>   dA((size_t) Mp * (Kl / 32), 0.0f);
            std::vector<uint8_t> scratch((size_t) (Kl / 32) * 34);
            uint64_t t1 = tcrv_q80::nowns();
            tcrv_q80::quant_pack_act((const float *) src1->data, Apack.data(), dA.data(), Ml, Kl, scratch.data());
            tcrv_q80::g_cyc_quant += tcrv_q80::nowns() - t1;

            std::vector<float> Cfp((size_t) Mp * Nl, 0.0f);
            uint64_t t2 = tcrv_q80::nowns();
            tcrv_q80::matmul_f32(Apack.data(), dA.data(), Bpack_p, dW_p, Cfp.data(), Mp, Nl, Kl);
            tcrv_q80::g_cyc_matmul += tcrv_q80::nowns() - t2;
            memcpy(Cf, Cfp.data(), (size_t) Ml * Nl * sizeof(float));

            static bool announced = false;
            if (!announced) {
                announced = true;
                fprintf(stderr,
                        "[TCRV-IME-Q80-BRIDGE] routed real q8_0 PREFILL mul_mat through tcrv IME kernel: "
                        "M=%ld N=%ld K=%ld (real vmadot 0xe210312b int8-direct) cache=%d\n",
                        Ml, Nl, Kl, (int) tcrv_q80::g_cache);
            }
        }
        ggml_barrier(params->threadpool);
        return true;
    }
};

static tcrv_q8_0_tensor_traits tcrv_q8_0_bridge;

'''
s = s.replace(cls_anchor, block + cls_anchor, 1)

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

t = open(F).read()
assert "class tcrv_q8_0_tensor_traits" in t, "trait class missing after write"
assert "static tcrv_q8_0_tensor_traits tcrv_q8_0_bridge;" in t, "static instance missing"
assert "return &ggml::cpu::riscv64_spacemit::tcrv_q8_0_bridge;" in t, "selection missing"
assert "g_wcache" in t, "[M1] cache missing"
assert "TCRV-Q80-PROF" in t, "[PROF] dump missing"
assert t.count("[TCRV-IME-Q80-BRIDGE]") >= 2, "markers missing"
print("patched OK; markers=%d; cache=1; prof=1; trait=1; selection=1" % t.count("TCRV-IME-Q80-BRIDGE"))
