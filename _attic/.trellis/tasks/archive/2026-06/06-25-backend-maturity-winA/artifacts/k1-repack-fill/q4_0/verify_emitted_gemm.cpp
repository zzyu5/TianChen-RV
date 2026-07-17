// White-box numeric harness for the COMPILER-EMITTED q4_0 16x1 repack GEMM.
//
// Drives the compiler-emitted GEMM (ggml_gemm_q4_0_16x1_q8_0, supplied by the
// thin ABI adapter over tcrv-opt's emitted symbol) against ggml's own scalar
// ggml_gemm_q4_0_16x1_q8_0_generic reference, over many n, random repacked
// inputs. Verdict = matches to fp32 rounding (norm = max_abs_err / rms(ref)).
//
// GEMM-only: the emitted node is the prefill GEMM; GEMV stays hand-written.
// Layout/repack/quantizer logic is copied verbatim from the prior
// verify_kernel.cpp (the hand-kernel harness), which is the proven reference.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK4_0 32
#define QK8_0 32

typedef uint16_t ggml_half;

static inline float half_to_float(ggml_half h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else { exp = 127 - 15 + 1; while ((man & 0x400) == 0) { man <<= 1; exp--; } man &= 0x3ff; f = (sign<<31)|(exp<<23)|(man<<13); }
    } else if (exp == 0x1f) { f = (sign<<31)|(0xff<<23)|(man<<13); }
    else { f = (sign<<31)|((exp-15+127)<<23)|(man<<13); }
    float r; memcpy(&r, &f, 4); return r;
}
static inline ggml_half float_to_half(float fv) {
    uint32_t x; memcpy(&x, &fv, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (exp <= 0) return (ggml_half)sign;
    if (exp >= 0x1f) return (ggml_half)(sign | 0x7c00);
    uint32_t h = sign | (exp << 10) | (man >> 13);
    if ((man & 0x1000) && ((man & 0x2fff) || (h & 1))) h++;
    return (ggml_half)h;
}

struct block_q4_0 { ggml_half d; uint8_t qs[QK4_0/2]; };
struct block_q4_0x16 { ggml_half d[16]; int8_t qs[QK8_0*8]; }; // 288 bytes total
struct block_q8_0x4 { ggml_half d[4]; int8_t qs[QK8_0*4]; };   // 136 bytes total

extern "C" {
// COMPILER-EMITTED GEMM (via adapter) + ggml's scalar reference + real quantizer.
void ggml_gemm_q4_0_16x1_q8_0(int n, float*, size_t, const void*, const void*, int, int);
void ggml_gemm_q4_0_16x1_q8_0_generic(int n, float*, size_t, const void*, const void*, int, int);
void ggml_quantize_mat_q8_0_4x1(const float*, void*, int64_t);
}

// Inline of static make_block_q4_0x16(in, 1) from repack.cpp (the ^0x88 bias bake).
static block_q4_0x16 make_block(const block_q4_0* in) {
    block_q4_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    const int end = QK4_0 * 8 / 1; // 256
    const uint8_t xor_mask = 0x88;
    for (int i = 0; i < end; ++i) {
        int src_id = i % 16, src_offset = i / 16;
        out.qs[i] = (int8_t)(in[src_id].qs[src_offset] ^ xor_mask);
    }
    return out;
}

static void make_random_q4_0_row(block_q4_0* row, int nblocks, std::mt19937& rng, int nonneg=0) {
    std::uniform_int_distribution<int> nib(nonneg ? 8 : 0, 15);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    for (int b = 0; b < nblocks; b++) {
        row[b].d = float_to_half(sc(rng));
        for (int k = 0; k < QK4_0/2; k++) row[b].qs[k] = (uint8_t)(nib(rng) | (nib(rng) << 4));
    }
}

int main(int argc, char** argv) {
    // Many n: the model's exact rows plus stress sizes, as the task asks.
    int ns[] = {64, 256, 4096, 11008, 14336};
    const int NN = sizeof(ns)/sizeof(ns[0]);
    int trials = (argc > 1) ? atoi(argv[1]) : 200;
    int nonneg = (argc > 2) ? atoi(argv[2]) : 0;
    std::mt19937 rng(20260619);
    std::uniform_real_distribution<float> act(nonneg ? 0.1f : -2.0f, 2.0f);

    double max_gemm = 0, sse_gemm = 0, maxrel_gemm = 0;
    long cnt_gemm = 0;
    const int NC = 16;
    for (int ni = 0; ni < NN; ni++) {
        int n = ns[ni];
        int nb = n / QK4_0;
        for (int t = 0; t < trials; t++) {
            std::vector<block_q4_0> w(16 * nb);
            for (int r = 0; r < 16; r++) make_random_q4_0_row(&w[r * nb], nb, rng, nonneg);
            std::vector<block_q4_0x16> vx(nb);
            block_q4_0 tmp[16];
            for (int x = 0; x < nb; x++) {
                for (int i = 0; i < 16; i++) tmp[i] = w[i * nb + x];
                vx[x] = make_block(tmp);
            }
            std::vector<float> af(4 * n);
            for (int i = 0; i < 4*n; i++) af[i] = act(rng);
            std::vector<block_q8_0x4> vy(nb);
            ggml_quantize_mat_q8_0_4x1(af.data(), vy.data(), n);
            float out_mine[4*16], out_ref[4*16];
            const size_t bs = 16;
            ggml_gemm_q4_0_16x1_q8_0(n, out_mine, bs, vx.data(), vy.data(), 4, NC);
            ggml_gemm_q4_0_16x1_q8_0_generic(n, out_ref, bs, vx.data(), vy.data(), 4, NC);
            for (int m = 0; m < 4; m++) for (int j = 0; j < 16; j++) {
                double v = (double)out_ref[m*bs+j];
                double e = fabs((double)out_mine[m*bs+j] - v);
                max_gemm = std::max(max_gemm, e);
                sse_gemm += v*v; cnt_gemm++;
                maxrel_gemm = std::max(maxrel_gemm, e/(fabs(v)+1e-6));
            }
        }
        printf("  n=%-6d trials=%d done\n", n, trials);
    }
    double rms_gemm = sqrt(sse_gemm/cnt_gemm);
    printf("VERIFY EMITTED q4_0_16x1 GEMM VLEN=128  trials=%d  n in {64,256,4096,11008,14336}  nonneg=%d\n", trials, nonneg);
    printf("  GEMM  max_abs_err=%.3e  rms(ref)=%.3e  norm=%.3e  (max_per_elem_rel=%.3e)\n",
           max_gemm, rms_gemm, max_gemm/rms_gemm, maxrel_gemm);
    bool ok = (max_gemm/rms_gemm < 1e-4);
    printf("  VERDICT: %s\n", ok ? "PASS (compiler-emitted GEMM matches generic within fp32 rounding)" : "FAIL");
    return ok ? 0 : 1;
}
