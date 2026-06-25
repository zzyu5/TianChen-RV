// White-box numeric harness for the COMPILER-EMITTED q4_1 16x1 repack GEMV.
//
// Drives the compiler-emitted q4_1 GEMV (ggml_gemv_q4_1_16x1_q8_1, supplied by
// the thin ABI adapter over tcrv-opt's emitted symbol) against a scalar q4_1
// reference, over many n, random repacked inputs. Verdict = matches to fp32
// rounding (norm = max_abs_err / rms(ref)) < 1e-4.
//
// q4_1 (Family B, scale+MIN, asymmetric) vs q4_0:
//   * nibbles decode UNSIGNED [0,15] (NO offset-binary -8 / ^0x88 bias);
//   * each block carries a per-row MIN m_x and a precomputed activation
//     scaled-sum s_y = d_y * sum(q8), and the fold is
//        sumf += (d_x*d_y)*sumi + m_x*s_y.
// The reference is a verbatim block-as-lane statement of EXACTLY the arithmetic
// the emitted kernel performs (unsigned nibble MAC then the dual-scale fold), so
// norm sits at fp rounding, NOT q8 quantization error. It dots the QUANTIZED
// activation, never raw fp.
//
// block_q4_1x16 (320 bytes): { fp16 d[16]; fp16 m[16]; uint8 qs[256]; }
//   qs is the SAME 16-way interleave as q4_0x16 but stored RAW (no ^0x88 bake):
//   byte i = block(i%16).qs[i/16].
// block_q8_1 (36 bytes): { fp16 d; fp16 s; int8 qs[32]; }  (s = d * sum(qs)).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK4_1 32
#define QK8_1 32

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

struct block_q4_1 { ggml_half d; ggml_half m; uint8_t qs[QK4_1/2]; };       // 20 bytes
struct block_q8_1 { ggml_half d; ggml_half s; int8_t qs[QK8_1]; };          // 36 bytes (gemv activation)
struct block_q4_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[QK8_1*8]; }; // 320 bytes

extern "C" {
// COMPILER-EMITTED q4_1 GEMV (via emitted_adapter_gemv_q4_1.cpp).
void ggml_gemv_q4_1_16x1_q8_1(int n, float*, size_t, const void*, const void*, int, int);
}

// Repack: 16 plain q4_1 blocks -> one block_q4_1x16. The nibbles are stored RAW
// (NO ^0x88) -- the q4_1 bias lives in the per-block MIN scale.
static block_q4_1x16 make_block(const block_q4_1* in) {
    block_q4_1x16 out;
    for (int i = 0; i < 16; i++) { out.d[i] = in[i].d; out.m[i] = in[i].m; }
    const int end = QK4_1 * 8 / 1; // 256
    for (int i = 0; i < end; ++i) {
        int src_id = i % 16, src_offset = i / 16;
        out.qs[i] = in[src_id].qs[src_offset]; // RAW, no bake
    }
    return out;
}

// Inline scalar reference: the EXACT block-as-lane unsigned-nibble MAC + dual
// scale fold the emitted kernel performs. Consumes the SAME repacked vx + plain
// q8_1 vy the kernel consumes.
//   sumi = sum_i ( (qs&0x0F)*a.qs[i] + (qs>>4)*a.qs[16+i] )   (UNSIGNED nibbles)
//   sumf[j] += (d_x[j]*d_y)*sumi + m_x[j]*s_y                 (s_y = a.s)
static void gemv_ref(int n, float* s, const void* vx, const void* vy, int nc) {
    const int qk = QK8_1;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const block_q8_1* a_ptr = (const block_q8_1*) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_1x16* b_ptr = (const block_q4_1x16*) vx + (x * nb);
        float sumf[16];
        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0f;
        for (int l = 0; l < nb; l++) {
            const float d_y = half_to_float(a_ptr[l].d);
            const float s_y = half_to_float(a_ptr[l].s);
            for (int j = 0; j < ncols_interleaved; j++) {
                int sumi = 0;
                for (int k = 0; k < qk / 2; k++) {
                    const uint8_t b = b_ptr[l].qs[k * ncols_interleaved + j];
                    const int v0 = b & 0x0F;        // low nibble [0,15]
                    const int v1 = b >> 4;          // high nibble [0,15]
                    sumi += v0 * a_ptr[l].qs[k] + v1 * a_ptr[l].qs[k + qk / 2];
                }
                const float d_x = half_to_float(b_ptr[l].d[j]);
                const float m_x = half_to_float(b_ptr[l].m[j]);
                sumf[j] += (d_x * d_y) * (float)sumi + m_x * s_y;
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

static void make_random_q4_1_row(block_q4_1* row, int nblocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    std::uniform_real_distribution<float> mn(-0.2f, 0.2f);
    for (int b = 0; b < nblocks; b++) {
        row[b].d = float_to_half(sc(rng));
        row[b].m = float_to_half(mn(rng));
        for (int k = 0; k < QK4_1/2; k++) row[b].qs[k] = (uint8_t)(nib(rng) | (nib(rng) << 4));
    }
}

int main(int argc, char** argv) {
    int ns[] = {64, 256, 4096, 11008, 14336};
    const int NN = sizeof(ns)/sizeof(ns[0]);
    int trials = (argc > 1) ? atoi(argv[1]) : 100;
    std::mt19937 rng(20260623);
    std::uniform_real_distribution<float> act(-2.0f, 2.0f);

    int ncs[] = {16, 32, 64, 336};
    const int NNC = sizeof(ncs)/sizeof(ncs[0]);
    bool all_ok = true;
    printf("VERIFY EMITTED q4_1_16x1 GEMV  trials=%d  n in {64,256,4096,11008,14336}\n", trials);
    for (int ci = 0; ci < NNC; ci++) {
        const int NC = ncs[ci];
        const int ngrp = NC / 16;
        double max_g = 0, sse_g = 0, maxrel_g = 0;
        long cnt_g = 0;
        for (int ni = 0; ni < NN; ni++) {
            int n = ns[ni];
            int nb = n / QK4_1;
            for (int t = 0; t < trials; t++) {
                // weights: NC rows x nb blocks (plain q4_1), repacked GROUP-major.
                std::vector<block_q4_1> w(NC * nb);
                for (int r = 0; r < NC; r++) make_random_q4_1_row(&w[r * nb], nb, rng);
                std::vector<block_q4_1x16> vx(ngrp * nb);
                for (int g = 0; g < ngrp; g++) {
                    block_q4_1 tmp[16];
                    for (int x = 0; x < nb; x++) {
                        for (int i = 0; i < 16; i++) tmp[i] = w[(g * 16 + i) * nb + x];
                        vx[g * nb + x] = make_block(tmp);
                    }
                }
                // ONE activation column (gemv) -> PLAIN q8_1 (block_q8_1, stride 36).
                std::vector<float> af(n);
                for (int i = 0; i < n; i++) af[i] = act(rng);
                std::vector<block_q8_1> vy(nb);
                for (int b = 0; b < nb; b++) {
                    float amax = 0; for (int j = 0; j < QK8_1; j++) amax = std::max(amax, fabsf(af[b*QK8_1+j]));
                    float d = amax / 127.0f, id = d ? 1.0f/d : 0.0f;
                    int isum = 0;
                    for (int j = 0; j < QK8_1; j++) { int q = (int)lroundf(af[b*QK8_1+j]*id); vy[b].qs[j] = (int8_t)q; isum += q; }
                    vy[b].d = float_to_half(d);
                    vy[b].s = float_to_half(d * (float)isum); // s = d * sum(q8)
                }
                std::vector<float> out_mine(NC), out_ref(NC);
                ggml_gemv_q4_1_16x1_q8_1(n, out_mine.data(), 0, vx.data(), vy.data(), 1, NC);
                gemv_ref(n, out_ref.data(), vx.data(), vy.data(), NC);
                for (int j = 0; j < NC; j++) {
                    double v = (double)out_ref[j];
                    double e = fabs((double)out_mine[j] - v);
                    max_g = std::max(max_g, e);
                    sse_g += v*v; cnt_g++;
                    maxrel_g = std::max(maxrel_g, e/(fabs(v)+1e-6));
                }
            }
        }
        double rms_g = sqrt(sse_g/cnt_g);
        bool ok = (max_g/rms_g < 1e-4);
        all_ok = all_ok && ok;
        printf("  NC=%-4d (%2d grp)  max_abs_err=%.3e  rms(ref)=%.3e  norm=%.3e  rel=%.3e  %s\n",
               NC, ngrp, max_g, rms_g, max_g/rms_g, maxrel_g, ok ? "PASS" : "FAIL");
    }
    printf("  VERDICT: %s\n", all_ok ? "PASS (compiler-emitted q4_1 GEMV matches scalar reference within fp32 rounding)" : "FAIL");
    return all_ok ? 0 : 1;
}
