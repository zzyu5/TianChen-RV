// White-box numeric harness for the COMPILER-EMITTED q4_0 16x1 repack GEMV.
//
// Drives the compiler-emitted GEMV (ggml_gemv_q4_0_16x1_q8_0, supplied by the
// thin ABI adapter over tcrv-opt's emitted symbol) against a scalar reference,
// over many n, random repacked inputs. Verdict = matches to fp32 rounding
// (norm = max_abs_err / rms(ref)) < 1e-4 (the GEMM got 1.34e-5).
//
// GEMV-only: the emitted node is the decode GEMV (single activation column).
// Activation format = PLAIN block_q8_0 (stride 34, scale +0, quants +2), NOT
// the GEMM's interleaved block_q8_0x4. Layout/repack/quantizer logic mirrors
// the proven verify_kernel.cpp (GEMV half) + verify_emitted_gemm.cpp norm bar.
//
// REFERENCE: inline scalar copy of ggml's own ggml_gemv_q4_0_16x1_q8_0_generic
// (repack.cpp:1370). Verbatim block-as-lane decode of the ^0x88-baked repacked
// weight qs + plain q8_0 activation quants, integer MAC then fp scale-fold —
// the exact arithmetic the emitted kernel performs (so norm sits at fp rounding,
// NOT q8 quantization error). This dots the QUANTIZED activation, never raw fp.
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
struct block_q8_0 { ggml_half d; int8_t qs[QK8_0]; };           // 34 bytes (gemv activation)
struct block_q4_0x16 { ggml_half d[16]; int8_t qs[QK8_0*8]; };  // 288 bytes total

extern "C" {
// COMPILER-EMITTED GEMV (via emitted_adapter_gemv.cpp).
void ggml_gemv_q4_0_16x1_q8_0(int n, float*, size_t, const void*, const void*, int, int);
}

// Inline of static make_block_q4_0x16(in, 1) from repack.cpp (the ^0x88 bake).
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

// Inline scalar reference: verbatim copy of ggml_gemv_q4_0_16x1_q8_0_generic
// (repack.cpp:1370). Consumes the SAME repacked vx + plain q8_0 vy the kernel
// consumes; integer block-as-lane MAC then fp16-scale fold.
static void gemv_ref(int n, float* s, const void* vx, const void* vy, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;
    float sumf[16];
    int sumi;
    const block_q8_0* a_ptr = (const block_q8_0*) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x16* b_ptr = (const block_q4_0x16*) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0f;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * half_to_float(b_ptr[l].d[j]) * half_to_float(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
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
    // Stress sizes: the model's exact rows plus small/odd sizes, as the task asks.
    int ns[] = {64, 256, 4096, 11008, 14336};
    const int NN = sizeof(ns)/sizeof(ns[0]);
    int trials = (argc > 1) ? atoi(argv[1]) : 200;
    int nonneg = (argc > 2) ? atoi(argv[2]) : 0;
    std::mt19937 rng(20260622);
    std::uniform_real_distribution<float> act(nonneg ? 0.1f : -2.0f, 2.0f);

    // Multi-column-group: real decode runs nc=336/352 (21-22 groups). NC>16
    // exercises the emitted GEMV's OUTER nc/16 loop + cross-group pointer math
    // (weight vx += x*nb*288, output s += x*16) that NC=16 (one group, x=0) never
    // touches. NC=336 mirrors the live decode width exactly.
    int ncs[] = {16, 32, 64, 336};
    const int NNC = sizeof(ncs)/sizeof(ncs[0]);
    bool all_ok = true;
    printf("VERIFY EMITTED q4_0_16x1 GEMV VLEN=128  trials=%d  n in {64,256,4096,11008,14336}  nonneg=%d\n", trials, nonneg);
    for (int ci = 0; ci < NNC; ci++) {
        const int NC = ncs[ci];
        const int ngrp = NC / 16;
        double max_gemv = 0, sse_gemv = 0, maxrel_gemv = 0;
        long cnt_gemv = 0;
        for (int ni = 0; ni < NN; ni++) {
            int n = ns[ni];
            int nb = n / QK4_0;
            for (int t = 0; t < trials; t++) {
                // weights: NC rows x nb blocks (plain q4_0), repacked GROUP-major
                // (group g holds rows g*16..g*16+15; vx[g*nb + x] = make_block).
                std::vector<block_q4_0> w(NC * nb);
                for (int r = 0; r < NC; r++) make_random_q4_0_row(&w[r * nb], nb, rng, nonneg);
                std::vector<block_q4_0x16> vx(ngrp * nb);
                for (int g = 0; g < ngrp; g++) {
                    block_q4_0 tmp[16];
                    for (int x = 0; x < nb; x++) {
                        for (int i = 0; i < 16; i++) tmp[i] = w[(g * 16 + i) * nb + x];
                        vx[g * nb + x] = make_block(tmp);
                    }
                }
                // ONE activation column (gemv) -> PLAIN q8_0 (block_q8_0, stride 34),
                // shared across all NC weight columns.
                std::vector<float> af(n);
                for (int i = 0; i < n; i++) af[i] = act(rng);
                std::vector<block_q8_0> vy(nb);
                for (int b = 0; b < nb; b++) {
                    float amax = 0; for (int j = 0; j < QK8_0; j++) amax = std::max(amax, fabsf(af[b*QK8_0+j]));
                    float d = amax / 127.0f, id = d ? 1.0f/d : 0.0f;
                    vy[b].d = float_to_half(d);
                    for (int j = 0; j < QK8_0; j++) vy[b].qs[j] = (int8_t)lroundf(af[b*QK8_0+j]*id);
                }
                std::vector<float> out_mine(NC), out_ref(NC);
                ggml_gemv_q4_0_16x1_q8_0(n, out_mine.data(), 0, vx.data(), vy.data(), 1, NC);
                gemv_ref(n, out_ref.data(), vx.data(), vy.data(), NC);
                for (int j = 0; j < NC; j++) {
                    double v = (double)out_ref[j];
                    double e = fabs((double)out_mine[j] - v);
                    max_gemv = std::max(max_gemv, e);
                    sse_gemv += v*v; cnt_gemv++;
                    maxrel_gemv = std::max(maxrel_gemv, e/(fabs(v)+1e-6));
                }
            }
        }
        double rms_gemv = sqrt(sse_gemv/cnt_gemv);
        bool ok = (max_gemv/rms_gemv < 1e-4);
        all_ok = all_ok && ok;
        printf("  NC=%-4d (%2d grp)  max_abs_err=%.3e  rms(ref)=%.3e  norm=%.3e  rel=%.3e  %s\n",
               NC, ngrp, max_gemv, rms_gemv, max_gemv/rms_gemv, maxrel_gemv, ok ? "PASS" : "FAIL");
    }
    printf("  VERDICT: %s\n", all_ok ? "PASS (compiler-emitted GEMV matches generic across NC=16/32/64/336 within fp32 rounding)" : "FAIL");
    return all_ok ? 0 : 1;
}
