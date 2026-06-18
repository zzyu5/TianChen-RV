// Standalone white-box correctness harness for the VLEN=128 q4_0 16x1 repack kernels.
// Compares ggml_gemm/gemv_q4_0_16x1_q8_0 (our new VLEN=128 path) against the
// obviously-correct scalar *_generic twins, at the model's exact n (4096, 11008).
// This is a complete test: the only new VLEN-specific code is the two kernels;
// repack-transform and the q8_0_4x1 quantizer are shared/scalar.
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

// fp16 helpers (round-to-nearest-even) so weights/scales survive a ggml_half round-trip.
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
    if (exp <= 0) return (ggml_half)sign; // flush tiny to zero (fine for test data)
    if (exp >= 0x1f) return (ggml_half)(sign | 0x7c00);
    // round to nearest even
    uint32_t h = sign | (exp << 10) | (man >> 13);
    if ((man & 0x1000) && ((man & 0x2fff) || (h & 1))) h++;
    return (ggml_half)h;
}

struct block_q4_0 { ggml_half d; uint8_t qs[QK4_0/2]; };
struct block_q8_0 { ggml_half d; int8_t qs[QK8_0]; };
// repacked weight: 16 blocks interleaved
struct block_q4_0x16 { ggml_half d[16]; int8_t qs[QK8_0*8]; }; // 256 bytes qs
// repacked activation: 4 cols interleaved (blocklen=1)
struct block_q8_0x4 { ggml_half d[4]; int8_t qs[QK8_0*4]; };

extern "C" {
void ggml_gemv_q4_0_16x1_q8_0(int n, float*, size_t, const void*, const void*, int, int);
void ggml_gemv_q4_0_16x1_q8_0_generic(int n, float*, size_t, const void*, const void*, int, int);
void ggml_gemm_q4_0_16x1_q8_0(int n, float*, size_t, const void*, const void*, int, int);
void ggml_gemm_q4_0_16x1_q8_0_generic(int n, float*, size_t, const void*, const void*, int, int);
void ggml_quantize_mat_q8_0_4x1(const float*, void*, int64_t);
}

// Inline of static make_block_q4_0x16(in, 1) from repack.cpp.
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
    // q4_0 decoded value = (nibble - 8). For nonneg decoded weights, draw nibble in [8,15].
    std::uniform_int_distribution<int> nib(nonneg ? 8 : 0, 15);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    for (int b = 0; b < nblocks; b++) {
        row[b].d = float_to_half(sc(rng));
        for (int k = 0; k < QK4_0/2; k++) row[b].qs[k] = (uint8_t)(nib(rng) | (nib(rng) << 4));
    }
}

int main(int argc, char** argv) {
    int ns[] = {4096, 11008};
    int trials = (argc > 1) ? atoi(argv[1]) : 200;
    int nonneg = (argc > 2) ? atoi(argv[2]) : 0; // discriminator mode: all-nonnegative data
    std::mt19937 rng(20260618);
    std::uniform_real_distribution<float> act(nonneg ? 0.1f : -2.0f, 2.0f);

    // metric: max absolute error, and accumulate sum-of-squares of reference for RMS norm.
    double max_gemv = 0, max_gemm = 0;          // max abs err
    double sse_gemv = 0, sse_gemm = 0;          // sum ref^2
    long   cnt_gemv = 0, cnt_gemm = 0;
    double maxrel_gemv = 0, maxrel_gemm = 0;    // per-element rel (for reference only)
    for (int ni = 0; ni < 2; ni++) {
        int n = ns[ni];
        int nb = n / QK4_0;
        const int NC = 16;          // one 16-row group
        // GEMV: 1 activation column.
        for (int t = 0; t < trials; t++) {
            // weights: 16 rows x nb blocks (plain q4_0), then repack.
            std::vector<block_q4_0> w(16 * nb);
            for (int r = 0; r < 16; r++) make_random_q4_0_row(&w[r * nb], nb, rng, nonneg);
            std::vector<block_q4_0x16> vx(nb);
            block_q4_0 tmp[16];
            for (int x = 0; x < nb; x++) {
                for (int i = 0; i < 16; i++) tmp[i] = w[i * nb + x];
                vx[x] = make_block(tmp);
            }
            // 1 activation column -> q8_0 plain (gemv uses block_q8_0).
            std::vector<float> af(n);
            for (int i = 0; i < n; i++) af[i] = act(rng);
            std::vector<block_q8_0> vy(nb);
            for (int b = 0; b < nb; b++) {
                float amax = 0; for (int j = 0; j < QK8_0; j++) amax = std::max(amax, fabsf(af[b*QK8_0+j]));
                float d = amax / 127.0f, id = d ? 1.0f/d : 0.0f;
                vy[b].d = float_to_half(d);
                for (int j = 0; j < QK8_0; j++) vy[b].qs[j] = (int8_t)lroundf(af[b*QK8_0+j]*id);
            }
            float out_mine[16], out_ref[16];
            ggml_gemv_q4_0_16x1_q8_0(n, out_mine, 0, vx.data(), vy.data(), 1, NC);
            ggml_gemv_q4_0_16x1_q8_0_generic(n, out_ref, 0, vx.data(), vy.data(), 1, NC);
            for (int j = 0; j < 16; j++) {
                double e = fabs((double)out_mine[j] - (double)out_ref[j]);
                max_gemv = std::max(max_gemv, e);
                sse_gemv += (double)out_ref[j]*(double)out_ref[j]; cnt_gemv++;
                maxrel_gemv = std::max(maxrel_gemv, e/(fabs((double)out_ref[j])+1e-6));
            }
        }
        // GEMM: 4 activation columns (nr=4), block_q8_0x4 via the real quantizer.
        for (int t = 0; t < trials; t++) {
            std::vector<block_q4_0> w(16 * nb);
            for (int r = 0; r < 16; r++) make_random_q4_0_row(&w[r * nb], nb, rng, nonneg);
            std::vector<block_q4_0x16> vx(nb);
            block_q4_0 tmp[16];
            for (int x = 0; x < nb; x++) {
                for (int i = 0; i < 16; i++) tmp[i] = w[i * nb + x];
                vx[x] = make_block(tmp);
            }
            // 4 activation rows, layout x[row*n + col]; quantizer expects that.
            std::vector<float> af(4 * n);
            for (int i = 0; i < 4*n; i++) af[i] = act(rng);
            std::vector<block_q8_0x4> vy(nb);
            ggml_quantize_mat_q8_0_4x1(af.data(), vy.data(), n);
            float out_mine[4*16], out_ref[4*16];
            const size_t bs = 16;
            ggml_gemm_q4_0_16x1_q8_0(n, out_mine, bs, vx.data(), vy.data(), 4, NC);
            ggml_gemm_q4_0_16x1_q8_0_generic(n, out_ref, bs, vx.data(), vy.data(), 4, NC);
            for (int m = 0; m < 4; m++) for (int j = 0; j < 16; j++) {
                double v=(double)out_ref[m*bs+j];
                double e = fabs((double)out_mine[m*bs+j] - v);
                max_gemm = std::max(max_gemm, e);
                sse_gemm += v*v; cnt_gemm++;
                maxrel_gemm = std::max(maxrel_gemm, e/(fabs(v)+1e-6));
            }
        }
    }
    double rms_gemv = sqrt(sse_gemv/cnt_gemv), rms_gemm = sqrt(sse_gemm/cnt_gemm);
    printf("VERIFY q4_0_16x1 VLEN=128  trials=%d  n in {4096,11008}  nonneg=%d\n", trials, nonneg);
    printf("  GEMV  max_abs_err=%.3e  rms(ref)=%.3e  norm=%.3e  (max_per_elem_rel=%.3e)\n",
           max_gemv, rms_gemv, max_gemv/rms_gemv, maxrel_gemv);
    printf("  GEMM  max_abs_err=%.3e  rms(ref)=%.3e  norm=%.3e  (max_per_elem_rel=%.3e)\n",
           max_gemm, rms_gemm, max_gemm/rms_gemm, maxrel_gemm);
    bool ok = (max_gemv/rms_gemv < 1e-4) && (max_gemm/rms_gemm < 1e-4);
    printf("  VERDICT: %s\n", ok ? "PASS (matches generic within fp32 rounding)" : "FAIL");
    return ok ? 0 : 1;
}
