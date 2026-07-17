// CROSS-CHECK: the COMPILER-EMITTED q4_1 repack GEMV vs the SEPARATELY-AUTHORED,
// separately-validated in-tree q4_1 block-dot (GgmlBlockDotQ41Q81Op). This kills
// residual circularity in the scalar self-test: instead of comparing the new
// kernel against a reference I hand-wrote, it compares it against a DIFFERENT
// compiler-emitted kernel (ggml_vec_dot_q4_1_q8_1) that predates this work and
// has its own lit + numeric coverage.
//
// For each of 16 weight rows: the block-dot dots that ONE row (plain block_q4_1)
// against the plain block_q8_1 activation, producing scalar s[row]. The repack
// GEMV dots ALL 16 rows at once from the block_q4_1x16 repacked layout. If both
// emit the same q4_1 arithmetic the 16 results must agree to fp rounding.
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
struct block_q8_1 { ggml_half d; ggml_half s; int8_t qs[QK8_1]; };          // 36 bytes
struct block_q4_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[QK8_1*8]; }; // 320 bytes

extern "C" {
void ggml_gemv_q4_1_16x1_q8_1(int n, float*, size_t, const void*, const void*, int, int); // repack GEMV
void tcrv_emitc_bd_bd(size_t n, float* s, const uint8_t* vx, const uint8_t* vy);           // block-dot (one row)
}

static block_q4_1x16 make_block(const block_q4_1* in) {
    block_q4_1x16 out;
    for (int i = 0; i < 16; i++) { out.d[i] = in[i].d; out.m[i] = in[i].m; }
    const int end = QK4_1 * 8;
    for (int i = 0; i < end; ++i) out.qs[i] = in[i % 16].qs[i / 16];
    return out;
}
static void make_random_q4_1_row(block_q4_1* row, int nblocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f), mn(-0.2f, 0.2f);
    for (int b = 0; b < nblocks; b++) {
        row[b].d = float_to_half(sc(rng)); row[b].m = float_to_half(mn(rng));
        for (int k = 0; k < QK4_1/2; k++) row[b].qs[k] = (uint8_t)(nib(rng) | (nib(rng) << 4));
    }
}

int main(int argc, char** argv) {
    int ns[] = {64, 256, 4096, 11008};
    const int NN = sizeof(ns)/sizeof(ns[0]);
    int trials = (argc > 1) ? atoi(argv[1]) : 60;
    std::mt19937 rng(424242);
    std::uniform_real_distribution<float> act(-2.0f, 2.0f);
    double maxerr = 0, sse = 0; long cnt = 0; bool ok = true;
    printf("CROSSCHECK q4_1 repack-GEMV vs in-tree q4_1 block-dot  trials=%d\n", trials);
    for (int ni = 0; ni < NN; ni++) {
        int n = ns[ni], nb = n / QK4_1;
        for (int t = 0; t < trials; t++) {
            std::vector<block_q4_1> w(16 * nb);
            for (int r = 0; r < 16; r++) make_random_q4_1_row(&w[r * nb], nb, rng);
            std::vector<block_q4_1x16> vx(nb);
            { block_q4_1 tmp[16];
              for (int x = 0; x < nb; x++) { for (int i = 0; i < 16; i++) tmp[i] = w[i * nb + x]; vx[x] = make_block(tmp); } }
            std::vector<float> af(n); for (int i = 0; i < n; i++) af[i] = act(rng);
            std::vector<block_q8_1> vy(nb);
            for (int b = 0; b < nb; b++) {
                float amax = 0; for (int j = 0; j < QK8_1; j++) amax = std::max(amax, fabsf(af[b*QK8_1+j]));
                float d = amax / 127.0f, id = d ? 1.0f/d : 0.0f; int isum = 0;
                for (int j = 0; j < QK8_1; j++) { int q = (int)lroundf(af[b*QK8_1+j]*id); vy[b].qs[j] = (int8_t)q; isum += q; }
                vy[b].d = float_to_half(d); vy[b].s = float_to_half(d * (float)isum);
            }
            // repack GEMV: 16 outputs at once.
            std::vector<float> gemv_out(16);
            ggml_gemv_q4_1_16x1_q8_1(n, gemv_out.data(), 0, vx.data(), vy.data(), 1, 16);
            // block-dot: dot each of the 16 plain q4_1 rows separately.
            for (int r = 0; r < 16; r++) {
                float bd = 0;
                tcrv_emitc_bd_bd((size_t)n, &bd, (const uint8_t*)&w[r * nb], (const uint8_t*)vy.data());
                double e = fabs((double)gemv_out[r] - (double)bd);
                maxerr = std::max(maxerr, e); sse += (double)bd*(double)bd; cnt++;
            }
        }
    }
    double rms = sqrt(sse/cnt);
    ok = (maxerr/rms < 1e-4);
    printf("  max_abs_err=%.3e  rms=%.3e  norm=%.3e  %s\n", maxerr, rms, maxerr/rms, ok ? "PASS" : "FAIL");
    printf("  VERDICT: %s\n", ok ? "PASS (repack GEMV agrees with separately-authored in-tree block-dot)" : "FAIL");
    return ok ? 0 : 1;
}
