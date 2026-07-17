// Micro-benchmark for the COMPILER-EMITTED q4_0 16x1 repack GEMV on K1 (VLEN=256).
// Times one variant (the emitted symbol pulled in via emitted_adapter_gemv.cpp ->
// emitted-repack-gemv.cpp). Decode-like shape: nc=336, n=4096. Best-of-N over many
// reps. Build BOTH variants with the IDENTICAL march (rv64gcv_zvfh) so the only
// difference is the emitted strip width (8-lane x2 vs 16-lane x1).
//
// Layout setup mirrors verify_emitted_gemv.cpp exactly (repacked GROUP-major vx +
// plain block_q8_0 vy). No correctness check here -- correctness is the separate
// verify harness; this is timing only.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>

#define QK4_0 32
#define QK8_0 32
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
struct block_q4_0 { ggml_half d; uint8_t qs[QK4_0/2]; };
struct block_q8_0 { ggml_half d; int8_t qs[QK8_0]; };
struct block_q4_0x16 { ggml_half d[16]; int8_t qs[QK8_0*8]; };

extern "C" { void ggml_gemv_q4_0_16x1_q8_0(int n, float*, size_t, const void*, const void*, int, int); }

static block_q4_0x16 make_block(const block_q4_0* in) {
    block_q4_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    const int end = QK4_0 * 8;
    const uint8_t xm = 0x88;
    for (int i = 0; i < end; ++i) { int sid = i % 16, soff = i / 16; out.qs[i] = (int8_t)(in[sid].qs[soff] ^ xm); }
    return out;
}
static void make_row(block_q4_0* row, int nb, std::mt19937& rng) {
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    for (int b = 0; b < nb; b++) { row[b].d = float_to_half(sc(rng)); for (int k = 0; k < QK4_0/2; k++) row[b].qs[k] = (uint8_t)(nib(rng) | (nib(rng) << 4)); }
}
static inline double now_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9 + ts.tv_nsec; }

int main(int argc, char** argv) {
    int n  = (argc > 1) ? atoi(argv[1]) : 4096;
    int NC = (argc > 2) ? atoi(argv[2]) : 336;
    int reps = (argc > 3) ? atoi(argv[3]) : 2000;
    int bestof = (argc > 4) ? atoi(argv[4]) : 9;
    const char* tag = (argc > 5) ? argv[5] : "?";
    int nb = n / QK4_0;
    int ngrp = NC / 16;
    std::mt19937 rng(20260622);
    std::uniform_real_distribution<float> act(-2.0f, 2.0f);

    std::vector<block_q4_0> w(NC * nb);
    for (int r = 0; r < NC; r++) make_row(&w[r*nb], nb, rng);
    std::vector<block_q4_0x16> vx(ngrp * nb);
    for (int g = 0; g < ngrp; g++) { block_q4_0 tmp[16];
        for (int x = 0; x < nb; x++) { for (int i = 0; i < 16; i++) tmp[i] = w[(g*16+i)*nb + x]; vx[g*nb + x] = make_block(tmp); } }
    std::vector<float> af(n); for (int i = 0; i < n; i++) af[i] = act(rng);
    std::vector<block_q8_0> vy(nb);
    for (int b = 0; b < nb; b++) { float amax=0; for (int j=0;j<QK8_0;j++) amax=std::max(amax,fabsf(af[b*QK8_0+j]));
        float d = amax/127.0f, id = d ? 1.0f/d : 0.0f; vy[b].d = float_to_half(d);
        for (int j=0;j<QK8_0;j++) vy[b].qs[j] = (int8_t)lroundf(af[b*QK8_0+j]*id); }
    std::vector<float> out(NC);

    // warm up
    for (int i=0;i<50;i++) ggml_gemv_q4_0_16x1_q8_0(n, out.data(), 0, vx.data(), vy.data(), 1, NC);

    double best = 1e30;
    volatile float sink = 0;
    for (int b = 0; b < bestof; b++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) ggml_gemv_q4_0_16x1_q8_0(n, out.data(), 0, vx.data(), vy.data(), 1, NC);
        double t1 = now_ns();
        sink += out[0] + out[NC-1];
        double per = (t1 - t0) / reps;
        if (per < best) best = per;
    }
    printf("BENCH variant=%-5s n=%d nc=%d reps=%d bestof=%d  best=%.1f ns/call  (sink=%g)\n",
           tag, n, NC, reps, bestof, best, (double)sink);
    return 0;
}
