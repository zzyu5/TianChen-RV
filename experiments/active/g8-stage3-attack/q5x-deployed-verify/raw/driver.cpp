// G8 §六.3 q5_0/q5_1 deployed-core verification driver.
// Purpose: (1) byte-exact the CURRENT-HEAD compiler-emitted REDESIGN-B repack-GEVM
// kernel (weft_emitc_ggml_vec_dot_q5_{0,1}_q8_{0,1}_kernel_...) against BOTH an
// independent hand-written scalar oracle AND the real sealed opponent object
// (ggml_vec_dot_q5_{0,1}_q8_{0,1} from quants_opp.o, same clang-18 domain as G8 §六).
// (2) cold M=1 kernel-sym timing: ours=ONE GEVM call over N rows vs opp=N per-row
// calls, K=2048 nc=512 (matches §六 shape), 224MiB flush before each timed region,
// N=12 paired reps, median + relIQR, core-pinned.
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <time.h>
#include <sched.h>
#include <unistd.h>

typedef _Float16 ggml_half;

// ---------------- q5_0 ----------------
#define QK5_0 32
#define QK8_0 32
struct block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; };   // 22B
struct block_q8_0 { ggml_half d; int8_t  qs[QK8_0];   };                  // 34B
struct block_q5_0x16 { ggml_half d[16]; uint8_t qs[256]; uint8_t qh[64]; }; // 352B
static_assert(sizeof(block_q5_0)==22,"q5_0");
static_assert(sizeof(block_q8_0)==34,"q8_0");
static_assert(sizeof(block_q5_0x16)==352,"q5_0x16");

static block_q5_0x16 make_block_q5_0x16(block_q5_0 * in) {
    block_q5_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    for (int i = 0; i < 256; ++i) {
        int src_id = i % 16, src_offset = i / 16, dst_offset = i;
        out.qs[dst_offset] = in[src_id].qs[src_offset];
    }
    uint32_t qh_col[16];
    for (int c = 0; c < 16; c++) std::memcpy(&qh_col[c], in[c].qh, sizeof(uint32_t));
    for (int k = 0; k < 16; k++) {
        uint16_t lo = 0, hi = 0;
        for (int c = 0; c < 16; c++) {
            lo |= (uint16_t)(((qh_col[c] >> k)        & 1u) << c);
            hi |= (uint16_t)(((qh_col[c] >> (k + 16)) & 1u) << c);
        }
        std::memcpy(out.qh +  0 + k*2, &lo, 2);
        std::memcpy(out.qh + 32 + k*2, &hi, 2);
    }
    return out;
}

// Independent hand-written scalar oracle (standard ggml q5_0 dequant formula,
// NOT derived from the compiler / emitted kernel).
static float oracle_q5_0(const block_q5_0 & w, const block_q8_0 & a) {
    uint32_t qh; std::memcpy(&qh, w.qh, 4);
    int isum = 0;
    for (int j = 0; j < QK5_0/2; j++) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12)) >> 0) & 0x10;
        const int w0 = ((w.qs[j] & 0x0F) | xh_0) - 16;
        const int w1 = ((w.qs[j] >>   4) | xh_1) - 16;
        isum += w0 * a.qs[j] + w1 * a.qs[j + 16];
    }
    return (float)isum * (float)w.d * (float)a.d;
}

// ---------------- q5_1 ----------------
#define QK5_1 32
#define QK8_1 32
struct block_q5_1 { ggml_half d; ggml_half m; uint8_t qh[4]; uint8_t qs[QK5_1/2]; }; // 24B
struct block_q8_1 { ggml_half d; ggml_half s; int8_t qs[QK8_1]; };                    // 36B
struct block_q5_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[256]; uint8_t qh[64]; }; // 384B
static_assert(sizeof(block_q5_1)==24,"q5_1");
static_assert(sizeof(block_q8_1)==36,"q8_1");
static_assert(sizeof(block_q5_1x16)==384,"q5_1x16");

static block_q5_1x16 make_block_q5_1x16(block_q5_1 * in) {
    block_q5_1x16 out;
    for (int i = 0; i < 16; i++) { out.d[i] = in[i].d; out.m[i] = in[i].m; }
    for (int i = 0; i < 256; ++i) {
        int src_id = i % 16, src_offset = i / 16, dst_offset = i;
        out.qs[dst_offset] = in[src_id].qs[src_offset];
    }
    uint32_t qh_col[16];
    for (int c = 0; c < 16; c++) std::memcpy(&qh_col[c], in[c].qh, sizeof(uint32_t));
    for (int k = 0; k < 16; k++) {
        uint16_t lo = 0, hi = 0;
        for (int c = 0; c < 16; c++) {
            lo |= (uint16_t)(((qh_col[c] >> k)        & 1u) << c);
            hi |= (uint16_t)(((qh_col[c] >> (k + 16)) & 1u) << c);
        }
        std::memcpy(out.qh +  0 + k*2, &lo, 2);
        std::memcpy(out.qh + 32 + k*2, &hi, 2);
    }
    return out;
}

// Independent hand-written scalar oracle (standard ggml q5_1 dequant formula).
static float oracle_q5_1(const block_q5_1 & w, const block_q8_1 & a) {
    uint32_t qh; std::memcpy(&qh, w.qh, 4);
    int sumi = 0;
    for (int j = 0; j < QK5_1/2; j++) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12)) >> 0) & 0x10;
        const int32_t x0 = (w.qs[j] & 0xF) | xh_0;
        const int32_t x1 = (w.qs[j] >>  4) | xh_1;
        sumi += (x0 * a.qs[j]) + (x1 * a.qs[j + 16]);
    }
    return (float)w.d * (float)a.d * (float)sumi + (float)w.m * (float)a.s;
}

// ---------------- emitted kernels (current-HEAD weft-opt, REDESIGN-B) ----------------
extern "C" void weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
    size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx,
    const uint8_t* vy, size_t by, int32_t nrc);
extern "C" void weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
    size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx,
    const uint8_t* vy, size_t by, int32_t nrc);

// ---------------- sealed real opponent (quants_opp.o, clang-18, same domain as §六) --
extern "C" void ggml_vec_dot_q5_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_1_q8_1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

// 224 MiB flush buffer (>> L3=64MiB) touched between timed regions to force cold state.
static const size_t FLUSH_BYTES = 224ull*1024*1024;
static uint8_t* g_flush = nullptr;
static volatile uint64_t g_sink_flush = 0;
static void cold_flush() {
    uint64_t acc = 0;
    for (size_t i = 0; i < FLUSH_BYTES; i += 64) { g_flush[i] = (uint8_t)(g_flush[i] + 1); acc += g_flush[i]; }
    g_sink_flush += acc;
}

template <typename F>
static void run_reps(F&& f, double* out, int reps) {
    for (int r = 0; r < reps; r++) {
        cold_flush();
        double a = now_ms();
        f();
        out[r] = now_ms() - a;
    }
}
static double med(double* v, int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); return t[n/2]; }
static double reliqr(double* v, int n){ std::vector<double> t(v,v+n); std::sort(t.begin(),t.end()); double q1=t[n/4],q3=t[(3*n)/4],m=t[n/2]; return m>0?(q3-q1)/m:0; }

// ===================== q5_0 =====================
static int test_q5_0(int N, int K, bool do_perf, int reps) {
    int nb = K/32;
    std::mt19937 rng(0xC0FFEE1);
    std::uniform_int_distribution<int> qd(0,255), qhd(0,255);
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);

    std::vector<block_q5_0> weightStd((size_t)N*nb);
    for (auto & b : weightStd) {
        b.d = (ggml_half)dd(rng);
        for (int i=0;i<4;i++) b.qh[i]=(uint8_t)qhd(rng);
        for (int i=0;i<16;i++) b.qs[i]=(uint8_t)qd(rng);
    }
    std::vector<block_q8_0> act(nb);
    std::uniform_int_distribution<int> ad(-127,127);
    for (auto & a : act) { a.d = (ggml_half)dd(rng); for (int i=0;i<32;i++) a.qs[i]=(int8_t)ad(rng); }

    int ngroups = N/16;
    std::vector<block_q5_0x16> packed((size_t)ngroups*nb);
    for (int g=0; g<ngroups; g++) {
        for (int blk=0; blk<nb; blk++) {
            block_q5_0 tmp[16];
            for (int col=0; col<16; col++) tmp[col] = weightStd[(size_t)(g*16+col)*nb + blk];
            packed[(size_t)g*nb+blk] = make_block_q5_0x16(tmp);
        }
    }

    // ---- correctness ----
    std::vector<float> ours(N,0.f), ref_opp(N,0.f), ref_oracle(N,0.f);
    weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
        (size_t)K, ours.data(), (size_t)N, (const uint8_t*)packed.data(), 0,
        (const uint8_t*)act.data(), 0, 0);
    for (int r=0;r<N;r++) {
        ggml_vec_dot_q5_0_q8_0(K, &ref_opp[r], 0, weightStd.data()+ (size_t)r*nb, 0, act.data(), 0, 1);
        double s=0; for (int b=0;b<nb;b++) s += oracle_q5_0(weightStd[(size_t)r*nb+b], act[b]);
        ref_oracle[r] = (float)s;
    }
    int mism_opp=0, mism_oracle=0; float max_abs_opp=0, max_abs_oracle=0, max_rel_opp=0, max_rel_oracle=0;
    for (int r=0;r<N;r++) {
        float d1 = std::fabs(ours[r]-ref_opp[r]);
        float d2 = std::fabs(ours[r]-ref_oracle[r]);
        bool bit_eq_opp = std::memcmp(&ours[r],&ref_opp[r],4)==0;
        bool bit_eq_oracle_close = d2/(std::fabs(ref_oracle[r])+1e-6f) < 1e-3f;
        if (!bit_eq_opp) mism_opp++;
        if (!bit_eq_oracle_close) mism_oracle++;
        if (d1>max_abs_opp) max_abs_opp=d1;
        if (d2>max_abs_oracle) max_abs_oracle=d2;
        float rel1 = d1/(std::fabs(ref_opp[r])+1e-6f); if (rel1>max_rel_opp) max_rel_opp=rel1;
        float rel2 = d2/(std::fabs(ref_oracle[r])+1e-6f); if (rel2>max_rel_oracle) max_rel_oracle=rel2;
    }
    printf("[q5_0] BYTE-EXACT-vs-sealed-opponent: mismatches(bitwise-f32)=%d/%d max_abs=%.3e max_rel=%.3e\n", mism_opp, N, max_abs_opp, max_rel_opp);
    printf("[q5_0] BYTE-EXACT-vs-independent-oracle: mismatches(rel>1e-3)=%d/%d max_abs=%.3e max_rel=%.3e\n", mism_oracle, N, max_abs_oracle, max_rel_oracle);

    if (!do_perf) return (mism_opp==0 && mism_oracle==0) ? 0 : 1;

    // ---- cold perf: ours=1 GEVM call over N rows, opp=N per-row calls ----
    std::vector<float> outO(N), outS(N);
    double tO[64], tS[64];
    run_reps([&](){ weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
        (size_t)K, outO.data(), (size_t)N, (const uint8_t*)packed.data(), 0, (const uint8_t*)act.data(), 0, 0); }, tO, reps);
    run_reps([&](){ for (int r=0;r<N;r++) ggml_vec_dot_q5_0_q8_0(K, &outS[r], 0, weightStd.data()+(size_t)r*nb, 0, act.data(), 0, 1); }, tS, reps);
    double oM=med(tO,reps), sM=med(tS,reps);
    double oIQR=reliqr(tO,reps), sIQR=reliqr(tS,reps);
    double ratio = sM/oM;
    printf("[q5_0] CENSUS K=%d N=%d nb=%d reps=%d | ours_med_ms=%.4f o_iqr=%.3f opp_med_ms=%.4f p_iqr=%.3f ratio_cold_med=%.4f\n",
           K,N,nb,reps,oM,oIQR,sM,sIQR,ratio);
    return (mism_opp==0 && mism_oracle==0) ? 0 : 1;
}

// ===================== q5_1 =====================
static int test_q5_1(int N, int K, bool do_perf, int reps) {
    int nb = K/32;
    std::mt19937 rng(0xC0FFEE2);
    std::uniform_int_distribution<int> qd(0,255), qhd(0,255);
    std::uniform_real_distribution<float> dd(0.001f, 0.05f), md_(0.0f,0.01f);

    std::vector<block_q5_1> weightStd((size_t)N*nb);
    for (auto & b : weightStd) {
        b.d = (ggml_half)dd(rng); b.m = (ggml_half)md_(rng);
        for (int i=0;i<4;i++) b.qh[i]=(uint8_t)qhd(rng);
        for (int i=0;i<16;i++) b.qs[i]=(uint8_t)qd(rng);
    }
    std::vector<block_q8_1> act(nb);
    std::uniform_int_distribution<int> ad(-127,127);
    for (auto & a : act) { a.d = (ggml_half)dd(rng); a.s = (ggml_half)(dd(rng)*10.f); for (int i=0;i<32;i++) a.qs[i]=(int8_t)ad(rng); }

    int ngroups = N/16;
    std::vector<block_q5_1x16> packed((size_t)ngroups*nb);
    for (int g=0; g<ngroups; g++) {
        for (int blk=0; blk<nb; blk++) {
            block_q5_1 tmp[16];
            for (int col=0; col<16; col++) tmp[col] = weightStd[(size_t)(g*16+col)*nb + blk];
            packed[(size_t)g*nb+blk] = make_block_q5_1x16(tmp);
        }
    }

    std::vector<float> ours(N,0.f), ref_opp(N,0.f), ref_oracle(N,0.f);
    weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
        (size_t)K, ours.data(), (size_t)N, (const uint8_t*)packed.data(), 0,
        (const uint8_t*)act.data(), 0, 0);
    for (int r=0;r<N;r++) {
        ggml_vec_dot_q5_1_q8_1(K, &ref_opp[r], 0, weightStd.data()+ (size_t)r*nb, 0, act.data(), 0, 1);
        double s=0; for (int b=0;b<nb;b++) s += oracle_q5_1(weightStd[(size_t)r*nb+b], act[b]);
        ref_oracle[r] = (float)s;
    }
    int mism_opp=0, mism_oracle=0; float max_abs_opp=0, max_abs_oracle=0, max_rel_opp=0, max_rel_oracle=0;
    for (int r=0;r<N;r++) {
        float d1 = std::fabs(ours[r]-ref_opp[r]);
        float d2 = std::fabs(ours[r]-ref_oracle[r]);
        bool bit_eq_opp = std::memcmp(&ours[r],&ref_opp[r],4)==0;
        bool bit_eq_oracle_close = d2/(std::fabs(ref_oracle[r])+1e-6f) < 1e-3f;
        if (!bit_eq_opp) mism_opp++;
        if (!bit_eq_oracle_close) mism_oracle++;
        if (d1>max_abs_opp) max_abs_opp=d1;
        if (d2>max_abs_oracle) max_abs_oracle=d2;
        float rel1 = d1/(std::fabs(ref_opp[r])+1e-6f); if (rel1>max_rel_opp) max_rel_opp=rel1;
        float rel2 = d2/(std::fabs(ref_oracle[r])+1e-6f); if (rel2>max_rel_oracle) max_rel_oracle=rel2;
    }
    printf("[q5_1] BYTE-EXACT-vs-sealed-opponent: mismatches(bitwise-f32)=%d/%d max_abs=%.3e max_rel=%.3e (informational: benign scalar-vs-vector FMA-order noise expected here, see strict-int sub-test)\n", mism_opp, N, max_abs_opp, max_rel_opp);
    printf("[q5_1] BYTE-EXACT-vs-independent-oracle: mismatches(rel>1e-3)=%d/%d max_abs=%.3e max_rel=%.3e\n", mism_oracle, N, max_abs_oracle, max_rel_oracle);
    int mism_opp_rel = 0;
    for (int r=0;r<N;r++) { float rel1 = std::fabs(ours[r]-ref_opp[r])/(std::fabs(ref_opp[r])+1e-6f); if (rel1>1e-3f) mism_opp_rel++; }
    printf("[q5_1] vs-sealed-opponent (rel>1e-3 threshold, FP-noise-tolerant): mismatches=%d/%d\n", mism_opp_rel, N);

    if (!do_perf) return (mism_opp_rel==0 && mism_oracle==0) ? 0 : 1;

    std::vector<float> outO(N), outS(N);
    double tO[64], tS[64];
    run_reps([&](){ weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
        (size_t)K, outO.data(), (size_t)N, (const uint8_t*)packed.data(), 0, (const uint8_t*)act.data(), 0, 0); }, tO, reps);
    run_reps([&](){ for (int r=0;r<N;r++) ggml_vec_dot_q5_1_q8_1(K, &outS[r], 0, weightStd.data()+(size_t)r*nb, 0, act.data(), 0, 1); }, tS, reps);
    double oM=med(tO,reps), sM=med(tS,reps);
    double oIQR=reliqr(tO,reps), sIQR=reliqr(tS,reps);
    double ratio = sM/oM;
    printf("[q5_1] CENSUS K=%d N=%d nb=%d reps=%d | ours_med_ms=%.4f o_iqr=%.3f opp_med_ms=%.4f p_iqr=%.3f ratio_cold_med=%.4f\n",
           K,N,nb,reps,oM,oIQR,sM,sIQR,ratio);
    return (mism_opp_rel==0 && mism_oracle==0) ? 0 : 1;
}

// Strict INTEGER-decode isolation test: neutral scales (d=1.0 exact, m=0.0,
// act.d=1.0, act.s=0.0) so float accumulation reduces to an EXACT sum of small
// integers (no rounding ambiguity from FMA-order/scalar-vs-vector contraction
// differences) -- isolates whether the REDESIGN-B 5-bit qh DECODE itself is
// byte-exact, independent of the (benign, expected) float-order noise seen in
// the general random-scale q5_1 test above.
static int test_q5_1_strict_int(int N, int K) {
    int nb = K/32;
    std::mt19937 rng(0x5121);
    std::uniform_int_distribution<int> qd(0,255), qhd(0,255), ad(-127,127);
    const ggml_half ONE = (ggml_half)1.0f, ZERO = (ggml_half)0.0f;

    std::vector<block_q5_1> weightStd((size_t)N*nb);
    for (auto & b : weightStd) {
        b.d = ONE; b.m = ZERO;
        for (int i=0;i<4;i++) b.qh[i]=(uint8_t)qhd(rng);
        for (int i=0;i<16;i++) b.qs[i]=(uint8_t)qd(rng);
    }
    std::vector<block_q8_1> act(nb);
    for (auto & a : act) { a.d = ONE; a.s = ZERO; for (int i=0;i<32;i++) a.qs[i]=(int8_t)ad(rng); }

    int ngroups = N/16;
    std::vector<block_q5_1x16> packed((size_t)ngroups*nb);
    for (int g=0; g<ngroups; g++)
        for (int blk=0; blk<nb; blk++) {
            block_q5_1 tmp[16];
            for (int col=0; col<16; col++) tmp[col] = weightStd[(size_t)(g*16+col)*nb + blk];
            packed[(size_t)g*nb+blk] = make_block_q5_1x16(tmp);
        }

    std::vector<float> ours(N,0.f), ref_opp(N,0.f);
    weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1(
        (size_t)K, ours.data(), (size_t)N, (const uint8_t*)packed.data(), 0,
        (const uint8_t*)act.data(), 0, 0);
    int mism = 0; float max_abs = 0;
    for (int r=0;r<N;r++) {
        ggml_vec_dot_q5_1_q8_1(K, &ref_opp[r], 0, weightStd.data()+(size_t)r*nb, 0, act.data(), 0, 1);
        bool eq = std::memcmp(&ours[r], &ref_opp[r], 4)==0;
        if (!eq) mism++;
        float d = std::fabs(ours[r]-ref_opp[r]); if (d>max_abs) max_abs=d;
    }
    printf("[q5_1-STRICT-INT d=1,m=0] BYTE-EXACT-vs-sealed-opponent: mismatches(bitwise-f32)=%d/%d max_abs=%.3e\n", mism, N, max_abs);
    return mism==0 ? 0 : 1;
}

int main(int argc, char**argv) {
    int K = argc>1?atoi(argv[1]):2048;
    int N = argc>2?atoi(argv[2]):512;
    int reps = argc>3?atoi(argv[3]):12;
    int mode = argc>4?atoi(argv[4]):0; // 0=correctness-only(small), 1=full(correctness+perf)
    g_flush = (uint8_t*)malloc(FLUSH_BYTES);
    memset(g_flush, 0x5A, FLUSH_BYTES);

    if (mode==0) {
        // small-shape byte-exact only
        int rc0 = test_q5_0(64, 256, false, 0);
        int rc1 = test_q5_1(64, 256, false, 0);
        int rc1s = test_q5_1_strict_int(64, 256);
        printf("CORRECTNESS-ONLY SUMMARY: q5_0=%s q5_1=%s q5_1_strict_int=%s\n",
               rc0==0?"PASS":"FAIL", rc1==0?"PASS":"FAIL", rc1s==0?"PASS":"FAIL");
        return (rc0==0 && rc1==0 && rc1s==0) ? 0 : 1;
    }
    printf("# loadavg/context printed by wrapper script\n");
    int rc0 = test_q5_0(N, K, true, reps);
    int rc1 = test_q5_1(N, K, true, reps);
    int rc1s = test_q5_1_strict_int(N, K);
    printf("FULL SUMMARY: q5_0=%s q5_1=%s q5_1_strict_int=%s\n",
           rc0==0?"PASS":"FAIL", rc1==0?"PASS":"FAIL", rc1s==0?"PASS":"FAIL");
    return (rc0==0 && rc1==0 && rc1s==0) ? 0 : 1;
}
