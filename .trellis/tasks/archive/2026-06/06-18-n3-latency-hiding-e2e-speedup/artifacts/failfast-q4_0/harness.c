// harness.c -- timing + accuracy driver for the q4_0_q8_0 latency-hiding probe.
//
// Compiled as a SEPARATE TU from kernels.c so -O3 cannot hoist the dot out of
// the best-of-N loop.
//
// ACCURACY METHODOLOGY (per advisor):
//   * Ground truth = fp64 fold (sumi is an exact integer; fp16 scales widen to
//     fp64 exactly; the fp64 sum of ~344 terms is effectively truth).
//   * For each kernel we report rel-err vs fp64 (the honest "is this drop-in
//     accurate" number) AND rel-err vs V0 (ggml-real), because the task asks
//     for vs-V0. V0's OWN rel-err vs fp64 is the yardstick: a variant whose
//     fp64 rel-err is <= V0's is at least as accurate as what ships.
//   * Inputs are REPRESENTATIVE: per-block fp16 scales constrained to a
//     realistic magnitude (~1e-3 .. ~1), quantised payload bytes full-range.
//
// TIMING: best-of-N min, taskset-pinned, warmup, n in {4096, 11008}.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define XS 18
#define YS 34

typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);

extern void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_v1_k2(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_v1_k4(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_v1_k8(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_v2_p8(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_v3(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_ceiling(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_ceiling_k2(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_ceiling_k4(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_ceiling_k8(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_v2_slide(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_red_k4(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_v4(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_transposed(size_t, float *, const uint8_t *, const uint8_t *);

// ---- RNG ----
static uint32_t rng = 0x2468ace0u;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

// fp16 (binary16) construct from a finite float; round-to-nearest-even via the
// host. We build representative scales by drawing a float in a realistic range
// and converting through _Float16.
static inline uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static inline float h2f(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }

// Representative scale: q4_0/q8_0 per-block scales are roughly |max|/8 and
// |max|/127 of a block of activations/weights -- O(1e-3 .. 1) in practice.
// Draw a magnitude log-uniform in [2^-12, 2^0] ~ [2.4e-4, 1.0], random sign on
// the activation side only (weights scale d is non-negative in q4_0; sign is
// carried by the offset-binary nibbles -- but to stress fold cancellation we
// let the *product* sumi carry sign naturally).
static float rscale(void) {
    // log-uniform magnitude in [2^-12, 2^0].
    int e = -(int)(xr() % 13);           // exponent in [-12, 0]
    float m = 1.0f + (float)(xr() & 0xFFFF) / 65536.0f;  // mantissa in [1,2)
    return ldexpf(m, e);
}

static void fill(uint8_t *vx, uint8_t *vy, int nb) {
    for (int ib = 0; ib < nb; ++ib) {
        uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        // representative scales (round-tripped through fp16 so kernels and
        // reference see identical stored bits).
        uint16_t dx = f2h(rscale());
        uint16_t dy = f2h(rscale());
        memcpy(xb, &dx, 2);
        memcpy(yb, &dy, 2);
        for (int i = 0; i < 16; ++i) xb[2 + i] = (uint8_t)(xr() & 0xFF);  // q4_0 nibbles
        for (int i = 0; i < 32; ++i) yb[2 + i] = (uint8_t)(xr() & 0xFF);  // q8_0 int8
    }
}

// fp64 ground-truth dot (decode in scalar exactly as the kernel does).
static double ref_fp64(size_t n, const uint8_t *vx, const uint8_t *vy) {
    const int nb = (int)n / 32;
    double sum = 0.0;
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy + (size_t)ib * YS;
        uint16_t hx, hy; memcpy(&hx, xb, 2); memcpy(&hy, yb, 2);
        double dx = (double)h2f(hx), dy = (double)h2f(hy);
        const int8_t *yq = (const int8_t *)(yb + 2);
        long sumi = 0;
        for (int k = 0; k < 16; ++k) {
            int nib = xb[2 + k];
            int lo = (nib & 0x0F) - 8;
            int hi = (nib >> 4) - 8;
            sumi += (long)lo * yq[k];
            sumi += (long)hi * yq[k + 16];
        }
        sum += (double)sumi * dx * dy;
    }
    return sum;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double tb(kfn fn, size_t n, int it, const uint8_t *vx, const uint8_t *vy) {
    double t0 = now(); float s; for (int i = 0; i < it; i++) fn(n, &s, vx, vy);
    double dt = now() - t0; (void)s; return dt / it;
}

struct K { const char *name; kfn fn; int is_ceiling; };

int main(void) {
    struct K ks[] = {
        {"V0_ggml",   kern_ggml,    0},
        {"V1_k2",     kern_v1_k2,   0},
        {"V1_k4",     kern_v1_k4,   0},
        {"V1_k8",     kern_v1_k8,   0},
        {"V2_p8",     kern_v2_p8,   0},
        {"V2_slide",  kern_v2_slide,0},
        {"V4",        kern_v4,      0},
        {"V3",        kern_v3,      0},
        {"RED_k4",    kern_red_k4,  1},
        {"TRANSPOSED",kern_transposed, 1},
        {"CEILING",    kern_ceiling,    1},
        {"CEILING_k2", kern_ceiling_k2, 1},
        {"CEILING_k4", kern_ceiling_k4, 1},
        {"CEILING_k8", kern_ceiling_k8, 1},
    };
    const int K = sizeof(ks) / sizeof(ks[0]);

    // ---------------- ACCURACY ----------------
    // Many representative inputs; track max + mean rel-err vs fp64 and vs V0.
    const int ACC_TRIALS = 2000;
    int acc_ns[] = {4096, 11008};
    const int ANN = 2;
    double maxerr64[32], sumerr64[32];
    double maxerrV0[32], sumerrV0[32];
    long errcnt = 0;
    for (int k = 0; k < K; ++k) { maxerr64[k] = 0; sumerr64[k] = 0; maxerrV0[k] = 0; sumerrV0[k] = 0; }

    for (int t = 0; t < ACC_TRIALS; ++t) {
        for (int a = 0; a < ANN; ++a) {
            int n = acc_ns[a], nb = n / 32;
            uint8_t *vx = (uint8_t *)malloc((size_t)nb * XS);
            uint8_t *vy = (uint8_t *)malloc((size_t)nb * YS);
            fill(vx, vy, nb);
            double ref = ref_fp64(n, vx, vy);
            double denom = fabs(ref) > 1e-30 ? fabs(ref) : 1e-30;
            float v0; kern_ggml(n, &v0, vx, vy);
            for (int k = 0; k < K; ++k) {
                if (ks[k].is_ceiling) continue;
                float g; ks[k].fn(n, &g, vx, vy);
                double e64 = fabs((double)g - ref) / denom;
                double dV0 = fabs((double)v0) > 1e-30 ? fabs((double)v0) : 1e-30;
                double eV0 = fabs((double)g - (double)v0) / dV0;
                if (e64 > maxerr64[k]) maxerr64[k] = e64;
                sumerr64[k] += e64;
                if (eV0 > maxerrV0[k]) maxerrV0[k] = eV0;
                sumerrV0[k] += eV0;
            }
            errcnt++;
            free(vx); free(vy);
        }
    }
    printf("# ACCURACY (rel-err over %d representative inputs x {4096,11008})\n", ACC_TRIALS);
    printf("# %-10s  %12s %12s  %12s %12s\n", "variant", "max_err_fp64", "mean_err_fp64", "max_err_vsV0", "mean_err_vsV0");
    for (int k = 0; k < K; ++k) {
        if (ks[k].is_ceiling) continue;
        printf("ACC %-10s  %12.3e %12.3e  %12.3e %12.3e\n",
               ks[k].name, maxerr64[k], sumerr64[k] / errcnt, maxerrV0[k], sumerrV0[k] / errcnt);
    }

    // ---------------- TIMING ----------------
    int timing_ns[] = {4096, 11008};
    const int TNN = 2;
    const int iters = 2000, reps = 200;
    for (int a = 0; a < TNN; ++a) {
        int n = timing_ns[a], nb = n / 32;
        uint8_t *vx = (uint8_t *)malloc((size_t)nb * XS);
        uint8_t *vy = (uint8_t *)malloc((size_t)nb * YS);
        fill(vx, vy, nb);
        // warmup
        for (int k = 0; k < K; ++k) for (int i = 0; i < iters; i++) { float s; ks[k].fn(n, &s, vx, vy); }
        double best[32]; for (int k = 0; k < K; ++k) best[k] = 1e18;
        for (int r = 0; r < reps; r++)
            for (int k = 0; k < K; k++) {
                double p = tb(ks[k].fn, n, iters, vx, vy);
                if (p < best[k]) best[k] = p;
            }
        double v0best = best[0];
        printf("# TIMING n=%d (best-of-%d min, %d iters/rep)\n", n, reps, iters);
        for (int k = 0; k < K; ++k)
            printf("TIME n=%d %-10s %10.1f ns  speedup=%.3f\n",
                   n, ks[k].name, best[k], v0best / best[k]);
        free(vx); free(vy);
    }
    printf("DONE\n");
    return 0;
}
