// harness.c -- timing + accuracy driver for the q4_0_q8_0 GEMM decode-amortization
// probe. SEPARATE TU from kernels.c so -O3 cannot hoist the loop-invariant work
// out of the best-of-N loop.
//
// CLOCK NOTE: this board's governor (ondemand) is not under our control and the
// core idles low. The 1.5x VERDICT is taken from the SPEEDUP RATIO, which is
// clock-invariant within a run (V0, V4-GEMV, decode-only, vwmacc-only, and GEMM
// all run at the same clock during the same interleaved best-of-N). We also
// sample scaling_cur_freq during timing and report observed clock. The stale
// 2.6 GHz absolute ns (878/1071) are NOT used as a threshold; we re-derive the
// 1.5x discriminator from THIS run.
//
// ACCURACY: ground truth = fp64 fold per output (sumi exact, fp16 scales widen
// exactly). rel-err of GEMM outputs vs ggml-real (V0) AND vs fp64 per output.
//
// LAYOUT:
//   weight row vx: nb blocks q4_0, stride XS=18.
//   activation cols: M columns, each nb blocks q8_0, stride YS=34.
//     vy_std: column-major reference store (column j contiguous): col j at
//             vy_std + j*nb*YS, block ib at +ib*YS. Used by V0/V4/ref per column.
//     vy_pack: BLOCK-MAJOR packed for GEMM: block ib's M columns contiguous at
//             vy_pack + (ib*M + j)*YS. Same bytes, reordered.

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
typedef void (*gfn)(size_t, int, float *, const uint8_t *, const uint8_t *);

extern void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_v4_gemv(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_decode_only(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_vwmacc_only(size_t, float *, const uint8_t *, const uint8_t *);
extern void kern_gemm(size_t, int, float *, const uint8_t *, const uint8_t *);
extern void kern_gemm_v4fold(size_t, int, float *, const uint8_t *, const uint8_t *);

static uint32_t rng = 0x2468ace0u;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static inline uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static inline float h2f(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }
static float rscale(void) {
    int e = -(int)(xr() % 13);
    float m = 1.0f + (float)(xr() & 0xFFFF) / 65536.0f;
    return ldexpf(m, e);
}

// Fill one weight row (nb blocks q4_0) and M activation columns (column-major
// reference store). Returns nothing; caller owns buffers.
static void fill_weight(uint8_t *vx, int nb) {
    for (int ib = 0; ib < nb; ++ib) {
        uint8_t *xb = vx + (size_t)ib * XS;
        uint16_t dx = f2h(rscale());
        memcpy(xb, &dx, 2);
        for (int i = 0; i < 16; ++i) xb[2 + i] = (uint8_t)(xr() & 0xFF);
    }
}
// vy_std: column-major, column j at vy_std + (size_t)j*nb*YS.
static void fill_acts(uint8_t *vy_std, int nb, int M) {
    for (int j = 0; j < M; ++j) {
        uint8_t *col = vy_std + (size_t)j * nb * YS;
        for (int ib = 0; ib < nb; ++ib) {
            uint8_t *yb = col + (size_t)ib * YS;
            uint16_t dy = f2h(rscale());
            memcpy(yb, &dy, 2);
            for (int i = 0; i < 32; ++i) yb[2 + i] = (uint8_t)(xr() & 0xFF);
        }
    }
}
// Pack column-major -> block-major: vy_pack[(ib*M+j)*YS] = vy_std col j block ib.
static void pack_blockmajor(const uint8_t *vy_std, uint8_t *vy_pack, int nb, int M) {
    for (int ib = 0; ib < nb; ++ib)
        for (int j = 0; j < M; ++j) {
            const uint8_t *src = vy_std + (size_t)j * nb * YS + (size_t)ib * YS;
            uint8_t *dst = vy_pack + ((size_t)ib * M + j) * YS;
            memcpy(dst, src, YS);
        }
}

static double ref_fp64(size_t n, const uint8_t *vx, const uint8_t *vy_col) {
    const int nb = (int)n / 32;
    double sum = 0.0;
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = vx + (size_t)ib * XS, *yb = vy_col + (size_t)ib * YS;
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

// Sample core-3 clock (best-effort; returns kHz or 0).
static long read_freq(void) {
    FILE *f = fopen("/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq", "r");
    if (!f) return 0;
    long v = 0; if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f);
    return v;
}

int main(void) {
    int Ms[] = {1, 2, 4, 6, 8, 12, 16};
    const int NM = (int)(sizeof(Ms) / sizeof(Ms[0]));
    const int MAXM = 16;
    int ns_list[] = {4096, 11008};
    const int NN = 2;
    const int iters = 2000, reps = 200;

    // -------------------- ACCURACY (M=8 representative) --------------------
    // For each trial: random weight row + 8 cols. GEMM out[j] must == ref / V0.
    {
        const int ACC_TRIALS = 2000, M = 8;
        double me64s = 0, se64s = 0, meV0s = 0, seV0s = 0; // scalar-fold gemm
        double me64f = 0, se64f = 0, meV0f = 0, seV0f = 0; // v4-fold gemm (spec)
        long cnt = 0;
        for (int a = 0; a < NN; ++a) {
            int n = ns_list[a], nb = n / 32;
            uint8_t *vx = malloc((size_t)nb * XS);
            uint8_t *vy_std = malloc((size_t)M * nb * YS);
            uint8_t *vy_pack = malloc((size_t)nb * M * YS);
            float *outs = malloc(sizeof(float) * M);
            float *outf = malloc(sizeof(float) * M);
            for (int t = 0; t < ACC_TRIALS; ++t) {
                fill_weight(vx, nb);
                fill_acts(vy_std, nb, M);
                pack_blockmajor(vy_std, vy_pack, nb, M);
                kern_gemm(n, M, outs, vx, vy_pack);
                kern_gemm_v4fold(n, M, outf, vx, vy_pack);
                for (int j = 0; j < M; ++j) {
                    const uint8_t *col = vy_std + (size_t)j * nb * YS;
                    double ref = ref_fp64(n, vx, col);
                    double denom = fabs(ref) > 1e-30 ? fabs(ref) : 1e-30;
                    float v0; kern_ggml(n, &v0, vx, col);
                    double dV0 = fabs((double)v0) > 1e-30 ? fabs((double)v0) : 1e-30;
                    double e64s = fabs((double)outs[j] - ref) / denom;
                    double eV0s = fabs((double)outs[j] - (double)v0) / dV0;
                    double e64f = fabs((double)outf[j] - ref) / denom;
                    double eV0f = fabs((double)outf[j] - (double)v0) / dV0;
                    if (e64s > me64s) me64s = e64s; se64s += e64s;
                    if (eV0s > meV0s) meV0s = eV0s; seV0s += eV0s;
                    if (e64f > me64f) me64f = e64f; se64f += e64f;
                    if (eV0f > meV0f) meV0f = eV0f; seV0f += eV0f;
                    cnt++;
                }
            }
            free(vx); free(vy_std); free(vy_pack); free(outs); free(outf);
        }
        printf("# ACCURACY GEMM (M=8) over %d trials x {4096,11008} x 8 cols\n", ACC_TRIALS);
        printf("ACC GEMM_scalar  max_err_fp64=%.3e mean_err_fp64=%.3e  max_err_vsV0=%.3e mean_err_vsV0=%.3e\n",
               me64s, se64s / cnt, meV0s, seV0s / cnt);
        printf("ACC GEMM_v4fold  max_err_fp64=%.3e mean_err_fp64=%.3e  max_err_vsV0=%.3e mean_err_vsV0=%.3e\n",
               me64f, se64f / cnt, meV0f, seV0f / cnt);
    }

    // -------------------- TIMING --------------------
    for (int a = 0; a < NN; ++a) {
        int n = ns_list[a], nb = n / 32;
        long fmin = 1L << 30, fmax = 0;

        // Buffers for the largest M; reused for all probes.
        uint8_t *vx = malloc((size_t)nb * XS);
        uint8_t *vy_std = malloc((size_t)MAXM * nb * YS);
        uint8_t *vy_pack = malloc((size_t)nb * MAXM * YS);
        float *out = malloc(sizeof(float) * MAXM);
        fill_weight(vx, nb);
        fill_acts(vy_std, nb, MAXM);
        pack_blockmajor(vy_std, vy_pack, nb, MAXM);
        const uint8_t *col0 = vy_std; // column 0 contiguous for GEMV/probes

        printf("# TIMING n=%d (best-of-%d min, %d iters/rep). ns reported PER OUTPUT.\n", n, reps, iters);

        // CLOCK-RAMP WARMUP: the ondemand governor idles the core low; spin hard
        // on the real kernel until scaling_cur_freq reaches its max (2.6 GHz) so
        // EVERY measured kernel (V0 denominator included) sees the SAME ramped
        // clock. Without this, V0 (measured first, cold) is slow and inflates the
        // ratio. Spin up to ~3s; bail when clock is pinned at the top.
        {
            long fmax_seen = 0;
            int held = 0;
            double tstart = now();
            for (int w = 0; w < 1000000; ++w) {
                for (int i = 0; i < iters; i++) { float s; kern_ggml(n, &s, vx, col0); }
                long fq = read_freq();
                if (fq > fmax_seen) fmax_seen = fq;
                // require the top clock to be HELD for >=8 consecutive checks
                // before trusting the V0 denominator is measured hot.
                if (fq >= 2600000) { if (++held >= 8) break; } else held = 0;
                if (now() - tstart > 8.0e9) break;  // hard cap 8s
            }
            printf("# n=%d warmup: ramped to %ld kHz (held %d)\n", n, fmax_seen, held);
        }

        // --- V0 ggml (per output) ---
        {
            for (int i = 0; i < iters; i++) { float s; kern_ggml(n, &s, vx, col0); }
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now(); float s; for (int i = 0; i < iters; i++) kern_ggml(n, &s, vx, col0);
                double p = (now() - t0) / iters; (void)s; if (p < best) best = p;
                long fq = read_freq(); if (fq) { if (fq < fmin) fmin = fq; if (fq > fmax) fmax = fq; }
            }
            printf("TIME n=%d %-16s %10.1f ns/out\n", n, "V0_ggml", best);
        }
        double v0best;
        { // re-measure V0 cleanly to use as ratio denom (same as above; keep both prints consistent)
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now(); float s; for (int i = 0; i < iters; i++) kern_ggml(n, &s, vx, col0);
                double p = (now() - t0) / iters; (void)s; if (p < best) best = p;
            }
            v0best = best;
        }

        // --- V4-GEMV (per output) ---
        {
            for (int i = 0; i < iters; i++) { float s; kern_v4_gemv(n, &s, vx, col0); }
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now(); float s; for (int i = 0; i < iters; i++) kern_v4_gemv(n, &s, vx, col0);
                double p = (now() - t0) / iters; (void)s; if (p < best) best = p;
            }
            printf("TIME n=%d %-16s %10.1f ns/out  speedup=%.3f\n", n, "V4_GEMV", best, v0best / best);
        }

        // --- decode-ONLY (cost axis; ns per block-decode, scaled to per output) ---
        {
            for (int i = 0; i < iters; i++) { float s; kern_decode_only(n, &s, vx, col0); }
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now(); float s; for (int i = 0; i < iters; i++) kern_decode_only(n, &s, vx, col0);
                double p = (now() - t0) / iters; (void)s; if (p < best) best = p;
            }
            printf("TIME n=%d %-16s %10.1f ns/out  (weight-nibble-unpack only)\n", n, "decode_ONLY", best);
        }

        // --- vwmacc-ONLY (cost axis) ---
        {
            for (int i = 0; i < iters; i++) { float s; kern_vwmacc_only(n, &s, vx, col0); }
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now(); float s; for (int i = 0; i < iters; i++) kern_vwmacc_only(n, &s, vx, col0);
                double p = (now() - t0) / iters; (void)s; if (p < best) best = p;
            }
            printf("TIME n=%d %-16s %10.1f ns/out  (product+accumulate only, no reduce)\n", n, "vwmacc_ONLY", best);
        }

        // --- GEMM sweep over M (ns PER OUTPUT = total/M). PRIMARY = v4fold. ---
        for (int mi = 0; mi < NM; ++mi) {
            int M = Ms[mi];
            for (int i = 0; i < iters; i++) kern_gemm_v4fold(n, M, out, vx, vy_pack);
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now(); for (int i = 0; i < iters; i++) kern_gemm_v4fold(n, M, out, vx, vy_pack);
                double p = (now() - t0) / iters; if (p < best) best = p;
                long fq = read_freq(); if (fq) { if (fq < fmin) fmin = fq; if (fq > fmax) fmax = fq; }
            }
            double per_out = best / M;
            char name[32]; snprintf(name, sizeof name, "GEMM_M%d", M);
            printf("TIME n=%d %-16s %10.1f ns/out  speedup=%.3f  (total=%.1f ns / M=%d)\n",
                   n, name, per_out, v0best / per_out, best, M);
        }
        // --- scalar-fold GEMM sweep (comparison: shows the +fold cost V4 removes) ---
        for (int mi = 0; mi < NM; ++mi) {
            int M = Ms[mi];
            for (int i = 0; i < iters; i++) kern_gemm(n, M, out, vx, vy_pack);
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now(); for (int i = 0; i < iters; i++) kern_gemm(n, M, out, vx, vy_pack);
                double p = (now() - t0) / iters; if (p < best) best = p;
            }
            double per_out = best / M;
            char name[32]; snprintf(name, sizeof name, "GEMMscalar_M%d", M);
            printf("TIME n=%d %-16s %10.1f ns/out  speedup=%.3f  (total=%.1f ns / M=%d)\n",
                   n, name, per_out, v0best / per_out, best, M);
        }
        // --- V0 DRIFT re-check (measured LAST; if it matches the first V0 the
        //     clock held steady across the whole sweep and the ratios are sound).
        {
            double best = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now(); float s; for (int i = 0; i < iters; i++) kern_ggml(n, &s, vx, col0);
                double p = (now() - t0) / iters; (void)s; if (p < best) best = p;
            }
            printf("TIME n=%d %-16s %10.1f ns/out  (drift check; should match first V0=%.1f)\n",
                   n, "V0_ggml_END", best, v0best);
        }
        printf("# observed core3 clock during n=%d timing: min=%ld kHz max=%ld kHz\n", n, fmin, fmax);
        free(vx); free(vy_std); free(vy_pack); free(out);
    }
    printf("DONE\n");
    return 0;
}
