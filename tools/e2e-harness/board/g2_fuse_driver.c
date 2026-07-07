/* g2_fuse_driver.c — G2 [FUSE] rms_norm->mul paired hardware micro (rvv/VLEN128).
 *
 * Faithful reconstruction of the emit census (experiments/.../g2-fuse-rms-norm-mul/emit_census.txt,
 * reproduced from tcrv-opt --tcrv-rvv-lower-to-emitc of the fused epilogue lit at HEAD 0ad9cf6d):
 *
 *   FUSED (one pass / row): scalar-double Sx^2 reduce -> scale=1/sqrtf(mean+eps);
 *     vector m8 loop:  vx=vle32(x) ; vy=vfmul_vf(vx,scale) ; vw=vle32(w) ; vz=vfmul_vv(vy,vw) ; vse32(z)
 *   UNFUSED (two passes / whole tensor):
 *     pass1 rms_norm : reduce ; vx=vle32(x) ; vy=vfmul_vf(vx,scale) ; vse32(y)      <- materializes y[]
 *     pass2 mul      : vy=vle32(y) ; vw=vle32(w) ; vz=vfmul_vv(vy,vw) ; vse32(z)     <- reloads y[]
 *
 * The ONLY difference is the intermediate row-tensor y[] round trip (write in pass1, read in pass2).
 * At tensor scale (each tensor > L3) y[] genuinely round-trips through DRAM, so the delta isolates
 * exactly the 8*n bytes/row the fusion eliminates. Both kernels are compiled from THIS single file
 * with ONE clang -O3 -march (preflight(0) toolchain symmetry trivially satisfied, no fast-math =>
 * the double reduce stays serial, matching the byte-exact emit).
 *
 * Modes:
 *   paired  <rounds>   : interleaved [fused, unfused, fused-sentinel] per round; prints per-round ns
 *                        + median/IQR/min for each; fingerprints (DCE guard + cross-check).
 *   fused   <iters>    : run only fused <iters> times (for `perf stat` counter attribution).
 *   unfused <iters>    : run only unfused <iters> times.
 *   verify             : fused-Z vs unfused-Z max-ULP + fold checksum (on-hardware value equality).
 *
 * Env: ROWS (default 8192), NEMB (default 4096), EPS (default 1e-5). Working set per tensor =
 *      ROWS*NEMB*4 bytes; X+Y+Z sized so each > L3 and total > 3xL3 (hygiene, reported).
 */
#include <riscv_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static size_t ROWS = 8192;
static size_t NEMB = 4096;
static float  EPS  = 1e-5f;

/* ---- one-row primitives (faithful to the emit census) ---------------------- */

static inline float row_scale(const float *x, size_t n, float eps) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) { float xi = x[i]; sum += (double)(xi * xi); }
    float mean = (float)(sum / (double)n);
    return 1.0f / sqrtf(mean + eps);
}

/* FUSED: normalize + mul(w) spliced into one m8 loop; y[] never stored/reloaded. */
static void fused_kernel(size_t rows, size_t n, const float *X, const float *W,
                         float *Z, float eps) {
    for (size_t r = 0; r < rows; r++) {
        const float *x = X + r * n;
        float *z = Z + r * n;
        float scale = row_scale(x, n, eps);
        size_t vl;
        for (size_t i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
            vfloat32m8_t vy = __riscv_vfmul_vf_f32m8(vx, scale, vl);
            vfloat32m8_t vw = __riscv_vle32_v_f32m8(W + i, vl);
            vfloat32m8_t vz = __riscv_vfmul_vv_f32m8(vy, vw, vl);
            __riscv_vse32_v_f32m8(z + i, vz, vl);
        }
    }
}

/* UNFUSED pass1: rms_norm, writes the intermediate y[] row-tensor. */
static void unfused_pass1_rmsnorm(size_t rows, size_t n, const float *X,
                                  float *Y, float eps) {
    for (size_t r = 0; r < rows; r++) {
        const float *x = X + r * n;
        float *y = Y + r * n;
        float scale = row_scale(x, n, eps);
        size_t vl;
        for (size_t i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
            vfloat32m8_t vy = __riscv_vfmul_vf_f32m8(vx, scale, vl);
            __riscv_vse32_v_f32m8(y + i, vy, vl);
        }
    }
}

/* UNFUSED pass2: separate mul kernel, reloads y[] and multiplies by w[]. */
static void unfused_pass2_mul(size_t rows, size_t n, const float *Y,
                              const float *W, float *Z) {
    for (size_t r = 0; r < rows; r++) {
        const float *y = Y + r * n;
        float *z = Z + r * n;
        size_t vl;
        for (size_t i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + i, vl);
            vfloat32m8_t vw = __riscv_vle32_v_f32m8(W + i, vl);
            vfloat32m8_t vz = __riscv_vfmul_vv_f32m8(vy, vw, vl);
            __riscv_vse32_v_f32m8(z + i, vz, vl);
        }
    }
}

static void unfused_kernel(size_t rows, size_t n, const float *X, const float *W,
                           float *Y, float *Z, float eps) {
    unfused_pass1_rmsnorm(rows, n, X, Y, eps);
    unfused_pass2_mul(rows, n, Y, W, Z);
}

/* ---- helpers --------------------------------------------------------------- */

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* flush buffer >> L3 to force cold, symmetric cache state before each timed region */
static volatile uint64_t g_sink = 0;
static void flush_caches(char *fb, size_t fbytes) {
    uint64_t acc = 0;
    for (size_t i = 0; i < fbytes; i += 64) { fb[i] ^= (char)(i); acc += (uint64_t)fb[i]; }
    g_sink += acc;
}

static uint64_t fold_z(const float *Z, size_t rows, size_t n) {
    /* strided FNV-ish fold over Z (DCE guard + cross-check fingerprint) */
    uint64_t h = 1469598103934665603ULL;
    size_t total = rows * n;
    size_t step = (total / 65536) | 1;
    for (size_t i = 0; i < total; i += step) {
        uint32_t bits;
        memcpy(&bits, &Z[i], 4);
        h ^= bits; h *= 1099511628211ULL;
    }
    return h;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static void stats(double *v, int k, double *med, double *iqr, double *mn) {
    double *s = malloc(sizeof(double) * k);
    memcpy(s, v, sizeof(double) * k);
    qsort(s, k, sizeof(double), cmp_double);
    *mn = s[0];
    *med = (k & 1) ? s[k/2] : 0.5 * (s[k/2 - 1] + s[k/2]);
    double q1 = s[k/4], q3 = s[(3*k)/4];
    *iqr = q3 - q1;
    free(s);
}

static void init_data(float *X, float *W, size_t rows, size_t n) {
    /* deterministic pseudo-random-ish, non-trivial so scale != 1 and mul != identity */
    for (size_t i = 0; i < n; i++) W[i] = 0.5f + 0.001f * (float)(i % 257);
    for (size_t r = 0; r < rows; r++) {
        float *x = X + r * n;
        for (size_t i = 0; i < n; i++) {
            uint32_t s = (uint32_t)(r * 2654435761u + i * 40503u + 7u);
            s ^= s >> 13; s *= 0x9E3779B1u; s ^= s >> 16;
            x[i] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;  /* [-1,1) */
        }
    }
}

int main(int argc, char **argv) {
    if (getenv("ROWS")) ROWS = strtoull(getenv("ROWS"), 0, 10);
    if (getenv("NEMB")) NEMB = strtoull(getenv("NEMB"), 0, 10);
    if (getenv("EPS"))  EPS  = strtof(getenv("EPS"), 0);
    const char *mode = argc > 1 ? argv[1] : "paired";

    size_t rows = ROWS, n = NEMB;
    size_t tbytes = rows * n * sizeof(float);
    size_t fbytes = 128ull * 1024 * 1024;   /* 128 MiB flush buffer >> L3(64MiB) */

    float *X = aligned_alloc(64, tbytes);
    float *W = aligned_alloc(64, n * sizeof(float));
    float *Y = aligned_alloc(64, tbytes);
    float *Z = aligned_alloc(64, tbytes);
    char  *FB = aligned_alloc(64, fbytes);
    if (!X || !W || !Y || !Z || !FB) { fprintf(stderr, "alloc fail\n"); return 2; }
    init_data(X, W, rows, n);
    memset(Y, 0, tbytes); memset(Z, 0, tbytes); memset(FB, 1, fbytes);

    double ws_mib = (double)(3 * tbytes) / (1024.0 * 1024.0);  /* X+Y+Z */
    double l3_mib = 64.0;

    if (strcmp(mode, "verify") == 0) {
        float *Z2 = aligned_alloc(64, tbytes);
        fused_kernel(rows, n, X, W, Z, EPS);
        unfused_kernel(rows, n, X, W, Y, Z2, EPS);
        size_t total = rows * n; uint64_t maxulp = 0; size_t ndiff = 0;
        for (size_t i = 0; i < total; i++) {
            uint32_t a, b; memcpy(&a, &Z[i], 4); memcpy(&b, &Z2[i], 4);
            if (a != b) {
                ndiff++;
                uint64_t ua = a, ub = b;
                uint64_t d = ua > ub ? ua - ub : ub - ua;
                if (d > maxulp) maxulp = d;
            }
        }
        printf("VERIFY fused_vs_unfused ndiff=%zu/%zu max_ulp=%llu fold_fused=0x%016llx fold_unfused=0x%016llx\n",
               ndiff, total, (unsigned long long)maxulp,
               (unsigned long long)fold_z(Z, rows, n),
               (unsigned long long)fold_z(Z2, rows, n));
        free(Z2);
        return 0;
    }

    if (strcmp(mode, "fused") == 0 || strcmp(mode, "unfused") == 0) {
        long iters = argc > 2 ? atol(argv[2]) : 3;
        int fused = strcmp(mode, "fused") == 0;
        for (long it = 0; it < iters; it++) {
            if (fused) fused_kernel(rows, n, X, W, Z, EPS);
            else       unfused_kernel(rows, n, X, W, Y, Z, EPS);
        }
        printf("PERFMODE mode=%s iters=%ld rows=%zu nemb=%zu tensor_MiB=%.1f fold_z=0x%016llx\n",
               mode, iters, rows, n, (double)tbytes/(1024.0*1024.0),
               (unsigned long long)fold_z(Z, rows, n));
        return 0;
    }

    /* paired mode */
    int rounds = argc > 2 ? atoi(argv[2]) : 12;
    double *tf = malloc(sizeof(double) * rounds);
    double *tu = malloc(sizeof(double) * rounds);
    double *ts = malloc(sizeof(double) * rounds);  /* fused sentinel */
    uint64_t foldf = 0, foldu = 0;
    /* one warm iteration to fault-in pages (not timed) */
    fused_kernel(rows, n, X, W, Z, EPS);
    unfused_kernel(rows, n, X, W, Y, Z, EPS);
    for (int r = 0; r < rounds; r++) {
        double t0, t1;
        flush_caches(FB, fbytes);
        t0 = now_ns(); fused_kernel(rows, n, X, W, Z, EPS);        t1 = now_ns();
        tf[r] = t1 - t0; foldf = fold_z(Z, rows, n);

        flush_caches(FB, fbytes);
        t0 = now_ns(); unfused_kernel(rows, n, X, W, Y, Z, EPS);   t1 = now_ns();
        tu[r] = t1 - t0; foldu = fold_z(Z, rows, n);

        flush_caches(FB, fbytes);
        t0 = now_ns(); fused_kernel(rows, n, X, W, Z, EPS);        t1 = now_ns();
        ts[r] = t1 - t0;

        fprintf(stderr, "round %2d  fused=%.3fms  unfused=%.3fms  sentinel=%.3fms\n",
                r, tf[r]/1e6, tu[r]/1e6, ts[r]/1e6);
    }
    double mf, if_, nf, mu, iu, nu, ms, is_, ns_;
    stats(tf, rounds, &mf, &if_, &nf);
    stats(tu, rounds, &mu, &iu, &nu);
    stats(ts, rounds, &ms, &is_, &ns_);
    double ratio = mu / mf;                 /* >1 => fused faster */
    double drift = ms / mf;                 /* ~1 => runs comparable (noise floor) */
    /* DRAM-stream throughput proxy from wall time: fused moves ~2 tensors, unfused ~4 */
    double gbps_f = ((double)(2 * tbytes)) / (mf) ;   /* bytes/ns = GB/s */
    double gbps_u = ((double)(4 * tbytes)) / (mu) ;
    printf("PAIRED rounds=%d rows=%zu nemb=%zu tensor_MiB=%.1f workingset_MiB=%.1f L3_MiB=%.1f hygiene=%s\n",
           rounds, rows, n, (double)tbytes/(1024.0*1024.0), ws_mib, l3_mib,
           ws_mib > 3.0 * l3_mib ? "COLD(ws>3xL3)" : "WARN(ws<3xL3)");
    printf("PAIRED fused_ms   median=%.3f iqr=%.3f min=%.3f iqr_pct=%.3f\n", mf/1e6, if_/1e6, nf/1e6, 100.0*if_/mf);
    printf("PAIRED unfused_ms median=%.3f iqr=%.3f min=%.3f iqr_pct=%.3f\n", mu/1e6, iu/1e6, nu/1e6, 100.0*iu/mu);
    printf("PAIRED sentinel_ms median=%.3f iqr=%.3f min=%.3f (fused repeat)\n", ms/1e6, is_/1e6, ns_/1e6);
    printf("PAIRED speedup_unfused_over_fused=%.4f drift_sentinel=%.4f gbps_fused=%.3f gbps_unfused=%.3f\n",
           ratio, drift, gbps_f, gbps_u);
    printf("PAIRED fold_fused=0x%016llx fold_unfused=0x%016llx values_equal=%s\n",
           (unsigned long long)foldf, (unsigned long long)foldu, foldf == foldu ? "YES" : "NO");
    return 0;
}
