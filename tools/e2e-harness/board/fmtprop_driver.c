/* fmtprop_driver.c — G2 [FMT-PROP] rms_norm->mul->quantize(q8_0) paired hardware micro (rvv/VLEN128).
 *
 * Faithful reconstruction of the emit census of commit 2b814e46
 * ([FMT-PROP] 曳光弹贯通 — rms_norm→mul→quantize 融合链, real emitter
 *  emitElementwiseRmsNormReduceStrip + emitQuantizeQ80BlockBody):
 *
 *   FUSED (one pass / row): scalar-double Sx^2 reduce -> scale=1/sqrtf(mean+eps);
 *     per 32-elem q8_0 block (vl=32, e32m8 == one block at VLEN128):
 *       vx=vle32(x); vy=vfmul_vf(vx,scale); vw=vle32(w); vz=vfmul_vv(vy,vw)   [register-kept]
 *       -> q8_0 body on vz: vabs=vfabs(vz); amax=vfredmax(vabs); d=amax/127; id=1/d;
 *          vs=vfmul_vf(vz,id); vi16=vfncvt_x(vs,RMM); vi8=vncvt(vi16); vse8_v_i8m2(qs)
 *          store d as fp16.  The f32 activation z[] is NEVER stored, NEVER reloaded.
 *
 *   UNFUSED (two passes / whole tensor), the pre-[FMT-PROP] baseline = the already-fused
 *   rms_norm->mul (prior G2 leg) that lands f32 + an INDEPENDENT quantize_row_q8_0 pass:
 *     pass1 rms_norm->mul : reduce; vx=vle32(x); vy=vfmul_vf(vx,scale); vw=vle32(w);
 *                           vz=vfmul_vv(vy,vw); vse32(z)          <- materializes f32 z[]
 *     pass2 quantize_row  : vz=vle32(z);  [same q8_0 body as fused]  -> qs/d
 *                                                                  <- reloads f32 z[]
 *
 * The ONLY difference is the intermediate f32 activation z[] round trip that FMT-PROP
 * eliminates: 4*n store (pass1) + 4*n reload (pass2) = 2*n*4 = 8*n bytes/row. At tensor
 * scale (each tensor > L3) z[] genuinely round-trips DRAM, so the delta isolates exactly
 * those 8*n bytes/row. Both kernels are compiled from THIS single file with ONE
 * clang -O3 -march (preflight(0) toolchain symmetry total; no fast-math => the double
 * reduce stays serial). The q8_0 quantize body is the SAME shared helper on both sides,
 * so the emitted q8_0 output is byte-identical (register-kept vz vs store/reload vz are
 * IEEE binary32-lossless) — value equality is verified on hardware.  [NG-4]: isolated A/B
 * on the memory axis, NOT a ggml beat, NOT an e2e [PERF-1] eight-gate.
 *
 * Modes:
 *   paired  <rounds>   : interleaved [fused, unfused, fused-sentinel] per round; median/IQR/min.
 *   fused   <iters>    : run only fused (for `perf stat` counter attribution).
 *   unfused <iters>    : run only unfused.
 *   verify             : fused-Q vs unfused-Q byte diff + fold checksum (on-hardware equality).
 *
 * Env: ROWS (default 8192), NEMB (default 4096, multiple of 32), EPS (default 1e-5).
 */
#include <riscv_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define QK8_0 32

static size_t ROWS = 8192;
static size_t NEMB = 4096;
static float  EPS  = 1e-5f;

/* block_q8_0 = 2-byte fp16 scale + 32 int8 quants = 34 bytes (ggml layout). */
#define BLK_BYTES 34

static inline float row_scale(const float *x, size_t n, float eps) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) { float xi = x[i]; sum += (double)(xi * xi); }
    float mean = (float)(sum / (double)n);
    return 1.0f / sqrtf(mean + eps);
}

/* SHARED q8_0 block body (faithful to emitQuantizeQ80BlockBody): one 32-elem block from a
 * register-kept vz (f32m8, vl=32) -> block_q8_0 at (qs_out, d_out). Identical on both sides,
 * so fused (vz in-register) and unfused (vz reloaded from f32 z[]) produce byte-identical q8_0. */
static inline void q8_0_block_from_vec(vfloat32m8_t vz, int8_t *qs_out, _Float16 *d_out, size_t vl) {
    vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(vz, vl);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vabs, vzero, vl);
    float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
    float d  = amax / 127.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    vfloat32m8_t vs = __riscv_vfmul_vf_f32m8(vz, id, vl);
    /* RMM=4 (round-nearest, ties-away) to match ggml's roundf-style quantize (same literal
     * as ggml_quantize_mat_q8_0 in arch/riscv/repack.cpp). */
    vint16m4_t vi16 = __riscv_vfncvt_x_f_w_i16m4_rm(vs, 4, vl);
    vint8m2_t  vi8  = __riscv_vncvt_x_x_w_i8m2(vi16, vl);
    __riscv_vse8_v_i8m2(qs_out, vi8, vl);
    *d_out = (_Float16)d;
}

/* FUSED: rms_norm->mul->quantize in one pass; f32 z[] never stored/reloaded. */
static void fused_kernel(size_t rows, size_t n, const float *X, const float *W,
                         uint8_t *Q, float eps) {
    size_t nb = n / QK8_0;
    for (size_t r = 0; r < rows; r++) {
        const float *x = X + r * n;
        uint8_t *q = Q + r * nb * BLK_BYTES;
        float scale = row_scale(x, n, eps);
        for (size_t b = 0; b < nb; b++) {
            size_t off = b * QK8_0;
            size_t vl = __riscv_vsetvl_e32m8(QK8_0);
            vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + off, vl);
            vfloat32m8_t vy = __riscv_vfmul_vf_f32m8(vx, scale, vl);
            vfloat32m8_t vw = __riscv_vle32_v_f32m8(W + off, vl);
            vfloat32m8_t vz = __riscv_vfmul_vv_f32m8(vy, vw, vl);
            uint8_t *blk = q + b * BLK_BYTES;
            _Float16 d;
            q8_0_block_from_vec(vz, (int8_t *)(blk + 2), &d, vl);
            memcpy(blk, &d, 2);
        }
    }
}

/* UNFUSED pass1: rms_norm->mul (the prior fused leg), writes f32 activation z[]. */
static void unfused_pass1_rmsnorm_mul(size_t rows, size_t n, const float *X,
                                      const float *W, float *Z, float eps) {
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

/* UNFUSED pass2: independent quantize_row_q8_0, reloads f32 z[]. */
static void unfused_pass2_quantize(size_t rows, size_t n, const float *Z, uint8_t *Q) {
    size_t nb = n / QK8_0;
    for (size_t r = 0; r < rows; r++) {
        const float *z = Z + r * n;
        uint8_t *q = Q + r * nb * BLK_BYTES;
        for (size_t b = 0; b < nb; b++) {
            size_t off = b * QK8_0;
            size_t vl = __riscv_vsetvl_e32m8(QK8_0);
            vfloat32m8_t vz = __riscv_vle32_v_f32m8(z + off, vl);
            uint8_t *blk = q + b * BLK_BYTES;
            _Float16 d;
            q8_0_block_from_vec(vz, (int8_t *)(blk + 2), &d, vl);
            memcpy(blk, &d, 2);
        }
    }
}

static void unfused_kernel(size_t rows, size_t n, const float *X, const float *W,
                           float *Z, uint8_t *Q, float eps) {
    unfused_pass1_rmsnorm_mul(rows, n, X, W, Z, eps);
    unfused_pass2_quantize(rows, n, Z, Q);
}

/* ---- helpers --------------------------------------------------------------- */
static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
static volatile uint64_t g_sink = 0;
static void flush_caches(char *fb, size_t fbytes) {
    uint64_t acc = 0;
    for (size_t i = 0; i < fbytes; i += 64) { fb[i] ^= (char)(i); acc += (uint64_t)fb[i]; }
    g_sink += acc;
}
static uint64_t fold_q(const uint8_t *Q, size_t bytes) {
    uint64_t h = 1469598103934665603ULL;
    size_t step = (bytes / 131072) | 1;
    for (size_t i = 0; i < bytes; i += step) { h ^= Q[i]; h *= 1099511628211ULL; }
    return h;
}
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y);
}
static void stats(double *v, int k, double *med, double *iqr, double *mn) {
    double *s = malloc(sizeof(double) * k); memcpy(s, v, sizeof(double) * k);
    qsort(s, k, sizeof(double), cmp_double);
    *mn = s[0]; *med = (k & 1) ? s[k/2] : 0.5 * (s[k/2 - 1] + s[k/2]);
    *iqr = s[(3*k)/4] - s[k/4]; free(s);
}
static void init_data(float *X, float *W, size_t rows, size_t n) {
    for (size_t i = 0; i < n; i++) W[i] = 0.5f + 0.001f * (float)(i % 257);
    for (size_t r = 0; r < rows; r++) {
        float *x = X + r * n;
        for (size_t i = 0; i < n; i++) {
            uint32_t s = (uint32_t)(r * 2654435761u + i * 40503u + 7u);
            s ^= s >> 13; s *= 0x9E3779B1u; s ^= s >> 16;
            x[i] = ((float)(s & 0xFFFF) / 32768.0f) - 1.0f;
        }
    }
}

int main(int argc, char **argv) {
    if (getenv("ROWS")) ROWS = strtoull(getenv("ROWS"), 0, 10);
    if (getenv("NEMB")) NEMB = strtoull(getenv("NEMB"), 0, 10);
    if (getenv("EPS"))  EPS  = strtof(getenv("EPS"), 0);
    const char *mode = argc > 1 ? argv[1] : "paired";

    size_t rows = ROWS, n = NEMB;
    if (n % QK8_0) { fprintf(stderr, "NEMB must be multiple of 32\n"); return 2; }
    size_t tbytes = rows * n * sizeof(float);           /* X, Z each */
    size_t qbytes = rows * (n / QK8_0) * BLK_BYTES;     /* q8_0 output */
    size_t fbytes = 128ull * 1024 * 1024;

    float   *X = aligned_alloc(64, tbytes);
    float   *W = aligned_alloc(64, n * sizeof(float));
    float   *Z = aligned_alloc(64, tbytes);
    uint8_t *Q = aligned_alloc(64, qbytes);
    char    *FB = aligned_alloc(64, fbytes);
    if (!X || !W || !Z || !Q || !FB) { fprintf(stderr, "alloc fail\n"); return 2; }
    init_data(X, W, rows, n);
    memset(Z, 0, tbytes); memset(Q, 0, qbytes); memset(FB, 1, fbytes);

    double ws_mib = (double)(tbytes + tbytes + qbytes) / (1024.0*1024.0);  /* unfused ws: X+Z+Q */
    double l3_mib = 64.0;

    if (strcmp(mode, "verify") == 0) {
        uint8_t *Q2 = aligned_alloc(64, qbytes);
        fused_kernel(rows, n, X, W, Q, EPS);
        unfused_kernel(rows, n, X, W, Z, Q2, EPS);
        size_t ndiff = 0, firstdiff = (size_t)-1;
        for (size_t i = 0; i < qbytes; i++)
            if (Q[i] != Q2[i]) { ndiff++; if (firstdiff == (size_t)-1) firstdiff = i; }
        printf("VERIFY fused_vs_unfused q8_0 ndiff_bytes=%zu/%zu firstdiff=%zd byte_exact=%s "
               "fold_fused=0x%016llx fold_unfused=0x%016llx\n",
               ndiff, qbytes, (ssize_t)(ndiff ? (ssize_t)firstdiff : -1),
               ndiff == 0 ? "YES" : "NO",
               (unsigned long long)fold_q(Q, qbytes), (unsigned long long)fold_q(Q2, qbytes));
        free(Q2);
        return 0;
    }

    if (strcmp(mode, "fused") == 0 || strcmp(mode, "unfused") == 0) {
        long iters = argc > 2 ? atol(argv[2]) : 3;
        int fused = strcmp(mode, "fused") == 0;
        for (long it = 0; it < iters; it++) {
            if (fused) fused_kernel(rows, n, X, W, Q, EPS);
            else       unfused_kernel(rows, n, X, W, Z, Q, EPS);
        }
        printf("PERFMODE mode=%s iters=%ld rows=%zu nemb=%zu tensor_MiB=%.1f qout_MiB=%.1f fold_q=0x%016llx\n",
               mode, iters, rows, n, (double)tbytes/(1024.0*1024.0),
               (double)qbytes/(1024.0*1024.0), (unsigned long long)fold_q(Q, qbytes));
        return 0;
    }

    int rounds = argc > 2 ? atoi(argv[2]) : 12;
    double *tf = malloc(sizeof(double)*rounds);
    double *tu = malloc(sizeof(double)*rounds);
    double *ts = malloc(sizeof(double)*rounds);
    uint64_t foldf = 0, foldu = 0;
    fused_kernel(rows, n, X, W, Q, EPS);
    unfused_kernel(rows, n, X, W, Z, Q, EPS);
    for (int r = 0; r < rounds; r++) {
        double t0, t1;
        flush_caches(FB, fbytes);
        t0 = now_ns(); fused_kernel(rows, n, X, W, Q, EPS);       t1 = now_ns();
        tf[r] = t1 - t0; foldf = fold_q(Q, qbytes);
        flush_caches(FB, fbytes);
        t0 = now_ns(); unfused_kernel(rows, n, X, W, Z, Q, EPS);  t1 = now_ns();
        tu[r] = t1 - t0; foldu = fold_q(Q, qbytes);
        flush_caches(FB, fbytes);
        t0 = now_ns(); fused_kernel(rows, n, X, W, Q, EPS);       t1 = now_ns();
        ts[r] = t1 - t0;
        fprintf(stderr, "round %2d  fused=%.3fms  unfused=%.3fms  sentinel=%.3fms\n",
                r, tf[r]/1e6, tu[r]/1e6, ts[r]/1e6);
    }
    double mf, if_, nf, mu, iu, nu, ms, is_, ns_;
    stats(tf, rounds, &mf, &if_, &nf);
    stats(tu, rounds, &mu, &iu, &nu);
    stats(ts, rounds, &ms, &is_, &ns_);
    double ratio = mu / mf;
    double drift = ms / mf;
    /* DRAM-stream bytes/row: fused reads 4n(x)+writes ~1.06n(q); unfused adds z[] 8n round trip. */
    printf("PAIRED rounds=%d rows=%zu nemb=%zu tensor_MiB=%.1f qout_MiB=%.1f workingset_MiB=%.1f L3_MiB=%.1f hygiene=%s\n",
           rounds, rows, n, (double)tbytes/(1024.0*1024.0), (double)qbytes/(1024.0*1024.0),
           ws_mib, l3_mib, ws_mib > 3.0*l3_mib ? "COLD(ws>3xL3)" : "WARN(ws<3xL3)");
    printf("PAIRED fused_ms   median=%.3f iqr=%.3f min=%.3f iqr_pct=%.3f\n", mf/1e6, if_/1e6, nf/1e6, 100.0*if_/mf);
    printf("PAIRED unfused_ms median=%.3f iqr=%.3f min=%.3f iqr_pct=%.3f\n", mu/1e6, iu/1e6, nu/1e6, 100.0*iu/mu);
    printf("PAIRED sentinel_ms median=%.3f iqr=%.3f min=%.3f (fused repeat)\n", ms/1e6, is_/1e6, ns_/1e6);
    double elim_mib = (double)(2 * tbytes) / (1024.0*1024.0);  /* predicted z[] round trip = 2*n*4*rows */
    printf("PAIRED speedup_unfused_over_fused=%.4f drift_sentinel=%.4f predicted_elim_MiB=%.1f wall_delta_ms=%.3f\n",
           ratio, drift, elim_mib, (mu - mf)/1e6);
    printf("PAIRED fold_fused=0x%016llx fold_unfused=0x%016llx values_equal=%s\n",
           (unsigned long long)foldf, (unsigned long long)foldu, foldf == foldu ? "YES" : "NO");
    return 0;
}
