// INC-9 q4_1 x q8_1 RIGOROUS microbench (ssh rvv).
//
// Measures the COMPILER's autotuner-selected structured emission for the ggml
// ggml_vec_dot_q4_1_q8_1 kernel against ggml's REAL hand-written RVV kernel
// (quants.c:277-326). q4_1 is Family B (scale+MIN, asymmetric): the compiler
// emits the UNSIGNED-nibble decode + the (d_x*d_y)*sumi + m_x*s_y fold.
//
//   ggml(real,separate-TU) : ggml_vec_dot_q4_1_q8_1 (m1, 1 vwredsum/block, no
//                            outer unroll) -- the kernel we replace
//   fullv_autotuned        : full-V AUTOTUNER PICK (m1, factor=4, elided) -- the
//                            latency-overlap lever ggml does NOT use
//   zve32x_autotuned       : zve32x AUTOTUNER PICK (m1, factor=2, robust) -- the
//                            VLEN-robust shape (timed here on the full-V board for
//                            a same-board comparison; it is correct at any VLEN)
//
// Methodology mirrors the q4_0/q8_0 probes so ns/call is comparable: taskset -c 3
// pinned, best-of-N min ns/call, warmup discarded, identical random data,
// correctness-gated (each shape bit-exact vs ggml on the bench data before
// timing), INTERLEAVED rounds (drift hits all kernels equally). Build:
// -march=rv64gcv -O3 -ffp-contract=fast. HONEST: report match/beat/lose as
// measured.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 32
#define Q4_STRIDE 20
#define Q8_STRIDE 36

static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);
extern "C" {
void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
// The compiler-emitted kernels (their exported symbol names).
void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_fullv(
    size_t, float *, const uint8_t *, const uint8_t *);
void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_zve32x(
    size_t, float *, const uint8_t *, const uint8_t *);
}

// ggml's REAL hand-written q4_1 RVV kernel inline (correctness reference, never
// timed -- the TIMED ggml is the separate-TU kern_ggml).
static void ggml_real(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / QK;
  float sumf = 0;
  size_t vl = QK / 2;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xq = vx + ib * Q4_STRIDE + 4;
    const int8_t *yq = (const int8_t *)(vy + ib * Q8_STRIDE + 4);
    uint16_t xd, xm, yd, ys;
    memcpy(&xd, vx + ib * Q4_STRIDE + 0, 2);
    memcpy(&xm, vx + ib * Q4_STRIDE + 2, 2);
    memcpy(&yd, vy + ib * Q8_STRIDE + 0, 2);
    memcpy(&ys, vy + ib * Q8_STRIDE + 2, 2);
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xq, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1(yq, vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1(yq + 16, vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
    vint8m1_t v1 = __riscv_vreinterpret_v_u8m1_i8m1(x_l);
    vint16m2_t m1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    vint16m2_t m2 = __riscv_vwmacc_vv_i16m2(m1, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t r = __riscv_vwredsum_vs_i16m2_i32m1(m2, z, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(r);
    sumf += (fp16_to_fp32(xd) * fp16_to_fp32(yd)) * sumi +
            fp16_to_fp32(xm) * fp16_to_fp32(ys);
  }
  *s = sumf;
}

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void put_fp16(uint8_t *p) {
  _Float16 d = (_Float16)(((float)(int)(xr() % 2001) - 1000.0f) / 256.0f);
  uint16_t dh; memcpy(&dh, &d, 2); memcpy(p, &dh, 2);
}
static void fill_q4(uint8_t *b, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *p = b + i * Q4_STRIDE;
    put_fp16(p + 0); put_fp16(p + 2);
    for (int j = 0; j < QK / 2; j++) p[4 + j] = (uint8_t)(xr() % 256);
  }
}
static void fill_q8(uint8_t *b, int nb) {
  for (int i = 0; i < nb; i++) {
    uint8_t *p = b + i * Q8_STRIDE;
    put_fp16(p + 0); put_fp16(p + 2);
    for (int j = 0; j < QK; j++) p[4 + j] = (uint8_t)(int8_t)(xr() % 256);
  }
}

static double now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

struct Entry { const char *name; kfn fn; double best; double sum; int cnt; };

static double timed_batch(kfn fn, int n, int iters, const uint8_t *vx,
                          const uint8_t *vy) {
  double t0 = now_ns();
  float s;
  for (int i = 0; i < iters; i++)
    fn(n, &s, vx, vy);
  return (now_ns() - t0) / iters;
}

int main() {
  const int n = 4096;
  const int nb = n / QK;
  const int iters = 2000;
  const int reps = 200;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * Q4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)nb * Q8_STRIDE);
  fill_q4(vx, nb);
  fill_q8(vy, nb);

  Entry kernels[] = {
      {"ggml(real,separate-TU)", kern_ggml, 1e18, 0, 0},
      {"fullv_autotuned(m1,mb4,elided)",
       tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_fullv,
       1e18, 0, 0},
      {"zve32x_autotuned(m1,mb2,robust)",
       tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_zve32x,
       1e18, 0, 0},
  };
  const int K = sizeof(kernels) / sizeof(kernels[0]);

  // Correctness gate: every shape must be bit-exact vs ggml on the bench data.
  float ref = 0;
  ggml_real(n, &ref, vx, vy);
  for (Entry &e : kernels) {
    float got = 0;
    e.fn(n, &got, vx, vy);
    if (memcmp(&got, &ref, 4) != 0) {
      printf("CORRECTNESS GATE FAILED: %s = %.9g != ggml %.9g\n", e.name, got, ref);
      return 1;
    }
  }
  printf("correctness gate: all shapes bit-exact vs ggml on bench data\n");

  for (Entry &e : kernels)
    for (int i = 0; i < iters; i++) { float s; e.fn(n, &s, vx, vy); }

  for (int r = 0; r < reps; r++)
    for (int k = 0; k < K; k++) {
      double per = timed_batch(kernels[k].fn, n, iters, vx, vy);
      if (per < kernels[k].best) kernels[k].best = per;
      kernels[k].sum += per;
      kernels[k].cnt += 1;
    }

  double ggml_ns = kernels[0].best;
  printf("%-40s %10s %10s   %s\n", "shape", "best-ns", "mean-ns", "ggml/best");
  for (Entry &e : kernels) {
    double mean = e.sum / e.cnt;
    double speedup = ggml_ns / e.best; // >1 => our shape is FASTER than ggml
    printf("%-40s %10.1f %10.1f   %.3fx\n", e.name, e.best, mean, speedup);
  }
  free(vx);
  free(vy);
  return 0;
}
