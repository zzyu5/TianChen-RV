// harness.cpp -- drives the iq2_xxs GRID-of-8 decode vec_dot kernels (CORE EmitC,
// EXPLICIT integer_core_lmul: m2 = production default anchor, m1 = VLEN256 refinement)
// for the theta20 measured-table board check on `ssh k1` (VLEN256, the ONLY board where
// m1 is legal -- e8m1 VLMAX 32 == the 32-element sub-block; at VLEN128 m1 is verifier-
// rejected, VLMAX 16 < 32).
//
// The two kernels (kernels/iq2xxs_{m2,m1}.cpp) are the byte-for-byte
//   weft-opt <scaffold, integer_core_lmul=X [+minimum_vlen=256]> --weft-rvv-lower-to-emitc
//   | mlir-translate-20 --mlir-to-cpp
// emit of the SAME iq2_xxs typed super-block SCALAR-accumulator GRID loop body, with ONLY
// the extern-"C" symbol renamed to iq2xxs_{m2,m1}. m2 uses vint8m2/vint16m4/vint64m4 wide
// gathers; m1 uses vint8m1/vint16m2/vint64m2. BOARD FINDING: the anchors are VLEN-PINNED by
// the pair-batched vget register-group geometry -- exactly ONE is byte-exact vs the
// INDEPENDENT scalar oracle per VLEN (m2 @VLEN128 / m1 @VLEN256), the other silent-wrong.
// The per-arm mism counts below (m2_vs_oracle_mism / m1_vs_oracle_mism) expose this: at any
// board VLEN one arm is mism=0 (correct anchor) and the other is mism=M (wrong-VLEN anchor).
//
// Kernel ABI (both): f(const uint8_t *weight, const uint8_t *act, float *s, size_t n).
//   weight (block_iq2_xxs, stride 66): d(fp16)@0 | qs[32]u16 @2 (per sub-block ib32 the
//     8 bytes qs[ib32*8..]: bytes[0..3] = 4 grid INDICES, bytes[4..7] = aux1 little-endian
//     -> ls = 2*(aux1>>28)+1, sign selector l = (aux1>>7l)&127).
//   act (block_q8_K, stride 292): d(fp32)@0 | qs[256]i8 @4 | bsums[16]i16 @260 (unused).
//   *s = 0.125f * sum_ib d_ib * bsum_ib  (decode vec_dot, ONE float per weight row).
// Decode GEVM: M weight rows x ONE shared activation; out[r] = dot(weight_row_r, act).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include "oracle_tables.h"

extern "C" void iq2xxs_m2(const uint8_t *w, const uint8_t *a, float *s, size_t n);
extern "C" void iq2xxs_m1(const uint8_t *w, const uint8_t *a, float *s, size_t n);

typedef _Float16 f16;

static uint64_t rng_state;
static inline uint32_t xr() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)rng_state;
}
static inline uint8_t rnd_u8() { return (uint8_t)(xr() & 0xff); }
static inline int8_t rnd_i8() { return (int8_t)((int)(xr() % 255) - 127); }
static inline f16 rnd_scale() {
  return (f16)(0.008f + (float)(xr() % 48) * 0.001f);
}

// block sizes.
static const size_t WB = 66;   // sizeof block_iq2_xxs
static const size_t AB = 292;  // sizeof block_q8_K

// build M weight rows (each nb block_iq2_xxs) + ONE activation (nb block_q8_K).
static void build(std::vector<uint8_t> &W, std::vector<uint8_t> &A, size_t M,
                  size_t K, uint64_t seed) {
  rng_state = seed ? seed : 0x9e3779b97f4a7c15ull;
  size_t nb = K / 256;
  W.assign(M * nb * WB, 0);
  A.assign(nb * AB, 0);
  for (size_t r = 0; r < M; ++r)
    for (size_t b = 0; b < nb; ++b) {
      uint8_t *blk = &W[(r * nb + b) * WB];
      ((f16 *)blk)[0] = rnd_scale();          // fp16 d @0
      for (size_t k = 0; k < 64; ++k)          // qs[32] u16 = 64 raw bytes @2
        blk[2 + k] = rnd_u8();
    }
  for (size_t b = 0; b < nb; ++b) {
    uint8_t *blk = &A[b * AB];
    ((float *)blk)[0] = 0.01f + (float)(xr() % 40) * 0.001f;  // fp32 d @0
    for (size_t p = 0; p < 256; ++p)           // qs[256] i8 @4
      blk[4 + p] = (uint8_t)rnd_i8();
    // bsums @260 left 0 (kernel never reads them)
  }
}

// INDEPENDENT scalar oracle (ZERO-MODEL): scalar nested loops over the ggml iq2_xxs
// algorithm, sign decode via the hand-embedded ksigns_iq2xs/kmask_iq2xs (NOT the kernel's
// vectorized batched gather nor its expanded signs64).
static float oracle_dot(const uint8_t *xw, const uint8_t *ya, size_t K) {
  size_t nb = K / 256;
  float sumf = 0.0f;
  for (size_t i = 0; i < nb; ++i) {
    const uint8_t *xb = xw + i * WB;
    const uint8_t *yb = ya + i * AB;
    float d = (float)*(const f16 *)(xb) * *(const float *)(yb);
    const uint16_t *q2 = (const uint16_t *)(xb + 2);
    const int8_t *q8 = (const int8_t *)(yb + 4);
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
      uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
      q2 += 4;
      uint32_t ls = 2 * (aux1 >> 28) + 1;
      const uint8_t *aux8 = (const uint8_t *)&aux0;  // 4 grid indices
      int32_t sumi = 0;
      for (int l = 0; l < 4; ++l) {
        const int8_t *grid = (const int8_t *)(oracle_grid + aux8[l]);
        uint8_t signs = ksigns_iq2xs[(aux1 >> 7 * l) & 127];
        for (int j = 0; j < 8; ++j)
          sumi += (int)grid[j] * (int)q8[j] * ((signs & kmask_iq2xs[j]) ? -1 : 1);
        q8 += 8;
      }
      bsum += sumi * (int)ls;
    }
    sumf += d * (float)bsum;
  }
  return 0.125f * sumf;
}

static std::vector<uint8_t> g_junk(64ull * 1024 * 1024);
static void flush_caches() {
  volatile uint64_t s = 0;
  for (size_t i = 0; i < g_junk.size(); i += 64)
    s += g_junk[i];
  (void)s;
}

static uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

typedef void (*kern_t)(const uint8_t *, const uint8_t *, float *, size_t);

static uint64_t time_cold(kern_t k, size_t M, size_t K, const uint8_t *W,
                          const uint8_t *A, float *out, int reps) {
  size_t nb = K / 256, rowbytes = nb * WB;
  std::vector<uint64_t> samples;
  for (int r = 0; r < reps; ++r) {
    flush_caches();
    uint64_t t0 = now_ns();
    for (size_t row = 0; row < M; ++row)
      k(W + row * rowbytes, A, &out[row], K);
    uint64_t t1 = now_ns();
    samples.push_back(t1 - t0);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

int main() {
  struct Cfg {
    size_t M, K;
    const char *tag;
  };
  Cfg cfgs[] = {
      {512, 2048, "M=512 K=2048 cache-resident"},
      {8192, 4096, "M=8192 K=4096 ~8.6MB"},
      {32768, 4096, "M=32768 K=4096 ~34.6MB DRAM-bound"},
  };
  int reps = 7;
  // Per-arm correctness vs the INDEPENDENT scalar oracle, counted over ALL M rows: this
  // exposes the VLEN-pinning. m2_mism==0 iff m2 is the correct anchor at this board's VLEN;
  // m1_mism==0 iff m1 is. Exactly ONE is 0 per VLEN (m2@128 / m1@256) -- the wall.
  printf("footprint,seed,weight_bytes,m2_median_ns,m1_median_ns,m1_over_m2,"
         "m2_vs_oracle_mism,m1_vs_oracle_mism,rows\n");
  for (auto &c : cfgs) {
    size_t nb = c.K / 256, rowbytes = nb * WB;
    for (uint64_t seed = 0; seed < 2; ++seed) {
      std::vector<uint8_t> W, A;
      build(W, A, c.M, c.K, seed + 1);
      std::vector<float> ref(c.M), o_m2(c.M, 0.0f), o_m1(c.M, 0.0f);
      for (size_t r = 0; r < c.M; ++r)
        ref[r] = oracle_dot(W.data() + r * rowbytes, A.data(), c.K);
      for (size_t r = 0; r < c.M; ++r) {
        iq2xxs_m2(W.data() + r * rowbytes, A.data(), &o_m2[r], c.K);
        iq2xxs_m1(W.data() + r * rowbytes, A.data(), &o_m1[r], c.K);
      }
      size_t m2_mism = 0, m1_mism = 0;
      for (size_t i = 0; i < c.M; ++i) {
        if (o_m2[i] != ref[i]) ++m2_mism;
        if (o_m1[i] != ref[i]) ++m1_mism;
      }
      uint64_t t_m2 = time_cold(iq2xxs_m2, c.M, c.K, W.data(), A.data(),
                                o_m2.data(), reps);
      uint64_t t_m1 = time_cold(iq2xxs_m1, c.M, c.K, W.data(), A.data(),
                                o_m1.data(), reps);
      printf("%s,%llu,%zu,%llu,%llu,%.4f,%zu,%zu,%zu\n", c.tag,
             (unsigned long long)seed, W.size(), (unsigned long long)t_m2,
             (unsigned long long)t_m1, (double)t_m1 / (double)t_m2,
             m2_mism, m1_mism, c.M);
      fflush(stdout);
    }
  }
  return 0;
}
