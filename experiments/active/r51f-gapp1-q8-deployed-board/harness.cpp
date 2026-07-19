// harness.cpp -- drives the DEPLOYED q8_0 repack-GEVM kernels (CORE==PROD emit,
// NOT the W2 standalone chains) for the [GAP-P1]-loosen rollout board check.
//
// The two kernels (kernels/q8_gevm_{mf2,m1}.cpp) are the byte-for-byte
// weft-opt --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc
// | mlir-translate --mlir-to-cpp emit of the SAME q8_0 decode repack-GEVM request
// (test/Conversion/RVV/rvv-lower-quant-contraction-q8-0-decode-repack-gevm.mlir):
//   * q8_gevm_mf2 = the selector's mf2 default  (pre-fill binary emit).
//   * q8_gevm_m1  = the selector's m1 whole-LMUL (measured-table-fill binary emit).
// Only the accumulator LMUL differs (half_lanes 8+8 vs 16); the arithmetic is
// identical, so the deployed kernels must be BYTE-EXACT to each other AND to the
// scalar oracle (3-arm, ZERO-MODEL mism=0).
//
// Kernel ABI (both): f(K, out, M, weight_x16, bx, act_plain, by, nrc).
//   weight (block_q8_0x16): per (col_group cg in 0..M/16, block b in 0..K/32) a
//     544-byte block at weight + (cg*(K/32)+b)*544: bytes[0..31] = 16 fp16 column
//     scales; bytes[32 + p*16 + c] = int8 weight (position p in 0..32, column c
//     in 0..16).
//   act (block_q8_0): per block b a 34-byte block at act + b*34: bytes[0..1] =
//     fp16 act scale; bytes[2 + p] = int8 activation (position p).
//   out[col] = sum_b ( (sum_p w_i8[c][b][p]*a_i8[b][p]) * (wscale[c][b]*ascale[b]) ).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

extern "C" void q8_gevm_mf2(size_t K, float *out, size_t M, const uint8_t *w,
                            size_t bx, const uint8_t *a, size_t by, int32_t nrc);
extern "C" void q8_gevm_m1(size_t K, float *out, size_t M, const uint8_t *w,
                           size_t bx, const uint8_t *a, size_t by, int32_t nrc);

typedef _Float16 f16;

static uint64_t rng_state;
static inline uint32_t xr() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)rng_state;
}
static inline int8_t rnd_i8() { return (int8_t)((int)(xr() % 255) - 127); }
static inline f16 rnd_scale() {
  return (f16)(0.008f + (float)(xr() % 48) * 0.001f);
}

static void build(std::vector<uint8_t> &W, std::vector<uint8_t> &A, size_t M,
                  size_t K, uint64_t seed) {
  rng_state = seed ? seed : 0x9e3779b97f4a7c15ull;
  size_t nblk = K / 32;
  size_t ncg = M / 16;
  W.assign(ncg * nblk * 544, 0);
  A.assign(nblk * 34, 0);
  for (size_t cg = 0; cg < ncg; ++cg)
    for (size_t b = 0; b < nblk; ++b) {
      uint8_t *blk = &W[(cg * nblk + b) * 544];
      for (size_t c = 0; c < 16; ++c)
        ((f16 *)blk)[c] = rnd_scale();
      for (size_t p = 0; p < 32; ++p)
        for (size_t c = 0; c < 16; ++c)
          blk[32 + p * 16 + c] = (uint8_t)rnd_i8();
    }
  for (size_t b = 0; b < nblk; ++b) {
    uint8_t *blk = &A[b * 34];
    ((f16 *)blk)[0] = rnd_scale();
    for (size_t p = 0; p < 32; ++p)
      blk[2 + p] = (uint8_t)rnd_i8();
  }
}

static void oracle(const std::vector<uint8_t> &W, const std::vector<uint8_t> &A,
                   std::vector<float> &out, size_t M, size_t K) {
  size_t nblk = K / 32;
  out.assign(M, 0.0f);
  for (size_t col = 0; col < M; ++col) {
    size_t cg = col / 16, lane = col % 16;
    float acc = 0.0f;
    for (size_t b = 0; b < nblk; ++b) {
      const uint8_t *blk = &W[(cg * nblk + b) * 544];
      const uint8_t *ablk = &A[b * 34];
      int32_t isum = 0;
      for (size_t p = 0; p < 32; ++p) {
        int8_t w = (int8_t)blk[32 + p * 16 + lane];
        int8_t a = (int8_t)ablk[2 + p];
        isum += (int32_t)w * (int32_t)a;
      }
      f16 ws = ((const f16 *)blk)[lane];
      f16 as = ((const f16 *)ablk)[0];
      float scale = (float)ws * (float)as;
      acc = fmaf((float)isum, scale, acc);
    }
    out[col] = acc;
  }
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

typedef void (*kern_t)(size_t, float *, size_t, const uint8_t *, size_t,
                       const uint8_t *, size_t, int32_t);

static uint64_t time_cold(kern_t k, size_t K, float *out, size_t M,
                          const uint8_t *w, const uint8_t *a, int reps) {
  std::vector<uint64_t> samples;
  for (int r = 0; r < reps; ++r) {
    flush_caches();
    uint64_t t0 = now_ns();
    k(K, out, M, w, 0, a, 0, 1);
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
      {256, 1024, "M=256 K=1024 cache-resident"},
      {2048, 4096, "M=2048 K=4096 ~8.5MB"},
      {8192, 4096, "M=8192 K=4096 ~35.6MB DRAM-bound"},
  };
  int reps = 7;
  printf("footprint,seed,weight_bytes,mf2_median_ns,m1_median_ns,m1_over_mf2,byte_exact_3arm\n");
  int all_exact = 1;
  for (auto &c : cfgs) {
    for (uint64_t seed = 0; seed < 2; ++seed) {
      std::vector<uint8_t> W, A;
      build(W, A, c.M, c.K, seed + 1);
      std::vector<float> ref;
      oracle(W, A, ref, c.M, c.K);
      std::vector<float> o_mf2(c.M, 0.0f), o_m1(c.M, 0.0f);
      q8_gevm_mf2(c.K, o_mf2.data(), c.M, W.data(), 0, A.data(), 0, 1);
      q8_gevm_m1(c.K, o_m1.data(), c.M, W.data(), 0, A.data(), 0, 1);
      int exact = (memcmp(o_mf2.data(), ref.data(), c.M * sizeof(float)) == 0) &&
                  (memcmp(o_m1.data(), ref.data(), c.M * sizeof(float)) == 0);
      if (!exact) {
        all_exact = 0;
        for (size_t i = 0; i < c.M; ++i)
          if (o_mf2[i] != ref[i] || o_m1[i] != ref[i]) {
            fprintf(stderr,
                    "MISMATCH %s seed %llu col %zu: ref=%.9g mf2=%.9g m1=%.9g\n",
                    c.tag, (unsigned long long)seed, i, ref[i], o_mf2[i],
                    o_m1[i]);
            break;
          }
      }
      uint64_t mf2 = time_cold(q8_gevm_mf2, c.K, o_mf2.data(), c.M, W.data(),
                               A.data(), reps);
      uint64_t m1 = time_cold(q8_gevm_m1, c.K, o_m1.data(), c.M, W.data(),
                              A.data(), reps);
      printf("%s,%llu,%zu,%llu,%llu,%.4f,%s\n", c.tag,
             (unsigned long long)seed, W.size(), (unsigned long long)mf2,
             (unsigned long long)m1, (double)m1 / (double)mf2,
             exact ? "PASS" : "FAIL");
      fflush(stdout);
    }
  }
  fprintf(stderr, "3-arm byte-exact (all footprints/seeds): %s\n",
          all_exact ? "PASS mism=0" : "FAIL");
  return all_exact ? 0 : 1;
}
