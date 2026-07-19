// harness_gemm.cpp -- drives the DEPLOYED q8_0 repack-GEMM (PREFILL) kernels
// (CORE==PROD emit) for the [GAP-P1]-loosen rollout board check on the PREFILL
// regime. Unlike the GEVM (decode) crossover, the mf2 GEMM folds columnsPerPass=4
// (amortizing weight-strip loads across 4 activation columns) while the m1 GEMM
// uses columnsPerPass=1 (one 16-lane f32m4 strip) -- a DIFFERENT tradeoff, so it
// is measured SEPARATELY here (not extrapolated from the GEVM).
//
// Kernel ABI: gemm(M, out_stride, K, out, N, weight_x16, act_x4).
//   weight (block_q8_0x16): per (col_group cg in 0..N/16, block b in 0..K/32) a
//     544-byte block at weight+(cg*(K/32)+b)*544: [0..31]=16 f16 col scales;
//     [32+p*16+cl]=int8 weight (position p, column cl in 0..16).
//   act (block_q8_0x4): per (row_group rg in 0..M/4, block b) a 136-byte block at
//     act+(rg*(K/32)+b)*136: [r*2]=f16 scale for row r (r in 0..4);
//     [8+p*4+r]=int8 activation (position p, row r).
//   out[row*N + cg*16 + cl] = sum_b (sum_p w*a) * (wscale*ascale), row = rg*4+r.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

extern "C" void q8_gemm_mf2(size_t M, size_t obs, size_t K, float *out, size_t N,
                            const uint8_t *w, const uint8_t *a);
extern "C" void q8_gemm_m1(size_t M, size_t obs, size_t K, float *out, size_t N,
                           const uint8_t *w, const uint8_t *a);
typedef _Float16 f16;

static uint64_t rs;
static inline uint32_t xr() {
  rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs;
}
static inline int8_t ri8() { return (int8_t)((int)(xr() % 255) - 127); }
static inline f16 rsc() { return (f16)(0.008f + (float)(xr() % 48) * 0.001f); }

static void build(std::vector<uint8_t> &W, std::vector<uint8_t> &A, size_t M,
                  size_t N, size_t K, uint64_t seed) {
  rs = seed ? seed : 0x9e3779b97f4a7c15ull;
  size_t nb = K / 32, ncg = N / 16, nrg = M / 4;
  W.assign(ncg * nb * 544, 0);
  A.assign(nrg * nb * 136, 0);
  for (size_t cg = 0; cg < ncg; ++cg)
    for (size_t b = 0; b < nb; ++b) {
      uint8_t *blk = &W[(cg * nb + b) * 544];
      for (size_t c = 0; c < 16; ++c) ((f16 *)blk)[c] = rsc();
      for (size_t p = 0; p < 32; ++p)
        for (size_t c = 0; c < 16; ++c) blk[32 + p * 16 + c] = (uint8_t)ri8();
    }
  for (size_t rg = 0; rg < nrg; ++rg)
    for (size_t b = 0; b < nb; ++b) {
      uint8_t *blk = &A[(rg * nb + b) * 136];
      for (size_t r = 0; r < 4; ++r) ((f16 *)blk)[r] = rsc();
      for (size_t p = 0; p < 32; ++p)
        for (size_t r = 0; r < 4; ++r) blk[8 + p * 4 + r] = (uint8_t)ri8();
    }
}

static void oracle(const std::vector<uint8_t> &W, const std::vector<uint8_t> &A,
                   std::vector<float> &out, size_t M, size_t N, size_t K) {
  size_t nb = K / 32;
  out.assign(M * N, 0.0f);
  for (size_t rg = 0; rg < M / 4; ++rg)
    for (size_t r = 0; r < 4; ++r) {
      size_t row = rg * 4 + r;
      for (size_t cg = 0; cg < N / 16; ++cg)
        for (size_t cl = 0; cl < 16; ++cl) {
          size_t col = cg * 16 + cl;
          float acc = 0.0f;
          for (size_t b = 0; b < nb; ++b) {
            const uint8_t *wb = &W[(cg * nb + b) * 544];
            const uint8_t *ab = &A[(rg * nb + b) * 136];
            int32_t isum = 0;
            for (size_t p = 0; p < 32; ++p)
              isum += (int32_t)(int8_t)wb[32 + p * 16 + cl] *
                      (int32_t)(int8_t)ab[8 + p * 4 + r];
            float scale = (float)((const f16 *)wb)[cl] * (float)((const f16 *)ab)[r];
            acc = fmaf((float)isum, scale, acc);
          }
          out[row * N + col] = acc;
        }
    }
}

static std::vector<uint8_t> junk(64ull * 1024 * 1024);
static void flush() {
  volatile uint64_t s = 0;
  for (size_t i = 0; i < junk.size(); i += 64) s += junk[i];
  (void)s;
}
static uint64_t now_ns() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
typedef void (*kern_t)(size_t, size_t, size_t, float *, size_t, const uint8_t *,
                       const uint8_t *);
static uint64_t tcold(kern_t k, size_t M, size_t K, float *o, size_t N,
                      const uint8_t *w, const uint8_t *a, int reps) {
  std::vector<uint64_t> s;
  for (int r = 0; r < reps; ++r) {
    flush();
    uint64_t t0 = now_ns();
    k(M, N, K, o, N, w, a);
    uint64_t t1 = now_ns();
    s.push_back(t1 - t0);
  }
  std::sort(s.begin(), s.end());
  return s[s.size() / 2];
}

int main() {
  struct Cfg { size_t M, N, K; const char *tag; };
  Cfg cfgs[] = {
      {256, 256, 1024, "M=256 N=256 K=1024"},
      {512, 256, 2048, "M=512 N=256 K=2048"},
      {512, 512, 2048, "M=512 N=512 K=2048"},
  };
  int reps = 5;
  printf("footprint,seed,mf2_median_ns,m1_median_ns,m1_over_mf2,byte_exact_3arm\n");
  int all = 1;
  for (auto &c : cfgs)
    for (uint64_t seed = 0; seed < 2; ++seed) {
      std::vector<uint8_t> W, A;
      build(W, A, c.M, c.N, c.K, seed + 1);
      std::vector<float> ref;
      oracle(W, A, ref, c.M, c.N, c.K);
      std::vector<float> omf2(c.M * c.N, 0.0f), om1(c.M * c.N, 0.0f);
      q8_gemm_mf2(c.M, c.N, c.K, omf2.data(), c.N, W.data(), A.data());
      q8_gemm_m1(c.M, c.N, c.K, om1.data(), c.N, W.data(), A.data());
      int ex = (memcmp(omf2.data(), ref.data(), c.M * c.N * sizeof(float)) == 0) &&
               (memcmp(om1.data(), ref.data(), c.M * c.N * sizeof(float)) == 0);
      if (!ex) {
        all = 0;
        for (size_t i = 0; i < c.M * c.N; ++i)
          if (omf2[i] != ref[i] || om1[i] != ref[i]) {
            fprintf(stderr, "MISMATCH %s seed %llu idx %zu: ref=%.9g mf2=%.9g m1=%.9g\n",
                    c.tag, (unsigned long long)seed, i, ref[i], omf2[i], om1[i]);
            break;
          }
      }
      uint64_t mf2 = tcold(q8_gemm_mf2, c.M, c.K, omf2.data(), c.N, W.data(), A.data(), reps);
      uint64_t m1 = tcold(q8_gemm_m1, c.M, c.K, om1.data(), c.N, W.data(), A.data(), reps);
      printf("%s,%llu,%llu,%llu,%.4f,%s\n", c.tag, (unsigned long long)seed,
             (unsigned long long)mf2, (unsigned long long)m1,
             (double)m1 / (double)mf2, ex ? "PASS" : "FAIL");
      fflush(stdout);
    }
  fprintf(stderr, "3-arm byte-exact (GEMM, all): %s\n", all ? "PASS mism=0" : "FAIL");
  return all ? 0 : 1;
}
