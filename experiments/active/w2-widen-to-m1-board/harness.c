// harness.c -- W2 §③ widen-to-m1 board driver.
//
// Workload = a decode-time GEVM: out[M] = W[M x K] . act[K], one length-K int8
// widening dot-reduce per output row. This is the memory-traffic profile the
// repack path actually serves at decode. We run it with the mf2 chain and the
// m1 chain and report the per-chain cost so the [GAP-P1] crossover is a MEASURED
// board number, not a projection.
//
// Discipline:
//   * byte-exact: every chain's full out[] is compared element-wise to the
//     scalar oracle; a single mismatch aborts (ZERO-MODEL: recompute from the
//     raw inputs, 0 mismatch required).
//   * cold: each measured pass is preceded by a cache-eviction stream over a
//     buffer larger than LLC, so the GEVM starts cold (DRAM-bound, decode-like).
//   * 2 seeds: the whole sweep runs for two independent RNG seeds; a claim only
//     stands if it holds on both.
//
// Prints one CSV row per (format, chain, seed): min/median cold cycles over
// NPASS cold passes, plus the byte-exact verdict.

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int32_t dot_q8_scalar(const int8_t *, const int8_t *, size_t);
int32_t dot_q4_scalar(const uint8_t *, const int8_t *, size_t);
int32_t dot_q8_mf2(const int8_t *, const int8_t *, size_t);
int32_t dot_q8_m1(const int8_t *, const int8_t *, size_t);
int32_t dot_q4_mf2(const uint8_t *, const int8_t *, int8_t *, size_t);
int32_t dot_q4_m1(const uint8_t *, const int8_t *, int8_t *, size_t);

#ifndef K_DIM
#define K_DIM 4096
#endif
#ifndef M_DIM
#define M_DIM 2048
#endif
#ifndef NPASS
#define NPASS 5
#endif

// Wall-clock nanoseconds (CLOCK_MONOTONIC): reliable in userspace, unlike
// rdcycle which SIGILLs on this board's Linux config. The m1-vs-mf2 comparison
// is a RATIO on the same clock, so the unit choice does not bias it.
static inline uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// xorshift64 -- deterministic per-seed fill.
static uint64_t rng_state;
static inline uint8_t rng_byte(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint8_t)(rng_state >> 24);
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

// Large eviction buffer (> any LLC on this class of board).
#define EVICT_BYTES (64u * 1024u * 1024u)
static volatile uint8_t *g_evict;
static void evict_caches(void) {
  volatile uint64_t sink = 0;
  for (size_t i = 0; i < EVICT_BYTES; i += 64)
    sink += g_evict[i];
  (void)sink;
}

int main(int argc, char **argv) {
  const size_t K = K_DIM, M = M_DIM;
  const uint64_t seeds[2] = {0x123456789abcdef0ULL, 0x0fedcba987654321ULL};

  g_evict = malloc(EVICT_BYTES);
  memset((void *)g_evict, 1, EVICT_BYTES);

  int8_t *w8 = malloc(M * K);                 // q8 weights (int8)
  uint8_t *w4 = malloc(M * (K / 2));          // q4 weights (packed nibbles)
  int8_t *act = malloc(K);                    // activation (int8)
  int8_t *scratch = malloc(K);                // q4 unpack scratch
  int32_t *out = malloc(M * sizeof(int32_t)); // kernel output
  int32_t *ref = malloc(M * sizeof(int32_t)); // scalar oracle
  if (!w8 || !w4 || !act || !scratch || !out || !ref) {
    fprintf(stderr, "alloc failed\n");
    return 2;
  }

  printf("# W2 widen-to-m1 board: GEVM out[M]=W[MxK].act[K]  M=%zu K=%zu "
         "NPASS=%d (cold)\n",
         M, K, NPASS);
  printf("format,chain,seed,min_ns,median_ns,byte_exact\n");

  for (int s = 0; s < 2; ++s) {
    rng_state = seeds[s] ? seeds[s] : 1;
    for (size_t i = 0; i < M * K; ++i)
      w8[i] = (int8_t)rng_byte();
    for (size_t i = 0; i < M * (K / 2); ++i)
      w4[i] = rng_byte();
    for (size_t i = 0; i < K; ++i)
      act[i] = (int8_t)rng_byte();

    // ---- q8 format: oracle then mf2 / m1 ----
    for (size_t r = 0; r < M; ++r)
      ref[r] = dot_q8_scalar(w8 + r * K, act, K);

    struct {
      const char *name;
      int is_m1;
    } q8chains[2] = {{"mf2", 0}, {"m1", 1}};
    for (int c = 0; c < 2; ++c) {
      uint64_t cyc[NPASS];
      int exact = 1;
      for (int p = 0; p < NPASS; ++p) {
        evict_caches();
        uint64_t t0 = now_ns();
        for (size_t r = 0; r < M; ++r)
          out[r] = q8chains[c].is_m1 ? dot_q8_m1(w8 + r * K, act, K)
                                     : dot_q8_mf2(w8 + r * K, act, K);
        uint64_t t1 = now_ns();
        cyc[p] = t1 - t0;
        for (size_t r = 0; r < M; ++r)
          if (out[r] != ref[r]) {
            exact = 0;
            break;
          }
      }
      qsort(cyc, NPASS, sizeof(uint64_t), cmp_u64);
      printf("q8,%s,%d,%llu,%llu,%s\n", q8chains[c].name, s,
             (unsigned long long)cyc[0], (unsigned long long)cyc[NPASS / 2],
             exact ? "PASS" : "FAIL");
    }

    // ---- q4 format: oracle then mf2 / m1 ----
    for (size_t r = 0; r < M; ++r)
      ref[r] = dot_q4_scalar(w4 + r * (K / 2), act, K);

    struct {
      const char *name;
      int is_m1;
    } q4chains[2] = {{"mf2", 0}, {"m1", 1}};
    for (int c = 0; c < 2; ++c) {
      uint64_t cyc[NPASS];
      int exact = 1;
      for (int p = 0; p < NPASS; ++p) {
        evict_caches();
        uint64_t t0 = now_ns();
        for (size_t r = 0; r < M; ++r)
          out[r] = q4chains[c].is_m1
                       ? dot_q4_m1(w4 + r * (K / 2), act, scratch, K)
                       : dot_q4_mf2(w4 + r * (K / 2), act, scratch, K);
        uint64_t t1 = now_ns();
        cyc[p] = t1 - t0;
        for (size_t r = 0; r < M; ++r)
          if (out[r] != ref[r]) {
            exact = 0;
            break;
          }
      }
      qsort(cyc, NPASS, sizeof(uint64_t), cmp_u64);
      printf("q4,%s,%d,%llu,%llu,%s\n", q4chains[c].name, s,
             (unsigned long long)cyc[0], (unsigned long long)cyc[NPASS / 2],
             exact ? "PASS" : "FAIL");
    }
  }

  free((void *)g_evict);
  free(w8);
  free(w4);
  free(act);
  free(scratch);
  free(out);
  free(ref);
  return 0;
}
