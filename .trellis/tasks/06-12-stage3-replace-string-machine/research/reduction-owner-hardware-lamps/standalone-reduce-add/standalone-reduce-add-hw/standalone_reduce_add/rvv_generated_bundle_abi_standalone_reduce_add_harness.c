#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int32_t make_lhs_value(size_t index, int pattern) {
  if (pattern == 0)
    return (int32_t)(((index % 7) < 3)
                         ? -((int32_t)(index % 31) + 1)
                         : ((int32_t)(index % 37) + 2));
  return (int32_t)(((index % 5) < 2)
                       ? ((int32_t)(index % 41) - 19)
                       : -((int32_t)(index % 23) + 5));
}

static int run_case(size_t n, int32_t seed, int pattern) {
  /* expected: (int32_t)(acc[0] + sum_i(lhs[i])) */
  size_t alloc_n = n == 0 ? 1 : n;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t acc[1];
  int32_t out[4];
  if (!lhs) {
    fprintf(stderr, "allocation failed for n=%zu\n", n);
    return 11;
  }

  int32_t expected = seed;
  for (size_t index = 0; index < alloc_n; ++index) {
    lhs[index] = make_lhs_value(index, pattern);
    if (index < n)
      expected = (int32_t)(expected + lhs[index]);
  }
  acc[0] = seed;
  for (size_t index = 0; index < sizeof(out) / sizeof(out[0]); ++index)
    out[index] = (int32_t)0x5a5a5a5a;

  tcrv_emitc_explicit_selected_body_standalone_reduce_add_kernel_explicit_selected_body_rvv_standalone_reduce_add(lhs, acc, out, n);

  if (out[0] != expected) {
    fprintf(stderr,
            "standalone_reduce_add mismatch n=%zu seed=%d pattern=%d got=%d expected=%d\n",
            n, seed, pattern, out[0], expected);
    free(lhs);
    return 12;
  }
  if (acc[0] != seed) {
    fprintf(stderr,
            "standalone_reduce_add mutated seed input n=%zu pattern=%d got=%d expected=%d\n",
            n, pattern, acc[0], seed);
    free(lhs);
    return 13;
  }
  for (size_t index = 0; index < alloc_n; ++index) {
    int32_t expected_lhs = make_lhs_value(index, pattern);
    if (lhs[index] != expected_lhs) {
      fprintf(stderr,
              "standalone_reduce_add mutated source input n=%zu seed=%d pattern=%d index=%zu got=%d expected=%d\n",
              n, seed, pattern, index, lhs[index], expected_lhs);
      free(lhs);
      return 14;
    }
  }
  for (size_t index = 1; index < sizeof(out) / sizeof(out[0]); ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "standalone_reduce_add touched scalar-output sentinel n=%zu seed=%d pattern=%d index=%zu got=%d sentinel=%d\n",
              n, seed, pattern, index, out[index], (int32_t)0x5a5a5a5a);
      free(lhs);
      return 15;
    }
  }

  free(lhs);
  printf("standalone_reduce_add case n=%zu seed=%d pattern=%d ok scalar_out=%d tail_preserved source_preserved\n",
         n, seed, pattern, out[0]);
  return 0;
}

int main(void) {
  const size_t counts[] = {0, 1, 16, 17, 257};
  const int32_t seeds[] = {(int32_t)-11, (int32_t)17};
  const int patterns[] = {0, 1};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const size_t seed_count = sizeof(seeds) / sizeof(seeds[0]);
  const size_t pattern_count = sizeof(patterns) / sizeof(patterns[0]);
  for (size_t seed_index = 0; seed_index < seed_count; ++seed_index) {
    for (size_t pattern_index = 0; pattern_index < pattern_count;
         ++pattern_index) {
      for (size_t count_index = 0; count_index < count_count; ++count_index) {
        int status = run_case(counts[count_index], seeds[seed_index],
                              patterns[pattern_index]);
        if (status != 0)
          return status;
      }
    }
  }
  printf("tcrv_rvv_generated_bundle_abi_standalone_reduce_add_ok counts=0,1,16,17,257 seeds=-11,17 patterns=0,1\n");
  printf("PASS op=standalone_reduce_add counts=0,1,16,17,257 seeds=-11,17 patterns=0,1\n");
  return 0;
}
