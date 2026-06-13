#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int run_case(size_t n, int pattern) {
  /* expected: (int32_t)lhs[index] */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int16_t *lhs = (int16_t *)malloc(sizeof(int16_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!lhs || !out) {
    fprintf(stderr, "allocation failed for n=%zu\n", n);
    free(lhs);
    free(out);
    return 11;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    if (pattern == 0) {
      lhs[index] = (int16_t)(((index % 2) == 0) ? -((int)(index % 127) + 1) : ((int)(index % 127) + 1));
    } else {
      lhs[index] = (int16_t)(((index % 6) == 0)
                                 ? -32768
                                 : ((index % 6) == 1)
                                       ? -257
                                       : ((index % 6) == 2)
                                             ? -1
                                             : ((index % 6) == 3)
                                                   ? 0
                                                   : ((index % 6) == 4)
                                                         ? 255
                                                         : (1024 + (int)(index % 31)));
    }
    out[index] = (int32_t)0x5a5a5a5a;
  }

  tcrv_emitc_pre_realized_body_widen_i16_to_i32_kernel_pre_realized_body_rvv_widen_i16_to_i32(lhs, out, n);

  size_t negative_lanes = 0;
  size_t positive_lanes = 0;
  size_t wide_magnitude_lanes = 0;
  for (size_t index = 0; index < n; ++index) {
    int32_t expected = (int32_t)lhs[index];
    if (lhs[index] < 0)
      ++negative_lanes;
    if (lhs[index] > 0)
      ++positive_lanes;
    if (lhs[index] <= -129 || lhs[index] >= 129)
      ++wide_magnitude_lanes;
    if (out[index] != expected) {
      fprintf(stderr,
              "widen_i16_to_i32 mismatch n=%zu pattern=%d index=%zu got=%d expected=%d lhs=%d\n",
              n, pattern, index, out[index], expected, lhs[index]);
      free(lhs);
      free(out);
      return 12;
    }
    if (lhs[index] < 0 && out[index] >= 0) {
      fprintf(stderr,
              "widen_i16_to_i32 sign-extension failed n=%zu pattern=%d index=%zu lhs=%d out=%d\n",
              n, pattern, index, lhs[index], out[index]);
      free(lhs);
      free(out);
      return 13;
    }
  }

  for (size_t index = n; index < alloc_n; ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "widen_i16_to_i32 touched tail sentinel n=%zu pattern=%d raw_index=%zu got=%d sentinel=%d\n",
              n, pattern, index, out[index], (int32_t)0x5a5a5a5a);
      free(lhs);
      free(out);
      return 14;
    }
  }

  if (n > 1 && (negative_lanes == 0 || positive_lanes == 0)) {
    fprintf(stderr,
            "widen_i16_to_i32 sign-extension coverage missing n=%zu pattern=%d negative_lanes=%zu positive_lanes=%zu\n",
            n, pattern, negative_lanes, positive_lanes);
    free(lhs);
    free(out);
    return 15;
  }
  if (pattern == 1 && n > 1 && wide_magnitude_lanes == 0) {
    fprintf(stderr,
            "widen_i16_to_i32 wide-magnitude coverage missing n=%zu pattern=%d\n",
            n, pattern);
    free(lhs);
    free(out);
    return 16;
  }

  free(lhs);
  free(out);
  printf("widen_i16_to_i32 case n=%zu pattern=%d ok sign_extension_checked two_input_patterns_checked tail_preserved\n",
         n, pattern);
  return 0;
}

int main(void) {
  const size_t counts[] = {1, 7, 16, 17, 257};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const int pattern_count = 2;
  for (size_t index = 0; index < count_count; ++index) {
    for (int pattern = 0; pattern < pattern_count; ++pattern) {
      int status = run_case(counts[index], pattern);
      if (status != 0)
        return status;
    }
  }
  printf("tcrv_rvv_generated_bundle_abi_widen_i16_to_i32_ok counts=1,7,16,17,257\n");
  printf("PASS op=widen_i16_to_i32 counts=1,7,16,17,257\n");
  return 0;
}
