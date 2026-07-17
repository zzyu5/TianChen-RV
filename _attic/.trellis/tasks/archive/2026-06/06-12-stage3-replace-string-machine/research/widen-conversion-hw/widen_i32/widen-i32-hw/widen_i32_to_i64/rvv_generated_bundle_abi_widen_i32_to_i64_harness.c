#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int run_case(size_t n, int pattern) {
  /* expected: (int64_t)lhs[index] */
  size_t alloc_n = n + 5;
  if (alloc_n == 5 && n == 0)
    alloc_n = 6;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int64_t *out = (int64_t *)malloc(sizeof(int64_t) * alloc_n);
  if (!lhs || !out) {
    fprintf(stderr, "allocation failed for n=%zu\n", n);
    free(lhs);
    free(out);
    return 11;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    if (pattern == 0) {
      lhs[index] = ((index % 2) == 0 ? (int32_t)(-((int32_t)(index + 1) * 65537)) : (int32_t)(((int32_t)(index + 1) * 65537)));
    } else {
      lhs[index] = (int32_t)(((index % 7) == 0)
                                 ? INT32_MIN
                                 : ((index % 7) == 1)
                                       ? -123456789
                                       : ((index % 7) == 2)
                                             ? -1
                                             : ((index % 7) == 3)
                                                   ? 0
                                                   : ((index % 7) == 4)
                                                         ? 1
                                                         : ((index % 7) == 5)
                                                               ? 123456789
                                                               : INT32_MAX);
    }
    out[index] = (int64_t)0x5a5a5a5a5a5a5a5aLL;
  }

  tcrv_emitc_explicit_selected_body_widen_i32_to_i64_kernel_explicit_selected_body_rvv_widen_i32_to_i64(lhs, out, n);

  size_t negative_lanes = 0;
  size_t positive_lanes = 0;
  size_t wide_magnitude_lanes = 0;
  for (size_t index = 0; index < n; ++index) {
    int64_t expected = (int64_t)lhs[index];
    if (lhs[index] < 0)
      ++negative_lanes;
    if (lhs[index] > 0)
      ++positive_lanes;
    if (lhs[index] > 32767 || lhs[index] < -32768)
      ++wide_magnitude_lanes;
    if (out[index] != expected) {
      fprintf(stderr,
              "widen_i32_to_i64 mismatch n=%zu index=%zu got=%lld expected=%lld lhs=%d\n",
              n, index, (long long)out[index], (long long)expected, lhs[index]);
      free(lhs);
      free(out);
      return 12;
    }
    if (lhs[index] < 0 && out[index] >= 0) {
      fprintf(stderr,
              "widen_i32_to_i64 sign-extension failed n=%zu index=%zu lhs=%d out=%lld\n",
              n, index, lhs[index], (long long)out[index]);
      free(lhs);
      free(out);
      return 13;
    }
  }

  for (size_t index = n; index < alloc_n; ++index) {
    if (out[index] != (int64_t)0x5a5a5a5a5a5a5a5aLL) {
      fprintf(stderr,
              "widen_i32_to_i64 touched tail sentinel n=%zu raw_index=%zu got=%lld sentinel=%lld\n",
              n, index, (long long)out[index],
              (long long)(int64_t)0x5a5a5a5a5a5a5a5aLL);
      free(lhs);
      free(out);
      return 14;
    }
  }

  if (n > 1 && (negative_lanes == 0 || positive_lanes == 0)) {
    fprintf(stderr,
            "widen_i32_to_i64 sign-extension coverage missing n=%zu negative_lanes=%zu positive_lanes=%zu\n",
            n, negative_lanes, positive_lanes);
    free(lhs);
    free(out);
    return 15;
  }
  if (n > 0 && wide_magnitude_lanes == 0) {
    fprintf(stderr,
            "widen_i32_to_i64 wide-magnitude coverage missing n=%zu\n", n);
    free(lhs);
    free(out);
    return 16;
  }

  free(lhs);
  free(out);
  printf("widen_i32_to_i64 case n=%zu pattern=%d ok sign_extension_checked wide_magnitude_checked tail_preserved two_input_patterns_checked\n",
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
  printf("tcrv_rvv_generated_bundle_abi_widen_i32_to_i64_ok counts=1,7,16,17,257\n");
  printf("PASS op=widen_i32_to_i64 counts=1,7,16,17,257\n");
  return 0;
}
