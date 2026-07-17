#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static int run_case(size_t n) {
  /* expected: rhs[chunk_start] + sum(lhs[chunk_start:chunk_start+vl]) */
  size_t alloc_n = n == 0 ? 1 : n;
  int32_t *lhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *rhs = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  if (!lhs || !rhs || !out) {
    fprintf(stderr, "allocation failed for n=%zu\n", n);
    free(lhs);
    free(rhs);
    free(out);
    return 11;
  }

  for (size_t index = 0; index < n; ++index) {
    lhs[index] = (int32_t)(((index % 4) < 2) ? -((int32_t)(index % 29) + 1) : ((int32_t)(index % 31) + 3));
    rhs[index] = (int32_t)(((index % 3) == 0) ? -((int32_t)(1000 + index)) : (int32_t)(1000 + (int32_t)(index * 3)));
    out[index] = (int32_t)0x5a5a5a5a;
  }

  tcrv_emitc_explicit_selected_body_reduce_add_kernel_explicit_selected_body_rvv_reduce_add(lhs, rhs, out, n);

  if (n > 0) {
    size_t full_chunk_vl = __riscv_vsetvl_e32m1(n);
    if (full_chunk_vl == 0) {
      fprintf(stderr, "reduce_add invalid full_chunk_vl=0 for n=%zu\n", n);
      free(lhs);
      free(rhs);
      free(out);
      return 13;
    }

    for (size_t chunk_start = 0; chunk_start < n; chunk_start += full_chunk_vl) {
      size_t vl = __riscv_vsetvl_e32m1(n - chunk_start);
      int32_t expected = rhs[chunk_start];
      for (size_t lane = 0; lane < vl; ++lane)
        expected = (int32_t)(expected + lhs[chunk_start + lane]);

      if (out[chunk_start] != expected) {
        fprintf(stderr,
                "reduce_add mismatch n=%zu chunk_start=%zu got=%d expected=%d rhs_seed=%d vl=%zu\n",
                n, chunk_start, out[chunk_start], expected, rhs[chunk_start], vl);
        free(lhs);
        free(rhs);
        free(out);
        return 12;
      }

      for (size_t lane = 1; lane < vl; ++lane) {
        size_t index = chunk_start + lane;
        if (out[index] != (int32_t)0x5a5a5a5a) {
          fprintf(stderr,
                  "reduce_add touched non-result lane n=%zu index=%zu got=%d sentinel=%d\n",
                  n, index, out[index], (int32_t)0x5a5a5a5a);
          free(lhs);
          free(rhs);
          free(out);
          return 14;
        }
      }
    }
  }

  free(lhs);
  free(rhs);
  free(out);
  printf("reduce_add case n=%zu ok\n", n);
  return 0;
}

int main(void) {
  const size_t counts[] = {0, 1, 16, 17, 257};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  for (size_t index = 0; index < count_count; ++index) {
    int status = run_case(counts[index]);
    if (status != 0)
      return status;
  }
  printf("tcrv_rvv_generated_bundle_abi_reduce_add_ok counts=0,1,16,17,257\n");
  printf("PASS op=reduce_add counts=0,1,16,17,257\n");
  return 0;
}
