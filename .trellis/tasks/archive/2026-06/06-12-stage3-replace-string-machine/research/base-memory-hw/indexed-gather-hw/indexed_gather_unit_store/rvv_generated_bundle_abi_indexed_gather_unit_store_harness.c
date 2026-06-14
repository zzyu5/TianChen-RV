#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h"

static uint32_t make_index(size_t logical_index, size_t n, int pattern) {
  if (n == 0)
    return 0;
  size_t mixed;
  if (pattern == 0) {
    mixed = (logical_index * 5 + 3) % n;
    if ((logical_index % 2) == 0)
      mixed = (n - 1) - mixed;
  } else {
    mixed = (logical_index * 7 + 1 + ((logical_index % 3) * 2)) % n;
    if ((logical_index % 4) == 1)
      mixed = (n - 1) - mixed;
  }
  return (uint32_t)mixed;
}

static int32_t init_data_value(size_t index, int pattern) {
  if (pattern == 0)
    return (int32_t)(101 + (int32_t)(index * 9));
  return (int32_t)(((index % 2) == 0)
                       ? (int32_t)(-700 - (int32_t)(index * 11))
                       : (int32_t)(900 + (int32_t)(index * 13)));
}

static int run_case(size_t n, int pattern) {
  /* expected: data[indices[index]] */
  size_t alloc_n = n == 0 ? 1 : n;
  size_t out_alloc_n = alloc_n + 8;
  int32_t *data = (int32_t *)malloc(sizeof(int32_t) * alloc_n);
  uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * alloc_n);
  int32_t *out = (int32_t *)malloc(sizeof(int32_t) * out_alloc_n);
  if (!data || !indices || !out) {
    fprintf(stderr, "allocation failed for n=%zu\n", n);
    free(data);
    free(indices);
    free(out);
    return 11;
  }

  for (size_t index = 0; index < alloc_n; ++index) {
    data[index] = init_data_value(index, pattern);
    indices[index] = make_index(index, alloc_n, pattern);
  }
  for (size_t index = 0; index < out_alloc_n; ++index) {
    out[index] = (int32_t)0x5a5a5a5a;
  }

  tcrv_emitc_explicit_selected_body_indexed_gather_unit_store_kernel_explicit_selected_body_rvv_indexed_gather_unit_store(data, indices, out, n);

  size_t output_order_distinguishing_lanes = 0;
  for (size_t index = 0; index < n; ++index) {
    if ((size_t)indices[index] >= n) {
      fprintf(stderr,
              "indexed_gather_unit_store generated out-of-range index n=%zu pattern=%d index=%zu gather_index=%u\n",
              n, pattern, index, indices[index]);
      free(data);
      free(indices);
      free(out);
      return 12;
    }
    int32_t expected = data[indices[index]];
    if ((size_t)indices[index] != index && data[indices[index]] != data[index])
      ++output_order_distinguishing_lanes;
    if (out[index] != expected) {
      fprintf(stderr,
              "indexed_gather_unit_store mismatch n=%zu pattern=%d index=%zu gather_index=%u got=%d expected=%d data_at_index=%d\n",
              n, pattern, index, indices[index], out[index], expected,
              data[indices[index]]);
      free(data);
      free(indices);
      free(out);
      return 13;
    }
  }

  for (size_t index = n; index < out_alloc_n; ++index) {
    if (out[index] != (int32_t)0x5a5a5a5a) {
      fprintf(stderr,
              "indexed_gather_unit_store touched tail sentinel n=%zu pattern=%d raw_index=%zu got=%d sentinel=%d\n",
              n, pattern, index, out[index], (int32_t)0x5a5a5a5a);
      free(data);
      free(indices);
      free(out);
      return 14;
    }
  }

  if (n > 3 && output_order_distinguishing_lanes == 0) {
    fprintf(stderr,
            "indexed_gather_unit_store vacuous indexed gather check failed n=%zu pattern=%d indices=[%u,%u,%u]\n",
            n, pattern, indices[0], indices[1], indices[2]);
    free(data);
    free(indices);
    free(out);
    return 15;
  }

  free(data);
  free(indices);
  free(out);
  printf("indexed_gather_unit_store case n=%zu pattern=%d ok non_monotonic_indexed_gather element_offset_indices output_order_distinguishing_lanes=%zu unit_store_output tail_preserved runtime_n_avl_honored\n",
         n, pattern, output_order_distinguishing_lanes);
  return 0;
}

int main(void) {
  const size_t counts[] = {0, 1, 16, 17, 257};
  const size_t count_count = sizeof(counts) / sizeof(counts[0]);
  const int index_patterns[] = {0, 1};
  const size_t index_pattern_count =
      sizeof(index_patterns) / sizeof(index_patterns[0]);
  for (size_t index = 0; index < count_count; ++index) {
    for (size_t pattern_index = 0; pattern_index < index_pattern_count;
         ++pattern_index) {
      int status = run_case(counts[index], index_patterns[pattern_index]);
      if (status != 0)
        return status;
    }
  }
  printf("tcrv_rvv_generated_bundle_abi_indexed_gather_unit_store_ok counts=0,1,16,17,257 index_patterns=2 element_offset_indices unit_store_output tail_preserved runtime_n_avl_honored\n");
  printf("PASS op=indexed_gather_unit_store counts=0,1,16,17,257 index_patterns=2 element_offset_indices unit_store_output tail_preserved runtime_n_avl_honored\n");
  return 0;
}
