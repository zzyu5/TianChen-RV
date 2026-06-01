#ifndef TCRV_CLASSROOM_LLAMA_Q8_0_Q8_0_H
#define TCRV_CLASSROOM_LLAMA_Q8_0_Q8_0_H

#include <stddef.h>
#include <stdint.h>

#define TCRV_CLASSROOM_QK8_0 32

typedef struct {
  float d;
  int8_t qs[TCRV_CLASSROOM_QK8_0];
} tcrv_classroom_block_q8_0;

#ifdef __cplusplus
extern "C" {
#endif

void tcrv_llama_q8_0_q8_0_reference_rvv(
    const tcrv_classroom_block_q8_0 *x,
    const tcrv_classroom_block_q8_0 *y,
    float *out,
    size_t n);

#ifdef __cplusplus
}
#endif

#endif

