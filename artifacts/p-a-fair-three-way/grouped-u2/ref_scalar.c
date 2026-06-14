
#include <stdint.h>
#include <stddef.h>
/* genuine scalar: int8*int8 -> i32 reduce, then * f32 scale. rv64gc, no vector ISA. */
__attribute__((noinline)) void
ref_scalar_byte(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                float scale, float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (int32_t)lhs[i] * (int32_t)rhs[i];
  out[0] = ((float)sum) * scale;
}
