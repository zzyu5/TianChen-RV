#include <stdint.h>
#include <stddef.h>
/* genuine scalar oracle: int8*int8 -> i32 reduce + acc[0], then * f32 scale.
   Compiled rv64gc (no vector ISA) -- the _generic reference for byte-exact. */
__attribute__((noinline)) void
ref_scalar_byte(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                float scale, float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (int32_t)lhs[i] * (int32_t)rhs[i];
  out[0] = ((float)sum) * scale;
}
