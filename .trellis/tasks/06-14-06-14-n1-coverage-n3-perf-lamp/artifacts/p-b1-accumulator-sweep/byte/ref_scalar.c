
#include <stdint.h>
#include <stddef.h>
/* genuine scalar int8 dot. rv64gc -> NO vector ISA (objdump-verified on board). */
__attribute__((noinline)) void
ref_scalar_byte(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                float scale, float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (int32_t)lhs[i] * (int32_t)rhs[i];
  out[0] = ((float)sum) * scale;
}
