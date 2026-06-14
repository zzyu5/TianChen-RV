
#include <stdint.h>
#include <stddef.h>
static int32_t sx_i4(uint8_t nib) {
  int32_t v = (int32_t)(nib & 0x0fu);
  return v >= 8 ? v - 16 : v;
}
/* genuine scalar packed-i4: two signed i4 per byte (low, high nibble). */
__attribute__((noinline)) void
ref_scalar_packed_i4(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                     float scale, float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i) {
    uint8_t lb = (uint8_t)lhs[i], rb = (uint8_t)rhs[i];
    sum += sx_i4(lb & 0x0fu) * sx_i4(rb & 0x0fu);
    sum += sx_i4((lb >> 4) & 0x0fu) * sx_i4((rb >> 4) & 0x0fu);
  }
  out[0] = ((float)sum) * scale;
}
