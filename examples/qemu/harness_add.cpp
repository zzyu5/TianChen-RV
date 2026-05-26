#include <cstddef>
#include <cstdint>
#include <cstdio>

extern "C" void tcrv_emitc_rvv_i32_add_kernel_rvv_i32_add(
    const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

int main() {
  constexpr size_t n = 37;
  int32_t lhs[n];
  int32_t rhs[n];
  int32_t out[n];

  for (size_t i = 0; i < n; ++i) {
    lhs[i] = static_cast<int32_t>(i * 3 - 17);
    rhs[i] = static_cast<int32_t>(41 - static_cast<int32_t>(i * 2));
    out[i] = -777777;
  }

  tcrv_emitc_rvv_i32_add_kernel_rvv_i32_add(lhs, rhs, out, n);

  for (size_t i = 0; i < n; ++i) {
    int32_t expected = lhs[i] + rhs[i];
    if (out[i] != expected) {
      std::printf("mismatch at %zu: got %d expected %d\n", i, out[i],
                  expected);
      return 1;
    }
  }

  std::printf("rvv classroom proof ok: %zu lanes checked\n", n);
  return 0;
}
