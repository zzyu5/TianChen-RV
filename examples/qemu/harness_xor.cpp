#include <cstddef>
#include <cstdint>
#include <cstdio>

// generated.cpp should provide this exported kernel. This harness is paired
// with test/Target/RVV/emitc-to-cpp-xor.mlir.
extern "C" void tcrv_emitc_rvv_i32_xor_kernel_rvv_i32_xor(
    const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

#ifndef TCRV_N
#define TCRV_N 41
#endif

static int32_t make_lhs(size_t i) {
  return static_cast<int32_t>((i * 17) ^ 0x13572468);
}

static int32_t make_rhs(size_t i) {
  return static_cast<int32_t>((i * 5 + 11) ^ 0x24681357);
}

static int32_t scalar_oracle(int32_t lhs, int32_t rhs) { return lhs ^ rhs; }

int main() {
  constexpr size_t n = TCRV_N;
  int32_t lhs[n];
  int32_t rhs[n];
  int32_t out[n];

  for (size_t i = 0; i < n; ++i) {
    lhs[i] = make_lhs(i);
    rhs[i] = make_rhs(i);
    out[i] = -777777;
  }

  tcrv_emitc_rvv_i32_xor_kernel_rvv_i32_xor(lhs, rhs, out, n);

  for (size_t i = 0; i < n; ++i) {
    int32_t expected = scalar_oracle(lhs[i], rhs[i]);
    if (out[i] != expected) {
      std::printf("mismatch at %zu: got %d expected %d\n", i, out[i],
                  expected);
      return 1;
    }
  }

  const size_t preview = n < 8 ? n : 8;
  for (size_t i = 0; i < preview; ++i) {
    std::printf("lane[%zu]: %d xor %d = %d\n", i, lhs[i], rhs[i], out[i]);
  }
  std::printf("rvv classroom xor proof ok: %zu lanes checked\n", n);
  return 0;
}

