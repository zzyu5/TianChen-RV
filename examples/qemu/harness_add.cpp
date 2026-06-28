// Worked-example harness for the elementwise-add slice.
// Pairs with the compiler-generated kernel (add_generated.cpp). The entry name
// is the compiler-derived `tcrv_emitc_<kernelSym>_<variantSym>`; the harness
// declares exactly that symbol, supplies inputs, calls the kernel, and checks
// every lane against a scalar oracle. Real-hardware proof target: SpacemiT X60 (ssh k1).
#include <cstddef>
#include <cstdint>
#include <cstdio>

// === compiler-generated kernel entry (must match add_generated.cpp) ===
extern "C" void tcrv_emitc_explicit_selected_body_add_kernel_explicit_selected_body_rvv_i32_add(
    const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);

static int32_t make_lhs(size_t i) { return static_cast<int32_t>(i * 3 - 17); }
static int32_t make_rhs(size_t i) { return static_cast<int32_t>(41 - static_cast<int32_t>(i * 2)); }
static int32_t scalar_oracle(int32_t a, int32_t b) { return a + b; }

int main() {
  const size_t n = 1031;  // deliberately not a multiple of VLEN, to exercise the tail
  static int32_t lhs[n], rhs[n], out[n];
  for (size_t i = 0; i < n; ++i) { lhs[i] = make_lhs(i); rhs[i] = make_rhs(i); out[i] = 0; }

  tcrv_emitc_explicit_selected_body_add_kernel_explicit_selected_body_rvv_i32_add(lhs, rhs, out, n);

  for (size_t i = 0; i < n; ++i) {
    int32_t expected = scalar_oracle(lhs[i], rhs[i]);
    if (out[i] != expected) {
      std::printf("mismatch at %zu: got %d expected %d\n", i, out[i], expected);
      return 1;
    }
  }
  std::printf("rvv classroom add slice proof ok: %zu lanes checked\n", n);
  return 0;
}
