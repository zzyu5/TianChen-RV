// Hardware self-check harness for the Stage-3 conversion-emitted UNSIGNED
// widening product + widening product-reduce-add kernels. Runs on real ssh rvv
// (riscv64 + clang -march=rv64gcv). Compares the RVV-intrinsic kernel output
// against a plain-C unsigned reference. PASS iff byte-exact.
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>

extern "C" void
tcrv_emitc_explicit_selected_body_unsigned_widening_product_kernel_explicit_selected_body_rvv_unsigned_widening_product(
    const uint8_t *v1, const uint8_t *v2, uint16_t *v3, size_t v4);

extern "C" void
tcrv_emitc_explicit_selected_body_unsigned_product_reduce_kernel_explicit_selected_body_rvv_unsigned_product_reduce(
    const uint8_t *v1, const uint8_t *v2, const uint32_t *v3, uint32_t *v4,
    size_t v5);

int main() {
  const size_t n = 257; // odd, > one VL chunk, to exercise the tail
  std::vector<uint8_t> a(n), b(n);
  for (size_t i = 0; i < n; ++i) {
    a[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);   // up to 255
    b[i] = static_cast<uint8_t>((i * 13 + 11) & 0xFF); // up to 255
  }

  // --- unsigned widening product (elementwise u8 x u8 -> u16) ---
  std::vector<uint16_t> prod(n, 0), prod_ref(n, 0);
  tcrv_emitc_explicit_selected_body_unsigned_widening_product_kernel_explicit_selected_body_rvv_unsigned_widening_product(
      a.data(), b.data(), prod.data(), n);
  for (size_t i = 0; i < n; ++i)
    prod_ref[i] = static_cast<uint16_t>(static_cast<uint16_t>(a[i]) *
                                        static_cast<uint16_t>(b[i]));
  int prod_fail = 0;
  for (size_t i = 0; i < n; ++i)
    if (prod[i] != prod_ref[i]) {
      if (prod_fail < 5)
        printf("PROD MISMATCH i=%zu got=%u ref=%u\n", i, prod[i], prod_ref[i]);
      ++prod_fail;
    }

  // --- unsigned widening product-reduce-add (sum of u8*u8 into u32, seeded) ---
  const uint32_t seed = 1000u;
  uint32_t acc = seed;
  uint32_t out = 0;
  tcrv_emitc_explicit_selected_body_unsigned_product_reduce_kernel_explicit_selected_body_rvv_unsigned_product_reduce(
      a.data(), b.data(), &acc, &out, n);
  uint32_t ref = seed;
  for (size_t i = 0; i < n; ++i)
    ref += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
  int reduce_fail = (out != ref);
  if (reduce_fail)
    printf("REDUCE MISMATCH got=%u ref=%u\n", out, ref);

  if (prod_fail == 0 && reduce_fail == 0) {
    printf("PASS unsigned_widening_product (n=%zu)\n", n);
    printf("PASS unsigned_widening_product_reduce_add out=%u ref=%u\n", out, ref);
    printf("ALL PASS\n");
    return 0;
  }
  printf("FAIL prod_mismatches=%d reduce_fail=%d\n", prod_fail, reduce_fail);
  return 1;
}
