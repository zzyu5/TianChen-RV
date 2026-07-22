#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" void
weft_emitc_q4_0_dequant_kernel_scalar_fallback_first_slice(
    int n, float *output, const std::uint8_t *weights);

int main() {
  alignas(2) std::uint8_t weights[36] = {
      0x00, 0x40,
      0x98, 0x7a, 0x98, 0x7a, 0x98, 0x7a, 0x98, 0x7a,
      0x98, 0x7a, 0x98, 0x7a, 0x98, 0x7a, 0x98, 0x7a,
      0x00, 0x38,
      0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
      0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
  };
  std::uint8_t original[36];
  std::memcpy(original, weights, sizeof(weights));

  constexpr float canary = 12345.25f;
  float guarded[66];
  for (float &value : guarded)
    value = -777.0f;
  guarded[0] = canary;
  guarded[65] = canary;

  weft_emitc_q4_0_dequant_kernel_scalar_fallback_first_slice(
      64, guarded + 1, weights);

  if (guarded[0] != canary || guarded[65] != canary) {
    std::fprintf(stderr, "output canary changed\n");
    return 1;
  }
  if (std::memcmp(original, weights, sizeof(weights)) != 0) {
    std::fprintf(stderr, "packed input changed\n");
    return 2;
  }

  const float *output = guarded + 1;
  for (int lane = 0; lane < 16; ++lane) {
    const float expectedLow = lane % 2 == 0 ? 0.0f : 4.0f;
    const float expectedHigh = lane % 2 == 0 ? 2.0f : -2.0f;
    if (output[lane] != expectedLow || output[lane + 16] != expectedHigh) {
      std::fprintf(stderr, "block 0 mismatch at lane %d\n", lane);
      return 3;
    }
    if (output[lane + 32] != -4.0f || output[lane + 48] != 3.5f) {
      std::fprintf(stderr, "block 1 mismatch at lane %d\n", lane);
      return 4;
    }
  }
  return 0;
}
