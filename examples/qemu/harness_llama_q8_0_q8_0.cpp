#include "llama_q8_0_q8_0.h"

#include <cmath>
#include <cstdio>

namespace {

constexpr size_t kBlocks = 5;
constexpr size_t kN = kBlocks * TCRV_CLASSROOM_QK8_0;

int8_t make_x(size_t block, size_t lane) {
  const int value = (int)((block * 17 + lane * 5 + 11) % 63) - 31;
  return (int8_t)value;
}

int8_t make_y(size_t block, size_t lane) {
  const int value = (int)((block * 13 + lane * 7 + 3) % 59) - 29;
  return (int8_t)value;
}

float make_x_scale(size_t block) {
  return 0.03125f + 0.00390625f * (float)((block % 5) + 1);
}

float make_y_scale(size_t block) {
  return 0.046875f + 0.001953125f * (float)((block % 7) + 1);
}

float scalar_reference(const tcrv_classroom_block_q8_0 *x,
                       const tcrv_classroom_block_q8_0 *y,
                       size_t n) {
  const size_t nb = n / TCRV_CLASSROOM_QK8_0;
  float sumf = 0.0f;
  for (size_t ib = 0; ib < nb; ++ib) {
    int32_t block_sum = 0;
    for (size_t lane = 0; lane < TCRV_CLASSROOM_QK8_0; ++lane) {
      block_sum += (int32_t)x[ib].qs[lane] * (int32_t)y[ib].qs[lane];
    }
    sumf += (float)block_sum * x[ib].d * y[ib].d;
  }
  return sumf;
}

void fill_blocks(tcrv_classroom_block_q8_0 *x,
                 tcrv_classroom_block_q8_0 *y) {
  for (size_t ib = 0; ib < kBlocks; ++ib) {
    x[ib].d = make_x_scale(ib);
    y[ib].d = make_y_scale(ib);
    for (size_t lane = 0; lane < TCRV_CLASSROOM_QK8_0; ++lane) {
      x[ib].qs[lane] = make_x(ib, lane);
      y[ib].qs[lane] = make_y(ib, lane);
    }
  }
}

} // namespace

int main() {
  tcrv_classroom_block_q8_0 x[kBlocks];
  tcrv_classroom_block_q8_0 y[kBlocks];
  fill_blocks(x, y);

  float got = -9999.0f;
  tcrv_llama_q8_0_q8_0_reference_rvv(x, y, &got, kN);

  const float expected = scalar_reference(x, y, kN);
  const float diff = std::fabs(got - expected);
  const float tolerance = 1.0e-4f;

  std::printf("llama q8_0_q8_0 classroom baseline\n");
  std::printf("n=%zu blocks=%zu qk=%d\n", kN, kBlocks,
              TCRV_CLASSROOM_QK8_0);
  for (size_t ib = 0; ib < kBlocks && ib < 3; ++ib) {
    int32_t block_sum = 0;
    for (size_t lane = 0; lane < TCRV_CLASSROOM_QK8_0; ++lane) {
      block_sum += (int32_t)x[ib].qs[lane] * (int32_t)y[ib].qs[lane];
    }
    std::printf("block[%zu]: dot_i32=%d x.d=%0.8f y.d=%0.8f\n",
                ib, block_sum, x[ib].d, y[ib].d);
  }
  std::printf("got=%0.8f expected=%0.8f diff=%0.8f\n",
              got, expected, diff);

  if (diff > tolerance) {
    std::fprintf(stderr,
                 "mismatch: got=%0.8f expected=%0.8f tolerance=%0.8f\n",
                 got, expected, tolerance);
    return 1;
  }

  std::printf("rvv classroom llama q8_0_q8_0 proof ok\n");
  return 0;
}

