// INC-40 nvfp4 COMPILER-KNOB control (ssh rvv). Two compiler-emitted kernels that
// differ ONLY in the MLIR `codebook=` attr -- the real FP4 e2m1 kvalues_mxfp4 vs a
// LINEAR nibble-8 table -- produce DIFFERENT hardware results on the SAME input bytes,
// each matching its OWN table's math. Proves the codebook DenseI8ArrayAttr is the live,
// load-bearing knob through compile -> hardware (not a dead/ignored attr).
//
// Both kernel TUs are emitted by the SAME compiler from the SAME input.mlir modulo the
// one codebook= attr; their emitted bodies are byte-identical except the table decl.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

// Two emitter outputs. They define the same symbol name, so each is wrapped in its own
// renamed TU via the macro trick: compile each separately, link both.
extern "C" void
tcrv_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_ggml_vec_dot_nvfp4_q8_0(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);  // the REAL FP4 kernel
extern "C" void nvfp4_linear_kernel(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);  // the LINEAR-table kernel

static const int QK = 64, QK_SUB = 16, NVFP4_STRIDE = 36, Q8_STRIDE = 34, N_SUB = 4;

static const int8_t kvalues_mxfp4[16] = {0, 1, 2,  3,  4,  6,  8,  12,
                                         0, -1, -2, -3, -4, -6, -8, -12};
static const int8_t kvalues_linear[16] = {-8, -7, -6, -5, -4, -3, -2, -1,
                                          0,  1,  2,  3,  4,  5,  6,  7};

static inline float ggml_ue4m3_to_fp32(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.0f;
  int exp = (x >> 3) & 0xF, man = x & 0x7;
  float raw = (exp == 0) ? ldexpf((float)man, -9)
                         : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
  return raw * 0.5f;
}
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf; std::memcpy(&hf, &h, 2); return (float)hf;
}

static void ref(int n, float *s, const uint8_t *x, const uint8_t *y, const int8_t *t) {
  const int nb = n / QK; float sumf = 0;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = x + (size_t)ib * NVFP4_STRIDE;
    for (int s_idx = 0; s_idx < N_SUB; ++s_idx) {
      const float d = ggml_ue4m3_to_fp32(xb[s_idx]);
      const int q8_block = s_idx / 2, q8_off = (s_idx % 2) * QK_SUB;
      const uint8_t *yb = y + (size_t)(2 * ib + q8_block) * Q8_STRIDE;
      uint16_t ydh; std::memcpy(&ydh, yb, 2);
      const float dy = fp16_to_fp32(ydh);
      const uint8_t *xqs = xb + 4 + s_idx * (QK_SUB / 2);
      const int8_t *yqs = (const int8_t *)(yb + 2);
      int lo = 0, hi = 0;
      for (int j = 0; j < QK_SUB / 2; ++j) {
        lo += yqs[q8_off + j] * t[xqs[j] & 0xf];
        hi += yqs[q8_off + j + QK_SUB / 2] * t[xqs[j] >> 4];
      }
      sumf += dy * d * (lo + hi);
    }
  }
  *s = sumf;
}

static uint32_t rng = 0x12345678u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

int main() {
  int n = 1024, nb = n / QK, q8nb = n / 32;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * NVFP4_STRIDE);
  uint8_t *vy = (uint8_t *)malloc((size_t)q8nb * Q8_STRIDE);
  for (int i = 0; i < nb; i++) {
    for (int s = 0; s < N_SUB; s++) vx[(size_t)i * NVFP4_STRIDE + s] = 0x38; // ~1.0
    for (int j = 0; j < QK / 2; j++) vx[(size_t)i * NVFP4_STRIDE + 4 + j] = (uint8_t)(xr() % 256);
  }
  for (int i = 0; i < q8nb; i++) {
    _Float16 d = (_Float16)1.0f; uint16_t dh; std::memcpy(&dh, &d, 2);
    std::memcpy(vy + (size_t)i * Q8_STRIDE, &dh, 2);
    for (int j = 0; j < 32; j++) vy[(size_t)i * Q8_STRIDE + 2 + j] = 100; // +100 (non-zero sum)
  }

  float real_hw = 0, lin_hw = 0, real_ref = 0, lin_ref = 0;
  tcrv_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_ggml_vec_dot_nvfp4_q8_0((size_t)n, &real_hw, vx, vy);
  nvfp4_linear_kernel((size_t)n, &lin_hw, vx, vy);
  ref(n, &real_ref, vx, vy, kvalues_mxfp4);
  ref(n, &lin_ref, vx, vy, kvalues_linear);

  printf("real-codebook kernel HW=%.6g (ref=%.6g)\n", real_hw, real_ref);
  printf("linear-codebook kernel HW=%.6g (ref=%.6g)\n", lin_hw, lin_ref);
  int ok = 1;
  if (std::memcmp(&real_hw, &real_ref, 4) != 0) { printf("FAIL: real kernel != real ref\n"); ok = 0; }
  if (std::memcmp(&lin_hw, &lin_ref, 4) != 0)   { printf("FAIL: linear kernel != linear ref\n"); ok = 0; }
  if (std::memcmp(&real_hw, &lin_hw, 4) == 0)   { printf("FAIL: the two codebooks gave the SAME HW result (attr not load-bearing!)\n"); ok = 0; }
  if (ok) printf("KNOB OK: the codebook attr flips the HW result; each matches its own table -> the codebook is the live load-bearing knob through compile->hardware\n");
  free(vx); free(vy);
  return ok ? 0 : 1;
}
