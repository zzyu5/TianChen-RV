// SCALAR oracle for q1_0 x q8_0 block-dot — structurally INDEPENDENT of any RVV
// vlm_v_b{4,8} sign-mask decode or vwredsum reduce. This is the byte-exact gate
// reference: a misconception in the bit->lane sign decode or the 32-lane reduce in
// the m1 (VLEN256) emit shows up ONLY against an independent scalar decode (ggml's
// own vlm_v_b4 _vl128 ref would SHARE such a bug and falsely agree).
//
// q1_0 super-block = 128 elements = 4 q8_0 sub-blocks of 32. block_q1_0 = {fp16 d;
// uint8 qs[16]} (18B). bit (8*b + i) of qs -> lane (8*b + i): set => +q8, clear => -q8.
// vlm_v_b{ratio} loads element j's mask bit from qs[j>>3] bit (j&7) (LSB-first), so
// the scalar decode uses the SAME LSB-first byte/bit indexing.
//
// fp fold MIRRORS ggml exactly (so the gate is 0.000e+00, not a fp-reassoc 1e-5):
// per sub-block k integer dot over 32 lanes -> red; acc += (float)y.d_k * (float)red;
// after 4 sub-blocks sumf += (float)x.d * acc.
#include <stdint.h>
#include <string.h>

extern "C" void kern_scalar(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int qk = 128;
  const int WB = 18, YB = 34;
  const int nb = (int)n / qk;
  float sumf = 0.0f;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * WB;
    uint16_t xdh; memcpy(&xdh, xb + 0, 2);
    _Float16 xf; memcpy(&xf, &xdh, 2);
    const float xd = (float)xf;
    float acc = 0.0f;
    for (int k = 0; k < 4; ++k) {
      const uint8_t *yb = vy + ((size_t)ib * 4 + k) * YB;
      const uint8_t *qsbits = xb + 2 + 4 * k;          // 4 bytes = 32 sign bits
      const int8_t  *yqs   = (const int8_t *)(yb + 2);  // 32 q8 quants
      uint16_t ydh; memcpy(&ydh, yb + 0, 2);
      _Float16 yf; memcpy(&yf, &ydh, 2);
      int32_t red = 0;
      for (int j = 0; j < 32; ++j) {
        int bit = (qsbits[j >> 3] >> (j & 7)) & 1;     // LSB-first, mirrors vlm_v_b
        int q = (int)yqs[j];
        red += bit ? q : -q;                            // set => +q8, clear => -q8
      }
      acc += (float)yf * (float)red;
    }
    sumf += xd * acc;
  }
  *s = sumf;
}
