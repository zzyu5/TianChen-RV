// DIAGNOSTIC matched-association reference (the tq1_0 pattern). A pure-SCALAR
// reference that reproduces ggml's iq4_xs INTEGER decode EXACTLY but uses OURS's
// per-sub-block DISTRIBUTED fp fold (sumf += d1*(float)sumi per sub-block) instead
// of ggml's single super-block fp fold (sumf += d*(sumi1+sumi2)). If OURS hits
// rel-norm 0.0 against THIS while showing a small nonzero rel-norm vs the verbatim
// ggml_ref.cpp, that isolates the divergence to fp32 reassociation (NOT an integer
// decode bug) -- re-confirming the integer core is bit-exact, exactly as for tq1_0.
//
// kvalues_iq4nl 16-entry codebook; per sub-block s (0..7): the signed 6-bit scale is
// ls = ((scales_l[s/2] >> (4*(s&1))) & 0xf) | (((scales_h >> (2*s)) & 3) << 4) - 32,
// d1 = fp16(x.d)*y.d * (float)ls, sumi = sum over 32 elems of codebook[nibble]*q8.
#include <stdint.h>
#include <string.h>

#define QK_K 256

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

extern "C" void kern_ggml_matched(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int WB = 136, YB = 292;
  const int nb = (int)n / QK_K;
  float sumf = 0.0f;
  for (int ibl = 0; ibl < nb; ++ibl) {
    const uint8_t *xb = vx + (size_t)ibl * WB, *yb = vy + (size_t)ibl * YB;
    const uint8_t *iq4 = xb + 8;                  // qs[128]@8
    const int8_t  *q8  = (const int8_t *)(yb + 4);// q8_K.qs[256]@4
    uint16_t scales_h; memcpy(&scales_h, xb + 2, 2);
    const uint8_t *scales_l = xb + 4;
    float yd; memcpy(&yd, yb + 0, 4);
    uint16_t xdh; memcpy(&xdh, xb + 0, 2); _Float16 xd; memcpy(&xd, &xdh, 2);
    const float d4d8 = (float)xd * yd;
    // 8 sub-blocks of 32 elems. Sub-block s consumes packed bytes [16*s .. 16*s+16):
    // lo nibble -> q8[32*s .. 32*s+16), hi nibble -> q8[32*s+16 .. 32*s+32). This
    // mirrors OURS's two-16-lane (lo,hi) split per 16 packed bytes per sub-block.
    for (int sb = 0; sb < 8; ++sb) {
      int ls = (int)(((scales_l[sb >> 1] >> (4 * (sb & 1))) & 0xf) | (((scales_h >> (2 * sb)) & 3) << 4)) - 32;
      const uint8_t *px = iq4 + 16 * sb;
      const int8_t  *py = q8  + 32 * sb;
      int sumi = 0;
      for (int k = 0; k < 16; ++k) {
        int lo = px[k] & 0xf;
        int hi = (px[k] >> 4) & 0xf;
        sumi += (int)kvalues_iq4nl[lo] * (int)py[k];       // lo -> q8[0..16)
        sumi += (int)kvalues_iq4nl[hi] * (int)py[16 + k];  // hi -> q8[16..32)
      }
      float d1 = d4d8 * (float)ls;
      sumf += d1 * (float)sumi;   // OURS's DISTRIBUTED per-sub-block fp fold
    }
  }
  *s = sumf;
}
