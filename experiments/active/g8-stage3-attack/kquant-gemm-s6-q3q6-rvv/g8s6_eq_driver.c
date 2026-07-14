/* G8 S6 strip-outer vs PLAIN byte-exact equality driver (q6_K / q3_K GEMM).
 * Fills the repacked-weight + q8_Kx4 activation buffers with deterministic data
 * (valid fp16/fp32 scales so no NaN, random quant bytes), runs BOTH the S6 and
 * PLAIN emitted GEMM kernels on the SAME bytes, and memcmp's the fp32 output.
 * Bit-identical output over all shapes == the G1 numeric byte-exact gate
 * (S6-tiled == PLAIN, DYNAMIC per-bit). Both kernels are the ACTUAL exported
 * .kernel.c (symbols renamed _S6 / _PLAIN by sed). */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern void kern_S6(size_t n, float* s, const uint8_t* vx, const uint8_t* vy,
                    size_t nr, size_t nc, size_t bs);
extern void kern_PLAIN(size_t n, float* s, const uint8_t* vx, const uint8_t* vy,
                       size_t nr, size_t nc, size_t bs);

/* fp16 encode of a small value in [-1,1) so the fold never NaNs */
static uint16_t f2h(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t man = x & 0x7FFFFFu;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
  return (uint16_t)(sign | (exp << 10) | (man >> 13));
}

static uint32_t rng = 0x12345678u;
static uint32_t nextr(void){ rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }

int main(int argc, char** argv) {
  /* Shapes: K (mult 256), nr (mult 4), nc (mult 16). Weight block_q?_Kx16 stride
   * WSTRIDE; activation block_q8_Kx4 stride 1168. Passed as env-independent consts. */
  size_t WSTRIDE = argc > 1 ? (size_t)atol(argv[1]) : 3360; /* q6_K=3360, q3_K=3504 */
  int Ks[]   = {256, 512, 2048};
  int nrs[]  = {4, 8, 64};
  int ncs[]  = {16, 32, 512};
  size_t ASTRIDE = 1168;
  int fails = 0, shapes = 0;
  for (int ki=0; ki<3; ++ki) for (int ri=0; ri<3; ++ri) for (int ci=0; ci<3; ++ci) {
    int K = Ks[ki], nr = nrs[ri], nc = ncs[ci];
    int nb = K/256;
    size_t nrg = nr/4, ncg = nc/16;
    /* weight buffer: ncg col-groups, each nb blocks of WSTRIDE bytes */
    size_t wbytes = ncg * nb * WSTRIDE;
    size_t abytes = nrg * nb * ASTRIDE;
    uint8_t* vx = (uint8_t*)malloc(wbytes);
    uint8_t* vy = (uint8_t*)malloc(abytes);
    for (size_t i=0;i<wbytes;++i) vx[i] = (uint8_t)(nextr() & 0xFF);
    for (size_t i=0;i<abytes;++i) vy[i] = (uint8_t)(nextr() & 0xFF);
    /* overwrite the fp16 weight d[16] @0 (32 bytes) per weight block with small vals */
    for (size_t cg=0; cg<ncg; ++cg) for (int b=0;b<nb;++b) {
      uint8_t* blk = vx + (cg*nb + b)*WSTRIDE;
      for (int j=0;j<16;++j){ float v = ((int)(nextr()&0xFF)-128)/512.0f; uint16_t h=f2h(v); memcpy(blk + j*2, &h, 2); }
    }
    /* overwrite the fp32 activation d[4] @0 per activation block with small vals */
    for (size_t rg=0; rg<nrg; ++rg) for (int b=0;b<nb;++b) {
      uint8_t* blk = vy + (rg*nb + b)*ASTRIDE;
      for (int c=0;c<4;++c){ float v = ((int)(nextr()&0xFF)-128)/512.0f; memcpy(blk + c*4, &v, 4); }
    }
    size_t bs = nc;
    float* s6  = (float*)calloc((size_t)nr*bs, 4);
    float* spl = (float*)calloc((size_t)nr*bs, 4);
    kern_S6((size_t)K, s6,  vx, vy, (size_t)nr, (size_t)nc, bs);
    kern_PLAIN((size_t)K, spl, vx, vy, (size_t)nr, (size_t)nc, bs);
    int mism = memcmp(s6, spl, (size_t)nr*bs*4) ? 1 : 0;
    /* count mismatching floats for detail */
    int nmis = 0; for (size_t i=0;i<(size_t)nr*bs;++i) if (memcmp(&s6[i],&spl[i],4)) ++nmis;
    printf("K=%d nr=%d nc=%d : %s (mism_floats=%d / %d)\n",
           K, nr, nc, mism? "MISMATCH":"byte-exact", nmis, nr*nc);
    if (mism) ++fails;
    ++shapes;
    free(vx); free(vy); free(s6); free(spl);
  }
  printf("=== %d/%d shapes byte-exact; fails=%d ===\n", shapes-fails, shapes, fails);
  return fails ? 1 : 0;
}
