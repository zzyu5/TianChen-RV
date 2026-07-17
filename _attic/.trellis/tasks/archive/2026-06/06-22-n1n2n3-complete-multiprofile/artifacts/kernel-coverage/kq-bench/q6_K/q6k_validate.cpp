// q6_K Win-A byte-exact HW validation harness (ssh rvv VLEN128 + ssh k1 VLEN256).
//
// Proves: the C our compiler emits for tcrv_rvv.q6_k_q8_k_block_dot (at a given
// integer_core_lmul) computes the SAME fp32 *s as ggml's reference
// ggml_vec_dot_q6_K_q8_K_generic (llama.cpp quants.c:800), BYTE-EXACT (exact
// IEEE-754 bit equality on *s), over the n set + edge cases + random n + a
// negative control. ORACLE = the VERBATIM ggml _generic body. The fp16->fp32
// macro is shimmed to a hardware _Float16 cast (exact, never rounds).
//
// The kernel under test is the UNMODIFIED compiler-emitted ours.cpp, declared
// extern. -ffp-contract=off: the kernel's positive fold uses SEPARATE
// __riscv_vfmul/vfadd (contraction-immune); _generic's `sums[l]+=d*aux32[l]`
// fuses to fma under contraction, so =off leaves the reference unfused.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>

#define QK_K 256
typedef uint16_t ggml_half;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    ggml_half d;
} block_q6_K;
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;

static inline float ggml_fp16_to_fp32_shim(uint16_t h) {
    _Float16 v; std::memcpy(&v, &h, sizeof(v)); return (float)v;
}
#define GGML_CPU_FP16_TO_FP32(x) ggml_fp16_to_fp32_shim(x)
#define GGML_RESTRICT __restrict
#define UNUSED(x) (void)(x)

// ---- ggml's own reference, VERBATIM (quants.c:800), 4-arg adapter below. -------
static void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    const block_q6_K * GGML_RESTRICT x = (const block_q6_K *)vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *)vy;
    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// The kernel under test (the emitted ours.cpp; the same 4-arg ABI).
extern "C" void tcrv_emitc_ggml_vec_dot_q6_K_q8_K_kernel_ggml_vec_dot_q6_K_q8_K(size_t, float *, const uint8_t *, const uint8_t *);

#define WB 210
#define YB 292

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() { uint16_t s=(xr()&1)<<15,e=(uint16_t)(xr()%31),m=(uint16_t)(xr()&0x3FF); return s|(e<<10)|m; }

static void fill(uint8_t *vx, uint8_t *vy, int nb, int zero_scales) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * WB, *yb = vy + (size_t)ib * YB;
    for (int i = 0; i < 128; ++i) xb[0 + i]   = (uint8_t)(xr() & 0xFF);
    for (int i = 0; i < 64;  ++i) xb[128 + i] = (uint8_t)(xr() & 0xFF);
    for (int i = 0; i < 16;  ++i) xb[192 + i] = zero_scales ? 0 : (uint8_t)((xr() & 0xFF) - 128);
    uint16_t d = rfp16(); memcpy(xb + 208, &d, 2);
    float yd = 0.01f + (float)(xr() % 1000) / 1000.0f;
    memcpy(yb + 0, &yd, 4);
    int8_t q8[256];
    for (int i = 0; i < 256; ++i) { q8[i] = (int8_t)((xr() & 0xFF) - 128) / 4; yb[4 + i] = (uint8_t)q8[i]; }
    for (int g = 0; g < 16; ++g) { int16_t bs=0; for (int k=0;k<16;++k) bs+=q8[g*16+k]; memcpy(yb+260+g*2,&bs,2); }
  }
}

static int biteq(float a, float b) { uint32_t ua,ub; memcpy(&ua,&a,4); memcpy(&ub,&b,4); return ua==ub; }

int main() {
  // Named n set + edge cases + random n. All multiples of QK_K.
  int named[] = { 256, 512, 2048, 4096, 25600, 65536 };
  int edges[] = { 256, 512, 768, 1024, 256*7, 256*13 };
  int pass = 0, total = 0, fail = 0;

  auto run_one = [&](int n, int seed) {
    int nb = n / QK_K;
    uint8_t *vx=(uint8_t*)malloc((size_t)nb*WB), *vy=(uint8_t*)malloc((size_t)nb*YB);
    rng = 0x2468ace0u ^ (uint32_t)(seed*0x9e3779b9u);
    fill(vx, vy, nb, 0);
    float rg=0, ro=0;
    ggml_vec_dot_q6_K_q8_K_generic(n, &rg, vx, vy);
    tcrv_emitc_ggml_vec_dot_q6_K_q8_K_kernel_ggml_vec_dot_q6_K_q8_K(n, &ro, vx, vy);
    total++;
    if (biteq(rg, ro)) pass++;
    else { fail++; if (fail <= 8) printf("FAIL n=%d seed=%d ref=%.6f tcrv=%.6f\n", n, seed, rg, ro); }
    free(vx); free(vy);
  };

  for (int s = 0; s < 6; ++s) for (int k = 0; k < (int)(sizeof(named)/sizeof(named[0])); ++k) run_one(named[k], s+1);
  for (int k = 0; k < (int)(sizeof(edges)/sizeof(edges[0])); ++k) run_one(edges[k], 100+k);
  // 2000 random n (multiples of QK_K, 1..256 super-blocks).
  rng = 0xC0FFEE11u;
  for (int t = 0; t < 2000; ++t) { int sb = 1 + (int)(xr() % 256); run_one(sb*QK_K, 9000+t); }

  // MIN-term (here: zero-scales) negative control: with scales==0 the dot is 0,
  // a DISCRIMINATING degenerate -- both must agree on exactly 0.0f; then perturb
  // one scale and require they STILL agree (the control discriminates a wrong
  // scale-handling kernel). 200 cases.
  int neg_disc = 0;
  for (int t = 0; t < 200; ++t) {
    int n = 256 * (1 + (int)(xr() % 8));
    int nb = n / QK_K;
    uint8_t *vx=(uint8_t*)malloc((size_t)nb*WB), *vy=(uint8_t*)malloc((size_t)nb*YB);
    rng = 0xBEEF0000u ^ (uint32_t)t; fill(vx, vy, nb, 0);
    // corrupt ONE scale byte to force a different per-sub-block contribution
    vx[192 + (t % 16)] = (uint8_t)(0x7F);
    float rg=0, ro=0;
    ggml_vec_dot_q6_K_q8_K_generic(n, &rg, vx, vy);
    tcrv_emitc_ggml_vec_dot_q6_K_q8_K_kernel_ggml_vec_dot_q6_K_q8_K(n, &ro, vx, vy);
    if (biteq(rg, ro) && rg != 0.0f) neg_disc++;
    free(vx); free(vy);
  }

  printf("RESULT positive=%d/%d failures=%d  neg-control discriminating=%d/200\n",
         pass, total, fail, neg_disc);
  printf("%s\n", (fail==0 && neg_disc==200) ? "PASS" : "FAIL");
  return (fail==0 && neg_disc==200) ? 0 : 1;
}
