/* tq10_vecdot_driver.c -- byte-exact gate + cold timing for the owned FUSED tq1_0
 * vec_dot leaf vs the deployed ggml opponent. B线第二块 P1 (公式墙攻坚).
 *
 * modes:
 *   verify <corpus_blocks> <seed>
 *       ZERO-MODEL byte-exact gate: for each of `corpus_blocks` independent random
 *       super-blocks, compare OURS (fused leaf) vs an INDEPENDENT scalar ORACLE
 *       (recomputes ALL arithmetic from raw input bytes, no shared intermediates) vs
 *       FACTORY (deployed ggml_vec_dot_tq1_0_q8_K). Integer core is byte-exact
 *       (0 tolerance); the fp32 scale fold is reported with a ULP bound. Then a
 *       3-arm anti-hollow probe (oracle-fault / DUT-fault / factory-fault) proves the
 *       comparison actually bites (each injected fault must flip >=1 differing byte).
 *   time <ours|factory> <n> <rounds> <pool_mib> <flush_mib> <seed>
 *       cache-cold paired micro (oversized pool swept once/round). Prints the same
 *       MICRO line shape as tools/e2e-harness/board/format_micro_driver.c so the
 *       cold ratio is formed exactly as the census sweep.
 *
 * PORTABLE host C for the scalar ORACLE + driver; the OURS leaf (riscv_vector.h) and
 * the FACTORY .o are linked on-board. No git writes;仓库侧零板端产物.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint16_t ggml_half;
#define QK_K 256

/* block_tq1_0 (54B): qs[48] @0, qh[4] @48, d(fp16) @52.  d is LAST (offset != 0). */
typedef struct { uint8_t qs[(QK_K - 4*QK_K/64)/5]; uint8_t qh[QK_K/64]; ggml_half d; } block_tq1_0;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; }                    block_q8_K;
_Static_assert(sizeof(block_tq1_0) == 54, "tq1_0 54");
_Static_assert(offsetof(block_tq1_0, d) == 52, "tq1_0 d@52");
_Static_assert(sizeof(block_q8_K) == 292, "q8_K 292");

/* ---- OURS: the owned FUSED leaf (resolved on board) ---- */
extern void weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);
/* ---- FACTORY: the deployed ggml dispatched vec_dot (compiler-symmetric build) ---- */
extern void ggml_vec_dot_tq1_0_q8_K(int n, float *s, size_t bs, const void *vx,
                                    size_t bx, const void *vy, size_t by, int nrc);

static float f16_to_f32(uint16_t h);  /* fwd decl (defined below, oracle-only) */

/* ---- INDEPENDENT scalar ZERO-MODEL oracle (recompute from raw bytes) ---- */
/* pow3 mod-256 base-3 decode + exact q8 index map, integer sum, single fp32 fold.
 * Deliberately NOT sharing any intermediate with the vectorized leaf. */
static float oracle_tq1_0(const block_tq1_0 *x, const block_q8_K *y) {
  static const int pow3[5] = {1, 3, 9, 27, 81};
  int sum = 0;
  /* main qs: j=0, l=0..4, m=0..31 -> q8[l*32 + m] */
  for (int l = 0; l < 5; ++l)
    for (int m = 0; m < 32; ++m) {
      uint8_t q = (uint8_t)((int)x->qs[m] * pow3[l]);
      uint16_t xi = (uint16_t)(((uint16_t)q * 3) >> 8);
      sum += ((int)xi - 1) * (int)y->qs[l * 32 + m];
    }
  /* tail qs: j=32, l=0..4, m=0..15 -> q8[160 + l*16 + m] */
  for (int l = 0; l < 5; ++l)
    for (int m = 0; m < 16; ++m) {
      uint8_t q = (uint8_t)((int)x->qs[32 + m] * pow3[l]);
      uint16_t xi = (uint16_t)(((uint16_t)q * 3) >> 8);
      sum += ((int)xi - 1) * (int)y->qs[160 + l * 16 + m];
    }
  /* qh: l=0..3, j=0..3 -> q8[240 + l*4 + j] */
  for (int l = 0; l < 4; ++l)
    for (int j = 0; j < 4; ++j) {
      uint8_t q = (uint8_t)((int)x->qh[j] * pow3[l]);
      uint16_t xi = (uint16_t)(((uint16_t)q * 3) >> 8);
      sum += ((int)xi - 1) * (int)y->qs[240 + l * 4 + j];
    }
  /* single-scale fp32 fold: (float)sum * (fp16(x.d) * y.d). fp16->f32 via a portable
   * scalar decode so the oracle needs no zfh (keeps it host-independent). */
  return (float)sum * (f16_to_f32(x->d) * y->d);
}

/* portable fp16 -> f32 (IEEE754 half), used ONLY by the scalar oracle. */
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t man  = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) { bits = sign; }
    else { /* subnormal */
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; exp--; }
      man &= 0x3FF;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000 | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f; memcpy(&f, &bits, 4); return f;
}

/* ---- rng (xorshift; identical family to format_micro_driver) ---- */
static uint64_t rng;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 32); }
static int8_t   ri8(void) { return (int8_t)((int)(xr() % 255) - 127); }
static float    rf32(float lo, float hi) { return lo + (hi - lo) * ((float)(xr() & 0xffffff) / (float)0x1000000); }
static uint16_t rf16(void) {  /* finite moderate f16 (exp in [10..17]) */
  uint16_t s = (uint16_t)((xr() & 1) << 15);
  uint16_t e = (uint16_t)((10 + (xr() % 8)) & 0x1F);
  uint16_t m = (uint16_t)(xr() & 0x3FF);
  return (uint16_t)(s | (e << 10) | m);
}
/* int_mode: force both scales to 1.0 so the fold `sumf = (float)sumi * 1.0` is EXACT
 * (|sumi| < 2^24). Then ours==oracle byte-exact IFF the integer core matches (0-tol
 * integer gate, no float-scale rounding in the way). float_mode uses random finite
 * scales to exercise the fp32 fold and measure its ULP bound (numerics.reassoc_ok). */
static void fill_weight(block_tq1_0 *x, int int_mode) {
  uint8_t *p = (uint8_t *)x;
  for (size_t j = 0; j < sizeof(*x); ++j) p[j] = (uint8_t)(xr() & 0xff);
  x->d = int_mode ? (uint16_t)0x3C00 /* fp16 1.0 */ : rf16();
}
static void fill_q8(block_q8_K *y, int int_mode) {
  y->d = int_mode ? 1.0f : rf32(0.001f, 0.05f);
  for (int j = 0; j < QK_K; ++j) y->qs[j] = ri8();
  for (int g = 0; g < QK_K / 16; ++g) { int s = 0; for (int j = 0; j < 16; ++j) s += y->qs[g*16+j]; y->bsums[g] = (int16_t)s; }
}

static int ulp(float a, float b) {
  int32_t ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
  if (ia < 0) ia = 0x7fffffff - ia;
  if (ib < 0) ib = 0x7fffffff - ib;
  long d = (long)ia - (long)ib; return d < 0 ? (int)-d : (int)d;
}
static int diff_bytes(float a, float b) {
  uint8_t pa[4], pb[4]; memcpy(pa, &a, 4); memcpy(pb, &b, 4);
  int d = 0; for (int i = 0; i < 4; ++i) d += (pa[i] != pb[i]); return d;
}

/* ============================ verify mode ============================ */
static int run_verify(long corpus, uint64_t seed) {
  rng = seed | 1ull;
  block_tq1_0 x; block_q8_K y;

  /* ---- (1) INT-mode gate: d=1.0 both sides, fold exact -> integer core 0-tolerance ---- */
  long int_mism_oo = 0;   /* ours vs oracle: MUST be 0 (integer core byte-exact) */
  long int_mism_fo = 0;   /* factory vs oracle: sanity (opponent int core, expect 0) */
  for (long b = 0; b < corpus; ++b) {
    fill_weight(&x, 1); fill_q8(&y, 1);
    float o = 0, f = 0, g = oracle_tq1_0(&x, &y);
    weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(QK_K, &o, (const uint8_t*)&x, (const uint8_t*)&y);
    ggml_vec_dot_tq1_0_q8_K(QK_K, &f, 0, &x, 0, &y, 0, 1);
    if (diff_bytes(o, g)) int_mism_oo++;
    if (diff_bytes(f, g)) int_mism_fo++;
  }

  /* ---- (2) FLOAT-mode: random finite scales -> fp32 fold ULP bound ---- */
  int max_ulp_oo = 0, max_ulp_fo = 0, max_ulp_of = 0; long fmode_bytediv_oo = 0;
  for (long b = 0; b < corpus; ++b) {
    fill_weight(&x, 0); fill_q8(&y, 0);
    float o = 0, f = 0, g = oracle_tq1_0(&x, &y);
    weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(QK_K, &o, (const uint8_t*)&x, (const uint8_t*)&y);
    ggml_vec_dot_tq1_0_q8_K(QK_K, &f, 0, &x, 0, &y, 0, 1);
    int uoo = ulp(o, g), ufo = ulp(f, g), uof = ulp(o, f);
    if (uoo > max_ulp_oo) max_ulp_oo = uoo;
    if (ufo > max_ulp_fo) max_ulp_fo = ufo;
    if (uof > max_ulp_of) max_ulp_of = uof;
    if (diff_bytes(o, g)) fmode_bytediv_oo++;
  }

  /* ---- (3) 3-arm anti-hollow (INT-mode, all three bit-identical -> any 1-bit ---- */
  /* corruption is ALWAYS detected; no accidental self-repair of a pre-existing ULP). */
  fill_weight(&x, 1); fill_q8(&y, 1);
  float o = 0, f = 0, g = oracle_tq1_0(&x, &y);
  weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(QK_K, &o, (const uint8_t*)&x, (const uint8_t*)&y);
  ggml_vec_dot_tq1_0_q8_K(QK_K, &f, 0, &x, 0, &y, 0, 1);
  int base_agree = (diff_bytes(o, g) == 0) && (diff_bytes(f, g) == 0); /* pre-condition */
  float o_bad; { uint32_t bits; memcpy(&bits, &o, 4); bits ^= 1u; memcpy(&o_bad, &bits, 4); }  /* DUT-fault */
  float g_bad; { uint32_t bits; memcpy(&bits, &g, 4); bits ^= 1u; memcpy(&g_bad, &bits, 4); }  /* oracle-fault */
  float f_bad; { uint32_t bits; memcpy(&bits, &f, 4); bits ^= 1u; memcpy(&f_bad, &bits, 4); }  /* factory-fault */
  int arm_dut     = diff_bytes(o_bad, g) >= 1;
  int arm_oracle  = diff_bytes(o, g_bad) >= 1;
  int arm_factory = diff_bytes(f_bad, g) >= 1;
  int anti_hollow_ok = base_agree && arm_dut && arm_oracle && arm_factory;

  int pass = (int_mism_oo == 0) && anti_hollow_ok;
  printf("VERIFY tq1_0 corpus=%ld*2 "
         "INT[ours_vs_oracle_mism=%ld fac_vs_oracle_mism=%ld] "
         "FLOAT[max_ulp(ours,oracle)=%d max_ulp(fac,oracle)=%d max_ulp(ours,fac)=%d bytediv(ours,oracle)=%ld] "
         "antihollow[base_agree=%d DUT=%d oracle=%d factory=%d]=%s "
         "numerics.reassoc_ok=%s RESULT=%s\n",
         corpus, int_mism_oo, int_mism_fo,
         max_ulp_oo, max_ulp_fo, max_ulp_of, fmode_bytediv_oo,
         base_agree, arm_dut, arm_oracle, arm_factory, anti_hollow_ok ? "RED-all" : "HOLLOW",
         (max_ulp_oo <= 2 && max_ulp_of <= 2) ? "true" : "false", pass ? "GREEN" : "FAIL");
  return pass ? 0 : 1;
}

/* ============================ time mode ============================ */
static int cmp_d(const void *a, const void *b) { double x = *(const double*)a, y = *(const double*)b; return (x>y)-(x<y); }
static double pctl(const double *v, int n, double p) { int i = (int)(p*(n-1)+0.5); if(i<0)i=0; if(i>=n)i=n-1; return v[i]; }
static uint64_t *g_flush = NULL; static size_t g_flush_n = 0; static volatile uint64_t g_sink = 0;
static void flush_do(void) { if(!g_flush) return; uint64_t s=0; for(size_t i=0;i<g_flush_n;i++) s+=g_flush[i]; g_sink+=s; }

static int run_time(const char *side, long n, int rounds, long pool_mib, long flush_mib, uint64_t seed) {
  rng = seed | 1ull;
  int is_ours = !strcmp(side, "ours"), is_fact = !strcmp(side, "factory");
  if (!is_ours && !is_fact) { fprintf(stderr, "side must be ours|factory\n"); return 2; }
  long nblk = n / QK_K; if (nblk < 1) nblk = 1;
  size_t pair = sizeof(block_tq1_0) + sizeof(block_q8_K);
  long pool_blocks = (long)(((uint64_t)pool_mib*1024*1024) / pair);
  if (pool_blocks < nblk*2) pool_blocks = nblk*2;
  pool_blocks -= (pool_blocks % nblk);
  long ncalls = pool_blocks / nblk;
  size_t wsb = (size_t)pool_blocks * pair;

  uint8_t     *wpool = (uint8_t*)malloc((size_t)pool_blocks * sizeof(block_tq1_0));
  block_q8_K  *ypool = (block_q8_K*)malloc((size_t)pool_blocks * sizeof(block_q8_K));
  if (!wpool || !ypool) { fprintf(stderr, "alloc fail\n"); return 1; }
  for (long b = 0; b < pool_blocks; ++b) { fill_weight((block_tq1_0*)(wpool + (size_t)b*sizeof(block_tq1_0)), 0); fill_q8(&ypool[b], 0); }

  if (flush_mib > 0) { g_flush_n = (size_t)flush_mib*1024*1024/sizeof(uint64_t);
    g_flush = (uint64_t*)malloc(g_flush_n*sizeof(uint64_t));
    if (g_flush) for (size_t i=0;i<g_flush_n;i++) g_flush[i]=i*2654435761u+1; }
  char strat[64];
  if (flush_mib>0 && g_flush) snprintf(strat,sizeof strat,"oversized-pool:%ldMiB+flush:%ldMiB",pool_mib,flush_mib);
  else                        snprintf(strat,sizeof strat,"oversized-pool:%ldMiB",pool_mib);

  { long off=0; float s; volatile float w=0;   /* warmup (dropped) */
    for (long c=0;c<ncalls;c++){ long base=off%(pool_blocks-nblk+1); s=0;
      if (is_ours) weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot((size_t)n,&s,wpool+(size_t)base*sizeof(block_tq1_0),(const uint8_t*)&ypool[base]);
      else         ggml_vec_dot_tq1_0_q8_K((int)n,&s,0,wpool+(size_t)base*sizeof(block_tq1_0),0,&ypool[base],0,1);
      w+=s; off+=nblk; } (void)w; }
  flush_do();

  double *rns = (double*)malloc((size_t)rounds*sizeof(double));
  struct timespec t0,t1; double total=0; uint64_t fp=1469598103934665603ull;
  for (int r=0;r<rounds;r++){ long off=0; volatile float acc=0;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for (long c=0;c<ncalls;c++){ long base=off%(pool_blocks-nblk+1); float s=0;
      if (is_ours) weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot((size_t)n,&s,wpool+(size_t)base*sizeof(block_tq1_0),(const uint8_t*)&ypool[base]);
      else         ggml_vec_dot_tq1_0_q8_K((int)n,&s,0,wpool+(size_t)base*sizeof(block_tq1_0),0,&ypool[base],0,1);
      acc+=s; uint32_t sb; memcpy(&sb,&s,4); fp^=sb; fp*=1099511628211ull; off+=nblk; }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double secs=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9; total+=secs;
    rns[r]=secs*1e9/(double)(ncalls*nblk); (void)acc; flush_do(); }

  qsort(rns,rounds,sizeof(double),cmp_d);
  double med=pctl(rns,rounds,0.50), q1=pctl(rns,rounds,0.25), q3=pctl(rns,rounds,0.75), mn=rns[0];
  double gbs=(double)wsb*(double)rounds/total/1e9;
  printf("MICRO fmt=tq1_0 side=%s n=%ld blocks_per_call=%ld pool_blocks=%ld working_set_bytes=%zu "
         "cache_strategy=%s rounds=%d ns_per_block_median=%.3f ns_per_block_iqr=%.3f ns_per_block_min=%.3f "
         "achieved_GBs=%.3f fingerprint=0x%016llx secs=%.4f flush_sink=%llu\n",
         side,n,nblk,pool_blocks,wsb,strat,rounds,med,(q3-q1),mn,gbs,
         (unsigned long long)fp,total,(unsigned long long)g_sink);
  free(rns); free(wpool); free(ypool); free(g_flush);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s verify <corpus> <seed> | time <ours|factory> <n> <rounds> <pool_mib> <flush_mib> <seed>\n", argv[0]); return 2; }
  if (!strcmp(argv[1], "verify")) {
    long corpus = (argc>2)? atol(argv[2]) : 4096;
    uint64_t seed = (argc>3)? strtoull(argv[3],0,0) : 0xBEEF01ull;
    return run_verify(corpus, seed);
  }
  if (!strcmp(argv[1], "time")) {
    const char *side = (argc>2)? argv[2] : "ours";
    long n        = (argc>3)? atol(argv[3]) : 4096;
    int  rounds   = (argc>4)? atoi(argv[4]) : 20;
    long pool_mib = (argc>5)? atol(argv[5]) : 256;
    long flush_mib= (argc>6)? atol(argv[6]) : 0;
    uint64_t seed = (argc>7)? strtoull(argv[7],0,0) : 0xBEEF01ull;
    return run_time(side, n, rounds, pool_mib, flush_mib, seed);
  }
  fprintf(stderr, "unknown mode '%s'\n", argv[1]); return 2;
}
