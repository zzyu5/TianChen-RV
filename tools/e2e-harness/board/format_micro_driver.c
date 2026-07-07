/* format_micro_driver.c -- cache-HYGIENE paired micro driver for the 8 constructed
 * super-block-grid vec_dot kernels {iq3_s, iq2_s, iq2_xs, iq2_xxs, iq3_xxs, iq4_xs,
 * tq2_0, tq1_0} (lineA format-micro, step 4; TQ = TriLM ternary siblings, `d` LAST in
 * block). Times ONE side per invocation (side=ours|factory);
 * the paired orchestrator (format_micro_paired.sh) interleaves the two sides and
 * forms the ratio. Keeping each side in its own timed pass matches board_ab.sh.
 *
 * CACHE HYGIENE (the reason the batch#1 perf leg was left OPEN):
 *   A cache-RESIDENT micro (tiny working set reused N times) reports a speedup that
 *   does NOT survive real decode, where weights stream COLD from DRAM. So this driver
 *   streams a POOL whose total footprint exceeds the board L2 by several x: each timed
 *   ROUND sweeps the ENTIRE oversized pool exactly once (ncalls * blocks_per_call ==
 *   pool_blocks), so every super-block is touched cold. Optionally a do_bench-style
 *   flush buffer is swept between rounds/calls (flush_mib>0). The achieved bytes/s is
 *   reported so the orchestrator can roofline-classify honestly (bandwidth- vs
 *   latency/compute-bound). A run whose working_set_bytes does not exceed the board L2
 *   by the required margin is marked NOT-HYGIENIC by the orchestrator and the cell row
 *   is STALE.
 *
 * PORTABILITY: pure host C (no riscv_vector.h / no intrinsics) so the STRUCTURE self-
 * tests + links + runs on the x86 host against scalar stubs (format_micro_*_stub.c).
 * The board build links the real constructed kernels (.o) + the real ggml factory .o.
 *
 * argv: <fmt> <side:ours|factory> <n_elems> <rounds> <pool_mib> <flush_mib> <seed>
 *   defaults:            n=4096   rounds=12  pool_mib=64  flush_mib=0  seed=0x1234567
 * one machine-parseable line:
 *   MICRO fmt=<f> side=<s> n=<n> blocks_per_call=<b> pool_blocks=<PB> working_set_bytes=<WB>
 *         cache_strategy=<..> rounds=<R> ns_per_block_median=<m> ns_per_block_iqr=<i>
 *         ns_per_block_min=<mn> achieved_GBs=<g> fingerprint=<hex> secs=<t>
 */
#define _POSIX_C_SOURCE 199309L      /* clock_gettime / CLOCK_MONOTONIC under strict -std */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint16_t ggml_half;
#define QK_K 256

/* ---- ggml block layouts (natural contiguous == ggml on-disk layout) ---- */
typedef struct { ggml_half d; uint16_t qs[QK_K/8]; }                                              block_iq2_xxs; /* 66  */
typedef struct { ggml_half d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; }                     block_iq2_xs;  /* 74  */
typedef struct { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } block_iq2_s;   /* 82  */
typedef struct { ggml_half d; uint8_t qs[3*QK_K/8]; }                                             block_iq3_xxs; /* 98  */
typedef struct { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32];
                 uint8_t signs[QK_K/8]; uint8_t scales[QK_K/64]; }                                 block_iq3_s;   /* 110 */
typedef struct { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; }  block_iq4_xs;  /* 136 */
/* TQ (TriLM ternary) blocks: NOTE `d` is the LAST field (offset != 0, unlike IQ) */
typedef struct { uint8_t qs[QK_K/4]; ggml_half d; }                                                block_tq2_0;   /* 66  (qs@0, d@64) */
typedef struct { uint8_t qs[(QK_K-4*QK_K/64)/5]; uint8_t qh[QK_K/64]; ggml_half d; }                block_tq1_0;   /* 54  (qs@0, qh@48, d@52) */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; }                               block_q8_K;    /* 292 */

_Static_assert(sizeof(block_iq2_xxs)==66,  "iq2_xxs");
_Static_assert(sizeof(block_iq2_xs)==74,   "iq2_xs");
_Static_assert(sizeof(block_iq2_s)==82,    "iq2_s");
_Static_assert(sizeof(block_iq3_xxs)==98,  "iq3_xxs");
_Static_assert(sizeof(block_iq3_s)==110,   "iq3_s");
_Static_assert(sizeof(block_iq4_xs)==136,  "iq4_xs");
_Static_assert(sizeof(block_tq2_0)==66,    "tq2_0");
_Static_assert(sizeof(block_tq1_0)==54,    "tq1_0");
_Static_assert(offsetof(block_tq2_0,d)==64,"tq2_0 d@64");
_Static_assert(offsetof(block_tq1_0,d)==52,"tq1_0 d@52");
_Static_assert(sizeof(block_q8_K)==292,    "q8_K");

/* ---- side ours: the exported constructed kernels (extern "C", resolved on board) ---- */
extern void tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_rvv_iq3_s_q8_K_block_dot   (size_t n, float*s, const uint8_t*vx, const uint8_t*vy);
extern void tcrv_emitc_ggml_vec_dot_iq2_s_q8_K_kernel_rvv_iq2_s_q8_K_block_dot   (size_t n, float*s, const uint8_t*vx, const uint8_t*vy);
extern void tcrv_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot (size_t n, float*s, const uint8_t*vx, const uint8_t*vy);
extern void tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot(size_t n, float*s, const uint8_t*vx, const uint8_t*vy);
extern void tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_rvv_iq3_xxs_q8_K_block_dot(size_t n, float*s, const uint8_t*vx, const uint8_t*vy);
extern void tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot (size_t n, float*s, const uint8_t*vx, const uint8_t*vy);
extern void tcrv_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_rvv_tq2_0_q8_K_block_dot   (size_t n, float*s, const uint8_t*vx, const uint8_t*vy);
extern void tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot   (size_t n, float*s, const uint8_t*vx, const uint8_t*vy);

/* ---- side factory: the ggml DISPATCHED vec_dot (extern; format_micro_opponent.sh
 *      compiles factory.o from pinned ggml source and provides these symbols) ---- */
extern void ggml_vec_dot_iq3_s_q8_K  (int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);
extern void ggml_vec_dot_iq2_s_q8_K  (int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);
extern void ggml_vec_dot_iq2_xs_q8_K (int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);
extern void ggml_vec_dot_iq2_xxs_q8_K(int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);
extern void ggml_vec_dot_iq3_xxs_q8_K(int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);
extern void ggml_vec_dot_iq4_xs_q8_K (int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);
extern void ggml_vec_dot_tq2_0_q8_K  (int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);
extern void ggml_vec_dot_tq1_0_q8_K  (int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);

/* ---- format descriptor: name, weight block size, f16-scale offset in block, side fn ptrs ----
 * d_off = byte offset of the ggml_half `d` scale within a weight block. IQ formats keep
 * `d` FIRST (offset 0); the TQ (TriLM) formats keep it LAST (tq2_0@64, tq1_0@52). fill_weight
 * writes a finite moderate f16 there so the scalar fold `sumf += (float)sumi * d` stays finite. */
typedef void (*ours_fn)(size_t, float*, const uint8_t*, const uint8_t*);
typedef void (*fact_fn)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
typedef struct { const char* name; size_t wsize; size_t d_off; ours_fn ours; fact_fn fact; } fmt_desc;

static const fmt_desc FMTS[] = {
  {"iq3_s",   sizeof(block_iq3_s),   0,  tcrv_emitc_ggml_vec_dot_iq3_s_q8_K_kernel_rvv_iq3_s_q8_K_block_dot,     ggml_vec_dot_iq3_s_q8_K},
  {"iq2_s",   sizeof(block_iq2_s),   0,  tcrv_emitc_ggml_vec_dot_iq2_s_q8_K_kernel_rvv_iq2_s_q8_K_block_dot,     ggml_vec_dot_iq2_s_q8_K},
  {"iq2_xs",  sizeof(block_iq2_xs),  0,  tcrv_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot,   ggml_vec_dot_iq2_xs_q8_K},
  {"iq2_xxs", sizeof(block_iq2_xxs), 0,  tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot, ggml_vec_dot_iq2_xxs_q8_K},
  {"iq3_xxs", sizeof(block_iq3_xxs), 0,  tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_rvv_iq3_xxs_q8_K_block_dot, ggml_vec_dot_iq3_xxs_q8_K},
  {"iq4_xs",  sizeof(block_iq4_xs),  0,  tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot,   ggml_vec_dot_iq4_xs_q8_K},
  {"tq2_0",   sizeof(block_tq2_0),   64, tcrv_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_rvv_tq2_0_q8_K_block_dot,     ggml_vec_dot_tq2_0_q8_K},
  {"tq1_0",   sizeof(block_tq1_0),   52, tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot,     ggml_vec_dot_tq1_0_q8_K},
};
#define NFMT (int)(sizeof(FMTS)/sizeof(FMTS[0]))

/* ---- rng (xorshift, identical to bd_verify_driver) ---- */
static uint64_t rng;
static uint32_t xr(void){ rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; return (uint32_t)(rng>>32); }
static int8_t   ri8(void){ return (int8_t)((int)(xr()%255) - 127); }
static float    rf32(float lo, float hi){ return lo + (hi-lo)*((float)(xr()&0xffffff)/(float)0x1000000); }
/* finite, moderate f16 for the d field (exponent in [10..17]) */
static uint16_t rf16(void){
    uint16_t s = (uint16_t)((xr()&1)<<15);
    uint16_t e = (uint16_t)((10 + (xr()%8)) & 0x1F);
    uint16_t m = (uint16_t)(xr()&0x3FF);
    return (uint16_t)(s|(e<<10)|m);
}
/* generic weight fill: all bytes random, then the f16 `d` scale (at d_off) overwritten
 * finite. Valid for ALL 8 formats because every grid/sign/scale index is a MASKED
 * bit-field (in range for any byte) and the ternary qs planes are pure arithmetic decode;
 * this isolates decode+dot timing without a quantizer in the loop. d_off differs by family
 * (IQ: 0 / tq2_0: 64 / tq1_0: 52) so the finite scale lands on the real `d` field. */
static void fill_weight(uint8_t* blk, size_t wsize, size_t d_off){
    for(size_t j=0;j<wsize;j++) blk[j]=(uint8_t)(xr()&0xff);
    uint16_t d = rf16(); memcpy(blk+d_off, &d, 2);
}
static void fill_q8_K(block_q8_K* y){
    y->d = rf32(0.001f, 0.05f);
    for(int j=0;j<QK_K;j++) y->qs[j]=ri8();
    for(int g=0; g<QK_K/16; g++){ int s=0; for(int j=0;j<16;j++) s+=y->qs[g*16+j]; y->bsums[g]=(int16_t)s; }
}

/* ---- percentile stats over R round-times ---- */
static int cmp_d(const void*a, const void*b){ double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }
static double pct(const double* v, int n, double p){          /* nearest-rank on sorted v */
    int idx = (int)(p*(n-1) + 0.5); if(idx<0)idx=0; if(idx>=n)idx=n-1; return v[idx];
}

/* do_bench-style L2 flush buffer (file scope: clang has no nested functions) */
static uint64_t* g_flushbuf = NULL; static size_t g_flush_n = 0; static volatile uint64_t g_flush_sink = 0;
static void flush_do(void){ if(!g_flushbuf) return; uint64_t s=0; for(size_t i=0;i<g_flush_n;i++) s+=g_flushbuf[i]; g_flush_sink+=s; }

int main(int argc, char** argv){
    if(argc<3){ fprintf(stderr,"usage: %s <fmt> <ours|factory> [n] [rounds] [pool_mib] [flush_mib] [seed]\n",argv[0]); return 2; }
    const char* fmt  = argv[1];
    const char* side = argv[2];
    long   n        = (argc>3)? atol(argv[3]) : 4096;
    int    rounds   = (argc>4)? atoi(argv[4]) : 12;
    long   pool_mib = (argc>5)? atol(argv[5]) : 64;
    long   flush_mib= (argc>6)? atol(argv[6]) : 0;
    rng = (argc>7)? (strtoull(argv[7],0,0)|1ull) : 0x1234567ull;

    const fmt_desc* fd = NULL;
    for(int i=0;i<NFMT;i++) if(!strcmp(fmt,FMTS[i].name)){ fd=&FMTS[i]; break; }
    if(!fd){ fprintf(stderr,"unknown fmt '%s'\n",fmt); return 2; }
    int is_ours = !strcmp(side,"ours"); int is_fact = !strcmp(side,"factory");
    if(!is_ours && !is_fact){ fprintf(stderr,"side must be ours|factory\n"); return 2; }
    if(rounds<10){ fprintf(stderr,"WARN rounds=%d < 10 (N>=10 required for median+IQR)\n",rounds); }

    long nblk = n/QK_K; if(nblk<1) nblk=1;
    size_t pair_bytes = fd->wsize + sizeof(block_q8_K);
    /* pool sized to exceed L2 by several x; round pool_blocks DOWN to a multiple of nblk
     * so an integer number of calls sweeps the WHOLE pool exactly once per round. */
    long pool_blocks = (long)(((uint64_t)pool_mib*1024*1024) / pair_bytes);
    if(pool_blocks < nblk*2) pool_blocks = nblk*2;
    pool_blocks -= (pool_blocks % nblk);
    long ncalls = pool_blocks / nblk;
    size_t working_set_bytes = (size_t)pool_blocks * pair_bytes;

    uint8_t*    wpool = (uint8_t*)malloc((size_t)pool_blocks * fd->wsize);
    block_q8_K* ypool = (block_q8_K*)malloc((size_t)pool_blocks * sizeof(block_q8_K));
    if(!wpool || !ypool){ fprintf(stderr,"alloc fail (%zu B pool)\n",working_set_bytes); return 1; }
    for(long b=0;b<pool_blocks;b++){ fill_weight(wpool + (size_t)b*fd->wsize, fd->wsize, fd->d_off); fill_q8_K(&ypool[b]); }

    /* optional do_bench-style L2 flush buffer swept between rounds */
    if(flush_mib>0){ g_flush_n = (size_t)flush_mib*1024*1024/sizeof(uint64_t);
                     g_flushbuf = (uint64_t*)malloc(g_flush_n*sizeof(uint64_t));
                     if(g_flushbuf) for(size_t i=0;i<g_flush_n;i++) g_flushbuf[i]=i*2654435761u+1; }
    char strat[64];
    if(flush_mib>0 && g_flushbuf) snprintf(strat,sizeof strat,"oversized-pool:%ldMiB+per-iter-flush:%ldMiB",pool_mib,flush_mib);
    else                         snprintf(strat,sizeof strat,"oversized-pool:%ldMiB",pool_mib);

    /* warmup pass (dropped) */
    { long off=0; float s; volatile float w=0;
      for(long c=0;c<ncalls;c++){ long base=off%(pool_blocks-nblk+1); s=0;
        if(is_ours) fd->ours((size_t)n,&s,wpool+(size_t)base*fd->wsize,(const uint8_t*)&ypool[base]);
        else        fd->fact((int)n,&s,0,wpool+(size_t)base*fd->wsize,0,&ypool[base],0,1);
        w+=s; off+=nblk; } (void)w; }
    flush_do();

    double* round_ns_per_block = (double*)malloc((size_t)rounds*sizeof(double));
    struct timespec t0,t1; double total_secs=0; uint64_t fp=1469598103934665603ull; /* FNV-1a over outputs */

    for(int r=0; r<rounds; r++){
        long off=0; volatile float acc=0;
        clock_gettime(CLOCK_MONOTONIC,&t0);
        for(long c=0;c<ncalls;c++){
            long base = off % (pool_blocks - nblk + 1);      /* rotate: each call cold */
            float s=0;
            if(is_ours) fd->ours((size_t)n,&s,wpool+(size_t)base*fd->wsize,(const uint8_t*)&ypool[base]);
            else        fd->fact((int)n,&s,0,wpool+(size_t)base*fd->wsize,0,&ypool[base],0,1);
            acc += s;
            uint32_t sb; memcpy(&sb,&s,4); fp ^= sb; fp *= 1099511628211ull;
            off += nblk;
        }
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double secs = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        total_secs += secs;
        round_ns_per_block[r] = secs*1e9 / (double)(ncalls*nblk);
        (void)acc;
        flush_do();
    }

    qsort(round_ns_per_block, rounds, sizeof(double), cmp_d);
    double med = pct(round_ns_per_block,rounds,0.50);
    double q1  = pct(round_ns_per_block,rounds,0.25);
    double q3  = pct(round_ns_per_block,rounds,0.75);
    double mn  = round_ns_per_block[0];
    /* achieved bytes/s over the whole measured stream (pool swept `rounds` times) */
    double achieved_GBs = (double)working_set_bytes*(double)rounds / total_secs / 1e9;

    printf("MICRO fmt=%s side=%s n=%ld blocks_per_call=%ld pool_blocks=%ld working_set_bytes=%zu "
           "cache_strategy=%s rounds=%d ns_per_block_median=%.3f ns_per_block_iqr=%.3f "
           "ns_per_block_min=%.3f achieved_GBs=%.3f fingerprint=0x%016llx secs=%.4f flush_sink=%llu\n",
           fmt, side, n, nblk, pool_blocks, working_set_bytes, strat, rounds,
           med, (q3-q1), mn, achieved_GBs, (unsigned long long)fp, total_secs,
           (unsigned long long)g_flush_sink);

    free(round_ns_per_block); free(wpool); free(ypool); free(g_flushbuf);
    return 0;
}
