/* gemm_timing_driver.c -- STEADY-STATE timing driver for the front-door
 * CONSTRUCTED q4_0 16x1-REPACKED GEMM (PREFILL, M>1) kernel. Sibling of
 * ../silicon-validation-gemm/gemm_verify_driver.c (SAME repack layout + ABI);
 * that one proves bit-exactness, this one measures ns/call + GMAC/s so the
 * [GAP-P1] mf2(underfed)-vs-m1(wide) core selection can be A/B timed on-board.
 *
 * Repack layouts + 7-role ABI are IDENTICAL to gemm_verify_driver.c (kept in
 * sync by construction). We additionally:
 *   - print VLEN(bits) (preflight gate-4 fingerprint = __riscv_vlenb()*8),
 *   - warm up, then time `iters` back-to-back kernel calls under one
 *     CLOCK_MONOTONIC span => ns_per_call (steady-state, compute-bound),
 *   - emit a checksum over out[] (defeats DCE + proves pre/post are the SAME
 *     number, i.e. the mf2->m1 flip is byte/value-identical by construction).
 *
 * argv: <K(mult32)> <nr(mult4)> <nc(mult16)> <iters> <seed>
 * stdout (one line, grep-able by the A/B harness):
 *   GEMM VLEN=<bits> K=.. nr=.. nc=.. iters=.. kernel_ns=<ns/call> gmacs=<x> cksum=<hex>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

typedef uint16_t ggml_half;
#define QK4_0 32
#define QK8_0 32
typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0; /* 18 */
typedef struct { ggml_half d; int8_t  qs[QK8_0];    } block_q8_0; /* 34 */
_Static_assert(sizeof(block_q4_0)==18,"q4_0");
_Static_assert(sizeof(block_q8_0)==34,"q8_0");

/* constructed GEMM, 7-role ABI (nr, bs, n, s, nc, vx, vy) */
extern void tcrv_emitc_ggml_gemm_q4_0_q8_0_repack_gemm_kernel_ggml_gemm_q4_0_q8_0_repack_gemm(
    size_t nr, size_t bs, size_t n, float*s, size_t nc, const uint8_t*vx, const uint8_t*vy);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static int8_t  ri8(void){ return (int8_t)((int)(xr()%255)-127); }
static uint8_t ru8(void){ return (uint8_t)(xr()&0xff); }
static uint16_t rf16(void){ uint16_t s=(uint16_t)((xr()&1)<<15),e=(uint16_t)((10+(xr()%8))&0x1F),m=(uint16_t)(xr()&0x3FF); return (uint16_t)(s|(e<<10)|m); }

static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }

int main(int argc, char**argv){
    if(argc<6){ fprintf(stderr,"usage: %s K nr nc iters seed\n",argv[0]); return 2; }
    int K=atoi(argv[1]), nr=atoi(argv[2]), nc=atoi(argv[3]), iters=atoi(argv[4]);
    rng=(uint64_t)strtoull(argv[5],0,0)|1ull;
    if(K%32||nr%4||nc%16){ fprintf(stderr,"K%%32,nr%%4,nc%%16 required\n"); return 2; }
    long vlen_bits = (long)__riscv_vlenb()*8;
    int nb=K/QK4_0, grp_c=nc/16, grp_r=nr/4;
    size_t wbytes=(size_t)grp_c*nb*288, abytes=(size_t)grp_r*nb*136;

    block_q4_0 *W=malloc((size_t)nc*nb*sizeof*W);
    block_q8_0 *A=malloc((size_t)nr*nb*sizeof*A);
    uint8_t *rw=malloc(wbytes), *ra=malloc(abytes);
    float *out=malloc((size_t)nr*nc*sizeof(float));
    if(!W||!A||!rw||!ra||!out){ fprintf(stderr,"OOM\n"); return 3; }

    /* one fixed random instance (col-major W, row-major A) + repack */
    for(int c=0;c<nc;c++) for(int b=0;b<nb;b++){ W[c*nb+b].d=rf16(); for(int j=0;j<QK4_0/2;j++) W[c*nb+b].qs[j]=ru8(); }
    for(int r=0;r<nr;r++) for(int b=0;b<nb;b++){ A[r*nb+b].d=rf16(); for(int j=0;j<QK8_0;j++) A[r*nb+b].qs[j]=ri8(); }
    for(int g=0;g<grp_c;g++) for(int b=0;b<nb;b++){
        uint8_t *blk = rw + ((size_t)g*nb + b)*288;
        for(int cc=0;cc<16;cc++){ int c=g*16+cc; memcpy(blk + cc*2, &W[c*nb+b].d, 2); }
        for(int l=0;l<QK4_0/2;l++) for(int cc=0;cc<16;cc++){ int c=g*16+cc;
            blk[32 + l*16 + cc] = (uint8_t)(W[c*nb+b].qs[l] ^ 0x88); }
    }
    for(int g=0;g<grp_r;g++) for(int b=0;b<nb;b++){
        uint8_t *blk = ra + ((size_t)g*nb + b)*136;
        for(int rr=0;rr<4;rr++){ int r=g*4+rr; memcpy(blk + rr*2, &A[r*nb+b].d, 2); }
        for(int l=0;l<QK8_0/2;l++) for(int rr=0;rr<4;rr++){ int r=g*4+rr;
            blk[8  + l*4 + rr] = (uint8_t)A[r*nb+b].qs[l];
            blk[72 + l*4 + rr] = (uint8_t)A[r*nb+b].qs[l+QK8_0/2]; }
    }

    /* warmup (absorbs cold I/D-cache; 3 calls) */
    for(int w=0; w<3; w++)
        tcrv_emitc_ggml_gemm_q4_0_q8_0_repack_gemm_kernel_ggml_gemm_q4_0_q8_0_repack_gemm(
            (size_t)nr,(size_t)nc,(size_t)K,out,(size_t)nc,rw,ra);

    /* timed: best-of-5 passes, each `iters` back-to-back calls; report MIN pass */
    double best_ns_per_call = 1e30;
    volatile double sink=0;
    for(int pass=0; pass<5; pass++){
        double t0=now_ns();
        for(int it=0; it<iters; it++)
            tcrv_emitc_ggml_gemm_q4_0_q8_0_repack_gemm_kernel_ggml_gemm_q4_0_q8_0_repack_gemm(
                (size_t)nr,(size_t)nc,(size_t)K,out,(size_t)nc,rw,ra);
        double t1=now_ns();
        double npc=(t1-t0)/(double)iters;
        if(npc<best_ns_per_call) best_ns_per_call=npc;
        sink += out[0]+out[(size_t)nr*nc-1];
    }

    /* checksum over out[] (int-bit fold; identical iff same numeric result) */
    uint64_t cks=1469598103934665603ULL; /* FNV-1a over the float bytes */
    for(size_t i=0;i<(size_t)nr*nc;i++){ uint32_t u; memcpy(&u,&out[i],4);
        for(int k=0;k<4;k++){ cks^=(u>>(k*8))&0xff; cks*=1099511628211ULL; } }

    double macs=(double)nr*(double)nc*(double)K; /* one MAC per (r,c,element) */
    double gmacs = macs / best_ns_per_call;      /* macs/ns == GMAC/s */
    printf("GEMM VLEN=%ld K=%d nr=%d nc=%d iters=%d kernel_ns=%.1f gmacs=%.4f cksum=%016llx sink=%.1f\n",
           vlen_bits,K,nr,nc,iters,best_ns_per_call,gmacs,(unsigned long long)cks,(double)sink);
    free(W);free(A);free(rw);free(ra);free(out);
    return 0;
}
