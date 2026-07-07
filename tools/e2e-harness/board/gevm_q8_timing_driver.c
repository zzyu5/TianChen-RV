/* gevm_q8_timing_driver.c -- STEADY-STATE timing driver for the CONSTRUCTED
 * symmetric q8_0 x q8_0 16-way block-as-lane REPACK GEVM (decode, M=1) kernel.
 * Isolates the q8_0 strip-width core so the narrow(half_lanes=8) vs wide(16) vs
 * whole-LMUL(m1) forms can be A/B timed at VLEN256 for the [GAP-P1] per-format
 * "should the selector widen q8_0" decision. Layout DERIVED from the emitted body:
 *   WEIGHT block_q8_0x16 (544 B), per 16-COL group g, per block b:
 *     blk = rw + (g*nb + b)*544
 *     bytes[0..31]    : 16 x f16 col scales; col cc at blk[cc*2]
 *     bytes[32..543]  : qs lane l(0..31), col cc -> blk[32 + l*16 + cc] (int8)
 *   ACT block_q8_0 (34 B), per block b:  blk = ra + b*34; d=f16 at [0], qs[l]=[2+l]
 *   OUTPUT: out[c], c in [0,nc).   n = K (contraction elems), nc = #cols.
 * ABI (5-role): (n, s, vx=weight, vy=act, nc).
 *
 * argv: <K(mult32)> <nc(mult16)> <iters> <seed>
 * stdout: GEVM VLEN=<bits> K=.. nc=.. iters=.. kernel_ns=<ns/call> gbps=<x> cksum=<hex>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

typedef uint16_t ggml_half;
#define QK8_0 32
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0; /* 34 */
_Static_assert(sizeof(block_q8_0)==34,"q8_0");

extern void tcrv_emitc_ggml_repack_gemv_q8_0_q8_0_kernel_ggml_repack_gemv_q8_0_q8_0(
    size_t n, float*s, const uint8_t*vx, const uint8_t*vy, size_t nc);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static int8_t  ri8(void){ return (int8_t)((int)(xr()%255)-127); }
static uint16_t rf16(void){ uint16_t s=(uint16_t)((xr()&1)<<15),e=(uint16_t)((10+(xr()%8))&0x1F),m=(uint16_t)(xr()&0x3FF); return (uint16_t)(s|(e<<10)|m); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s K nc iters seed\n",argv[0]); return 2; }
    int K=atoi(argv[1]), nc=atoi(argv[2]), iters=atoi(argv[3]);
    rng=(uint64_t)strtoull(argv[4],0,0)|1ull;
    if(K%32||nc%16){ fprintf(stderr,"K%%32,nc%%16 required\n"); return 2; }
    long vlen_bits=(long)__riscv_vlenb()*8;
    int nb=K/QK8_0, grp_c=nc/16;
    size_t wbytes=(size_t)grp_c*nb*544, abytes=(size_t)nb*34;

    block_q8_0 *W=malloc((size_t)nc*nb*sizeof*W);  /* col-major W[c*nb+b] */
    block_q8_0 *A=malloc((size_t)nb*sizeof*A);      /* one act vector */
    uint8_t *rw=malloc(wbytes), *ra=malloc(abytes);
    float *out=malloc((size_t)nc*sizeof(float));
    if(!W||!A||!rw||!ra||!out){ fprintf(stderr,"OOM\n"); return 3; }

    for(int c=0;c<nc;c++) for(int b=0;b<nb;b++){ W[c*nb+b].d=rf16(); for(int j=0;j<QK8_0;j++) W[c*nb+b].qs[j]=ri8(); }
    for(int b=0;b<nb;b++){ A[b].d=rf16(); for(int j=0;j<QK8_0;j++) A[b].qs[j]=ri8(); }
    /* weight repack block_q8_0x16 */
    for(int g=0;g<grp_c;g++) for(int b=0;b<nb;b++){
        uint8_t *blk=rw+((size_t)g*nb+b)*544;
        for(int cc=0;cc<16;cc++){ int c=g*16+cc; memcpy(blk+cc*2,&W[c*nb+b].d,2); }
        for(int l=0;l<QK8_0;l++) for(int cc=0;cc<16;cc++){ int c=g*16+cc;
            blk[32 + l*16 + cc]=(uint8_t)W[c*nb+b].qs[l]; }
    }
    /* activation q8_0 vector */
    for(int b=0;b<nb;b++){ uint8_t*blk=ra+(size_t)b*34; memcpy(blk,&A[b].d,2);
        for(int l=0;l<QK8_0;l++) blk[2+l]=(uint8_t)A[b].qs[l]; }

    for(int w=0;w<3;w++)
        tcrv_emitc_ggml_repack_gemv_q8_0_q8_0_kernel_ggml_repack_gemv_q8_0_q8_0((size_t)K,out,rw,ra,(size_t)nc);

    double best=1e30; volatile double sink=0;
    for(int pass=0;pass<5;pass++){
        double t0=now_ns();
        for(int it=0;it<iters;it++)
            tcrv_emitc_ggml_repack_gemv_q8_0_q8_0_kernel_ggml_repack_gemv_q8_0_q8_0((size_t)K,out,rw,ra,(size_t)nc);
        double t1=now_ns(); double npc=(t1-t0)/(double)iters;
        if(npc<best) best=npc; sink+=out[0]+out[nc-1];
    }
    uint64_t cks=1469598103934665603ULL;
    for(int i=0;i<nc;i++){ uint32_t u; memcpy(&u,&out[i],4); for(int k=0;k<4;k++){ cks^=(u>>(k*8))&0xff; cks*=1099511628211ULL; } }
    /* bytes streamed per call ~= weight (dominant): grp_c*nb*544 + act nb*34 */
    double bytes=(double)wbytes+(double)abytes;
    double gbps=bytes/best; /* bytes/ns == GB/s */
    printf("GEVM VLEN=%ld K=%d nc=%d iters=%d kernel_ns=%.1f gbps=%.4f cksum=%016llx sink=%.1f\n",
           vlen_bits,K,nc,iters,best,gbps,(unsigned long long)cks,(double)sink);
    free(W);free(A);free(rw);free(ra);free(out);
    return 0;
}
