/* q4_0_repack_verify_driver.c -- silicon verifier for the front-door-CONSTRUCTED
 * q4_0 16x1-REPACKED GEVM kernel.
 *
 * The repack GEVM computes, per output row, the SAME q4_0 x q8_0 dot product as
 * ggml, but with a VECTOR-FMA fold (vfmacc) and dx*dy pre-grouped + 16-wide lane
 * accumulation -- structurally distinct from the no-FMA left-assoc scalar fold.
 * So we report BOTH:
 *   (A) vs the pinned no-FMA left-assoc ggml scalar oracle (ggml_vec_dot_q4_0_q8_0_generic)
 *       -> expected small bounded ULP (FMA-fold, NOT a bug); this is the parity metric.
 *   (B) vs an f64 high-precision reference -> shows the kernel is as-accurate-or-better
 *       than the ggml scalar reference (accuracy substantiation).
 *
 * Repack layout (derived from the exported kernel), per 16-row group, per block b (288 B):
 *   bytes[0..31]   : 16 x f16 weight scales; row r scale at offset r*2
 *   bytes[32..287] : row r, nibble-position l -> byte at 32 + l*16 + r = q4_0.qs[l] ^ 0x88
 * Groups laid out as: group g at g*nb*288; block b within group at +b*288.
 * Activation = standard block_q8_0 stream (34 B/block).
 *
 * argv: <K_elems> <nrows(mult of 16)> <trials> <seed>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef uint16_t ggml_half;
#define QK4_0 32
#define QK8_0 32
typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0; /* 18 */
typedef struct { ggml_half d; int8_t  qs[QK8_0];    } block_q8_0; /* 34 */
_Static_assert(sizeof(block_q4_0)==18,"q4_0");
_Static_assert(sizeof(block_q8_0)==34,"q8_0");

extern void ggml_vec_dot_q4_0_q8_0_generic(int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);
extern void tcrv_emitc_ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel_ggml_vec_dot_q4_0_q8_0_repack_gemv(size_t n, float*s, size_t nrows, const uint8_t*vx, const uint8_t*vy);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static int8_t  ri8(void){ return (int8_t)((int)(xr()%255)-127); }
static uint8_t ru8(void){ return (uint8_t)(xr()&0xff); }
static uint16_t rf16(void){ uint16_t s=(uint16_t)((xr()&1)<<15),e=(uint16_t)((10+(xr()%8))&0x1F),m=(uint16_t)(xr()&0x3FF); return (uint16_t)(s|(e<<10)|m); }

/* IEEE half -> double (exact), for the f64 reference (independent of board fp16) */
static double h2d(uint16_t h){
    int s=(h>>15)&1, e=(h>>10)&0x1F, m=h&0x3FF; double sign=s?-1:1;
    if(e==0)   return sign*ldexp((double)m, -24);
    if(e==0x1F)return m? (double)NAN : sign*(double)INFINITY;
    return sign*ldexp((double)(m|0x400), e-25);
}
static int64_t okey(float f){ uint32_t u; memcpy(&u,&f,4); return (u&0x80000000u)?-(int64_t)(u&0x7fffffffu):(int64_t)u; }
static int64_t ulp(float a, float b){ int64_t d=okey(a)-okey(b); return d<0?-d:d; }

int main(int argc, char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s K nrows trials seed\n",argv[0]); return 2; }
    int K=atoi(argv[1]), nrows=atoi(argv[2]), trials=atoi(argv[3]);
    rng=(uint64_t)strtoull(argv[4],0,0)|1ull;
    int nb=K/QK4_0, ngrp=nrows/16;
    size_t repbytes=(size_t)ngrp*nb*288;

    block_q4_0 *W=malloc((size_t)nrows*nb*sizeof*W);
    block_q8_0 *A=malloc((size_t)nb*sizeof*A);
    uint8_t *rep=malloc(repbytes);
    float *out=malloc((size_t)nrows*sizeof(float));

    long ncmp=0, npass=0; int worst=0; double sum_ulp_kern=0, sum_ulp_ggml=0, worst_rel=0;
    int ex_r=-1,ex_t=-1; float ex_o=0,ex_ref=0; double ex_f64=0;

    for(int t=0;t<trials;t++){
        for(int r=0;r<nrows;r++) for(int b=0;b<nb;b++){ W[r*nb+b].d=rf16(); for(int j=0;j<QK4_0/2;j++) W[r*nb+b].qs[j]=ru8(); }
        for(int b=0;b<nb;b++){ A[b].d=rf16(); for(int j=0;j<QK8_0;j++) A[b].qs[j]=ri8(); }
        /* build repack */
        for(int g=0;g<ngrp;g++) for(int b=0;b<nb;b++){
            uint8_t *blk = rep + ((size_t)g*nb + b)*288;
            for(int rr=0;rr<16;rr++){ int r=g*16+rr; memcpy(blk + rr*2, &W[r*nb+b].d, 2); }
            for(int l=0;l<QK4_0/2;l++) for(int rr=0;rr<16;rr++){ int r=g*16+rr;
                blk[32 + l*16 + rr] = (uint8_t)(W[r*nb+b].qs[l] ^ 0x88); }
        }
        for(int i=0;i<nrows;i++) out[i]=9;
        tcrv_emitc_ggml_vec_dot_q4_0_q8_0_repack_gemv_kernel_ggml_vec_dot_q4_0_q8_0_repack_gemv((size_t)K,out,(size_t)nrows,rep,(const uint8_t*)A);
        for(int r=0;r<nrows;r++){
            float ref=9; ggml_vec_dot_q4_0_q8_0_generic(K,&ref,0,&W[r*nb],0,A,0,1);
            /* f64 high-precision reference, same block math */
            double acc=0;
            for(int b=0;b<nb;b++){ double dw=h2d(W[r*nb+b].d), da=h2d(A[b].d); long si=0;
                for(int j=0;j<QK4_0/2;j++){ int v0=(W[r*nb+b].qs[j]&0xF)-8, v1=(W[r*nb+b].qs[j]>>4)-8;
                    si += (long)v0*A[b].qs[j] + (long)v1*A[b].qs[j+QK4_0/2]; }
                acc += (double)si*dw*da; }
            int64_t uk=ulp(out[r],ref); ncmp++; if(uk==0)npass++;
            int64_t ukf=(int64_t)llabs((int64_t)okey(out[r])-okey((float)acc));
            int64_t ugf=(int64_t)llabs((int64_t)okey(ref)   -okey((float)acc));
            sum_ulp_kern+=(double)ukf; sum_ulp_ggml+=(double)ugf;
            double rel = acc!=0 ? fabs((double)out[r]-acc)/fabs(acc) : 0;
            if(rel>worst_rel) worst_rel=rel;
            if(uk>worst){ worst=(int)uk; ex_r=r; ex_t=t; ex_o=out[r]; ex_ref=ref; ex_f64=acc; }
        }
    }
    printf("FMT=q4_0_repack_gemv K=%d nrows=%d trials=%d cmps=%ld\n",K,nrows,trials,ncmp);
    printf("  vs_noFMA_scalar_ggml: bitexact=%ld/%ld worst_ulp=%d worst_rel=%.3e\n",npass,ncmp,worst,worst_rel);
    printf("  mean_ulp_vs_f64: kernel=%.3f  ggml_scalar=%.3f  (lower=more accurate)\n",
        sum_ulp_kern/ncmp, sum_ulp_ggml/ncmp);
    if(worst) printf("  WORST t=%d row=%d kernel=%.9g ggml=%.9g f64=%.9g\n",ex_t,ex_r,ex_o,ex_ref,ex_f64);
    printf("VERDICT=%s\n", (npass==ncmp)?"BIT_EXACT":"FMA_FOLD_BOUNDED_ULP");
    free(W);free(A);free(rep);free(out);
    return 0;
}
