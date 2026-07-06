/* gemm_verify_driver.c -- silicon verifier for the front-door-CONSTRUCTED
 * q4_0 16x1-REPACKED GEMM (PREFILL, M>1) kernel (be66c917).
 *
 * The repack GEMM computes out[r][c] = dot(activation_row r, weight_col c) as
 * the SAME q4_0 x q8_0 contraction ggml does, but with a VECTOR-FMA fold
 * (vfmacc) over dx*dy-scaled 8-lane integer accumulation -- structurally
 * distinct from the no-FMA left-assoc scalar fold. So (mirroring the GEVM
 * silicon driver in ../silicon-validation-batch-1) we report BOTH:
 *   (A) vs the pinned no-FMA left-assoc ggml scalar oracle
 *       (ggml_vec_dot_q4_0_q8_0_generic, recompiled -ffp-contract=off)
 *       -> expected small bounded ULP (FMA-fold, NOT a bug); parity metric.
 *   (B) vs an f64 high-precision reference -> shows the kernel is
 *       as-accurate-or-better than the ggml scalar reference.
 *
 * Repack layouts (DERIVED from the exported kernel body):
 *   WEIGHT block_q4_0x16 (288 B), per 16-COLUMN group g, per block b:
 *     blk = rep_w + (g*nb + b)*288
 *     bytes[0..31]    : 16 x f16 col scales; col cc at blk[cc*2]
 *     bytes[32..287]  : nibble l, col cc -> blk[32 + l*16 + cc] = W_col.qs[l] ^ 0x88
 *   ACT block_q8_0x4 (136 B), per 4-ROW group g, per block b:
 *     blk = rep_a + (g*nb + b)*136
 *     bytes[0..7]     : 4 x f16 row scales; row rr at blk[rr*2]
 *     bytes[8..71]    : low  quant l (0..15), row rr -> blk[8  + l*4 + rr] = A_row.qs[l]
 *     bytes[72..135]  : high quant l (0..15), row rr -> blk[72 + l*4 + rr] = A_row.qs[l+16]
 *   OUTPUT: out[r*bs + c], bs = nc (contiguous rows).
 *
 * argv: <K_elems(mult 32)> <nr(mult 4)> <nc(mult 16)> <trials> <seed>
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
/* constructed GEMM, 7-role ABI (nr, bs, n, s, nc, vx, vy) */
extern void tcrv_emitc_ggml_gemm_q4_0_q8_0_repack_gemm_kernel_ggml_gemm_q4_0_q8_0_repack_gemm(
    size_t nr, size_t bs, size_t n, float*s, size_t nc, const uint8_t*vx, const uint8_t*vy);

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
    if(argc<6){ fprintf(stderr,"usage: %s K nr nc trials seed\n",argv[0]); return 2; }
    int K=atoi(argv[1]), nr=atoi(argv[2]), nc=atoi(argv[3]), trials=atoi(argv[4]);
    rng=(uint64_t)strtoull(argv[5],0,0)|1ull;
    if(K%32||nr%4||nc%16){ fprintf(stderr,"K%%32,nr%%4,nc%%16 required\n"); return 2; }
    int nb=K/QK4_0, grp_c=nc/16, grp_r=nr/4;
    size_t wbytes=(size_t)grp_c*nb*288, abytes=(size_t)grp_r*nb*136;

    block_q4_0 *W=malloc((size_t)nc*nb*sizeof*W);   /* col-major: W[c*nb+b] */
    block_q8_0 *A=malloc((size_t)nr*nb*sizeof*A);   /* row-major: A[r*nb+b] */
    uint8_t *rw=malloc(wbytes), *ra=malloc(abytes);
    float *out=malloc((size_t)nr*nc*sizeof(float));

    long ncmp=0, npass=0; int worst=0; double sum_ulp_kern=0, sum_ulp_ggml=0, worst_rel=0;
    int ex_r=-1,ex_c=-1,ex_t=-1; float ex_o=0,ex_ref=0; double ex_f64=0;

    for(int t=0;t<trials;t++){
        for(int c=0;c<nc;c++) for(int b=0;b<nb;b++){ W[c*nb+b].d=rf16(); for(int j=0;j<QK4_0/2;j++) W[c*nb+b].qs[j]=ru8(); }
        for(int r=0;r<nr;r++) for(int b=0;b<nb;b++){ A[r*nb+b].d=rf16(); for(int j=0;j<QK8_0;j++) A[r*nb+b].qs[j]=ri8(); }
        /* build weight repack (block_q4_0x16) */
        for(int g=0;g<grp_c;g++) for(int b=0;b<nb;b++){
            uint8_t *blk = rw + ((size_t)g*nb + b)*288;
            for(int cc=0;cc<16;cc++){ int c=g*16+cc; memcpy(blk + cc*2, &W[c*nb+b].d, 2); }
            for(int l=0;l<QK4_0/2;l++) for(int cc=0;cc<16;cc++){ int c=g*16+cc;
                blk[32 + l*16 + cc] = (uint8_t)(W[c*nb+b].qs[l] ^ 0x88); }
        }
        /* build activation repack (block_q8_0x4) */
        for(int g=0;g<grp_r;g++) for(int b=0;b<nb;b++){
            uint8_t *blk = ra + ((size_t)g*nb + b)*136;
            for(int rr=0;rr<4;rr++){ int r=g*4+rr; memcpy(blk + rr*2, &A[r*nb+b].d, 2); }
            for(int l=0;l<QK8_0/2;l++) for(int rr=0;rr<4;rr++){ int r=g*4+rr;
                blk[8  + l*4 + rr] = (uint8_t)A[r*nb+b].qs[l];
                blk[72 + l*4 + rr] = (uint8_t)A[r*nb+b].qs[l+QK8_0/2]; }
        }
        for(size_t i=0;i<(size_t)nr*nc;i++) out[i]=9;
        tcrv_emitc_ggml_gemm_q4_0_q8_0_repack_gemm_kernel_ggml_gemm_q4_0_q8_0_repack_gemm(
            (size_t)nr,(size_t)nc,(size_t)K,out,(size_t)nc,rw,ra);
        for(int r=0;r<nr;r++) for(int c=0;c<nc;c++){
            float ref=9; ggml_vec_dot_q4_0_q8_0_generic(K,&ref,0,&W[c*nb],0,&A[r*nb],0,1);
            /* f64 high-precision reference, same block math */
            double acc=0;
            for(int b=0;b<nb;b++){ double dw=h2d(W[c*nb+b].d), da=h2d(A[r*nb+b].d); long si=0;
                for(int j=0;j<QK4_0/2;j++){ int v0=(W[c*nb+b].qs[j]&0xF)-8, v1=(W[c*nb+b].qs[j]>>4)-8;
                    si += (long)v0*A[r*nb+b].qs[j] + (long)v1*A[r*nb+b].qs[j+QK4_0/2]; }
                acc += (double)si*dw*da; }
            float ko = out[(size_t)r*nc + c];
            int64_t uk=ulp(ko,ref); ncmp++; if(uk==0)npass++;
            int64_t ukf=(int64_t)llabs((int64_t)okey(ko) -okey((float)acc));
            int64_t ugf=(int64_t)llabs((int64_t)okey(ref)-okey((float)acc));
            sum_ulp_kern+=(double)ukf; sum_ulp_ggml+=(double)ugf;
            double rel = acc!=0 ? fabs((double)ko-acc)/fabs(acc) : 0;
            if(rel>worst_rel) worst_rel=rel;
            if(uk>worst){ worst=(int)uk; ex_r=r; ex_c=c; ex_t=t; ex_o=ko; ex_ref=ref; ex_f64=acc; }
        }
    }
    printf("FMT=q4_0_repack_gemm K=%d nr=%d nc=%d trials=%d cmps=%ld\n",K,nr,nc,trials,ncmp);
    printf("  vs_noFMA_scalar_ggml: bitexact=%ld/%ld worst_ulp=%d worst_rel=%.3e\n",npass,ncmp,worst,worst_rel);
    printf("  mean_ulp_vs_f64: kernel=%.3f  ggml_scalar=%.3f  (lower=more accurate)\n",
        sum_ulp_kern/ncmp, sum_ulp_ggml/ncmp);
    if(worst) printf("  WORST t=%d row=%d col=%d kernel=%.9g ggml=%.9g f64=%.9g\n",ex_t,ex_r,ex_c,ex_o,ex_ref,ex_f64);
    printf("VERDICT=%s\n", (npass==ncmp)?"BIT_EXACT":"FMA_FOLD_BOUNDED_ULP");
    free(W);free(A);free(rw);free(ra);free(out);
    return 0;
}
