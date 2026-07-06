/* bd_verify_driver_iq23.c -- silicon bit-exact verifier for the front-door-
 * CONSTRUCTED super-block grid block-dot kernels iq3_xxs and iq2_xxs
 * (silicon-validation batch-2, closes the 2 remaining flip debts).
 *
 * Method (identical discipline to silicon-validation-batch-1):
 *   - Generate N independent random VALID quantized block streams; the SAME raw
 *     bytes are fed to BOTH the kernel-under-test and the oracle, so this isolates
 *     the decode+dot arithmetic (no quantizer in the loop). Random bytes are valid
 *     for these two formats: qs bytes index the 256-entry grid tables and the
 *     7-bit sign fields index ksigns_iq2xs[128] -- every value is in range.
 *   - Kernel-under-test = the exported RISC-V C from
 *       tcrv-opt <front-door-materialize> --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
 *   - Oracle = ggml's OWN generic scalar reference ggml_vec_dot_<fmt>_generic
 *     (compiled from the board's ggml-cpu/quants.c at -ffp-contract=off = the
 *     pinned no-FMA left-associative ggml scalar oracle).
 *   - Bit-for-bit compare of s[0]; report worst-ULP + N-of-N.
 *
 * argv: <fmt: iq3_xxs|iq2_xxs> <n_elems> <trials> <seed>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t ggml_half;
#define QK8_0  32
#define QK_K   256

/* ---- ggml block layouts (natural contiguous == ggml on-disk layout) ---- */
typedef struct { ggml_half d; uint16_t qs[QK_K/8]; }   block_iq2_xxs; /* 66 */
typedef struct { ggml_half d; uint8_t  qs[3*QK_K/8]; } block_iq3_xxs; /* 98 */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K; /* 292 */

_Static_assert(sizeof(block_iq2_xxs)==66, "iq2_xxs");
_Static_assert(sizeof(block_iq3_xxs)==98, "iq3_xxs");
_Static_assert(sizeof(block_q8_K)==292,   "q8_K");

/* ---- oracle: ggml generic scalar references (from quants.c, -ffp-contract=off) ---- */
extern void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);
extern void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float*s, size_t bs, const void*vx, size_t bx, const void*vy, size_t by, int nrc);

/* ---- kernel-under-test (exported constructed C) ---- */
extern void tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot(size_t n, float*s, const uint8_t*vx, const uint8_t*vy);
extern void tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_rvv_iq3_xxs_q8_K_block_dot(size_t n, float*s, const uint8_t*vx, const uint8_t*vy);

/* ---- rng + helpers (identical to batch-1) ---- */
static uint64_t rng;
static uint32_t xr(void){ rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; return (uint32_t)(rng>>32); }
static int8_t   ri8(void){ return (int8_t)((int)(xr()%255) - 127); }
static uint8_t  ru8(void){ return (uint8_t)(xr()&0xff); }
/* finite, moderate f16: exponent in [10..17] -> magnitude ~ [2^-5, 2^2] */
static uint16_t rf16(void){
    uint16_t s = (uint16_t)((xr()&1)<<15);
    uint16_t e = (uint16_t)((10 + (xr()%8)) & 0x1F);
    uint16_t m = (uint16_t)(xr()&0x3FF);
    return (uint16_t)(s|(e<<10)|m);
}
static float rf32(float lo, float hi){ return lo + (hi-lo)*((float)(xr()&0xffffff)/(float)0x1000000); }

/* ---- ULP distance on ordered-int mapping (bit-exact <=> 0) ---- */
static int64_t okey(float f){ uint32_t u; memcpy(&u,&f,4); return (u&0x80000000u)? -(int64_t)(u&0x7fffffffu) : (int64_t)u; }
static int64_t ulp(float a, float b){ int64_t d=okey(a)-okey(b); return d<0?-d:d; }

static void fill_q8_K(block_q8_K*y, int nb){
    for(int i=0;i<nb;i++){
        y[i].d = rf32(0.001f, 0.05f);
        for(int j=0;j<QK_K;j++) y[i].qs[j]=ri8();
        for(int g=0; g<QK_K/16; g++){ int s=0; for(int j=0;j<16;j++) s+=y[i].qs[g*16+j]; y[i].bsums[g]=(int16_t)s; }
    }
}

int main(int argc, char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s fmt n trials seed\n",argv[0]); return 2; }
    const char*fmt=argv[1];
    int n=atoi(argv[2]), trials=atoi(argv[3]);
    rng = (uint64_t)strtoull(argv[4],0,0) | 1ull;

    int worst=0, npass=0; float ex_o=0,ex_r=0; int ex_t=-1;

    if(!strcmp(fmt,"iq2_xxs")){
        int nb=n/QK_K;
        block_iq2_xxs*x=malloc(nb*sizeof*x); block_q8_K*y=malloc(nb*sizeof*y);
        for(int t=0;t<trials;t++){
            for(int i=0;i<nb;i++){ x[i].d=rf16(); for(int j=0;j<QK_K/8;j++) x[i].qs[j]=(uint16_t)(xr()&0xffff); }
            fill_q8_K(y,nb);
            float so=9,sr=9;
            tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot((size_t)n,&so,(const uint8_t*)x,(const uint8_t*)y);
            ggml_vec_dot_iq2_xxs_q8_K_generic(n,&sr,0,x,0,y,0,1);
            int64_t u=ulp(so,sr); if(u==0)npass++; if(u>worst){worst=(int)u;ex_o=so;ex_r=sr;ex_t=t;}
        }
        free(x);free(y);
    } else if(!strcmp(fmt,"iq3_xxs")){
        int nb=n/QK_K;
        block_iq3_xxs*x=malloc(nb*sizeof*x); block_q8_K*y=malloc(nb*sizeof*y);
        for(int t=0;t<trials;t++){
            for(int i=0;i<nb;i++){ x[i].d=rf16(); for(int j=0;j<3*QK_K/8;j++) x[i].qs[j]=ru8(); }
            fill_q8_K(y,nb);
            float so=9,sr=9;
            tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_rvv_iq3_xxs_q8_K_block_dot((size_t)n,&so,(const uint8_t*)x,(const uint8_t*)y);
            ggml_vec_dot_iq3_xxs_q8_K_generic(n,&sr,0,x,0,y,0,1);
            int64_t u=ulp(so,sr); if(u==0)npass++; if(u>worst){worst=(int)u;ex_o=so;ex_r=sr;ex_t=t;}
        }
        free(x);free(y);
    } else { fprintf(stderr,"unknown fmt %s\n",fmt); return 2; }

    printf("FMT=%s n=%d trials=%d bitexact=%d/%d worst_ulp=%d\n", fmt,n,trials,npass,trials,worst);
    if(worst){ printf("WORST trial=%d ours=%.9g ref=%.9g (bits ours=0x%08x ref=0x%08x)\n",
        ex_t,ex_o,ex_r,*(uint32_t*)&ex_o,*(uint32_t*)&ex_r); }
    printf("VERDICT=%s\n", (npass==trials)?"BIT_EXACT":"MISMATCH");
    return (npass==trials)?0:1;
}
