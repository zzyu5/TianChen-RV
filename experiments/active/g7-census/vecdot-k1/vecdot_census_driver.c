/* vecdot_census_driver.c — G7 L1 k1-lane: vec_dot (block-dot) cold-start census.
 *
 * ONE binary, ONE process, SAME session/core. Measures HOT (single-buffer warmup+best-of-N)
 * AND COLD (L2-flush before EACH timed region, paired A/B, N reps, best+median+relIQR) for
 * OUR weft-emitted block-dot vec_dot kernel vs the OPPONENT's real as-shipped ggml_vec_dot_*,
 * driven as a GEVM: output[M][nc] where each cell = one vec_dot of length K.
 *   FLAT (q4_0/q4_1/q5_0/q5_1/q8_0): ours emits ggml's own block-dot => parity-by-adoption.
 *   K-quant (q2_K..q6_K):           ours-block-dot (KQuant aux32 int-core) vs ggml-generic/vl256.
 *
 * BOTH sides consume IDENTICAL plain block bytes (no repack) => a fair kernel-axis A/B with the
 * exact same memory footprint. Control flow of block-dot decode is DATA-INDEPENDENT, so the
 * INT-mode bounded fill gives valid steady-state timing AND doubles as the byte-exact certificate:
 * super-block d = fp16 1.0, dmin/min = 0 (min term vanishes), small integer payloads => both sides
 * fold an EXACT integer in fp32 => require ours[k]==opp[k] bit-for-bit (ZERO-MODEL vs independent
 * separately-compiled stock ggml oracle). [NG-4] kernel-axis datapoint, NOT e2e, NOT a sealed Win.
 * Main tree + build/ UNTOUCHED; no git.
 *
 * argv: <fmt> <K(mult blockK)> <M(rows)> <nc(cols)> <hot_iters> <cold_reps> <seed> [verify_only=0]
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

typedef uint16_t ggml_half;
#define QK_K 256

/* ---- OUR weft-emitted block-dot vec_dot kernels ---- */
/* FLAT q4_0/q8_0: ggml-shaped 9-arg (n,s,bs,vx,bx,vy,by,nrc,aux) */
extern void weft_emitc_ggml_vec_dot_q4_0_q8_0_kernel_rvv_q4_0_q8_0_block_dot(size_t,float*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,int32_t,const int32_t*);
extern void weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(size_t,float*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,int32_t,const int32_t*);
/* FLAT q4_1/q5_0/q5_1: compact 5-arg (n,s,vx,vy,aux) */
extern void weft_emitc_ggml_vec_dot_q4_1_q8_1_kernel_rvv_q4_1_q8_1_block_dot(size_t,float*,const uint8_t*,const uint8_t*,const int32_t*);
extern void weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_rvv_q5_0_q8_0_block_dot(size_t,float*,const uint8_t*,const uint8_t*,const int32_t*);
extern void weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_rvv_q5_1_q8_1_block_dot(size_t,float*,const uint8_t*,const uint8_t*,const int32_t*);
/* K-quant: 4-arg (n,s,vx,vy) */
extern void weft_emitc_ggml_vec_dot_q2_K_q8_K_kernel_rvv_q2_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_q3_K_q8_K_kernel_rvv_q3_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_q6_K_q8_K_kernel_rvv_q6_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);

/* ---- OPPONENT: real as-shipped ggml block-dot, linked from libggml-cpu.so ---- */
extern void ggml_vec_dot_q4_0_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q4_1_q8_1(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q5_0_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q5_1_q8_1(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q8_0_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q2_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q3_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q4_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q5_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q6_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

/* L2-flush: k1 has 512KiB L2, NO L3 => streaming 32 MiB (64x L2) guarantees cold */
#define FLUSH_BYTES (32u*1024u*1024u)
static volatile uint64_t g_sink=0;
static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }

/* format id */
enum { F_q40=0,F_q41,F_q50,F_q51,F_q80, F_q2K,F_q3K,F_q4K,F_q5K,F_q6K, NF };
static const char* FNAME[NF]={"q4_0","q4_1","q5_0","q5_1","q8_0","q2_K","q3_K","q4_K","q5_K","q6_K"};
/* weight plain block bytes */
static const int WBLK[NF]={18,20,22,24,34, 84,110,144,176,210};
/* activation plain block bytes: FLAT q8_0=34/q8_1=36 ; K-quant q8_K=292 */
static const int ABLK[NF]={34,36,34,36,34, 292,292,292,292,292};
/* block contraction length: FLAT 32, K-quant 256 */
static const int BLKK[NF]={32,32,32,32,32, 256,256,256,256,256};
/* d/dmin(min) fp16 offsets in weight block; -1 = none */
static const int W_DOFF[NF]={0,0,0,0,0, 80,108,0,0,208};
static const int W_MOFF[NF]={-1,2,-1,2,-1, 82,-1,2,2,-1};   /* dmin/min offset, -1=absent */
/* activation d field: FLAT fp16@0 ; K-quant float@0 */

static void fill_weight(uint8_t* w, int fmt, int nc, int nb){
    size_t tot=(size_t)nc*nb*WBLK[fmt];
    for(size_t i=0;i<tot;i++) w[i]=(uint8_t)(xr()&0x0F);   /* small nibbles/scales => bounded exact int */
    /* stamp d=1.0 (fp16 0x3C00), dmin/min=0 per block */
    for(int c=0;c<nc;c++)for(int b=0;b<nb;b++){
        uint8_t* blk=w+((size_t)c*nb+b)*WBLK[fmt];
        uint16_t one=0x3C00, zero=0x0000;
        memcpy(blk+W_DOFF[fmt],&one,2);
        if(W_MOFF[fmt]>=0) memcpy(blk+W_MOFF[fmt],&zero,2);
    }
}
static void fill_act(uint8_t* a, int fmt, int M, int nb){
    for(int r=0;r<M;r++)for(int b=0;b<nb;b++){
        uint8_t* blk=a+((size_t)r*nb+b)*ABLK[fmt];
        if(ABLK[fmt]==292){ /* q8_K: float d@0, int8 qs@4[256], int16 bsums@260[16] */
            float one=1.0f; memcpy(blk,&one,4);
            int8_t* qs=(int8_t*)(blk+4);
            for(int i=0;i<256;i++) qs[i]=(xr()&1)?1:-1;
            int16_t* bs=(int16_t*)(blk+260);
            for(int g=0;g<16;g++){ int s=0; for(int i=0;i<16;i++) s+=qs[g*16+i]; bs[g]=(int16_t)s; }
        } else { /* q8_0(34): fp16 d@0,int8 qs@2[32] ; q8_1(36): fp16 d@0,s@2,int8 qs@4[32] */
            uint16_t one=0x3C00, zero=0x0000; memcpy(blk,&one,2);
            int qoff = (ABLK[fmt]==36)?4:2;
            if(ABLK[fmt]==36) memcpy(blk+2,&zero,2);        /* q8_1 s field (m term off anyway) */
            int8_t* qs=(int8_t*)(blk+qoff);
            for(int i=0;i<32;i++) qs[i]=(xr()&1)?1:-1;
        }
    }
}

int main(int argc,char**argv){
    if(argc<8){ fprintf(stderr,"usage: %s fmt K M nc hot_iters cold_reps seed [verify_only]\n",argv[0]); return 2; }
    const char* fs=argv[1];
    int fmt=-1; for(int i=0;i<NF;i++) if(!strcmp(fs,FNAME[i])) fmt=i;
    if(fmt<0){ fprintf(stderr,"bad fmt %s\n",fs); return 2; }
    int K=atoi(argv[2]),M=atoi(argv[3]),nc=atoi(argv[4]),hiters=atoi(argv[5]),reps=atoi(argv[6]);
    rng=(uint64_t)strtoull(argv[7],0,0)|1ull;
    int vonly = argc>8?atoi(argv[8]):0;
    int bk=BLKK[fmt];
    if(K%bk){ fprintf(stderr,"K must be mult of %d\n",bk); return 2; }
    if(reps<10) reps=10; if(hiters<1) hiters=1;
    int nb=K/bk;
    long vlen=(long)__riscv_vlenb()*8;

    size_t wbytes=(size_t)nc*nb*WBLK[fmt], abytes=(size_t)M*nb*ABLK[fmt];
    uint8_t* W=aligned_alloc(64,wbytes);
    uint8_t* A=aligned_alloc(64,abytes);
    float* Our=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    float* Opp=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    g_flush=aligned_alloc(64,FLUSH_BYTES);
    if(!W||!A||!Our||!Opp||!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    fill_weight(W,fmt,nc,nb); fill_act(A,fmt,M,nb); memset(g_flush,1,FLUSH_BYTES);

    int32_t aux[4]={0,0,0,0};
    #define WCOL(c) (W+(size_t)(c)*nb*WBLK[fmt])
    #define AROW(r) (A+(size_t)(r)*nb*ABLK[fmt])
    #define OUR_ONE(o,wc,ar) do{ switch(fmt){ \
        case F_q40: weft_emitc_ggml_vec_dot_q4_0_q8_0_kernel_rvv_q4_0_q8_0_block_dot((size_t)K,(o),0,(wc),0,(ar),0,1,aux); break; \
        case F_q80: weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot((size_t)K,(o),0,(wc),0,(ar),0,1,aux); break; \
        case F_q41: weft_emitc_ggml_vec_dot_q4_1_q8_1_kernel_rvv_q4_1_q8_1_block_dot((size_t)K,(o),(wc),(ar),aux); break; \
        case F_q50: weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_rvv_q5_0_q8_0_block_dot((size_t)K,(o),(wc),(ar),aux); break; \
        case F_q51: weft_emitc_ggml_vec_dot_q5_1_q8_1_kernel_rvv_q5_1_q8_1_block_dot((size_t)K,(o),(wc),(ar),aux); break; \
        case F_q2K: weft_emitc_ggml_vec_dot_q2_K_q8_K_kernel_rvv_q2_K_q8_K_block_dot((size_t)K,(o),(wc),(ar)); break; \
        case F_q3K: weft_emitc_ggml_vec_dot_q3_K_q8_K_kernel_rvv_q3_K_q8_K_block_dot((size_t)K,(o),(wc),(ar)); break; \
        case F_q4K: weft_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot((size_t)K,(o),(wc),(ar)); break; \
        case F_q5K: weft_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot((size_t)K,(o),(wc),(ar)); break; \
        default:    weft_emitc_ggml_vec_dot_q6_K_q8_K_kernel_rvv_q6_K_q8_K_block_dot((size_t)K,(o),(wc),(ar)); } }while(0)
    #define OPP_ONE(o,wc,ar) do{ switch(fmt){ \
        case F_q40: ggml_vec_dot_q4_0_q8_0(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_q80: ggml_vec_dot_q8_0_q8_0(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_q41: ggml_vec_dot_q4_1_q8_1(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_q50: ggml_vec_dot_q5_0_q8_0(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_q51: ggml_vec_dot_q5_1_q8_1(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_q2K: ggml_vec_dot_q2_K_q8_K(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_q3K: ggml_vec_dot_q3_K_q8_K(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_q4K: ggml_vec_dot_q4_K_q8_K(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_q5K: ggml_vec_dot_q5_K_q8_K(K,(o),0,(wc),0,(ar),0,1); break; \
        default:    ggml_vec_dot_q6_K_q8_K(K,(o),0,(wc),0,(ar),0,1); } }while(0)
    #define OUR_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) OUR_ONE(&Our[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)
    #define OPP_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) OPP_ONE(&Opp[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)

    /* ================= VERIFY (byte-exact vs stock ggml oracle) ================= */
    OUR_GRID(); OPP_GRID();
    long long mism=0; int64_t worst_ulp=0;
    for(size_t k=0;k<(size_t)M*nc;k++){
        float a=Our[k], b=Opp[k];
        if(memcmp(&a,&b,4)!=0) mism++;
        uint32_t ua,ub; memcpy(&ua,&a,4); memcpy(&ub,&b,4);
        int64_t ka=(ua&0x80000000u)?-(int64_t)(ua&0x7fffffffu):(int64_t)ua;
        int64_t kb=(ub&0x80000000u)?-(int64_t)(ub&0x7fffffffu):(int64_t)ub;
        int64_t u=ka-kb; if(u<0)u=-u; if(u>worst_ulp)worst_ulp=u;
    }
    printf("VERIFY fmt=%s K=%d M=%d nc=%d : int_byte_mismatch=%lld worst_ulp=%lld sample_our=%.6f opp=%.6f => %s\n",
           FNAME[fmt],K,M,nc,mism,(long long)worst_ulp,Our[0],Opp[0], mism? "*** MISMATCH ***":"BYTE-EXACT-vs-stock-ggml OK");
    if(vonly){ free(W);free(A);free(Our);free(Opp);free(g_flush); return mism?1:0; }

    double macs=(double)M*(double)nc*(double)K;

    /* ================= HOT (single buffer, warmup + best-of-N of hiters grids) ============= */
    for(int w=0;w<3;w++){ OUR_GRID(); }
    double ours_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++) OUR_GRID(); double npc=(now_ns()-t0)/(double)hiters; if(npc<ours_hot)ours_hot=npc; }
    for(int w=0;w<2;w++){ OPP_GRID(); }
    double opp_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++) OPP_GRID(); double npc=(now_ns()-t0)/(double)hiters; if(npc<opp_hot)opp_hot=npc; }

    /* ================= COLD (32 MiB flush before EACH timed region; paired; N reps) ========== */
    double* ours=malloc(sizeof(double)*reps);
    double* opp =malloc(sizeof(double)*reps);
    cold_flush(); OUR_GRID(); cold_flush(); OPP_GRID();
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns(); OUR_GRID(); ours[p]=now_ns()-t0;
        cold_flush(); double t1=now_ns(); OPP_GRID(); opp[p] =now_ns()-t1;
    }
    double nf=1e30; for(int p=0;p<reps;p++){ cold_flush(); double t0=now_ns(); double d=now_ns()-t0; if(d<nf)nf=d; }
    double ob=1e30,pb=1e30; for(int p=0;p<reps;p++){ if(ours[p]<ob)ob=ours[p]; if(opp[p]<pb)pb=opp[p]; }
    qsort(ours,reps,sizeof(double),cmp_d); qsort(opp,reps,sizeof(double),cmp_d);
    double om=ours[reps/2], pm=opp[reps/2];
    double o_q1=ours[reps/4], o_q3=ours[(3*reps)/4], p_q1=opp[reps/4], p_q3=opp[(3*reps)/4];
    double o_iqr = om>0?100.0*(o_q3-o_q1)/om:0.0, p_iqr = pm>0?100.0*(p_q3-p_q1)/pm:0.0;

    printf("CENSUS fmt=%s VLEN=%ld K=%d M=%d nc=%d hiters=%d reps=%d nf_ns=%.0f | "
           "HOT ours_ns=%.0f opp_ns=%.0f ratio_hot=%.4f | "
           "COLD ours_med_ns=%.0f o_iqr=%.2f opp_med_ns=%.0f p_iqr=%.2f ratio_cold_med=%.4f ratio_cold_best=%.4f | "
           "byte_mismatch=%lld worst_ulp=%lld\n",
           FNAME[fmt],vlen,K,M,nc,hiters,reps,nf,
           ours_hot,opp_hot,(macs/ours_hot)/(macs/opp_hot),
           om,o_iqr,pm,p_iqr,(macs/om)/(macs/pm),(macs/ob)/(macs/pb),
           mism,(long long)worst_ulp);
    free(W);free(A);free(Our);free(Opp);free(g_flush);free(ours);free(opp);
    return mism?1:0;
}
