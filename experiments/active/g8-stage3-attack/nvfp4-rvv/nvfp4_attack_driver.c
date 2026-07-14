/* nvfp4_attack_driver.c — G8 §六.3 nvfp4 rvv 攻坚 G1/G2 driver.
 * Derived from g7-census/iqfp4-dequant-rvv/iqfp4_vecdot_driver.c (nvfp4-only slice),
 * SAME cold protocol (224MiB flush, paired A/B, N reps, median+relIQR), SAME shapes
 * (K=2048 default, M=1/8, nc=512) as the sealed §六 cold-remeasure so numbers are
 * directly comparable to summary_0p8_rvv.csv row `fp4-vecdot,nvfp4,...,0.661,0.565,0.662`.
 *
 * THREE candidates measured pairwise against the SAME opponent grid each rep:
 *   BASE   = weft_emitc_..._block_dot            (original emitted kernel, unmodified)
 *   ATTACK = weft_emitc_..._block_dot_ATTACK      (re-roll + hoist-scale construction attempt)
 *   OPP    = ggml_vec_dot_nvfp4_q8_0              (real dispatch: generic, no riscv arch path)
 *
 * G1 gate: ATTACK integer-bit-exact vs BASE (DYNAMIC S(new)==baseline) AND vs OPP oracle.
 * argv: <K> <M> <nc> <hot_iters> <cold_reps> <seed>
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

extern void weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot_ATTACK(size_t,float*,const uint8_t*,const uint8_t*);
extern void ggml_vec_dot_nvfp4_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }

#define FLUSH_BYTES (224u*1024u*1024u)
static volatile uint64_t g_sink=0;
static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }

/* nvfp4: WBLK=36 (4 UE4M3 scale bytes + 32 nibble bytes, K=64) ; q8_0 activation ABLK=34,KA=32 */
#define WBLK 36
#define KW 64
#define ABLK 34
#define KA 32

static void fill_weight(uint8_t* w, int nc, int nbw){
    size_t tot=(size_t)nc*nbw*WBLK;
    for(size_t i=0;i<tot;i++) w[i]=(uint8_t)(xr()&0x0F);
    for(int c=0;c<nc;c++)for(int b=0;b<nbw;b++){
        uint8_t* blk=w+((size_t)c*nbw+b)*WBLK;
        blk[0]=0x38; blk[1]=0x38; blk[2]=0x38; blk[3]=0x38; /* UE4M3 ~1.0 sub-scales, matches sealed harness */
    }
}
static void fill_act(uint8_t* a, int M, int nba){
    for(int r=0;r<M;r++)for(int b=0;b<nba;b++){
        uint8_t* blk=a+((size_t)r*nba+b)*ABLK;
        uint16_t one=0x3C00; memcpy(blk,&one,2);
        int8_t* qs=(int8_t*)(blk+2);
        for(int i=0;i<32;i++) qs[i]=(xr()&1)?1:-1;
    }
}

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):2048, M=argc>2?atoi(argv[2]):1, nc=argc>3?atoi(argv[3]):512;
    int hiters=argc>4?atoi(argv[4]):8, reps=argc>5?atoi(argv[5]):12;
    rng=(argc>6?strtoull(argv[6],0,0):12345ull)|1ull;
    if(K%KW){ fprintf(stderr,"K must be mult of %d\n",KW); return 2; }
    if(reps<10) reps=10; if(hiters<1) hiters=1;
    int nbw=K/KW, nba=K/KA;
    long vlen=(long)__riscv_vlenb()*8;

    size_t wbytes=(size_t)nc*nbw*WBLK, abytes=(size_t)M*nba*ABLK;
    uint8_t* W=aligned_alloc(64,wbytes);
    uint8_t* A=aligned_alloc(64,abytes);
    float* Base=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    float* Attack=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    float* Opp=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    g_flush=aligned_alloc(64,FLUSH_BYTES);
    if(!W||!A||!Base||!Attack||!Opp||!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    fill_weight(W,nc,nbw); fill_act(A,M,nba); memset(g_flush,1,FLUSH_BYTES);

    #define WCOL(c) (W+(size_t)(c)*nbw*WBLK)
    #define AROW(r) (A+(size_t)(r)*nba*ABLK)
    #define BASE_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot((size_t)K,&Base[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)
    #define ATTACK_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot_ATTACK((size_t)K,&Attack[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)
    #define OPP_GRID()   do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) ggml_vec_dot_nvfp4_q8_0(K,&Opp[(size_t)r*nc+c],0,WCOL(c),0,AROW(r),0,1); }while(0)

    /* ============ G1: VERIFY (int-bit-exact) ATTACK vs BASE (DYNAMIC S(new)==baseline) AND vs OPP oracle ============ */
    BASE_GRID(); ATTACK_GRID(); OPP_GRID();
    long long mism_ab=0, mism_ao=0, mism_bo=0;
    for(size_t k=0;k<(size_t)M*nc;k++){
        if(memcmp(&Attack[k],&Base[k],4)!=0) mism_ab++;
        if(memcmp(&Attack[k],&Opp[k],4)!=0) mism_ao++;
        if(memcmp(&Base[k],&Opp[k],4)!=0) mism_bo++;
    }
    printf("G1 fmt=nvfp4 K=%d M=%d nc=%d : ATTACK-vs-BASE int_byte_mismatch=%lld | ATTACK-vs-OPP int_byte_mismatch=%lld | BASE-vs-OPP int_byte_mismatch=%lld | sample base=%.6f attack=%.6f opp=%.6f => %s\n",
           K,M,nc,mism_ab,mism_ao,mism_bo,Base[0],Attack[0],Opp[0],
           (mism_ab==0 && mism_ao==0) ? "G1-GREEN(byte-exact)" : "*** G1 MISMATCH ***");

    double macs=(double)M*(double)nc*(double)K;

    /* ============ HOT ============ */
    for(int w=0;w<3;w++){ BASE_GRID(); }
    double base_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++) BASE_GRID(); double npc=(now_ns()-t0)/(double)hiters; if(npc<base_hot)base_hot=npc; }
    for(int w=0;w<3;w++){ ATTACK_GRID(); }
    double attack_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++) ATTACK_GRID(); double npc=(now_ns()-t0)/(double)hiters; if(npc<attack_hot)attack_hot=npc; }
    for(int w=0;w<2;w++){ OPP_GRID(); }
    double opp_hot=1e30;
    for(int p=0;p<8;p++){ double t0=now_ns(); for(int it=0;it<hiters;it++) OPP_GRID(); double npc=(now_ns()-t0)/(double)hiters; if(npc<opp_hot)opp_hot=npc; }

    /* ============ COLD (paired, N reps) — BASE,ATTACK,OPP each round to control drift ============ */
    double* base=malloc(sizeof(double)*reps);
    double* atk =malloc(sizeof(double)*reps);
    double* opp =malloc(sizeof(double)*reps);
    cold_flush(); BASE_GRID(); cold_flush(); ATTACK_GRID(); cold_flush(); OPP_GRID();
    for(int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns(); BASE_GRID();   base[p]=now_ns()-t0;
        cold_flush(); double t1=now_ns(); ATTACK_GRID(); atk[p] =now_ns()-t1;
        cold_flush(); double t2=now_ns(); OPP_GRID();    opp[p] =now_ns()-t2;
    }
    qsort(base,reps,sizeof(double),cmp_d); qsort(atk,reps,sizeof(double),cmp_d); qsort(opp,reps,sizeof(double),cmp_d);
    double bm=base[reps/2], am=atk[reps/2], pm=opp[reps/2];
    double b_q1=base[reps/4], b_q3=base[(3*reps)/4], a_q1=atk[reps/4], a_q3=atk[(3*reps)/4], p_q1=opp[reps/4], p_q3=opp[(3*reps)/4];
    double b_iqr=bm>0?100.0*(b_q3-b_q1)/bm:0.0, a_iqr=am>0?100.0*(a_q3-a_q1)/am:0.0, p_iqr=pm>0?100.0*(p_q3-p_q1)/pm:0.0;

    printf("G2 fmt=nvfp4 VLEN=%ld K=%d M=%d nc=%d hiters=%d reps=%d | "
           "HOT base_ns=%.0f attack_ns=%.0f opp_ns=%.0f ratio_hot_base=%.4f ratio_hot_attack=%.4f | "
           "COLD base_med_ns=%.0f b_iqr=%.2f attack_med_ns=%.0f a_iqr=%.2f opp_med_ns=%.0f p_iqr=%.2f "
           "ratio_cold_base=%.4f ratio_cold_attack=%.4f attack_over_base=%.4f\n",
           vlen,K,M,nc,hiters,reps,
           base_hot,attack_hot,opp_hot,(macs/base_hot)/(macs/opp_hot),(macs/attack_hot)/(macs/opp_hot),
           bm,b_iqr,am,a_iqr,pm,p_iqr,
           (macs/bm)/(macs/pm),(macs/am)/(macs/pm),bm/am);

    free(W);free(A);free(Base);free(Attack);free(Opp);free(g_flush);free(base);free(atk);free(opp);
    return (mism_ab||mism_ao)?1:0;
}
