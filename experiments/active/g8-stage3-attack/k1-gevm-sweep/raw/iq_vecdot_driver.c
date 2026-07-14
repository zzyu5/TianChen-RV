/* iqfp4_vecdot_driver.c — G7 L1 k1-lane: iq/fp4 codebook-gather vec_dot cold-start census.
 *
 * ONE binary, ONE process, SAME session/core. HOT (warmup+best-of-N) + COLD (L2-flush before EACH
 * timed region, paired A/B, N reps, median+relIQR) for OUR weft-emitted grid/codebook block-dot
 * vec_dot vs the OPPONENT's real as-shipped ggml_vec_dot_<fmt>, driven as a GEVM: output[M][nc],
 * each cell = one vec_dot of length K.
 *   iq1_s / iq1_m : super-block grid-gather (QK_K=256) vs q8_K activation (292B).
 *   iq4_nl        : 16-entry integer codebook (QK4_NL=32) vs q8_0 activation (34B).
 *   nvfp4         : E2M1 4-bit fp codebook, UE4M3 sub-scales (QK_NVFP4=64) vs q8_0 activation.
 *
 * BOTH sides consume IDENTICAL plain block bytes (no repack) => fair kernel-axis A/B, same footprint.
 * Control flow of the decode is DATA-INDEPENDENT, so the bounded fill gives valid steady-state timing
 * AND doubles as the byte-exact certificate: weight scale d = fp16 1.0 (or UE4M3 1.0 for nvfp4),
 * activation q8 payload = +-1 => decoded products are exact-representable (integers / multiples of
 * 0.5 / 0.125) => require ours[k]==opp[k] bit-for-bit vs the independent separately-compiled stock
 * ggml oracle (ZERO-MODEL). iq1_m has no explicit d field (scale reconstructed from scattered bits)
 * => reported honestly (mismatch/worst_ulp). [NG-4] kernel-axis datapoint, NOT e2e, NOT a sealed Win.
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

/* ---- OUR weft-emitted grid/codebook block-dot vec_dot kernels ---- */
extern void weft_emitc_ggml_vec_dot_iq1_s_q8_K_kernel_rvv_iq1_s_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_iq1_m_q8_K_kernel_rvv_iq1_m_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot(size_t,float*,const uint8_t*,const uint8_t*,const int32_t*);
extern void weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot(size_t,float*,const uint8_t*,const uint8_t*);

/* ---- OPPONENT: real as-shipped ggml vec_dot, linked from libggml-cpu.so ---- */
extern void ggml_vec_dot_iq1_s_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_iq1_m_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_iq4_nl_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_nvfp4_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);

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
enum { F_iq1s=0,F_iq1m,F_iq4nl,F_nvfp4, NF };
static const char* FNAME[NF]={"iq1_s","iq1_m","iq4_nl","nvfp4"};
/* weight plain block bytes / weight block K */
static const int WBLK[NF]={50,56,18,36};
static const int KW[NF]  ={256,256,32,64};
/* activation plain block bytes / activation block K : q8_K=292/256 ; q8_0=34/32 */
static const int ABLK[NF]={292,292,34,34};
static const int KA[NF]  ={256,256,32,32};

static void fill_weight(uint8_t* w, int fmt, int nc, int nbw){
    size_t tot=(size_t)nc*nbw*WBLK[fmt];
    for(size_t i=0;i<tot;i++) w[i]=(uint8_t)(xr()&0x0F);   /* small grid indices / nibbles => bounded */
    for(int c=0;c<nc;c++)for(int b=0;b<nbw;b++){
        uint8_t* blk=w+((size_t)c*nbw+b)*WBLK[fmt];
        uint16_t one=0x3C00;
        if(fmt==F_iq1s){ memcpy(blk+0,&one,2); }               /* iq1_s d@0 = fp16 1.0 */
        else if(fmt==F_iq4nl){ memcpy(blk+0,&one,2); }         /* iq4_nl d@0 = fp16 1.0 */
        else if(fmt==F_nvfp4){ blk[0]=0x38; blk[1]=0x38; blk[2]=0x38; blk[3]=0x38; } /* UE4M3 ~1.0 sub-scales */
        /* iq1_m: no explicit d field; scale reconstructed from scattered scales/qh bits => leave small */
    }
}
static void fill_act(uint8_t* a, int fmt, int M, int nba){
    for(int r=0;r<M;r++)for(int b=0;b<nba;b++){
        uint8_t* blk=a+((size_t)r*nba+b)*ABLK[fmt];
        if(ABLK[fmt]==292){ /* q8_K: float d@0, int8 qs@4[256], int16 bsums@260[16] */
            float one=1.0f; memcpy(blk,&one,4);
            int8_t* qs=(int8_t*)(blk+4);
            for(int i=0;i<256;i++) qs[i]=(xr()&1)?1:-1;
            int16_t* bs=(int16_t*)(blk+260);
            for(int g=0;g<16;g++){ int s=0; for(int i=0;i<16;i++) s+=qs[g*16+i]; bs[g]=(int16_t)s; }
        } else { /* q8_0(34): fp16 d@0, int8 qs@2[32] */
            uint16_t one=0x3C00; memcpy(blk,&one,2);
            int8_t* qs=(int8_t*)(blk+2);
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
    if(K%KW[fmt]){ fprintf(stderr,"K must be mult of %d\n",KW[fmt]); return 2; }
    if(reps<10) reps=10; if(hiters<1) hiters=1;
    int nbw=K/KW[fmt], nba=K/KA[fmt];
    long vlen=(long)__riscv_vlenb()*8;

    size_t wbytes=(size_t)nc*nbw*WBLK[fmt], abytes=(size_t)M*nba*ABLK[fmt];
    uint8_t* W=aligned_alloc(64,wbytes);
    uint8_t* A=aligned_alloc(64,abytes);
    float* Our=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    float* Opp=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    g_flush=aligned_alloc(64,FLUSH_BYTES);
    if(!W||!A||!Our||!Opp||!g_flush){ fprintf(stderr,"OOM\n"); return 3; }
    fill_weight(W,fmt,nc,nbw); fill_act(A,fmt,M,nba); memset(g_flush,1,FLUSH_BYTES);

    int32_t aux[4]={0,0,0,0};
    #define WCOL(c) (W+(size_t)(c)*nbw*WBLK[fmt])
    #define AROW(r) (A+(size_t)(r)*nba*ABLK[fmt])
    #define OUR_ONE(o,wc,ar) do{ switch(fmt){ \
        case F_iq1s:  weft_emitc_ggml_vec_dot_iq1_s_q8_K_kernel_rvv_iq1_s_q8_K_block_dot((size_t)K,(o),(wc),(ar)); break; \
        case F_iq1m:  weft_emitc_ggml_vec_dot_iq1_m_q8_K_kernel_rvv_iq1_m_q8_K_block_dot((size_t)K,(o),(wc),(ar)); break; \
        case F_iq4nl: weft_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_rvv_iq4_nl_q8_0_block_dot((size_t)K,(o),(wc),(ar),aux); break; \
        default:      weft_emitc_ggml_vec_dot_nvfp4_q8_0_kernel_rvv_nvfp4_q8_0_block_dot((size_t)K,(o),(wc),(ar)); } }while(0)
    #define OPP_ONE(o,wc,ar) do{ switch(fmt){ \
        case F_iq1s:  ggml_vec_dot_iq1_s_q8_K(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_iq1m:  ggml_vec_dot_iq1_m_q8_K(K,(o),0,(wc),0,(ar),0,1); break; \
        case F_iq4nl: ggml_vec_dot_iq4_nl_q8_0(K,(o),0,(wc),0,(ar),0,1); break; \
        default:      ggml_vec_dot_nvfp4_q8_0(K,(o),0,(wc),0,(ar),0,1); } }while(0)
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

    /* ================= HOT (single buffer, warmup + best-of-N) ============= */
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
