/* iq2_vecdot_driver.c — iq2 GRID-codebook vec_dot @rvv cold-start harness (B线 block2 P2).
 *
 * op = vec_dot (block_iq2_{xxs,xs} · block_q8_K -> f32 scalar · grid-codebook · M=1 · HAS reduction).
 *
 *   OURS   = owned weft-emitted iq2 grid vec_dot leaf (kernels/<fmt>.kernel.c, linked as .o).
 *            Reproduces the opponent's OWN readable structure: __riscv_vluxei16 grid-decode over
 *            the 256/512-entry iq2 codebook + tiny __riscv_vwredsum reductions. Verbatim from
 *            weft-opt --materialize-<fmt>-block-dot-source-front-door --weft-rvv-lower-to-emitc.
 *            [公式墙对位: owned C-intrinsic vs opponent C-intrinsic — NOT inline-asm.]
 *   OPP    = the board's REAL DEPLOYED symbol ggml_vec_dot_iq2_<fmt>_q8_K, linked from
 *            libggml-cpu.so — NOT reimplemented. @VLEN128 the dispatch thunk lands on the
 *            hand-tuned `_vl128` specialization (tier=手调 STRONG, not the cheap `_generic`).
 *   ORACLE = libcall-free pure-integer INDEPENDENT recompute (h2f, int64, ZERO-MODEL from the
 *            actual input bytes; scalar grid idx/sign/scale decode that shares NO bit->lane
 *            decode with either vector kernel; canonical ggml codebook constants in
 *            iq2_oracle_tables.h — the sign fold is a per-bit ksigns test, NOT the emitter's
 *            expanded signs64 plane). [K-5], 实验宪法 §1.8 结构无关.
 *
 * byte-exact ([K-5]): ours == oracle (ZERO-MODEL primary) AND ours == ggml-deployed
 * (deployed-path correctness) AND ggml == oracle. Three independent code paths agreeing
 * = anti-hollow by construction; INJECT arms prove the gate bites.
 *
 * INT-mode bounded fill (weight super-block d = fp16 1.0, activation q8 = +/-1): the grid
 * bytes, per-lane signs, and per-sub-block ls scales are all integers, so both sides fold an
 * EXACT integer in fp32 (|bsum| < 2^24 at K<=2048) and *s = 0.125f*sumf is an exact power-of-2
 * scale => the answer is order-independent => byte-exact is a valid correctness certificate AND
 * gives valid steady-state cold timing (grid-decode control flow is data-independent).
 * [NG-4] kernel-axis M=1 datapoint, NOT e2e, NOT a sealed Win.
 *
 * The build/run script owns ALL persistence; this driver only prints to stdout (ISSUE-090 契约).
 * Main tree + build/ UNTOUCHED; no git.
 *
 * argv: <fmt> <K> <M> <nc> <reps> <seed> <inject> <flush_mb>
 *   reps == 0 : VERIFY-only  (byte-exact 3-way + fp16 golden + anti-hollow, NO timing)
 *   reps  > 0 : MEASURE       (verify first, abort on fail, then paired cold timing)
 *   inject 0=clean · 1=corrupt-OURS-out · 2=corrupt-ORACLE-out · 3=corrupt-DEPLOYED-ggml-out (3-arm anti-hollow)
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
#include "iq2_oracle_tables.h"

#define QK_K 256

/* ---- OURS: owned weft-emitted iq2 grid vec_dot leaves (4-arg block-dot ABI) ---- */
extern void weft_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);

/* ---- OPP: real as-shipped ggml, linked from libggml-cpu.so ---- */
extern void ggml_vec_dot_iq2_xxs_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_iq2_xs_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_cpu_init(void);

/* ---- pure-integer fp16 -> fp32 (independent of ggml; golden self-tested) ---- */
static float h2f(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000u)<<16;
    uint32_t exp =(h>>10)&0x1Fu;
    uint32_t man = h&0x3FFu;
    uint32_t f;
    if(exp==0){
        if(man==0){ f=sign; }
        else { exp=127u-15u+1u; while(!(man&0x400u)){ man<<=1; exp--; } man&=0x3FFu; f=sign|(exp<<23)|(man<<13); }
    } else if(exp==0x1Fu){ f=sign|0x7F800000u|(man<<13); }
    else { f=sign|((exp-15u+127u)<<23)|(man<<13); }
    float r; memcpy(&r,&f,4); return r;
}
static int selftest_fp16(void){
    struct G { uint16_t b; float v; };
    const struct G g[] = {
        {0x3C00,1.0f},{0xBC00,-1.0f},{0x3800,0.5f},{0x4000,2.0f},{0x0000,0.0f},
        {0x3555,0.333251953125f},{0x0400,6.103515625e-05f},
        {0x0001,5.9604644775390625e-08f},{0x7BFF,65504.0f},
    };
    int fails=0;
    for(unsigned i=0;i<sizeof(g)/sizeof(g[0]);++i){ float r=h2f(g[i].b); if(memcmp(&r,&g[i].v,4)!=0) fails++; }
    return fails;
}

/* ======================= INDEPENDENT ZERO-MODEL ORACLES ======================= */
/* iq2_xxs block 66B { fp16 d@0 ; uint16 qs[32]@2 } . iq2_xs block 74B { fp16 d@0 ;   */
/* uint16 qs[32]@2 ; uint8 scales[8]@66 } . q8_K activ 292B { f32 d@0 ; i8 qs[256]@4 ; */
/* i16 bsums[16]@260 } (iq2 vec_dot IGNORES bsums).                                    */
static inline int8_t gridbyte(int64_t entry, int j){ return (int8_t)((uint64_t)entry >> (8*j)); }

static void oracle_iq2_xxs(size_t n, float* s, const uint8_t* vx, const uint8_t* vy){
    const int nb=(int)(n/QK_K);
    float sumf=0.0f;
    for(int ib=0;ib<nb;++ib){
        const uint8_t* xb=vx+(size_t)ib*66;
        const uint8_t* yb=vy+(size_t)ib*292;
        uint16_t dh; memcpy(&dh,xb+0,2);
        float xd=h2f(dh), yd; memcpy(&yd,yb+0,4);
        uint16_t qs[32]; memcpy(qs,xb+2,64);
        const int8_t* q8=(const int8_t*)(yb+4);
        int q8pos=0; int64_t bsum=0;
        for(int ib32=0;ib32<8;++ib32){
            uint32_t a0=(uint32_t)qs[4*ib32] | ((uint32_t)qs[4*ib32+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib32+2] | ((uint32_t)qs[4*ib32+3]<<16);
            uint8_t aux8[4]={(uint8_t)a0,(uint8_t)(a0>>8),(uint8_t)(a0>>16),(uint8_t)(a0>>24)};
            int64_t ls=2*(a1>>28)+1;
            int64_t sumi=0;
            for(int l=0;l<4;++l){
                int idx=aux8[l];
                int64_t g=iq2xxs_grid[idx];
                uint8_t signs=ksigns_iq2xs[(a1>>(7*l))&127];
                for(int j=0;j<8;++j){
                    int sgn=(signs&(1u<<j))?-1:1;
                    sumi += (int64_t)q8[q8pos++]*gridbyte(g,j)*sgn;
                }
            }
            bsum += sumi*ls;
        }
        sumf += (xd*yd)*(float)bsum;
    }
    *s=0.125f*sumf;
}
static void oracle_iq2_xs(size_t n, float* s, const uint8_t* vx, const uint8_t* vy){
    const int nb=(int)(n/QK_K);
    float sumf=0.0f;
    for(int ib=0;ib<nb;++ib){
        const uint8_t* xb=vx+(size_t)ib*74;
        const uint8_t* yb=vy+(size_t)ib*292;
        uint16_t dh; memcpy(&dh,xb+0,2);
        float xd=h2f(dh), yd; memcpy(&yd,yb+0,4);
        uint16_t qs[32]; memcpy(qs,xb+2,64);
        const uint8_t* sc=xb+66;
        const int8_t* q8=(const int8_t*)(yb+4);
        int q8pos=0; int64_t bsum=0;
        for(int ib32=0;ib32<8;++ib32){
            int64_t ls1=2*(sc[ib32]&0xf)+1;
            int64_t ls2=2*(sc[ib32]>>4)+1;
            int64_t sumi=0;
            for(int l=0;l<2;++l){                 /* groups 0-1 weighted by ls1 */
                uint16_t w=qs[4*ib32+l];
                int idx=w&511; int64_t g=iq2xs_grid[idx];
                uint8_t signs=ksigns_iq2xs[w>>9];
                for(int j=0;j<8;++j){ int sgn=(signs&(1u<<j))?-1:1; sumi += (int64_t)q8[q8pos++]*gridbyte(g,j)*sgn; }
            }
            bsum += sumi*ls1;
            sumi=0;
            for(int l=2;l<4;++l){                 /* groups 2-3 weighted by ls2 */
                uint16_t w=qs[4*ib32+l];
                int idx=w&511; int64_t g=iq2xs_grid[idx];
                uint8_t signs=ksigns_iq2xs[w>>9];
                for(int j=0;j<8;++j){ int sgn=(signs&(1u<<j))?-1:1; sumi += (int64_t)q8[q8pos++]*gridbyte(g,j)*sgn; }
            }
            bsum += sumi*ls2;
        }
        sumf += (xd*yd)*(float)bsum;
    }
    *s=0.125f*sumf;
}

/* ======================= format registry ======================= */
enum { F_iq2xxs=0, F_iq2xs, NF };
static const char* FNAME[NF]={"iq2_xxs","iq2_xs"};
static const int   WBLK [NF]={66,74};
static const int   ABLK [NF]={292,292};

static void our_one(int fmt,size_t K,float* o,const uint8_t* wc,const uint8_t* ar){
    if(fmt==F_iq2xxs) weft_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_rvv_iq2_xxs_q8_K_block_dot(K,o,wc,ar);
    else              weft_emitc_ggml_vec_dot_iq2_xs_q8_K_kernel_rvv_iq2_xs_q8_K_block_dot(K,o,wc,ar);
}
static void opp_one(int fmt,int K,float* o,const uint8_t* wc,const uint8_t* ar){
    if(fmt==F_iq2xxs) ggml_vec_dot_iq2_xxs_q8_K(K,o,0,wc,0,ar,0,1);
    else              ggml_vec_dot_iq2_xs_q8_K(K,o,0,wc,0,ar,0,1);
}
static void ora_one(int fmt,size_t K,float* o,const uint8_t* wc,const uint8_t* ar){
    if(fmt==F_iq2xxs) oracle_iq2_xxs(K,o,wc,ar);
    else              oracle_iq2_xs(K,o,wc,ar);
}

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }
static inline uint32_t f2u(float f){ uint32_t u; memcpy(&u,&f,4); return u; }

/* INT-mode bounded fill: exact-integer fold => order-independent byte-exact + valid cold.
 * Weight bytes are fully random (grid idx / sign sel / scale nibbles all valid by construction);
 * only the fp16 super-block d is pinned to 1.0 so the fp32 scale is exact. */
static void fill_weight(uint8_t* w,int fmt,int nc,int nb){
    size_t tot=(size_t)nc*nb*WBLK[fmt];
    for(size_t i=0;i<tot;i++) w[i]=(uint8_t)(xr()&0xFF);
    for(int c=0;c<nc;c++)for(int b=0;b<nb;b++){
        uint8_t* blk=w+((size_t)c*nb+b)*WBLK[fmt];
        uint16_t one=0x3C00; memcpy(blk+0,&one,2);  /* fp16 d = 1.0 */
    }
}
static void fill_act(uint8_t* a,int fmt,int M,int nb){
    for(int r=0;r<M;r++)for(int b=0;b<nb;b++){
        uint8_t* blk=a+((size_t)r*nb+b)*ABLK[fmt];   /* q8_K: f32 d@0, i8 qs@4, i16 bsums@260 */
        float one=1.0f; memcpy(blk,&one,4);
        int8_t* qs=(int8_t*)(blk+4);
        for(int i=0;i<256;i++) qs[i]=(xr()&1)?1:-1;
        for(int g=0;g<16;g++){ int s=0; for(int i=0;i<16;i++) s+=qs[g*16+i]; int16_t v=(int16_t)s; memcpy(blk+260+2*g,&v,2); }
    }
}

int main(int argc,char** argv){
    if(argc<9){ fprintf(stderr,"usage: %s fmt K M nc reps seed inject flush_mb\n",argv[0]); return 2; }
    const char* fs=argv[1];
    int fmt=-1; for(int i=0;i<NF;i++) if(!strcmp(fs,FNAME[i])) fmt=i;
    if(fmt<0){ fprintf(stderr,"bad fmt %s (driver serves iq2_xxs/iq2_xs)\n",fs); return 2; }
    int K=atoi(argv[2]),M=atoi(argv[3]),nc=atoi(argv[4]),reps=atoi(argv[5]);
    rng=(uint64_t)strtoull(argv[6],0,0)|1ull;
    int INJECT=atoi(argv[7]), FLUSHMB=atoi(argv[8]);
    if(K%QK_K){ fprintf(stderr,"K must be mult of %d\n",QK_K); return 2; }
    int nb=K/QK_K;
    long vlen=(long)__riscv_vlenb()*8;

    ggml_cpu_init();

    size_t wbytes=(size_t)nc*nb*WBLK[fmt], abytes=(size_t)M*nb*ABLK[fmt];
    uint8_t* W=aligned_alloc(64,wbytes);
    uint8_t* A=aligned_alloc(64,abytes);
    float* Our=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    float* Opp=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    float* Ora=aligned_alloc(64,(size_t)M*nc*sizeof(float));
    size_t FB=(size_t)((FLUSHMB>0?FLUSHMB:224))*1024u*1024u;
    uint8_t* Fl=aligned_alloc(64,FB);
    if(!W||!A||!Our||!Opp||!Ora||!Fl){ fprintf(stderr,"OOM\n"); return 3; }
    fill_weight(W,fmt,nc,nb); fill_act(A,fmt,M,nb); memset(Fl,1,FB);

    #define WCOL(c) (W+(size_t)(c)*nb*WBLK[fmt])
    #define AROW(r) (A+(size_t)(r)*nb*ABLK[fmt])
    #define OUR_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) our_one(fmt,(size_t)K,&Our[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)
    #define OPP_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) opp_one(fmt,K,&Opp[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)
    #define ORA_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) ora_one(fmt,(size_t)K,&Ora[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)

    int f16fail=selftest_fp16();

    /* ===== correctness: 3-way byte-exact over the full grid ===== */
    /* 3 anti-hollow arms, one per INDEPENDENT path's OUTPUT (each proves the corresponding
     * pairwise comparison is live -> the gate is not hollow). A shared-INPUT corruption is
     * deliberately NOT used: all three paths read the same bytes, so it changes all three
     * consistently and (correctly) does not break agreement -- it tests input sensitivity,
     * not gate hollowness. */
    OUR_GRID(); OPP_GRID(); ORA_GRID();
    if(INJECT==1) Our[0]+=1.0f;     /* DUT (ours) fault: ours-vs-{ggml,oracle} MUST flip */
    if(INJECT==2) Ora[0]+=1.0f;     /* oracle fault: {ours,ggml}-vs-oracle MUST flip */
    if(INJECT==3) Opp[0]+=1.0f;     /* deployed-ggml (leaf) fault: {ours,oracle}-vs-ggml MUST flip */
    long long m_og=0,m_oi=0,m_gi=0; int64_t worst=0;
    for(size_t k=0;k<(size_t)M*nc;k++){
        uint32_t uo=f2u(Our[k]),ug=f2u(Opp[k]),ui=f2u(Ora[k]);
        if(uo!=ug) m_og++; if(uo!=ui) m_oi++; if(ug!=ui) m_gi++;
        int64_t ko=(uo&0x80000000u)?-(int64_t)(uo&0x7fffffffu):(int64_t)uo;
        int64_t ki=(ui&0x80000000u)?-(int64_t)(ui&0x7fffffffu):(int64_t)ui;
        int64_t d=ko-ki; if(d<0)d=-d; if(d>worst)worst=d;
    }
    int byte_exact=(m_og==0)&&(m_oi==0)&&(m_gi==0)&&(f16fail==0);

    printf("# VECDOT_CORRECT fmt=%s op=vec_dot engine=rvv regime=decode inject=%d seed=0x%llX K=%d M=%d nc=%d nb=%d vlen=%ld\n",
           FNAME[fmt],INJECT,(unsigned long long)(rng&0xffffffffull),K,M,nc,nb,vlen);
    printf("# fp16_primitive_golden=%s (h2f 9/9 textbook IEEE)\n", f16fail?"FAIL":"PASS");
    printf("# result_ours=%.9g result_ggml_deployed=%.9g result_int_oracle=%.9g\n",(double)Our[0],(double)Opp[0],(double)Ora[0]);
    printf("# bits_ours=0x%08x bits_ggml=0x%08x bits_int_oracle=0x%08x worst_ulp=%lld\n",f2u(Our[0]),f2u(Opp[0]),f2u(Ora[0]),(long long)worst);
    printf("# BYTE_EXACT ours_vs_ggml_deployed=%s ours_vs_int_oracle=%s ggml_vs_int_oracle=%s ALL=%s (grid mism og=%lld oi=%lld gi=%lld)\n",
           m_og?"false":"true", m_oi?"false":"true", m_gi?"false":"true", byte_exact?"true":"false", m_og,m_oi,m_gi);

    if(INJECT!=0){
        printf("# ANTIHOLLOW inject=%d expect_byte_exact=false observed=%s -> %s\n",
               INJECT, byte_exact?"true":"false", byte_exact?"HOLLOW-FAIL":"BITES-OK");
        free(W);free(A);free(Our);free(Opp);free(Ora);free(Fl);
        return byte_exact?3:0;   /* fault that did NOT break byte-exact = hollow gate */
    }
    if(reps==0){ free(W);free(A);free(Our);free(Opp);free(Ora);free(Fl); return byte_exact?0:1; }
    if(!byte_exact){ printf("# CORRECTNESS FAIL - aborting before timing\n"); free(W);free(A);free(Our);free(Opp);free(Ora);free(Fl); return 2; }

    /* ===== cold timing (flush > LLC before EACH paired region; N reps; median+relIQR) ===== */
    volatile uint64_t fsink=0;
    #define FLUSH() do{ uint64_t s=0; for(size_t i=0;i<FB;i+=64){ Fl[i]^=(uint8_t)i; s+=Fl[i]; } fsink+=s; }while(0)
    double* to=malloc(sizeof(double)*reps);
    double* tg=malloc(sizeof(double)*reps);
    FLUSH(); OUR_GRID(); FLUSH(); OPP_GRID();          /* warm the code, cold the data */
    for(int p=0;p<reps;p++){
        FLUSH(); double a0=now_ns(); OUR_GRID(); to[p]=now_ns()-a0;
        FLUSH(); double b0=now_ns(); OPP_GRID(); tg[p]=now_ns()-b0;
    }
    qsort(to,reps,sizeof(double),cmp_d); qsort(tg,reps,sizeof(double),cmp_d);
    double om=to[reps/2], gm=tg[reps/2];
    double o_iqr=om>0?100.0*(to[(3*reps)/4]-to[reps/4])/om:0.0;
    double g_iqr=gm>0?100.0*(tg[(3*reps)/4]-tg[reps/4])/gm:0.0;
    printf("# VECDOT_COLD fmt=%s seed=0x%llX reps=%d | ours_med_ns=%.0f(iqr%.2f%%) oppX_med_ns=%.0f(iqr%.2f%%) | ratio_cold_X=%.4f | opp=ggml_vec_dot_%s_q8_K(deployed _vl128) flush=%dMiB fsink=%llu\n",
           FNAME[fmt],(unsigned long long)strtoull(argv[6],0,0),reps,om,o_iqr,gm,g_iqr,gm/om,FNAME[fmt],(FLUSHMB>0?FLUSHMB:224),(unsigned long long)fsink);
    free(W);free(A);free(Our);free(Opp);free(Ora);free(Fl);free(to);free(tg);
    return 0;
}
