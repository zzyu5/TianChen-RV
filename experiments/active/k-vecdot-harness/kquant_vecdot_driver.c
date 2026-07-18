/* kquant_vecdot_driver.c — K-quant vec_dot @rvv cold-start harness driver (K 线).
 *
 * op = vec_dot (block_qX_K · block_q8_K -> f32 scalar · HAS fp reduction).
 *
 *   OURS   = owned weft-emitted block-dot kernel (kernels/<fmt>.kernel.c, linked as .o;
 *            KQuant aux32 integer-core; VLEN-invariant emit). Verbatim from weft-translate.
 *   OPP    = the board's REAL DEPLOYED symbol ggml_vec_dot_<fmt>_q8_K, linked from
 *            libggml-cpu.so — NOT reimplemented. @VLEN128 the dispatch thunk lands on the
 *            hand-tuned `_vl128` specialization (§对手法 3.4 部署事实要件; caliber probed by
 *            the harness objdump = 手调档 STRONG, not the cheap _generic path).
 *   ORACLE = libcall-free pure-integer INDEPENDENT recompute (h2f, int64, ZERO-MODEL from
 *            actual input bytes; [K-5], 实验宪法 §1.8 结构无关). A third code path that shares
 *            NO bit->lane decode with either vector kernel.
 *
 * byte-exact ([K-5]): ours == oracle (ZERO-MODEL primary) AND ours == ggml-deployed
 * (deployed-path correctness) AND ggml == oracle. Three independent code paths agreeing
 * = anti-hollow by construction; INJECT arms prove the gate bites.
 *
 * INT-mode bounded fill (super-block d = fp16 1.0, dmin/min = 0, small nibble payloads,
 * activation qs = +/-1): both sides fold an EXACT integer in fp32 (< 2^24 at K=2048) =>
 * the answer is order-independent => byte-exact is a valid correctness certificate AND
 * gives valid steady-state cold timing (block-dot decode control flow is data-independent).
 * [NG-4] kernel-axis datapoint, NOT e2e, NOT a sealed Win.
 *
 * The harness (tools/bench/cells/vec_dot.sh) owns ALL persistence; this driver only prints
 * to stdout (ISSUE-090 契约). Main tree + build/ UNTOUCHED; no git.
 *
 * argv: <fmt> <K> <M> <nc> <reps> <seed> <inject> <flush_mb>
 *   reps == 0 : VERIFY-only  (byte-exact 3-way + fp16 golden + anti-hollow, NO timing)
 *   reps  > 0 : MEASURE       (verify first, abort on fail, then paired cold timing)
 *   inject 0=clean · 1=corrupt-OURS-out · 2=corrupt-ORACLE-const (verify-mode anti-hollow)
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK_K 256

/* ---- OURS: owned weft-emitted K-quant block-dot vec_dot kernels (4-arg) ---- */
extern void weft_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_q6_K_q8_K_kernel_rvv_q6_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);

/* ---- OPP: real as-shipped ggml, linked from libggml-cpu.so ---- */
extern void ggml_vec_dot_q4_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q6_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
/* ggml fp16 lookup-table init (no-op fast-path on zfh boards, but harmless insurance). */
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
static uint16_t f2h(float x){
    uint32_t u; memcpy(&u,&x,4);
    uint32_t sign=(u>>16)&0x8000u; int32_t exp=(int32_t)((u>>23)&0xFF)-127+15; uint32_t man=u&0x7FFFFFu;
    if(exp<=0){ return (uint16_t)sign; }
    if(exp>=0x1F){ return (uint16_t)(sign|0x7C00u); }
    return (uint16_t)(sign|((uint32_t)exp<<10)|(man>>13));
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
/* q4_K block 144B { fp16 d@0 ; fp16 dmin@2 ; u8 scales[12]@4 ; u8 qs[128]@16 } */
/* q8_K activ 292B { f32 d@0 ; i8 qs[256]@4 ; i16 bsums[16]@260 }               */
static void oracle_q4_K(size_t n, float* s, const uint8_t* vx, const uint8_t* vy){
    const int nb=(int)(n/QK_K);
    static const uint32_t kmask1=0x3f3f3f3fu,kmask2=0x0f0f0f0fu,kmask3=0x03030303u;
    float sumf=0.0f;
    for(int ib=0;ib<nb;++ib){
        const uint8_t* xb=vx+(size_t)ib*144;
        const uint8_t* yb=vy+(size_t)ib*292;
        uint16_t dh,dmh; memcpy(&dh,xb+0,2); memcpy(&dmh,xb+2,2);
        float xd=h2f(dh), xdm=h2f(dmh), yd; memcpy(&yd,yb+0,4);
        const uint8_t* q4=xb+16;
        const int8_t*  q8=(const int8_t*)(yb+4);
        const uint8_t* bs=yb+260;   /* i16 bsums, read via memcpy for alignment safety */
        int8_t a[QK_K]; int8_t* ap=a; const uint8_t* q4p=q4;
        for(int j=0;j<QK_K/64;++j){
            for(int l=0;l<32;++l) ap[l]=(int8_t)(q4p[l]&0xF);      ap+=32;
            for(int l=0;l<32;++l) ap[l]=(int8_t)(q4p[l]>>4);       ap+=32; q4p+=32;
        }
        uint32_t utmp[4]; memcpy(utmp,xb+4,12);
        utmp[3]=((utmp[2]>>4)&kmask2)|(((utmp[1]>>6)&kmask3)<<4);
        uint32_t uaux=utmp[1]&kmask1;
        utmp[1]=(utmp[2]&kmask2)|(((utmp[0]>>6)&kmask3)<<4);
        utmp[2]=uaux; utmp[0]&=kmask1;
        const uint8_t* sc =(const uint8_t*)&utmp[0];
        const uint8_t* mn =(const uint8_t*)&utmp[2];
        int64_t total=0;
        for(int k=0;k<QK_K;++k) total += (int64_t)sc[k/32]*(int)q8[k]*(int)a[k];
        int64_t smin=0;
        for(int g=0;g<QK_K/16;++g){ int16_t b16; memcpy(&b16,bs+2*g,2); smin += (int64_t)b16*(int)mn[g/2]; }
        sumf += (xd*yd)*(float)total;
        sumf -= (xdm*yd)*(float)smin;
    }
    *s=sumf;
}
/* q6_K block 210B { u8 ql[128]@0 ; u8 qh[64]@128 ; i8 scales[16]@192 ; fp16 d@208 } */
static void oracle_q6_K(size_t n, float* s, const uint8_t* vx, const uint8_t* vy){
    const int nb=(int)(n/QK_K);
    float sumf=0.0f;
    for(int ib=0;ib<nb;++ib){
        const uint8_t* xb=vx+(size_t)ib*210;
        const uint8_t* yb=vy+(size_t)ib*292;
        uint16_t dh; memcpy(&dh,xb+208,2);
        float xd=h2f(dh), yd; memcpy(&yd,yb+0,4);
        const uint8_t* ql=xb+0; const uint8_t* qh=xb+128;
        const int8_t*  sc=(const int8_t*)(xb+192);
        const int8_t*  q8=(const int8_t*)(yb+4);
        int8_t a[QK_K]; int8_t* ap=a; const uint8_t* qlp=ql; const uint8_t* qhp=qh;
        for(int j=0;j<QK_K;j+=128){
            for(int l=0;l<32;++l){
                ap[l+ 0]=(int8_t)(((qlp[l+ 0]&0xF)|(((qhp[l]>>0)&3)<<4)))-32;
                ap[l+32]=(int8_t)(((qlp[l+32]&0xF)|(((qhp[l]>>2)&3)<<4)))-32;
                ap[l+64]=(int8_t)(((qlp[l+ 0]>>4 )|(((qhp[l]>>4)&3)<<4)))-32;
                ap[l+96]=(int8_t)(((qlp[l+32]>>4 )|(((qhp[l]>>6)&3)<<4)))-32;
            }
            ap+=128; qlp+=64; qhp+=32;
        }
        int64_t total=0;
        for(int k=0;k<QK_K;++k) total += (int64_t)sc[k/16]*(int)q8[k]*(int)a[k];
        sumf += (xd*yd)*(float)total;
    }
    *s=sumf;
}

/* ======================= format registry ======================= */
enum { F_q4K=0, F_q6K, NF };
static const char* FNAME[NF]={"q4_K","q6_K"};
static const int   WBLK [NF]={144,210};
static const int   ABLK [NF]={292,292};
static const int   W_DOFF[NF]={0,208};   /* fp16 d offset in weight block */
static const int   W_MOFF[NF]={2,-1};    /* fp16 dmin offset, -1 = absent */

static void our_one(int fmt,size_t K,float* o,const uint8_t* wc,const uint8_t* ar){
    if(fmt==F_q4K) weft_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot(K,o,wc,ar);
    else           weft_emitc_ggml_vec_dot_q6_K_q8_K_kernel_rvv_q6_K_q8_K_block_dot(K,o,wc,ar);
}
static void opp_one(int fmt,int K,float* o,const uint8_t* wc,const uint8_t* ar){
    if(fmt==F_q4K) ggml_vec_dot_q4_K_q8_K(K,o,0,wc,0,ar,0,1);
    else           ggml_vec_dot_q6_K_q8_K(K,o,0,wc,0,ar,0,1);
}
static void ora_one(int fmt,size_t K,float* o,const uint8_t* wc,const uint8_t* ar){
    if(fmt==F_q4K) oracle_q4_K(K,o,wc,ar);
    else           oracle_q6_K(K,o,wc,ar);
}

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }
static inline uint32_t f2u(float f){ uint32_t u; memcpy(&u,&f,4); return u; }

/* INT-mode bounded fill: exact-integer fold => order-independent byte-exact + valid cold */
static void fill_weight(uint8_t* w,int fmt,int nc,int nb){
    size_t tot=(size_t)nc*nb*WBLK[fmt];
    for(size_t i=0;i<tot;i++) w[i]=(uint8_t)(xr()&0x0F);
    for(int c=0;c<nc;c++)for(int b=0;b<nb;b++){
        uint8_t* blk=w+((size_t)c*nb+b)*WBLK[fmt];
        uint16_t one=0x3C00,zero=0x0000;
        memcpy(blk+W_DOFF[fmt],&one,2);
        if(W_MOFF[fmt]>=0) memcpy(blk+W_MOFF[fmt],&zero,2);
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
    if(fmt<0){ fprintf(stderr,"bad fmt %s (driver serves q4_K/q6_K)\n",fs); return 2; }
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
    OUR_GRID(); OPP_GRID(); ORA_GRID();
    if(INJECT==1) Our[0]+=1.0f;     /* DUT-output fault: byte-exact MUST flip */
    if(INJECT==2) Ora[0]+=1.0f;     /* oracle-constant fault: cross-check MUST catch */
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
    printf("# VECDOT_COLD fmt=%s seed=0x%llX reps=%d | ours_med_ns=%.0f(iqr%.2f%%) oppX_med_ns=%.0f(iqr%.2f%%) | ratio_cold_X=%.4f | opp=ggml_vec_dot_%s_q8_K(deployed) flush=%dMiB fsink=%llu\n",
           FNAME[fmt],(unsigned long long)strtoull(argv[6],0,0),reps,om,o_iqr,gm,g_iqr,gm/om,FNAME[fmt],(FLUSHMB>0?FLUSHMB:224),(unsigned long long)fsink);
    free(W);free(A);free(Our);free(Opp);free(Ora);free(Fl);free(to);free(tg);
    return 0;
}
