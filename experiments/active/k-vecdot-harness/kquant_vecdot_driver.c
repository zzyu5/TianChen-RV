/* kquant_vecdot_driver.c — K-quant vec_dot @rvv cold-start harness driver (K 线).
 *
 * op = vec_dot (block_qX_K · block_q8_K -> f32 scalar · HAS fp reduction).
 *
 *   OURS   = owned weft-emitted block-dot kernel (kernels/<fmt>.kernel.c, linked as .o;
 *            KQuant aux32 integer-core; VLEN-invariant emit). Verbatim from weft-translate.
 *   OPP    = the board's REAL DEPLOYED symbol ggml_vec_dot_<fmt>_q8_K, linked from
 *            libggml-cpu.so — NOT reimplemented.  The harness probes the actual board-width
 *            dispatch path and reports its caliber separately: q2/q3/q4/q6 currently reach
 *            hand-tuned `_vl128/_vl256` specializations, while q5 reaches the exported
 *            general-vector body.  A real deployed path is never replaced merely to make
 *            the opponent look stronger or weaker (§对手法 3.4).
 *   ORACLE = libcall-free pure-integer INDEPENDENT recompute (h2f, int64, ZERO-MODEL from
 *            actual input bytes; [K-5], 实验宪法 §1.8 结构无关). A third code path that shares
 *            NO bit->lane decode with either vector kernel.
 *
 * byte-exact ([K-5]): ours == oracle (ZERO-MODEL primary) AND ours == ggml-deployed
 * (deployed-path correctness) AND ggml == oracle. Three independent code paths agreeing
 * = anti-hollow by construction; INJECT arms prove the gate bites.
 *
 * INT-mode bounded fill (super-block d = fp16 1.0, activation qs = +/-1): both sides
 * fold an EXACT integer in fp32 (< 2^24 at K=2048), so byte-exact is a valid
 * structure-independent certificate.  The standard timing corpus keeps dmin=0 for
 * continuity; verify additionally runs a min-active corpus for q2_K/q4_K/q5_K with
 * dmin=1 and non-zero group sums.  The oracle must observe a non-zero min contribution,
 * otherwise that arm fails as hollow ([K-5b], ISSUE-114).
 *
 * This driver and its cell only print to stdout; the canonical bench runner alone owns
 * repository-side immutable-run persistence (ISSUE-090 契约). Main tree + build/ UNTOUCHED.
 *
 * argv: <fmt> <K> <M> <nc> <reps> <seed> <inject> <flush_mb> [min_active]
 *   reps == 0 : VERIFY-only  (byte-exact 3-way + fp16 golden + anti-hollow, NO timing)
 *   reps  > 0 : MEASURE       (verify first, abort on fail, then paired cold timing)
 *   inject 0=clean · 1=corrupt-OURS-out · 2=corrupt-ORACLE-const
 *          3=corrupt-q8-bsums (input-contract negative arm; must fail before evaluation)
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK_K 256
_Static_assert(QK_K==256,"B2 K-quant harness requires the canonical QK_K=256 ABI");

/* ---- OURS: owned weft-emitted K-quant block-dot vec_dot kernels (4-arg) ---- */
extern void weft_emitc_ggml_vec_dot_q2_K_q8_K_kernel_rvv_q2_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_q3_K_q8_K_kernel_rvv_q3_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void weft_emitc_ggml_vec_dot_q6_K_q8_K_kernel_rvv_q6_K_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);

/* ---- OPP: real as-shipped ggml, linked from libggml-cpu.so ---- */
extern void ggml_vec_dot_q2_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q3_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q4_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_q5_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
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
static uint32_t g_q2_values;
static uint32_t g_q3_values;
static uint32_t g_q3_high_bits;
static uint32_t g_q4_values;
static uint32_t g_q5_values;
static uint32_t g_q5_high_bits;
static int g_min_term_nonzero;

/* q2_K block 84B { scales[16]@0 ; qs[64]@16 ; fp16 d@80 ; fp16 dmin@82 } */
static void oracle_q2_K(size_t n, float* s, const uint8_t* vx, const uint8_t* vy){
    const int nb=(int)(n/QK_K);
    float sumf=0.0f;
    for(int ib=0;ib<nb;++ib){
        const uint8_t* xb=vx+(size_t)ib*84;
        const uint8_t* yb=vy+(size_t)ib*292;
        uint16_t dh,dmh; memcpy(&dh,xb+80,2); memcpy(&dmh,xb+82,2);
        float xd=h2f(dh), xdm=h2f(dmh), yd; memcpy(&yd,yb,4);
        const int8_t* q8=(const int8_t*)(yb+4);
        int64_t total=0, smin=0;
        for(int nn=0;nn<2;++nn){
            const uint8_t* q=xb+16+nn*32;
            for(int j=0;j<4;++j){
                const int shift=2*j;
                for(int half=0;half<2;++half){
                    const uint8_t sm=xb[nn*8+2*j+half];
                    const int sc=sm&0x0f, mn=sm>>4;
                    const int base=nn*128+j*32+half*16;
                    for(int l=0;l<16;++l){
                        const int qv=(q[half*16+l]>>shift)&3;
                        g_q2_values|=1u<<qv;
                        total+=(int64_t)sc*qv*(int)q8[base+l];
                        smin+=(int64_t)mn*(int)q8[base+l];
                    }
                }
            }
        }
        if(smin!=0) g_min_term_nonzero=1;
        sumf+=(xd*yd)*(float)total;
        sumf-=(xdm*yd)*(float)smin;
    }
    *s=sumf;
}

/* q3_K block 110B { hmask[32]@0 ; qs[64]@32 ; scales[12]@96 ; fp16 d@108 } */
static void oracle_q3_K(size_t n, float* s, const uint8_t* vx, const uint8_t* vy){
    static const uint32_t kmask1=0x03030303u,kmask2=0x0f0f0f0fu;
    const int nb=(int)(n/QK_K);
    float sumf=0.0f;
    for(int ib=0;ib<nb;++ib){
        const uint8_t* xb=vx+(size_t)ib*110;
        const uint8_t* yb=vy+(size_t)ib*292;
        uint16_t dh; memcpy(&dh,xb+108,2);
        float xd=h2f(dh),yd; memcpy(&yd,yb,4);
        const uint8_t* hm=xb;
        const uint8_t* qs=xb+32;
        const int8_t* q8=(const int8_t*)(yb+4);
        uint32_t aux[4]; memcpy(aux,xb+96,12);
        const uint32_t tmp=aux[2];
        aux[2]=((aux[0]>>4)&kmask2)|(((tmp>>4)&kmask1)<<4);
        aux[3]=((aux[1]>>4)&kmask2)|(((tmp>>6)&kmask1)<<4);
        aux[0]=(aux[0]&kmask2)|(((tmp>>0)&kmask1)<<4);
        aux[1]=(aux[1]&kmask2)|(((tmp>>2)&kmask1)<<4);
        const int8_t* scales=(const int8_t*)aux;
        int64_t total=0;
        for(int nn=0;nn<2;++nn){
            const uint8_t* q=qs+nn*32;
            for(int j=0;j<4;++j){
                const int shift=2*j,mbit=nn*4+j;
                for(int half=0;half<2;++half){
                    const int scale=(int)scales[nn*8+2*j+half]-32;
                    const int base=nn*128+j*32+half*16;
                    for(int l=0;l<16;++l){
                        const int qbits=(q[half*16+l]>>shift)&3;
                        const int high=(hm[half*16+l]>>mbit)&1;
                        const int qv=qbits-(high?0:4);
                        g_q3_values|=1u<<qbits;
                        g_q3_high_bits|=1u<<high;
                        total+=(int64_t)scale*qv*(int)q8[base+l];
                    }
                }
            }
        }
        sumf+=(xd*yd)*(float)total;
    }
    *s=sumf;
}

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
        for(int k=0;k<QK_K;++k){
            g_q4_values|=1u<<(unsigned)a[k];
            total += (int64_t)sc[k/32]*(int)q8[k]*(int)a[k];
        }
        int64_t smin=0;
        for(int g=0;g<QK_K/16;++g){ int16_t b16; memcpy(&b16,bs+2*g,2); smin += (int64_t)b16*(int)mn[g/2]; }
        if(smin!=0) g_min_term_nonzero=1;
        sumf += (xd*yd)*(float)total;
        sumf -= (xdm*yd)*(float)smin;
    }
    *s=sumf;
}

static void get_scale_min_k4(int j,const uint8_t* q,uint8_t* d,uint8_t* m){
    if(j<4){ *d=q[j]&63; *m=q[j+4]&63; return; }
    *d=(q[j+4]&0x0f)|((q[j-4]>>6)<<4);
    *m=(q[j+4]>>4)|((q[j]>>6)<<4);
}

static void pack_scale_min_k4(uint8_t* q,int salt){
    uint8_t sc[8],mn[8];
    memset(q,0,12);
    for(int j=0;j<8;++j){
        sc[j]=(uint8_t)(1+((j+salt)%7));
        mn[j]=(uint8_t)(1+((3*j+salt)%15));
    }
    for(int j=0;j<4;++j){ q[j]=sc[j]; q[j+4]=mn[j]; }
    for(int j=4;j<8;++j){
        q[j+4]=(uint8_t)((sc[j]&0x0f)|((mn[j]&0x0f)<<4));
        q[j-4]=(uint8_t)((q[j-4]&0x3f)|((sc[j]>>4)<<6));
        q[j]=(uint8_t)((q[j]&0x3f)|((mn[j]>>4)<<6));
    }
}

/* q5_K block 176B { fp16 d@0 ; fp16 dmin@2 ; scales[12]@4 ; qh[32]@16 ; qs[128]@48 } */
static void oracle_q5_K(size_t n,float* s,const uint8_t* vx,const uint8_t* vy){
    const int nb=(int)(n/QK_K);
    float sumf=0.0f;
    for(int ib=0;ib<nb;++ib){
        const uint8_t* xb=vx+(size_t)ib*176;
        const uint8_t* yb=vy+(size_t)ib*292;
        uint16_t dh,dmh; memcpy(&dh,xb,2); memcpy(&dmh,xb+2,2);
        float xd=h2f(dh),xdm=h2f(dmh),yd; memcpy(&yd,yb,4);
        const uint8_t* scales=xb+4;
        const uint8_t* qh=xb+16;
        const uint8_t* qs=xb+48;
        const int8_t* q8=(const int8_t*)(yb+4);
        int64_t total=0,smin=0;
        for(int seg=0;seg<4;++seg){
            uint8_t sc0,mn0,sc1,mn1;
            get_scale_min_k4(2*seg+0,scales,&sc0,&mn0);
            get_scale_min_k4(2*seg+1,scales,&sc1,&mn1);
            const uint8_t* q=qs+seg*32;
            for(int l=0;l<32;++l){
                const int high0=(qh[l]>>(2*seg))&1;
                const int qv0=(q[l]&0x0f)+(high0<<4);
                const int i0=seg*64+l;
                g_q5_values|=1u<<(unsigned)(qv0&15);
                g_q5_high_bits|=1u<<high0;
                total+=(int64_t)sc0*qv0*(int)q8[i0];
                smin+=(int64_t)mn0*(int)q8[i0];

                const int high1=(qh[l]>>(2*seg+1))&1;
                const int qv1=(q[l]>>4)+(high1<<4);
                const int i1=seg*64+32+l;
                g_q5_values|=1u<<(unsigned)(qv1&15);
                g_q5_high_bits|=1u<<high1;
                total+=(int64_t)sc1*qv1*(int)q8[i1];
                smin+=(int64_t)mn1*(int)q8[i1];
            }
        }
        if(smin!=0) g_min_term_nonzero=1;
        sumf+=(xd*yd)*(float)total;
        sumf-=(xdm*yd)*(float)smin;
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
enum { F_q2K=0, F_q3K, F_q4K, F_q5K, F_q6K, NF };
static const char* FNAME[NF]={"q2_K","q3_K","q4_K","q5_K","q6_K"};
static const int   WBLK [NF]={84,110,144,176,210};
static const int   ABLK [NF]={292,292,292,292,292};
static const int   W_DOFF[NF]={80,108,0,0,208};  /* fp16 d offset */
static const int   W_MOFF[NF]={82,-1,2,2,-1};    /* fp16 dmin, -1 = absent */

static void our_one(int fmt,size_t K,float* o,const uint8_t* wc,const uint8_t* ar){
    switch(fmt){
    case F_q2K: weft_emitc_ggml_vec_dot_q2_K_q8_K_kernel_rvv_q2_K_q8_K_block_dot(K,o,wc,ar); break;
    case F_q3K: weft_emitc_ggml_vec_dot_q3_K_q8_K_kernel_rvv_q3_K_q8_K_block_dot(K,o,wc,ar); break;
    case F_q4K: weft_emitc_ggml_vec_dot_q4_K_q8_K_kernel_rvv_q4_K_q8_K_block_dot(K,o,wc,ar); break;
    case F_q5K: weft_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(K,o,wc,ar); break;
    case F_q6K: weft_emitc_ggml_vec_dot_q6_K_q8_K_kernel_rvv_q6_K_q8_K_block_dot(K,o,wc,ar); break;
    }
}
static void opp_one(int fmt,int K,float* o,const uint8_t* wc,const uint8_t* ar){
    switch(fmt){
    case F_q2K: ggml_vec_dot_q2_K_q8_K(K,o,0,wc,0,ar,0,1); break;
    case F_q3K: ggml_vec_dot_q3_K_q8_K(K,o,0,wc,0,ar,0,1); break;
    case F_q4K: ggml_vec_dot_q4_K_q8_K(K,o,0,wc,0,ar,0,1); break;
    case F_q5K: ggml_vec_dot_q5_K_q8_K(K,o,0,wc,0,ar,0,1); break;
    case F_q6K: ggml_vec_dot_q6_K_q8_K(K,o,0,wc,0,ar,0,1); break;
    }
}
static void ora_one(int fmt,size_t K,float* o,const uint8_t* wc,const uint8_t* ar){
    switch(fmt){
    case F_q2K: oracle_q2_K(K,o,wc,ar); break;
    case F_q3K: oracle_q3_K(K,o,wc,ar); break;
    case F_q4K: oracle_q4_K(K,o,wc,ar); break;
    case F_q5K: oracle_q5_K(K,o,wc,ar); break;
    case F_q6K: oracle_q6_K(K,o,wc,ar); break;
    }
}

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }
static inline uint32_t f2u(float f){ uint32_t u; memcpy(&u,&f,4); return u; }
static void* alloc64(size_t n){ return aligned_alloc(64,(n+63u)&~(size_t)63u); }

/* INT-mode bounded fill: exact-integer fold => order-independent byte-exact + valid cold. */
static void fill_weight(uint8_t* w,int fmt,int nc,int nb,int min_active){
    size_t tot=(size_t)nc*nb*WBLK[fmt];
    for(size_t i=0;i<tot;i++) w[i]=(uint8_t)xr();
    for(int c=0;c<nc;c++)for(int b=0;b<nb;b++){
        uint8_t* blk=w+((size_t)c*nb+b)*WBLK[fmt];
        uint16_t one=0x3C00,zero=0x0000;
        memcpy(blk+W_DOFF[fmt],&one,2);
        if(W_MOFF[fmt]>=0) memcpy(blk+W_MOFF[fmt],min_active?&one:&zero,2);
        if(min_active && fmt==F_q2K){
            for(int i=0;i<16;++i)
                blk[i]=(uint8_t)((1+((i+c+b)%7))|((1+((3*i+c+b)%15))<<4));
        }
        if(min_active && (fmt==F_q4K || fmt==F_q5K)){
            pack_scale_min_k4(blk+4,c+3*b);
        }
    }
}
static void fill_act(uint8_t* a,int fmt,int M,int nb,int min_active){
    for(int r=0;r<M;r++)for(int b=0;b<nb;b++){
        uint8_t* blk=a+((size_t)r*nb+b)*ABLK[fmt];   /* q8_K: f32 d@0, i8 qs@4, i16 bsums@260 */
        float one=1.0f; memcpy(blk,&one,4);
        int8_t* qs=(int8_t*)(blk+4);
        for(int g=0;g<16;++g)for(int i=0;i<16;++i){
            const int mag=1+(g%7);
            qs[g*16+i]=min_active ? (int8_t)(i==0?-mag:mag)
                                  : (int8_t)((xr()&1)?1:-1);
        }
        for(int g=0;g<16;g++){ int s=0; for(int i=0;i<16;i++) s+=qs[g*16+i]; int16_t v=(int16_t)s; memcpy(blk+260+2*g,&v,2); }
    }
}

static int validate_act_bsums(const uint8_t* a,int fmt,int M,int nb){
    for(int r=0;r<M;++r)for(int b=0;b<nb;++b){
        const uint8_t* blk=a+((size_t)r*nb+b)*ABLK[fmt];
        const int8_t* qs=(const int8_t*)(blk+4);
        for(int g=0;g<16;++g){
            int expected=0; for(int i=0;i<16;++i) expected+=qs[g*16+i];
            int16_t stored; memcpy(&stored,blk+260+2*g,2);
            if(stored!=expected) return 0;
        }
    }
    return 1;
}

static int corpus_ok(int fmt,int min_active){
    int ok=1;
    if(fmt==F_q2K) ok=(g_q2_values&0x0f)==0x0f;
    else if(fmt==F_q3K) ok=((g_q3_values&0x0f)==0x0f)&&((g_q3_high_bits&0x03)==0x03);
    else if(fmt==F_q4K) ok=(g_q4_values&0xffffu)==0xffffu;
    else if(fmt==F_q5K) ok=((g_q5_values&0xffffu)==0xffffu)&&((g_q5_high_bits&0x03)==0x03);
    if(min_active && W_MOFF[fmt]>=0) ok=ok&&g_min_term_nonzero;
    return ok;
}

int main(int argc,char** argv){
    if(argc<9){ fprintf(stderr,"usage: %s fmt K M nc reps seed inject flush_mb [min_active]\n",argv[0]); return 2; }
    const char* fs=argv[1];
    int fmt=-1; for(int i=0;i<NF;i++) if(!strcmp(fs,FNAME[i])) fmt=i;
    if(fmt<0){ fprintf(stderr,"bad fmt %s (driver serves q2_K/q3_K/q4_K/q5_K/q6_K)\n",fs); return 2; }
    int K=atoi(argv[2]),M=atoi(argv[3]),nc=atoi(argv[4]),reps=atoi(argv[5]);
    const unsigned long long INPUT_SEED=strtoull(argv[6],0,0);
    rng=(uint64_t)INPUT_SEED|1ull;
    int INJECT=atoi(argv[7]), FLUSHMB=atoi(argv[8]);
    int MIN_ACTIVE=argc>9?atoi(argv[9]):0;
    if(K<=0 || K%QK_K){ fprintf(stderr,"K must be a positive multiple of %d\n",QK_K); return 2; }
    if(M<=0 || nc<=0 || reps<0 || FLUSHMB<0){ fprintf(stderr,"M/nc must be positive; reps/flush_mb non-negative\n"); return 2; }
    if(INJECT<0 || INJECT>3){ fprintf(stderr,"inject must be 0..3\n"); return 2; }
    if(MIN_ACTIVE<0 || MIN_ACTIVE>1){ fprintf(stderr,"min_active must be 0 or 1\n"); return 2; }
    if(INJECT!=0 && reps!=0){ fprintf(stderr,"fault injection is verify-only\n"); return 2; }
    if(MIN_ACTIVE && W_MOFF[fmt]<0){ fprintf(stderr,"fmt %s has no min term\n",fs); return 2; }
    if(MIN_ACTIVE && reps!=0){ fprintf(stderr,"min_active is a correctness-only arm\n"); return 2; }
    int nb=K/QK_K;
    long vlen=(long)__riscv_vlenb()*8;

    ggml_cpu_init();

    size_t wbytes=(size_t)nc*nb*WBLK[fmt], abytes=(size_t)M*nb*ABLK[fmt];
    uint8_t* W=alloc64(wbytes);
    uint8_t* A=alloc64(abytes);
    float* Our=alloc64((size_t)M*nc*sizeof(float));
    float* Opp=alloc64((size_t)M*nc*sizeof(float));
    float* Ora=alloc64((size_t)M*nc*sizeof(float));
    size_t FB=(size_t)((FLUSHMB>0?FLUSHMB:224))*1024u*1024u;
    uint8_t* Fl=alloc64(FB);
    if(!W||!A||!Our||!Opp||!Ora||!Fl){ fprintf(stderr,"OOM\n"); return 3; }
    fill_weight(W,fmt,nc,nb,MIN_ACTIVE); fill_act(A,fmt,M,nb,MIN_ACTIVE); memset(Fl,1,FB);
    if(INJECT==3){ int16_t bad; memcpy(&bad,A+260,2); ++bad; memcpy(A+260,&bad,2); }
    const int bsum_ok=validate_act_bsums(A,fmt,M,nb);
    printf("# INPUT_CONTRACT q8_K_bsums_match_qs=%s inject=%d -> %s\n",
           bsum_ok?"true":"false",INJECT,bsum_ok?"PASS":(INJECT==3?"BITES-OK":"FAIL"));
    if(INJECT==3){
        free(W);free(A);free(Our);free(Opp);free(Ora);free(Fl);
        return bsum_ok?4:0;
    }
    if(!bsum_ok){
        free(W);free(A);free(Our);free(Opp);free(Ora);free(Fl);
        return 4;
    }

    #define WCOL(c) (W+(size_t)(c)*nb*WBLK[fmt])
    #define AROW(r) (A+(size_t)(r)*nb*ABLK[fmt])
    #define OUR_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) our_one(fmt,(size_t)K,&Our[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)
    #define OPP_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) opp_one(fmt,K,&Opp[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)
    #define ORA_GRID() do{ for(int r=0;r<M;r++)for(int c=0;c<nc;c++) ora_one(fmt,(size_t)K,&Ora[(size_t)r*nc+c],WCOL(c),AROW(r)); }while(0)

    int f16fail=selftest_fp16();

    /* ===== correctness: 3-way byte-exact over the full grid ===== */
    OUR_GRID(); OPP_GRID(); ORA_GRID();
    int counterfactual_ok=1;
    if(MIN_ACTIVE){
        const size_t count=(size_t)M*nc;
        float* active_ours=malloc(count*sizeof(float));
        float* active_ora=malloc(count*sizeof(float));
        if(!active_ours||!active_ora){ fprintf(stderr,"OOM counterfactual\n"); return 3; }
        memcpy(active_ours,Our,count*sizeof(float));
        memcpy(active_ora,Ora,count*sizeof(float));
        const uint16_t zero=0;
        for(int c=0;c<nc;++c)for(int b=0;b<nb;++b)
            memcpy(W+((size_t)c*nb+b)*WBLK[fmt]+W_MOFF[fmt],&zero,2);
        OUR_GRID(); ORA_GRID();
        size_t changed_ours=0,changed_ora=0,nominal_mism=0;
        for(size_t i=0;i<count;++i){
            if(memcmp(&active_ours[i],&Our[i],4)!=0) ++changed_ours;
            if(memcmp(&active_ora[i],&Ora[i],4)!=0) ++changed_ora;
            if(memcmp(&Our[i],&Ora[i],4)!=0) ++nominal_mism;
        }
        counterfactual_ok=changed_ours>0&&changed_ora>0&&nominal_mism==0;
        printf("# MIN_TERM_COUNTERFACTUAL fmt=%s active_vs_dmin0_ours=%zu/%zu oracle=%zu/%zu dmin0_ours_vs_oracle_mism=%zu min_term_nonzero=%s -> %s\n",
               FNAME[fmt],changed_ours,count,changed_ora,count,nominal_mism,
               g_min_term_nonzero?"true":"false",counterfactual_ok?"BITES-OK":"HOLLOW-FAIL");
        memcpy(Our,active_ours,count*sizeof(float));
        memcpy(Ora,active_ora,count*sizeof(float));
        free(active_ours); free(active_ora);
    }
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
    const int corpus=corpus_ok(fmt,MIN_ACTIVE);
    int byte_exact=(m_og==0)&&(m_oi==0)&&(m_gi==0)&&(f16fail==0)&&corpus&&counterfactual_ok;

    printf("# VECDOT_CORRECT fmt=%s op=vec_dot engine=rvv regime=micro-fixed inject=%d min_active=%d seed=0x%llX K=%d M=%d nc=%d nb=%d vlen=%ld\n",
           FNAME[fmt],INJECT,MIN_ACTIVE,INPUT_SEED,K,M,nc,nb,vlen);
    printf("# fp16_primitive_golden=%s (h2f 9/9 textbook IEEE)\n", f16fail?"FAIL":"PASS");
    printf("# result_ours=%.9g result_ggml_deployed=%.9g result_int_oracle=%.9g\n",(double)Our[0],(double)Opp[0],(double)Ora[0]);
    printf("# bits_ours=0x%08x bits_ggml=0x%08x bits_int_oracle=0x%08x worst_ulp=%lld\n",f2u(Our[0]),f2u(Opp[0]),f2u(Ora[0]),(long long)worst);
    printf("# BYTE_EXACT ours_vs_ggml_deployed=%s ours_vs_int_oracle=%s ggml_vs_int_oracle=%s ALL=%s (grid mism og=%lld oi=%lld gi=%lld)\n",
           m_og?"false":"true", m_oi?"false":"true", m_gi?"false":"true", byte_exact?"true":"false", m_og,m_oi,m_gi);
    printf("# CORPUS fmt=%s q2_mask=0x%x q3_mask=0x%x q3_high=0x%x q4_mask=0x%x q5_mask=0x%x q5_high=0x%x min_active=%d min_term_nonzero=%s -> %s\n",
           FNAME[fmt],g_q2_values,g_q3_values,g_q3_high_bits,g_q4_values,
           g_q5_values,g_q5_high_bits,MIN_ACTIVE,g_min_term_nonzero?"true":"false",
           corpus?"COMPLETE":"INCOMPLETE");

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
