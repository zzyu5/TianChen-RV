/* harness_8x8.cpp — G8 §六.3 P1-followup: k1 K-quant GEMM vs REAL native-layout dispatched opponent.
 *
 * Fixes the prior P1 DEFERRAL of q5_K/q6_K 8x8: instead of hand-building the x8 interleave
 * (which gave 0.09-0.13 GMAC/s and was mis-labeled a cache-thrash artifact), we drive ggml's OWN
 * exported repack (`ggml::cpu::repack::repack<block_qX_K,I,N>`) + activation quantizers
 * (`ggml_quantize_mat_q8_K_4x8` for the 8x8/INTER_SIZE=8 path, `_4x1` for the 16x1/INTER_SIZE=1 path)
 * from the STOCK libggml-cpu.so (md5 871169a0), producing the exact native buffers the shipped
 * `ggml_gemm_qX_K_*_q8_K` kernels expect. Correctness is cross-checked against an INDEPENDENT scalar
 * dequant-dot reference (this file's own dequantize_row_* copies of the ggml quant DEFINITION),
 * proving the harness is fed correctly (NOT an artifact) before any timing is trusted.
 *
 * q4_K is a POSITIVE CONTROL for the 16x1 feeding (census established ours q4_K == ggml 16x1, nbad=0).
 * q2_K correctness: OURS q2_K (weft vl=16) is fed the canonical ggml repack<q2_K,1,16> + q8_Kx4(4x1)
 * and compared to the SAME independent scalar reference.
 *
 * [NG-4] kernel-axis micro datapoint. NOT e2e. Stock .so read-only (direct link, no dlopen).
 * argv: <fmt=q5_K|q6_K|q2_K|q4_K> <K> <nr> <nc> <cold_reps> <seed>
 */
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#define QK_K 256
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;

/* ---- plain block structs (exact ggml-common.h field order) ---- */
typedef struct { uint8_t scales[16]; uint8_t qs[64]; ggml_half d; ggml_half dmin; } block_q2_K; /*84*/
typedef struct { ggml_half d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[128]; } block_q4_K; /*144*/
typedef struct { ggml_half d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K; /*176*/
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; ggml_half d; } block_q6_K; /*210*/
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K; /*292*/
static_assert(sizeof(block_q2_K)==84,"q2");
static_assert(sizeof(block_q4_K)==144,"q4");
static_assert(sizeof(block_q5_K)==176,"q5");
static_assert(sizeof(block_q6_K)==210,"q6");
static_assert(sizeof(block_q8_K)==292,"q8");
/* interleaved packed sizes (repack.h): q4_Kx16=2304 q5_Kx8=1408 q6_Kx8=1680 q2_Kx16=1344 q8_Kx4=1168 */
#define SZ_Q8KX4 1168

/* ---- minimal ggml_tensor compatible with ggml.h layout (repack reads type/ne/data) ---- */
#define GGML_MAX_DIMS 4
#define GGML_MAX_OP_PARAMS 64
#define GGML_MAX_SRC 10
#define GGML_MAX_NAME 64
enum ggml_type { GGML_TYPE_F32=0, GGML_TYPE_Q2_K=10, GGML_TYPE_Q4_K=12, GGML_TYPE_Q5_K=13, GGML_TYPE_Q6_K=14, GGML_TYPE_Q8_K=15 };
struct ggml_tensor {
    enum ggml_type type;
    void * buffer;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    int op;
    int32_t op_params[GGML_MAX_OP_PARAMS/sizeof(int32_t)];
    int32_t flags;
    void * src[GGML_MAX_SRC];
    void * view_src;
    size_t view_offs;
    void * data;
    char name[GGML_MAX_NAME];
    void * extra;
    char padding[8];
};

/* ---- STOCK .so exported ggml repack free functions (C++ mangled, bound via asm label) ---- */
extern int ggml_repack_q4k_16x1(struct ggml_tensor*, const void*, size_t)
    asm("_ZN4ggml3cpu6repack6repackI10block_q4_KLl1ELl16EEEiP11ggml_tensorPKvm");
extern int ggml_repack_q5k_8x8(struct ggml_tensor*, const void*, size_t)
    asm("_ZN4ggml3cpu6repack6repackI10block_q5_KLl8ELl8EEEiP11ggml_tensorPKvm");
extern int ggml_repack_q6k_8x8(struct ggml_tensor*, const void*, size_t)
    asm("_ZN4ggml3cpu6repack6repackI10block_q6_KLl8ELl8EEEiP11ggml_tensorPKvm");
extern int ggml_repack_q2k_16x1(struct ggml_tensor*, const void*, size_t)
    asm("_ZN4ggml3cpu6repack6repackI10block_q2_KLl1ELl16EEEiP11ggml_tensorPKvm");

/* ---- STOCK .so C-linkage kernels + quantizers ---- */
extern "C" {
void ggml_gemm_q5_K_8x8_q8_K(int,float*,size_t,const void*,const void*,int,int);
void ggml_gemm_q6_K_8x8_q8_K(int,float*,size_t,const void*,const void*,int,int);
void ggml_gemm_q2_K_16x1_q8_K(int,float*,size_t,const void*,const void*,int,int);
void ggml_gemm_q4_K_16x1_q8_K(int,float*,size_t,const void*,const void*,int,int);
void ggml_quantize_mat_q8_K_4x8(const float*,void*,int64_t);   /* INTER_SIZE=8 (q5/q6 8x8) */
void ggml_quantize_mat_q8_K_4x1(const float*,void*,int64_t);   /* INTER_SIZE=1 (q2/q4 16x1) */
void quantize_row_q8_K(const float*,void*,int64_t);
void quantize_row_q5_K(const float*,void*,int64_t);
void quantize_row_q6_K(const float*,void*,int64_t);
void quantize_row_q2_K(const float*,void*,int64_t);
void quantize_row_q4_K(const float*,void*,int64_t);
void ggml_vec_dot_q5_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q6_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q2_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_q4_K_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
/* OURS weft-exported vl=16 repack-GEMM (n,s,vx,vy,nr,nc,bs) */
void weft_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
void weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
void weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
void weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
}

/* ---- fp16 -> fp32 (IEEE) ---- */
static inline float fp16(ggml_half h){
    uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,out;
    if(e==0){ if(m==0) out=s<<31; else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; out=(s<<31)|(e<<23)|(m<<13); } }
    else if(e==31){ out=(s<<31)|(0xff<<23)|(m<<13); }
    else out=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&out,4); return f;
}

/* ============ INDEPENDENT scalar dequant reference (ggml quant DEFINITION, re-derived) ============ */
static inline void gsm4(int j,const uint8_t*q,uint8_t*d,uint8_t*m){
    if(j<4){*d=q[j]&63;*m=q[j+4]&63;} else {*d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j-0]>>6)<<4);}
}
static void deq_q2_K(const block_q2_K* x, float* y, int nb){
    for(int i=0;i<nb;i++){ float d=fp16(x[i].d), mn=fp16(x[i].dmin); const uint8_t* q=x[i].qs; int is=0; float dl,ml;
        for(int n=0;n<QK_K;n+=128){ int shift=0;
            for(int j=0;j<4;++j){ uint8_t sc=x[i].scales[is++]; dl=d*(sc&0xF); ml=mn*(sc>>4);
                for(int l=0;l<16;++l) *y++=dl*((int8_t)((q[l]>>shift)&3))-ml;
                sc=x[i].scales[is++]; dl=d*(sc&0xF); ml=mn*(sc>>4);
                for(int l=0;l<16;++l) *y++=dl*((int8_t)((q[l+16]>>shift)&3))-ml;
                shift+=2; }
            q+=32; } }
}
static void deq_q4_K(const block_q4_K* x, float* y, int nb){
    for(int i=0;i<nb;i++){ const uint8_t* q=x[i].qs; float d=fp16(x[i].d), mn=fp16(x[i].dmin); int is=0; uint8_t sc,m;
        for(int j=0;j<QK_K;j+=64){ gsm4(is+0,x[i].scales,&sc,&m); float d1=d*sc,m1=mn*m;
            gsm4(is+1,x[i].scales,&sc,&m); float d2=d*sc,m2=mn*m;
            for(int l=0;l<32;++l) *y++=d1*(q[l]&0xF)-m1;
            for(int l=0;l<32;++l) *y++=d2*(q[l]>>4)-m2;
            q+=32; is+=2; } }
}
static void deq_q5_K(const block_q5_K* x, float* y, int nb){
    for(int i=0;i<nb;i++){ const uint8_t* ql=x[i].qs; const uint8_t* qh=x[i].qh;
        float d=fp16(x[i].d), mn=fp16(x[i].dmin); int is=0; uint8_t sc,m,u1=1,u2=2;
        for(int j=0;j<QK_K;j+=64){ gsm4(is+0,x[i].scales,&sc,&m); float d1=d*sc,m1=mn*m;
            gsm4(is+1,x[i].scales,&sc,&m); float d2=d*sc,m2=mn*m;
            for(int l=0;l<32;++l) *y++=d1*((ql[l]&0xF)+((qh[l]&u1)?16:0))-m1;
            for(int l=0;l<32;++l) *y++=d2*((ql[l]>>4)+((qh[l]&u2)?16:0))-m2;
            ql+=32; is+=2; u1<<=2; u2<<=2; } }
}
static void deq_q6_K(const block_q6_K* x, float* y, int nb){
    for(int i=0;i<nb;i++){ float d=fp16(x[i].d); const uint8_t* ql=x[i].ql; const uint8_t* qh=x[i].qh; const int8_t* sc=x[i].scales;
        for(int n=0;n<QK_K;n+=128){ for(int l=0;l<32;++l){ int is=l/16;
                int8_t q1=(int8_t)((ql[l+0]&0xF)|(((qh[l]>>0)&3)<<4))-32;
                int8_t q2=(int8_t)((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
                int8_t q3=(int8_t)((ql[l+0]>>4)|(((qh[l]>>4)&3)<<4))-32;
                int8_t q4=(int8_t)((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
                y[l+0]=d*sc[is+0]*q1; y[l+32]=d*sc[is+2]*q2; y[l+64]=d*sc[is+4]*q3; y[l+96]=d*sc[is+6]*q4; }
            y+=128; ql+=64; qh+=32; sc+=8; } }
}

/* ---- rng / timing / flush ---- */
static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static float frand(void){ return ((float)(xr()&0xffff)/32768.0f - 1.0f)*0.5f; }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:(x>y?1:0); }
#define FLUSH_BYTES (224u*1024u*1024u)
static volatile uint64_t g_sink=0; static uint8_t* g_flush=0;
static void cold_flush(void){ uint64_t s=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]^=(uint8_t)i; s+=g_flush[i]; } g_sink+=s; }

int main(int argc,char**argv){
    if(argc<7){ fprintf(stderr,"usage: %s q5_K|q6_K|q2_K|q4_K K nr nc cold_reps seed\n",argv[0]); return 2; }
    const char* fmt=argv[1];
    int K=atoi(argv[2]),nr=atoi(argv[3]),nc=atoi(argv[4]),reps=atoi(argv[5]);
    rng=(uint64_t)strtoull(argv[6],0,0)|1ull;
    int fi=!strcmp(fmt,"q5_K")?5:!strcmp(fmt,"q6_K")?6:!strcmp(fmt,"q2_K")?2:!strcmp(fmt,"q4_K")?4:0;
    if(!fi){ fprintf(stderr,"bad fmt\n"); return 2; }
    if(K%QK_K||nr%16||nc%16){ fprintf(stderr,"need K%%256, nr%%16, nc%%16\n"); return 2; }
    if(reps<10) reps=10;
    int nb=K/QK_K;
    int is16x1 = (fi==2||fi==4);
    int row_iv = is16x1?16:8;
    size_t plainW=(fi==2)?sizeof(block_q2_K):(fi==4)?sizeof(block_q4_K):(fi==5)?sizeof(block_q5_K):sizeof(block_q6_K);
    size_t packWsz=(fi==2)?1344:(fi==4)?2304:(fi==5)?1408:1680;   /* sizeof(block_qX_KxN) */
    size_t oursWstride=(fi==2)?1344:(fi==4)?2304:(fi==5)?2816:3360; /* ours vl=16 x16 group stride */

    /* ---------- PLAIN weights (nc cols, nb blocks) by quantizing random floats ---------- */
    uint8_t* Wplain=(uint8_t*)aligned_alloc(64,(size_t)nc*nb*plainW);
    float* wf=(float*)malloc(sizeof(float)*K);
    for(int c=0;c<nc;c++){ for(int i=0;i<K;i++) wf[i]=frand(); void* dst=Wplain+(size_t)c*nb*plainW;
        if(fi==2) quantize_row_q2_K(wf,dst,K); else if(fi==4) quantize_row_q4_K(wf,dst,K);
        else if(fi==5) quantize_row_q5_K(wf,dst,K); else quantize_row_q6_K(wf,dst,K); }
    free(wf);

    /* ---------- ggml native repack -> opponent weight buffer ---------- */
    size_t oppWbytes=(size_t)(nc/row_iv)*nb*packWsz;
    uint8_t* Wopp=(uint8_t*)aligned_alloc(64,oppWbytes);
    struct ggml_tensor t; memset(&t,0,sizeof t);
    t.type=(fi==2)?GGML_TYPE_Q2_K:(fi==4)?GGML_TYPE_Q4_K:(fi==5)?GGML_TYPE_Q5_K:GGML_TYPE_Q6_K;
    t.ne[0]=K; t.ne[1]=nc; t.ne[2]=1; t.ne[3]=1; t.data=Wopp;
    size_t din=(size_t)nc*nb*plainW;
    int rc=(fi==2)?ggml_repack_q2k_16x1(&t,Wplain,din):(fi==4)?ggml_repack_q4k_16x1(&t,Wplain,din)
          :(fi==5)?ggml_repack_q5k_8x8(&t,Wplain,din):ggml_repack_q6k_8x8(&t,Wplain,din);
    if(rc!=0){ fprintf(stderr,"REPACK_FAIL rc=%d\n",rc); return 4; }

    /* ---------- activations (nr rows): plain q8_K + interleaved (4x1 for 16x1, 4x8 for 8x8) ---------- */
    float* af=(float*)malloc((size_t)nr*K*sizeof(float));
    for(size_t i=0;i<(size_t)nr*K;i++) af[i]=frand();
    block_q8_K* Aplain=(block_q8_K*)aligned_alloc(64,(size_t)nr*nb*sizeof(block_q8_K));
    for(int r=0;r<nr;r++) quantize_row_q8_K(&af[(size_t)r*K],&Aplain[(size_t)r*nb],K);
    uint8_t* Aopp=(uint8_t*)aligned_alloc(64,(size_t)(nr/4)*nb*SZ_Q8KX4);
    for(int g=0;g<nr/4;g++){ const float* src=&af[(size_t)g*4*K]; void* dst=Aopp+(size_t)g*nb*SZ_Q8KX4;
        if(is16x1) ggml_quantize_mat_q8_K_4x1(src,dst,K); else ggml_quantize_mat_q8_K_4x8(src,dst,K); }
    free(af);

    /* ---------- reference: dequant(weights) . dequant(activation) ---------- */
    float* Wdeq=(float*)malloc((size_t)nc*K*sizeof(float));
    for(int c=0;c<nc;c++){ const void* wb=Wplain+(size_t)c*nb*plainW;
        if(fi==2) deq_q2_K((const block_q2_K*)wb,&Wdeq[(size_t)c*K],nb);
        else if(fi==4) deq_q4_K((const block_q4_K*)wb,&Wdeq[(size_t)c*K],nb);
        else if(fi==5) deq_q5_K((const block_q5_K*)wb,&Wdeq[(size_t)c*K],nb);
        else deq_q6_K((const block_q6_K*)wb,&Wdeq[(size_t)c*K],nb); }
    float* Adeq=(float*)malloc((size_t)nr*K*sizeof(float));
    for(int r=0;r<nr;r++) for(int b=0;b<nb;b++){ float d=Aplain[(size_t)r*nb+b].d;
        const int8_t* qs=Aplain[(size_t)r*nb+b].qs; for(int i=0;i<QK_K;i++) Adeq[(size_t)r*K+b*QK_K+i]=d*qs[i]; }
    float* ref=(float*)malloc((size_t)nr*nc*sizeof(float));
    for(int r=0;r<nr;r++) for(int c=0;c<nc;c++){ double acc=0; const float* wc=&Wdeq[(size_t)c*K]; const float* ar=&Adeq[(size_t)r*K];
        for(int i=0;i<K;i++) acc+=(double)wc[i]*ar[i]; ref[(size_t)r*nc+c]=(float)acc; }

    /* ---------- opponent GEMM (native-fed) ---------- */
    float* Sopp=(float*)aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    #define OPP() do{ if(fi==5) ggml_gemm_q5_K_8x8_q8_K(K,Sopp,(size_t)nc,Wopp,Aopp,nr,nc); \
                      else if(fi==6) ggml_gemm_q6_K_8x8_q8_K(K,Sopp,(size_t)nc,Wopp,Aopp,nr,nc); \
                      else if(fi==4) ggml_gemm_q4_K_16x1_q8_K(K,Sopp,(size_t)nc,Wopp,Aopp,nr,nc); \
                      else ggml_gemm_q2_K_16x1_q8_K(K,Sopp,(size_t)nc,Wopp,Aopp,nr,nc); }while(0)
    OPP();
    auto chk=[&](float* S,int transp,double& mrel,int& nbad){ mrel=0; nbad=0;
        for(int r=0;r<nr;r++)for(int c=0;c<nc;c++){ float v=transp?S[(size_t)c*nr+r]:S[(size_t)r*nc+c]; float g=ref[(size_t)r*nc+c];
            double den=fabs(g)>fabs(v)?fabs(g):fabs(v); double d=fabs((double)v-g); double rel=den>1e-9?d/den:0;
            if(rel>mrel)mrel=rel; if(rel>1e-2)nbad++; } };
    double mrel0,mrel1; int nbad0,nbad1; chk(Sopp,0,mrel0,nbad0); chk(Sopp,1,mrel1,nbad1);
    int transp=(nbad1<nbad0)?1:0; double opp_mrel=transp?mrel1:mrel0; int opp_nbad=transp?nbad1:nbad0;

    /* secondary block-dot cross-check (validates independent ref) */
    double vd_mrel=0; int vd_n=0;
    for(int r=0;r<nr&&r<8;r++)for(int c=0;c<nc&&c<8;c++){ float o; const void* wc=Wplain+(size_t)c*nb*plainW; const void* ar=&Aplain[(size_t)r*nb];
        if(fi==5) ggml_vec_dot_q5_K_q8_K(K,&o,0,wc,0,ar,0,1); else if(fi==6) ggml_vec_dot_q6_K_q8_K(K,&o,0,wc,0,ar,0,1);
        else if(fi==4) ggml_vec_dot_q4_K_q8_K(K,&o,0,wc,0,ar,0,1); else ggml_vec_dot_q2_K_q8_K(K,&o,0,wc,0,ar,0,1);
        float g=ref[(size_t)r*nc+c]; double den=fabs(g)>fabs(o)?fabs(g):fabs(o),d=fabs((double)o-g),rel=den>1e-9?d/den:0; if(rel>vd_mrel)vd_mrel=rel; vd_n++; }

    /* ---------- OURS correctness on canonical 16x1 buffers (q2_K target, q4_K positive control) ---------- */
    double ours_mrel=-1; int ours_nbad=-1, ours_transp=0;
    if(is16x1){
        float* Sc=(float*)aligned_alloc(64,(size_t)nr*nc*sizeof(float));
        if(fi==4) weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K((size_t)K,Sc,Wopp,Aopp,(size_t)nr,(size_t)nc,(size_t)nc);
        else weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K((size_t)K,Sc,Wopp,Aopp,(size_t)nr,(size_t)nc,(size_t)nc);
        double m0,m1; int n0,n1; chk(Sc,0,m0,n0); chk(Sc,1,m1,n1);
        ours_transp=(n1<n0)?1:0; ours_mrel=ours_transp?m1:m0; ours_nbad=ours_transp?n1:n0; free(Sc);
    }

    /* ---------- OURS timing buffers (own x16 layout, random-but-finite; opp-independent) ---------- */
    size_t oursWbytes=(size_t)(nc/16)*nb*oursWstride, oursAbytes=(size_t)(nr/4)*nb*SZ_Q8KX4;
    uint8_t* WoursT=(uint8_t*)aligned_alloc(64,oursWbytes);
    uint8_t* AoursT=(uint8_t*)aligned_alloc(64,oursAbytes);
    float* Sours=(float*)aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    for(size_t i=0;i<oursWbytes;i++) WoursT[i]=(uint8_t)(xr()&0xff);
    for(size_t i=0;i<oursAbytes;i++) AoursT[i]=(uint8_t)(xr()&0xff);
    const uint16_t H=0x2C00;
    for(int g=0;g<nc/16;g++)for(int l=0;l<nb;l++){ uint16_t* p=(uint16_t*)(WoursT+((size_t)g*nb+l)*oursWstride); for(int j=0;j<32;j++) p[j]=H; }
    for(int g=0;g<nr/4;g++)for(int l=0;l<nb;l++){ float* p=(float*)(AoursT+((size_t)g*nb+l)*SZ_Q8KX4); for(int j=0;j<4;j++) p[j]=0.01f; }
    #define OURS() do{ if(fi==5) weft_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K((size_t)K,Sours,WoursT,AoursT,(size_t)nr,(size_t)nc,(size_t)nc); \
                       else if(fi==6) weft_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K((size_t)K,Sours,WoursT,AoursT,(size_t)nr,(size_t)nc,(size_t)nc); \
                       else if(fi==4) weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K((size_t)K,Sours,WoursT,AoursT,(size_t)nr,(size_t)nc,(size_t)nc); \
                       else weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K((size_t)K,Sours,WoursT,AoursT,(size_t)nr,(size_t)nc,(size_t)nc); }while(0)

    double macs=(double)nr*(double)nc*(double)K;
    volatile double sink=0;

    /* ---------- COLD paired A/B ---------- */
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH_BYTES); memset(g_flush,1,FLUSH_BYTES);
    double* to=(double*)malloc(sizeof(double)*reps); double* tp=(double*)malloc(sizeof(double)*reps);
    cold_flush(); OURS(); cold_flush(); OPP();
    for(int p=0;p<reps;p++){
        cold_flush(); double a=now_ns(); OURS(); to[p]=now_ns()-a; sink+=Sours[0]+Sours[(size_t)nr*nc-1];
        cold_flush(); double b=now_ns(); OPP();  tp[p]=now_ns()-b; sink+=Sopp[0]+Sopp[(size_t)nr*nc-1];
    }
    qsort(to,reps,sizeof(double),cmpd); qsort(tp,reps,sizeof(double),cmpd);
    double om=to[reps/2], pm=tp[reps/2], ob=to[0], pb=tp[0];
    double oiqr=om>0?100.0*(to[(3*reps)/4]-to[reps/4])/om:0, piqr=pm>0?100.0*(tp[(3*reps)/4]-tp[reps/4])/pm:0;

    printf("H8 fmt=%s K=%d nr=%d nc=%d reps=%d | "
           "XCHK_ref(opp) max_rel=%.3e nbad=%d/%d transp=%d | vecdot_ref max_rel=%.3e n=%d | ",
           fmt,K,nr,nc,reps, opp_mrel,opp_nbad,nr*nc,transp, vd_mrel,vd_n);
    if(is16x1) printf("OURS_%s max_rel=%.3e nbad=%d/%d transp=%d | ", fmt,ours_mrel,ours_nbad,nr*nc,ours_transp);
    printf("COLD ours_med_ns=%.0f oiqr=%.2f%% opp_med_ns=%.0f piqr=%.2f%% "
           "ours_gmacs=%.4f opp_gmacs=%.4f ratio_med=%.4f ratio_best=%.4f | sink=%.1f\n",
           om,oiqr,pm,piqr, macs/om,macs/pm,(macs/om)/(macs/pm),(macs/ob)/(macs/pb),(double)sink+(double)g_sink);
    return 0;
}
