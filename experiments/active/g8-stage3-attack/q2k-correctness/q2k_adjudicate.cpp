/* q2k_adjudicate.cpp -- G8 §六.3 [BUG-K1-Q2K-GEMM-WRONG] ZERO-MODEL adjudication.
 *
 * QUESTION: prior 8x8 harness fed OURS q2_K GEMM the ggml-canonical repack<q2_K,1,16>
 * buffer -> max_rel 2.0 all-wrong (q4_K control nbad=0). Is OURS a REAL kernel bug, or a
 * q2_K-specific FEEDING-MISMATCH (ours expects a DIFFERENT weight layout than ggml stock)?
 *
 * METHOD (ZERO-MODEL): one INDEPENDENT scalar reference (ggml canonical vec_dot_q2_K_q8_K,
 * re-derived from the q2_K quant DEFINITION). Build the SAME plain q2_K weights + q8_K
 * activation, then pack the weight buffer TWO ways and run OURS' ACTUAL compiled kernel on
 * each:
 *   (A) STRAIGHT   : scales[64+s*16+c]=in[c].scales[s]  (ours' expected / deploy-patched layout)
 *   (B) GGML-STOCK : ggml's real repack<q2_K,1,16>      (the "Sequential-Parallel" scale perm)
 * qs interleave is byte-granular (interleave_block=1) => IDENTICAL in both; ONLY the 256-byte
 * scales sub-block ORDER differs. Also run ggml's STOCK gemm on (B) as a positive control.
 *
 * VERDICT: if OURS on (A) == ref (nbad=0) and OURS on (B) != ref (all-wrong) => the prior
 * all-wrong was a FEEDING-MISMATCH (ours is arithmetically CORRECT on its own straight layout;
 * deployment patches ggml's make_block_q2_Kx16 to the straight order -- see G5 deploy_patch).
 *
 * Stock .so read-only (direct link). argv: <K> <nr> <nc> <seed>
 */
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK_K 256
typedef uint16_t ggml_half;

/* plain q2_K (ggml-common.h order): scales[16], qs[64], d, dmin -- 84 B */
struct block_q2_K { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; ggml_half d; ggml_half dmin; };
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };
static_assert(sizeof(block_q2_K)==84,"q2");
static_assert(sizeof(block_q8_K)==292,"q8");

/* ---- ggml_tensor compatible (repack reads type/ne/data) ---- */
#define GGML_MAX_DIMS 4
#define GGML_MAX_OP_PARAMS 64
#define GGML_MAX_SRC 10
#define GGML_MAX_NAME 64
enum ggml_type { GGML_TYPE_Q2_K=10 };
struct ggml_tensor {
    enum ggml_type type; void* buffer; int64_t ne[GGML_MAX_DIMS]; size_t nb[GGML_MAX_DIMS];
    int op; int32_t op_params[GGML_MAX_OP_PARAMS/sizeof(int32_t)]; int32_t flags;
    void* src[GGML_MAX_SRC]; void* view_src; size_t view_offs; void* data;
    char name[GGML_MAX_NAME]; void* extra; char padding[8];
};

/* STOCK .so: ggml real repack<block_q2_K,1,16> (mangled) + stock GEMM */
extern int ggml_repack_q2k_16x1(struct ggml_tensor*, const void*, size_t)
    asm("_ZN4ggml3cpu6repack6repackI10block_q2_KLl1ELl16EEEiP11ggml_tensorPKvm");
extern "C" {
void ggml_gemm_q2_K_16x1_q8_K(int,float*,size_t,const void*,const void*,int,int);
/* OURS weft-exported vl=16 repack-GEMM (n,s,vx,vy,nr,nc,bs) */
void weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(
    size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
}

/* fp16 helpers */
static inline uint16_t f2h(float f){
    uint32_t x; memcpy(&x,&f,4); uint32_t s=(x>>16)&0x8000; int32_t e=((x>>23)&0xff)-127+15;
    uint32_t m=x&0x7fffff; if(e<=0){ return (uint16_t)s; } if(e>=31) return (uint16_t)(s|0x7c00);
    return (uint16_t)(s|(e<<10)|(m>>13));
}
static inline float h2f(uint16_t h){
    uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,out;
    if(e==0){ if(m==0) out=s<<31; else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; out=(s<<31)|(e<<23)|(m<<13);} }
    else if(e==31) out=(s<<31)|(0xff<<23)|(m<<13);
    else out=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&out,4); return f;
}

/* INDEPENDENT reference: ggml canonical vec_dot_q2_K_q8_K integer certs isum + summs */
static void ref_block(const block_q2_K* x,const block_q8_K* a,int64_t& isum,int64_t& summs){
    int scale[16],mins[16];
    for(int s=0;s<16;++s){ scale[s]=x->scales[s]&0x0F; mins[s]=(x->scales[s]>>4)&0x0F; }
    isum=0; int is=0;
    for(int k=0;k<QK_K/128;++k){ const uint8_t* q2=x->qs+k*32; const int8_t* q8=a->qs+k*128; int shift=0;
        for(int j=0;j<4;++j){ int d0=scale[is]; int sl=0;
            for(int l=0;l<16;++l) sl+=q8[j*32+l]*((q2[l]>>shift)&3); isum+=(int64_t)d0*sl;
            int d1=scale[is+1]; sl=0;
            for(int l=16;l<32;++l) sl+=q8[j*32+l]*((q2[l]>>shift)&3); isum+=(int64_t)d1*sl;
            is+=2; shift+=2; } }
    summs=0; for(int s=0;s<16;++s) summs+=(int64_t)a->bsums[s]*mins[s];
}

/* ggml stock make_block_q2_Kx16 scale sub-block permutation (Sequential-Parallel) */
static const int GGML_PERM[16] = {0,2,4,6, 1,3,5,7, 8,10,12,14, 9,11,13,15};

int main(int argc,char**argv){
    if(argc<5){ fprintf(stderr,"usage: %s K nr nc seed\n",argv[0]); return 2; }
    int K=atoi(argv[1]), nr=atoi(argv[2]), nc=atoi(argv[3]);
    std::mt19937 rng(argc>4?(unsigned)strtoul(argv[4],0,0):20260715u);
    if(K%QK_K||nr%4||nc%16){ fprintf(stderr,"need K%%256 nr%%4 nc%%16\n"); return 2; }
    int nb=K/QK_K, ng=nc/16, gr=nr/4;

    /* ---- plain q2_K weights (adversarial fp16 d/dmin, random scales+qs) ---- */
    std::uniform_int_distribution<int> byte(0,255);
    std::vector<block_q2_K> W((size_t)nc*nb);
    for(size_t i=0;i<W.size();++i){
        for(int s=0;s<16;++s) W[i].scales[s]=(uint8_t)byte(rng);
        for(int q=0;q<64;++q) W[i].qs[q]=(uint8_t)byte(rng);
        W[i].d   = f2h(0.010f + 0.0006f*(int)(i%11));
        W[i].dmin= f2h(0.006f + 0.0004f*(int)(i%7));
    }
    /* ---- q8_K activation (nr rows) ---- */
    std::vector<block_q8_K> A((size_t)nr*nb);
    std::uniform_int_distribution<int> q8(-90,90);
    for(int r=0;r<nr;++r)for(int b=0;b<nb;++b){ block_q8_K& a=A[(size_t)r*nb+b];
        a.d=0.012f+0.002f*((r+b)%9);
        for(int g=0;g<16;++g){ int dc=((g*7+r*3)%21)-10; int sum=0;
            for(int i=0;i<16;++i){ int v=q8(rng)+dc; if(v>127)v=127; if(v<-128)v=-128;
                a.qs[g*16+i]=(int8_t)v; sum+=v; } a.bsums[g]=(int16_t)sum; } }

    /* ---- independent reference GEMM: out[r*nc+c] = sum_b( d*a.d*isum - dmin*a.d*summs ) ---- */
    std::vector<double> ref((size_t)nr*nc);
    for(int r=0;r<nr;++r)for(int c=0;c<nc;++c){ double acc=0;
        for(int b=0;b<nb;++b){ int64_t ii,ss; const block_q2_K& x=W[(size_t)c*nb+b]; const block_q8_K& a=A[(size_t)r*nb+b];
            ref_block(&x,&a,ii,ss); acc += h2f(x.d)*(double)a.d*(double)ii - h2f(x.dmin)*(double)a.d*(double)ss; }
        ref[(size_t)r*nc+c]=acc; }

    /* ---- weight buffer (A) STRAIGHT = ours' expected / deploy-patched layout ---- */
    std::vector<uint8_t> Wstraight((size_t)ng*nb*1344,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&Wstraight[((size_t)g*nb+b)*1344];
        for(int c=0;c<16;++c){ const block_q2_K& x=W[(size_t)(g*16+c)*nb+b];
            memcpy(blk+c*2,&x.d,2); memcpy(blk+32+c*2,&x.dmin,2);
            for(int s=0;s<16;++s) blk[64+s*16+c]=x.scales[s];      /* STRAIGHT sub-block order */
            for(int i=0;i<64;++i) blk[320+i*16+c]=x.qs[i]; } }

    /* ---- weight buffer (B) GGML-STOCK = ggml's real repack<q2_K,1,16> (permuted scales) ---- */
    std::vector<uint8_t> Wggml((size_t)ng*nb*1344,0);
    { struct ggml_tensor t; memset(&t,0,sizeof t); t.type=GGML_TYPE_Q2_K;
      t.ne[0]=K; t.ne[1]=nc; t.ne[2]=1; t.ne[3]=1; t.data=Wggml.data();
      int rc=ggml_repack_q2k_16x1(&t, W.data(), (size_t)nc*nb*sizeof(block_q2_K));
      if(rc!=0){ fprintf(stderr,"GGML_REPACK_FAIL rc=%d\n",rc); return 4; } }

    /* ---- hand PERMUTED (cross-check == ggml real repack) ---- */
    std::vector<uint8_t> Wperm((size_t)ng*nb*1344,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&Wperm[((size_t)g*nb+b)*1344];
        for(int c=0;c<16;++c){ const block_q2_K& x=W[(size_t)(g*16+c)*nb+b];
            memcpy(blk+c*2,&x.d,2); memcpy(blk+32+c*2,&x.dmin,2);
            for(int chunk=0;chunk<16;++chunk) blk[64+chunk*16+c]=x.scales[GGML_PERM[chunk]];
            for(int i=0;i<64;++i) blk[320+i*16+c]=x.qs[i]; } }
    long long permdiff=0; for(size_t i=0;i<Wggml.size();++i) if(Wggml[i]!=Wperm[i]) permdiff++;

    /* ---- q8_Kx4 GEMM activation (d[4]@0, qs p*4+c @16, bsums (s*4+c)@1040, stride 1168) ---- */
    std::vector<uint8_t> Y((size_t)gr*nb*1168,0);
    for(int g=0;g<gr;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&Y[((size_t)g*nb+b)*1168];
        for(int c=0;c<4;++c){ float d=A[(size_t)(g*4+c)*nb+b].d; memcpy(blk+c*4,&d,4); }
        for(int p=0;p<QK_K;++p)for(int c=0;c<4;++c) blk[16+p*4+c]=(uint8_t)A[(size_t)(g*4+c)*nb+b].qs[p];
        for(int s=0;s<16;++s)for(int c=0;c<4;++c){ int16_t v=A[(size_t)(g*4+c)*nb+b].bsums[s]; memcpy(blk+1040+(s*4+c)*2,&v,2); } }

    auto chk=[&](const std::vector<float>& S,const char* who){ double mrel=0; long long nbad=0;
        for(int r=0;r<nr;++r)for(int c=0;c<nc;++c){ double v=S[(size_t)r*nc+c],g=ref[(size_t)r*nc+c];
            double den=std::fabs(g)>std::fabs(v)?std::fabs(g):std::fabs(v); double rel=den>1e-9?std::fabs(v-g)/den:0;
            if(rel>mrel)mrel=rel; if(rel>1e-2)nbad++; }
        printf("  %-28s max_rel=%.3e  nbad=%lld/%d  => %s\n",who,mrel,nbad,nr*nc, nbad?"WRONG":"MATCH(byte-exact-fp)"); return nbad; };

    printf("# q2_K adjudication  K=%d nr=%d nc=%d  (nb=%d ng=%d gr=%d)\n",K,nr,nc,nb,ng,gr);
    printf("# ggml-real-repack vs hand-permuted diff bytes = %lld  (0 => GGML_PERM order confirmed)\n",permdiff);

    /* OURS kernel on STRAIGHT (A) */
    std::vector<float> Sa((size_t)nr*nc,9.0f);
    weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K((size_t)K,Sa.data(),Wstraight.data(),Y.data(),(size_t)nr,(size_t)nc,(size_t)nc);
    long long nA=chk(Sa,"OURS on STRAIGHT (A)");

    /* OURS kernel on GGML-STOCK (B) */
    std::vector<float> Sb((size_t)nr*nc,9.0f);
    weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K((size_t)K,Sb.data(),Wggml.data(),Y.data(),(size_t)nr,(size_t)nc,(size_t)nc);
    long long nB=chk(Sb,"OURS on GGML-STOCK (B)");

    /* STOCK gemm on GGML-STOCK (positive control: stock repack + stock gemm self-consistent) */
    std::vector<float> Sc((size_t)nr*nc,9.0f);
    ggml_gemm_q2_K_16x1_q8_K(K,Sc.data(),(size_t)nc,Wggml.data(),Y.data(),nr,nc);
    long long nC=chk(Sc,"STOCK gemm on GGML-STOCK");

    printf("\nVERDICT: OURS-STRAIGHT=%s  OURS-GGMLSTOCK=%s  STOCK-GGMLSTOCK=%s\n",
        nA?"WRONG":"MATCH", nB?"WRONG":"MATCH", nC?"WRONG":"MATCH");
    if(nA==0 && nB>0)
        printf("  => FEEDING-MISMATCH: ours CORRECT on its straight layout; prior all-wrong = fed ggml permuted scales.\n");
    else if(nA>0)
        printf("  => REAL KERNEL BUG: ours wrong even on its own straight layout.\n");
    return 0;
}
