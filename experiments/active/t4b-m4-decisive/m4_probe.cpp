/* [G3-cert-hardening M4] DECISIVE clean positional: OUR compiler-emitted q4_K/q5_K repack-GEMM
 * kernel  vs  ggml's OWN repack GEMM, on a REAL Q4_K_M model tensor (dmin!=0), with the SAME
 * ggml mat-quant q8 activation (ggml_quantize_mat_q8_K_4x1) fed to BOTH. No hand oracle in the
 * head-to-head loop -- we call ggml's own function directly.
 *
 * q4_K opponent = ggml_gemm_q4_K_16x1_q8_K_generic  (ggml's OWN portable reference of the q4_K
 *   16x1 repack GEMM; the RVV variant ggml_gemm_q4_K_16x1_q8_K hardcodes vl=16 on f32m2/u8m2 =>
 *   VLEN256-only, structurally broken at this board's VLEN128 -- demonstrated below).
 * q5_K: the Q4_K_M model ships NO native Q5_K tensor and ggml ships NO q5_K 16x1 gemm (only
 *   q5_Kx8), so q5_K uses REAL-model-derived weights (dequantize a real Q6_K tensor -> requantize
 *   to Q5_K, dmin!=0) and an INDEPENDENT integer-exact oracle built from the on-disk q5_K blocks
 *   + the SAME deinterleaved mat-quant q8. Same shared min-fold as q4_K (emitRepackKQuantGemmBodyQ4K).
 *
 * Pre-registered reading:
 *   integer bit-exact (kernel captured MAIN/MIN == exact recompute)  AND  float output within
 *   bounded ULP of ggml's own generic / the integer-exact oracle (every arithmetic term accounted)
 *      => CASE CLOSED (kernel correct, claims recoverable).
 *   any un-attributable macroscopic mismatch => CASE REOPENED.
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "kquant_repacker.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cstdlib>

/* ---- ggml's own quantizers / GEMMs / (de)quantizers, dynamically exported by the .so ---- */
extern "C" void  ggml_quantize_mat_q8_K_4x1(const float* x, void* vy, int64_t k);
extern "C" void  ggml_gemm_q4_K_16x1_q8_K        (int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc);
extern "C" void  ggml_gemm_q4_K_16x1_q8_K_generic(int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc);
extern "C" void  quantize_row_q5_K   (const float* x, void* vy, int64_t k);
extern "C" void  dequantize_row_q6_K (const void* vx, float* y, int64_t k);
extern "C" void  dequantize_row_q4_K (const void* vx, float* y, int64_t k);
/* our compiler-emitted kernels, exposed by the reversible M4 patch (visibility default wrappers) */
extern "C" void  tcrv_m4_call_q4K(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
extern "C" void  tcrv_m4_call_q5K(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
/* pure-observation integer capture globals defined in the instrumented q4_K kernel TU */
extern "C" {
  extern int   tcrv_cap_want, tcrv_cap_nblk, tcrv_cap_hits;
  extern int   tcrv_main_i[8][4][8];
  extern int   tcrv_min_i [8][4][8];
  extern float tcrv_d[8][8], tcrv_dmin[8][8], tcrv_ad[8][4];
  extern short tcrv_bsums[8][64];
}

struct blk_q8_Kx4 { float d[4]; int8_t qs[1024]; int16_t bsums[64]; };   /* 1168, mat-quant output */

static float h2f(uint16_t h){
    uint32_t s=(h>>15)&1u,e=(h>>10)&0x1Fu,m=h&0x3FFu,o;
    if(e==0){ if(m==0)o=s<<31; else{int ee=-14;while(!(m&0x400u)){m<<=1;ee--;}m&=0x3FFu;o=(s<<31)|((uint32_t)(ee+127)<<23)|(m<<13);} }
    else if(e==0x1Fu)o=(s<<31)|(0xFFu<<23)|(m<<13);
    else o=(s<<31)|((e-15u+127u)<<23)|(m<<13);
    float f; memcpy(&f,&o,4); return f;
}
static void gsm4(int j,const uint8_t*q,uint8_t*d,uint8_t*m){
    if(j<4){*d=q[j]&63;*m=q[j+4]&63;} else{*d=(q[j+4]&0xF)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j]>>6)<<4);} }
static long ulp_dist(float a,float b){
    int32_t ia,ib; memcpy(&ia,&a,4); memcpy(&ib,&b,4);
    if(ia<0)ia=0x80000000-ia; if(ib<0)ib=0x80000000-ib;
    return llabs((long)ia-(long)ib);
}

static uint64_t RNG=0x243F6A8885A308D3ull;
static uint32_t xr(){ RNG^=RNG<<13;RNG^=RNG>>7;RNG^=RNG<<17;return (uint32_t)(RNG>>32); }

/* ---- read a named tensor's raw bytes straight from the gguf file (no full-model load) ---- */
static bool load_tensor(const char* path,const char* name,std::vector<uint8_t>&out,
                        int64_t&ne0,int64_t&ne1,ggml_type&ty){
    ggml_context* mctx=nullptr; gguf_init_params p{true,&mctx};
    gguf_context* g=gguf_init_from_file(path,p); if(!g){printf("gguf open FAIL\n");return false;}
    int64_t id=gguf_find_tensor(g,name); if(id<0){printf("tensor '%s' not found\n",name);gguf_free(g);return false;}
    ty=gguf_get_tensor_type(g,id);
    ggml_tensor* t=ggml_get_tensor(mctx,name); ne0=t->ne[0]; ne1=t->ne[1];
    size_t nbytes=ggml_nbytes(t);
    size_t abs=gguf_get_data_offset(g)+gguf_get_tensor_offset(g,id);
    out.resize(nbytes);
    FILE* f=fopen(path,"rb"); fseek(f,(long)abs,SEEK_SET);
    size_t got=fread(out.data(),1,nbytes,f); fclose(f); gguf_free(g);
    if(got!=nbytes){printf("short read %zu/%zu\n",got,nbytes);return false;}
    return true;
}

struct Metrics{ double max_abs=0,max_rel=0; long max_ulp=0; int nexact=0,ntot=0; };
static Metrics compare(const std::vector<float>&A,const std::vector<float>&B){
    Metrics M; M.ntot=(int)A.size();
    for(size_t k=0;k<A.size();++k){
        double a=A[k],b=B[k],ab=std::fabs(a-b);
        if(ab>M.max_abs)M.max_abs=ab;
        double den=std::fabs(b)>1.0?std::fabs(b):1.0; double r=ab/den; if(r>M.max_rel)M.max_rel=r;
        long u=ulp_dist(A[k],B[k]); if(u>M.max_ulp)M.max_ulp=u;
        if(a==b)M.nexact++;
    }
    return M;
}
static void report(const char*tag,const Metrics&M){
    printf("    %-34s max_abs=%.3e  max_rel(|.|>1)=%.3e  max_ulp=%ld  exact=%d/%d\n",
           tag,M.max_abs,M.max_rel,M.max_ulp,M.nexact,M.ntot);
}

/* ================= q4_K decisive ================= */
static int run_q4K(const char* model,const char* tname){
    printf("\n================ q4_K  tensor=%s ================\n",tname);
    std::vector<uint8_t> raw; int64_t ne0,ne1; ggml_type ty;
    if(!load_tensor(model,tname,raw,ne0,ne1,ty)) return 2;
    if(ty!=GGML_TYPE_Q4_K){ printf("not Q4_K\n"); return 2; }
    int K=(int)ne0, N=(int)ne1, nb=K/256;
    printf("  real tensor: type=Q4_K K=%d N=%d nb=%d  (%zu bytes)\n",K,N,nb,raw.size());
    const kqr_block_q4_K* W=(const kqr_block_q4_K*)raw.data();   /* on-disk == kqr_block_q4_K (144B) */

    /* dmin!=0 census on the REAL tensor */
    int dmin_nz=0,dtot=N*nb; float dmn_min=1e30f,dmn_max=0,d_s=0,dmn_s=0;
    for(int c=0;c<N;++c)for(int b=0;b<nb;++b){ float dm=h2f(W[(size_t)c*nb+b].dmin); if(dm!=0){dmin_nz++; if(dm<dmn_min)dmn_min=dm; if(dm>dmn_max)dmn_max=dm;} }
    d_s=h2f(W[0].d); dmn_s=h2f(W[0].dmin);
    printf("  dmin!=0 blocks: %d/%d  (min=%.3e max=%.3e) sample d=%.6g dmin=%.6g  MIN-TERM ACTIVE=%s\n",
           dmin_nz,dtot,dmn_min,dmn_max,d_s,dmn_s,dmin_nz>0?"YES":"NO");

    /* activation: 4 rows x K, realistic [-5,5]; mat-quant via ggml's own quantizer */
    const int nr=4;
    std::vector<float> act((size_t)nr*K);
    RNG=0x243F6A8885A308D3ull^0x4b34ull;
    for(size_t i=0;i<act.size();++i) act[i]=0.05f*(float)((int)(xr()%201)-100);
    std::vector<blk_q8_Kx4> vy(nb);
    ggml_quantize_mat_q8_K_4x1(act.data(), vy.data(), K);        /* SAME q8 fed to BOTH kernels */

    /* weights repacked by the M1 repacker (block_q4_Kx16, stride 2304) -- identical input to both */
    std::vector<uint8_t> vx(kqr_bytes_q4_K(N,nb));
    kqr_repack_q4_K(vx.data(), W, N, nb);

    std::vector<float> A((size_t)nr*N,0), B((size_t)nr*N,0), C((size_t)nr*N,0);
    /* OUR kernel (instrumented, capture ON for cols0..7/grp0) */
    tcrv_cap_nblk=0; tcrv_cap_hits=0; tcrv_cap_want=1;
    tcrv_m4_call_q4K((size_t)K, A.data(), vx.data(), (const uint8_t*)vy.data(), (size_t)nr, (size_t)N, (size_t)N);
    tcrv_cap_want=0;
    /* ggml's OWN generic (VLEN-independent ground truth) */
    ggml_gemm_q4_K_16x1_q8_K_generic(K, B.data(), N, vx.data(), vy.data(), nr, N);
    /* ggml's RVV variant (VLEN256-only; expected broken at VLEN128) */
    ggml_gemm_q4_K_16x1_q8_K        (K, C.data(), N, vx.data(), vy.data(), nr, N);

    /* independent integer-exact double oracle G from on-disk W + deinterleaved mat-quant q8 */
    std::vector<float> G((size_t)nr*N,0.0f);
    for(int r=0;r<nr;++r)for(int c=0;c<N;++c){
        double acc=0.0;
        for(int b=0;b<nb;++b){
            const kqr_block_q4_K* xb=&W[(size_t)c*nb+b];
            long mains=0,mins=0; uint8_t sc,m; const uint8_t* q=xb->qs; int is=0;
            for(int jg=0;jg<256;jg+=64){
                gsm4(is+0,xb->scales,&sc,&m); int sc0=sc,m0=m;
                gsm4(is+1,xb->scales,&sc,&m); int sc1=sc,m1=m;
                for(int l=0;l<32;l++){int w=q[l]&0xF;   int a8=vy[b].qs[(jg+l)*4+r];    mains+=(long)sc0*w*a8; mins+=(long)m0*a8;}
                for(int l=0;l<32;l++){int w=q[l]>>4;    int a8=vy[b].qs[(jg+32+l)*4+r]; mains+=(long)sc1*w*a8; mins+=(long)m1*a8;}
                q+=32; is+=2;
            }
            double d=h2f(xb->d), dmin=h2f(xb->dmin), ad=vy[b].d[r];
            acc += ad*( d*(double)mains - dmin*(double)mins );
        }
        G[(size_t)r*N+c]=(float)acc;
    }

    /* integer bit-exact capture check on the REAL tensor (cols 0..7, blocks 0..min(nb,8)) */
    int nblk=tcrv_cap_nblk; if(nblk>nb)nblk=nb; if(nblk>8)nblk=8;
    int main_bad=0,min_bad=0;
    for(int b=0;b<nblk;b++)for(int r=0;r<nr;r++)for(int c=0;c<8;c++){
        const kqr_block_q4_K* xb=&W[(size_t)c*nb+b];
        long mains=0,mins=0; uint8_t sc,m; const uint8_t* q=xb->qs; int is=0;
        for(int jg=0;jg<256;jg+=64){
            gsm4(is+0,xb->scales,&sc,&m); int sc0=sc,m0=m;
            gsm4(is+1,xb->scales,&sc,&m); int sc1=sc,m1=m;
            for(int l=0;l<32;l++){int w=q[l]&0xF; int a8=vy[b].qs[(jg+l)*4+r];    mains+=(long)sc0*w*a8; mins+=(long)m0*a8;}
            for(int l=0;l<32;l++){int w=q[l]>>4;  int a8=vy[b].qs[(jg+32+l)*4+r]; mains+=(long)sc1*w*a8; mins+=(long)m1*a8;}
            q+=32; is+=2;
        }
        if(mains!=(long)tcrv_main_i[b][r][c]) main_bad++;
        if(mins !=(long)tcrv_min_i [b][r][c]) min_bad++;
    }
    printf("  [INTEGER capture] kernel MAIN/MIN vs exact recompute-from-mat-q8 (cols0..7, blk0..%d, rows0..3):\n",nblk-1);
    printf("      hits=%d nblk=%d  MAIN mismatches=%d  MIN mismatches=%d  => %s\n",
           tcrv_cap_hits,tcrv_cap_nblk,main_bad,min_bad,
           (main_bad==0&&min_bad==0)?"INTEGER BIT-EXACT":"*** INTEGER DIVERGENCE ***");

    printf("  [FLOAT head-to-head] (real dmin!=0 weights, same mat-quant q8 into both):\n");
    Metrics AB=compare(A,B), AG=compare(A,G), BG=compare(B,G);
    report("OURS  vs ggml_generic (A vs B)",AB);
    report("OURS  vs int-exact oracle (A vs G)",AG);
    report("ggml_generic vs int-exact (B vs G)",BG);
    /* demonstrate the RVV variant is VLEN256-only (broken here) */
    Metrics CG=compare(C,G); int c_hi_bad=0;
    for(int r=0;r<nr;r++)for(int c=0;c<N;c++){ if(std::fabs(C[(size_t)r*N+c]-G[(size_t)r*N+c])>1e-2*(std::fabs(G[(size_t)r*N+c])+1e-3)) c_hi_bad++; }
    printf("    %-34s max_abs=%.3e  bad-elems(>1%%)=%d/%d  (RVV vl=16 hardcode => VLEN256-only)\n",
           "ggml_RVV vs int-exact (C vs G)",CG.max_abs,c_hi_bad,(int)C.size());

    /* sample row0 cols0..7 side by side */
    printf("  sample row0 c0..7:  OURS / ggml_gen / int-exact\n");
    for(int c=0;c<8;c++) printf("    c%d: %12.5f / %12.5f / %12.5f\n",c,A[c],B[c],G[c]);

    bool closed = (main_bad==0 && min_bad==0) && (AG.max_rel<1e-4) && (AB.max_rel<1e-4);
    printf("  >>> q4_K VERDICT: %s (int-exact:%s  A~G rel=%.2e  A~ggml rel=%.2e)\n",
           closed?"CASE CLOSED":"CASE REOPENED",
           (main_bad==0&&min_bad==0)?"YES":"NO",AG.max_rel,AB.max_rel);
    return closed?0:1;
}

/* ================= q5_K decisive (real-derived weights + integer-exact oracle) ================= */
static int run_q5K(const char* model,const char* src_q6_name){
    printf("\n================ q5_K  (real-derived from %s) ================\n",src_q6_name);
    std::vector<uint8_t> raw; int64_t ne0,ne1; ggml_type ty;
    if(!load_tensor(model,src_q6_name,raw,ne0,ne1,ty)) return 2;
    if(ty!=GGML_TYPE_Q6_K){ printf("source not Q6_K\n"); return 2; }
    int K=(int)ne0, N=(int)ne1, nb=K/256;
    printf("  source Q6_K K=%d N=%d nb=%d -> dequant -> requant Q5_K (real-model-derived)\n",K,N,nb);

    /* dequant each output-col (row of K) from Q6_K, requant to Q5_K on-disk blocks */
    size_t q6_row_bytes = raw.size()/(size_t)N;      /* bytes per K-row of q6_K */
    std::vector<kqr_block_q5_K> W5((size_t)N*nb);
    std::vector<float> f32row(K);
    for(int c=0;c<N;++c){
        dequantize_row_q6_K(raw.data()+(size_t)c*q6_row_bytes, f32row.data(), K);
        quantize_row_q5_K(f32row.data(), &W5[(size_t)c*nb], K);
    }
    int dmin_nz=0,dtot=N*nb; float dmn_min=1e30f,dmn_max=0;
    for(int c=0;c<N;++c)for(int b=0;b<nb;++b){ float dm=h2f(W5[(size_t)c*nb+b].dmin); if(dm!=0){dmin_nz++; if(dm<dmn_min)dmn_min=dm; if(dm>dmn_max)dmn_max=dm;} }
    printf("  q5_K dmin!=0 blocks: %d/%d (min=%.3e max=%.3e) sample d=%.6g dmin=%.6g  MIN-TERM ACTIVE=%s\n",
           dmin_nz,dtot,dmn_min,dmn_max,h2f(W5[0].d),h2f(W5[0].dmin),dmin_nz>0?"YES":"NO");

    /* activation + SAME mat-quant q8 */
    const int nr=4;
    std::vector<float> act((size_t)nr*K);
    RNG=0x243F6A8885A308D3ull^0x5c5cull;
    for(size_t i=0;i<act.size();++i) act[i]=0.05f*(float)((int)(xr()%201)-100);
    std::vector<blk_q8_Kx4> vy(nb);
    ggml_quantize_mat_q8_K_4x1(act.data(), vy.data(), K);

    std::vector<uint8_t> vx(kqr_bytes_q5_K(N,nb));
    kqr_repack_q5_K(vx.data(), W5.data(), N, nb);

    std::vector<float> A((size_t)nr*N,0);
    tcrv_m4_call_q5K((size_t)K, A.data(), vx.data(), (const uint8_t*)vy.data(), (size_t)nr, (size_t)N, (size_t)N);

    /* independent integer-exact double oracle for q5_K (5-bit weight = nibble | qh<<4) */
    std::vector<float> G((size_t)nr*N,0.0f);
    for(int r=0;r<nr;++r)for(int c=0;c<N;++c){
        double acc=0.0;
        for(int b=0;b<nb;++b){
            const kqr_block_q5_K* xb=&W5[(size_t)c*nb+b];
            const uint8_t* ql=xb->qs; const uint8_t* qh=xb->qh;
            long mains=0,mins=0; uint8_t sc,m; int is=0,u1=1,u2=2;
            for(int jg=0;jg<256;jg+=64){
                gsm4(is+0,xb->scales,&sc,&m); int sc0=sc,m0=m;
                gsm4(is+1,xb->scales,&sc,&m); int sc1=sc,m1=m;
                for(int l=0;l<32;l++){ int w=(ql[l]&0xF)+((qh[l]&u1)?16:0); int a8=vy[b].qs[(jg+l)*4+r];    mains+=(long)sc0*w*a8; mins+=(long)m0*a8; }
                for(int l=0;l<32;l++){ int w=(ql[l]>>4) +((qh[l]&u2)?16:0); int a8=vy[b].qs[(jg+32+l)*4+r]; mains+=(long)sc1*w*a8; mins+=(long)m1*a8; }
                ql+=32; is+=2; u1<<=2; u2<<=2;
            }
            double d=h2f(xb->d), dmin=h2f(xb->dmin), ad=vy[b].d[r];
            acc += ad*( d*(double)mains - dmin*(double)mins );
        }
        G[(size_t)r*N+c]=(float)acc;
    }

    Metrics AG=compare(A,G);
    printf("  [FLOAT] q5_K OURS vs int-exact oracle (real-derived dmin!=0, same mat-quant q8):\n");
    report("OURS vs int-exact oracle (A vs G)",AG);
    printf("  sample row0 c0..7:  OURS / int-exact\n");
    for(int c=0;c<8;c++) printf("    c%d: %12.5f / %12.5f\n",c,A[c],G[c]);
    bool closed = (AG.max_rel<1e-4);
    printf("  >>> q5_K VERDICT: %s (A~G rel=%.2e)\n",closed?"CASE CLOSED":"CASE REOPENED",AG.max_rel);
    return closed?0:1;
}

int main(int argc,char**argv){
    const char* model = argc>1?argv[1]:"/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf";
    const char* q4name= argc>2?argv[2]:"blk.0.attn_k.weight";
    const char* q6name= argc>3?argv[3]:"blk.0.attn_v.weight";
    printf("=== M4 DECISIVE POSITIONAL: our repack-GEMM vs ggml's own, REAL Q4_K_M tensor, VLEN128 ===\n");
    printf("    sizeof(blk_q8_Kx4)=%zu (expect 1168)  sizeof(kqr_block_q4_K)=%zu(144) kqr_block_q5_K=%zu(176)\n",
           sizeof(blk_q8_Kx4),sizeof(kqr_block_q4_K),sizeof(kqr_block_q5_K));
    int rc=0;
    rc |= run_q4K(model,q4name);
    rc |= run_q5K(model,q6name);
    printf("\n=== M4 SUMMARY: %s ===\n", rc==0?"ALL CASE CLOSED (kernel correct on real model tensors, dmin!=0)":"REOPENED / error");
    return rc;
}
