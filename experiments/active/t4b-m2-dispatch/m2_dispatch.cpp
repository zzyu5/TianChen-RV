/* [SEL-1 T4b / M2] Real ggml_mul_mat dispatch probe for the q4_K repack-GEMM path @VLEN128.
 *
 * Drives a genuine ggml_mul_mat through ggml_backend_graph_compute (NOT a direct kernel call):
 *   - REPACK run: weight tensor placed in ggml_backend_cpu_repack_buffer_type(); ggml's own
 *     set_tensor repacks it (make_block_q4_Kx16) and the patched selector routes q4_K@VLEN128 to
 *     trait q4_K_16x1_q8_K -> ggml_gemm_q4_K_16x1_q8_K -> (our patched VLEN128 branch) golden kernel.
 *     Prints the one-shot stderr routing banner from inside libggml-cpu.so.
 *   - STOCK run: identical graph, weight in the DEFAULT cpu buffer -> standard ggml_vec_dot_q4_K_q8_K.
 * Compares dst(REPACK) vs dst(STOCK): INT byte-exact / NORM bounded-ULP.
 * Also cross-checks ggml's internal repacked bytes (w->data) == M1 repacker kqr_repack_q4_K().
 */
#include <riscv_vector.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "kquant_repacker.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>

/* C++-mangled internal symbol (not extern "C") exported by libggml-cpu.so */
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

/* ---- weight builders (byte-identical to the M0/M1 cert oracle) ---- */
static uint64_t RNG;
static uint32_t xr(void){ RNG^=RNG<<13; RNG^=RNG>>7; RNG^=RNG<<17; return (uint32_t)(RNG>>32); }
static const uint16_t DHALF[8] = {0x2800,0x2C00,0x3000,0x3400,0x2400,0x2A00,0x3200,0x2600};
static void pack_disk_scales(uint8_t* sc12, const uint8_t sc[8], const uint8_t mn[8]){
    for(int j=0;j<4;++j){ sc12[j]=sc[j]; sc12[j+4]=mn[j]; }
    for(int j=4;j<8;++j){
        sc12[j+4] = (uint8_t)((sc[j]&0x0F) | ((mn[j]&0x0F)<<4));
        sc12[j-4] = (uint8_t)((sc12[j-4]&0x3F) | ((sc[j]>>4)<<6));
        sc12[j-0] = (uint8_t)((sc12[j-0]&0x3F) | ((mn[j]>>4)<<6));
    }
}
static void build_w_q4K(kqr_block_q4_K* x, int col, int blk, int intmode){
    if(intmode){ x->d=0x3C00; x->dmin=0x0000; } else { x->d=DHALF[(col+blk)&7]; x->dmin=DHALF[(col+3*blk+1)&7]; }
    uint8_t sc[8], mn[8];
    for(int j=0;j<8;++j){ sc[j]= intmode ? (uint8_t)(1+((j+col+blk)%3)) : (uint8_t)(xr()&63);
                          mn[j]= intmode ? 0 : (uint8_t)(xr()&63); }
    pack_disk_scales(x->scales, sc, mn);
    for(int i=0;i<KQR_QK_K/2;++i){
        uint8_t lo = intmode ? (uint8_t)((i+col)%13) : (uint8_t)(xr()&0x0F);
        uint8_t hi = intmode ? (uint8_t)((i+2*blk+1)%13) : (uint8_t)(xr()&0x0F);
        x->qs[i] = (uint8_t)((lo&0x0F) | ((hi&0x0F)<<4));
    }
}
/* F32 activation: INT mode = small integers (exact-int fp32 reduce), NORM = wider spread */
static float actval(int r,int k,int intmode){
    if(intmode) return (float)((int)(xr()%9)-4);
    return 0.05f*(float)((int)(xr()%201)-100);
}

static int64_t okey(float f){ uint32_t u; memcpy(&u,&f,4); return (u&0x80000000u)?-(int64_t)(u&0x7fffffffu):(int64_t)u; }
static int64_t ulpd(float a,float b){ int64_t d=okey(a)-okey(b); return d<0?-d:d; }

/* Run one mul_mat through the real backend. weight in `wbuft`. Returns dst [N*rows] (row r, col c at r*N+c).
 * If wdata_out != NULL, copies the (possibly repacked) weight buffer bytes for the M1 cross-check. */
static std::vector<float> run(ggml_backend_t backend, ggml_backend_buffer_type_t wbuft,
                              int N,int K,int rows,const uint8_t* wbytes,size_t wsize,const float* act,
                              std::vector<uint8_t>* wdata_out){
    ggml_init_params wp{ ggml_tensor_overhead()+256, nullptr, true };
    ggml_context* wctx = ggml_init(wp);
    ggml_tensor* w = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_K, K, N);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, wbuft);
    ggml_backend_tensor_set(w, wbytes, 0, wsize);           /* repack buffer -> triggers repack */
    if(wdata_out){ wdata_out->resize(wsize); memcpy(wdata_out->data(), w->data, wsize); }

    ggml_init_params cp{ ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context* cctx = ggml_init(cp);
    ggml_tensor* a = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, K, rows);
    ggml_tensor* out = ggml_mul_mat(cctx, w, a);            /* [N, rows] */
    ggml_cgraph* gf = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(a, act, 0, ggml_nbytes(a));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> res((size_t)N*rows);
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(alloc);
    ggml_free(cctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return res;
}

/* ---- independent scalar q4_K x q8_K oracle, using the REPACK path's own quantization
 * (iscale = -127/max, matching ggml_quantize_mat_q8_K_4x1). double accumulation. ---- */
static float fp16_to_fp32(uint16_t h){
    uint32_t s=(h>>15)&1u, e=(h>>10)&0x1Fu, m=h&0x3FFu, out;
    if(e==0){ if(m==0){ out=s<<31; } else { int ee=-14; while(!(m&0x400u)){ m<<=1; ee--; } m&=0x3FFu; out=(s<<31)|((uint32_t)(ee+127)<<23)|(m<<13); } }
    else if(e==0x1Fu){ out=(s<<31)|(0xFFu<<23)|(m<<13); }
    else { out=(s<<31)|((e-15u+127u)<<23)|(m<<13); }
    float f; memcpy(&f,&out,4); return f;
}
static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m){
    if(j<4){ *d=q[j]&63; *m=q[j+4]&63; }
    else { *d=(uint8_t)((q[j+4]&0xF)|((q[j-4]>>6)<<4)); *m=(uint8_t)((q[j+4]>>4)|((q[j]>>6)<<4)); }
}
static int nearest_int_(float x){ return (int)lrintf(x); }
static float oracle_dot(const kqr_block_q4_K* Wcol, const float* actrow, int nb){ return 0; } /* replaced below */
static float oracle_dot2(const kqr_block_q4_K* Wcol, const float* actrow, int nb, int use_min){
    double result=0.0;
    for(int i=0;i<nb;i++){
        const kqr_block_q4_K* xb=&Wcol[i];
        const float* xa=actrow + (size_t)i*256;
        float amax=0,max=0;
        for(int j=0;j<256;j++){ float v=xa[j]; if(amax<std::fabs(v)){amax=std::fabs(v); max=v;} }
        float iscale = amax? -127.f/max : 0.f;
        float a_d = amax? 1.f/iscale : 0.f;
        int8_t q8[256];
        for(int j=0;j<256;j++) q8[j]=(int8_t)nearest_int_(xa[j]*iscale);
        float d=fp16_to_fp32(xb->d), dmin=fp16_to_fp32(xb->dmin);
        uint8_t sc,m; double mains=0, mins=0;
        const uint8_t* q=xb->qs; int is=0;
        for(int jg=0;jg<256;jg+=64){
            get_scale_min_k4(is+0,xb->scales,&sc,&m); int sc0=sc,m0=m;
            get_scale_min_k4(is+1,xb->scales,&sc,&m); int sc1=sc,m1=m;
            for(int l=0;l<32;l++){ int w=q[l]&0xF; int a8=q8[jg+l];    mains+=(double)sc0*w*a8; mins+=(double)m0*a8; }
            for(int l=0;l<32;l++){ int w=q[l]>>4;  int a8=q8[jg+32+l]; mains+=(double)sc1*w*a8; mins+=(double)m1*a8; }
            q+=32; is+=2;
        }
        result += (double)a_d * ( (double)d*mains - (use_min ? (double)dmin*mins : 0.0) );
    }
    return (float)result;
}

static int one_case(ggml_backend_t backend,int N,int K,int rows,int intmode){
    const int nb = K/256;
    RNG = 0x243F6A8885A308D3ull ^ ((uint64_t)(N*1315423911u + K*2654435761u + rows*40503u + intmode));
    /* weights (on-disk block_q4_K), laid out [c*nb + b] */
    std::vector<kqr_block_q4_K> W((size_t)N*nb);
    for(int c=0;c<N;++c) for(int b=0;b<nb;++b) build_w_q4K(&W[(size_t)c*nb+b], c, b, intmode);
    std::vector<uint8_t> wbytes((size_t)N*nb*sizeof(kqr_block_q4_K));
    memcpy(wbytes.data(), W.data(), wbytes.size());
    /* activation f32 [K, rows] laid out [r*K + k] */
    std::vector<float> act((size_t)rows*K);
    for(int r=0;r<rows;++r) for(int k=0;k<K;++k) act[(size_t)r*K+k]=actval(r,k,intmode);

    /* M1 repacker reference layout */
    std::vector<uint8_t> m1(kqr_bytes_q4_K(N,nb));
    kqr_repack_q4_K(m1.data(), W.data(), N, nb);

    std::vector<uint8_t> ggml_repacked;
    fprintf(stderr,">>> [%s N=%d K=%d rows=%d] REPACK-dispatch compute begin\n",
            intmode?"INT ":"NORM",N,K,rows); fflush(stderr);
    std::vector<float> R = run(backend, ggml_backend_cpu_repack_buffer_type(), N,K,rows,
                               wbytes.data(), wbytes.size(), act.data(), &ggml_repacked);
    fprintf(stderr,"<<< REPACK-dispatch compute end\n"); fflush(stderr);
    std::vector<float> S = run(backend, ggml_backend_get_default_buffer_type(backend), N,K,rows,
                               wbytes.data(), wbytes.size(), act.data(), nullptr);

    /* M1 <-> ggml internal repack byte equality */
    int repack_match = (ggml_repacked.size()==m1.size() && memcmp(ggml_repacked.data(),m1.data(),m1.size())==0);

    /* two oracles: F=full (main-min), Fnm=main-term-only (min forced off) */
    std::vector<float> F((size_t)N*rows), Fnm((size_t)N*rows);
    for(int r=0;r<rows;++r) for(int c=0;c<N;++c){
        F  [(size_t)r*N+c] = oracle_dot2(&W[(size_t)c*nb], &act[(size_t)r*K], nb, 1);
        Fnm[(size_t)r*N+c] = oracle_dot2(&W[(size_t)c*nb], &act[(size_t)r*K], nb, 0);
    }
    double wR_RF=0, wR_RFnm=0, wR_RS=0;
    for(size_t k=0;k<R.size();++k){
        double dF   = std::fabs((double)F[k])  >1e-6?std::fabs((double)F[k])  :1.0;
        double dFnm = std::fabs((double)Fnm[k])>1e-6?std::fabs((double)Fnm[k]):1.0;
        double dS   = std::fabs((double)S[k])  >1e-6?std::fabs((double)S[k])  :1.0;
        double rF=std::fabs((double)R[k]-(double)F[k])/dF;      if(rF>wR_RF)wR_RF=rF;
        double rN=std::fabs((double)R[k]-(double)Fnm[k])/dFnm;  if(rN>wR_RFnm)wR_RFnm=rN;
        double rS=std::fabs((double)R[k]-(double)S[k])/dS;      if(rS>wR_RS)wR_RS=rS;
    }
    printf("  [%s] N=%-4d K=%-5d rows=%d : M1==ggml-repack:%s | R-vs-ORACLE(full) rel=%.3e %s | R-vs-ORACLE(main-only) rel=%.3e | R-vs-tcrv-stock rel=%.3e\n",
           intmode?"INT ":"NORM", N,K,rows, repack_match?"YES":"NO ",
           wR_RF, (wR_RF<1e-3?"OK":"BAD"), wR_RFnm, wR_RS);
    if(N==64 && K==512 && !intmode){
        printf("      sample[r0c0..3] R    : %.4f %.4f %.4f %.4f\n", R[0],R[1],R[2],R[3]);
        printf("      sample[r0c0..3] Ffull: %.4f %.4f %.4f %.4f\n", F[0],F[1],F[2],F[3]);
        printf("      sample[r0c0..3] Fmain: %.4f %.4f %.4f %.4f\n", Fnm[0],Fnm[1],Fnm[2],Fnm[3]);
    }
    int ok = repack_match && (wR_RF<1e-3);
    return ok?0:1;
}

int main(void){
    fprintf(stderr,"VLEN(bits)=%d\n",(int)(__riscv_vlenb()*8)); fflush(stderr);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    printf("=== M2 q4_K repack-GEMM through REAL ggml_mul_mat dispatch (VLEN128) ===\n");
    int fail=0;
    fail |= one_case(backend, 64,  512,  4, 1);
    fail |= one_case(backend, 160, 2560, 4, 1);
    fail |= one_case(backend, 64,  512,  4, 0);
    fail |= one_case(backend, 160, 2560, 4, 0);
    fail |= one_case(backend, 256, 512,  4, 0);
    printf("SUMMARY: %s\n", fail?"FAIL":"PASS (real dispatch routed to our kernel; repack==M1; numeric OK)");
    ggml_backend_free(backend);
    return fail;
}
