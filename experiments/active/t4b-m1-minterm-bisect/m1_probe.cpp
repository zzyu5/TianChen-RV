/* [G3-minterm-fix M1] Instrumented dispatch probe for the q4_K repack-GEMM min-term @VLEN128.
 *
 * Built on M2's m2_dispatch.cpp (same real ggml_mul_mat dispatch, same weight/activation
 * builders, same independent scalar oracle). ADDS:
 *   (1) an INTEGER per-(block,row,col) oracle for the MAIN accumulator (sum sc*w*q8) and the
 *       MIN accumulator (sum m*q8) -- exactly the two integer quantities the kernel holds in
 *       v47..v53 (main) and v46 (min) right before the fp16 d/dmin fold.
 *   (2) a readout of the kernel's captured intermediates (tcrv_main_i/tcrv_min_i/tcrv_d/
 *       tcrv_dmin/tcrv_ad, defined in the instrumented kernel TU inside libggml-cpu.so) for
 *       the NORM N=64 K=512 case, weight cols 0..7, printed SIDE-BY-SIDE with the oracle.
 * This BISECTS the min-term: if MIN integer accumulator matches oracle but output is wrong,
 * the defect is the fp16 dmin column application; if MIN integer accumulator is wrong, the
 * defect is upstream (min-value unpack or bsums pairing). CONFIRM-ONLY, no kernel math change.
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

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

/* ggml's two q8_K quantizers, dynamically exported by libggml-cpu.so:
 *  - ggml_quantize_mat_q8_K_4x1 : quantizes 4 rows -> interleaved q8_Kx4 (what the REPACK kernel eats)
 *  - quantize_row_q8_K          : quantizes 1 row  -> q8_K            (what the STOCK vec_dot eats) */
extern "C" void ggml_quantize_mat_q8_K_4x1(const float* x, void* vy, int64_t k);
extern "C" void quantize_row_q8_K(const float* x, void* vy, int64_t k);
/* exact ggml layouts (QK_K=256): sizeof(block_q8_Kx4)=1168 == activation_block_stride */
struct blk_q8_K   { float d;    int8_t qs[256];  int16_t bsums[16]; };
struct blk_q8_Kx4 { float d[4]; int8_t qs[1024]; int16_t bsums[64]; };

/* ==== capture globals defined inside the instrumented kernel TU (libggml-cpu.so) ==== */
extern "C" {
  extern int   tcrv_cap_want;
  extern int   tcrv_cap_nblk;
  extern int   tcrv_cap_hits;
  extern int   tcrv_main_i[8][4][8];
  extern int   tcrv_min_i [8][4][8];
  extern float tcrv_d      [8][8];
  extern float tcrv_dmin   [8][8];
  extern float tcrv_ad     [8][4];
  extern short tcrv_bsums  [8][64];
}

/* ---- weight/activation builders (byte-identical to M2/M2c oracle) ---- */
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
static float actval(int r,int k,int intmode){
    if(intmode) return (float)((int)(xr()%9)-4);
    return 0.05f*(float)((int)(xr()%201)-100);
}
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

/* INTEGER per-(block,row,col) oracle: returns the exact integer MAIN (sum sc*w*q8) and
 * MIN (sum m*q8) accumulators the kernel holds for one 256-block, one weight column c,
 * one activation row r -- plus d/dmin/a_d for that (c,r,block). Matches the kernel's v47/v46. */
static void oracle_int_block(const kqr_block_q4_K* xb, const float* xa_block,
                             long* main_i, long* min_i, float* d_o, float* dmin_o, float* ad_o){
    float amax=0,max=0;
    for(int j=0;j<256;j++){ float v=xa_block[j]; if(amax<std::fabs(v)){amax=std::fabs(v); max=v;} }
    float iscale = amax? -127.f/max : 0.f;
    *ad_o = amax? 1.f/iscale : 0.f;
    int8_t q8[256];
    for(int j=0;j<256;j++) q8[j]=(int8_t)nearest_int_(xa_block[j]*iscale);
    *d_o = fp16_to_fp32(xb->d); *dmin_o = fp16_to_fp32(xb->dmin);
    long mains=0, mins=0; uint8_t sc,m; const uint8_t* q=xb->qs; int is=0;
    for(int jg=0;jg<256;jg+=64){
        get_scale_min_k4(is+0,xb->scales,&sc,&m); int sc0=sc,m0=m;
        get_scale_min_k4(is+1,xb->scales,&sc,&m); int sc1=sc,m1=m;
        for(int l=0;l<32;l++){ int w=q[l]&0xF; int a8=q8[jg+l];    mains+=(long)sc0*w*a8; mins+=(long)m0*a8; }
        for(int l=0;l<32;l++){ int w=q[l]>>4;  int a8=q8[jg+32+l]; mains+=(long)sc1*w*a8; mins+=(long)m1*a8; }
        q+=32; is+=2;
    }
    *main_i = mains; *min_i = mins;
}

static std::vector<float> run(ggml_backend_t backend, ggml_backend_buffer_type_t wbuft,
                              int N,int K,int rows,const uint8_t* wbytes,size_t wsize,const float* act,
                              std::vector<uint8_t>* wdata_out){
    ggml_init_params wp{ ggml_tensor_overhead()+256, nullptr, true };
    ggml_context* wctx = ggml_init(wp);
    ggml_tensor* w = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_K, K, N);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, wbuft);
    ggml_backend_tensor_set(w, wbytes, 0, wsize);
    if(wdata_out){ wdata_out->resize(wsize); memcpy(wdata_out->data(), w->data, wsize); }
    ggml_init_params cp{ ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context* cctx = ggml_init(cp);
    ggml_tensor* a = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, K, rows);
    ggml_tensor* out = ggml_mul_mat(cctx, w, a);
    ggml_cgraph* gf = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(a, act, 0, ggml_nbytes(a));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> res((size_t)N*rows);
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(alloc); ggml_free(cctx);
    ggml_backend_buffer_free(wbuf); ggml_free(wctx);
    return res;
}

static int one_case(ggml_backend_t backend,int N,int K,int rows,int intmode,int capture){
    const int nb = K/256;
    RNG = 0x243F6A8885A308D3ull ^ ((uint64_t)(N*1315423911u + K*2654435761u + rows*40503u + intmode));
    std::vector<kqr_block_q4_K> W((size_t)N*nb);
    for(int c=0;c<N;++c) for(int b=0;b<nb;++b) build_w_q4K(&W[(size_t)c*nb+b], c, b, intmode);
    std::vector<uint8_t> wbytes((size_t)N*nb*sizeof(kqr_block_q4_K));
    memcpy(wbytes.data(), W.data(), wbytes.size());
    std::vector<float> act((size_t)rows*K);
    for(int r=0;r<rows;++r) for(int k=0;k<K;++k) act[(size_t)r*K+k]=actval(r,k,intmode);

    std::vector<uint8_t> m1(kqr_bytes_q4_K(N,nb));
    kqr_repack_q4_K(m1.data(), W.data(), N, nb);
    std::vector<uint8_t> ggml_repacked;

    if(capture){ tcrv_cap_nblk=0; tcrv_cap_hits=0; tcrv_cap_want=1; }
    std::vector<float> R = run(backend, ggml_backend_cpu_repack_buffer_type(), N,K,rows,
                               wbytes.data(), wbytes.size(), act.data(), &ggml_repacked);
    if(capture){ tcrv_cap_want=0; }
    std::vector<float> S = run(backend, ggml_backend_get_default_buffer_type(backend), N,K,rows,
                               wbytes.data(), wbytes.size(), act.data(), nullptr);
    int repack_match = (ggml_repacked.size()==m1.size() && memcmp(ggml_repacked.data(),m1.data(),m1.size())==0);

    std::vector<float> F((size_t)N*rows), Fnm((size_t)N*rows);
    for(int r=0;r<rows;++r) for(int c=0;c<N;++c){
        F  [(size_t)r*N+c] = oracle_dot2(&W[(size_t)c*nb], &act[(size_t)r*K], nb, 1);
        Fnm[(size_t)r*N+c] = oracle_dot2(&W[(size_t)c*nb], &act[(size_t)r*K], nb, 0);
    }
    double wR_RF=0, wR_RFnm=0, wR_RS=0;
    for(size_t k=0;k<R.size();++k){
        double dF=std::fabs((double)F[k])>1e-6?std::fabs((double)F[k]):1.0;
        double dFnm=std::fabs((double)Fnm[k])>1e-6?std::fabs((double)Fnm[k]):1.0;
        double dS=std::fabs((double)S[k])>1e-6?std::fabs((double)S[k]):1.0;
        double rF=std::fabs((double)R[k]-(double)F[k])/dF;   if(rF>wR_RF)wR_RF=rF;
        double rN=std::fabs((double)R[k]-(double)Fnm[k])/dFnm; if(rN>wR_RFnm)wR_RFnm=rN;
        double rS=std::fabs((double)R[k]-(double)S[k])/dS;   if(rS>wR_RS)wR_RS=rS;
    }
    printf("  [%s] N=%-4d K=%-5d rows=%d : M1==ggml-repack:%s | R-vs-ORACLE(full) rel=%.3e %s | R-vs-ORACLE(main) rel=%.3e | R-vs-stock rel=%.3e\n",
           intmode?"INT ":"NORM", N,K,rows, repack_match?"YES":"NO ", wR_RF,(wR_RF<1e-3?"OK":"BAD"),wR_RFnm,wR_RS);

    if(capture){
        printf("\n  ===== [G3 M1 BISECT] NORM N=%d K=%d rows=%d : kernel-vs-oracle per (block,row,col) =====\n", N,K,rows);
        printf("  capture hits=%d nblk=%d (expect hits = nblk*1 for group0)\n", tcrv_cap_hits, tcrv_cap_nblk);
        printf("  Per-output-column full ratio R/F (row0, cols 0..7):\n   ");
        for(int c=0;c<8;c++) printf(" c%d=%.3f", c, (double)R[0*N+c]/(double)F[0*N+c]);
        printf("\n");
        /* error distribution: is the 'worst rel' a cancellation artifact (small F) or systematic? */
        {
            int wi=-1; double wabs=0, wF=0; int cnt1=0, cntp1=0; double maxabsRS=0; int osAgree=0, osTot=0;
            for(size_t k=0;k<R.size();++k){
                double aRF=std::fabs((double)R[k]-(double)F[k]);
                if(aRF>wabs){wabs=aRF; wi=(int)k; wF=(double)F[k];}
                double dF=std::fabs((double)F[k])>1e-6?std::fabs((double)F[k]):1.0;
                if(aRF/dF>0.01) cnt1++; if(aRF/dF>0.001) cntp1++;
                double aRS=std::fabs((double)R[k]-(double)S[k]); if(aRS>maxabsRS)maxabsRS=aRS;
                double dS=std::fabs((double)S[k])>1e-6?std::fabs((double)S[k]):1.0;
                double aFS=std::fabs((double)F[k]-(double)S[k]);
                osTot++; if(aFS/dS<1e-3) osAgree++;
            }
            printf("  [dist] R-vs-F: max ABS err=%.4f at idx=%d (F=%.4f, |F| %s); elems rel>1%%: %d/%zu ; rel>0.1%%: %d/%zu\n",
                   wabs, wi, wF, (std::fabs(wF)<5.0?"SMALL->cancellation-amplified":"LARGE->systematic"), cnt1, R.size(), cntp1, R.size());
            printf("  [dist] R-vs-STOCK max ABS err=%.4f ; oracle-vs-STOCK agree(rel<1e-3): %d/%d (LOW => oracle q8 != ggml q8 -> integer bisect noise is a MODELING gap not a kernel bug)\n",
                   maxabsRS, osAgree, osTot);
        }
        /* integer oracle for cols 0..7, rows 0..3, all blocks; compare to kernel capture */
        int nblk = tcrv_cap_nblk; if(nblk> nb) nblk=nb; if(nblk>8) nblk=8;
        int main_bad=0, min_bad=0, d_bad=0, dmin_bad=0, ad_bad=0;
        for(int b=0;b<nblk;b++){
            printf("\n  --- block %d ---\n", b);
            printf("   col:            ");   for(int c=0;c<8;c++) printf("%10d ", c); printf("\n");
            for(int r=0;r<4;r++){
                long om[8], on[8]; float od[8], odm[8], oad=0;
                for(int c=0;c<8;c++){
                    float dd,ddm,aa;
                    oracle_int_block(&W[(size_t)c*nb + b], &act[(size_t)r*K + (size_t)b*256],
                                     &om[c], &on[c], &dd, &ddm, &aa);
                    od[c]=dd; odm[c]=ddm; oad=aa;
                }
                printf("   r%d MAIN kern:   ",r); for(int c=0;c<8;c++) printf("%10d ", tcrv_main_i[b][r][c]); printf("\n");
                printf("   r%d MAIN orac:   ",r); for(int c=0;c<8;c++) printf("%10ld ", om[c]); printf("\n");
                printf("   r%d MIN  kern:   ",r); for(int c=0;c<8;c++) printf("%10d ", tcrv_min_i[b][r][c]); printf("\n");
                printf("   r%d MIN  orac:   ",r); for(int c=0;c<8;c++) printf("%10ld ", on[c]); printf("\n");
                printf("   r%d MIN  d(k/o): ",r); for(int c=0;c<8;c++){ if(tcrv_main_i[b][r][c]!=(int)om[c]) main_bad++; if(tcrv_min_i[b][r][c]!=(int)on[c]) min_bad++; }
                for(int c=0;c<8;c++) printf("%10s ", (tcrv_min_i[b][r][c]==(int)on[c])?"ok":"DIFF"); printf("\n");
                if(r==0){
                    printf("   b%d d    kern:   ",b); for(int c=0;c<8;c++) printf("%10.5f ", tcrv_d[b][c]); printf("\n");
                    printf("   b%d d    orac:   ",b); for(int c=0;c<8;c++) printf("%10.5f ", od[c]); printf("\n");
                    printf("   b%d dmin kern:   ",b); for(int c=0;c<8;c++) printf("%10.5f ", tcrv_dmin[b][c]); printf("\n");
                    printf("   b%d dmin orac:   ",b); for(int c=0;c<8;c++) printf("%10.5f ", odm[c]); printf("\n");
                    printf("   b%d ad(k=%.5f o=%.5f)\n", b, tcrv_ad[b][0], oad);
                    for(int c=0;c<8;c++){ if(std::fabs(tcrv_d[b][c]-od[c])>1e-3) d_bad++; if(std::fabs(tcrv_dmin[b][c]-odm[c])>1e-3) dmin_bad++; }
                }
            }
        }
        /* ---- q8-model-FREE min decomposition: use the kernel's OWN captured bsums ----
         * oracle weight-min m(sub,col) is q8-free (pure weight decode). Recompute
         *   min_int[row][col] = sum_sub m(sub,col) * (kbsums[(2sub)*4+row]+kbsums[(2sub+1)*4+row])
         * and compare to the kernel's captured integer min-acc v46. If BIT-EXACT, the kernel's
         * min-value unpack + bsums pairing + accumulation are correct and the only reason the
         * plain oracle differed was its own q8 (activation) quantization. Also diff kernel bsums
         * vs oracle bsums to localize any activation-side difference. */
        printf("\n  ===== [q8-FREE] min-term decomposition using KERNEL's own captured bsums =====\n");
        int minfree_bad=0, bsum_diff=0, bsum_maxabs=0;
        for(int b=0;b<nblk;b++){
            /* oracle weight mins m(sub,col) for cols 0..7 */
            int mval[8][8]; /* [sub][col] */
            for(int c=0;c<8;c++){ const kqr_block_q4_K* xb=&W[(size_t)c*nb + b];
                for(int sub=0;sub<8;sub++){ uint8_t dd,mm; get_scale_min_k4(sub,xb->scales,&dd,&mm); mval[sub][c]=mm; } }
            /* oracle bsums region from oracle q8: bsums_reg[g16*4+row] = sum_16 q8 */
            short obs[64];
            for(int r=0;r<4;r++){
                const float* xa = &act[(size_t)r*K + (size_t)b*256];
                float amax=0,max=0; for(int j=0;j<256;j++){ float v=xa[j]; if(amax<std::fabs(v)){amax=std::fabs(v);max=v;} }
                float iscale=amax? -127.f/max:0.f;
                for(int g=0;g<16;g++){ int s=0; for(int j=0;j<16;j++) s+= (int)(int8_t)nearest_int_(xa[g*16+j]*iscale); obs[g*4+r]=(short)s; }
            }
            for(int i=0;i<64;i++){ int d=tcrv_bsums[b][i]-obs[i]; if(d){bsum_diff++; if(std::abs(d)>bsum_maxabs)bsum_maxabs=std::abs(d);} }
            for(int r=0;r<4;r++) for(int c=0;c<8;c++){
                long mi=0; for(int sub=0;sub<8;sub++) mi += (long)mval[sub][c]*((long)tcrv_bsums[b][(2*sub)*4+r]+(long)tcrv_bsums[b][(2*sub+1)*4+r]);
                if(mi != (long)tcrv_min_i[b][r][c]) minfree_bad++;
            }
        }
        printf("  [q8-FREE] min_int(oracle-weight-min x KERNEL-bsums) vs kernel v46: mismatches=%d (0 => kernel min math CORRECT given its bsums)\n", minfree_bad);
        printf("  [q8-FREE] kernel-bsums vs oracle-bsums: diffs=%d/%d maxabs=%d (nonzero => activation q8 quant differs oracle-vs-ggml)\n", bsum_diff, nblk*64, bsum_maxabs);

        /* ===== ZERO-MODEL ggml-exact disambiguation: kernel-bug vs q8-quant-path =====
         * Call ggml's OWN mat-quant (kernel input) and row-quant (stock input) on the SAME
         * activation. (a) diff their bsums -> tells if the two ggml q8 paths differ at all.
         * (b) recompute integer main/min from the mat-quant q8 the kernel actually eats and
         * compare BIT-EXACT to the kernel's captured v47/v46 -> tells if the kernel computes
         * q4_K.q8 correctly for its input. */
        printf("\n  ===== [ZERO-MODEL] ggml mat-quant (kernel) vs row-quant (stock), and kernel-vs-mat-quant-recompute =====\n");
        {
            std::vector<blk_q8_Kx4> mq(nb);
            ggml_quantize_mat_q8_K_4x1(act.data(), mq.data(), K);           /* 4 rows interleaved */
            std::vector<blk_q8_K> rq((size_t)4*nb);
            for(int r=0;r<4;r++) quantize_row_q8_K(&act[(size_t)r*K], &rq[(size_t)r*nb], K);
            int matrow_bsum_diff=0, matrow_bsum_max=0, ker_mat_bsum_diff=0;
            int mainexact_bad=0, minexact_bad=0;
            for(int b=0;b<nb;b++){
                /* (a) mat bsums [g16*4+row] vs row bsums [g16] */
                for(int g=0;g<16;g++) for(int r=0;r<4;r++){
                    int md=mq[b].bsums[g*4+r], rd=rq[(size_t)r*nb+b].bsums[g];
                    if(md!=rd){matrow_bsum_diff++; if(std::abs(md-rd)>matrow_bsum_max)matrow_bsum_max=std::abs(md-rd);}
                }
                if(b<nblk) for(int i=0;i<64;i++) if(mq[b].bsums[i]!=tcrv_bsums[b][i]) ker_mat_bsum_diff++;
                /* (b) recompute integer main/min from mat-quant q8 (deinterleave: q8[r][e]=qs[e*4+r]) */
                if(b<nblk) for(int r=0;r<4;r++) for(int c=0;c<8;c++){
                    const kqr_block_q4_K* xb=&W[(size_t)c*nb + b];
                    long mains=0, mins=0; uint8_t sc,m; const uint8_t* q=xb->qs; int is=0;
                    for(int jg=0;jg<256;jg+=64){
                        get_scale_min_k4(is+0,xb->scales,&sc,&m); int sc0=sc,m0=m;
                        get_scale_min_k4(is+1,xb->scales,&sc,&m); int sc1=sc,m1=m;
                        for(int l=0;l<32;l++){ int w=q[l]&0xF; int a8=mq[b].qs[(jg+l)*4+r];    mains+=(long)sc0*w*a8; mins+=(long)m0*a8; }
                        for(int l=0;l<32;l++){ int w=q[l]>>4;  int a8=mq[b].qs[(jg+32+l)*4+r]; mains+=(long)sc1*w*a8; mins+=(long)m1*a8; }
                        q+=32; is+=2;
                    }
                    if(mains != (long)tcrv_main_i[b][r][c]) mainexact_bad++;
                    if(mins  != (long)tcrv_min_i [b][r][c]) minexact_bad++;
                }
            }
            /* (c) element-wise qs: mat (deinterleaved) vs row; + recompute main/min from ROW q8 vs kernel */
            int qs_diff=0, qs_max=0, mainrow_bad=0, minrow_bad=0;
            for(int b=0;b<nb;b++){
                for(int r=0;r<4;r++) for(int e=0;e<256;e++){ int mv=mq[b].qs[e*4+r], rv=rq[(size_t)r*nb+b].qs[e];
                    if(mv!=rv){qs_diff++; if(std::abs(mv-rv)>qs_max)qs_max=std::abs(mv-rv);} }
                if(b<nblk) for(int r=0;r<4;r++) for(int c=0;c<8;c++){
                    const kqr_block_q4_K* xb=&W[(size_t)c*nb + b]; const int8_t* q8=rq[(size_t)r*nb+b].qs;
                    long mains=0, mins=0; uint8_t sc,m; const uint8_t* q=xb->qs; int is=0;
                    for(int jg=0;jg<256;jg+=64){ get_scale_min_k4(is+0,xb->scales,&sc,&m); int sc0=sc,m0=m;
                        get_scale_min_k4(is+1,xb->scales,&sc,&m); int sc1=sc,m1=m;
                        for(int l=0;l<32;l++){ int w=q[l]&0xF; int a8=q8[jg+l];    mains+=(long)sc0*w*a8; mins+=(long)m0*a8; }
                        for(int l=0;l<32;l++){ int w=q[l]>>4;  int a8=q8[jg+32+l]; mains+=(long)sc1*w*a8; mins+=(long)m1*a8; }
                        q+=32; is+=2; }
                    if(mains!=(long)tcrv_main_i[b][r][c]) mainrow_bad++;
                    if(mins !=(long)tcrv_min_i [b][r][c]) minrow_bad++;
                }
            }
            printf("  [ZERO-MODEL] sizeof(blk_q8_K)=%zu (ggml expect 292) sizeof(blk_q8_Kx4)=%zu (expect 1168)\n", sizeof(blk_q8_K), sizeof(blk_q8_Kx4));
            printf("  [ZERO-MODEL] (a) ggml mat-quant bsums vs ggml row-quant bsums: diffs=%d/%d maxabs=%d\n", matrow_bsum_diff, nb*64, matrow_bsum_max);
            printf("               kernel captured bsums vs ggml mat-quant bsums: diffs=%d (0 => kernel eats exactly ggml mat-quant)\n", ker_mat_bsum_diff);
            printf("  [ZERO-MODEL] (c) mat-quant qs (deint) vs row-quant qs element-wise: diffs=%d/%d maxabs=%d (0 => two ggml q8 IDENTICAL)\n", qs_diff, nb*4*256, qs_max);
            printf("               kernel v47 MAIN vs recompute-from-ROW-q8: mismatches=%d | kernel v46 MIN vs ROW: mismatches=%d\n", mainrow_bad, minrow_bad);
            printf("  [ZERO-MODEL] (b) kernel v47 MAIN vs recompute-from-mat-q8: mismatches=%d  |  kernel v46 MIN vs recompute: mismatches=%d\n", mainexact_bad, minexact_bad);
            /* (d) scan for first mat-vs-row q8 divergences and dump full context per differing (row,block) */
            {
                printf("  [ZERO-MODEL] (d) first mat(generic)-vs-row(riscv SIMD) q8 divergences, with per-row d and act:\n");
                int shown=0;
                for(int b=0;b<nb && shown<6;b++) for(int r=0;r<4 && shown<6;r++){
                    /* count diffs for this (r,b) */
                    int nd=0, firste=-1; for(int e=0;e<256;e++){ if(mq[b].qs[e*4+r]!=rq[(size_t)r*nb+b].qs[e]){ nd++; if(firste<0)firste=e; } }
                    if(nd==0) continue;
                    const float* xa=&act[(size_t)r*K + (size_t)b*256]; float amax=0,mx=0; for(int j=0;j<256;j++){float v=xa[j]; if(amax<std::fabs(v)){amax=std::fabs(v);mx=v;}}
                    printf("    (row=%d blk=%d) qs-diffs=%d/256 firstdiff-elt=%d | oracle max=%.5f 1/iscale=%.6f  mat.d=%.6f  row.d=%.6f\n",
                           r,b,nd,firste, mx, (mx?-1.f/(-127.f/mx):0.f), mq[b].d[r], rq[(size_t)r*nb+b].d);
                    printf("       around elt %d:  e:act / orac / mat / row\n", firste);
                    for(int e=firste; e<firste+6 && e<256; e++){
                        float isc = amax? -127.f/mx:0.f;
                        printf("         %2d: %8.4f / %4d / %4d / %4d\n", e, xa[e], nearest_int_(xa[e]*isc), (int)mq[b].qs[e*4+r], (int)rq[(size_t)r*nb+b].qs[e]);
                    }
                    shown++;
                }
                if(shown==0) printf("       (no mat-vs-row divergence found in captured rows/blocks)\n");
            }
            printf("               (both 0 => kernel computes q4_K.q8 BIT-EXACT for its input; discrepancy vs stock/oracle is the mat-vs-row q8 path, NOT a kernel defect)\n");
        }

        printf("\n  ===== BISECT VERDICT: MAIN mismatches=%d  MIN mismatches=%d  d mismatches=%d  dmin mismatches=%d | q8FREE-minmath-bad=%d  bsum-diffs=%d =====\n",
               main_bad, min_bad, d_bad, dmin_bad, minfree_bad, bsum_diff);
        if(min_bad>0)  printf("  -> MIN integer accumulator (v46) DIVERGES from oracle: defect is UPSTREAM of fp16 fold (min-value unpack or bsums pairing).\n");
        if(dmin_bad>0) printf("  -> dmin fp16->f32 per-column load (v5762) DIVERGES: defect is the dmin COLUMN application in the float fold.\n");
        if(min_bad==0 && dmin_bad==0 && main_bad==0) printf("  -> ALL captured integer/float intermediates MATCH oracle for cols 0..7 (min defect NOT in the captured tile0 quantities -- see notes).\n");
    }
    return (repack_match && (wR_RF<1e-3))?0:1;
}

int main(void){
    fprintf(stderr,"VLEN(bits)=%d\n",(int)(__riscv_vlenb()*8)); fflush(stderr);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    printf("=== M1 q4_K repack-GEMM min-term BISECT through REAL ggml_mul_mat dispatch (VLEN128) ===\n");
    int fail=0;
    fail |= one_case(backend, 64,  512,  4, 1, 0);
    fail |= one_case(backend, 64,  512,  4, 0, 1);   /* <-- capture this one */
    fail |= one_case(backend, 160, 2560, 4, 0, 0);
    printf("\nSUMMARY: %s\n", fail?"NUMERIC-DIVERGENCE-PRESENT (expected: min-term bug)":"clean");
    ggml_backend_free(backend);
    return 0;
}
