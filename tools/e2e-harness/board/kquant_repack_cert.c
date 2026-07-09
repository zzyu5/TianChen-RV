/* kquant_repack_cert.c -- [SEL-1 T4b / M1] GENERAL per-tensor byte-exact / bounded-ULP certificate
 * for the kquant_repacker.h offline repacker, over ARBITRARY GEMM shapes, for q4_K AND q5_K.
 * Generalizes the M0 one-shot q4_K verifier (kquant_repack_verify_q4K.c, commit 663214e9) to any
 * shape and both K-quant formats, driving the SHARED repacker (kqr_repack_q4_K / kqr_repack_q5_K /
 * kqr_interleave_q8_K).
 *
 * For each (format, shape): synthesize original ggml blocks -> repack with kquant_repacker.h ->
 * run OUR golden emitted repack-GEMM kernel (whole mul_mat, one call) -> compare against the board's
 * OWN stock ggml integer path grid (ggml_vec_dot_q{4,5}_K_q8_K per (row,col), the exact block-dot
 * ggml's prefill mul_mat dispatches to at VLEN128 where the K-quant repack trait is a NULL no-op).
 * ggml is ground truth; NO hand-rolled reference. NOT a perf probe.
 *
 * TWO certificates per shape:
 *   (INT)  unit d=1.0 (0x3C00), dmin=0 (min term vanishes), magnitude-bounded integer core =>
 *          both sides fold an EXACT integer in fp32 => require Or[k]==Oref[k] bit-for-bit (0 ULP).
 *          BYTE-EXACT-INTEGER-vs-stock-ggml iff 0 mismatch. Exercises the repacked scale-split,
 *          nibble interleave, (q5_K) qh 5th-bit plane, and the q8_K activation interleave.
 *   (NORM) adversarial fp16 d AND dmin (min*bsums term ACTIVE), real fp32 activation d, true per-16
 *          bsums. fp32 fold-order differs (our vector kernel vs ggml scalar loop) => small ULP
 *          EXPECTED (pure fp32 reassociation, benign iff rel small). Reports worst ULP + worst rel.
 *
 * Build: clang-17 -O2 -march=rv64gcv_zfh_zvfh... -x c++ (both golden kernels) + this ; link the
 *        board's libggml-cpu.so (-lggml-cpu -lggml-base -lggml -lm). See kquant_repack_cert.sh.
 * argv: [fmt=all|q4_K|q5_K] [seed]
 */
#include "kquant_repacker.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

/* OUR exported repack-GEMM kernels (n, s, vx=repacked weights, vy=interleaved acts, nr, nc, bs) */
extern "C" void tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
extern "C" void tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
/* STOCK ggml block-dots dispatched at VLEN128 (reference), linked from libggml-cpu.so */
extern "C" void ggml_vec_dot_q4_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                                       const void* vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                                       const void* vy, size_t by, int nrc);

static int64_t okey(float f){ uint32_t u; memcpy(&u,&f,4); return (u&0x80000000u)?-(int64_t)(u&0x7fffffffu):(int64_t)u; }
static int64_t ulp(float a,float b){ int64_t d=okey(a)-okey(b); return d<0?-d:d; }
static uint64_t RNG;
static uint32_t xr(void){ RNG^=RNG<<13; RNG^=RNG>>7; RNG^=RNG<<17; return (uint32_t)(RNG>>32); }
/* small positive fp16 bit patterns (exact, no float->half encoder needed) */
static const uint16_t DHALF[8] = {0x2800,0x2C00,0x3000,0x3400,0x2400,0x2A00,0x3200,0x2600};

/* pack the ggml on-disk scales[12] from 8 sub-scales + 8 sub-mins (inverse of get_scale_min_k4) */
static void pack_disk_scales(uint8_t* sc12, const uint8_t sc[8], const uint8_t mn[8]){
    for(int j=0;j<4;++j){ sc12[j]=sc[j]; sc12[j+4]=mn[j]; }
    for(int j=4;j<8;++j){
        sc12[j+4] = (uint8_t)((sc[j]&0x0F) | ((mn[j]&0x0F)<<4));
        sc12[j-4] = (uint8_t)((sc12[j-4]&0x3F) | ((sc[j]>>4)<<6));
        sc12[j-0] = (uint8_t)((sc12[j-0]&0x3F) | ((mn[j]>>4)<<6));
    }
}
/* fill the shared q4_K/q5_K header (d,dmin,scales) + qs; qh filled by caller for q5_K */
static void fill_hdr(uint16_t* d, uint16_t* dmin, uint8_t* sc12, uint8_t* qs,
                     int col, int blk, int intmode){
    if(intmode){ *d=0x3C00; *dmin=0x0000; }
    else       { *d=DHALF[(col+blk)&7]; *dmin=DHALF[(col+3*blk+1)&7]; }
    uint8_t sc[8], mn[8];
    for(int j=0;j<8;++j){
        sc[j]= intmode ? (uint8_t)(1+((j+col+blk)%3)) : (uint8_t)(xr()&63);
        mn[j]= intmode ? 0                            : (uint8_t)(xr()&63);
    }
    pack_disk_scales(sc12, sc, mn);
    for(int i=0;i<KQR_QK_K/2;++i){
        uint8_t lo = intmode ? (uint8_t)((i+col)%13)     : (uint8_t)(xr()&0x0F);
        uint8_t hi = intmode ? (uint8_t)((i+2*blk+1)%13) : (uint8_t)(xr()&0x0F);
        qs[i] = (uint8_t)((lo&0x0F) | ((hi&0x0F)<<4));
    }
}
static void build_w_q4K(kqr_block_q4_K* x, int col, int blk, int intmode){
    fill_hdr(&x->d,&x->dmin,x->scales,x->qs,col,blk,intmode);
}
static void build_w_q5K(kqr_block_q5_K* x, int col, int blk, int intmode){
    fill_hdr(&x->d,&x->dmin,x->scales,x->qs,col,blk,intmode);
    for(int m=0;m<KQR_QK_K/8;++m)
        x->qh[m] = intmode ? (uint8_t)((m*7+col+blk)&0xFF) : (uint8_t)(xr()&0xFF);
}
static void build_a(kqr_block_q8_K* a, int blk, int intmode){
    a->d = intmode ? 1.0f : (0.008f + 0.0013f*(float)(blk%7));
    for(int g=0;g<KQR_QK_K/16;++g){ int sum=0;
        for(int i=0;i<16;++i){
            int v = intmode ? ((int)(xr()%9)-4) : ((int)(xr()%181)-90);
            a->qs[g*16+i]=(int8_t)v; sum+=v;
        }
        a->bsums[g]=(int16_t)sum;
    }
}

struct Sh{ int nr,nc,n; };

static int run_fmt(const char* fmt, unsigned seed,
                   long long& gWorstIntMism, int& gWorstUlp, double& gWorstRel){
    int isQ5 = !strcmp(fmt,"q5_K");
    /* mixed shapes: small/large, non-power-of-2 nc, multi-block K, tall/wide -- generality proof */
    std::vector<Sh> shM = {{4,16,256},{4,32,512},{8,64,256},{12,48,768},{16,256,512},{20,80,1024},{4,160,2560}};
    int fail=0;
    for(int mode=0;mode<2;++mode){
        printf("\n  == %s / %s ==\n", fmt,
               mode? "NORM (adversarial fp16 d+dmin, min*bsums ACTIVE, bounded-ULP vs stock ggml)"
                   : "INT  (unit d, dmin=0, magnitude-bounded, BYTE-EXACT-INTEGER vs stock ggml)");
        for(auto s: shM){
            int nr=s.nr, nc=s.nc, n=s.n, nb=n/KQR_QK_K;
            RNG = ((uint64_t)seed<<1 | 1ull) ^ ((uint64_t)(nr*131+nc*7+n) * 0x9E3779B97F4A7C15ull)
                  ^ ((uint64_t)(isQ5?0xA5:0x5A)<<40);
            std::vector<float> Ours((size_t)nr*nc,-123.0f), Ref((size_t)nr*nc,-456.0f);
            std::vector<kqr_block_q8_K> A((size_t)nr*nb);
            for(int r=0;r<nr;++r)for(int b=0;b<nb;++b) build_a(&A[(size_t)r*nb+b], r*31+b, !mode);
            std::vector<uint8_t> Ar(kqr_bytes_q8_K(nr,nb));
            kqr_interleave_q8_K(Ar.data(), A.data(), nr, nb);
            std::vector<uint8_t> Wr;

            if(!isQ5){
                std::vector<kqr_block_q4_K> W((size_t)nc*nb);
                for(int c=0;c<nc;++c)for(int b=0;b<nb;++b) build_w_q4K(&W[(size_t)c*nb+b],c,b,!mode);
                Wr.resize(kqr_bytes_q4_K(nc,nb));
                kqr_repack_q4_K(Wr.data(), W.data(), nc, nb);
                tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
                    (size_t)n, Ours.data(), Wr.data(), Ar.data(), (size_t)nr, (size_t)nc, (size_t)nc);
                for(int c=0;c<nc;++c)for(int r=0;r<nr;++r)
                    ggml_vec_dot_q4_K_q8_K(n,&Ref[(size_t)r*nc+c],0,&W[(size_t)c*nb],0,&A[(size_t)r*nb],0,1);
            } else {
                std::vector<kqr_block_q5_K> W((size_t)nc*nb);
                for(int c=0;c<nc;++c)for(int b=0;b<nb;++b) build_w_q5K(&W[(size_t)c*nb+b],c,b,!mode);
                Wr.resize(kqr_bytes_q5_K(nc,nb));
                kqr_repack_q5_K(Wr.data(), W.data(), nc, nb);
                tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(
                    (size_t)n, Ours.data(), Wr.data(), Ar.data(), (size_t)nr, (size_t)nc, (size_t)nc);
                for(int c=0;c<nc;++c)for(int r=0;r<nr;++r)
                    ggml_vec_dot_q5_K_q8_K(n,&Ref[(size_t)r*nc+c],0,&W[(size_t)c*nb],0,&A[(size_t)r*nb],0,1);
            }

            long long mism=0; int wU=0; double wR=0;
            for(size_t k=0;k<(size_t)nr*nc;++k){
                float a=Ours[k], b=Ref[k];
                if(!mode){ if(memcmp(&a,&b,4)!=0) ++mism; }
                int u=(int)ulp(a,b); if(u>wU)wU=u;
                double den=std::fabs((double)b);
                double rel = den>1e-9 ? std::fabs((double)a-(double)b)/den : std::fabs((double)a-(double)b);
                if(rel>wR)wR=rel;
            }
            if(!mode){
                if(mism>gWorstIntMism)gWorstIntMism=mism; if(mism)fail=1;
                printf("     GEMM nr=%-3d nc=%-4d K=%-5d : int_mismatch=%lld  worstULP=%d   %s\n",
                       nr,nc,n,mism,wU, mism? "*** INT MISMATCH ***":"byte-exact-int OK");
            } else {
                if(wU>gWorstUlp)gWorstUlp=wU; if(wR>gWorstRel)gWorstRel=wR; if(wR>1e-3)fail=1;
                printf("     GEMM nr=%-3d nc=%-4d K=%-5d : worstULP=%-4d worstRel=%.3e   %s\n",
                       nr,nc,n,wU,wR, wR>1e-3? "*** REL TOO LARGE ***":"bounded-ULP OK (fp32 reassoc)");
            }
        }
    }
    return fail;
}

int main(int argc,char**argv){
    std::string which = argc>1 ? argv[1] : "all";
    unsigned seed = argc>2 ? (unsigned)strtoul(argv[2],0,0) : 20260709u;
    printf("# M1 GENERAL K-quant repacker per-tensor certificate  (repacker=kquant_repacker.h)\n");
    printf("#   q4_K kernel md5 b0b5beac (block_q4_Kx16/2304) ; q5_K kernel md5 ba30ba54 (block_q5_Kx16/2816)\n");
    printf("#   ref = STOCK ggml_vec_dot_q{4,5}_K_q8_K (VLEN128 block-dot, repack trait NULL)\n");

    long long worstIntMism=0; int worstUlp=0; double worstRel=0; int fail=0;
    if(which=="all"||which=="q4_K") fail |= run_fmt("q4_K",seed,worstIntMism,worstUlp,worstRel);
    if(which=="all"||which=="q5_K") fail |= run_fmt("q5_K",seed,worstIntMism,worstUlp,worstRel);

    printf("\n# SUMMARY: worst_int_mismatch=%lld  worst_norm_ULP=%d  worst_norm_rel=%.3e  =>  %s\n",
           worstIntMism, worstUlp, worstRel, fail? "FAIL":"PASS");
    return fail?1:0;
}
