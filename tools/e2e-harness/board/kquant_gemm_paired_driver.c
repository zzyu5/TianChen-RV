/* kquant_gemm_paired_driver.c — [KQUANT-L1] prefill paired A/B on rvv/VLEN128.
 *
 * Times OUR exported repack GEMM (q4_K or q5_K, front-door lowered from HEAD 039133ea,
 * compiled from the .kernel.c exported by tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate)
 * against the OPPONENT's REAL dispatched path at VLEN128: ggml_vec_dot_q{4,5}_K_q8_K, linked
 * directly from the board's own libggml-cpu.so (build-gcc15-rv64gcv). The dispatch probe proved
 * that at VLEN=128 ggml's repack trait selector returns NULLPTR for q4_K (case 128 = TODO no-op)
 * and q5_K (no riscv branch), so ggml's prefill mul_mat falls back to exactly this per-(row,col)
 * block-dot — the fair opponent for a prefill GEMM.
 *
 * Both sides compute the SAME logical matmul shape (K x nr x nc), macs = nr*nc*K. The control
 * flow of both the repack kernel and the K-quant block-dot is DATA-INDEPENDENT (K-quant decode
 * has no data branches; ggml's internal vl128/vl256 split is by __riscv_vlenb(), a constant), so
 * random byte fills give valid steady-state GMAC/s. Correctness of OUR kernel is established
 * separately by the board oracle at construction (bounded-norm PASS, this is a THROUGHPUT probe).
 *
 * [NG-4] DISCIPLINE: this is an L1 path-candidate datapoint (opponent has no working repack@128,
 * we do). It is NOT a [PERF-1] eight-gate beat — the eight gates are not walked here. Reported as
 * a gap, not a win.
 *
 * argv: <fmt=q4_K|q5_K> <K(mult256)> <nr(mult4)> <nc(mult16)> <iters> <seed>
 * stdout: KQGEMM fmt=.. VLEN=.. K=.. nr=.. nc=.. ours_ns=.. ours_gmacs=.. opp_ns=.. opp_gmacs=.. ratio_ours_over_opp=..
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK_K 256
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;
typedef struct { ggml_half d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;          /* 144 */
typedef struct { ggml_half d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K; /* 176 */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;                             /* 292 */
_Static_assert(sizeof(block_q4_K)==144,"q4_K");
_Static_assert(sizeof(block_q5_K)==176,"q5_K");
_Static_assert(sizeof(block_q8_K)==292,"q8_K");

/* OUR exported repack GEMM kernels: (n, s, vx=weights, vy=activations, nr, nc, bs) */
extern void tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
extern void tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);

/* OPPONENT: real ggml block-dot, linked from libggml-cpu.so */
extern void ggml_vec_dot_q4_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
extern void ggml_vec_dot_q5_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static void fill_rand(uint8_t* p, size_t n){ for(size_t i=0;i<n;i++) p[i]=(uint8_t)(xr()&0xff); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }

int main(int argc, char** argv){
    if(argc<7){ fprintf(stderr,"usage: %s q4_K|q5_K K nr nc iters seed\n",argv[0]); return 2; }
    const char* fmt=argv[1];
    int K=atoi(argv[2]), nr=atoi(argv[3]), nc=atoi(argv[4]), iters=atoi(argv[5]);
    rng=(uint64_t)strtoull(argv[6],0,0)|1ull;
    int is_q5 = !strcmp(fmt,"q5_K");
    if(!is_q5 && strcmp(fmt,"q4_K")){ fprintf(stderr,"fmt must be q4_K or q5_K\n"); return 2; }
    if(K%QK_K||nr%4||nc%16){ fprintf(stderr,"K%%256, nr%%4, nc%%16 required\n"); return 2; }
    long vlen_bits=(long)__riscv_vlenb()*8;
    int nb=K/QK_K, grp_c=nc/16, grp_r=nr/4;

    /* OUR side: repacked interleaved buffers (content arbitrary; sizes/strides exact) */
    size_t wstride = is_q5 ? 2816 : 2304;         /* block_qX_Kx16 group stride */
    size_t astride = 1168;                        /* block_q8_Kx4 group stride  */
    size_t wbytes=(size_t)grp_c*nb*wstride, abytes=(size_t)grp_r*nb*astride;
    uint8_t* Wr=aligned_alloc(64,wbytes);
    uint8_t* Ar=aligned_alloc(64,abytes);
    float*   Or=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    /* OPPONENT side: plain per-row ggml blocks + q8_K activation rows */
    size_t wblk = is_q5 ? sizeof(block_q5_K) : sizeof(block_q4_K);
    uint8_t* Wo=aligned_alloc(64,(size_t)nc*nb*wblk);
    block_q8_K* Ao=aligned_alloc(64,(size_t)nr*nb*sizeof(block_q8_K));
    float*   Oo=aligned_alloc(64,(size_t)nr*nc*sizeof(float));
    if(!Wr||!Ar||!Or||!Wo||!Ao||!Oo){ fprintf(stderr,"OOM\n"); return 3; }
    fill_rand(Wr,wbytes); fill_rand(Ar,abytes);
    fill_rand(Wo,(size_t)nc*nb*wblk); fill_rand((uint8_t*)Ao,(size_t)nr*nb*sizeof(block_q8_K));
    /* Make all fp16/fp32 SCALE fields finite-and-small so the fp accumulation stays a real number
     * (the integer decode+vwmacc dominates runtime and is data-independent; this only keeps the DCE
     * sink finite and defends the timing against any denormal/nan objection). Layouts are exact. */
    const uint16_t H = 0x2C00;   /* fp16 0.0625 */
    /* our repacked weights: per group/block, d strip @[0,32), dmin strip @[32,64) */
    for(int g=0;g<grp_c;g++) for(int l=0;l<nb;l++){ uint16_t* p=(uint16_t*)(Wr+((size_t)g*nb+l)*wstride);
        for(int j=0;j<32;j++) p[j]=H; }
    /* our repacked activations: per group/block, 4 fp32 d @[0,16) */
    for(int g=0;g<grp_r;g++) for(int l=0;l<nb;l++){ float* p=(float*)(Ar+((size_t)g*nb+l)*astride);
        for(int j=0;j<4;j++) p[j]=0.01f; }
    /* opponent weights: block d,dmin (first 2 fp16) */
    for(size_t i=0;i<(size_t)nc*nb;i++){ uint16_t* p=(uint16_t*)(Wo+i*wblk); p[0]=H; p[1]=H; }
    /* opponent activations: q8_K d (fp32) small-positive */
    for(size_t i=0;i<(size_t)nr*nb;i++) Ao[i].d=0.01f;

    /* ---- OUR repack GEMM: warmup + best-of-5 ---- */
    for(int w=0;w<3;w++){
        if(is_q5) tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc);
        else      tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc);
    }
    double ours_best=1e30; volatile double sink=0;
    for(int pass=0;pass<5;pass++){
        double t0=now_ns();
        for(int it=0;it<iters;it++){
            if(is_q5) tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc);
            else      tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K((size_t)K,Or,Wr,Ar,(size_t)nr,(size_t)nc,(size_t)nc);
        }
        double npc=(now_ns()-t0)/(double)iters;
        if(npc<ours_best) ours_best=npc;
        sink+=Or[0]+Or[(size_t)nr*nc-1];
    }

    /* ---- OPPONENT block-dot GEMM: warmup + best-of-5 ---- */
    #define OPP_GEMM() do{ for(int c=0;c<nc;c++){ const uint8_t* wc=Wo+(size_t)c*nb*wblk; \
        for(int r=0;r<nr;r++){ const block_q8_K* ar=Ao+(size_t)r*nb; \
            if(is_q5) ggml_vec_dot_q5_K_q8_K(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); \
            else      ggml_vec_dot_q4_K_q8_K(K,&Oo[(size_t)r*nc+c],0,wc,0,ar,0,1); } } }while(0)
    for(int w=0;w<2;w++){ OPP_GEMM(); }
    double opp_best=1e30;
    for(int pass=0;pass<5;pass++){
        double t0=now_ns();
        for(int it=0;it<iters;it++){ OPP_GEMM(); }
        double npc=(now_ns()-t0)/(double)iters;
        if(npc<opp_best) opp_best=npc;
        sink+=Oo[0]+Oo[(size_t)nr*nc-1];
    }

    double macs=(double)nr*(double)nc*(double)K;
    double ours_g=macs/ours_best, opp_g=macs/opp_best;
    printf("KQGEMM fmt=%s VLEN=%ld K=%d nr=%d nc=%d iters=%d ours_ns=%.1f ours_gmacs=%.4f opp_ns=%.1f opp_gmacs=%.4f ratio_ours_over_opp=%.3f sink=%.1f\n",
           fmt,vlen_bits,K,nr,nc,iters,ours_best,ours_g,opp_best,opp_g,ours_g/opp_g,(double)sink);
    free(Wr);free(Ar);free(Or);free(Wo);free(Ao);free(Oo);
    return 0;
}
