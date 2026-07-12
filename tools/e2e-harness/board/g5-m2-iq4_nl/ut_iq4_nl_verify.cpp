// [G5-M2 iq4_nl] MIRAGE de-risk silicon UT (board rvv, VLEN128).
// Proves OUR EMITTED vl=8 iq4_nl repack GEVM + GEMM kernels, reading the UPSTREAM
// make_block_iq4_nlx16 interleaved weight layout (stride 288) + the block_q8_0x4
// activation layout (stride 136) / plain block_q8_0 (stride 34), are BYTE-EXACT to an
// INDEPENDENT scalar iq4_nl x q8_0 codebook dot oracle.
//   * INT test  : all d = fp16(1.0) -> pure i32 isum, exact in float (bounded shapes) ->
//                 INT-mismatch MUST be 0 (byte-exact integer). This is the hard gate.
//   * NORM test : adversarial random fp16 d, f64 reference, bounded relative error.
// If INT-mismatch != 0 the interleaver/kernel is wrong -> STOP (do NOT build/deploy).
//
// Weight interleave == upstream ggml make_block_iq4_nlx16 (repack.cpp:3691):
//     out.d[c] = in[c].d;   out.qs[j*16 + c] = in[c].qs[j]   (c 0..15 col, j 0..15 byte)
// Activation x4 interleave (block_q8_0x4, stride 136): qs[e*4 + m] = plain[m].qs[e], d[m]=plain[m].d
// iq4_nl decode: elem e in 0..15 -> kvalues[qs[e]&0xf]; elem e+16 -> kvalues[qs[e]>>4].
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <riscv_vector.h>

#define QK 32
using half = _Float16;

static const int8_t kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

// OUR emitted kernels (current-HEAD ABI)
//   GEVM: (size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)
//   GEMM: (size_t nr, size_t bs, size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)
extern "C" void tcrv_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);
extern "C" void tcrv_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(
    size_t nr, size_t bs, size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);

// plain blocks
struct block_iq4_nl { uint16_t d; uint8_t qs[QK/2]; };          // 18
struct block_q8_0    { uint16_t d; int8_t  qs[QK];   };          // 34
struct block_iq4_nlx16 { uint16_t d[16]; uint8_t qs[QK*8]; };    // 288
struct block_q8_0x4    { uint16_t d[4];  int8_t  qs[QK*4]; };    // 136

static uint64_t rng;
static uint32_t xr(){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static uint16_t f2h(float f){ half h=(half)f; uint16_t u; memcpy(&u,&h,2); return u; }
static float    h2f(uint16_t u){ half h; memcpy(&h,&u,2); return (float)h; }

// upstream make_block_iq4_nlx16 (byte-identical to repack.cpp:3691, blck_size_interleave==1)
static block_iq4_nlx16 make_wx16(const block_iq4_nl in[16]){
    block_iq4_nlx16 out;
    for(int c=0;c<16;c++) out.d[c]=in[c].d;
    const int end = QK*8; // 256
    for(int i=0;i<end;i++){ int src_id=i%16, src_off=i/16; out.qs[i]=in[src_id].qs[src_off]; }
    return out;
}
// block_q8_0x4 (qs[e*4+m], d[m]) -- matches emitted GEMM act indexing (offset e*4+m)
static block_q8_0x4 make_ax4(const block_q8_0 in[4]){
    block_q8_0x4 out;
    for(int m=0;m<4;m++) out.d[m]=in[m].d;
    for(int e=0;e<QK;e++) for(int m=0;m<4;m++) out.qs[e*4+m]=in[m].qs[e];
    return out;
}
// independent scalar oracle: iq4_nl weight column x q8_0 act row, over nb blocks.
static double oracle_dot(const block_iq4_nl* wcol, const block_q8_0* arow, int nb){
    double sumf=0.0;
    for(int l=0;l<nb;l++){
        long isum=0;
        for(int e=0;e<QK/2;e++){
            int lo=kvalues_iq4nl[wcol[l].qs[e]&0xf];
            int hi=kvalues_iq4nl[wcol[l].qs[e]>>4];
            isum += (long)arow[l].qs[e]      * lo;
            isum += (long)arow[l].qs[e+QK/2] * hi;
        }
        sumf += (double)h2f(wcol[l].d) * (double)h2f(arow[l].d) * (double)isum;
    }
    return sumf;
}

int main(int argc,char**argv){
    rng = (argc>1? strtoull(argv[1],0,0):20260712ull)|1ull;
    long vlen=(long)__riscv_vlenb()*8;
    printf("[ut_iq4_nl] VLEN=%ld seed=%llu\n", vlen, (unsigned long long)rng);

    // shapes: {nb, nc(mult16), nr(mult4)}
    struct Sh{int nb,nc,nr;};
    std::vector<Sh> shapes = {{2,16,4},{4,32,4},{4,16,8},{8,16,4},{2,48,4},{4,16,16}};

    long int_mismatch_total=0;
    double worst_norm=0.0, worst_int=0.0;

    for(int mode=0;mode<2;mode++){ // 0=INT(d=1) 1=NORM(rand d)
        const char* mn = mode? "NORM" : "INT ";
        for(auto sh: shapes){
            int nb=sh.nb, nc=sh.nc, nr=sh.nr;
            // plain weights per column, plain acts per row
            std::vector<block_iq4_nl> W((size_t)nc*nb);
            std::vector<block_q8_0>    A((size_t)nr*nb);
            for(auto&b:W){ b.d = mode? f2h((h2f(0x3C00)*0.0f)+0.05f*(1+(xr()&7))) : 0x3C00;
                           for(auto&q:b.qs) q=(uint8_t)(xr()&0xff); }
            for(auto&b:A){ b.d = mode? f2h(0.05f*(1+(xr()&7))) : 0x3C00;
                           for(int e=0;e<QK;e++) b.qs[e]=(int8_t)((mode? (xr()%255)-127 : (xr()%17)-8)); }
            // interleave weight (all nc columns) and act-x4 (all nr rows)
            int gc=nc/16, gr=nr/4;
            std::vector<block_iq4_nlx16> Wx((size_t)gc*nb);
            for(int g=0;g<gc;g++) for(int l=0;l<nb;l++){
                block_iq4_nl tmp[16];
                for(int c=0;c<16;c++) tmp[c]=W[(size_t)(g*16+c)*nb + l];
                Wx[(size_t)g*nb + l]=make_wx16(tmp);
            }
            std::vector<block_q8_0x4> Ax((size_t)gr*nb);
            for(int g=0;g<gr;g++) for(int l=0;l<nb;l++){
                block_q8_0 tmp[4];
                for(int m=0;m<4;m++) tmp[m]=A[(size_t)(g*4+m)*nb + l];
                Ax[(size_t)g*nb + l]=make_ax4(tmp);
            }
            int n=nb*QK;
            // --- GEVM: row 0 vs all columns (plain act row 0) ---
            std::vector<float> og(nc,0.f);
            tcrv_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0(
                (size_t)n, og.data(), (const uint8_t*)Wx.data(), (const uint8_t*)&A[0], (size_t)nc);
            for(int c=0;c<nc;c++){
                double ref = oracle_dot(&W[(size_t)c*nb], &A[0], nb);
                double diff = std::fabs((double)og[c]-ref);
                double den = std::fabs(ref)+1e-9;
                if(mode==0){ if(diff>0.5){int_mismatch_total++;} worst_int=std::max(worst_int,diff); }
                else worst_norm=std::max(worst_norm,diff/den);
            }
            // --- GEMM: nr rows x nc cols ---
            std::vector<float> om((size_t)nr*nc,0.f);
            tcrv_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(
                (size_t)nr,(size_t)nc,(size_t)n, om.data(),
                (const uint8_t*)Wx.data(), (const uint8_t*)Ax.data(), (size_t)nc);
            for(int r=0;r<nr;r++) for(int c=0;c<nc;c++){
                double ref = oracle_dot(&W[(size_t)c*nb], &A[(size_t)r*nb], nb);
                double diff = std::fabs((double)om[(size_t)r*nc+c]-ref);
                double den = std::fabs(ref)+1e-9;
                if(mode==0){ if(diff>0.5){int_mismatch_total++;} worst_int=std::max(worst_int,diff); }
                else worst_norm=std::max(worst_norm,diff/den);
            }
            printf("  %s nb=%d nc=%d nr=%d : gevm+gemm checked\n", mn, nb, nc, nr);
        }
    }
    printf("== INT_mismatch_total=%ld worst_int_absdiff=%.3f worst_norm_reldiff=%.3e ==\n",
           int_mismatch_total, worst_int, worst_norm);
    if(int_mismatch_total==0 && worst_norm < 1e-4){
        printf("UT_VERDICT: SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM (GREEN)\n"); return 0;
    }
    printf("UT_VERDICT: RED (mismatch)\n"); return 1;
}
