// q8_0 repack GEVM — Win-B·micro: OUR compiler-emitted q8_0 repack GEVM (block-as-lane,
// decode, the NARROW = mf2/VLEN128 arm the SG2044 tune actually ships) vs ggml's OWN
// shipped RVV kernel for q8_0, ggml_vec_dot_q8_0_q8_0 (quants.c:435). This is the
// algorithm-change contribution per N3-METHODOLOGY: Win-B baseline = ggml's REAL RVV
// kernel, NOT scalar/naive.
//
// THE BASELINE (critical): at VLEN128 ggml does NOT route q8_0 through its repack —
// repack.cpp:4713 `case 128: { break; }` for GGML_TYPE_Q8_0 returns nullptr (the
// q8_0_16x1 repack is VLEN256-only, case 256). So ggml falls back to the plain per-row
// block-dot ggml_vec_dot_q8_0_q8_0. UNLIKE q4_K, q8_0 has NO `_vl128`/`_vl256` split —
// there is a SINGLE `#if defined(__riscv_v)` body (vle8_v_i8m2 -> vwmul_vv -> per-block
// vwredsum cross-lane reduction). That generic RVV body is what runs at VLEN128 and is
// the methodology-correct Win-B baseline (a weaker, non-VLEN-specialized baseline than
// q4_K's hand-tuned _vl128 inline-asm — which is exactly why a q8_0 win is more plausible).
// Lifted VERBATIM + a same-output GEVM loop: call the dot nc times (one per output column)
// over the ORIGINAL pre-repack block_q8_0 weights (same values our repack GEVM consumes).
//
// "Ours" = NARROW: the Win-A finding proves NARROW (mf2, no-stamp default) is BYTE-IDENTICAL
// to the rv64gcv stamp = the SG2044 tune's ACTUAL emit. Win-B compares the arm the tune ships.
//
// Data construction (block_q8_0/x16, make_block16, activation quant) is lifted from the
// PASS-validated Win-A harness ablation_micro_q80.cpp; the reference is the ggml dot, plus
// best-of-reps min timing of the full nc-output vector for BOTH sides.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <random>
#include <algorithm>
#include <riscv_vector.h>

#define QK8_0 32
typedef uint16_t ggml_half;
#define GGML_RESTRICT __restrict__
#define UNUSED(x) (void)(x)
// fp16->fp32 via native _Float16 (matches the kernel's vle16+vfwcvt path AND ggml's stored
// fp16 bits; NO ggml_table_f32_f16 dependency -> agreement stays the d*d*sumi fold residual).
static inline ggml_half f2h(float fv){ _Float16 x=(_Float16)fv; ggml_half h; memcpy(&h,&x,2); return h; }
static inline float h2f(ggml_half h){ _Float16 x; memcpy(&x,&h,2); return (float)x; }
#define GGML_CPU_FP16_TO_FP32(x) (h2f((x)))

// plain q8_0 block: fp16 scale d @0, 32 int8 quants @2 -> stride 34 (== ggml block_q8_0)
struct block_q8_0   { ggml_half d; int8_t qs[QK8_0]; };                       // 34
static_assert(sizeof(block_q8_0)==34, "q8_0 stride 34");
// repacked weight group: 16 fp16 scales @0 (32B), 16*32 int8 quants @32 -> stride 544
struct block_q8_0x16{ ggml_half d[16]; int8_t qs[QK8_0*16]; };               // 544
static_assert(sizeof(block_q8_0x16)==544, "q8_0x16 stride 544");

// the compiler-EMITTED q8_0 repack GEVM kernel — NARROW (mf2, 2x8 strip) = SG2044 tune emit.
extern "C" void k_gemv_NARROW(size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4, size_t v5);

// ================== ggml's OWN shipped RVV kernel for q8_0 ==================
// Lifted VERBATIM from llama.cpp ggml/src/ggml-cpu/arch/riscv/quants.c:435 (the
// `#if defined(__riscv_v)` body of ggml_vec_dot_q8_0_q8_0). This is the Win-B baseline.
// nrc==1, bs/bx/by unused (GEVM single-column dot). It does a per-block cross-lane
// vwredsum (nb reductions) — the structural contrast our block-as-lane NARROW removes.
static __attribute__((noinline)) void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = (const block_q8_0 *)vx;
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 *)vy;

    int ib = 0;
    float sumf = 0;

    size_t vl = qk;
    for (; ib < nb; ++ib) {
        // load elements
        vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[ib].qs, vl);
        vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[ib].qs, vl);

        vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);

        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }
    *s = sumf;
}

// same-output GEVM loop over ggml's q8_0 dot: nc output columns, each a per-row dot of the
// ORIGINAL pre-repack block_q8_0 weight row (w + col*nb) with the q8_0 activation (vy, nb blocks).
static void ggml_gevm_q8_0(int n, int nc, const block_q8_0* w, const block_q8_0* vy, std::vector<float>& out){
    int nb=n/QK8_0;
    for(int c=0;c<nc;++c){
        float s=0;
        ggml_vec_dot_q8_0_q8_0(n,&s,0,(const void*)(w+(size_t)c*nb),0,(const void*)vy,0,1);
        out[c]=s;
    }
}

// Repack 16 plain q8_0 blocks -> one block_q8_0x16. byte i = block(i%16).qs[i/16] (block-as-lane).
static block_q8_0x16 make_block16(const block_q8_0* in){
    block_q8_0x16 o; for(int i=0;i<16;i++) o.d[i]=in[i].d;
    const int end=QK8_0*16; for(int i=0;i<end;++i){int si=i%16,so=i/16; o.qs[i]=in[si].qs[so];}
    return o;
}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
// agreement = max|a-b| / rms(b) over the full nc vector (b = reference = ggml)
static double norm_cmp(const std::vector<float>&a,const std::vector<float>&b){
    double me=0,sse=0; size_t nn=a.size(); for(size_t j=0;j<nn;++j){ double v=b[j],e=fabs((double)a[j]-v); me=std::max(me,e); sse+=v*v; }
    double rms=sqrt(sse/nn); return rms>0?me/rms:me;
}

int main(int argc,char**argv){
    int n = argc>1?atoi(argv[1]):4096;   // K (contraction = n)
    int nc= argc>2?atoi(argv[2]):4096;   // output cols (weight rows)
    int reps=argc>3?atoi(argv[3]):300;
    int nb=n/QK8_0; int ngrp=nc/16;
    std::mt19937 rng(20260624);
    std::uniform_real_distribution<float> sc(0.005f,0.05f), act(-2.f,2.f);
    std::uniform_int_distribution<int> q8(-127,127);

    // weights: nc rows x nb blocks of plain q8_0 (ggml reads these ORIGINALS directly;
    // ours reads them repacked). Row c lives at w + c*nb.
    std::vector<block_q8_0> w((size_t)nc*nb);
    for(size_t i=0;i<w.size();++i){ w[i].d=f2h(sc(rng)); for(int k=0;k<QK8_0;++k) w[i].qs[k]=(int8_t)q8(rng); }
    // repack into ngrp x nb block_q8_0x16 (lane r of group g <-> original row g*16+r)
    std::vector<block_q8_0x16> vx((size_t)ngrp*nb);
    for(int g=0;g<ngrp;++g){ block_q8_0 tmp[16]; for(int x=0;x<nb;++x){ for(int i=0;i<16;++i) tmp[i]=w[(size_t)(g*16+i)*nb+x]; vx[(size_t)g*nb+x]=make_block16(tmp); } }

    // activation: 1 column, quantized to q8_0 (nb blocks)
    std::vector<float> af(n); for(int i=0;i<n;++i) af[i]=act(rng);
    std::vector<block_q8_0> vy(nb);
    for(int b=0;b<nb;++b){ float amax=0; for(int j=0;j<QK8_0;++j) amax=std::max(amax,fabsf(af[b*QK8_0+j])); float d=amax/127.f,id=d?1.f/d:0.f; for(int j=0;j<QK8_0;++j){ int q=(int)lroundf(af[b*QK8_0+j]*id); vy[b].qs[j]=(int8_t)std::max(-127,std::min(127,q)); } vy[b].d=f2h(d); }

    std::vector<float> oOurs(nc), oGgml(nc);
    // single correctness pass: ours (NARROW repack GEVM) vs ggml's q8_0 dot GEVM
    k_gemv_NARROW((size_t)n,oOurs.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy.data(),(size_t)nc);
    ggml_gevm_q8_0(n,nc,w.data(),vy.data(),oGgml);
    double agree=norm_cmp(oOurs,oGgml);
    printf("WINB AGREE  n=%d nc=%d  norm(ours NARROW vs ggml q8_0 dot)=%.3e %s\n",
           n,nc,agree,(agree<1e-4)?"PASS":"FAIL");

    // warmup
    for(int i=0;i<10;++i){ k_gemv_NARROW((size_t)n,oOurs.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy.data(),(size_t)nc); ggml_gevm_q8_0(n,nc,w.data(),vy.data(),oGgml); }
    // best-of-reps min for both sides (full nc-output vector each)
    double bOurs=1e18,bGgml=1e18;
    for(int r=0;r<reps;++r){
        double t0=now(); k_gemv_NARROW((size_t)n,oOurs.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy.data(),(size_t)nc); double tO=now()-t0; if(tO<bOurs)bOurs=tO;
        double t1=now(); ggml_gevm_q8_0(n,nc,w.data(),vy.data(),oGgml); double tG=now()-t1; if(tG<bGgml)bGgml=tG;
    }
    // ratio = ggml / ours : >1 => ours FASTER (Win-B WIN); <1 => LOSS
    printf("WINB RESULT n=%d nc=%d  ours(NARROW)=%.1f ns  ggml(q8_0 dot)=%.1f ns  ratio(ggml/ours)=%.3fx  %s\n",
           n,nc,bOurs,bGgml,bGgml/bOurs,(bGgml/bOurs>1.0)?"WIN":"LOSS");
    return 0;
}
