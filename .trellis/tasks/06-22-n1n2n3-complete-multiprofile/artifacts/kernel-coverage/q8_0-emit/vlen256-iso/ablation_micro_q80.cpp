// ISO — q8_0 repack GEVM strip-count/ILP ablation at the SAME mf2 fractional core,
// VLEN256. BOTH arms i8mf2->i16m1->i32m2->f32m2 (IDENTICAL LMUL, host-grep-verified
// zero m1/m4 contaminant); the ONLY difference is half_lanes: WIDE=16 (ONE 16-lane
// i32m2 strip, serial accumulate chain) vs NARROW=8 (TWO 8-lane i32m2 strips, ILP-2).
// This ISOLATES the strip-count/ILP axis from the LMUL axis (the prior 5.5x rvv/VLEN128
// FINDING bundled mf2->m1 AND 2-strip->1-strip; here LMUL is held fixed at mf2).
// Both compiler-emitted from the SAME tcrv_rvv.repack_gemv_q8_0_q8_0 op via plain
// --tcrv-rvv-lower-to-emitc (no march stamp), IRs identical except half_lanes 8 vs 16.
// half_lanes=16 is what deriveRepackHalfLanes(256,16)=16 (the VLEN256 selector) PICKS.
// Family A (symmetric, FULL int8): NO nibble decode, NO min, NO q8_1 scaled-sum;
// in-block i32 accumulation (vwmul i8xi8->i16, vwadd_wv i32 += i16), dual fp16 d_x*d_y.
// Plus a SCALAR q8_0 dequant-matmul ORACLE (correctness) consuming the SAME bytes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <riscv_vector.h>

#define QK8_0 32
typedef uint16_t ggml_half;

// plain q8_0 block: fp16 scale d @0, 32 int8 quants @2 -> stride 34
struct block_q8_0   { ggml_half d; int8_t qs[QK8_0]; };                       // 34
// repacked weight group: 16 fp16 scales @0 (32B), 16*32 int8 quants @32 -> stride 544
struct block_q8_0x16{ ggml_half d[16]; int8_t qs[QK8_0*16]; };               // 544

static inline ggml_half f2h(float fv){ _Float16 x=(_Float16)fv; ggml_half h; memcpy(&h,&x,2); return h; }
static inline float h2f(ggml_half h){ _Float16 x; memcpy(&x,&h,2); return (float)x; }

// the two compiler-EMITTED q8_0 repack GEVM kernels (renamed symbols)
#include "k_gemv_NARROW.cpp"
#include "k_gemv_WIDE.cpp"

// Repack 16 plain q8_0 blocks -> one block_q8_0x16. int8 quants stored interleaved:
// byte i = block(i%16).qs[i/16]  (16-way block-as-lane interleave, same shape as q4_0).
static block_q8_0x16 make_block16(const block_q8_0* in){
    block_q8_0x16 o; for(int i=0;i<16;i++) o.d[i]=in[i].d;
    const int end=QK8_0*16; for(int i=0;i<end;++i){int si=i%16,so=i/16; o.qs[i]=in[si].qs[so];}
    return o;
}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double norm_cmp(const std::vector<float>&a,const std::vector<float>&b){
    double me=0,sse=0; size_t nn=a.size(); for(size_t j=0;j<nn;++j){ double v=b[j],e=fabs((double)a[j]-v); me=std::max(me,e); sse+=v*v; }
    double rms=sqrt(sse/nn); return rms>0?me/rms:me;
}

// SCALAR q8_0 dequant-matmul ORACLE consuming the SAME repacked weights (vx) and
// plain q8_0 activation (vy): for output col c, sum over blocks l of
// d_x[c,l]*d_y[l] * sum_k( w_int8 * a_int8 ).  Integer accumulation in i32.
static void oracle_scalar(int n,int nc,const block_q8_0x16* vx,const block_q8_0* vy,std::vector<float>&out){
    int nb=n/QK8_0; int ngrp=nc/16;
    for(int g=0;g<ngrp;++g){
        for(int r=0;r<16;++r){           // which of the 16 interleaved rows in the group
            double acc=0;
            for(int l=0;l<nb;++l){
                const block_q8_0x16& bx=vx[(size_t)g*nb+l];
                const block_q8_0&    by=vy[l];
                int32_t sumi=0;
                for(int k=0;k<QK8_0;++k){
                    int8_t w=bx.qs[k*16+r];   // interleaved: position k, lane r
                    int8_t a=by.qs[k];
                    sumi += (int32_t)w*(int32_t)a;
                }
                float dx=h2f(bx.d[r]), dy=h2f(by.d);
                acc += (double)dx*(double)dy*(double)sumi;
            }
            out[g*16+r]=(float)acc;
        }
    }
}

int main(int argc,char**argv){
    int n = argc>1?atoi(argv[1]):4096;   // K (contraction)
    int nc= argc>2?atoi(argv[2]):4096;   // output cols (weight rows)
    int reps=argc>3?atoi(argv[3]):300;
    int nb=n/QK8_0; int ngrp=nc/16;
    std::mt19937 rng(20260624);
    std::uniform_real_distribution<float> sc(0.005f,0.05f), act(-2.f,2.f);
    std::uniform_int_distribution<int> q8(-127,127);

    // weights: nc rows x nb blocks of plain q8_0
    std::vector<block_q8_0> w((size_t)nc*nb);
    for(size_t i=0;i<w.size();++i){ w[i].d=f2h(sc(rng)); for(int k=0;k<QK8_0;++k) w[i].qs[k]=(int8_t)q8(rng); }
    // repack into ngrp x nb block_q8_0x16
    std::vector<block_q8_0x16> vx((size_t)ngrp*nb);
    for(int g=0;g<ngrp;++g){ block_q8_0 tmp[16]; for(int x=0;x<nb;++x){ for(int i=0;i<16;++i) tmp[i]=w[(size_t)(g*16+i)*nb+x]; vx[(size_t)g*nb+x]=make_block16(tmp); } }

    // ---------- GEVM (decode, 1 activation column) ----------
    std::vector<float> af(n); for(int i=0;i<n;++i) af[i]=act(rng);
    std::vector<block_q8_0> vy(nb);
    for(int b=0;b<nb;++b){ float amax=0; for(int j=0;j<QK8_0;++j) amax=std::max(amax,fabsf(af[b*QK8_0+j])); float d=amax/127.f,id=d?1.f/d:0.f; for(int j=0;j<QK8_0;++j){ int q=(int)lroundf(af[b*QK8_0+j]*id); vy[b].qs[j]=(int8_t)std::max(-127,std::min(127,q)); } vy[b].d=f2h(d); }

    std::vector<float> oN(nc),oW(nc),oRef(nc);
    k_gemv_NARROW((size_t)n,oN.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy.data(),(size_t)nc);
    k_gemv_WIDE  ((size_t)n,oW.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy.data(),(size_t)nc);
    oracle_scalar(n,nc,vx.data(),vy.data(),oRef);

    double nm_wn=norm_cmp(oW,oN);
    double nm_oracleN=norm_cmp(oN,oRef);
    double nm_oracleW=norm_cmp(oW,oRef);
    printf("GEVM AGREE  n=%d nc=%d  norm(WIDE vs NARROW)=%.3e %s\n",n,nc,nm_wn,nm_wn==0.0?"BYTE-EXACT":(nm_wn<1e-4?"PASS":"FAIL"));
    printf("GEVM ORACLE n=%d nc=%d  norm(NARROW vs scalar)=%.3e  norm(WIDE vs scalar)=%.3e  %s\n",
           n,nc,nm_oracleN,nm_oracleW,(nm_oracleN<1e-4&&nm_oracleW<1e-4)?"PASS":"FAIL");

    for(int i=0;i<10;++i){ k_gemv_NARROW((size_t)n,oN.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy.data(),(size_t)nc); k_gemv_WIDE((size_t)n,oW.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy.data(),(size_t)nc);}
    double bN=1e18,bW=1e18;
    for(int r=0;r<reps;++r){
        double t0=now(); k_gemv_NARROW((size_t)n,oN.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy.data(),(size_t)nc); double tN=now()-t0; if(tN<bN)bN=tN;
        double t1=now(); k_gemv_WIDE  ((size_t)n,oW.data(),(const uint8_t*)vx.data(),(const uint8_t*)vy.data(),(size_t)nc); double tW=now()-t1; if(tW<bW)bW=tW;
    }
    printf("GEVM RESULT n=%d nc=%d  NARROW=%.0f ns  WIDE=%.0f ns  ratio(NARROW/WIDE)=%.3fx\n",n,nc,bN,bW,bN/bW);
    return 0;
}
