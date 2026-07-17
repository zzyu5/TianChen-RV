// S1 scalar duel driver — real no-V silicon (超锐/scalar board, rv64gc, clang-18).
//
//   OURS   = owned weft_scalar emitted kernel (verbatim from weft-translate).
//   OPP    = the board's REAL DEPLOYED ggml symbol ggml_vec_dot_tq2_0_q8_K
//            (on a no-V build the dispatch thunk collapses to _generic — the true
//            deployed path; §对手法 3.4 部署事实要件, §板册 3.7). Linked from
//            libggml-cpu.so, NOT reimplemented — this is the genuine S1 opponent
//            (upgrade over the A3 rehearsal's inline reference).
//   ORACLE = libcall-free pure-integer independent recompute (weft_h2f, int64,
//            ZERO-MODEL from actual input bytes; [K-5]). Third independent impl.
//
// byte-exact ([K-5]): ours == oracle (ZERO-MODEL primary) AND ours == ggml-deployed
// (deployed-path correctness) AND ggml == oracle (opponent vs same oracle). Three
// independent code paths agreeing = anti-hollow by construction.
//
// enablement domain: scalar-oracle sanity, [L-6] scalar NEVER a contribution
// baseline — timing is diagnostic, NON-Win, must NOT enter the system ledger.
//
// The harness (scalar_vec_dot.sh) owns all persistence; this driver only prints to
// stdout (ISSUE-090 contract). Modes via argv.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>

#include "weft_scalar_tq2_0_kernel.cpp"   // OURS

extern "C" float    weft_h2f(uint16_t);   // fp16util.cpp (pure-integer)
extern "C" uint16_t weft_f2h(float);

// The board's REAL deployed ggml symbol (no-V build → dispatch thunk == generic).
//   void ggml_vec_dot_tq2_0_q8_K(int n, float* s, size_t bs,
//                                const void* vx, size_t bx,
//                                const void* vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_tq2_0_q8_K(int, float*, size_t,
                                        const void*, size_t,
                                        const void*, size_t, int);
// ggml's GGML_CPU_FP16_TO_FP32 on the rv64gc (no-zfh) generic path uses a lookup
// table populated by ggml_cpu_init(); without it x.d folds to 0 (silent all-zero
// output). Must be called before any ggml vec_dot.
extern "C" void ggml_cpu_init(void);

// tq2_0 weight block 66B { u8 qs[64]@0 ; fp16 d @64 }
// q8_K  activ  block 292B { f32 d @0 ; i8 qs[256]@4 ; i16 bsums[16]@260 }

// Fully libcall-free independent oracle (pure-integer weft_h2f, int64 accum).
// ZERO-MODEL: recomputed from actual input bytes, independent contraction.
static void oracle_int_tq2_0_q8_K(int n, float* s, const uint8_t* vx, const int8_t* vy) {
  const int nb = n / 256;
  float sumf = 0.0f;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t* xq = vx + (size_t)ib * 66;
    const int8_t*  yb = vy + (size_t)ib * 292;
    const int8_t*  yq = yb + 4;
    int64_t isum = 0;
    for (int k = 0; k < 2; ++k)
      for (int l = 0; l < 4; ++l)
        for (int m = 0; m < 32; ++m) {
          int trit = (((xq + k*32)[m] >> (2*l)) & 3) - 1;
          isum += (int64_t)(yq + k*128)[l*32 + m] * trit;
        }
    uint16_t hb; std::memcpy(&hb, xq + 64, 2);
    float dx = weft_h2f(hb);
    float dy = *(const float*)(yb);
    sumf += (float)isum * (dy * dx);
  }
  *s = sumf;
}

static inline uint32_t f2u(float f){ uint32_t u; std::memcpy(&u,&f,4); return u; }
static double median(std::vector<double>& v){ std::sort(v.begin(),v.end()); size_t n=v.size();
  return n%2? v[n/2] : 0.5*(v[n/2-1]+v[n/2]); }
static double reliqr(std::vector<double>& v){ std::sort(v.begin(),v.end()); size_t n=v.size();
  if(n<4) return 0.0; double q1=v[n/4], q3=v[(3*n)/4], med=median(v);
  return med>0? 100.0*(q3-q1)/med : 0.0; }

static int selftest_fp16(){
  struct G { uint16_t b; float v; };
  const G g[] = {
    {0x3C00,1.0f},{0xBC00,-1.0f},{0x3800,0.5f},{0x4000,2.0f},{0x0000,0.0f},
    {0x3555,0.333251953125f},{0x0400,6.103515625e-05f},
    {0x0001,5.9604644775390625e-08f},{0x7BFF,65504.0f},
  };
  int fails=0;
  for (auto& e : g) if (f2u(weft_h2f(e.b)) != f2u(e.v)) fails++;
  return fails;
}

int main(int argc, char** argv){
  int   NB     = (argc>1)? atoi(argv[1]) : 4096;
  int   REP    = (argc>2)? atoi(argv[2]) : 25;
  unsigned SEED= (argc>3)? (unsigned)strtoul(argv[3],0,0) : 0xA3C0FFEEu;
  int   INJECT = (argc>4)? atoi(argv[4]) : 0;   // 0 clean · 1 corrupt-OURS-out · 2 corrupt-oracle
  int   FLUSHMB= (argc>5)? atoi(argv[5]) : 0;   // cache flush MiB between cold reps (0=off)
  int   N = NB * 256;

  ggml_cpu_init();                               // populate ggml fp16 lookup table

  std::vector<uint8_t> W((size_t)NB*66);
  std::vector<int8_t>  A((size_t)NB*292);
  std::mt19937 rng(SEED);
  std::uniform_int_distribution<int> b8(0,255), i8(-127,127);
  std::uniform_real_distribution<float> sc(0.01f, 0.5f);
  for (int ib=0; ib<NB; ++ib){
    uint8_t* xq = W.data() + (size_t)ib*66;
    for (int i=0;i<64;++i) xq[i] = (uint8_t)b8(rng);
    uint16_t dxb = weft_f2h(sc(rng));
    std::memcpy(xq+64, &dxb, 2);
    int8_t* yb = A.data() + (size_t)ib*292;
    float dy = sc(rng);
    std::memcpy(yb, &dy, 4);
    for (int i=0;i<256;++i) yb[4+i] = (int8_t)i8(rng);
    std::memset(yb+260, 0, 32);
  }

  int f16fail = selftest_fp16();

  float s_ours=0.f, s_ggml=0.f, s_int=0.f;
  weft_emitc_tq2_0_kernel_scalar_fallback_first_slice(N, &s_ours, W.data(), A.data());
  ggml_vec_dot_tq2_0_q8_K(N, &s_ggml, 0, W.data(), 0, A.data(), 0, 1);
  oracle_int_tq2_0_q8_K(N, &s_int, W.data(), A.data());

  if (INJECT == 1) s_ours += 1.0f;                 // DUT-output fault: byte-exact MUST flip
  if (INJECT == 2) s_int  += 1.0f;                 // oracle-constant fault: cross-check MUST catch

  uint32_t u_ours=f2u(s_ours), u_ggml=f2u(s_ggml), u_int=f2u(s_int);
  int be_ggml = (u_ours==u_ggml);
  int be_int  = (u_ours==u_int);
  int gg_int  = (u_ggml==u_int);
  int byte_exact = be_ggml && be_int && gg_int && f16fail==0;

  printf("# VECDOT_CORRECT fmt=tq2_0 op=vec_dot engine=scalar regime=prefill inject=%d seed=0x%X n=%d nb=%d\n",
         INJECT, SEED, N, NB);
  printf("# fp16_primitive_golden=%s (weft_h2f 9/9 textbook IEEE)\n", f16fail? "FAIL":"PASS");
  printf("# result_ours=%.9g result_ggml_deployed=%.9g result_int_oracle=%.9g\n",
         (double)s_ours,(double)s_ggml,(double)s_int);
  printf("# bits_ours=0x%08x bits_ggml=0x%08x bits_int_oracle=0x%08x\n", u_ours,u_ggml,u_int);
  printf("# BYTE_EXACT ours_vs_ggml_deployed=%s ours_vs_int_oracle=%s ggml_vs_int_oracle=%s ALL=%s\n",
         be_ggml?"true":"false", be_int?"true":"false", gg_int?"true":"false", byte_exact?"true":"false");

  if (INJECT != 0) {
    // anti-hollow: a fault MUST break byte-exactness (gate bites).
    printf("# ANTIHOLLOW inject=%d expect_byte_exact=false observed=%s -> %s\n",
           INJECT, byte_exact?"true":"false", byte_exact? "HOLLOW-FAIL":"BITES-OK");
    return byte_exact ? 3 : 0;   // if a fault did NOT break it, that's a hollow gate
  }

  if (!byte_exact) { printf("# CORRECTNESS FAIL - aborting before timing\n"); return 2; }

  // ---- cold timing (measure mode) ----
  if (REP > 0) {
    std::vector<char> flush;
    if (FLUSHMB > 0) flush.assign((size_t)FLUSHMB*1024*1024, 1);
    volatile char fsink=0;
    volatile float sink=0.f;
    // warm
    for(int w=0;w<3;++w){ float t;
      weft_emitc_tq2_0_kernel_scalar_fallback_first_slice(N,&t,W.data(),A.data()); sink+=t;
      ggml_vec_dot_tq2_0_q8_K(N,&t,0,W.data(),0,A.data(),0,1); sink+=t; }
    std::vector<double> t_ours, t_ggml;
    for(int r=0;r<REP;++r){
      float t;
      if(FLUSHMB>0){ for(size_t i=0;i<flush.size();i+=64) fsink+=flush[i]; }
      auto a0=std::chrono::steady_clock::now();
      weft_emitc_tq2_0_kernel_scalar_fallback_first_slice(N,&t,W.data(),A.data());
      auto a1=std::chrono::steady_clock::now(); sink+=t;
      if(FLUSHMB>0){ for(size_t i=0;i<flush.size();i+=64) fsink+=flush[i]; }
      auto a2=std::chrono::steady_clock::now();
      ggml_vec_dot_tq2_0_q8_K(N,&t,0,W.data(),0,A.data(),0,1);
      auto a3=std::chrono::steady_clock::now(); sink+=t;
      t_ours.push_back(std::chrono::duration<double,std::nano>(a1-a0).count());
      t_ggml.push_back(std::chrono::duration<double,std::nano>(a3-a2).count());
    }
    double mo=median(t_ours), mg=median(t_ggml);
    printf("# VECDOT_COLD fmt=tq2_0 seed=0x%X | ours_med_ns=%.0f(iqr%.1f%%) oppX_med_ns=%.0f(iqr%.1f%%) | ratio_cold_X=%.4f | opp=ggml_vec_dot_tq2_0_q8_K(deployed·no-V==generic) domain=enablement-NONWIN(L-6) flush=%dMiB\n",
           SEED, mo, reliqr(t_ours), mg, reliqr(t_ggml), mg/mo, FLUSHMB);
    printf("# sink=%.6g fsink=%d\n", (double)sink, (int)fsink);
  }
  return 0;
}
