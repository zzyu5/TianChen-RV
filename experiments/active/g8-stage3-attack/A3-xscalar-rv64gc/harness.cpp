// A3 X-SCALAR rv64gc rehearsal harness (channel 1: real silicon, vector-absent).
//   OURS = owned weft_scalar emitted kernel (verbatim from weft-translate).
//   REF  = ggml canonical hand-written scalar tq2_0 x q8_K vec_dot; serves as
//          BOTH the independent byte-exact oracle (int64 contraction, ggml k/l/m
//          element order, recomputed from actual input bytes) AND the same-board
//          same-toolchain scalar PK opponent (Win-S sanity, enablement, NON-Win).
// Correctness gate (K-5 integer byte-exact) precedes timing.
// fp16 scale primitive (weft_h2f) is validated against textbook IEEE golden.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>

#include "weft_scalar_tq2_0_kernel.cpp"   // OURS (size_t from cstddef above)

extern "C" float    weft_h2f(uint16_t);   // from fp16util.cpp (pure-integer)
extern "C" uint16_t weft_f2h(float);

// tq2_0 weight block 66B { u8 qs[64]@0 ; fp16 d @64 }
// q8_K  activ  block 292B { f32 d @0 ; i8 qs[256]@4 ; i16 bsums[16]@260 }
static void ggml_ref_tq2_0_q8_K(int n, float* s, const uint8_t* vx, const int8_t* vy) {
  const int nb = n / 256;
  float sumf = 0.0f;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t* xq = vx + (size_t)ib * 66;
    const int8_t*  yb = vy + (size_t)ib * 292;
    const int8_t*  yq = yb + 4;
    int64_t isum = 0;                                    // int64: wider than kernel int32
    for (int k = 0; k < 2; ++k) {
      const uint8_t* q2 = xq + k * 32;
      const int8_t*  q8 = yq + k * 128;
      for (int l = 0; l < 4; ++l)
        for (int m = 0; m < 32; ++m) {
          int trit = ((q2[m] >> (2 * l)) & 3) - 1;       // field-extract, NO popcount
          isum += (int64_t)q8[l * 32 + m] * trit;
        }
    }
    float dy = *(const float*)(yb);
    float dx = (float)*(const _Float16*)(xq + 64);       // same __extendhfsf2 path as OURS
    sumf += (float)isum * (dy * dx);
  }
  *s = sumf;
}

// Fully libcall-free independent oracle: reads the fp16 scale via pure-integer
// weft_h2f (NO _Float16, NO __extendhfsf2). Byte-exact vs OURS proves the kernel's
// soft-fp16 read is genuinely correct, not both-wrong via a shared shim.
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

// validate the fp16 primitive against textbook IEEE-754 half golden values
static int selftest_fp16(){
  struct G { uint16_t b; float v; };
  const G g[] = {
    {0x3C00, 1.0f}, {0xBC00, -1.0f}, {0x3800, 0.5f}, {0x4000, 2.0f},
    {0x0000, 0.0f}, {0x3555, 0.333251953125f}, {0x0400, 6.103515625e-05f},
    {0x0001, 5.9604644775390625e-08f}, {0x7BFF, 65504.0f},
  };
  int fails=0;
  for (auto& e : g) if (f2u(weft_h2f(e.b)) != f2u(e.v)) {
    printf("  \"fp16_selftest_fail\": {\"bits\":\"0x%04x\",\"got\":%.9g,\"want\":%.9g},\n",
           e.b, (double)weft_h2f(e.b), (double)e.v); fails++;
  }
  return fails;
}

int main(int argc, char** argv){
  int NB = (argc>1)? atoi(argv[1]) : 8192;
  int REP = (argc>2)? atoi(argv[2]) : 61;
  int N = NB * 256;
  std::vector<uint8_t> W((size_t)NB*66);
  std::vector<int8_t>  A((size_t)NB*292);

  std::mt19937 rng(0xA3C0FFEEu);
  std::uniform_int_distribution<int> b8(0,255), i8(-127,127);
  std::uniform_real_distribution<float> sc(0.01f, 0.5f);
  for (int ib=0; ib<NB; ++ib){
    uint8_t* xq = W.data() + (size_t)ib*66;
    for (int i=0;i<64;++i) xq[i] = (uint8_t)b8(rng);
    uint16_t dxb = weft_f2h(sc(rng));                    // pure-integer f32->f16 (no libcall)
    std::memcpy(xq+64, &dxb, 2);
    int8_t* yb = A.data() + (size_t)ib*292;
    float dy = sc(rng);
    std::memcpy(yb, &dy, 4);
    for (int i=0;i<256;++i) yb[4+i] = (int8_t)i8(rng);
    std::memset(yb+260, 0, 32);
  }

  printf("{\n");
  printf("  \"target\": \"rvv-board run-as-noV (-march=rv64gc -mabi=lp64d), narrow-exempt V-board-run-as-noV\",\n");
  printf("  \"n_elements\": %d, \"nb_superblocks\": %d, \"reps\": %d,\n", N, NB, REP);
  int f16fail = selftest_fp16();
  printf("  \"fp16_primitive_golden\": %s,\n", f16fail? "\"FAIL\"":"\"PASS (9/9 textbook IEEE)\"");

  float s_ours=0.f, s_ref=0.f, s_int=0.f;
  weft_emitc_tq2_0_kernel_scalar_fallback_first_slice(N, &s_ours, W.data(), A.data());
  ggml_ref_tq2_0_q8_K(N, &s_ref, W.data(), A.data());
  oracle_int_tq2_0_q8_K(N, &s_int, W.data(), A.data());
  uint32_t u_ours=f2u(s_ours), u_ref=f2u(s_ref), u_int=f2u(s_int);
  int byte_exact = (u_ours==u_ref) && (u_ours==u_int) && f16fail==0;
  printf("  \"result_ours\": %.9g, \"result_ref\": %.9g, \"result_int_oracle\": %.9g,\n",
         (double)s_ours,(double)s_ref,(double)s_int);
  printf("  \"bits_ours\": \"0x%08x\", \"bits_ref\": \"0x%08x\", \"bits_int_oracle\": \"0x%08x\",\n",
         u_ours, u_ref, u_int);
  printf("  \"byte_exact_vs_ggml_ref\": %s, \"byte_exact_vs_libcall_free_oracle\": %s,\n",
         (u_ours==u_ref)?"true":"false", (u_ours==u_int)?"true":"false");
  printf("  \"byte_exact\": %s,\n", byte_exact? "true":"false");
  if(!byte_exact){ printf("  \"CORRECTNESS\": \"FAIL - aborting before timing\"\n}\n"); return 2; }

  volatile float sink=0.f;
  for(int w=0;w<5;++w){ float t;
    weft_emitc_tq2_0_kernel_scalar_fallback_first_slice(N,&t,W.data(),A.data()); sink+=t;
    ggml_ref_tq2_0_q8_K(N,&t,W.data(),A.data()); sink+=t; }
  std::vector<double> t_ours, t_ref;
  for(int r=0;r<REP;++r){
    float t;
    auto a0=std::chrono::steady_clock::now();
    weft_emitc_tq2_0_kernel_scalar_fallback_first_slice(N,&t,W.data(),A.data());
    auto a1=std::chrono::steady_clock::now(); sink+=t;
    ggml_ref_tq2_0_q8_K(N,&t,W.data(),A.data());
    auto a2=std::chrono::steady_clock::now(); sink+=t;
    t_ours.push_back(std::chrono::duration<double,std::nano>(a1-a0).count());
    t_ref .push_back(std::chrono::duration<double,std::nano>(a2-a1).count());
  }
  double mo=median(t_ours), mr=median(t_ref);
  printf("  \"median_ns_ours\": %.1f, \"median_ns_ref\": %.1f,\n", mo, mr);
  printf("  \"ratio_ref_over_ours\": %.4f,\n", mr/mo);
  printf("  \"pk_domain\": \"scalar-oracle sanity (Win-S), enablement, NON-Win\",\n");
  printf("  \"sink\": %.6g\n}\n", (double)sink);
  return 0;
}
