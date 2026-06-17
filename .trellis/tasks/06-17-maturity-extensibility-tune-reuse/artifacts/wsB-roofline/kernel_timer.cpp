/* WS-B roofline: time the 3 emitted ggml block-dot kernels in TWO regimes.
 *
 * Real ssh rvv (riscv64, VLEN=128, clang 18.1.3, -march=rv64gcv -mabi=lp64d).
 * rdcycle primary (frequency-independent cyc/dot); freq read during hot run for
 * absolute GMAC/s. Two regimes per the WS-B methodology:
 *   - CACHE-RESIDENT: one fixed (vx,vy) pair reused across reps -> working set
 *     stays in L1/L2 -> measures compute+decode ceiling, NOT DRAM.
 *   - STREAMING: weight buffer W >> LLC (many distinct rows), one reused y row
 *     -> each weight byte read from DRAM once, exactly the ggml mat-vec pattern.
 *
 * The emitted kernels are #included verbatim (the compiler output, unmodified).
 * q8_0 / q4_K take the 4-arg (n,s,vx,vy) form; q4_0 the 8-arg ggml ABI form.
 * DCE guard: we accumulate every result into a volatile sink and assert finite
 * + nonzero, so a hoisted/dead kernel is caught (it would read as ~0 cyc/dot).
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <riscv_vector.h>

// The emitted kernels already wrap their entry in extern "C"; include verbatim.
#include "kernel_q8_0_q8_0.cpp"   // tcrv_emitc_...q8_0_q8_0(n,s,vx,vy)
#include "kernel_q4_k_q8_k.cpp"   // tcrv_emitc_...q4_K_q8_K(n,s,vx,vy)
#include "kernel_q4_0_q8_0.cpp"   // tcrv_emitc_...q4_0_q8_0(n,s,bs,vx,bx,vy,by,nrc)

static inline uint64_t rdcycle(void){ uint64_t c; asm volatile("rdcycle %0":"=r"(c)); return c; }

// ---- block byte sizes (ggml) ----
static const int QK     = 32, QK_K = 256;
static const int Q80_W  = 34;   // block_q8_0: fp16 d + 32 i8
static const int Q80_A  = 34;
static const int Q40_W  = 18;   // block_q4_0: fp16 d + 16 packed-i4
static const int Q40_A  = 34;   // activation is q8_0
static const int Q4K_W  = 144;  // block_q4_K super-block (weights)
static const int Q4K_A  = 292;  // block_q8_K activation super-block

// ---- per-dot accounting ----
struct Acct { const char* name; int qk; int wstride; int astride; long macs_per_blk; };
// MACs counted as actual dot-product integer multiplies (qk per block).
static Acct ACCT_Q80 = {"q8_0_q8_0", QK,   Q80_W, Q80_A, QK};
static Acct ACCT_Q40 = {"q4_0_q8_0", QK,   Q40_W, Q40_A, QK};
static Acct ACCT_Q4K = {"q4_K_q8_K", QK_K, Q4K_W, Q4K_A, QK_K};

static uint64_t med(std::vector<uint64_t> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static uint64_t mn (std::vector<uint64_t> v){ return *std::min_element(v.begin(),v.end()); }
static uint64_t mx (std::vector<uint64_t> v){ return *std::max_element(v.begin(),v.end()); }

static void fill_rand(uint8_t* p, size_t n, uint64_t seed){
  for(size_t i=0;i<n;i++){ seed=seed*6364136223846793005ULL+1442695040888963407ULL; p[i]=(uint8_t)(seed>>56); }
}

static double g_freq=0;

int main(){
  const size_t n = 4096;                 // representative ggml row
  const long nb   = n/QK;                // 128 blocks (q8_0/q4_0)
  const long nbK  = n/QK_K;              // 16 super-blocks (q4_K)

  // ---- single-row buffers (cache-resident) ----
  std::vector<uint8_t> q80_w(nb*Q80_W), q80_a(nb*Q80_A);
  std::vector<uint8_t> q40_w(nb*Q40_W), q40_a(nb*Q40_A);
  std::vector<uint8_t> q4k_w(nbK*Q4K_W), q4k_a(nbK*Q4K_A);
  fill_rand(q80_w.data(),q80_w.size(),1); fill_rand(q80_a.data(),q80_a.size(),2);
  fill_rand(q40_w.data(),q40_w.size(),3); fill_rand(q40_a.data(),q40_a.size(),4);
  fill_rand(q4k_w.data(),q4k_w.size(),5); fill_rand(q4k_a.data(),q4k_a.size(),6);
  // keep fp16 scales small/finite (bytes 0-1 of each block) so output is finite
  auto tame=[&](std::vector<uint8_t>&b,int stride){ for(size_t o=0;o+1<b.size();o+=stride){ b[o]=0x00; b[o+1]=0x2c; } }; // ~0.06 fp16
  tame(q80_w,Q80_W); tame(q80_a,Q80_A); tame(q40_w,Q40_W); tame(q40_a,Q40_A);
  // q4_K activation: byte 0-3 = fp32 d (q8_K), tame to 0.05f; weights byte 0-1 fp16 d, 2-3 fp16 dmin
  for(size_t o=0;o+3<q4k_a.size();o+=Q4K_A){ float d=0.05f; memcpy(&q4k_a[o],&d,4); }
  for(size_t o=0;o+3<q4k_w.size();o+=Q4K_W){ q4k_w[o]=0x00;q4k_w[o+1]=0x2c;q4k_w[o+2]=0x00;q4k_w[o+3]=0x28; }

  volatile double sink=0; float s_out=0;

  // ---- frequency: warm hard on q8_0 then read during a hot batch ----
  for(int w=0;w<200000;w++){ tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(n,&s_out,q80_w.data(),q80_a.data()); sink+=s_out; }
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0); uint64_t f0=rdcycle();
  for(int w=0;w<300000;w++){ tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(n,&s_out,q80_w.data(),q80_a.data()); sink+=s_out; }
  uint64_t f1=rdcycle(); clock_gettime(CLOCK_MONOTONIC,&t1);
  double ns=(t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec); g_freq=(double)(f1-f0)/ns;
  printf("freq_GHz_sustained=%.4f (during hot q8_0 batch)\n",g_freq);

  // DCE/correctness guard: outputs must be finite and nonzero
  float oa=0,ob=0,oc=0;
  tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(n,&oa,q80_w.data(),q80_a.data());
  tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(n,&ob,0,q40_w.data(),0,q40_a.data(),0,1);
  tcrv_emitc_ggml_vec_dot_q4_K_q8_K_kernel_ggml_vec_dot_q4_K_q8_K(n,&oc,q4k_w.data(),q4k_a.data());
  printf("guard: q8_0=%.6g q4_0=%.6g q4_K=%.6g (must be finite & nonzero)\n",oa,ob,oc);
  if(!std::isfinite(oa)||!std::isfinite(ob)||!std::isfinite(oc)||oa==0||ob==0||oc==0){
    printf("GUARD FAILED -- aborting (kernel may be DCE'd or fed degenerate data)\n"); return 2;
  }

  // ---- per-call batch counts so cyc is per single dot ----
  // Each timed call does D dots; the lambda returns the summed output and we
  // divide the measured cycles by D by reporting via a D-aware wrapper below.
  const int D = 2000;   // dots per timed sample (amortizes rdcycle overhead)
  const int REPS = 31;

  // helper: run D dots, return cyc/dot by dividing inside a custom loop
  auto time_per_dot=[&](const char* name,const char* regime,Acct a,
                        auto perdot)->void{
    long blocks=(long)n/a.qk, macs=blocks*a.macs_per_blk;
    double bpd_full=(double)blocks*(a.wstride+a.astride), bpd_wt=(double)blocks*a.wstride;
    std::vector<uint64_t> samp; samp.reserve(REPS);
    for(int r=0;r<REPS;r++){
      uint64_t c0=rdcycle();
      for(int d=0;d<D;d++) perdot(d);
      uint64_t c1=rdcycle();
      samp.push_back(c1-c0);
    }
    double cyc_per_dot=(double)mn(samp)/D;
    double spread=100.0*(double)(mx(samp)-mn(samp))/(double)mn(samp);
    double mac_per_cyc=(double)macs/cyc_per_dot, gmacs=mac_per_cyc*g_freq;
    printf("  [%-9s|%-9s] cyc/dot=%.1f spread=%.1f%%  cyc/MAC=%.4f MAC/cyc=%.3f GMAC/s=%.2f  B/cyc(full)=%.3f B/cyc(wt)=%.3f\n",
      name,regime,cyc_per_dot,spread,cyc_per_dot/macs,mac_per_cyc,gmacs,bpd_full/cyc_per_dot,bpd_wt/cyc_per_dot);
  };

  printf("\n# CACHE-RESIDENT (single row reused; working set in L1/L2)  n=%zu D=%d REPS=%d\n",n,D,REPS);
  time_per_dot("q8_0_q8_0","cache",ACCT_Q80,[&](int){ float s; tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(n,&s,q80_w.data(),q80_a.data()); sink+=s; });
  time_per_dot("q4_0_q8_0","cache",ACCT_Q40,[&](int){ float s; tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(n,&s,0,q40_w.data(),0,q40_a.data(),0,1); sink+=s; });
  time_per_dot("q4_K_q8_K","cache",ACCT_Q4K,[&](int){ float s; tcrv_emitc_ggml_vec_dot_q4_K_q8_K_kernel_ggml_vec_dot_q4_K_q8_K(n,&s,q4k_w.data(),q4k_a.data()); sink+=s; });

  // ---- STREAMING: weight buffer >> LLC, one reused activation row ----
  // Allocate WROWS distinct weight rows (each = one full dot's weight bytes),
  // cycle through them so each weight byte is fetched from DRAM. Activation y
  // stays fixed (reused, amortized -> the matvec pattern).
  const long WBYTES_Q80 = (long)nb*Q80_W, WBYTES_Q40=(long)nb*Q40_W, WBYTES_Q4K=(long)nbK*Q4K_W;
  const size_t TARGET = 512ull*1024*1024;  // 512 MB >> any LLC
  long ROWS_Q80 = TARGET/WBYTES_Q80, ROWS_Q40=TARGET/WBYTES_Q40, ROWS_Q4K=TARGET/WBYTES_Q4K;
  printf("\n# STREAMING (weights from DRAM: %zu MB buffer, y reused)  rows: q8_0=%ld q4_0=%ld q4_K=%ld\n",
         TARGET/1024/1024,ROWS_Q80,ROWS_Q40,ROWS_Q4K);
  std::vector<uint8_t> W80((size_t)ROWS_Q80*WBYTES_Q80), W40((size_t)ROWS_Q40*WBYTES_Q40), W4K((size_t)ROWS_Q4K*WBYTES_Q4K);
  // tile-fill from the tamed single row so scales stay finite
  for(long r=0;r<ROWS_Q80;r++) memcpy(&W80[(size_t)r*WBYTES_Q80],q80_w.data(),WBYTES_Q80);
  for(long r=0;r<ROWS_Q40;r++) memcpy(&W40[(size_t)r*WBYTES_Q40],q40_w.data(),WBYTES_Q40);
  for(long r=0;r<ROWS_Q4K;r++) memcpy(&W4K[(size_t)r*WBYTES_Q4K],q4k_w.data(),WBYTES_Q4K);

  auto stream=[&](const char* name,Acct a,uint8_t* W,long rows,long wbytes,const uint8_t* y,
                  void(*kfn4)(size_t,float*,const uint8_t*,const uint8_t*),
                  void(*kfn8)(size_t,float*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,int))->void{
    long blocks=(long)n/a.qk, macs=blocks*a.macs_per_blk;
    double bpd_wt=(double)blocks*a.wstride, bpd_full=(double)blocks*(a.wstride+a.astride);
    // warm (page-in + freq) one full sweep
    for(long r=0;r<rows;r++){ float s; if(kfn4) kfn4(n,&s,W+(size_t)r*wbytes,y); else kfn8(n,&s,0,W+(size_t)r*wbytes,0,y,0,1); sink+=s; }
    std::vector<uint64_t> samp; samp.reserve(9);
    for(int rep=0;rep<9;rep++){
      uint64_t c0=rdcycle();
      for(long r=0;r<rows;r++){ float s; if(kfn4) kfn4(n,&s,W+(size_t)r*wbytes,y); else kfn8(n,&s,0,W+(size_t)r*wbytes,0,y,0,1); sink+=s; }
      uint64_t c1=rdcycle();
      samp.push_back(c1-c0);
    }
    double cyc_per_dot=(double)mn(samp)/rows;
    double spread=100.0*(double)(mx(samp)-mn(samp))/(double)mn(samp);
    double mac_per_cyc=(double)macs/cyc_per_dot, gmacs=mac_per_cyc*g_freq;
    printf("  [%-9s|streaming] cyc/dot=%.1f spread=%.1f%%  cyc/MAC=%.4f MAC/cyc=%.3f GMAC/s=%.2f  B/cyc(wt)=%.3f B/cyc(full)=%.3f\n",
      name,cyc_per_dot,spread,cyc_per_dot/macs,mac_per_cyc,gmacs,bpd_wt/cyc_per_dot,bpd_full/cyc_per_dot);
  };
  stream("q8_0_q8_0",ACCT_Q80,W80.data(),ROWS_Q80,WBYTES_Q80,q80_a.data(),tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0,nullptr);
  stream("q4_0_q8_0",ACCT_Q40,W40.data(),ROWS_Q40,WBYTES_Q40,q40_a.data(),nullptr,tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0);
  stream("q4_K_q8_K",ACCT_Q4K,W4K.data(),ROWS_Q4K,WBYTES_Q4K,q4k_a.data(),tcrv_emitc_ggml_vec_dot_q4_K_q8_K_kernel_ggml_vec_dot_q4_K_q8_K,nullptr);

  printf("\n(sink=%.3g)\n",(double)sink);
  return 0;
}
