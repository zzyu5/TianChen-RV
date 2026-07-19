// harness_gevm_q5.cpp -- drives the DEPLOYED q5_0 / q5_1 repack-GEVM kernels
// (CORE==PROD front-door emit) for the r51g [GAP-P1]-loosen family sweep.
// Same skeleton as harness_gevm_q4.cpp; adds the qh 5th-bit (stored as an 8-bit
// bitmask per (position,half)).
//
// q5_0 block (stride 352, x16): [0..31]=16 fp16 d; [32 + p*16 + c]=nibble byte
//   (lo=pos p, hi=pos p+16, each unsigned [0,15]); qh bitmask region @288:
//   lo qh byte @288 + p*2 + (c<8?0:1), hi qh byte @320 + p*2 + (c<8?0:1); bit (c%8).
//   value = nib - 16 + 16*qhbit  (offset-binary, in [-16,15]). No min.
//   act (block_q8_0, stride 34): [0..1]=fp16 d_a; [2+p]=int8.
//
// q5_1 block (stride 384, x16): [0..31]=16 fp16 d; [32..63]=16 fp16 m;
//   [64 + p*16 + c]=nibble byte; qh region @320: lo @320+p*2+(c/8), hi @352+p*2+(c/8).
//   value = nib + 16*qhbit  (unsigned [0,31]).  fold adds m*s_a.
//   act (block_q8_1, stride 36): [0..1]=fp16 d_a; [2..3]=fp16 s_a; [4+p]=int8.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

typedef _Float16 f16;

extern "C" void q50_gevm_mf2(size_t, float *, size_t, const uint8_t *, size_t, const uint8_t *, size_t, int32_t);
extern "C" void q50_gevm_m1 (size_t, float *, size_t, const uint8_t *, size_t, const uint8_t *, size_t, int32_t);
extern "C" void q51_gevm_mf2(size_t, float *, size_t, const uint8_t *, size_t, const uint8_t *, size_t, int32_t);
extern "C" void q51_gevm_m1 (size_t, float *, size_t, const uint8_t *, size_t, const uint8_t *, size_t, int32_t);
typedef void (*kern_t)(size_t, float *, size_t, const uint8_t *, size_t, const uint8_t *, size_t, int32_t);

static uint64_t rng_state;
static inline uint32_t xr(){ rng_state^=rng_state<<13; rng_state^=rng_state>>7; rng_state^=rng_state<<17; return (uint32_t)rng_state; }
static inline int8_t rnd_i8(){ return (int8_t)((int)(xr()%255)-127); }
static inline int rnd_raw5(){ return (int)(xr()%32); }   // raw 5-bit [0,31]
static inline f16 rnd_scale(){ return (f16)(0.008f + (float)(xr()%48)*0.001f); }
static inline void set_qh(uint8_t* qbyte, int col, int bit){ if(bit) *qbyte |= (uint8_t)(1u<<(col&7)); }

// ---------------- q5_0 ----------------
static void build_q50(std::vector<uint8_t>&W,std::vector<uint8_t>&A,size_t M,size_t K,uint64_t seed){
  rng_state=seed?seed:0x9e3779b97f4a7c15ull; size_t nblk=K/32,ncg=M/16;
  W.assign(ncg*nblk*352,0); A.assign(nblk*34,0);
  for(size_t cg=0;cg<ncg;++cg) for(size_t b=0;b<nblk;++b){ uint8_t* blk=&W[(cg*nblk+b)*352];
    for(size_t c=0;c<16;++c) ((f16*)blk)[c]=rnd_scale();
    for(size_t p=0;p<16;++p) for(size_t c=0;c<16;++c){
      int r_lo=rnd_raw5(), r_hi=rnd_raw5();          // positions p and p+16
      int nlo=r_lo&15, nhi=r_hi&15;
      blk[32 + p*16 + c] = (uint8_t)(nlo | (nhi<<4));
      set_qh(&blk[288 + p*2 + (c<8?0:1)], c, r_lo>>4);
      set_qh(&blk[320 + p*2 + (c<8?0:1)], c, r_hi>>4);
    }
  }
  for(size_t b=0;b<nblk;++b){ uint8_t* ab=&A[b*34]; ((f16*)ab)[0]=rnd_scale();
    for(size_t p=0;p<32;++p) ab[2+p]=(uint8_t)rnd_i8(); }
}
static void oracle_q50(const std::vector<uint8_t>&W,const std::vector<uint8_t>&A,std::vector<float>&out,size_t M,size_t K){
  size_t nblk=K/32; out.assign(M,0.0f);
  for(size_t col=0;col<M;++col){ size_t cg=col/16,c=col%16; float acc=0.0f;
    for(size_t b=0;b<nblk;++b){ const uint8_t* blk=&W[(cg*nblk+b)*352]; const uint8_t* ab=&A[b*34];
      int32_t isum=0;
      for(size_t pos=0;pos<32;++pos){ size_t p=pos%16; uint8_t byte=blk[32 + p*16 + c];
        int nib=(pos<16)?(byte&0xF):(byte>>4);
        size_t qoff=(pos<16)?(288 + p*2):(320 + p*2); qoff += (c<8?0:1);
        int qh=(blk[qoff]>>(c&7))&1;
        int q=nib - 16 + 16*qh;
        int a=(int8_t)ab[2+pos]; isum += q*a; }
      f16 dw=((const f16*)blk)[c]; f16 da=((const f16*)ab)[0];
      acc=fmaf((float)isum,(float)dw*(float)da,acc);
    }
    out[col]=acc;
  }
}
// ---------------- q5_1 ----------------
static void build_q51(std::vector<uint8_t>&W,std::vector<uint8_t>&A,size_t M,size_t K,uint64_t seed){
  rng_state=seed?seed:0x9e3779b97f4a7c15ull; size_t nblk=K/32,ncg=M/16;
  W.assign(ncg*nblk*384,0); A.assign(nblk*36,0);
  for(size_t cg=0;cg<ncg;++cg) for(size_t b=0;b<nblk;++b){ uint8_t* blk=&W[(cg*nblk+b)*384];
    for(size_t c=0;c<16;++c){ ((f16*)blk)[c]=rnd_scale(); ((f16*)(blk+32))[c]=rnd_scale(); }
    for(size_t p=0;p<16;++p) for(size_t c=0;c<16;++c){
      int r_lo=rnd_raw5(), r_hi=rnd_raw5(); int nlo=r_lo&15, nhi=r_hi&15;
      blk[64 + p*16 + c] = (uint8_t)(nlo | (nhi<<4));
      set_qh(&blk[320 + p*2 + (c<8?0:1)], c, r_lo>>4);
      set_qh(&blk[352 + p*2 + (c<8?0:1)], c, r_hi>>4);
    }
  }
  for(size_t b=0;b<nblk;++b){ uint8_t* ab=&A[b*36]; ((f16*)ab)[0]=rnd_scale();
    int64_t qs=0; for(size_t p=0;p<32;++p){ int8_t q=rnd_i8(); ab[4+p]=(uint8_t)q; qs+=q; }
    ((f16*)(ab+2))[0]=(f16)((float)((const f16*)ab)[0]*(float)qs); }
}
static void oracle_q51(const std::vector<uint8_t>&W,const std::vector<uint8_t>&A,std::vector<float>&out,size_t M,size_t K){
  size_t nblk=K/32; out.assign(M,0.0f);
  for(size_t col=0;col<M;++col){ size_t cg=col/16,c=col%16; float acc=0.0f;
    for(size_t b=0;b<nblk;++b){ const uint8_t* blk=&W[(cg*nblk+b)*384]; const uint8_t* ab=&A[b*36];
      int32_t isum=0;
      for(size_t pos=0;pos<32;++pos){ size_t p=pos%16; uint8_t byte=blk[64 + p*16 + c];
        int nib=(pos<16)?(byte&0xF):(byte>>4);
        size_t qoff=(pos<16)?(320 + p*2):(352 + p*2); qoff += (c<8?0:1);
        int qh=(blk[qoff]>>(c&7))&1;
        int q=nib + 16*qh;
        int a=(int8_t)ab[4+pos]; isum += q*a; }
      f16 dw=((const f16*)blk)[c]; f16 mw=((const f16*)(blk+32))[c];
      f16 da=((const f16*)ab)[0]; f16 sa=((const f16*)(ab+2))[0];
      acc=fmaf((float)isum,(float)dw*(float)da,acc);
      acc=acc+(float)mw*(float)sa;
    }
    out[col]=acc;
  }
}

static std::vector<uint8_t> g_junk(64ull*1024*1024);
static void flush_caches(){ volatile uint64_t s=0; for(size_t i=0;i<g_junk.size();i+=64) s+=g_junk[i]; (void)s; }
static uint64_t now_ns(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (uint64_t)ts.tv_sec*1000000000ull+ts.tv_nsec; }
static uint64_t time_cold(kern_t k,size_t K,float*out,size_t M,const uint8_t*w,const uint8_t*a,int reps){
  std::vector<uint64_t> s; for(int r=0;r<reps;++r){ flush_caches(); uint64_t t0=now_ns(); k(K,out,M,w,0,a,0,1); uint64_t t1=now_ns(); s.push_back(t1-t0);}
  std::sort(s.begin(),s.end()); return s[s.size()/2];
}
struct Fmt{ const char* name; kern_t mf2; kern_t m1;
  void(*build)(std::vector<uint8_t>&,std::vector<uint8_t>&,size_t,size_t,uint64_t);
  void(*oracle)(const std::vector<uint8_t>&,const std::vector<uint8_t>&,std::vector<float>&,size_t,size_t); };
int main(){
  Fmt fmts[]={ {"q5_0",q50_gevm_mf2,q50_gevm_m1,build_q50,oracle_q50},
               {"q5_1",q51_gevm_mf2,q51_gevm_m1,build_q51,oracle_q51} };
  struct Cfg{ size_t M,K; const char* tag; } cfgs[]={ {256,1024,"M=256 K=1024 cache-resident"},
    {2048,4096,"M=2048 K=4096 ~mid"},{8192,4096,"M=8192 K=4096 DRAM-bound"} };
  int reps=15;
  printf("format,footprint,seed,weight_bytes,mf2_median_ns,m1_median_ns,m1_over_mf2,byte_exact_3arm\n");
  int all_exact=1;
  for(auto& F:fmts) for(auto& c:cfgs) for(uint64_t seed=0;seed<2;++seed){
    std::vector<uint8_t> W,A; F.build(W,A,c.M,c.K,seed+1);
    std::vector<float> ref; F.oracle(W,A,ref,c.M,c.K);
    std::vector<float> o_mf2(c.M,0.0f),o_m1(c.M,0.0f);
    F.mf2(c.K,o_mf2.data(),c.M,W.data(),0,A.data(),0,1);
    F.m1 (c.K,o_m1.data(), c.M,W.data(),0,A.data(),0,1);
    int exact=(memcmp(o_mf2.data(),ref.data(),c.M*sizeof(float))==0)&&(memcmp(o_m1.data(),ref.data(),c.M*sizeof(float))==0);
    if(!exact){ all_exact=0; for(size_t i=0;i<c.M;++i) if(o_mf2[i]!=ref[i]||o_m1[i]!=ref[i]){
      fprintf(stderr,"MISMATCH %s %s seed %llu col %zu: ref=%.9g mf2=%.9g m1=%.9g\n",F.name,c.tag,(unsigned long long)seed,i,ref[i],o_mf2[i],o_m1[i]); break; } }
    uint64_t mf2=time_cold(F.mf2,c.K,o_mf2.data(),c.M,W.data(),A.data(),reps);
    uint64_t m1 =time_cold(F.m1, c.K,o_m1.data(), c.M,W.data(),A.data(),reps);
    printf("%s,%s,%llu,%zu,%llu,%llu,%.4f,%s\n",F.name,c.tag,(unsigned long long)seed,W.size(),
      (unsigned long long)mf2,(unsigned long long)m1,(double)m1/(double)mf2,exact?"PASS":"FAIL");
    fflush(stdout);
  }
  fprintf(stderr,"3-arm byte-exact: %s\n",all_exact?"PASS mism=0":"FAIL");
  return all_exact?0:1;
}
