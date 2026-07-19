// harness_gemm.cpp -- drives the DEPLOYED q4_0/q4_1/q5_0/q5_1 repack-GEMM (PREFILL)
// kernels (CORE==PROD emit) for the r51g family sweep. The prefill columnsPerPass
// tradeoff (mf2=4 vs m1=1) is a DIFFERENT crossover than the decode GEVM, so it is
// measured SEPARATELY (not extrapolated). Weight block layout = identical to GEVM;
// activation = block_q8_0x4 (q4_0/q5_0, 136B) or block_q8_1x4 (q4_1/q5_1, 144B).
//
// Kernel ABI: gemm(M, obs, K, out, N, weight, act).  out[row*N+col], row=rg*4+r.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
typedef _Float16 f16;

extern "C" void q40_gemm_mf2(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern "C" void q40_gemm_m1 (size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern "C" void q41_gemm_mf2(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern "C" void q41_gemm_m1 (size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern "C" void q50_gemm_mf2(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern "C" void q50_gemm_m1 (size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern "C" void q51_gemm_mf2(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
extern "C" void q51_gemm_m1 (size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);
typedef void(*kern_t)(size_t,size_t,size_t,float*,size_t,const uint8_t*,const uint8_t*);

static uint64_t rs;
static inline uint32_t xr(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (uint32_t)rs; }
static inline int8_t ri8(){ return (int8_t)((int)(xr()%255)-127); }
static inline int rnib_s(){ return (int)(xr()%16)-8; }
static inline int rnib_u(){ return (int)(xr()%16); }
static inline int rraw5(){ return (int)(xr()%32); }
static inline f16 rsc(){ return (f16)(0.008f+(float)(xr()%48)*0.001f); }
static inline void setqh(uint8_t* qb,int col,int bit){ if(bit) *qb|=(uint8_t)(1u<<(col&7)); }

// build activation block_q8_0x4 (wsum=0) or block_q8_1x4 (wsum=1)
static void build_act(std::vector<uint8_t>&A,size_t M,size_t K,int wsum){
  size_t nb=K/32,nrg=M/4; size_t stride=wsum?144:136;
  A.assign(nrg*nb*stride,0);
  for(size_t rg=0;rg<nrg;++rg)for(size_t b=0;b<nb;++b){ uint8_t* ab=&A[(rg*nb+b)*stride];
    size_t qoff=wsum?16:8;
    for(size_t r=0;r<4;++r){ ((f16*)(ab+r*2))[0]=rsc();
      int64_t sum=0; /*filled after quants*/ (void)sum; }
    for(size_t p=0;p<32;++p)for(size_t r=0;r<4;++r) ab[qoff+p*4+r]=(uint8_t)ri8();
    if(wsum){ for(size_t r=0;r<4;++r){ int64_t s=0; for(size_t p=0;p<32;++p) s+=(int8_t)ab[16+p*4+r];
      ((f16*)(ab+8+r*2))[0]=(f16)((float)((f16*)(ab+r*2))[0]*(float)s); } }
  }
}
static inline int actq(const uint8_t* ab,int pos,int r,int wsum){ return (int8_t)ab[(wsum?16:8)+pos*4+r]; }
static inline float actd(const uint8_t* ab,int r){ return (float)((const f16*)(ab+r*2))[0]; }
static inline float acts(const uint8_t* ab,int r){ return (float)((const f16*)(ab+8+r*2))[0]; }

// weight decode: returns quant value at (block ptr, pos, col) for each format
static inline int wq_q40(const uint8_t* wb,int pos,int c){ uint8_t byte=wb[32+(pos%16)*16+c];
  return (pos<16)?((int)(int8_t)(byte<<4)>>4):((int)((int8_t)byte>>4)); }
static inline int wq_q41(const uint8_t* wb,int pos,int c){ uint8_t byte=wb[64+(pos%16)*16+c];
  return (pos<16)?(byte&0xF):(byte>>4); }
static inline int wq_q50(const uint8_t* wb,int pos,int c){ int p=pos%16; uint8_t byte=wb[32+p*16+c];
  int nib=(pos<16)?(byte&0xF):(byte>>4); size_t qoff=(pos<16)?(288+p*2):(320+p*2); qoff+=(c<8?0:1);
  int qh=(wb[qoff]>>(c&7))&1; return nib-16+16*qh; }
static inline int wq_q51(const uint8_t* wb,int pos,int c){ int p=pos%16; uint8_t byte=wb[64+p*16+c];
  int nib=(pos<16)?(byte&0xF):(byte>>4); size_t qoff=(pos<16)?(320+p*2):(352+p*2); qoff+=(c<8?0:1);
  int qh=(wb[qoff]>>(c&7))&1; return nib+16*qh; }

struct Fmt{ const char* name; kern_t mf2,m1; size_t wstride; int wsum; int hasmin;
  int (*wq)(const uint8_t*,int,int);
  void (*wfill)(uint8_t*,int,int,uint64_t&); };

static void wfill_q40(uint8_t* blk,int,int,uint64_t&){ for(int c=0;c<16;++c) ((f16*)blk)[c]=rsc();
  for(int p=0;p<16;++p)for(int c=0;c<16;++c){ int lo=rnib_s(),hi=rnib_s(); blk[32+p*16+c]=(uint8_t)((lo&0xF)|((hi&0xF)<<4)); } }
static void wfill_q41(uint8_t* blk,int,int,uint64_t&){ for(int c=0;c<16;++c){ ((f16*)blk)[c]=rsc(); ((f16*)(blk+32))[c]=rsc(); }
  for(int p=0;p<16;++p)for(int c=0;c<16;++c){ int lo=rnib_u(),hi=rnib_u(); blk[64+p*16+c]=(uint8_t)((lo&0xF)|((hi&0xF)<<4)); } }
static void wfill_q50(uint8_t* blk,int,int,uint64_t&){ for(int c=0;c<16;++c) ((f16*)blk)[c]=rsc();
  for(int p=0;p<16;++p)for(int c=0;c<16;++c){ int rl=rraw5(),rh=rraw5(); blk[32+p*16+c]=(uint8_t)((rl&15)|((rh&15)<<4));
    setqh(&blk[288+p*2+(c<8?0:1)],c,rl>>4); setqh(&blk[320+p*2+(c<8?0:1)],c,rh>>4); } }
static void wfill_q51(uint8_t* blk,int,int,uint64_t&){ for(int c=0;c<16;++c){ ((f16*)blk)[c]=rsc(); ((f16*)(blk+32))[c]=rsc(); }
  for(int p=0;p<16;++p)for(int c=0;c<16;++c){ int rl=rraw5(),rh=rraw5(); blk[64+p*16+c]=(uint8_t)((rl&15)|((rh&15)<<4));
    setqh(&blk[320+p*2+(c<8?0:1)],c,rl>>4); setqh(&blk[352+p*2+(c<8?0:1)],c,rh>>4); } }

static void build_w(const Fmt&F,std::vector<uint8_t>&W,size_t N,size_t K){
  size_t nb=K/32,ncg=N/16; W.assign(ncg*nb*F.wstride,0);
  uint64_t dummy=0;
  for(size_t cg=0;cg<ncg;++cg)for(size_t b=0;b<nb;++b) F.wfill(&W[(cg*nb+b)*F.wstride],(int)cg,(int)b,dummy);
}
static void oracle(const Fmt&F,const std::vector<uint8_t>&W,const std::vector<uint8_t>&A,std::vector<float>&out,size_t M,size_t N,size_t K){
  size_t nb=K/32; size_t astride=F.wsum?144:136; out.assign(M*N,0.0f);
  for(size_t rg=0;rg<M/4;++rg)for(int r=0;r<4;++r){ size_t row=rg*4+r;
    for(size_t cg=0;cg<N/16;++cg)for(int cl=0;cl<16;++cl){ size_t col=cg*16+cl; float acc=0.0f;
      for(size_t b=0;b<nb;++b){ const uint8_t* wb=&W[(cg*nb+b)*F.wstride]; const uint8_t* ab=&A[(rg*nb+b)*astride];
        int32_t isum=0; for(int p=0;p<32;++p) isum += F.wq(wb,p,cl)*actq(ab,p,r,F.wsum);
        float dw=(float)((const f16*)wb)[cl]; float da=actd(ab,r);
        acc=fmaf((float)isum, dw*da, acc);
        if(F.hasmin){ float mw=(float)((const f16*)(wb+32))[cl]; acc=acc+mw*acts(ab,r); }
      }
      out[row*N+col]=acc;
    }
  }
}

static std::vector<uint8_t> junk(64ull*1024*1024);
static void flush(){ volatile uint64_t s=0; for(size_t i=0;i<junk.size();i+=64)s+=junk[i]; (void)s; }
static uint64_t now_ns(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (uint64_t)ts.tv_sec*1000000000ull+ts.tv_nsec; }
static uint64_t tcold(kern_t k,size_t M,size_t K,float*o,size_t N,const uint8_t*w,const uint8_t*a,int reps){
  std::vector<uint64_t> s; for(int r=0;r<reps;++r){ flush(); uint64_t t0=now_ns(); k(M,N,K,o,N,w,a); uint64_t t1=now_ns(); s.push_back(t1-t0);}
  std::sort(s.begin(),s.end()); return s[s.size()/2];
}
int main(){
  Fmt fmts[]={
    {"q4_0",q40_gemm_mf2,q40_gemm_m1,288,0,0,wq_q40,wfill_q40},
    {"q4_1",q41_gemm_mf2,q41_gemm_m1,320,1,1,wq_q41,wfill_q41},
    {"q5_0",q50_gemm_mf2,q50_gemm_m1,352,0,0,wq_q50,wfill_q50},
    {"q5_1",q51_gemm_mf2,q51_gemm_m1,384,1,1,wq_q51,wfill_q51},
  };
  struct Cfg{ size_t M,N,K; const char* tag; } cfgs[]={
    {256,256,1024,"M=256 N=256 K=1024"},{512,256,2048,"M=512 N=256 K=2048"},{512,512,2048,"M=512 N=512 K=2048"} };
  int reps=7;
  printf("format,footprint,seed,mf2_median_ns,m1_median_ns,m1_over_mf2,byte_exact_3arm\n");
  int all=1;
  for(auto& F:fmts)for(auto& c:cfgs)for(uint64_t seed=0;seed<2;++seed){
    rs=seed+1; std::vector<uint8_t> W; build_w(F,W,c.N,c.K);
    std::vector<uint8_t> A; build_act(A,c.M,c.K,F.wsum);
    std::vector<float> ref; oracle(F,W,A,ref,c.M,c.N,c.K);
    std::vector<float> omf2(c.M*c.N,0.0f),om1(c.M*c.N,0.0f);
    F.mf2(c.M,c.N,c.K,omf2.data(),c.N,W.data(),A.data());
    F.m1 (c.M,c.N,c.K,om1.data(), c.N,W.data(),A.data());
    int ex=(memcmp(omf2.data(),ref.data(),c.M*c.N*sizeof(float))==0)&&(memcmp(om1.data(),ref.data(),c.M*c.N*sizeof(float))==0);
    if(!ex){ all=0; for(size_t i=0;i<c.M*c.N;++i) if(omf2[i]!=ref[i]||om1[i]!=ref[i]){
      fprintf(stderr,"MISMATCH %s %s seed %llu idx %zu: ref=%.9g mf2=%.9g m1=%.9g\n",F.name,c.tag,(unsigned long long)seed,i,ref[i],omf2[i],om1[i]); break; } }
    uint64_t mf2=tcold(F.mf2,c.M,c.K,omf2.data(),c.N,W.data(),A.data(),reps);
    uint64_t m1 =tcold(F.m1, c.M,c.K,om1.data(), c.N,W.data(),A.data(),reps);
    printf("%s,%s,%llu,%llu,%llu,%.4f,%s\n",F.name,c.tag,(unsigned long long)seed,
      (unsigned long long)mf2,(unsigned long long)m1,(double)m1/(double)mf2,ex?"PASS":"FAIL");
    fflush(stdout);
  }
  fprintf(stderr,"3-arm byte-exact (GEMM, all): %s\n",all?"PASS mism=0":"FAIL");
  return all?0:1;
}
