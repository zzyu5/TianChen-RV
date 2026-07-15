// A2-batch7 — q4_K@ime KERNEL-SYM cold microbench (k1).
//   OURS   = weft format-keyed q4_K IME GEMM tile (vmadot two-level dmin/bsums fold
//            -> float), pre-decoded, with per-32-block activation scale d_a.
//   VENDOR = SpacemiT stock IME dispatch: q4_K -> q4_1x16 requant
//            (repack_q4_k_to_q4_1_16_bl / make_block_q4_1x16, with zero-points) then the
//            SAME ime1::gemm_kernel_i8i4 (zp != null). Vendor activation = quantize_a_4row_i8.
// W is DEFINED as dequant(random valid q4_K bytes) so both sides consume identical q4_K.
// Gates: OURS int32-core exact + f32 fold == reference (qa . W_dequant, per-block d_a).
//        VENDOR vs true-f32 W.X (Frobenius, tolerant of fp16(d1)+zp rounding) + vs ours.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <vector>

namespace spacemit_kernels { namespace ime1 {
size_t gemm_kernel_i8i4(size_t, const uint8_t*, const uint8_t*, const uint8_t*, float*, size_t, size_t, size_t, size_t);
void quantize_a_4row_i8(size_t, const float*, size_t, uint8_t*);
}}

typedef uint16_t ggml_half;
#define QK_K 256
#define QK4_1 32

// ------- vmadot leaf (0xe210312b) -------
static inline void vmadot_mac_kloop(const int8_t *A, const int8_t *B, long kt, int32_t *frag) {
  __asm__ volatile(
    "vsetvli t0,zero,e8,m1,ta,ma\n\tvmv.v.i v2,0\n\tvmv.v.i v3,0\n\t"
    "mv t2,%[kt]\n\tmv t3,%[pa]\n\tmv t4,%[pb]\n\t1:\n\tvle8.v v0,(t3)\n\tvle8.v v1,(t4)\n\t"
    "vmadot v2,v0,v1\n\taddi t3,t3,32\n\taddi t4,t4,32\n\taddi t2,t2,-1\n\tbnez t2,1b\n\t"
    "vsetvli t0,zero,e32,m1,ta,ma\n\tvse32.v v2,(%[pf])\n\taddi t5,%[pf],32\n\tvse32.v v3,(t5)\n\t"
    ::[pa]"r"(A),[pb]"r"(B),[kt]"r"(kt),[pf]"r"(frag)
    :"t0","t2","t3","t4","t5","v0","v1","v2","v3","memory");
}

// ------- fp16 <-> f32 -------
static uint16_t f32_to_fp16(float f){uint32_t x;memcpy(&x,&f,4);uint32_t s=(x>>16)&0x8000u;
  int32_t e=(int32_t)((x>>23)&0xFF)-127+15;uint32_t m=x&0x7FFFFFu;
  if(((x>>23)&0xFF)==0xFF)return (uint16_t)(s|0x7C00u|(m?0x200u:0));
  if(e>=0x1F)return (uint16_t)(s|0x7C00u);
  if(e<=0){if(e<-10)return (uint16_t)s;m|=0x800000u;uint32_t sh=(uint32_t)(14-e);uint32_t h=m>>sh;
    if((m>>(sh-1))&1)h++;return (uint16_t)(s|h);}
  uint16_t h=(uint16_t)(s|((uint32_t)e<<10)|(m>>13));if((m>>12)&1)h++;return h;}
static float fp16_to_f32(uint16_t h){uint32_t s=(uint32_t)(h&0x8000u)<<16;uint32_t e=(h>>10)&0x1Fu;
  uint32_t m=h&0x3FFu;uint32_t b;
  if(e==0u){if(m==0u)b=s;else{e=127u-15u+1u;while((m&0x400u)==0u){m<<=1;e--;}m&=0x3FFu;b=s|(e<<23)|(m<<13);}}
  else if(e==0x1Fu)b=s|0x7F800000u|(m<<13);else b=s|((e-15u+127u)<<23)|(m<<13);
  float f;memcpy(&f,&b,4);return f;}

// q4_K get_scale_min_k4
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *m){
  if(j<4){*sc=q[j]&63;*m=q[j+4]&63;}
  else{*sc=(q[j+4]&0xF)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j-0]>>6)<<4);}}
// q4_K nibble for element p (0..255) in a 144B ggml block
static int q4k_nibble(const uint8_t *blk, long p){long b=p/32,pl=p%32;const uint8_t*qs=blk+16;
  uint8_t byte=qs[(b/2)*32+pl];return (b&1)?(byte>>4):(byte&0x0F);}

// ================= OURS q4_K IME kernel (pre-decoded int8 weight, two-level fold + d_a) ==
// Bint8: [nt][nsb*8 subblocks][... ] fragment int8; here we pre-decode nibbles to int8
// laid out fragment-major [nt col-tile][ K/8 frag *32 ]. sc/m per (n,sb,b) precomputed.
static void ours_q4k_matmul(const int8_t *Apack, const float *dA, const int8_t *Bint8,
                            const uint8_t *scAll, const uint8_t *mAll,
                            const float *dW, const float *dminW,
                            float *Cf, long M, long N, long K) {
  const long mt=M/4, nt=N/4, nsb=K/256, kt=K/8;
  for(long mi=0;mi<mt;++mi){const int8_t *Arow=Apack+mi*4*K;
    for(long nj=0;nj<nt;++nj){const int8_t *Bcol=Bint8+nj*kt*32;
      for(long sb=0;sb<nsb;++sb){
        for(int b=0;b<8;++b){
          long blk32=sb*8+b;                       // 32-block index
          const int8_t *Ablk=Arow+blk32*4*32;       // 4 frags of this 32-block
          const int8_t *Bblk=Bcol+blk32*4*32;
          int32_t sumi[16]; vmadot_mac_kloop(Ablk,Bblk,4,sumi);
          int32_t asum[4]={0,0,0,0};
          for(int kf=0;kf<4;++kf)for(int ml=0;ml<4;++ml)for(int kl=0;kl<8;++kl)
            asum[ml]+=(int32_t)Ablk[kf*32+ml*8+kl];
          for(int ml=0;ml<4;++ml){long mo=mi*4+ml; float da=dA[mo*(nsb*8)+blk32];
            for(int nl=0;nl<4;++nl){long no=nj*4+nl;
              uint8_t sc=scAll[((no*nsb+sb)*8)+b], mm=mAll[((no*nsb+sb)*8)+b];
              float d=dW[no*nsb+sb], dmin=dminW[no*nsb+sb];
              Cf[mo*N+no]+=da*(d*(float)sc*(float)sumi[ml*4+nl] - dmin*(float)mm*(float)asum[ml]);
            }}
        }
      }
    }
  }
}

// ================= VENDOR q4_1x16 layout + repack =================
struct block_q4_1x16 { ggml_half d[16]; uint8_t zp[16]; uint8_t qs[256]; }; // 304B
// make_block_q4_1x16 (reproduced): in[16] each = {d(fp16), m(fp16), qs[16]}
struct blk41 { ggml_half d, m; uint8_t qs[16]; };
static block_q4_1x16 make_block_q4_1x16(const blk41 *in){
  block_q4_1x16 out;
  for(int i=0;i<16;i++){float d=fp16_to_f32(in[i].d),m=fp16_to_f32(in[i].m);
    float mid=-nearbyintf(m/d); mid=std::min(15.0f,std::max(0.0f,mid));
    out.d[i]=f32_to_fp16(d); out.zp[i]=(uint8_t)mid;}
  for(int i=0;i<16;i++)for(int j=0;j<QK4_1/4;j++)
    out.qs[i*QK4_1/4+j]=(in[i].qs[j]&0x0F)|((in[i].qs[j+QK4_1/4]&0x0F)<<4);
  for(int i=0;i<16;i++)for(int j=0;j<QK4_1/4;j++)
    out.qs[4*QK4_1+i*QK4_1/4+j]=((in[i].qs[j]&0xF0)>>4)|(in[i].qs[j+QK4_1/4]&0xF0);
  return out;
}

static double now_s(){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static volatile int64_t g_sink=0;
static void flush_cache(volatile int8_t*b,long n){int64_t s=0;for(long i=0;i<n;i+=64)s+=b[i];g_sink+=s;}
static double median_of(std::vector<double>&v){std::sort(v.begin(),v.end());size_t n=v.size();
  return n&1?v[n/2]:0.5*(v[n/2-1]+v[n/2]);}
static double reliqr(std::vector<double> v){std::sort(v.begin(),v.end());size_t n=v.size();
  double q1=v[n/4],q3=v[(3*n)/4],md=v[n/2];return md>0?(q3-q1)/md:0;}

int main(int argc,char**argv){
  long M=argc>1?atol(argv[1]):64, N=argc>2?atol(argv[2]):512, K=argc>3?atol(argv[3]):2048;
  int REPS=argc>4?atoi(argv[4]):25; unsigned seed=argc>5?(unsigned)strtoul(argv[5],0,0):0xC0FFEE1u;
  const long nsb=K/256, nb32=K/32, mt=M/4, nt=N/4, kt=K/8, q4kbb=144;
  printf("## q4_K@ime kernel-sym  M=%ld N=%ld K=%ld reps=%d seed=0x%X\n",M,N,K,REPS,seed);
  srand(seed);

  // ---- generate REALISTIC q4_K weight bytes (ggml layout), per (n, sb) ----
  // Constrain per-sub-block 6-bit sc in [20,63] (=> d1=d*sc>0, no degenerate block),
  // m in [0,15] (=> min term small vs scale => q4_1 zp faithful). Pack via inverse
  // get_scale_min_k4. d/dmin moderate fp16 (exp 14 ~0.5) so |W| is model-realistic.
  std::vector<uint8_t> Wq4k((size_t)N*nsb*q4kbb);
  for(long n=0;n<N;++n)for(long sb=0;sb<nsb;++sb){
    uint8_t *blk=&Wq4k[(n*nsb+sb)*q4kbb];
    uint16_t d=(uint16_t)((rand()&0x03FF)|(14<<10));
    uint16_t dm=(uint16_t)((rand()&0x03FF)|(13<<10));
    blk[0]=(uint8_t)(d&0xFF);blk[1]=(uint8_t)(d>>8);blk[2]=(uint8_t)(dm&0xFF);blk[3]=(uint8_t)(dm>>8);
    uint8_t sc[8],mm[8];
    for(int b=0;b<8;++b){sc[b]=(uint8_t)(20+rand()%44); mm[b]=(uint8_t)(rand()%16);}
    uint8_t *q=blk+4;
    for(int j=0;j<4;++j){
      q[j]   =(uint8_t)((sc[j]&63)   | (((sc[j+4]>>4)&3)<<6));
      q[j+4] =(uint8_t)((mm[j]&63)   | (((mm[j+4]>>4)&3)<<6));
      q[j+8] =(uint8_t)((sc[j+4]&0xF)| ((mm[j+4]&0xF)<<4));
    }
    for(int i=0;i<128;i++)blk[16+i]=(uint8_t)(rand()&0xFF); // nibbles (0..15 both halves)
  }
  // true W[n][k] = dequant(q4_K)
  std::vector<float> W((size_t)N*K);
  std::vector<uint8_t> scAll((size_t)N*nsb*8), mAll((size_t)N*nsb*8);
  std::vector<float> dW(N*nsb), dminW(N*nsb);
  for(long n=0;n<N;++n)for(long sb=0;sb<nsb;++sb){
    const uint8_t *blk=&Wq4k[(n*nsb+sb)*q4kbb];
    float d=fp16_to_f32((uint16_t)(blk[0]|(blk[1]<<8))), dmin=fp16_to_f32((uint16_t)(blk[2]|(blk[3]<<8)));
    dW[n*nsb+sb]=d; dminW[n*nsb+sb]=dmin;
    for(int b=0;b<8;++b){uint8_t sc,mm;get_scale_min_k4(b,blk+4,&sc,&mm);
      scAll[(n*nsb+sb)*8+b]=sc; mAll[(n*nsb+sb)*8+b]=mm;
      for(int pl=0;pl<32;++pl){long p=b*32+pl,k=sb*256+p;
        W[n*K+k]=d*(float)sc*(float)q4k_nibble(blk,p)-dmin*(float)mm;}}
  }
  // ---- activation X, canonical q8_0-per-32-block (ours) ----
  std::vector<float> X((size_t)M*K); for(auto&v:X)v=((float)rand()/2147483647.0f)*2.0f-1.0f;
  std::vector<float> dA((size_t)M*nb32); std::vector<int8_t> qa((size_t)M*K);
  for(long m=0;m<M;++m)for(long j=0;j<nb32;++j){
    float amax=0; for(int t=0;t<32;t++){float a=fabsf(X[m*K+j*32+t]);if(a>amax)amax=a;}
    float d=amax/127.0f; uint16_t dh=f32_to_fp16(d); float dr=fp16_to_f32(dh);
    float id=dr?1.0f/dr:0.0f; dA[m*nb32+j]=dr;
    for(int t=0;t<32;t++){int q=(int)roundf(X[m*K+j*32+t]*id);if(q<-127)q=-127;if(q>127)q=127;
      qa[m*K+j*32+t]=(int8_t)q;}
  }
  // ---- OURS packing: Apack fragment-major; Bint8 pre-decoded nibbles fragment-major ----
  std::vector<int8_t> Apack((size_t)mt*kt*32);
  for(long m=0;m<M;++m)for(long k=0;k<K;++k){long mi=m/4,ml=m%4,kf=k/8,kl=k%8;
    Apack[mi*4*K+kf*32+ml*8+kl]=qa[m*K+k];}
  std::vector<int8_t> Bint8((size_t)nt*kt*32);
  for(long n=0;n<N;++n)for(long k=0;k<K;++k){long nj=n/4,nl=n%4,kf=k/8,kl=k%8;
    long sb=k/256, p=k%256;
    int nib=q4k_nibble(&Wq4k[(n*nsb+sb)*q4kbb],p);
    // fragment layout: Bint8[nj col-tile][frag kf *32 + nl*8 + kl]
    Bint8[nj*kt*32 + kf*32 + nl*8 + kl]=(int8_t)nib;}

  // ---- VENDOR packing: q4_K -> q4_1x16 (per sub-block) + quantize_a ----
  // layout mirrors repack_q4_k_to_q4_1_16_bl: for grp of 16 rows, for sb, for j(8): make_block_q4_1x16
  // row_stride_b = PER-COLUMN stride = k_blks * get_repacked_block_type_size(q4_1=19);
  // b_col advances 16*row_stride_b = nb32*304 = one 16-col group (nb32 blocks of 304B).
  const long row_stride_b = nb32 * 19;                            // (sizeof(block_q4_0)+1)*32/QK4_1 = 19
  std::vector<uint8_t> Wpack((size_t)(N/16)*nb32*sizeof(block_q4_1x16));
  {
    block_q4_1x16 *dst=(block_q4_1x16*)Wpack.data();
    for(long grp=0;grp<N;grp+=16)for(long sb=0;sb<nsb;++sb)for(int j=0;j<8;++j){
      blk41 tmp[16];
      for(int i=0;i<16;i++){long n=grp+i;const uint8_t*blk=&Wq4k[(n*nsb+sb)*q4kbb];
        float d=fp16_to_f32((uint16_t)(blk[0]|(blk[1]<<8))),mn=fp16_to_f32((uint16_t)(blk[2]|(blk[3]<<8)));
        uint8_t sc,mm;get_scale_min_k4(j,blk+4,&sc,&mm);
        float d1=d*(float)sc, m1=mn*(float)mm;
        tmp[i].d=f32_to_fp16(d1); tmp[i].m=f32_to_fp16(-m1);
        const uint8_t*q=blk+16+(j/2)*QK4_1;  // qs + (j/2)*32
        if(j%2==0)for(int ii=0;ii<16;ii++)tmp[i].qs[ii]=(q[ii]&0x0F)|((q[ii+16]&0x0F)<<4);
        else for(int ii=0;ii<16;ii++)tmp[i].qs[ii]=((q[ii]&0xF0)>>4)|(q[ii+16]&0xF0);
      }
      *dst++=make_block_q4_1x16(tmp);
    }
  }
  const long block_stride_a=4+32, row_stride_a=nb32*block_stride_a;
  std::vector<uint8_t> QA((size_t)M*row_stride_a);
  for(long m0=0;m0<M;m0+=4)
    spacemit_kernels::ime1::quantize_a_4row_i8(32,&X[m0*K],K,&QA[m0*row_stride_a]);

  // ================= GATES =================
  std::vector<float> Cours(M*N,0.0f);
  ours_q4k_matmul(Apack.data(),dA.data(),Bint8.data(),scAll.data(),mAll.data(),
                  dW.data(),dminW.data(),Cours.data(),M,N,K);
  // ours vs reference (qa . W_dequant with per-block d_a) -- exact int core + f32 fold
  double ours_maxrel=0;
  for(long m=0;m<M;++m)for(long n=0;n<N;++n){double ref=0.0;
    for(long j=0;j<nb32;++j){long sb=j/8,b=j%8;double sumi=0,asum=0;
      const uint8_t*blk=&Wq4k[(n*nsb+sb)*q4kbb];
      for(int pl=0;pl<32;pl++){long p=b*32+pl;int a=qa[m*K+j*32+pl];
        sumi+=(double)a*q4k_nibble(blk,p);asum+=(double)a;}
      double d=dW[n*nsb+sb],dmin=dminW[n*nsb+sb];
      ref+=(double)dA[m*nb32+j]*(d*(double)scAll[(n*nsb+sb)*8+b]*sumi
                                 -dmin*(double)mAll[(n*nsb+sb)*8+b]*asum);}
    double ae=fabs((double)Cours[m*N+n]-ref),re=fabs(ref)>1e-6?ae/fabs(ref):ae;
    if(re>ours_maxrel)ours_maxrel=re;}
  // vendor run once
  std::vector<float> Cven(M*N,0.0f);
  for(long m0=0;m0<M;m0+=4){uint8_t*b_col=Wpack.data();
    for(long ni=0;ni<N;ni+=16){
      spacemit_kernels::ime1::gemm_kernel_i8i4(32,&QA[m0*row_stride_a],b_col,b_col,
                                               &Cven[m0*N+ni],4,16,nb32,N);
      b_col+=16*row_stride_b;}}
  double sse_vt=0,sse_ot=0,sse_vo=0,sst=0,truemax=0;
  for(long m=0;m<M;++m)for(long n=0;n<N;++n){double t=0.0;
    for(long k=0;k<K;++k)t+=(double)X[m*K+k]*(double)W[n*K+k];
    double v=Cven[m*N+n],o=Cours[m*N+n];
    sse_vt+=(v-t)*(v-t);sse_ot+=(o-t)*(o-t);sse_vo+=(v-o)*(v-o);sst+=t*t;
    if(fabs(t)>truemax)truemax=fabs(t);}
  double ven_vs_true=sqrt(sse_vt/sst),ours_vs_true=sqrt(sse_ot/sst),ven_vs_ours=sqrt(sse_vo/sst);
  printf("GATE ours: maxrel(vs qa.W_dequant)=%.3e\n",ours_maxrel);
  printf("GATE vendor(Frobenius-rel): vendor_vs_true=%.4f ours_vs_true=%.4f vendor_vs_ours=%.4f (|C|max=%.2f)\n",
         ven_vs_true,ours_vs_true,ven_vs_ours,truemax);
  for(int n=0;n<4&&n<N;++n){double t=0.0;for(long k=0;k<K;++k)t+=(double)X[k]*(double)W[n*K+k];
    printf("   n=%d true=%9.4f ours=%9.4f vendor=%9.4f\n",n,t,Cours[n],Cven[n]);}
  // q4_1 requant is faithful up to fp16(d1)+zp integer rounding => allow looser vendor tol
  int gate_ok=(ours_maxrel<1.2e-2)&&(ven_vs_ours<0.05)&&(ven_vs_true<0.12);
  printf("GATE %s\n",gate_ok?"PASS":"FAIL");
  if(!gate_ok){printf("ABORT: gate fail, no timing (no fabrication)\n");return 2;}

  // ================= COLD TIMING =================
  const long FLUSH=32L<<20; std::vector<int8_t> fb(FLUSH,1); std::vector<float> Ct(M*N);
  std::vector<double> t_ours,t_ven; int WARM=3;
  for(int r=0;r<REPS+WARM;++r){
    flush_cache(fb.data(),FLUSH);std::fill(Ct.begin(),Ct.end(),0.0f);double a0=now_s();
    ours_q4k_matmul(Apack.data(),dA.data(),Bint8.data(),scAll.data(),mAll.data(),dW.data(),dminW.data(),Ct.data(),M,N,K);
    double a1=now_s();g_sink+=(int64_t)Ct[0];
    flush_cache(fb.data(),FLUSH);std::fill(Ct.begin(),Ct.end(),0.0f);double a2=now_s();
    for(long m0=0;m0<M;m0+=4){uint8_t*b_col=Wpack.data();
      for(long ni=0;ni<N;ni+=16){spacemit_kernels::ime1::gemm_kernel_i8i4(32,&QA[m0*row_stride_a],b_col,b_col,&Ct[m0*N+ni],4,16,nb32,N);b_col+=16*row_stride_b;}}
    double a3=now_s();g_sink+=(int64_t)Ct[0];
    if(r>=WARM){t_ours.push_back((a1-a0)*1e3);t_ven.push_back((a3-a2)*1e3);}
  }
  double mo=median_of(t_ours),mv=median_of(t_ven);
  printf("COLD ms (median,N=%d): ours=%.4f (iqr%.1f%%)  vendor=%.4f (iqr%.1f%%)\n",
         REPS,mo,reliqr(t_ours)*100,mv,reliqr(t_ven)*100);
  printf("RATIO vendor/ours (>=0.8 PASS): %.4f\n",mv/mo);
  printf("SINK %lld\n",(long long)g_sink);
  return 0;
}
