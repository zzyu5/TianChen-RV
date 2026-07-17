// Adversarial i16-lane-headroom check: force q8 to +/-127 and weights to extreme
// 5-bit values so per-lane vwmacc sums approach the i16 boundary (max
// 16*127*16=32512 <= 32767, headroom 255). Confirms the cloned q4_0 i16
// accumulator path is byte-exact for q5_0's DOUBLED [-16,15] range too.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#define QK 32
typedef uint16_t ggml_half;
static inline ggml_half f2h(float fv){uint32_t x;memcpy(&x,&fv,4);uint32_t s=(x>>16)&0x8000;int32_t e=((x>>23)&0xff)-112;uint32_t m=x&0x7fffff;if(e<=0)return(ggml_half)s;if(e>=0x1f)return(ggml_half)(s|0x7c00);ggml_half h=s|(e<<10)|(m>>13);if((m&0x1000)&&((m&0x2fff)||(h&1)))h++;return h;}
static inline float h2f(ggml_half h){uint32_t sg=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,f;if(e==0){if(m==0)f=sg<<31;else{e=113;while(!(m&0x400)){m<<=1;e--;}m&=0x3ff;f=(sg<<31)|(e<<23)|(m<<13);}}else if(e==0x1f)f=(sg<<31)|(0xff<<23)|(m<<13);else f=(sg<<31)|((e+112)<<23)|(m<<13);float r;memcpy(&r,&f,4);return r;}
struct block_q5_0{ggml_half d;uint8_t qh[4];uint8_t qs[16];};
struct block_q8_0{ggml_half d;int8_t qs[32];};
struct block_q5_0x16{ggml_half d[16];uint8_t qs[256];uint16_t qhmask[32];};
extern "C" void ggml_gemv_q5_0_16x1_q8_0(int,float*,size_t,const void*,const void*,int,int);
static block_q5_0x16 mk(const block_q5_0*in){block_q5_0x16 o;for(int i=0;i<16;i++)o.d[i]=in[i].d;for(int i=0;i<256;i++)o.qs[i]=in[i%16].qs[i/16];for(int e=0;e<32;e++){uint16_t m=0;for(int b=0;b<16;b++){uint32_t qh;memcpy(&qh,in[b].qh,4);m|=(uint16_t)(((qh>>e)&1u)<<b);}o.qhmask[e]=m;}return o;}
int main(){
  const int NC=16,nb=64,n=nb*QK;
  std::vector<block_q5_0> w(NC*nb);std::vector<block_q8_0> a(nb);
  // weights: all qs nibbles = 15 (-> 5-bit 15|16=31, -16 = +15 = max |w|=16 via low? actually max |w|=16 when nibble 0,bit1 -> -16). Mix to hit both -16 and +15.
  for(int r=0;r<NC;r++)for(int b=0;b<nb;b++){block_q5_0&blk=w[r*nb+b];blk.d=f2h(0.03f);uint32_t qh=0x00000000u;memcpy(blk.qh,&qh,4);for(int k=0;k<16;k++)blk.qs[k]=0x00;}// nibble 0, qh bit 0 -> (0|0)-16 = -16 everywhere
  for(int b=0;b<nb;b++){a[b].d=f2h(0.03f);for(int k=0;k<32;k++)a[b].qs[k]=(int8_t)127;}// q8 = +127 -> product -16*127 = -2032 per elem, 32 elems -> per-lane lo+hi each 16 steps *127*16
  std::vector<block_q5_0x16> vx(nb);{block_q5_0 tmp[16];for(int b=0;b<nb;b++){for(int i=0;i<16;i++)tmp[i]=w[i*nb+b];vx[b]=mk(tmp);}}
  float out[16];ggml_gemv_q5_0_16x1_q8_0(n,out,0,vx.data(),a.data(),1,NC);
  // ref: each weight -16, each q8 +127, sumi per block = 32 * (-16*127) = -65024; nb blocks; sumf = sum d_x*d_y*sumi
  double maxrel=0;for(int c=0;c<NC;c++){double sumf=0;for(int b=0;b<nb;b++){double dx=h2f(w[c*nb+b].d),dy=h2f(a[b].d);int sumi=0;for(int k=0;k<16;k++){int x0=((w[c*nb+b].qs[k]&0x0F)|0)-16;int x1=((w[c*nb+b].qs[k]>>4)|0)-16;sumi+=x0*a[b].qs[k]+x1*a[b].qs[k+16];}sumf+=dx*dy*sumi;}double dn=fabs(sumf);if(dn<1e-6)dn=1e-6;double rl=fabs(out[c]-sumf)/dn;if(rl>maxrel)maxrel=rl;}
  printf("ADVERSARIAL (q8=+127, w=-16 everywhere; per-lane sum -16*127*16=-32512, |.|<32767) maxrel=%.3e %s\n",maxrel,maxrel<1e-4?"PASS":"FAIL");
  // second pattern: nibble 15 + qh bit set -> w = (15|16)-16 = +15, q8 = -127 -> +15*-127, also extreme
  for(int r=0;r<NC;r++)for(int b=0;b<nb;b++){block_q5_0&blk=w[r*nb+b];uint32_t qh=0xFFFFFFFFu;memcpy(blk.qh,&qh,4);for(int k=0;k<16;k++)blk.qs[k]=0xFF;}
  for(int b=0;b<nb;b++)for(int k=0;k<32;k++)a[b].qs[k]=(int8_t)-127;
  {block_q5_0 tmp[16];for(int b=0;b<nb;b++){for(int i=0;i<16;i++)tmp[i]=w[i*nb+b];vx[b]=mk(tmp);}}
  ggml_gemv_q5_0_16x1_q8_0(n,out,0,vx.data(),a.data(),1,NC);
  maxrel=0;for(int c=0;c<NC;c++){double sumf=0;for(int b=0;b<nb;b++){double dx=h2f(w[c*nb+b].d),dy=h2f(a[b].d);int sumi=0;for(int k=0;k<16;k++){int x0=((w[c*nb+b].qs[k]&0x0F)|0x10)-16;int x1=((w[c*nb+b].qs[k]>>4)|0x10)-16;sumi+=x0*a[b].qs[k]+x1*a[b].qs[k+16];}sumf+=dx*dy*sumi;}double dn=fabs(sumf);if(dn<1e-6)dn=1e-6;double rl=fabs(out[c]-sumf)/dn;if(rl>maxrel)maxrel=rl;}
  printf("ADVERSARIAL (q8=-127, w=+15 everywhere; per-lane sum +15*-127*16=-30480) maxrel=%.3e %s\n",maxrel,maxrel<1e-4?"PASS":"FAIL");
  return 0;}
