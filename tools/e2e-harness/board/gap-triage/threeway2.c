#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define QK_K 256
typedef uint16_t ggml_half;
typedef struct { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; }  block_iq4_xs;
typedef struct { uint8_t qs[(QK_K-4*QK_K/64)/5]; uint8_t qh[QK_K/64]; ggml_half d; }               block_tq1_0;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; }                                block_q8_K;
extern void tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void ggml_vec_dot_iq4_xs_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_iq4_xs_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_tq1_0_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_tq1_0_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
static uint64_t st;
static void seed(uint64_t s){st=s;}
static uint8_t rb(){ st=st*6364136223846793005ull+1442695040888963407ull; return (uint8_t)(st>>33); }
static int ulp(float a,float b){ int32_t ia,ib; memcpy(&ia,&a,4); memcpy(&ib,&b,4); if(ia<0)ia=0x7fffffff-ia; if(ib<0)ib=0x7fffffff-ib; long d=(long)ia-(long)ib; return d<0?(int)-d:(int)d; }
static void fill_q8(block_q8_K*q,int consistent,int fixd){ q->d = fixd?1.0f:( ( (float)(int8_t)rb() )/8.0f + 1.0f ); for(int i=0;i<QK_K;i++) q->qs[i]=(int8_t)rb();
  if(consistent){ for(int g=0;g<QK_K/16;g++){int s=0;for(int j=0;j<16;j++)s+=q->qs[g*16+j];q->bsums[g]=(int16_t)s;} } else { for(int g=0;g<QK_K/16;g++)q->bsums[g]=(int16_t)(rb()|(rb()<<8)); } }
static void run(const char*name,int consistent,int fixd,uint64_t s){
  block_q8_K y; seed(s);
  block_iq4_xs x4; uint8_t*p=(uint8_t*)&x4; for(unsigned i=0;i<sizeof x4;i++)p[i]=rb(); if(fixd)x4.d=0x3C00; fill_q8(&y,consistent,fixd);
  float o=0,f=0,g=0;
  tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot(QK_K,&o,(const uint8_t*)&x4,(const uint8_t*)&y);
  ggml_vec_dot_iq4_xs_q8_K(QK_K,&f,0,&x4,0,&y,0,1); ggml_vec_dot_iq4_xs_q8_K_generic(QK_K,&g,0,&x4,0,&y,0,1);
  printf("  [%s] iq4_xs ULP ours-gen=%d fac-gen=%d ours-fac=%d  (o=%.7g f=%.7g g=%.7g)\n",name,ulp(o,g),ulp(f,g),ulp(o,f),o,f,g);
  block_tq1_0 xt; p=(uint8_t*)&xt; for(unsigned i=0;i<sizeof xt;i++)p[i]=rb(); if(fixd)xt.d=0x3C00; fill_q8(&y,consistent,fixd);
  o=f=g=0;
  tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(QK_K,&o,(const uint8_t*)&xt,(const uint8_t*)&y);
  ggml_vec_dot_tq1_0_q8_K(QK_K,&f,0,&xt,0,&y,0,1); ggml_vec_dot_tq1_0_q8_K_generic(QK_K,&g,0,&xt,0,&y,0,1);
  printf("  [%s] tq1_0  ULP ours-gen=%d fac-gen=%d ours-fac=%d  (o=%.7g f=%.7g g=%.7g)\n",name,ulp(o,g),ulp(f,g),ulp(o,f),o,f,g);
}
int main(){
  printf("Regime A: fixed d=1.0 + consistent bsums (well-formed)\n"); run("A",1,1,0x1234567);
  printf("Regime B: random finite d + consistent bsums (isolate scale-driven fold-order)\n"); run("B",1,0,0x1234567);
  printf("Regime C: random d + INCONSISTENT random bsums (batch2c-style adversarial)\n"); run("C",0,0,0x1234567);
  printf("Regime C2 (diff seed): \n"); run("C2",0,0,0xDEADBEEF);
  return 0;
}
