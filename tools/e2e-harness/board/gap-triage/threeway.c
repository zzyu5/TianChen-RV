#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define QK_K 256
typedef uint16_t ggml_half;
typedef struct { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; }  block_iq4_xs;   /*136*/
typedef struct { uint8_t qs[(QK_K-4*QK_K/64)/5]; uint8_t qh[QK_K/64]; ggml_half d; }               block_tq1_0;    /*54*/
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; }                                block_q8_K;     /*292*/

extern void tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(size_t,float*,const uint8_t*,const uint8_t*);
extern void ggml_vec_dot_iq4_xs_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_iq4_xs_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_tq1_0_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern void ggml_vec_dot_tq1_0_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);

static uint64_t st=0x1234567ull;
static uint8_t rb(){ st=st*6364136223846793005ull+1442695040888963407ull; return (uint8_t)(st>>33); }
static int ulp(float a,float b){ int32_t ia,ib; memcpy(&ia,&a,4); memcpy(&ib,&b,4); if(ia<0)ia=0x7fffffff-ia; if(ib<0)ib=0x7fffffff-ib; long d=(long)ia-(long)ib; return d<0?(int)-d:(int)d; }

static void fill_q8(block_q8_K*q){ q->d=1.0f; for(int i=0;i<QK_K;i++) q->qs[i]=(int8_t)rb(); for(int g=0;g<QK_K/16;g++){int s=0;for(int j=0;j<16;j++)s+=q->qs[g*16+j];q->bsums[g]=(int16_t)s;} }

int main(){
  block_q8_K y; 
  /* ---- iq4_xs ---- */
  { block_iq4_xs x; uint8_t*p=(uint8_t*)&x; for(unsigned i=0;i<sizeof x;i++)p[i]=rb(); x.d=0x3C00; /*fp16 1.0*/ fill_q8(&y);
    float o=0,f=0,g=0;
    tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot(QK_K,&o,(const uint8_t*)&x,(const uint8_t*)&y);
    ggml_vec_dot_iq4_xs_q8_K(QK_K,&f,0,&x,0,&y,0,1);
    ggml_vec_dot_iq4_xs_q8_K_generic(QK_K,&g,0,&x,0,&y,0,1);
    printf("iq4_xs   ours=% .9g  factory=% .9g  generic=% .9g | ulp(ours,gen)=%d ulp(fac,gen)=%d ulp(ours,fac)=%d\n",o,f,g,ulp(o,g),ulp(f,g),ulp(o,f));
  }
  /* ---- tq1_0 ---- */
  { block_tq1_0 x; uint8_t*p=(uint8_t*)&x; for(unsigned i=0;i<sizeof x;i++)p[i]=rb(); x.d=0x3C00; fill_q8(&y);
    float o=0,f=0,g=0;
    tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(QK_K,&o,(const uint8_t*)&x,(const uint8_t*)&y);
    ggml_vec_dot_tq1_0_q8_K(QK_K,&f,0,&x,0,&y,0,1);
    ggml_vec_dot_tq1_0_q8_K_generic(QK_K,&g,0,&x,0,&y,0,1);
    printf("tq1_0    ours=% .9g  factory=% .9g  generic=% .9g | ulp(ours,gen)=%d ulp(fac,gen)=%d ulp(ours,fac)=%d\n",o,f,g,ulp(o,g),ulp(f,g),ulp(o,f));
  }
  return 0;
}
