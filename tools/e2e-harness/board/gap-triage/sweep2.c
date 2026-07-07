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
static uint64_t st=0x1234567ull;
static uint8_t rb(){ st=st*6364136223846793005ull+1442695040888963407ull; return (uint8_t)(st>>33); }
static int ulp(float a,float b){ int32_t ia,ib; memcpy(&ia,&a,4); memcpy(&ib,&b,4); if(ia<0)ia=0x7fffffff-ia; if(ib<0)ib=0x7fffffff-ib; long d=(long)ia-(long)ib; return d<0?(int)-d:(int)d; }
int main(){
  int N=50000;
  block_q8_K y;
  int mog4=0,mfg4=0,cog4=0,cfg4=0, mog1=0,mfg1=0,cog1=0,cfg1=0;
  for(int t=0;t<N;t++){
    y.d=0.5f+((float)(rb())/255.0f)*3.0f; for(int i=0;i<QK_K;i++)y.qs[i]=(int8_t)rb(); for(int g=0;g<QK_K/16;g++){int s=0;for(int j=0;j<16;j++)s+=y.qs[g*16+j];y.bsums[g]=(int16_t)s;}
    block_iq4_xs x4; uint8_t*p=(uint8_t*)&x4; for(unsigned i=0;i<sizeof x4;i++)p[i]=rb(); x4.d=(uint16_t)(0x3400+(rb()&0x0700));
    float o=0,f=0,g=0;
    tcrv_emitc_ggml_vec_dot_iq4_xs_q8_K_kernel_rvv_iq4_xs_q8_K_block_dot(QK_K,&o,(const uint8_t*)&x4,(const uint8_t*)&y);
    ggml_vec_dot_iq4_xs_q8_K(QK_K,&f,0,&x4,0,&y,0,1); ggml_vec_dot_iq4_xs_q8_K_generic(QK_K,&g,0,&x4,0,&y,0,1);
    int a=ulp(o,g),b=ulp(f,g); if(a>mog4)mog4=a; if(b>mfg4)mfg4=b; if(a)cog4++; if(b)cfg4++;
    block_tq1_0 xt; p=(uint8_t*)&xt; for(unsigned i=0;i<sizeof xt;i++)p[i]=rb(); xt.d=(uint16_t)(0x3400+(rb()&0x0700));
    o=f=g=0;
    tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_tq1_0_q8_K_block_dot(QK_K,&o,(const uint8_t*)&xt,(const uint8_t*)&y);
    ggml_vec_dot_tq1_0_q8_K(QK_K,&f,0,&xt,0,&y,0,1); ggml_vec_dot_tq1_0_q8_K_generic(QK_K,&g,0,&xt,0,&y,0,1);
    a=ulp(o,g);b=ulp(f,g); if(a>mog1)mog1=a; if(b>mfg1)mfg1=b; if(a)cog1++; if(b)cfg1++;
  }
  printf("N=%d blocks (fp16 d=1.0, q8_K.d=1.0, consistent bsums, random payloads)\n",N);
  printf("iq4_xs: OURS-vs-oracle  maxULP=%d  mismatchBlocks=%d/%d   FACTORY-vs-oracle maxULP=%d mismatchBlocks=%d/%d\n",mog4,cog4,N,mfg4,cfg4,N);
  printf("tq1_0 : OURS-vs-oracle  maxULP=%d  mismatchBlocks=%d/%d   FACTORY-vs-oracle maxULP=%d mismatchBlocks=%d/%d\n",mog1,cog1,N,mfg1,cfg1,N);
  return 0;
}
