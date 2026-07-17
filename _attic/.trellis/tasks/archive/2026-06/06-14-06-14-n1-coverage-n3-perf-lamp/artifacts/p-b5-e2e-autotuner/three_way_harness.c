#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
extern void ref_scalar_byte(const int8_t*,const int8_t*,const int32_t*,float,float*,size_t);
extern void naive_rvv_byte(const int8_t*,const int8_t*,const int32_t*,float,float*,size_t);
extern void tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t*,const int8_t*,const int32_t*,float,float*,size_t);
typedef void(*kfn)(const int8_t*,const int8_t*,const int32_t*,float,float*,size_t);
static uint32_t rs=0x12345678u; static int8_t r8(void){rs=rs*1664525u+1013904223u;return (int8_t)((rs>>17)&0xFF);}
static double ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec*1e9+t.tv_nsec;}
#define WARM 3
#define REP 9
#define IT 16
int main(void){
 const size_t SZ[]={257,4096,65536}; const char*NM[]={"genuine-scalar","naive-rvv","tuned-auto"};
 kfn FNS[]={ref_scalar_byte,naive_rvv_byte,tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize}; float scale=0.0125f; double sink=0;
 for(int s=0;s<3;++s){size_t n=SZ[s]; int8_t*l=malloc(n),*r=malloc(n); int32_t acc[1]={-7}; float out=0;
  for(size_t i=0;i<n;++i){l[i]=r8();r[i]=r8();}
  // correctness vs scalar oracle
  float oref; ref_scalar_byte(l,r,acc,scale,&oref,n);
  for(int v=0;v<3;++v){FNS[v](l,r,acc,scale,&out,n); if(fabsf(out-oref)>1e-5f*(1+fabsf(oref))){printf("CORRECTNESS n=%zu variant=%s FAIL out=%f ref=%f\n",n,NM[v],out,oref);return 1;}}
  printf("CORRECTNESS n=%zu ok oracle=%g\n",n,oref);
  for(int v=0;v<3;++v){double best=1e30;
   for(int w=0;w<WARM;++w){for(int it=0;it<IT;++it){FNS[v](l,r,acc,scale,&out,n);sink+=out;}}
   for(int rp=0;rp<REP;++rp){double t0=ns();for(int it=0;it<IT;++it){FNS[v](l,r,acc,scale,&out,n);sink+=out;}double per=(ns()-t0)/IT; if(per<best)best=per;}
   printf("SUMMARY variant=%s n=%zu best_per_iter_ns=%.3f\n",NM[v],n,best);}
  free(l);free(r);}
 printf("SINK=%g\n",sink); return 0;}
