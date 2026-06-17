// INC-9 q4_1 FULL SHAPE LADDER microbench (ssh rvv) -- the discriminating
// measurement the 3-point bench could not give. Times the WHOLE m1 grid
// {mb1,mb2,mb4} x {robust,elided} + ggml's real kernel, so the mb4<mb2 inversion
// (opposite of q4_0) is explained: which shape is actually fastest, and whether
// ANY shape beats ggml on the scale+MIN family. Same rig as microbench.cpp.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 32
#define Q4_STRIDE 20
#define Q8_STRIDE 36
static inline float f16(uint16_t h){_Float16 x;memcpy(&x,&h,2);return (float)x;}
typedef void (*kfn)(size_t,float*,const uint8_t*,const uint8_t*);
extern "C" {
void kern_ggml(size_t,float*,const uint8_t*,const uint8_t*);
void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb1r(size_t,float*,const uint8_t*,const uint8_t*);
void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb2r(size_t,float*,const uint8_t*,const uint8_t*);
void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb4r(size_t,float*,const uint8_t*,const uint8_t*);
void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb1e(size_t,float*,const uint8_t*,const uint8_t*);
void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb2e(size_t,float*,const uint8_t*,const uint8_t*);
void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb4e(size_t,float*,const uint8_t*,const uint8_t*);
}
static void ggml_real(size_t n,float*s,const uint8_t*vx,const uint8_t*vy){
  const int nb=(int)n/QK; float sumf=0; size_t vl=QK/2;
  for(int ib=0;ib<nb;++ib){
    const uint8_t*xq=vx+ib*Q4_STRIDE+4; const int8_t*yq=(const int8_t*)(vy+ib*Q8_STRIDE+4);
    uint16_t xd,xm,yd,ys; memcpy(&xd,vx+ib*Q4_STRIDE,2); memcpy(&xm,vx+ib*Q4_STRIDE+2,2);
    memcpy(&yd,vy+ib*Q8_STRIDE,2); memcpy(&ys,vy+ib*Q8_STRIDE+2,2);
    vuint8m1_t tx=__riscv_vle8_v_u8m1(xq,vl);
    vint8m1_t y0=__riscv_vle8_v_i8m1(yq,vl), y1=__riscv_vle8_v_i8m1(yq+16,vl);
    vuint8m1_t a=__riscv_vand_vx_u8m1(tx,0x0F,vl), l=__riscv_vsrl_vx_u8m1(tx,0x04,vl);
    vint8m1_t v0=__riscv_vreinterpret_v_u8m1_i8m1(a), v1=__riscv_vreinterpret_v_u8m1_i8m1(l);
    vint16m2_t m1=__riscv_vwmul_vv_i16m2(v0,y0,vl), m2=__riscv_vwmacc_vv_i16m2(m1,v1,y1,vl);
    vint32m1_t z=__riscv_vmv_v_x_i32m1(0,vl), r=__riscv_vwredsum_vs_i16m2_i32m1(m2,z,vl);
    int sumi=__riscv_vmv_x_s_i32m1_i32(r);
    sumf+=(f16(xd)*f16(yd))*sumi+f16(xm)*f16(ys);
  }
  *s=sumf;
}
static uint32_t rng=0x2468ace0u; static uint32_t xr(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void pf(uint8_t*p){_Float16 d=(_Float16)(((float)(int)(xr()%2001)-1000.0f)/256.0f);uint16_t h;memcpy(&h,&d,2);memcpy(p,&h,2);}
static void fq4(uint8_t*b,int nb){for(int i=0;i<nb;i++){uint8_t*p=b+i*Q4_STRIDE;pf(p);pf(p+2);for(int j=0;j<QK/2;j++)p[4+j]=(uint8_t)(xr()%256);}}
static void fq8(uint8_t*b,int nb){for(int i=0;i<nb;i++){uint8_t*p=b+i*Q8_STRIDE;pf(p);pf(p+2);for(int j=0;j<QK;j++)p[4+j]=(uint8_t)(int8_t)(xr()%256);}}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
struct E{const char*name;kfn fn;double best;};
static double tb(kfn fn,int n,int it,const uint8_t*vx,const uint8_t*vy){double t0=now();float s;for(int i=0;i<it;i++)fn(n,&s,vx,vy);return (now()-t0)/it;}
int main(){
  const int n=4096,nb=n/QK,iters=2000,reps=200;
  uint8_t*vx=(uint8_t*)malloc((size_t)nb*Q4_STRIDE),*vy=(uint8_t*)malloc((size_t)nb*Q8_STRIDE);
  fq4(vx,nb); fq8(vy,nb);
  E ks[]={
    {"ggml(real)",kern_ggml,1e18},
    {"mb1_robust",tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb1r,1e18},
    {"mb2_robust",tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb2r,1e18},
    {"mb4_robust",tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb4r,1e18},
    {"mb1_elided",tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb1e,1e18},
    {"mb2_elided",tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb2e,1e18},
    {"mb4_elided(FULLV-PICK)",tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1_mb4e,1e18},
  };
  const int K=sizeof(ks)/sizeof(ks[0]);
  float ref=0; ggml_real(n,&ref,vx,vy);
  for(E&e:ks){float g=0;e.fn(n,&g,vx,vy);if(memcmp(&g,&ref,4)!=0){printf("GATE FAIL: %s %.9g != %.9g\n",e.name,g,ref);return 1;}}
  printf("correctness gate: all 6 shapes bit-exact vs ggml\n");
  for(E&e:ks)for(int i=0;i<iters;i++){float s;e.fn(n,&s,vx,vy);}
  for(int r=0;r<reps;r++)for(int k=0;k<K;k++){double p=tb(ks[k].fn,n,iters,vx,vy);if(p<ks[k].best)ks[k].best=p;}
  double g=ks[0].best;
  printf("%-26s %10s   %s\n","shape","best-ns","ggml/best (>1 => faster than ggml)");
  for(E&e:ks)printf("%-26s %10.1f   %.3fx\n",e.name,e.best,g/e.best);
  free(vx);free(vy);
  return 0;
}
