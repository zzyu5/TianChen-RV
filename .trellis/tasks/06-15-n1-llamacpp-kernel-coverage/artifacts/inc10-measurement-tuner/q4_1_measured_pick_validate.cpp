#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define Q4S 20
#define Q8S 36
#define QK 32
extern "C" void kern_ggml(size_t,float*,const uint8_t*,const uint8_t*);
extern "C" void tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1(size_t,float*,const uint8_t*,const uint8_t*);
static inline float f16(uint16_t h){_Float16 x;memcpy(&x,&h,2);return (float)x;}
static void generic_ref(size_t n,float*s,const uint8_t*vx,const uint8_t*vy){
  const int nb=(int)n/QK; float sumf=0;
  for(int ib=0;ib<nb;++ib){
    const uint8_t*xq=vx+ib*Q4S+4; const int8_t*yq=(const int8_t*)(vy+ib*Q8S+4);
    uint16_t xd,xm,yd,ys; memcpy(&xd,vx+ib*Q4S,2);memcpy(&xm,vx+ib*Q4S+2,2);memcpy(&yd,vy+ib*Q8S,2);memcpy(&ys,vy+ib*Q8S+2,2);
    int sumi=0; for(int j=0;j<16;++j){int lo=xq[j]&0x0F,hi=xq[j]>>4;sumi+=lo*yq[j]+hi*yq[j+16];}
    sumf=sumf+((f16(xd)*f16(yd))*(float)sumi+f16(xm)*f16(ys));
  }
  *s=sumf;
}
static uint32_t rng=0x9911ee22u; static uint32_t xr(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void pf(uint8_t*p){_Float16 d=(_Float16)(((float)(int)(xr()%2001)-1000.0f)/256.0f);uint16_t h;memcpy(&h,&d,2);memcpy(p,&h,2);}
static void fq4(uint8_t*b,int nb){for(int i=0;i<nb;i++){uint8_t*p=b+i*Q4S;pf(p);pf(p+2);for(int j=0;j<QK/2;j++)p[4+j]=(uint8_t)(xr()%256);}}
static void fq8(uint8_t*b,int nb){for(int i=0;i<nb;i++){uint8_t*p=b+i*Q8S;pf(p);pf(p+2);for(int j=0;j<QK;j++)p[4+j]=(uint8_t)(int8_t)(xr()%256);}}
static int beq(float a,float b){uint32_t x,y;memcpy(&x,&a,4);memcpy(&y,&b,4);return x==y;}
int main(){
  int ns[]={32,64,128,256,512,1024,2048,4096,8192}; int tot=0,fg=0,fc=0;
  for(int t=0;t<400;++t)for(int k=0;k<9;++k){int n=ns[k],nb=n/QK;
    uint8_t*vx=(uint8_t*)malloc((size_t)nb*Q4S),*vy=(uint8_t*)malloc((size_t)nb*Q8S);fq4(vx,nb);fq8(vy,nb);
    float g=0,ge=0,m=0; kern_ggml(n,&g,vx,vy); generic_ref(n,&ge,vx,vy);
    tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1(n,&m,vx,vy);
    if(!beq(m,g)){if(fg<3)printf("FAIL ggml n=%d %.9g!=%.9g\n",n,m,g);fg++;}
    if(!beq(m,ge)){if(fc<3)printf("FAIL gen n=%d %.9g!=%.9g\n",n,m,ge);fc++;}
    tot++;free(vx);free(vy);}
  printf("q4_1 MEASURED-PICK (m1/1/elided) byte-exact: %d cases, vs ggml-real %d fail, vs _generic %d fail\n",tot,fg,fc);
  return (fg||fc)?1:0;
}
