#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
/* OURS f16lut kernel (emitted) */
#include "/tmp/gelu_f16lut_host.c"
/* OPP: verbatim ggml f16-LUT from opponent_ggml.cpp */
static inline uint16_t o_f32_to_f16(float f){ uint32_t x; memcpy(&x,&f,4);
 uint32_t sign=(x>>16)&0x8000u; int32_t exp=(int32_t)((x>>23)&0xff)-127+15; uint32_t man=x&0x7fffffu;
 if(exp<=0){ if(exp<-10) return (uint16_t)sign; man|=0x800000u; uint32_t sh=(uint32_t)(14-exp);
   uint16_t r=(uint16_t)(man>>sh); if((man>>(sh-1))&1) r++; return (uint16_t)(sign|r); }
 if(exp>=31) return (uint16_t)(sign|0x7c00u);
 uint16_t r=(uint16_t)(sign|((uint32_t)exp<<10)|(man>>13)); if((man>>12)&1) r++; return r; }
static inline float o_f16_to_f32(uint16_t h){ uint32_t sign=(uint32_t)(h&0x8000)<<16; uint32_t exp=(h>>10)&0x1f; uint32_t man=h&0x3ff; uint32_t o;
 if(exp==0){ if(man==0){o=sign;} else { exp=127-15+1; while(!(man&0x400)){man<<=1;exp--;} man&=0x3ff; o=sign|(exp<<23)|(man<<13);} }
 else if(exp==31){ o=sign|0x7f800000u|(man<<13); } else { o=sign|((exp+112)<<23)|(man<<13); }
 float f; memcpy(&f,&o,4); return f; }
static uint16_t tab[1<<16];
static inline float ggml_gelu_f32(float x){ return 0.5f*x*(1.0f+tanhf(0.79788456080286535587989211986876f*x*(1.0f+0.044715f*x*x))); }
static void opp_init(void){ for(int i=0;i<(1<<16);++i){ float f=o_f16_to_f32((uint16_t)i); tab[i]=o_f32_to_f16(ggml_gelu_f32(f)); } }
static void opp_gelu(int n, float* y, const float* x){ uint16_t t; for(int i=0;i<n;++i){ if(x[i]<=-10.0f)y[i]=0.0f; else if(x[i]>=10.0f)y[i]=x[i]; else { uint16_t fp16=o_f32_to_f16(x[i]); memcpy(&t,&fp16,sizeof(uint16_t)); y[i]=o_f16_to_f32(tab[t]); } } }
/* same rng as driver */
static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static float rf(void){ return ((float)(xr()&0xffffff)/(float)0xffffff)*2.0f-1.0f; }
static uint32_t ulp(float a,float b){ if(a==b)return 0; uint32_t ua,ub; memcpy(&ua,&a,4); memcpy(&ub,&b,4); if((ua>>31)!=(ub>>31))return 0xffffffu; return ua>ub?ua-ub:ub-ua; }
int main(){ opp_init();
  int Ns[]={512,1024,2048,3072,4096,5120,8192,16384};
  uint32_t g_maxulp_ours_vs_opp=0; double g_ours_rel=0,g_opp_rel=0; uint32_t g_ours_vs_lutoracle=0;
  for(int k=0;k<8;k++){ int n=Ns[k]; rng=(uint64_t)(12345+k)|1ull;
    float*x=(float*)malloc(n*4),*yo=(float*)malloc(n*4),*yp=(float*)malloc(n*4);
    for(int i=0;i<n;i++)x[i]=rf();
    weft_emitc_gelu_f32_kernel_gelu_f32((size_t)n,x,yo);   /* OURS */
    opp_gelu(n,yp,x);                                       /* OPP  */
    uint32_t mu=0; double orel=0,prel=0;
    for(int i=0;i<n;i++){ uint32_t u=ulp(yo[i],yp[i]); if(u>mu)mu=u;
      double ox=x[i]; double oref=0.5*ox*(1.0+tanh(0.7978845608028654*ox*(1.0+0.044715*ox*ox)));
      double den=fabs(oref)>1e-6?fabs(oref):1e-6;
      double ro=fabs((double)yo[i]-oref)/den, rp=fabs((double)yp[i]-oref)/den;
      if(ro>orel)orel=ro; if(rp>prel)prel=rp; }
    if(mu>g_maxulp_ours_vs_opp)g_maxulp_ours_vs_opp=mu;
    if(orel>g_ours_rel)g_ours_rel=orel; if(prel>g_opp_rel)g_opp_rel=prel;
    printf("n=%-6d ours_vs_opp_maxulp=%u  ours_rel_vs_fp64tanh=%.3e  opp_rel_vs_fp64tanh=%.3e\n",n,mu,orel,prel);
    free(x);free(yo);free(yp);
  }
  printf("\nSUMMARY ours_vs_opp_maxulp=%u (0=BYTE-EXACT same-tier)  ours_rel=%.3e  opp_rel=%.3e\n",
         g_maxulp_ours_vs_opp,g_ours_rel,g_opp_rel);
  return g_maxulp_ours_vs_opp==0?0:1; }
