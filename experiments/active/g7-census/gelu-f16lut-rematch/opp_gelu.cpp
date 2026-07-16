/* opp_gelu.cpp — VERBATIM ggml as-shipped GGML_GELU_FP16 f16-LUT opponent, extracted
 * byte-identical from experiments/active/g7-census/bclass-forward-ops/opponent_ggml.cpp
 * (vec.h:46/968/988). Same-precision-tier opponent for the G.0.3 gelu rematch. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
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
static uint16_t opp_gelu_table[1<<16];
static inline float ggml_gelu_f32(float x){ return 0.5f*x*(1.0f + tanhf(0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x))); }
extern "C" void opp_gelu_init(void){ for(int i=0;i<(1<<16);++i){ float f=o_f16_to_f32((uint16_t)i); opp_gelu_table[i]=o_f32_to_f16(ggml_gelu_f32(f)); } }
extern "C" void opp_ggml_vec_gelu_f32(int n, float* y, const float* x){ uint16_t t;
    for(int i=0;i<n;++i){ if(x[i] <= -10.0f){ y[i]=0.0f; } else if(x[i] >= 10.0f){ y[i]=x[i]; }
        else { uint16_t fp16=o_f32_to_f16(x[i]); memcpy(&t,&fp16,sizeof(uint16_t)); y[i]=o_f16_to_f32(opp_gelu_table[t]); } } }
