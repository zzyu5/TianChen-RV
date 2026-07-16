#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include <math.h>
/* G.0.3 f16-LUT gelu seam: byte-exact + same-algorithm to ggml as-shipped GGML_GELU_FP16 (vec.h:46/968/988). Table gather, no runtime tanhf. Call weft_gelu_f16lut_init() once before use (mirrors ggml_init's ggml_table_gelu_f16 build). */
static unsigned short weft_gelu_f16_table[1<<16];
static int weft_gelu_f16_ready=0;
static inline unsigned short weft_f32_to_f16(float f){
  unsigned int x; __builtin_memcpy(&x,&f,4);
  unsigned int sign=(x>>16)&0x8000u; int e=(int)((x>>23)&0xff)-127+15; unsigned int man=x&0x7fffffu;
  if(e<=0){ if(e<-10) return (unsigned short)sign; man|=0x800000u; unsigned int sh=(unsigned int)(14-e);
    unsigned short r=(unsigned short)(man>>sh); if((man>>(sh-1))&1) r++; return (unsigned short)(sign|r); }
  if(e>=31) return (unsigned short)(sign|0x7c00u);
  unsigned short r=(unsigned short)(sign|((unsigned int)e<<10)|(man>>13)); if((man>>12)&1) r++; return r;
}
static inline float weft_f16_to_f32(unsigned short h){
  unsigned int sign=(unsigned int)(h&0x8000)<<16; unsigned int e=(h>>10)&0x1f; unsigned int man=h&0x3ff; unsigned int o;
  if(e==0){ if(man==0){o=sign;} else { e=127-15+1; while(!(man&0x400)){man<<=1;e--;} man&=0x3ff; o=sign|(e<<23)|(man<<13);} }
  else if(e==31){ o=sign|0x7f800000u|(man<<13); }
  else { o=sign|((e+112)<<23)|(man<<13); }
  float f; __builtin_memcpy(&f,&o,4); return f;
}
static inline float weft_ggml_gelu_f32(float x){ return 0.5f*x*(1.0f + tanhf(0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x))); }
extern "C" void weft_gelu_f16lut_init(void){
  for(int i=0;i<(1<<16);++i){ float f=weft_f16_to_f32((unsigned short)i); weft_gelu_f16_table[i]=weft_f32_to_f16(weft_ggml_gelu_f32(f)); }
  weft_gelu_f16_ready=1;
}
static inline float weft_gelu_f16lut_scalar(float x){
  if(x <= -10.0f) return 0.0f;
  if(x >=  10.0f) return x;
  unsigned short t = weft_f32_to_f16(x);
  return weft_f16_to_f32(weft_gelu_f16_table[t]);
}
extern "C" void weft_emitc_gelu_f32_kernel_gelu_f32(size_t v1, const float* v2, float* v3) {
  // weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
  // weft_emitc.source_op=weft_rvv.setvl role=configure op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // weft_emitc.route_source_op=weft_rvv.elementwise_gelu_map role=compute op_interface=WEFTEmitCLowerableOpInterface
  for (size_t v5 = 0; v5 < v1; v5 += 1) {
    const float* v6 = v2 + v5;
    const float* v7 = (const float*) v6;
    const float v8 = v7[0];
    // weft_emitc.source_op=weft_rvv.elementwise_gelu_map role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weft_gelu_f16lut_scalar
    float v9 = weft_gelu_f16lut_scalar(v8);
    float* v10 = v3 + v5;
    float* v11 = (float*) v10;
    v11[0] = v9;
  }
  return;
}


