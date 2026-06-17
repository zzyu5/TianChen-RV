#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <riscv_vector.h>
void tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t* v1, const int8_t* v2, const int32_t* v3, float v4, float* v5, size_t v6) {
  size_t v7 = __riscv_vsetvl_e8m2(v6);
  vint32m8_t v8;
  size_t v9 = __riscv_vsetvlmax_e32m8();
  vint32m8_t v10 = __riscv_vmv_v_x_i32m8(0, v9);
  v8 = v10;
  for (size_t v11 = 0; v11 < v6; v11 += v7) {
    size_t v12 = v6 - v11;
    size_t v13 = __riscv_vsetvl_e8m2(v12);
    const int8_t* v14 = v1 + v11;
    vint8m2_t v15 = __riscv_vle8_v_i8m2(v14, v13);
    const int8_t* v16 = v2 + v11;
    vint8m2_t v17 = __riscv_vle8_v_i8m2(v16, v13);
    vint16m4_t v18 = __riscv_vwmul_vv_i16m4(v15, v17, v13);
    vint32m8_t v19 = v8;
    vint32m8_t v20 = __riscv_vwadd_wv_i32m8(v19, v18, v13);
    v8 = v20;
  }
  size_t v21 = __riscv_vsetvlmax_e32m1();
  vint32m1_t v22 = __riscv_vmv_v_x_i32m1(0, v21);
  vint32m8_t v23 = v8;
  vint32m1_t v24 = __riscv_vredsum_vs_i32m8_i32m1(v23, v22, v9);
  int32_t v25 = __riscv_vmv_x_s_i32m1_i32(v24);
  const int32_t v26 = v3[0];
  int32_t v27 = v26 + v25;
  float v28 = (float) v27;
  float v29 = v28 * v4;
  vfloat32m1_t v30 = __riscv_vfmv_v_f_f32m1(v29, 1);
  __riscv_vse32_v_f32m1(v5, v30, 1);
  return;
}
static float scalar_reference(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, size_t n){ int32_t sum=acc[0]; for(size_t i=0;i<n;++i) sum += (int32_t)lhs[i]*(int32_t)rhs[i]; return ((float)sum)*scale; }
static uint32_t rs=0x12345678u; static int8_t r8(void){ rs=rs*1664525u+1013904223u; return (int8_t)((rs>>17)&0xFF);}
int main(void){ const size_t sz[]={257,256,1024,4096,16384,65536}; int ap=1;
 for(size_t s=0;s<6;++s){ size_t n=sz[s]; int8_t*l=malloc(n),*r=malloc(n); int32_t acc[1]={-4096}; float scale=0.013725f, out=0.0f;
  for(size_t i=0;i<n;++i){l[i]=r8();r[i]=r8();}
  tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(l,r,acc,scale,&out,n);
  float ref=scalar_reference(l,r,acc,scale,n); float tol=1e-5f*(1.0f+fabsf(ref)); int p=fabsf(out-ref)<=tol;
  printf("n=%-6zu emitted=%.6f reference=%.6f abs_err=%.3e %s\n",n,out,ref,fabsf(out-ref),p?"PASS":"FAIL"); if(!p)ap=0; free(l);free(r);}
 printf("RESULT: %s\n", ap?"ALL_PASS":"FAIL"); return ap?0:1; }
