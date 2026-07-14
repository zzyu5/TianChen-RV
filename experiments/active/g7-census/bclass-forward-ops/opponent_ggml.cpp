/* opponent_ggml.cpp — G7 L1 B-class forward-op census: ggml AS-SHIPPED opponent kernels.
 *
 * These are VERBATIM transcriptions of ggml-cpu source (ggml-org/llama.cpp @ /home/kingdom/phdworks/llama.cpp),
 * wrapped extern "C" for symmetric A/B linking. Source provenance is cited per function.
 * Compiled with the SAME clang-18 + march as our emitted kernels ([CASE-COMPILER-ASYMMETRY] not
 * triggered: both clang-18 = k1 shipped compiler). The RVV branches (__riscv_v_intrinsic) fire under
 * -march=rv64gcv*, reproducing the as-shipped code path.
 *
 * silu / soft_max are NOT here — they are EXPORTED symbols (nm -D: T ggml_vec_silu_f32,
 * T ggml_vec_soft_max_f32) linked directly from the stock libggml-cpu.so (genuine shipped binary).
 *
 * Opponents transcribed here (inline-in-ggml, not exported => must compile from source):
 *   add   ggml_vec_add_f32    vec.h:89   (AVX2-only vec path; RV falls to scalar loop => autovec)
 *   mul   ggml_vec_mul_f32    vec.h:127  (scalar loop => autovec)
 *   cpy   ggml_vec_cpy_f32    vec.h:119  (scalar loop => autovec/memcpy)
 *   scale ggml_vec_scale_f32  vec.h:703  (NATIVE RVV m8 vfmul_vf — byte-identical algo to ours)
 *   gelu  ggml_vec_gelu_f32   vec.h:988  (GGML_GELU_FP16 defined @ vec.h:46 => f16 LUT lookup)
 *   rms_norm  ggml_compute_forward_rms_norm_f32  ops.cpp:3758 (scalar dbl reduce + memcpy + vec_scale m8)
 *   rope      ggml_compute_forward_rope_f32 / ggml_rope_cache_init  ops.cpp:5707 (cos/sin cache 2-pass)
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>

/* ---- f16<->f32 (matches ggml simd-mappings scalar fallback; used only for gelu LUT build) ---- */
static inline uint16_t o_f32_to_f16(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000u; int32_t exp=(int32_t)((x>>23)&0xff)-127+15; uint32_t man=x&0x7fffffu;
    if(exp<=0){ if(exp<-10) return (uint16_t)sign; man|=0x800000u; uint32_t sh=(uint32_t)(14-exp);
        uint16_t r=(uint16_t)(man>>sh); if((man>>(sh-1))&1) r++; return (uint16_t)(sign|r); }
    if(exp>=31) return (uint16_t)(sign|0x7c00u);
    uint16_t r=(uint16_t)(sign|((uint32_t)exp<<10)|(man>>13)); if((man>>12)&1) r++; return r;
}
static inline float o_f16_to_f32(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000)<<16; uint32_t exp=(h>>10)&0x1f; uint32_t man=h&0x3ff; uint32_t o;
    if(exp==0){ if(man==0){o=sign;} else { exp=127-15+1; while(!(man&0x400)){man<<=1;exp--;} man&=0x3ff; o=sign|(exp<<23)|(man<<13);} }
    else if(exp==31){ o=sign|0x7f800000u|(man<<13); }
    else { o=sign|((exp+112)<<23)|(man<<13); }
    float f; memcpy(&f,&o,4); return f;
}

/* ================= add / mul / cpy (ggml scalar loops; AVX2 vec path dead on RV) ================= */
/* vec.h:89 ggml_vec_add_f32(n, z, x, y). On RISC-V the __AVX2__ block is not compiled => scalar. */
extern "C" void opp_ggml_vec_add_f32(int n, float* z, const float* x, const float* y){
    int i=0;
    for(; i<n; ++i){ z[i]=x[i]+y[i]; }
}
/* vec.h:127 ggml_vec_mul_f32(n, z, x, y) — pure scalar loop. */
extern "C" void opp_ggml_vec_mul_f32(int n, float* z, const float* x, const float* y){
    for(int i=0;i<n;++i){ z[i]=x[i]*y[i]; }
}
/* vec.h:119 ggml_vec_cpy_f32(n, y, x) — pure scalar loop. */
extern "C" void opp_ggml_vec_cpy_f32(int n, float* y, const float* x){
    for(int i=0;i<n;++i){ y[i]=x[i]; }
}

/* ================= scale (ggml NATIVE RVV m8 — vec.h:703 __riscv_v_intrinsic branch) ============= */
extern "C" void opp_ggml_vec_scale_f32(int n, float* y, const float v){
    for (int i = 0, avl; i < n; i += avl) {
        avl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
        vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(ay, v, avl);
        __riscv_vse32_v_f32m8(&y[i], ny, avl);
    }
}

/* ================= gelu (ggml f16 LUT — GGML_GELU_FP16 defined @ vec.h:46) ======================= */
/* ggml.c ggml_init builds ggml_table_gelu_f16[i] = f16( ggml_gelu_f32( f16_to_f32(i) ) ). */
static uint16_t opp_gelu_table[1<<16];
static int opp_gelu_table_ready=0;
static inline float ggml_gelu_f32(float x){
    /* vec.h:968 */
    return 0.5f*x*(1.0f + tanhf(0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x)));
}
extern "C" void opp_gelu_init(void){
    for(int i=0;i<(1<<16);++i){
        float f=o_f16_to_f32((uint16_t)i);
        opp_gelu_table[i]=o_f32_to_f16(ggml_gelu_f32(f));
    }
    opp_gelu_table_ready=1;
}
/* vec.h:988 ggml_vec_gelu_f32 (GGML_GELU_FP16 path) */
extern "C" void opp_ggml_vec_gelu_f32(int n, float* y, const float* x){
    uint16_t t;
    for(int i=0;i<n;++i){
        if(x[i] <= -10.0f){ y[i]=0.0f; }
        else if(x[i] >= 10.0f){ y[i]=x[i]; }
        else { uint16_t fp16=o_f32_to_f16(x[i]); memcpy(&t,&fp16,sizeof(uint16_t)); y[i]=o_f16_to_f32(opp_gelu_table[t]); }
    }
}

/* ================= rms_norm (ops.cpp:3758 non-fused: scalar dbl reduce + memcpy + vec_scale m8) === */
extern "C" void opp_ggml_rms_norm_f32(int n, const float* x, float* y, float eps){
    double sum=0.0;
    for(int i=0;i<n;++i){ sum += (double)(x[i]*x[i]); }
    const float mean  = (float)(sum/(double)n);
    const float scale = 1.0f/sqrtf(mean+eps);
    memcpy(y, x, (size_t)n*sizeof(float));
    /* ggml_vec_scale_f32 (native RVV m8) */
    for (int i = 0, avl; i < n; i += avl) {
        avl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
        vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(ay, scale, avl);
        __riscv_vse32_v_f32m8(&y[i], ny, avl);
    }
}

/* ================= rope (ops.cpp ggml_rope_cache_init 2-pass; non-neox/non-yarn, attn=1) ========= */
/* Faithful to as-shipped: ggml precomputes a cos/sin cache for the row, then applies (2 memory passes
 * over the cache). theta_i = theta0 * theta_scale^i. Interleaved-pair (rope_norm) rotate.            */
extern "C" void opp_ggml_rope_norm_f32(int n, const float* x, float* y, float theta0, float theta_scale, float* cache){
    float theta = theta0;
    int half = n/2;
    for(int i=0;i<half;++i){ cache[2*i]=cosf(theta); cache[2*i+1]=sinf(theta); theta*=theta_scale; }
    for(int i=0;i<half;++i){
        float c=cache[2*i], s=cache[2*i+1];
        float x0=x[2*i], x1=x[2*i+1];
        y[2*i]   = x0*c - x1*s;
        y[2*i+1] = x0*s + x1*c;
    }
}
