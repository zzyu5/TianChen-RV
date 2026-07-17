// INC-38 iq1_s COMPILER-KNOB control (ssh rvv). Proves the $grid typed attr is the
// LIVE source of the emitted grid const -> HW result: the REAL-grid kernel and a
// PERTURBED-grid kernel (grid entry 5 low byte +1, emitted by the SAME compiler from a
// perturbed-attr input) MUST produce DIFFERENT *s on identical data. This rules out a
// hardcoded table -- the attr -> emitc const -> hardware flow is end-to-end live.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define QK_K 256
typedef struct { uint16_t d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

extern "C" void
tcrv_emitc_ggml_vec_dot_iq1_s_q8_K_kernel_ggml_vec_dot_iq1_s_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);   // real grid
extern "C" void
tcrv_emitc_ggml_vec_dot_iq1_s_q8_K_kernel_ggml_vec_dot_iq1_s_q8_K_PERT(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);   // perturbed grid

static uint64_t rng = 0xfeedface12345678ULL;
static uint32_t xr(){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static uint8_t rb(){ return (uint8_t)(xr()&0xff); }
static uint16_t rfp16(){ uint16_t e=(uint16_t)(9+(xr()%8)); return (uint16_t)(((xr()&1)<<15)|(e<<10)|(xr()&0x3ff)); }

int main(){
  int nb=8, n=nb*QK_K;
  block_iq1_s *X=(block_iq1_s*)malloc(nb*sizeof(block_iq1_s));
  block_q8_K *Y=(block_q8_K*)malloc(nb*sizeof(block_q8_K));
  for(int b=0;b<nb;++b){
    X[b].d=rfp16();
    for(int j=0;j<QK_K/8;++j) X[b].qs[j]=rb();
    // force qs index 5 (grid entry 5 is perturbed): set a qs byte to 5 with qh field 0
    // so the perturbed entry is actually addressed.
    for(int j=0;j<QK_K/32;++j) X[b].qh[j]=(uint16_t)(xr()&0x8fff); // group fields 0 except...
    X[b].qs[0]=5; X[b].qs[4]=5; X[b].qs[8]=5;  // index 5 hit in several sub-blocks
    float dy; uint16_t h=rfp16(); _Float16 hf; memcpy(&hf,&h,2); dy=(float)hf; Y[b].d=dy;
    for(int j=0;j<QK_K;++j) Y[b].qs[j]=(int8_t)rb();
    for(int g=0;g<QK_K/16;++g){int s=0;for(int j=0;j<16;++j)s+=Y[b].qs[g*16+j];Y[b].bsums[g]=(int16_t)s;}
  }
  float s_real=0.f, s_pert=0.f;
  tcrv_emitc_ggml_vec_dot_iq1_s_q8_K_kernel_ggml_vec_dot_iq1_s_q8_K((size_t)n,&s_real,(const uint8_t*)X,(const uint8_t*)Y);
  tcrv_emitc_ggml_vec_dot_iq1_s_q8_K_kernel_ggml_vec_dot_iq1_s_q8_K_PERT((size_t)n,&s_pert,(const uint8_t*)X,(const uint8_t*)Y);
  uint32_t a,b2; memcpy(&a,&s_real,4); memcpy(&b2,&s_pert,4);
  bool diverges = (a!=b2);
  printf("KNOB real-grid=%.6f perturbed-grid=%.6f -> %s\n", s_real, s_pert,
         diverges ? "DIVERGES (attr->const->HW flow live)" : "MATCHES (BUG: grid attr inert!)");
  free(X); free(Y);
  return diverges ? 0 : 1;
}
