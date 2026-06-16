// INC-30 COMPILER-KNOB negative control: two COMPILER-EMITTED kernels that differ
// ONLY in the MLIR codebook= attr (real kvalues_iq4nl vs the LINEAR nibble-8 table).
// Proves the codebook attr->emission->HW-result link end-to-end: same input bytes,
// only the compiler's codebook knob flipped -> the HW result flips.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static const int QK=32, IQ4_STRIDE=18, Q8_STRIDE=34;
static const int8_t kvalues_iq4nl[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
static const int8_t kvalues_linear[16]={-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7};
static inline float fp16(uint16_t h){_Float16 f;std::memcpy(&f,&h,2);return (float)f;}
extern "C" void tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_ggml_vec_dot_iq4_nl_q8_0(size_t,float*,const uint8_t*,const uint8_t*);
extern "C" void tcrv_LINEAR_codebook_kernel(size_t,float*,const uint8_t*,const uint8_t*);
static void gen_ref(int n,float*s,const uint8_t*x,const uint8_t*y,const int8_t*t){
  int nb=n/QK; float sumf=0;
  for(int ib=0;ib<nb;ib++){const uint8_t*xb=x+ib*IQ4_STRIDE;const uint8_t*yb=y+ib*Q8_STRIDE;
    const uint8_t*xqs=xb+2;const int8_t*yqs=(const int8_t*)(yb+2);uint16_t xd,yd;
    std::memcpy(&xd,xb,2);std::memcpy(&yd,yb,2);float d=fp16(yd)*fp16(xd);int s1=0,s2=0;
    for(int j=0;j<QK/2;j++){s1+=yqs[j]*t[xqs[j]&0xf];s2+=yqs[j+QK/2]*t[xqs[j]>>4];}
    sumf+=d*(s1+s2);} *s=sumf;}
static uint32_t rng=0x9e3779b9u; static uint32_t xr(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
int main(){
  int n=2048,nb=n/QK;
  uint8_t*vx=(uint8_t*)malloc(nb*IQ4_STRIDE),*vy=(uint8_t*)malloc(nb*Q8_STRIDE);
  for(int i=0;i<nb;i++){uint8_t*b=vx+i*IQ4_STRIDE;_Float16 d=(_Float16)(((float)(int)(xr()%2001)-1000.f)/256.f);std::memcpy(b,&d,2);
    for(int j=0;j<QK/2;j++){uint8_t nib=(uint8_t)(j&0xF);b[2+j]=(uint8_t)(nib|(nib<<4));}}
  for(int i=0;i<nb;i++){uint8_t*b=vy+i*Q8_STRIDE;_Float16 d=(_Float16)(((float)(int)(xr()%2001)-1000.f)/256.f);std::memcpy(b,&d,2);
    for(int j=0;j<QK;j++)b[2+j]=(uint8_t)(int8_t)((j&1)?-128:127);}
  float so_real=0, so_lin=0, ref_real=0, ref_lin=0;
  tcrv_emitc_ggml_vec_dot_iq4_nl_q8_0_kernel_ggml_vec_dot_iq4_nl_q8_0(n,&so_real,vx,vy); // compiler: real attr
  tcrv_LINEAR_codebook_kernel(n,&so_lin,vx,vy);                                          // compiler: linear attr
  gen_ref(n,&ref_real,vx,vy,kvalues_iq4nl);   // math with real table
  gen_ref(n,&ref_lin, vx,vy,kvalues_linear);  // math with linear table (= q4_0 on these bytes)
  printf("compiler(real-attr)   = %.9g   (ref real  = %.9g)\n",so_real,ref_real);
  printf("compiler(linear-attr) = %.9g   (ref linear= %.9g, == q4_0 on these bytes)\n",so_lin,ref_lin);
  int fail=0;
  if(std::memcmp(&so_real,&ref_real,4)!=0){printf("FAIL: real-attr kernel != real-table math\n");fail=1;}
  if(std::memcmp(&so_lin,&ref_lin,4)!=0){printf("FAIL: linear-attr kernel != linear-table math\n");fail=1;}
  if(std::memcmp(&so_real,&so_lin,4)==0){printf("FAIL(vacuous): the two compiler kernels gave the SAME result\n");fail=1;}
  if(!fail)printf("COMPILER-KNOB CONTROL OK: flipping ONLY the codebook= attr -> emitted decl flips -> HW result flips (real != linear), each matching its own table's math\n");
  free(vx);free(vy);return fail;}
