// q8_0 DEPLOYED repack-GEVM: OUR m1 whole-LMUL chain vs the ggml arch repack
// opponent ggml_gemv_q8_0_16x1_q8_0 (byte-for-byte from
// /home/ubuntu/vericurve-rv-lab/llama.cpp .../arch/riscv/repack.cpp). The ggml
// block_q8_0x16 layout is byte-identical to our x16 repack (544B: 16 fp16 d + 512
// int8 qs[i*16+c]); the activation block_q8_0 (34B) matches too -- so BOTH kernels
// consume the SAME weight/act bytes. 3-arm vs a scalar oracle.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <riscv_vector.h>
#define GGML_RESTRICT __restrict
#define UNUSED(x) ((void)(x))
#define QK8_0 32
typedef uint16_t ggml_half;
struct block_q8_0   { ggml_half d;      int8_t qs[32]; };       // 34B
struct block_q8_0x16{ ggml_half d[16];  int8_t qs[512]; };      // 544B
#include <cassert>
void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);
    UNUSED(bs);

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb);

        // 1x16 Accumulator
        vfloat32m2_t sumf = __riscv_vfmv_v_f_f32m2(0.0f, 16);

        for (int l = 0; l < nb; l++) {
            // 1x16 Integer Accumulator
            vint32m2_t sumi = __riscv_vmv_v_x_i32m2(0.0f, 16);

            // Accumulation loop.
            for (int i = 0; i < QK8_0; i++) {
                // Load `b_ptr`.
                const vint8mf2_t b_0 = __riscv_vle8_v_i8mf2((const int8_t *)&b_ptr[l].qs[i * 16], 16);
                // const vint16m1_t b_0_16 = __riscv_vwcvt_x_x_v_i16m1(b_0, 16);

                sumi = __riscv_vwadd_wv_i32m2(sumi, __riscv_vwmul_vx_i16m1(b_0, a_ptr[l].qs[i], 16), 16);
            }

            const vfloat16m1_t b_d = __riscv_vle16_v_f16m1((const _Float16 *)b_ptr[l].d, 16);
            const vfloat32m2_t d_0 = __riscv_vfwmul_vf_f32m2(b_d, *(const _Float16 *)&a_ptr[l].d, 16);

            sumf = __riscv_vfmacc_vv_f32m2(sumf, __riscv_vfcvt_f_x_v_f32m2(sumi, 16), d_0, 16);
        }

        __riscv_vse32_v_f32m2(s + x * 16, sumf, 16);
    }
}


// our deployed m1 kernel (renamed symbol q8_gevm_m1), ABI f(K,out,M,w,bx,a,by,nrc)
extern "C" void q8_gevm_m1(size_t,float*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,int32_t);
typedef _Float16 f16;
static uint64_t rs; static inline uint32_t xr(){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return(uint32_t)rs;}
static inline int8_t ri8(){return(int8_t)((int)(xr()%255)-127);} static inline f16 rsc(){return(f16)(0.008f+(float)(xr()%48)*0.001f);}
static void build(std::vector<uint8_t>&W,std::vector<uint8_t>&A,size_t M,size_t K,uint64_t seed){
  rs=seed?seed:0x9e3779b97f4a7c15ull; size_t nb=K/32,ncg=M/16; W.assign(ncg*nb*544,0); A.assign(nb*34,0);
  for(size_t cg=0;cg<ncg;++cg)for(size_t b=0;b<nb;++b){uint8_t*blk=&W[(cg*nb+b)*544];
    for(size_t c=0;c<16;++c)((f16*)blk)[c]=rsc(); for(size_t p=0;p<32;++p)for(size_t c=0;c<16;++c)blk[32+p*16+c]=(uint8_t)ri8();}
  for(size_t b=0;b<nb;++b){uint8_t*ab=&A[b*34];((f16*)ab)[0]=rsc();for(size_t p=0;p<32;++p)ab[2+p]=(uint8_t)ri8();}
}
static void oracle(const std::vector<uint8_t>&W,const std::vector<uint8_t>&A,std::vector<float>&out,size_t M,size_t K){
  size_t nb=K/32;out.assign(M,0.0f);
  for(size_t col=0;col<M;++col){size_t cg=col/16,c=col%16;float acc=0;
    for(size_t b=0;b<nb;++b){const uint8_t*blk=&W[(cg*nb+b)*544];const uint8_t*ab=&A[b*34];int32_t isum=0;
      for(size_t p=0;p<32;++p)isum+=(int32_t)(int8_t)blk[32+p*16+c]*(int32_t)(int8_t)ab[2+p];
      acc=fmaf((float)isum,(float)((const f16*)blk)[c]*(float)((const f16*)ab)[0],acc);}
    out[col]=acc;}
}
static std::vector<uint8_t> junk(64ull*1024*1024);
static void flush(){volatile uint64_t s=0;for(size_t i=0;i<junk.size();i+=64)s+=junk[i];(void)s;}
static uint64_t now_ns(){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return(uint64_t)ts.tv_sec*1000000000ull+ts.tv_nsec;}
int main(){
  struct Cfg{size_t M,K;const char*tag;}cfgs[]={{256,1024,"M=256 K=1024"},{2048,4096,"M=2048 K=4096"},{8192,4096,"M=8192 K=4096 DRAM"}};
  int reps=15;
  printf("footprint,seed,ours_ns,ggml_ns,ours_over_ggml,ours_exact,ggml_exact\n");
  for(auto&c:cfgs)for(uint64_t seed=0;seed<2;++seed){
    std::vector<uint8_t>W,A;build(W,A,c.M,c.K,seed+1);std::vector<float>ref;oracle(W,A,ref,c.M,c.K);
    std::vector<float>oo(c.M,0),og(c.M,0);
    q8_gevm_m1(c.K,oo.data(),c.M,W.data(),0,A.data(),0,1);
    ggml_gemv_q8_0_16x1_q8_0((int)c.K,og.data(),0,W.data(),A.data(),1,(int)c.M);
    int oe=memcmp(oo.data(),ref.data(),c.M*4)==0, ge=memcmp(og.data(),ref.data(),c.M*4)==0;
    std::vector<uint64_t> so,sg;
    for(int r=0;r<reps;++r){flush();uint64_t t0=now_ns();q8_gevm_m1(c.K,oo.data(),c.M,W.data(),0,A.data(),0,1);so.push_back(now_ns()-t0);
      flush();t0=now_ns();ggml_gemv_q8_0_16x1_q8_0((int)c.K,og.data(),0,W.data(),A.data(),1,(int)c.M);sg.push_back(now_ns()-t0);}
    std::sort(so.begin(),so.end());std::sort(sg.begin(),sg.end());
    printf("%s,%llu,%llu,%llu,%.4f,%s,%s\n",c.tag,(unsigned long long)seed,(unsigned long long)so[7],(unsigned long long)sg[7],
      (double)so[7]/(double)sg[7],oe?"PASS":"FAIL",ge?"PASS":"FAIL");fflush(stdout);
  }
  return 0;
}
