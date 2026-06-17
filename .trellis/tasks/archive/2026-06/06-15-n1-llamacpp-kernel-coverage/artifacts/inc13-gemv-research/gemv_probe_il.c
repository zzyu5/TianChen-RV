// inc13-gemv-research: INTERLEAVED-COLUMN cross-row GEMV probe (the real lever).
//
// THROWAWAY probe. Tests the mechanism the standard-layout probe could NOT test:
// amortize the INTEGER REDUCTION across columns. One wide vle8 loads 4 interleaved
// weight columns; one vwmul/vwmacc chain (at m4) produces interleaved products; a
// vnsrl/vadd REDUCTION TREE de-interleaves them into 4 column i32 sums with NO
// per-column vwredsum. Then a vector scale-fold across the 4 columns.
//
// This mirrors ggml's ggml_gemv_q4_0_8x8_q8_0 reduction cascade (repack.cpp:161-196),
// scaled to 4 columns to fit VLEN=128 (e8m4 = 64 lanes = 4 cols x 16 nibble-bytes).
//
// LAYOUT (we generate it; in a real kernel this is a one-time weight repack):
//   weights : block_q4_0x4-style. 4 fp16 scales, then 4*16=64 qs bytes interleaved
//             in 8-byte chunks across the 4 columns, XOR'd with 0x88 (offset-binary
//             -> two's complement, so decode is plain vsll/vsra, no -8).
//   activation: standard q8_0 (AoS), broadcast across columns.
//
// BASELINE = ggml per-row vec_dot called 4x over STANDARD q4_0 rows (the USED path).
// Bitwise-gated. Each column folds blocks ascending: sumi*(d_wt)*(d_act).
// NOTE on rounding: ggml's 8x8 gemv folds (sumi*d_act)*d_wt; vec_dot folds
//   (sumi*d_wt)*d_act. We replicate the VEC_DOT order here (multiply facc by d_act
//   first via vfmul_vf, THEN vfmacc by the d_wt column vector) to stay bitexact vs
//   the used path. The integer sumi is identical either way (integer add associative).
//
// Board: ssh rvv, VLEN=128 (VLENB=16). Build:
//   clang -march=rv64gcv_zfh_zvfh -mabi=lp64d -O3 -ffp-contract=fast
// Run pinned: taskset -c 3 ./gemv_probe_il <K> <N_rows> <iters> <repeats>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <riscv_vector.h>

#define QK 32
#define Q4_STRIDE 18
#define Q8_STRIDE 34

static inline float fp16_to_fp32(uint16_t h){ _Float16 hf; memcpy(&hf,&h,2); return (float)hf; }
static inline uint16_t fp32_to_fp16(float f){ _Float16 hf=(_Float16)f; uint16_t h; memcpy(&h,&hf,2); return h; }

// ---- ggml per-row vec_dot (standard q4_0 row vs standard q8_0 act) ----
static void vec_dot(int n, float *s, const uint8_t *x, const uint8_t *y){
  const int nb=n/QK; const size_t vl=QK/2; float sumf=0.0f;
  for(int ib=0;ib<nb;++ib){
    const uint8_t *xb=x+(size_t)ib*Q4_STRIDE, *yb=y+(size_t)ib*Q8_STRIDE;
    vuint8m1_t tx=__riscv_vle8_v_u8m1(xb+2,vl);
    vint8m1_t y0=__riscv_vle8_v_i8m1((const int8_t*)(yb+2),vl);
    vint8m1_t y1=__riscv_vle8_v_i8m1((const int8_t*)(yb+2+16),vl);
    vuint8m1_t xa=__riscv_vand_vx_u8m1(tx,0x0F,vl), xl=__riscv_vsrl_vx_u8m1(tx,4,vl);
    vint8m1_t v0=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(xa),8,vl);
    vint8m1_t v1=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(xl),8,vl);
    vint16m2_t m=__riscv_vwmul_vv_i16m2(v0,y0,vl);
    m=__riscv_vwmacc_vv_i16m2(m,v1,y1,vl);
    vint32m1_t z=__riscv_vmv_v_x_i32m1(0,vl);
    int sumi=__riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(m,z,vl));
    uint16_t dx,dy; memcpy(&dx,xb,2); memcpy(&dy,yb,2);
    sumf += sumi*fp16_to_fp32(dx)*fp16_to_fp32(dy);
  }
  *s=sumf;
}

// ============ INTERLEAVED 4-column weight block (block_q4_0x4 style) ============
// layout per block: 4 fp16 scales (8 bytes) + 64 interleaved+xor'd qs bytes.
// interleave: 8-byte chunks; chunk i belongs to column (i%4), source offset (i/4)*8.
// Build from 4 standard q4_0 column rows.  (== make_block_q4_0x4 with blck=8 idea,
//  but 4 columns; XOR 0x88.)
#define X4_STRIDE (8 + 64)
static void build_x4(uint8_t *dst, const uint8_t *cols[4], int nb){
  for(int ib=0; ib<nb; ++ib){
    uint8_t *o = dst + (size_t)ib*X4_STRIDE;
    for(int c=0;c<4;++c) memcpy(o + c*2, cols[c] + (size_t)ib*Q4_STRIDE, 2); // 4 scales
    uint8_t *qs = o + 8;
    const int end = QK*4/8/8; // = 64/8 = 8 chunks of 8 bytes? QK=32 -> 16 qbytes/col
    // 4 cols * 16 qbytes = 64 bytes = 8 chunks of 8.
    (void)end;
    for(int chunk=0; chunk<8; ++chunk){
      int c   = chunk % 4;            // which column
      int soff = (chunk/4)*8;         // byte offset within the column's 16 qs bytes
      const uint8_t *src = cols[c] + (size_t)ib*Q4_STRIDE + 2 + soff;
      uint64_t e; memcpy(&e, src, 8);
      e ^= 0x8888888888888888ULL;     // offset-binary -> two's complement
      memcpy(qs + chunk*8, &e, 8);
    }
  }
}

// ---- the interleaved 4-col kernel: 4 columns -> 4 fp32 outputs ----
// one vle8.m4 (64 bytes) loads all 4 columns; one vwmul/vwmacc chain at m4;
// ggml-style vnsrl/vadd reduction tree -> 4 i32 column sums; vector scale fold.
static void gemv_il4(int K, float *s4, const uint8_t *x4, const uint8_t *y){
  const int nb=K/QK;
  const size_t vl=QK;        // 32; m4 op width = vl*2 = 64 bytes loaded
  vfloat32m1_t sumf = __riscv_vfmv_v_f_f32m1(0.0f, 4);
  for(int ib=0; ib<nb; ++ib){
    const uint8_t *o = x4 + (size_t)ib*X4_STRIDE;
    const uint8_t *yb = y + (size_t)ib*Q8_STRIDE;
    // broadcast the activation block: ggml uses the 8-byte chunks of q8 replicated
    // via i64 splat. Here the interleave pairs col-chunk i with q8 chunk (i/4).
    // q8 has 32 i8; chunk0=bytes[0:8], chunk1=[8:16] -> lo half; chunk2,3 -> hi half.
    // PROBE-FIX: q8_0 qs starts at +2 (after fp16 header) -> misaligned i64 loads
    // trap-emulate on this board (no fast-misaligned). Copy to 8-aligned scratch
    // once/block (cheap) so the i64 broadcast loads are aligned, like ggml's
    // repacked block_q8_0x4 (qs at an 8-aligned offset). NOT a kernel cost -- a
    // probe artifact of driving the interleaved kernel from a STANDARD q8_0 block.
    int8_t ya[32] __attribute__((aligned(8)));
    memcpy(ya, yb + 2, 32);
    const int64_t a0=*(const int64_t*)&ya[0];
    const int64_t a1=*(const int64_t*)&ya[8];
    const int64_t a2=*(const int64_t*)&ya[16];
    const int64_t a3=*(const int64_t*)&ya[24];
    __asm__ __volatile__("" ::: "memory");
    // splat each 8-byte activation chunk across a 32-lane (m2) register: lane group
    // matches the 4-col interleave so vwmul lines up column c with its 8 q8 bytes.
    const vint8m2_t l0=__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, vl/4));
    const vint8m2_t l1=__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, vl/4));
    const vint8m2_t l2=__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, vl/4));
    const vint8m2_t l3=__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, vl/4));

    const vint8m4_t raw=__riscv_vle8_v_i8m4((const int8_t*)(o+8), vl*2); // 64 bytes
    const vint8m4_t lo=__riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(raw,4,vl*2),4,vl*2);
    const vint8m4_t hi=__riscv_vsra_vx_i8m4(raw,4,vl*2);
    const vint8m2_t lo0=__riscv_vget_v_i8m4_i8m2(lo,0), lo1=__riscv_vget_v_i8m4_i8m2(lo,1);
    const vint8m2_t hi0=__riscv_vget_v_i8m4_i8m2(hi,0), hi1=__riscv_vget_v_i8m4_i8m2(hi,1);

    vint16m4_t s0=__riscv_vwmul_vv_i16m4(lo0,l0,vl);
    s0=__riscv_vwmacc_vv_i16m4(s0,lo1,l1,vl);
    s0=__riscv_vwmacc_vv_i16m4(s0,hi0,l2,vl);
    s0=__riscv_vwmacc_vv_i16m4(s0,hi1,l3,vl);

    // ggml reduction tree -> 4 column sums in an i32m1 (4 lanes)
    const vuint32m4_t si=__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(s0));
    const vuint16m2_t h0=__riscv_vnsrl_wx_u16m2(si,0,vl/2);
    const vuint16m2_t h1=__riscv_vnsrl_wx_u16m2(si,16,vl/2);
    const vuint16m2_t h=__riscv_vadd_vv_u16m2(h0,h1,vl/2);
    const vuint32m2_t hi32=__riscv_vreinterpret_v_u16m2_u32m2(h);
    const vuint16m1_t g0=__riscv_vnsrl_wx_u16m1(hi32,0,vl/4);
    const vuint16m1_t g1=__riscv_vnsrl_wx_u16m1(hi32,16,vl/4);
    const vuint16m1_t g=__riscv_vadd_vv_u16m1(g0,g1,vl/4);
    const vuint32m1_t g32=__riscv_vreinterpret_v_u16m1_u32m1(g);
    const vint16mf2_t f0=__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(g32,0,4));
    const vint16mf2_t f1=__riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(g32,16,4));
    const vint32m1_t isum=__riscv_vwadd_vv_i32m1(f0,f1,4);
    const vfloat32m1_t facc=__riscv_vfcvt_f_x_v_f32m1(isum,4);

    // scale fold (VEC_DOT order: facc * d_act, then * d_wt[col]) -- bitexact vs vec_dot
    uint16_t dyb; memcpy(&dyb,yb,2); float dy=fp16_to_fp32(dyb);
    float bs[4]; for(int c=0;c<4;++c){ uint16_t d; memcpy(&d,o+c*2,2); bs[c]=fp16_to_fp32(d);}
    const vfloat32m1_t bsv=__riscv_vle32_v_f32m1(bs,4);
    const vfloat32m1_t t1=__riscv_vfmul_vf_f32m1(facc, dy, 4);      // (sumi)*d_act
    sumf=__riscv_vfmacc_vv_f32m1(sumf, t1, bsv, 4);                  // += *d_wt[col]
  }
  __riscv_vse32_v_f32m1(s4, sumf, 4);
}

// ============================ harness ============================
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
static void fill_q4(uint8_t *row,int nb,unsigned *sd){
  for(int b=0;b<nb;++b){ uint8_t *k=row+(size_t)b*Q4_STRIDE;
    uint16_t d=fp32_to_fp16(0.005f+(rand_r(sd)%100)*0.0003f); memcpy(k,&d,2);
    for(int j=0;j<QK/2;++j) k[2+j]=(uint8_t)(rand_r(sd)&0xFF);} }
static void fill_q8(uint8_t *y,int nb,unsigned *sd){
  for(int b=0;b<nb;++b){ uint8_t *k=y+(size_t)b*Q8_STRIDE;
    uint16_t d=fp32_to_fp16(0.01f+(rand_r(sd)%100)*0.0004f); memcpy(k,&d,2);
    for(int j=0;j<QK;++j) ((int8_t*)(k+2))[j]=(int8_t)((rand_r(sd)%255)-127);} }

int main(int argc,char**argv){
  int K=argc>1?atoi(argv[1]):4096;
  int N=argc>2?atoi(argv[2]):8;
  long iters=argc>3?atol(argv[3]):20000;
  int reps=argc>4?atoi(argv[4]):30;
  if(K%QK){fprintf(stderr,"K%%32\n");return 1;}
  if(N%4){fprintf(stderr,"N%%4\n");return 1;}
  int nb=K/QK; size_t rb=(size_t)nb*Q4_STRIDE;

  uint8_t *W=malloc((size_t)N*rb), *y=malloc((size_t)nb*Q8_STRIDE);
  float *sref=malloc(N*4), *s=malloc(N*4);
  unsigned sd=12345;
  for(int r=0;r<N;++r) fill_q4(W+(size_t)r*rb,nb,&sd);
  fill_q8(y,nb,&sd);

  // build interleaved weights: N/4 tiles of 4 columns
  uint8_t *Wil=malloc((size_t)(N/4)*nb*X4_STRIDE);
  for(int t=0;t<N/4;++t){
    const uint8_t *cols[4]={W+(size_t)(t*4+0)*rb,W+(size_t)(t*4+1)*rb,
                            W+(size_t)(t*4+2)*rb,W+(size_t)(t*4+3)*rb};
    build_x4(Wil+(size_t)t*nb*X4_STRIDE, cols, nb);
  }

  // reference: N x vec_dot over standard rows
  for(int r=0;r<N;++r) vec_dot(K,&sref[r],W+(size_t)r*rb,y);

  printf("# INTERLEAVED 4-col GEMV probe  K=%d N=%d iters=%ld reps=%d\n",K,N,iters,reps);
  printf("# ref s[0]=%.8e\n", sref[0]);

  // run+gate the interleaved kernel
  memset(s,0,N*4);
  for(int t=0;t<N/4;++t) gemv_il4(K,&s[t*4],Wil+(size_t)t*nb*X4_STRIDE,y);
  int ok=(memcmp(s,sref,N*4)==0);
  if(!ok){
    printf("# GATE FAIL -- s vs ref:\n");
    for(int r=0;r<N && r<8;++r) printf("#  r%d  got=%.8e  ref=%.8e\n",r,s[r],sref[r]);
  }

  // bench baseline (N x vec_dot)
  double best_a=1e30;
  for(int rp=0;rp<reps;++rp){ double t0=now_ns();
    for(long it=0;it<iters;++it) for(int r=0;r<N;++r) vec_dot(K,&s[r],W+(size_t)r*rb,y);
    double t1=now_ns(); double per=(t1-t0)/iters; if(per<best_a)best_a=per; }

  // bench interleaved
  double best_il=1e30;
  for(int rp=0;rp<reps;++rp){ double t0=now_ns();
    for(long it=0;it<iters;++it) for(int t=0;t<N/4;++t) gemv_il4(K,&s[t*4],Wil+(size_t)t*nb*X4_STRIDE,y);
    double t1=now_ns(); double per=(t1-t0)/iters; if(per<best_il)best_il=per; }

  printf("  a_vecdot (Nx vec_dot, USED path)   %9.1f ns/matvec  %7.1f ns/row\n", best_a, best_a/N);
  printf("  il4 (interleaved 4-col, tree-reduce) %9.1f ns/matvec  %7.1f ns/row  gate=%s  speedup=%.3fx\n",
         best_il, best_il/N, ok?"PASS":"FAIL", best_a/best_il);

  free(W);free(y);free(sref);free(s);free(Wil);
  return 0;
}
