/* kquant_repacker.h -- [SEL-1 T4b / M1] GENERAL offline K-quant weight repacker + q8_K
 * activation interleaver. Header-only, no deps beyond <cstdint>/<cstring>. Generalizes the
 * M0 one-shot q4_K pack (commit 663214e9) to ARBITRARY tensor shapes for q4_K AND q5_K, and
 * promotes the q8_K activation interleave to a SHARED helper (byte-identical for q4_K/q5_K).
 *
 * The repacked byte layouts are DERIVED (not trial-and-error) from the golden emitted repack-GEMM
 * kernels the board consumes:
 *   OUR q4_K kernel  golden_q4K.c              md5 b0b5beac  (block_q4_Kx16 stride 2304)
 *   OUR q5_K kernel  gemm_q5_K_q8_K.kernel.c   md5 ba30ba54  (block_q5_Kx16 stride 2816)
 *   activation       block_q8_Kx4              stride 1168 (shared, both formats)
 *
 * ---- block_q4_Kx16 stride 2304 ----
 *   d[16]     fp16 @0      d[c]    @ 2*c
 *   dmin[16]  fp16 @32     dmin[c] @ 32+2*c
 *   scales    @64  192B    ggml repack.cpp CUSTOM 6-bit split (NOT the on-disk scales[12]):
 *                          get_scale_min_k4(scales[12]) -> sc[0..7],mn[0..7]; then
 *                            lo strip g<4 @ 64+g*16+c  ; g>=4 @ 128+(g-4)*16+c :
 *                                byte = (mn[g]&0xF)<<4 | (sc[g]&0xF)
 *                            hi strip sb @ 192+sb*16+c :
 *                                [1:0]=sc[sb]>>4 [3:2]=mn[sb]>>4 [5:4]=sc[sb+4]>>4 [7:6]=mn[sb+4]>>4
 *   qs[128][16] @256 2048B  qs[i] col c @ 256 + i*16 + c   (raw block_q4_K.qs, straight interleave)
 *
 * ---- block_q5_Kx16 stride 2816 (== q4_K + a qh 5th-bit plane, SAME scale region) ----
 *   d[16]/dmin[16]/scales  IDENTICAL to q4_K (@0/@32/@64, 256B header)
 *   qh[32][16]  @256  512B  qh[m] col c @ 256 + m*16 + c   (raw block_q5_K.qh, STRAIGHT interleave)
 *   qs[128][16] @768 2048B  qs[i] col c @ 768 + i*16 + c   (raw block_q5_K.qs, straight interleave)
 *   NOTE: qs is pushed to @768 (AFTER the qh plane) -- NOT @256. The kernel assembles the 5-bit
 *   weight as  nibble | (qh_bit<<4)  where for nibble byte i the qh byte is qh[i%32] and the bit
 *   is 2*(i/32) (low nibble) / 2*(i/32)+1 (high nibble) -- exactly ggml's u1/u2<<=2 plane walk.
 *   A verbatim qh copy preserves all 8 planes; no bit transpose needed (confirmed vs golden kernel).
 *
 * ---- block_q8_Kx4 stride 1168 (activation, shared) ----
 *   d[4]    fp32 @0     d[r]    @ 4*r
 *   qs[256][4]     @16  qs[p] row r @ 16 + p*4 + r
 *   bsums[16][4] i16 @1040  bsums[g] row r @ 1040 + g*8 + r*2   (true per-16 sums, ggml quantize_row_q8_K)
 *
 * Weight tensor natural layout (ggml [ne0=K, ne1=N] row-major): block for output-column c
 * (=row of the weight matrix) and block-index b is src[(size_t)c*nb + b]; nc columns grouped in
 * 16s -> nc/16 x16 groups. Activation tensor: block for row r block b is src[(size_t)r*nb + b];
 * nr rows grouped in 4s -> nr/4 x4 groups. Precondition: nc%16==0, nr%4==0.
 */
#ifndef KQUANT_REPACKER_H
#define KQUANT_REPACKER_H
#include <cstdint>
#include <cstring>
#include <cstddef>

#define KQR_QK_K 256
#define KQR_K_SCALE_SIZE 12
typedef uint16_t kqr_half;

/* on-disk ggml block structs (ground-truth oracle inputs) */
struct kqr_block_q4_K { kqr_half d, dmin; uint8_t scales[KQR_K_SCALE_SIZE]; uint8_t qs[KQR_QK_K/2]; };            /* 144 */
struct kqr_block_q5_K { kqr_half d, dmin; uint8_t scales[KQR_K_SCALE_SIZE]; uint8_t qh[KQR_QK_K/8]; uint8_t qs[KQR_QK_K/2]; }; /* 176 */
struct kqr_block_q8_K { float d; int8_t qs[KQR_QK_K]; int16_t bsums[KQR_QK_K/16]; };                              /* 292 */

/* repacked group strides */
#define KQR_STRIDE_Q4Kx16 2304
#define KQR_STRIDE_Q5Kx16 2816
#define KQR_STRIDE_Q8Kx4  1168

/* ---- byte-size helpers for a whole tensor of the given shape ---- */
static inline size_t kqr_bytes_q4_K(int nc, int nb){ return (size_t)(nc/16)*nb*KQR_STRIDE_Q4Kx16; }
static inline size_t kqr_bytes_q5_K(int nc, int nb){ return (size_t)(nc/16)*nb*KQR_STRIDE_Q5Kx16; }
static inline size_t kqr_bytes_q8_K(int nr, int nb){ return (size_t)(nr/4)*nb*KQR_STRIDE_Q8Kx4; }

static inline void kqr_wr16(uint8_t* p, uint16_t h){ memcpy(p,&h,2); }

/* ggml canonical 6-bit K-scale unpack of the on-disk scales[12] -> 8 sc + 8 m */
static inline void kqr_get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m){
    if(j<4){ *d=q[j]&63; *m=q[j+4]&63; }
    else   { *d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j]>>6)<<4); }
}

/* SHARED scale-region encoder: writes the ggml repack.cpp CUSTOM 6-bit split for column c of a
 * q4_K OR q5_K x16 group (the header @0..256 is byte-identical for both formats). */
static inline void kqr_pack_scales16(uint8_t* blk, const uint8_t scales[KQR_K_SCALE_SIZE], int c){
    uint8_t sc[8], mn[8];
    for(int jj=0;jj<8;++jj) kqr_get_scale_min_k4(jj,scales,&sc[jj],&mn[jj]);
    for(int g=0;g<8;++g){
        int loOff = (g<4) ? (64 + g*16 + c) : (128 + (g-4)*16 + c);
        blk[loOff] = (uint8_t)(((mn[g]&0xF)<<4) | (sc[g]&0xF));
    }
    for(int sb=0;sb<4;++sb){
        blk[192 + sb*16 + c] = (uint8_t)(
            ((sc[sb]>>4)&3) | (((mn[sb]>>4)&3)<<2) |
            (((sc[sb+4]>>4)&3)<<4) | (((mn[sb+4]>>4)&3)<<6));
    }
}

/* ---- GENERAL q4_K weight repack: [nc][nb] original blocks -> block_q4_Kx16 dst (stride 2304) ---- */
static inline void kqr_repack_q4_K(uint8_t* dst, const kqr_block_q4_K* src, int nc, int nb){
    int ng = nc/16;
    memset(dst, 0, kqr_bytes_q4_K(nc,nb));
    for(int g=0; g<ng; ++g) for(int b=0; b<nb; ++b){
        uint8_t* blk = dst + ((size_t)g*nb + b)*KQR_STRIDE_Q4Kx16;
        for(int c=0; c<16; ++c){
            const kqr_block_q4_K& x = src[(size_t)(g*16+c)*nb + b];
            kqr_wr16(blk + 2*c,        x.d);
            kqr_wr16(blk + 32 + 2*c,   x.dmin);
            kqr_pack_scales16(blk, x.scales, c);
            for(int i=0;i<128;++i) blk[256 + i*16 + c] = x.qs[i];
        }
    }
}

/* ---- GENERAL q5_K weight repack: [nc][nb] original blocks -> block_q5_Kx16 dst (stride 2816) ----
 * q5_K = q4_K header + qh 5th-bit plane @256 (verbatim interleave) + qs pushed to @768. */
static inline void kqr_repack_q5_K(uint8_t* dst, const kqr_block_q5_K* src, int nc, int nb){
    int ng = nc/16;
    memset(dst, 0, kqr_bytes_q5_K(nc,nb));
    for(int g=0; g<ng; ++g) for(int b=0; b<nb; ++b){
        uint8_t* blk = dst + ((size_t)g*nb + b)*KQR_STRIDE_Q5Kx16;
        for(int c=0; c<16; ++c){
            const kqr_block_q5_K& x = src[(size_t)(g*16+c)*nb + b];
            kqr_wr16(blk + 2*c,        x.d);
            kqr_wr16(blk + 32 + 2*c,   x.dmin);
            kqr_pack_scales16(blk, x.scales, c);
            for(int m=0;m<KQR_QK_K/8;++m) blk[256 + m*16 + c] = x.qh[m];   /* 32 qh bytes @256 */
            for(int i=0;i<128;++i)        blk[768 + i*16 + c] = x.qs[i];   /* 128 qs bytes @768 */
        }
    }
}

/* ---- SHARED q8_K activation interleave: [nr][nb] blocks -> block_q8_Kx4 dst (stride 1168) ---- */
static inline void kqr_interleave_q8_K(uint8_t* dst, const kqr_block_q8_K* src, int nr, int nb){
    int ng = nr/4;
    memset(dst, 0, kqr_bytes_q8_K(nr,nb));
    for(int g=0; g<ng; ++g) for(int b=0; b<nb; ++b){
        uint8_t* blk = dst + ((size_t)g*nb + b)*KQR_STRIDE_Q8Kx4;
        for(int r=0; r<4; ++r){
            const kqr_block_q8_K& a = src[(size_t)(g*4+r)*nb + b];
            memcpy(blk + 4*r, &a.d, 4);
            for(int p=0;p<256;++p) blk[16 + p*4 + r] = (uint8_t)a.qs[p];
            for(int gg=0;gg<16;++gg){ int16_t bs=a.bsums[gg]; memcpy(blk + 1040 + gg*8 + r*2, &bs, 2); }
        }
    }
}

#endif /* KQUANT_REPACKER_H */
