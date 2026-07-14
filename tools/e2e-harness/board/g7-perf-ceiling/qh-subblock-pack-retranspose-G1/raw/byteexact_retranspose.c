/* byteexact_retranspose.c  --  G1 byte-exact hard gate for the q5_K qh RE-TRANSPOSE design.
 *
 * PROVES three things over a 16-column q5_K super-block group with RANDOM qh/qs:
 *   (A) PACKING ROUND-TRIP is a BIJECTION: re-transpose(qh) then invert == original qh
 *       (every high bit preserved, only relocated -- weight VALUES unchanged).
 *   (B) DECODE EQUIVALENCE: native-mask decode on the RE-TRANSPOSED layout ==
 *       current sub-block-bit-extract decode on the CURRENT layout == stock ggml scalar
 *       (the 5th bit injected into the nibble is identical for every (col,elem)).
 *   (C) FULL 5-bit WEIGHT byte-exact: nibble|(bit<<4) identical across all three.
 *
 * Geometry (block_q5_Kx16, stride 2816; kquant_repacker.h):
 *   16 columns interleaved; QK_K=256 elems/col; qh = 32 bytes/col, each byte holds
 *   8 sub-block bits (positions m=0..31, sub-blocks s=0..7).
 *   CURRENT pack   : blk[256 + m*16 + c] = qh_col[c][m]      (verbatim, 8 planes/byte)
 *   RE-TRANSPOSE   : per (m,s) a 16-bit mask, bit c = (qh_col[c][m]>>s)&1
 *                    stored @ 256 + (m*8 + s)*2  (2 bytes, little-endian)  -> 512B total (== current)
 *
 * ggml element walk (dequantize_row_q5_K):  for j in {0,64,128,192}: two 32-elem sub-blocks
 *   low : y = d1*((ql[l]&0xF) + (qh[l]&u1?16:0)) ; u1=1<<(2*(j/64))
 *   high: y = d2*((ql[l]>>4)  + (qh[l]&u2?16:0)) ; u2=2<<(2*(j/64))
 *   => element (j-block jb in 0..3, half hi in {0,1}, l in 0..31): sub-block s = 2*jb+hi, pos m=l.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NCOL 16
#define QK_K 256

static uint8_t qh_col[NCOL][QK_K/8];   /* original per-column qh (32 bytes = 8 planes) */
static uint8_t qs_col[NCOL][QK_K/2];   /* original per-column nibbles (128 bytes) */

static uint8_t blk_cur[2816];          /* current repacked group */
static uint8_t blk_new[2816];          /* re-transposed repacked group */

int main(void){
    srand(20260714u);
    for(int c=0;c<NCOL;++c){
        for(int m=0;m<QK_K/8;++m) qh_col[c][m] = (uint8_t)(rand() & 0xFF);
        for(int i=0;i<QK_K/2;++i) qs_col[c][i] = (uint8_t)(rand() & 0xFF);
    }

    /* ---- CURRENT pack (verbatim qh + straight nibble interleave) ---- */
    memset(blk_cur,0,sizeof blk_cur);
    for(int c=0;c<NCOL;++c){
        for(int m=0;m<QK_K/8;++m) blk_cur[256 + m*16 + c] = qh_col[c][m];   /* 512B qh @256 */
        for(int i=0;i<QK_K/2;++i) blk_cur[768 + i*16 + c] = qs_col[c][i];   /* 2048B qs @768 */
    }

    /* ---- RE-TRANSPOSE pack: qh -> per-(m,s) 16-bit column mask; nibbles UNCHANGED ---- */
    memset(blk_new,0,sizeof blk_new);
    for(int m=0;m<QK_K/8;++m){
        for(int s=0;s<8;++s){
            uint16_t mask = 0;
            for(int c=0;c<NCOL;++c) mask |= (uint16_t)(((qh_col[c][m]>>s)&1u) << c);
            int off = 256 + (m*8 + s)*2;                 /* 2 bytes / (m,s) */
            blk_new[off+0] = (uint8_t)(mask & 0xFF);
            blk_new[off+1] = (uint8_t)(mask >> 8);
        }
    }
    for(int c=0;c<NCOL;++c)
        for(int i=0;i<QK_K/2;++i) blk_new[768 + i*16 + c] = qs_col[c][i];   /* qs identical */

    /* ================= (A) PACKING ROUND-TRIP BIJECTION ================= */
    int fail_A = 0;
    for(int c=0;c<NCOL;++c){
        for(int m=0;m<QK_K/8;++m){
            uint8_t rebuilt = 0;
            for(int s=0;s<8;++s){
                int off = 256 + (m*8 + s)*2;
                uint16_t mask = (uint16_t)blk_new[off] | ((uint16_t)blk_new[off+1]<<8);
                rebuilt |= (uint8_t)(((mask>>c)&1u) << s);
            }
            if(rebuilt != qh_col[c][m]) fail_A++;
        }
    }
    /* nibble region must be byte-identical between the two packs */
    int fail_qs = 0;
    for(int i=768;i<2816;++i) if(blk_cur[i]!=blk_new[i]) fail_qs++;
    /* qh regions occupy the SAME 512 bytes and hold the SAME popcount (permutation) */
    long pc_cur=0, pc_new=0;
    for(int i=256;i<768;++i){ pc_cur+=__builtin_popcount(blk_cur[i]); pc_new+=__builtin_popcount(blk_new[i]); }

    /* ================= (B)/(C) DECODE EQUIVALENCE + FULL 5-bit WEIGHT ================= */
    /* iterate every element of every column, decode the 5th bit + full weight three ways */
    int fail_bit = 0, fail_w = 0, ntest = 0;
    for(int c=0;c<NCOL;++c){
        for(int jb=0;jb<4;++jb){          /* super-64 block 0..3 */
            for(int hi=0;hi<2;++hi){      /* low-nibble sub-block (0) / high-nibble sub-block (1) */
                int s = 2*jb + hi;        /* sub-block bit index 0..7 */
                for(int l=0;l<32;++l){    /* intra-sub-block position == qh byte index m */
                    int m = l;
                    /* nibble byte index in the col: ggml packs low->even32, high walks with ql+=32 per jb */
                    int nib_byte = jb*32 + l;               /* ql index within the 128-byte nibble region */
                    uint8_t packed = qs_col[c][nib_byte];
                    uint8_t nib = hi ? (uint8_t)(packed>>4) : (uint8_t)(packed & 0x0F);

                    /* -- stock ggml scalar 5th bit -- */
                    uint8_t u = (uint8_t)((hi?2:1) << (2*jb));       /* u1/u2<<=2 walk */
                    int bit_stock = (qh_col[c][m] & u) ? 1 : 0;

                    /* -- CURRENT decode: sub-block-bit extract from the verbatim byte -- */
                    uint8_t cur_byte = blk_cur[256 + m*16 + c];
                    int bit_cur = (cur_byte >> s) & 1;

                    /* -- RE-TRANSPOSE decode: vlm-style native mask, lane c = bit c -- */
                    int off = 256 + (m*8 + s)*2;
                    uint16_t nm = (uint16_t)blk_new[off] | ((uint16_t)blk_new[off+1]<<8);
                    int bit_new = (nm >> c) & 1;

                    if(!(bit_stock==bit_cur && bit_cur==bit_new)) fail_bit++;

                    /* full 5-bit weight (K-quant: NO offset-binary; bias lives in 6-bit MIN) */
                    int w_stock = nib | (bit_stock<<4);
                    int w_cur   = nib | (bit_cur  <<4);
                    int w_new   = nib | (bit_new  <<4);
                    if(!(w_stock==w_cur && w_cur==w_new)) fail_w++;
                    ntest++;
                }
            }
        }
    }

    printf("(A) qh round-trip bijection mismatches : %d  (0 == byte-exact permutation)\n", fail_A);
    printf("    nibble region byte-diff cur vs new : %d  (0 == nibbles untouched)\n", fail_qs);
    printf("    qh popcount  cur=%ld  new=%ld       : %s\n", pc_cur, pc_new, pc_cur==pc_new?"EQUAL (permutation)":"DIFFER!!");
    printf("(B) 5th-bit decode mismatches (stock==cur==new) : %d / %d\n", fail_bit, ntest);
    printf("(C) full 5-bit weight mismatches               : %d / %d\n", fail_w, ntest);
    int ok = (fail_A==0 && fail_qs==0 && pc_cur==pc_new && fail_bit==0 && fail_w==0);
    printf("RESULT: %s\n", ok?"PASS  (re-transpose is byte-exact: bijective repack + decode-equivalent)":"FAIL");
    return ok?0:1;
}
