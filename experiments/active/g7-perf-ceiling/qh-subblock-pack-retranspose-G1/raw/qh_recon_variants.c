/* qh_recon_variants.c -- rolled apples-to-apples of the q5_K GEVM qh-recon inner
 * per-(ii,h) tile, THREE variants, BYTE-IDENTICAL except the qh 5th-bit recon.
 * Compiled gcc-15.2 -march=rv64gcv_zvfh -mabi=lp64d -O3 (deployment domain, VLEN128).
 *
 * strip width = half = 8 columns (VLEN128 e8mf2 -> vl=8). Two arms (lo/hi nibble),
 * one super-half sub-block pair. This isolates ONLY the qh reconstruct delta:
 *   OLD_CUR   : current DEPLOYED emitter (RVVToEmitCBlockQuantLinear.cpp 9133-9142)
 *               loSel=vsrl(qh,s); loBit=vsll(vand(loSel,1),4); nLo=vor(loNib,loBit)  (4 op/arm)
 *   SB_KNEST  : sub-block native-mask (q5k-knest-G1 design, NOT deployed)
 *               m=vmsne(vand(qh,1<<s),0); nLo=vadd_mu(m,loNib,loNib,16)              (3 op/arm)
 *   RETRANS   : THIS design -- re-transposed qh is a per-lane bool plane (like q5_0)
 *               m=vlm_v_b8(mask_ptr); nLo=vadd_mu(m,loNib,loNib,16)                  (2 op/arm)
 */
#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

#define HALF 8
static const int sLoBit = 2, sHiBit = 3;   /* representative non-zero sub-block bits */

/* ---- OLD_CUR : current deployed sub-block-shift extract ---- */
void recon_old_cur(const uint8_t* packed, const uint8_t* qh,
                   vint8mf2_t* outLo, vint8mf2_t* outHi){
    size_t vl = HALF;
    vuint8mf2_t p   = __riscv_vle8_v_u8mf2(packed, vl);
    vuint8mf2_t loN = __riscv_vand_vx_u8mf2(p, 0x0F, vl);
    vuint8mf2_t hiN = __riscv_vsrl_vx_u8mf2(p, 4, vl);
    vuint8mf2_t qhs = __riscv_vle8_v_u8mf2(qh, vl);
    vuint8mf2_t loSel = __riscv_vsrl_vx_u8mf2(qhs, sLoBit, vl);
    vuint8mf2_t loBit = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(loSel,0x01,vl),4,vl);
    vuint8mf2_t hiSel = __riscv_vsrl_vx_u8mf2(qhs, sHiBit, vl);
    vuint8mf2_t hiBit = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(hiSel,0x01,vl),4,vl);
    *outLo = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vor_vv_u8mf2(loN, loBit, vl));
    *outHi = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vor_vv_u8mf2(hiN, hiBit, vl));
}

/* ---- SB_KNEST : sub-block native-mask, mask EXTRACTED from packed byte ---- */
void recon_sb_knest(const uint8_t* packed, const uint8_t* qh,
                    vint8mf2_t* outLo, vint8mf2_t* outHi){
    size_t vl = HALF;
    vuint8mf2_t p   = __riscv_vle8_v_u8mf2(packed, vl);
    vuint8mf2_t loN = __riscv_vand_vx_u8mf2(p, 0x0F, vl);
    vuint8mf2_t hiN = __riscv_vsrl_vx_u8mf2(p, 4, vl);
    vuint8mf2_t qhs = __riscv_vle8_v_u8mf2(qh, vl);
    vbool16_t loM = __riscv_vmsne_vx_u8mf2_b16(__riscv_vand_vx_u8mf2(qhs,(1u<<sLoBit),vl),0,vl);
    vbool16_t hiM = __riscv_vmsne_vx_u8mf2_b16(__riscv_vand_vx_u8mf2(qhs,(1u<<sHiBit),vl),0,vl);
    *outLo = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vadd_vx_u8mf2_mu(loM, loN, loN, 16, vl));
    *outHi = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vadd_vx_u8mf2_mu(hiM, hiN, hiN, 16, vl));
}

/* ---- RETRANS : re-transposed layout -> mask loaded DIRECTLY (vlm), lane c = bit c ---- */
void recon_retrans(const uint8_t* packed, const uint8_t* maskLo, const uint8_t* maskHi,
                   vint8mf2_t* outLo, vint8mf2_t* outHi){
    size_t vl = HALF;
    vuint8mf2_t p   = __riscv_vle8_v_u8mf2(packed, vl);
    vuint8mf2_t loN = __riscv_vand_vx_u8mf2(p, 0x0F, vl);
    vuint8mf2_t hiN = __riscv_vsrl_vx_u8mf2(p, 4, vl);
    vbool16_t loM = __riscv_vlm_v_b16(maskLo, vl);   /* re-transposed: 8-col contiguous bit plane */
    vbool16_t hiM = __riscv_vlm_v_b16(maskHi, vl);
    *outLo = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vadd_vx_u8mf2_mu(loM, loN, loN, 16, vl));
    *outHi = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vadd_vx_u8mf2_mu(hiM, hiN, hiN, 16, vl));
}
