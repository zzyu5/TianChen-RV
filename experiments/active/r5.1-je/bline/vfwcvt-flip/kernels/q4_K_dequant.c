#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void weft_emitc_dequant_q4_K_kernel_dequant_q4_K(size_t v1, const uint8_t* v2, float* v3) {
  size_t nb = v1 / 256;
  for (size_t ib = 0; ib < nb; ib += 1) {
    const uint8_t* v8 = v2 + ib * 144;
    float* yb = v3 + ib * 256;
    float d    = (float)*(const _Float16 *)(v8);
    float dmin = (float)*(const _Float16 *)(v8 + 2);
    const uint8_t* sc = v8 + 4;
    float da0  = d    * (float)(sc[0] & 63);
    float mla0 = dmin * (float)(sc[4] & 63);
    float db0  = d    * (float)(sc[1] & 63);
    float mlb0 = dmin * (float)(sc[5] & 63);
    float da1  = d    * (float)(sc[2] & 63);
    float mla1 = dmin * (float)(sc[6] & 63);
    float db1  = d    * (float)(sc[3] & 63);
    float mlb1 = dmin * (float)(sc[7] & 63);
    float da2  = d    * (float)((sc[8] & 0x0F) | ((sc[0] >> 6) << 4));
    float mla2 = dmin * (float)((sc[8] >> 4) | ((sc[4] >> 6) << 4));
    float db2  = d    * (float)((sc[9] & 0x0F) | ((sc[1] >> 6) << 4));
    float mlb2 = dmin * (float)((sc[9] >> 4) | ((sc[5] >> 6) << 4));
    float da3  = d    * (float)((sc[10] & 0x0F) | ((sc[2] >> 6) << 4));
    float mla3 = dmin * (float)((sc[10] >> 4) | ((sc[6] >> 6) << 4));
    float db3  = d    * (float)((sc[11] & 0x0F) | ((sc[3] >> 6) << 4));
    float mlb3 = dmin * (float)((sc[11] >> 4) | ((sc[7] >> 6) << 4));

    const uint8_t* q_0 = v8 + 16;
    vuint8m2_t qv_0 = __riscv_vle8_v_u8m2(q_0, 32);
    vuint8m2_t nlo_0 = __riscv_vand_vx_u8m2(qv_0, 15, 32);
    vuint16m4_t lo16_0 = __riscv_vzext_vf2_u16m4(nlo_0, 32);
    vfloat32m8_t lof_0 = __riscv_vfwcvt_f_xu_v_f32m8(lo16_0, 32);
    vfloat32m8_t loacc_0 = __riscv_vfmv_v_f_f32m8(mla0, 32);
    vfloat32m8_t lor_0 = __riscv_vfmsac_vf_f32m8(loacc_0, da0, lof_0, 32);
    __riscv_vse32_v_f32m8(yb + 0, lor_0, 32);
    vuint8m2_t nhi_0 = __riscv_vsrl_vx_u8m2(qv_0, 4, 32);
    vuint16m4_t hi16_0 = __riscv_vzext_vf2_u16m4(nhi_0, 32);
    vfloat32m8_t hif_0 = __riscv_vfwcvt_f_xu_v_f32m8(hi16_0, 32);
    vfloat32m8_t hiacc_0 = __riscv_vfmv_v_f_f32m8(mlb0, 32);
    vfloat32m8_t hir_0 = __riscv_vfmsac_vf_f32m8(hiacc_0, db0, hif_0, 32);
    __riscv_vse32_v_f32m8(yb + 32, hir_0, 32);
    const uint8_t* q_1 = v8 + 48;
    vuint8m2_t qv_1 = __riscv_vle8_v_u8m2(q_1, 32);
    vuint8m2_t nlo_1 = __riscv_vand_vx_u8m2(qv_1, 15, 32);
    vuint16m4_t lo16_1 = __riscv_vzext_vf2_u16m4(nlo_1, 32);
    vfloat32m8_t lof_1 = __riscv_vfwcvt_f_xu_v_f32m8(lo16_1, 32);
    vfloat32m8_t loacc_1 = __riscv_vfmv_v_f_f32m8(mla1, 32);
    vfloat32m8_t lor_1 = __riscv_vfmsac_vf_f32m8(loacc_1, da1, lof_1, 32);
    __riscv_vse32_v_f32m8(yb + 64, lor_1, 32);
    vuint8m2_t nhi_1 = __riscv_vsrl_vx_u8m2(qv_1, 4, 32);
    vuint16m4_t hi16_1 = __riscv_vzext_vf2_u16m4(nhi_1, 32);
    vfloat32m8_t hif_1 = __riscv_vfwcvt_f_xu_v_f32m8(hi16_1, 32);
    vfloat32m8_t hiacc_1 = __riscv_vfmv_v_f_f32m8(mlb1, 32);
    vfloat32m8_t hir_1 = __riscv_vfmsac_vf_f32m8(hiacc_1, db1, hif_1, 32);
    __riscv_vse32_v_f32m8(yb + 96, hir_1, 32);
    const uint8_t* q_2 = v8 + 80;
    vuint8m2_t qv_2 = __riscv_vle8_v_u8m2(q_2, 32);
    vuint8m2_t nlo_2 = __riscv_vand_vx_u8m2(qv_2, 15, 32);
    vuint16m4_t lo16_2 = __riscv_vzext_vf2_u16m4(nlo_2, 32);
    vfloat32m8_t lof_2 = __riscv_vfwcvt_f_xu_v_f32m8(lo16_2, 32);
    vfloat32m8_t loacc_2 = __riscv_vfmv_v_f_f32m8(mla2, 32);
    vfloat32m8_t lor_2 = __riscv_vfmsac_vf_f32m8(loacc_2, da2, lof_2, 32);
    __riscv_vse32_v_f32m8(yb + 128, lor_2, 32);
    vuint8m2_t nhi_2 = __riscv_vsrl_vx_u8m2(qv_2, 4, 32);
    vuint16m4_t hi16_2 = __riscv_vzext_vf2_u16m4(nhi_2, 32);
    vfloat32m8_t hif_2 = __riscv_vfwcvt_f_xu_v_f32m8(hi16_2, 32);
    vfloat32m8_t hiacc_2 = __riscv_vfmv_v_f_f32m8(mlb2, 32);
    vfloat32m8_t hir_2 = __riscv_vfmsac_vf_f32m8(hiacc_2, db2, hif_2, 32);
    __riscv_vse32_v_f32m8(yb + 160, hir_2, 32);
    const uint8_t* q_3 = v8 + 112;
    vuint8m2_t qv_3 = __riscv_vle8_v_u8m2(q_3, 32);
    vuint8m2_t nlo_3 = __riscv_vand_vx_u8m2(qv_3, 15, 32);
    vuint16m4_t lo16_3 = __riscv_vzext_vf2_u16m4(nlo_3, 32);
    vfloat32m8_t lof_3 = __riscv_vfwcvt_f_xu_v_f32m8(lo16_3, 32);
    vfloat32m8_t loacc_3 = __riscv_vfmv_v_f_f32m8(mla3, 32);
    vfloat32m8_t lor_3 = __riscv_vfmsac_vf_f32m8(loacc_3, da3, lof_3, 32);
    __riscv_vse32_v_f32m8(yb + 192, lor_3, 32);
    vuint8m2_t nhi_3 = __riscv_vsrl_vx_u8m2(qv_3, 4, 32);
    vuint16m4_t hi16_3 = __riscv_vzext_vf2_u16m4(nhi_3, 32);
    vfloat32m8_t hif_3 = __riscv_vfwcvt_f_xu_v_f32m8(hi16_3, 32);
    vfloat32m8_t hiacc_3 = __riscv_vfmv_v_f_f32m8(mlb3, 32);
    vfloat32m8_t hir_3 = __riscv_vfmsac_vf_f32m8(hiacc_3, db3, hif_3, 32);
    __riscv_vse32_v_f32m8(yb + 224, hir_3, 32);
  }
}
