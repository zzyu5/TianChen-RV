#!/usr/bin/env python3
# q5_K dequant leaf with the vfwcvt widen (same lever as q4_K). 5th-bit ORed into the
# u8 plane (range 0..31) BEFORE vzext.vf2 -> vfwcvt.f.xu.v (u16m4 -> f32m8). Byte-exact.
L=[]
def w(s): L.append(s)
w('#include <stddef.h>'); w('#include <stdint.h>'); w('#include <riscv_vector.h>')
w('extern "C" void weft_emitc_dequant_q5_K_kernel_dequant_q5_K(size_t v1, const uint8_t* v2, float* v3) {')
w('  size_t nb = v1 / 256;')
w('  for (size_t ib = 0; ib < nb; ib += 1) {')
w('    const uint8_t* v8 = v2 + ib * 176;')
w('    float* yb = v3 + ib * 256;')
w('    float d    = (float)*(const _Float16 *)(v8);')
w('    float dmin = (float)*(const _Float16 *)(v8 + 2);')
w('    const uint8_t* sc = v8 + 4;')
w('    const uint8_t* qh = v8 + 16;')
w('    vuint8m2_t qhv = __riscv_vle8_v_u8m2(qh, 32);')
for j in range(4):
    for tag, jj in (('a', 2*j), ('b', 2*j+1)):
        if jj < 4:
            scv=f'(sc[{jj}] & 63)'; mv=f'(sc[{jj+4}] & 63)'
        else:
            scv=f'((sc[{jj+4}] & 0x0F) | ((sc[{jj-4}] >> 6) << 4))'
            mv =f'((sc[{jj+4}] >> 4) | ((sc[{jj}] >> 6) << 4))'
        w(f'    float d{tag}{j}  = d    * (float){scv};')
        w(f'    float ml{tag}{j} = dmin * (float){mv};')
w('')
def half(j, hi, bitshift, dsc, mlsc, ystore, plane_src):
    p=f'{"hi" if hi else "lo"}_{j}'
    if hi:
        w(f'    vuint8m2_t nib_{p} = __riscv_vsrl_vx_u8m2({plane_src}, 4, 32);')
    else:
        w(f'    vuint8m2_t nib_{p} = __riscv_vand_vx_u8m2({plane_src}, 15, 32);')
    w(f'    vuint8m2_t sh_{p} = __riscv_vsrl_vx_u8m2(qhv, {bitshift}, 32);')
    w(f'    vuint8m2_t bit_{p} = __riscv_vand_vx_u8m2(sh_{p}, 1, 32);')
    w(f'    vuint8m2_t hb_{p} = __riscv_vsll_vx_u8m2(bit_{p}, 4, 32);')
    w(f'    vuint8m2_t plane_{p} = __riscv_vor_vv_u8m2(nib_{p}, hb_{p}, 32);')
    w(f'    vuint16m4_t w16_{p} = __riscv_vzext_vf2_u16m4(plane_{p}, 32);')
    w(f'    vfloat32m8_t f_{p} = __riscv_vfwcvt_f_xu_v_f32m8(w16_{p}, 32);')
    w(f'    vfloat32m8_t acc_{p} = __riscv_vfmv_v_f_f32m8({mlsc}, 32);')
    w(f'    vfloat32m8_t r_{p} = __riscv_vfmsac_vf_f32m8(acc_{p}, {dsc}, f_{p}, 32);')
    w(f'    __riscv_vse32_v_f32m8(yb + {ystore}, r_{p}, 32);')
for j in range(4):
    qoff=48+j*32
    ptok='v8 + 48;' if j==0 else f'v8 + {qoff};'
    w(f'    const uint8_t* qsp_{j} = {ptok}')
    w(f'    vuint8m2_t qv_{j} = __riscv_vle8_v_u8m2(qsp_{j}, 32);')
    half(j, False, 2*j,   f'da{j}', f'mla{j}', j*64,    f'qv_{j}')
    half(j, True,  2*j+1, f'db{j}', f'mlb{j}', j*64+32, f'qv_{j}')
w('  }'); w('}')
import sys; sys.stdout.write('\n'.join(L)+'\n')
