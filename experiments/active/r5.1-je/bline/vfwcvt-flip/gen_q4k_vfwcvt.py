#!/usr/bin/env python3
# q4_K dequant leaf: keep the 32-lane m8 f32 stage, but shrink the widen chain:
#   OLD: vzext.vf4 (u8m2->u32m8, 4x widen) + vreinterpret + vfcvt.f.x.v  [3 ops]
#   NEW: vzext.vf2 (u8m2->u16m4, 2x widen) + vfwcvt.f.xu.v (u16m4->f32m8) [2 ops, cheaper widen]
# Byte-exact: nibble is 0..15 (<2^31) so unsigned vfwcvt == the old signed vfcvt path.
L=[]
def w(s): L.append(s)
w('#include <stddef.h>'); w('#include <stdint.h>'); w('#include <riscv_vector.h>')
w('extern "C" void weft_emitc_dequant_q4_K_kernel_dequant_q4_K(size_t v1, const uint8_t* v2, float* v3) {')
w('  size_t nb = v1 / 256;')
w('  for (size_t ib = 0; ib < nb; ib += 1) {')
w('    const uint8_t* v8 = v2 + ib * 144;')
w('    float* yb = v3 + ib * 256;')
w('    float d    = (float)*(const _Float16 *)(v8);')
w('    float dmin = (float)*(const _Float16 *)(v8 + 2);')
w('    const uint8_t* sc = v8 + 4;')
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
for j in range(4):
    qoff=16+j*32
    ptok='v8 + 16;' if j==0 else f'v8 + {qoff};'
    w(f'    const uint8_t* q_{j} = {ptok}')
    w(f'    vuint8m2_t qv_{j} = __riscv_vle8_v_u8m2(q_{j}, 32);')
    # low nibble -> y[j*64 : +32]
    ylo=j*64
    w(f'    vuint8m2_t nlo_{j} = __riscv_vand_vx_u8m2(qv_{j}, 15, 32);')
    w(f'    vuint16m4_t lo16_{j} = __riscv_vzext_vf2_u16m4(nlo_{j}, 32);')
    w(f'    vfloat32m8_t lof_{j} = __riscv_vfwcvt_f_xu_v_f32m8(lo16_{j}, 32);')
    w(f'    vfloat32m8_t loacc_{j} = __riscv_vfmv_v_f_f32m8(mla{j}, 32);')
    w(f'    vfloat32m8_t lor_{j} = __riscv_vfmsac_vf_f32m8(loacc_{j}, da{j}, lof_{j}, 32);')
    w(f'    __riscv_vse32_v_f32m8(yb + {ylo}, lor_{j}, 32);')
    # high nibble -> y[j*64+32 : +32]
    yhi=j*64+32
    w(f'    vuint8m2_t nhi_{j} = __riscv_vsrl_vx_u8m2(qv_{j}, 4, 32);')
    w(f'    vuint16m4_t hi16_{j} = __riscv_vzext_vf2_u16m4(nhi_{j}, 32);')
    w(f'    vfloat32m8_t hif_{j} = __riscv_vfwcvt_f_xu_v_f32m8(hi16_{j}, 32);')
    w(f'    vfloat32m8_t hiacc_{j} = __riscv_vfmv_v_f_f32m8(mlb{j}, 32);')
    w(f'    vfloat32m8_t hir_{j} = __riscv_vfmsac_vf_f32m8(hiacc_{j}, db{j}, hif_{j}, 32);')
    w(f'    __riscv_vse32_v_f32m8(yb + {yhi}, hir_{j}, 32);')
w('  }'); w('}')
import sys; sys.stdout.write('\n'.join(L)+'\n')
