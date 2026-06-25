#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_silu_f32_kernel_ggml_vec_silu_f32(size_t v1, const float* v2, float* v3) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v4 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m2
  size_t v5 = __riscv_vsetvl_e32m2(v1);
  for (size_t v6 = 0; v6 < v1; v6 += v5) {
    size_t v7 = v1 - v6;
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m2
    size_t v8 = __riscv_vsetvl_e32m2(v7);
    const float* v9 = v2 + v6;
    const float* v10 = (const float*) v9;
    float* v11 = v3 + v6;
    float* v12 = (float*) v11;
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m2
    vfloat32m2_t v13 = __riscv_vle32_v_f32m2(v10, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfneg_v_f32m2
    vfloat32m2_t v14 = __riscv_vfneg_v_f32m2(v13, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v15 = __riscv_vfmv_v_f_f32m2(0x1.8p23f, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vf_f32m2
    vfloat32m2_t v16 = __riscv_vfmacc_vf_f32m2(v15, 0x1.715476p+0f, v14, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfsub_vv_f32m2
    vfloat32m2_t v17 = __riscv_vfsub_vv_f32m2(v16, v15, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfnmsac_vf_f32m2
    vfloat32m2_t v18 = __riscv_vfnmsac_vf_f32m2(v14, 0x1.62e4p-1f, v17, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfnmsac_vf_f32m2
    vfloat32m2_t v19 = __riscv_vfnmsac_vf_f32m2(v18, 0x1.7f7d1cp-20f, v17, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_f32m2_u32m2
    vuint32m2_t v20 = __riscv_vreinterpret_v_f32m2_u32m2(v16);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u32m2
    vuint32m2_t v21 = __riscv_vsll_vx_u32m2(v20, 23, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u32m2
    vuint32m2_t v22 = __riscv_vadd_vx_u32m2(v21, 0x3f800000, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m2_f32m2
    vfloat32m2_t v23 = __riscv_vreinterpret_v_u32m2_f32m2(v22);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfabs_v_f32m2
    vfloat32m2_t v24 = __riscv_vfabs_v_f32m2(v17, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmfgt_vf_f32m2_b16
    vbool16_t v25 = __riscv_vmfgt_vf_f32m2_b16(v24, 126.0f, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vv_f32m2
    vfloat32m2_t v26 = __riscv_vfmul_vv_f32m2(v19, v19, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v27 = __riscv_vfmul_vf_f32m2(v19, 0x1.ffffecp-1f, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v28 = __riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vf_f32m2
    vfloat32m2_t v29 = __riscv_vfmacc_vf_f32m2(v28, 0x1.555e66p-3f, v19, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v30 = __riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vf_f32m2
    vfloat32m2_t v31 = __riscv_vfmacc_vf_f32m2(v30, 0x1.0e4020p-7f, v19, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
    vfloat32m2_t v32 = __riscv_vfmacc_vv_f32m2(v29, v31, v26, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
    vfloat32m2_t v33 = __riscv_vfmacc_vv_f32m2(v27, v32, v26, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmfle_vf_f32m2_b16
    vbool16_t v34 = __riscv_vmfle_vf_f32m2_b16(v17, 0.0f, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u32m2
    vuint32m2_t v35 = __riscv_vmv_v_x_u32m2(0, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vxm_u32m2
    vuint32m2_t v36 = __riscv_vmerge_vxm_u32m2(v35, 0x82000000, v34, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u32m2
    vuint32m2_t v37 = __riscv_vadd_vx_u32m2(v36, 0x7f000000, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m2_f32m2
    vfloat32m2_t v38 = __riscv_vreinterpret_v_u32m2_f32m2(v37);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vv_u32m2
    vuint32m2_t v39 = __riscv_vsub_vv_u32m2(v21, v36, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m2_f32m2
    vfloat32m2_t v40 = __riscv_vreinterpret_v_u32m2_f32m2(v39);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
    vfloat32m2_t v41 = __riscv_vfmacc_vv_f32m2(v23, v23, v33, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
    vfloat32m2_t v42 = __riscv_vfmacc_vv_f32m2(v40, v40, v33, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vv_f32m2
    vfloat32m2_t v43 = __riscv_vfmul_vv_f32m2(v42, v38, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m2
    vfloat32m2_t v44 = __riscv_vmerge_vvm_f32m2(v41, v43, v25, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfabs_v_f32m2
    vfloat32m2_t v45 = __riscv_vfabs_v_f32m2(v17, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmfgt_vf_f32m2_b16
    vbool16_t v46 = __riscv_vmfgt_vf_f32m2_b16(v45, 192.0f, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vv_f32m2
    vfloat32m2_t v47 = __riscv_vfmul_vv_f32m2(v38, v38, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m2
    vfloat32m2_t v48 = __riscv_vmerge_vvm_f32m2(v44, v47, v46, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfadd_vf_f32m2
    vfloat32m2_t v49 = __riscv_vfadd_vf_f32m2(v48, 1.0f, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfdiv_vv_f32m2
    vfloat32m2_t v50 = __riscv_vfdiv_vv_f32m2(v13, v49, v8);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_silu_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v12, v50, v8);
  }
  return;
}


