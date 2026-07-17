#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" double tcrv_emitc_ggml_vec_soft_max_f32_kernel_ggml_vec_soft_max_f32(size_t v1, float* v2, const float* v3, float v4) {
  // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_rvv.setvl role=configure op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m1
  size_t v5 = __riscv_vsetvl_e32m1(v1);
  // tcrv_emitc.route_source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.local_variable=vsum source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
  vfloat64m1_t v6;
  // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f64m1
  vfloat64m1_t v7 = __riscv_vfmv_v_f_f64m1(0.0, 1);
  v6 = v7;
  // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m2
  size_t v8 = __riscv_vsetvl_e32m2(v1);
  for (size_t v9 = 0; v9 < v1; v9 += v8) {
    size_t v10 = v1 - v9;
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsetvl_e32m2
    size_t v11 = __riscv_vsetvl_e32m2(v10);
    const float* v12 = v3 + v9;
    const float* v13 = (const float*) v12;
    float* v14 = v2 + v9;
    float* v15 = (float*) v14;
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vle32_v_f32m2
    vfloat32m2_t v16 = __riscv_vle32_v_f32m2(v13, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfsub_vf_f32m2
    vfloat32m2_t v17 = __riscv_vfsub_vf_f32m2(v16, v4, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v18 = __riscv_vfmv_v_f_f32m2(0x1.8p23f, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vf_f32m2
    vfloat32m2_t v19 = __riscv_vfmacc_vf_f32m2(v18, 0x1.715476p+0f, v17, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfsub_vv_f32m2
    vfloat32m2_t v20 = __riscv_vfsub_vv_f32m2(v19, v18, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfnmsac_vf_f32m2
    vfloat32m2_t v21 = __riscv_vfnmsac_vf_f32m2(v17, 0x1.62e4p-1f, v20, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfnmsac_vf_f32m2
    vfloat32m2_t v22 = __riscv_vfnmsac_vf_f32m2(v21, 0x1.7f7d1cp-20f, v20, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_f32m2_u32m2
    vuint32m2_t v23 = __riscv_vreinterpret_v_f32m2_u32m2(v19);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsll_vx_u32m2
    vuint32m2_t v24 = __riscv_vsll_vx_u32m2(v23, 23, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u32m2
    vuint32m2_t v25 = __riscv_vadd_vx_u32m2(v24, 0x3f800000, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m2_f32m2
    vfloat32m2_t v26 = __riscv_vreinterpret_v_u32m2_f32m2(v25);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfabs_v_f32m2
    vfloat32m2_t v27 = __riscv_vfabs_v_f32m2(v20, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmfgt_vf_f32m2_b16
    vbool16_t v28 = __riscv_vmfgt_vf_f32m2_b16(v27, 126.0f, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vv_f32m2
    vfloat32m2_t v29 = __riscv_vfmul_vv_f32m2(v22, v22, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vf_f32m2
    vfloat32m2_t v30 = __riscv_vfmul_vf_f32m2(v22, 0x1.ffffecp-1f, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v31 = __riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vf_f32m2
    vfloat32m2_t v32 = __riscv_vfmacc_vf_f32m2(v31, 0x1.555e66p-3f, v22, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_v_f_f32m2
    vfloat32m2_t v33 = __riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vf_f32m2
    vfloat32m2_t v34 = __riscv_vfmacc_vf_f32m2(v33, 0x1.0e4020p-7f, v22, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
    vfloat32m2_t v35 = __riscv_vfmacc_vv_f32m2(v32, v34, v29, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
    vfloat32m2_t v36 = __riscv_vfmacc_vv_f32m2(v30, v35, v29, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmfle_vf_f32m2_b16
    vbool16_t v37 = __riscv_vmfle_vf_f32m2_b16(v20, 0.0f, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmv_v_x_u32m2
    vuint32m2_t v38 = __riscv_vmv_v_x_u32m2(0, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vxm_u32m2
    vuint32m2_t v39 = __riscv_vmerge_vxm_u32m2(v38, 0x82000000, v37, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vx_u32m2
    vuint32m2_t v40 = __riscv_vadd_vx_u32m2(v39, 0x7f000000, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m2_f32m2
    vfloat32m2_t v41 = __riscv_vreinterpret_v_u32m2_f32m2(v40);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vsub_vv_u32m2
    vuint32m2_t v42 = __riscv_vsub_vv_u32m2(v24, v39, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vreinterpret_v_u32m2_f32m2
    vfloat32m2_t v43 = __riscv_vreinterpret_v_u32m2_f32m2(v42);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
    vfloat32m2_t v44 = __riscv_vfmacc_vv_f32m2(v26, v26, v36, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmacc_vv_f32m2
    vfloat32m2_t v45 = __riscv_vfmacc_vv_f32m2(v43, v43, v36, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vv_f32m2
    vfloat32m2_t v46 = __riscv_vfmul_vv_f32m2(v45, v41, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m2
    vfloat32m2_t v47 = __riscv_vmerge_vvm_f32m2(v44, v46, v28, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfabs_v_f32m2
    vfloat32m2_t v48 = __riscv_vfabs_v_f32m2(v20, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmfgt_vf_f32m2_b16
    vbool16_t v49 = __riscv_vmfgt_vf_f32m2_b16(v48, 192.0f, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmul_vv_f32m2
    vfloat32m2_t v50 = __riscv_vfmul_vv_f32m2(v41, v41, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vmerge_vvm_f32m2
    vfloat32m2_t v51 = __riscv_vmerge_vvm_f32m2(v47, v50, v49, v11);
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vse32_v_f32m2
    __riscv_vse32_v_f32m2(v15, v51, v11);
    vfloat64m1_t v52 = v6;
    // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfwredusum_vs_f32m2_f64m1
    vfloat64m1_t v53 = __riscv_vfwredusum_vs_f32m2_f64m1(v51, v52, v11);
    // tcrv_emitc.assign target=vsum source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface
    v6 = v53;
  }
  vfloat64m1_t v54 = v6;
  // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vfmv_f_s_f64m1_f64
  double v55 = __riscv_vfmv_f_s_f64m1_f64(v54);
  // tcrv_emitc.source_op=tcrv_rvv.ggml_vec_soft_max_f32 role=compute op_interface=TCRVEmitCLowerableOpInterface callee=return
  double v56 = (double) v55;
  return v56;
}


