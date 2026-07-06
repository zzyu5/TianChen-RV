#include <stdint.h>
extern "C" void tcrv_emitc_tq2_0_kernel_scalar_fallback_first_slice(int v1, float* v2, const uint8_t* v3, const int8_t* v4) {
  // tcrv_emitc.route_source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute op_interface=TCRVEmitCLowerableOpInterface
  // tcrv_emitc.source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute step=super_block_count
  size_t v5 = (size_t) v1;
  size_t v6 = v5 / 256;
  float v7;
  v7 = 0.0f;
  // tcrv_emitc.source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute step=super_block_loop
  for (size_t v8 = 0; v8 < v6; v8 += 1) {
    size_t v9 = v8 * 66;
    const uint8_t* v10 = v3 + v9;
    size_t v11 = v8 * 292;
    const int8_t* v12 = v4 + v11;
    const int8_t* v13 = v12 + 4;
    int v14;
    v14 = 0;
    // tcrv_emitc.source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute step=plane_group_loop
    for (size_t v15 = 0; v15 < 64; v15 += 32) {
      for (size_t v16 = 0; v16 < 4; v16 += 1) {
        size_t v17 = v16 * 2;
        int v18 = (int) v17;
        for (size_t v19 = 0; v19 < 32; v19 += 1) {
          // tcrv_emitc.source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute step=ternary_decode_mac
          size_t v20 = v15 + v19;
          const uint8_t v21 = v10[v20];
          int v22 = (int) v21;
          int v23 = v22 >> v18;
          int v24 = v23 & 3;
          int v25 = v24 - 1;
          size_t v26 = v15 * 4;
          size_t v27 = v16 * 32;
          size_t v28 = v26 + v27;
          size_t v29 = v28 + v19;
          const int8_t v30 = v13[v29];
          int v31 = (int) v30;
          int v32 = v31 * v25;
          int v33 = v14;
          int v34 = v33 + v32;
          v14 = v34;
        }
      }
    }
    // tcrv_emitc.source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute step=fold_activation_d
    const float* v35 = (const float*) v12;
    const float v36 = v35[0];
    // tcrv_emitc.source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute step=fold_weight_d
    const uint8_t* v37 = v10 + 64;
    float v38 = (float)*(const _Float16 *)(v37);
    float v39 = v36 * v38;
    // tcrv_emitc.source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute step=scalar_fold
    int v40 = v14;
    float v41 = v7;
    v7 = v41 + (float) v40 * v39;
  }
  // tcrv_emitc.source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute step=store_s
  float v42 = v7;
  v2[0] = v42;
  return;
}


