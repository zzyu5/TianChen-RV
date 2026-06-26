// A structurally non-conforming ggml q4_K super-block block-dot source: the
// operator-identity signature is wrong (the q8_K activation operand is an f32
// memref, not the i8 memref the vec_dot identity requires). The front door must
// REJECT it fail-closed (I7), not silently construct a super-block block-dot op
// around a mismatched ABI.
module attributes {tcrv_rvv.source_front_door = "ggml_q4_K_q8_K_block_dot_source"} {
  func.func @source_q4_K_q8_K_bad(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xf32>) {
    return
  }
}
