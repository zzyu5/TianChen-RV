// A structurally non-conforming ggml iq4_nl codebook block-dot source: the
// operator-identity signature is wrong (the q8 activation operand is an f32 memref,
// not the i8 memref the vec_dot identity requires). The front door must REJECT it
// fail-closed (I7), not silently construct a codebook block-dot op around a
// mismatched ABI.
module attributes {tcrv_rvv.source_front_door = "ggml_iq4_nl_q8_0_block_dot_source"} {
  func.func @source_iq4_nl_q8_0_bad(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xf32>) {
    return
  }
}
