// FAIL-CLOSED negative data for the packed-i4 nibble integer-core front door: a
// source carrying the marker but a WRONG (4-arg q4_0 KERNEL-shaped) signature, NOT
// the 6-role nibble integer-core operator identity (weight/qlo/qhi memref<?xi8>,
// acc/out memref<?xi32>, n index). The packed-i4 front door must REJECT this
// (fail-closed, I7). Referenced via %S/Inputs by
// rvv-packed-i4-offset-binary-dot-source-front-door.mlir; not a standalone lit test
// (Inputs/lit.local.cfg sets config.suffixes = []).
module attributes {tcrv_rvv.source_front_door = "bounded_packed_i4_offset_binary_dot_source"} {
  func.func @source_wrong_signature(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}
