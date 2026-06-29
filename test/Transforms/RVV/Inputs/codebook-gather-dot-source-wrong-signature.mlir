// FAIL-CLOSED negative data for the codebook-gather integer-core front door: a
// source carrying the marker but a WRONG (4-arg codebook KERNEL-shaped) signature,
// NOT the 6-role codebook integer-core operator identity (weight/qlo/qhi
// memref<?xi8>, acc/out memref<?xi32>, n index). The codebook front door must
// REJECT this (fail-closed, I7). Referenced via %S/Inputs by
// rvv-codebook-gather-dot-source-front-door.mlir; not a standalone lit test
// (Inputs/lit.local.cfg sets config.suffixes = []).
module attributes {tcrv_rvv.source_front_door = "bounded_codebook_gather_dot_source"} {
  func.func @source_wrong_signature(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>) {
    return
  }
}
