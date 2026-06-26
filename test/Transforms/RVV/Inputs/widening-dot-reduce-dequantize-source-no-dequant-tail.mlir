// FAIL-CLOSED negative data for the dequant front door: the bare-dot tail with NO
// dequant (i32 store, no sitofp/mulf %scale, 5-arg signature). The dequant front
// door must REJECT this (fail-closed, I7) -- it requires the full sitofp + mulf
// scale + f32 store tail and the 6-arg dequant signature. Referenced via %S/Inputs
// by rvv-widening-dot-reduce-dequantize-source-front-door.mlir; not a standalone
// lit test (Inputs/lit.local.cfg sets config.suffixes = []).
module attributes {tcrv_rvv.source_front_door = "bounded_widening_dot_reduce_dequantize_source"} {
  func.func @source_no_dequant_tail(%lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xi32>, %acc: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i8
    %seed = memref.load %acc[%c0] : memref<?xi32>
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<32xi8>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<32xi8>
    %ae = arith.extsi %a : vector<32xi8> to vector<32xi32>
    %be = arith.extsi %b : vector<32xi8> to vector<32xi32>
    %p = arith.muli %ae, %be : vector<32xi32>
    %r = vector.multi_reduction <add>, %p, %seed [0] : vector<32xi32> to i32
    memref.store %r, %out[%c0] : memref<?xi32>
    return
  }
}
