// W2 §(2) -- F25 baked-g INVARIANCE judgment experiment (reduction path).
//
// F25 = selectIntegerCoreLMUL: its signature eats ONLY c (march -> minimum_vLEN);
// the geometry descriptor g (anchor {m1,m2}, quantFormat "plain-int8",
// blockLen = kContractionBlockLen = 32, factorCap = 1) is BAKED inline. The census
// flagged this as a potential facade ("baked g = a frozen free variable?").
//
// VERDICT: baked g is 恒真 (invariant) for the reduction front door's format set,
// and this is a MECHANICAL tie, not a coincidence: the SAME constant
// kContractionBlockLen (32) that parameterizes the descriptor ALSO gates the
// matcher's accept predicate (isStaticBlockVector: dimSize(0) == kContractionBlockLen).
// So a source whose block length is NOT 32 is REJECTED fail-closed BEFORE
// selectIntegerCoreLMUL ever runs -- g cannot vary within the accepted format set,
// so it is NOT a frozen free variable (NOT a facade), it is a structural constant.
//
// This is the g-axis half of the F25 judgment experiment; the c-axis half (θ =
// integer_core_lmul flips m2@VLEN128 -> m1@VLEN256 with the march fact) is proven
// by rvv-widening-dot-reduce-source-front-door.mlir. Together they show F25 is a
// LEGITIMATE single-side (c-only) selector, not a facade: c is the real load-bearing
// input, and g is provably non-varying (so no g-promotion is warranted).
//
// The "change the baked input" experiment for g: flip the block length 32 -> 16.
// The correct falsifier outcome is NOT "θ unchanged" but "input REJECTED" -- which
// is exactly what a structurally-baked (non-free) g must produce.

// RUN: weft-opt %s --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv --verify-diagnostics --split-input-file

// >> g = 32 (the ONLY accepted block length) is accepted: the front door
//      constructs a body (no diagnostic). Asserted positively elsewhere; here we
//      only need it to NOT error, so verify-diagnostics passes with no expected-error.
module attributes {weft_rvv.source_front_door = "bounded_widening_dot_reduce_source"} {
  func.func @accepted_k32(%lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xi32>, %acc: memref<?xi32>, %n: index) {
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

// -----

// >> g = 16 (block length flipped off the baked 32): REJECTED fail-closed. The
//      SAME constant kContractionBlockLen that the descriptor bakes also gates the
//      matcher, so g cannot be varied -- proving baked g is 恒真, not a facade.
module attributes {weft_rvv.source_front_door = "bounded_widening_dot_reduce_source"} {
  func.func @rejected_k16(%lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xi32>, %acc: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i8
    %seed = memref.load %acc[%c0] : memref<?xi32>
    // expected-error@+1 {{bounded RVV widening-dot-reduce source front door failed: source transfer_read must yield a static vector<32xi8>}}
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<16xi8>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<16xi8>
    %ae = arith.extsi %a : vector<16xi8> to vector<16xi32>
    %be = arith.extsi %b : vector<16xi8> to vector<16xi32>
    %p = arith.muli %ae, %be : vector<16xi32>
    %r = vector.multi_reduction <add>, %p, %seed [0] : vector<16xi32> to i32
    memref.store %r, %out[%c0] : memref<?xi32>
    return
  }
}
