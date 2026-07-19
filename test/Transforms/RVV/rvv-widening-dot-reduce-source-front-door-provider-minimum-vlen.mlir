// W2 §(1) B -- source-front-door PROVIDER PRODUCTION judgment experiment.
//
// Census: the RVV source front doors CONSTRUCT their own weft.exec.capability @rvv
// provider op but historically left it EMPTY (only id/kind/status), so every
// downstream resolveRVVMinimumVLEN fell back to re-parsing -march -- the provider
// was a hollow mirror and the probe->provider producer pass was wired into NO
// pipeline (its only caller was a test). Verdict for all 5 front doors: B (未 stamp
// = 产空 provider = 真债).
//
// Remedy (给探测层挂生产驱动): the front door now calls the ONE shared producer
// (materializeRVVProviderCapabilityAxes) right after it constructs the body, so the
// constructed provider CARRIES the c facts. This experiment proves it: a source
// input with NO provider stamp compiles to a provider op that carries a TYPED i64
// minimum_vlen, and that fact FLIPS with the march capability (128 vs 256) -- so a
// downstream consumer reads the provider fact, not a local -march re-parse (I1/I4).
//
// Byte-exact: the emitted BODY (setvl/load/product/reduce) and the EmitC C source
// are unchanged (the capability op is metadata, I2/I4 -- it does not lower to C);
// only the provider op gains the minimum_vlen (+ support-axis) facts it should have
// carried all along.

// RUN: weft-opt %s --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv | FileCheck %s --check-prefix=V128
// RUN: weft-opt %s --weft-rvv-materialize-widening-dot-reduce-source-front-door=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=V256

module attributes {weft_rvv.source_front_door = "bounded_widening_dot_reduce_source"} {
  func.func @source_widening_dot_reduce(%lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xi32>, %acc: memref<?xi32>, %n: index) {
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

// The constructed RVV provider now carries the TYPED minimum_vlen fact = 128 at the
// rv64gcv (VLEN128) tier.
// V128: weft.exec.capability @rvv
// V128-SAME: minimum_vlen = 128 : i64

// The SAME source, compiled for the zvl256b tier, produces a provider carrying
// minimum_vlen = 256 -- the fact FLIPS with the march capability (production path
// materializes it; downstream reads the provider, not -march).
// V256: weft.exec.capability @rvv
// V256-SAME: minimum_vlen = 256 : i64
