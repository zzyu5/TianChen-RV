// Track B auto-lowering, the DEQUANT rung -- one step ABOVE the bare-dot MVP
// (rvv-widening-dot-reduce-source-front-door.mlir). The COMPILER auto-CONSTRUCTS
// the tcrv_rvv RVV-dialect widening int8 dot-reduce body WITH a runtime-f32-scale
// dequant tail from a GENERIC vector-dialect source (vector.transfer_read x2 +
// arith.extsi x2 + arith.muli + vector.multi_reduction <add> + arith.sitofp +
// arith.mulf %scale + f32 memref.store), instead of a per-kernel hand emitter.
// The integer-core LMUL anchor is the RETURN VALUE of the SAME shared block-dot
// schedule authority the MVP uses (enumerateBlockDotShapeCandidates +
// selectGenericSchedule fed deriveMinimumVLEN(march)) -- NOT a hand switch. So
// the SAME generic source emits a BYTE-DIFFERENT RVV body by capability, NOW
// with the i32->f32 dequant fused in:
//   * VLEN128 (rv64gcv):        e8m2 -> i16m4 -> vwredsum_i16m4_i32m1 -> dequant
//   * VLEN256 (rv64gcv_zvl256b): e8m1 -> i16m2 -> vwredsum_i16m2_i32m1 -> dequant
//
// This proves the auto-lowering path SCALES from a bare integer dot-reduce to a
// dot+dequant contraction (the q8_0-style integer core + ONE runtime f32 scale),
// still capability-fact-driven, still from a generic vector source with NO per-
// kernel emitter. The generated EmitC is byte-identical to the hand-authored
// dequant body (same isLowPrecisionDequantBody sink), differing only in the
// kernel symbol name.
//
// HONEST SCOPE: this is the single-runtime-f32-scale signed-i8 dot dequant
// contraction; it is NOT the per-block fp16-scale ggml q8_0_q8_0 block-dot
// KERNEL (no nb = n / QK block loop, no per-block d_x . d_y reads). It is the
// second auto-lowered block; the per-block fp16-scale loop is a later rung.

// The auto-constructed dequant body (the e8m2 VLEN128 anchor).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv | FileCheck %s --check-prefix=BODY
//
// The capability-FLIP, asserted on the emitted INTRINSICS (not a metadata mirror):
// the SAME generic source emits e8m2/i16m4 at VLEN128 and e8m1/i16m2 at VLEN256,
// both ending in the i32->f32 runtime-scale dequant epilogue.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN128
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=VLEN256
//
// FAIL-CLOSED (I7): a structurally non-conforming source -- the missing-dequant-tail
// negative module below (a bare i32 dot store, NO sitofp/mulf %scale) -- is REJECTED,
// not silently lowered: the dequant front door requires the full sitofp + mulf scale
// + f32 store tail (and the 6-arg dequant signature).
// RUN: not tcrv-opt %S/Inputs/widening-dot-reduce-dequantize-source-no-dequant-tail.mlir --tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv 2>&1 | FileCheck %s --check-prefix=NODEQUANT

module attributes {tcrv_rvv.source_front_door = "bounded_widening_dot_reduce_dequantize_source"} {
  func.func @source_dequant_dot(%lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xf32>, %acc: memref<?xi32>, %scale: f32, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i8
    %seed = memref.load %acc[%c0] : memref<?xi32>
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<32xi8>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<32xi8>
    %ae = arith.extsi %a : vector<32xi8> to vector<32xi32>
    %be = arith.extsi %b : vector<32xi8> to vector<32xi32>
    %p = arith.muli %ae, %be : vector<32xi32>
    %r = vector.multi_reduction <add>, %p, %seed [0] : vector<32xi32> to i32
    %f = arith.sitofp %r : i32 to f32
    %s = arith.mulf %f, %scale : f32
    memref.store %s, %out[%c0] : memref<?xf32>
    return
  }
}

// ===================== AUTO-CONSTRUCTED tcrv_rvv DEQUANT BODY ===============
// The generic vector source becomes a tcrv.exec.kernel with the auto-built
// load/widening_product/standalone_reduce/DEQUANTIZE/store dequant body. NO
// per-kernel emitter authored this -- the front-door matcher constructed it from
// the VLEN128 gearbox-selected m2 byte anchor, adding the runtime-f32-scale
// dequant on top of the MVP's bare-dot body.
// BODY: tcrv.exec.kernel @rvv_widening_dot_reduce_dequantize_i8_from_vector_source
// BODY: tcrv.exec.variant @rvv_widening_dot_reduce_dequantize_i8
// The runtime f32 dequant scale ABI value (c_type "float", role dequant-scale-value).
// BODY: tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:scale", role = "dequant-scale-value"}
// BODY: tcrv_rvv.setvl
// BODY-SAME: lmul = "m2"
// BODY-SAME: sew = 8
// BODY: tcrv_rvv.load
// BODY-SAME: -> !tcrv_rvv.vector<i8, "m2">
// BODY: tcrv_rvv.load
// BODY-SAME: -> !tcrv_rvv.vector<i8, "m2">
// BODY: tcrv_rvv.widening_product
// BODY-SAME: signed-i8m2xi8m2-to-i16m4
// BODY-SAME: -> !tcrv_rvv.vector<i16, "m4">
// BODY: tcrv_rvv.standalone_reduce
// BODY-SAME: kind = "signed_widening_reduce_add"
// BODY-SAME: -> !tcrv_rvv.vector<i32, "m1">
// BODY: tcrv_rvv.dequantize
// BODY-SAME: dequant_relation = "signed-i32m1-to-f32m1-scale-f32"
// BODY-SAME: kind = "i32_to_f32_scaled"
// BODY-SAME: -> !tcrv_rvv.vector<f32, "m1">
// BODY: tcrv_rvv.store
// The conservative fallback is authored by the fallback-owning plugin.
// BODY: tcrv.exec.variant @rvv_widening_dot_reduce_dequantize_i8_scalar_fallback
// BODY-SAME: fallback_role = "conservative"
// BODY: tcrv.exec.case @rvv_widening_dot_reduce_dequantize_i8
// BODY: tcrv.exec.fallback @rvv_widening_dot_reduce_dequantize_i8_scalar_fallback

// ===================== VLEN128 (rv64gcv) -- the e8m2 anchor + dequant ========
// The gearbox selects the m2 byte anchor at the VLEN-128 tier -- vsetvl_e8m2, i8m2
// loads, the i16m4 widening product, the vwredsum_i16m4_i32m1 reduce, then the
// i32->f32 runtime-scale dequant epilogue (vfmv -> store f32).
// VLEN128: emitc.func @tcrv_emitc_rvv_widening_dot_reduce_dequantize_i8_from_vector_source_rvv_widening_dot_reduce_dequantize_i8(
// VLEN128: call_opaque "__riscv_vsetvl_e8m2"
// VLEN128-NOT: call_opaque "__riscv_vsetvl_e8m1"
// VLEN128: call_opaque "__riscv_vle8_v_i8m2"
// VLEN128: call_opaque "__riscv_vwmul_vv_i16m4"
// VLEN128: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN128-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN128: call_opaque "__riscv_vfmv_v_f_f32m1"
// VLEN128: call_opaque "__riscv_vse32_v_f32m1"
// VLEN128: return

// ===================== VLEN256 (rv64gcv_zvl256b) -- the FLIP to e8m1 =========
// The REAL VLEN fact FLIPS the anchor to m1: a BYTE-DIFFERENT kernel. vsetvl_e8m1
// (NOT e8m2), i8m1 loads, the i16m2 product (NOT i16m4), the vwredsum_i16m2_i32m1
// reduce -- ending in the SAME i32->f32 runtime-scale dequant epilogue.
// VLEN256: emitc.func @tcrv_emitc_rvv_widening_dot_reduce_dequantize_i8_from_vector_source_rvv_widening_dot_reduce_dequantize_i8(
// VLEN256: call_opaque "__riscv_vsetvl_e8m1"
// VLEN256-NOT: call_opaque "__riscv_vsetvl_e8m2"
// VLEN256: call_opaque "__riscv_vle8_v_i8m1"
// VLEN256: call_opaque "__riscv_vwmul_vv_i16m2"
// VLEN256: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// VLEN256-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// VLEN256: call_opaque "__riscv_vfmv_v_f_f32m1"
// VLEN256: call_opaque "__riscv_vse32_v_f32m1"
// VLEN256: return

// ===================== FAIL-CLOSED diagnostics (I7) =========================
// The bare-dot (no-dequant-tail) source is rejected: this front door requires the
// sitofp + mulf %scale + f32 store dequant tail (and the 6-arg dequant signature).
// NODEQUANT: bounded RVV widening-dot-reduce dequantize source front door failed
