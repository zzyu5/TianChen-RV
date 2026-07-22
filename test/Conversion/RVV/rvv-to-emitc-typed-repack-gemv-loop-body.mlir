// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_repack_gemv_loop_body"/kind = "plain_repack_loop"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "lane_wise_vector_scale"/fold_model = "unsupported_fold"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/half_lanes = 16 : i64/half_lanes = 12 : i64/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADHALF
// RUN: sed 's/repack_lane_wise_q4_x_i8_dot %vx/repack_lane_wise_q4_x_i8_dot %vy/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADBASE
// RUN: sed 's/repack_dual_fp16_scale_fold %vx/repack_dual_fp16_scale_fold %vy/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADFOLDBASE

// M-FLAT REPACK loop-scaffold milestone 3 (numHalves==1 arm) -- the q4_0 16x1-
// REPACKED GEVM's per-block LANE-WISE integer CORE *and* its dual-fp16 per-strip
// scale FOLD are now BOTH carried IN-REGION (the M2 loop-carried fold stub is
// gone). The region-carrying weft_rvv.typed_repack_gemv_loop_body holds the
// integer-core BRICK weft_rvv.repack_lane_wise_q4_x_i8_dot (the per-block seed
// i16 lo/hi -> nibble-step vwmacc loop -> lo/hi vwadd combine) FOLLOWED BY the
// scale-fold BRICK weft_rvv.repack_dual_fp16_scale_fold (per-strip vle16 weight
// scale -> ONE _Float16 activation scale -> vfwmul/vfcvt/vfmacc into the f32
// accumulator), both addressed off the block_index induction variable. The
// lowering GATES the emit on BOTH bricks' block_index-tied anti-bypass (block_index
// == region arg 0, bases == the loop-body's own ABI buffers) PLUS the fold's
// dataflow tie (fold consumes the integer brick's sumi + the loop-carried acc,
// yield names the fold's acc_next), then drives the CORE + FOLD from the SHARED
// emitRepackQ4LaneWiseIntegerCore / emitRepackDualFp16ScaleFold leaves -- BYTE-
// EXACT to the monolithic emitRepackGemvQ4_0Q8_0 (vle8 / sign-extension decode /
// scalar q8 quant reads / lane-wise vwmacc lo+hi / lo/hi vwadd combine / vle16
// weight scale / _Float16 act scale / vfwmul / vfcvt / vfmacc). This is the
// numHalves==1 (VLEN=256 fractional mf2) arm; Phase B generalizes the region to
// FULL-BODY byte-exact on EVERY arm -- the VLEN=128 two-8-lane-halves
// (numHalves==2) multi-accumulator arm and the RVV0.7 whole-LMUL m1/f32m4 arm are
// pinned by the EMPIRICAL region-vs-monolith byte-diff lits
// rvv-to-emitc-typed-repack-gemv-loop-body-vlen128-full-body.mlir /
// -rvv07-full-body.mlir. Monolith retirement + front-door construction are later
// steps. Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv), not tested here.
//
// This is an emit-consistency ("CORE == emission-plans") lit that locks the loop-
// nest + per-strip VECTOR accumulator skeleton AND the region-driven integer core
// + scale fold byte-exact to the monolithic emitRepackGemvQ4_0Q8_0 one-strip
// (VLEN=256 / RVV0.7) form. Numerical bit-exact-vs-ggml is pending-hardware (ssh
// rvv), not tested here.

module {
  weft.exec.kernel @rvv_typed_repack_gemv_loop_body_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_typed_repack_gemv_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 16 : i64, integer_core_lmul = "mf2", fold_model = "lane_wise_vector_scale"} {
        ^bb0(%block_index: index, %acc: !weft_rvv.vector<f32, "m2">):
          // M3 region body: the block_index-tied integer CORE brick (per-block
          // lane-wise nibble dot -> per-strip i32 sumi), THEN the dual-fp16 per-
          // strip scale FOLD brick that consumes the integer brick's sumi AND the
          // loop-carried acc and produces the folded-out acc_next the yield names
          // (the M2 loop-carried stub is gone -- the fold now mutates the acc).
          %sumi = weft_rvv.repack_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %block_index : index {kind = "repack_lane_wise_q4_x_i8_dot", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">
          %acc_next = weft_rvv.repack_dual_fp16_scale_fold %vx, %vy, %sumi, %acc, %vl block %block_index : index {kind = "repack_dual_fp16_scale_fold", weight_scale_byte_offset = 0 : i64, activation_scale_byte_offset = 0 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<f32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc_next : !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_rvv_typed_repack_gemv_loop_body_kernel_rvv_typed_repack_gemv_loop_body(

// nb = n / QK; nc_groups = nc / weight_interleave (the two structural divides).
// CHECK: %[[NB:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[NCG:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">

// The OUTER weight-column-GROUP loop over nc/16; per-group weight base (stride 288).
// CHECK: for %[[X:.*]] = %{{.*}} to %[[NCG]] step %{{.*}}  : !emitc.opaque<"size_t"> {
// CHECK: literal "288"

// The ONE 16-lane f32 strip accumulator: the mutable vfloat32m2 emitc.variable the
// SSA loop-carried acc maps to, seeded per group with a single vfmv_v_f_f32m2(0.0f).
// CHECK: %[[SUMF:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: assign %[[SEED]] : !emitc.opaque<"vfloat32m2_t"> to %[[SUMF]] : <!emitc.opaque<"vfloat32m2_t">>

// The inner contraction-BLOCK loop over nb; plain q8_0 activation stride 34.
// CHECK: for %[[L:.*]] = %{{.*}} to %[[NB]] step %{{.*}}  : !emitc.opaque<"size_t"> {
// CHECK: literal "34"

// ===== The region integer CORE (byte-exact to the monolith's integer part) =====
// ONE 16-lane i16 lo + ONE 16-lane i16 hi accumulator seed (NOT four 8-lane).
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The nibble-step loop; ONE disjoint repacked sub-load (rows 0..15 at qs[i*16+0]),
// then the plain sign-extension decode (NO vxor): b_lo=vsra(vsll(b,4),4); b_hi=vsra(b,4).
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsll_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsra_vx_i8mf2"
// TWO scalar q8 quant reads (low qs[i], high qs[16+i]) + TWO lane-wise vwmacc.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// ONE lo/hi combine vwadd_vv_i32m2 (the per-strip i32 sumi).
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// NO offset-binary xor (the repacked nibbles carry the ^0x88 bias already).
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8

// ===== The region dual-fp16 scale FOLD (M3, byte-exact to the monolith's fold),
// sitting right AFTER the integer core: per-strip vle16 weight scale -> ONE
// _Float16 activation scale -> vfwmul (d = d_x*d_y) -> vfcvt (sumi -> f32) ->
// load acc -> vfmacc -> assign back. The carried-OUT acc IS the fold result. =====
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: %[[ACCIN:.*]] = load %[[SUMF]] : <!emitc.opaque<"vfloat32m2_t">>
// CHECK: %[[ACCNEXT:.*]] = call_opaque "__riscv_vfmacc_vv_f32m2"(%[[ACCIN]],
// CHECK: assign %[[ACCNEXT]] : !emitc.opaque<"vfloat32m2_t"> to %[[SUMF]] : <!emitc.opaque<"vfloat32m2_t">>

// The per-strip lane-wise vector store (NO horizontal reduction): the final strip
// loaded from sumf and written straight through s + x*16 with vse32.
// CHECK: %[[FINAL:.*]] = load %[[SUMF]] : <!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The bounded surface is fail-closed on the loop kind, the fold_model fact, and
// the resource-aware strip width (I7). The per-strip f32 VECTOR loop-carried
// accumulator dtype is enforced by the verifier; BOTH bricks' block_index-tied
// anti-bypass is exercised by the positive path + the BADBASE negative (rewiring
// the integer brick's weight base) and the BADFOLDBASE negative (rewiring the
// fold brick's weight base to a foreign buffer fails closed).
// BADKIND: currently supports only kind "typed_repack_gemv_loop_body"
// BADFOLD: currently supports only fold_model "lane_wise_vector_scale"
// BADHALF: requires half_lanes in {8, 16} dividing weight_interleave
// BADBASE: failed to legalize operation 'weft.exec.variant'
// BADFOLDBASE: failed to legalize operation 'weft.exec.variant'
