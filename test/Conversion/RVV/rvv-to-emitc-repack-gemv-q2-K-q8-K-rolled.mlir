// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// G7 [GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP] whole-K-nest S6 roll (q2_K family member):
// the SAME q2_K x q8_K 16x1-REPACKED single-output-column GEVM (decode) front-door region
// as rvv-to-emitc-repack-gemv-q2-K-q8-K.mlir, but the loop-body op carries the OPTIONAL
// main_term_form = "rolled" SCHEDULE knob. The ROLLED schedule emits the dominant
// per-16-position inner mm-loop as ONE runtime emitc.for with the per-strip x per-2bit-lane
// i16 partials carried as RESIDENT SSA-register VariableOps (seeded above, load-accumulate-
// store inside). kernel-axis C1 maturity lever, ZERO perf green (roll = I-cache/vsetvli
// lever, NOT a density mover; [CASE-MICRO-E2E]). CAPABILITY-KEYED by the stamp, BYTE-EXACT
// to the unrolled emit by construction. Checks the ROLLED STRUCTURE only; numeric
// correctness rides the SHARED q2_K board oracle (unchanged).

module {
  weft.exec.kernel @ggml_repack_gemv_q2_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_q2_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit", qk = 256 : i64, weight_block_stride = 1344 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 320 : i64, activation_quant_byte_offset = 4 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, activation_bsums_byte_offset = 260 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "kquant_dmin_bsums_min", main_term_form = "rolled"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied q2_K integer-core BRICK: per-block lane-wise q2_K
          // dot -> the numHalves (2) per-strip i32 sumi. The typed emitter re-emits the
          // whole byte-exact q2_K body (2-bit weight peel + 4-bit packed scale/min +
          // dual d/dmin + bsums-min fold) from this brick's identity; the yield passes
          // the accs through.
          %sumi:2 = weft_rvv.repack_gemv_kquant_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_kquant_core", decode_model = "q2_K", weight_quant_byte_offset = 320 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemv_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K(
// The two outer runtime loops -- the weight-column-group loop (nc/16) and the
// per-strip contraction-block loop over nb -- SAME as the unrolled emit.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// The per-strip i16 partial is seeded into a RESIDENT VariableOp (emitc.variable +
// assign of vmv_v_x_i16m1) BEFORE the rolled inner loop -- NOT a materialized tile.
// CHECK: emitc.variable
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// ===== The ROLLED main-term marker: a THIRD (inner) runtime for-loop over the inner
// position group -- the compact per-element decode loop the full-unroll emit lacks.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// Inside it: reload the running i16 partial (load), weight x activation vwmacc_vx,
// assign the partial back (the rolled resident load-accumulate-store, byte-exact to
// the unrolled SSA accumulation).
// CHECK: load %{{.*}} : <!emitc.opaque<"vint16m1_t">>
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: assign %{{.*}} : !emitc.opaque<"vint16m1_t">
// After the rolled loop: promote the final i16 partials to i32 (scale-weighted
// vwmacc_vv) and run the SHARED end-of-block fold (unchanged).
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The MIN-term subtract survives (vfnmsac: sumf -= dmins_d * cvt(bsums)).
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
