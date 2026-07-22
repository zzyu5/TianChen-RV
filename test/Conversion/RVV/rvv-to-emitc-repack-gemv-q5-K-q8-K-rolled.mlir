// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// G7 [GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP] whole-K-nest S6 roll: the SAME q5_K x q8_K
// 16x1-REPACKED single-output-column GEVM (decode) front-door region as
// rvv-to-emitc-repack-gemv-q5-K-q8-K.mlir, but the loop-body op carries the OPTIONAL
// main_term_form = "rolled" SCHEDULE knob. Where the default (unrolled) emit fully
// STATIC-unrolls the per-super-half / per-pair / per-k / per-16-element decode nest (8751
// static v-insn, 2943 vsetvli, 209 whole-reg vs*r.v tile-roundtrip spills @VLEN128), the
// ROLLED schedule emits the dominant per-16-element inner ii-loop as ONE runtime emitc.for
// with the per-strip i16 partials carried as RESIDENT SSA-register VariableOps (the single
// wide accumulator -- seeded above, load-accumulate-store inside, NEVER a materialized stack
// decode tile). This is the whole-K-nest emitter maturity lever (kernel-axis C1, ZERO perf
// green -- ~1.78x instruction density > 1.10x gate + [CASE-MICRO-E2E] decode memory-wall):
// it CLOSES the tile-roundtrip (vs*r.v 209 -> 0) + collapses the vsetvli tax. CAPABILITY-KEYED
// by the main_term_form stamp (the explicit A/B forcing override + capability policy pin),
// BYTE-EXACT to the unrolled emit by construction (identical integer vwmacc accumulation
// order -- only the ii-loop is materialized instead of unrolled). This lit checks the ROLLED
// STRUCTURE only; numeric correctness rides the SHARED q5_K board oracle (unchanged).
module {
  weft.exec.kernel @ggml_repack_gemv_q5_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_q5_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 4 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 260 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "kquant_dmin_bsums_min", main_term_form = "rolled"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied q5_K integer-core BRICK: per-block lane-wise q5_K dot ->
          // the numHalves (2) per-strip i32 sumi. The rolled schedule re-emits the whole
          // byte-exact q5_K body with the dominant per-16-element inner ii-loop rolled.
          %sumi:2 = weft_rvv.repack_gemv_kquant_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_kquant_core", decode_model = "q5_K", weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
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
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_q5_K_q8_K_kernel_ggml_repack_gemv_q5_K_q8_K(
// The two outer runtime loops -- the weight-column-group loop (nc/16) and the
// per-strip contraction-block loop over nb -- SAME as the unrolled emit.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// The per-strip i16 partial is seeded into a RESIDENT VariableOp (emitc.variable +
// assign of vmv_v_x_i16m1) BEFORE the rolled inner loop -- NOT a materialized tile.
// CHECK: emitc.variable
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// ===== The ROLLED main-term marker: a THIRD (inner) runtime for-loop over the 16-element
// group -- the compact per-element decode loop the full-unroll emit does NOT have.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// Inside it: reload the running i16 partial (load), decode the q5_K nibble+qh 5th bit
// (vle8 + the vand 0x01 qh mask), weight x activation vwmacc_vx, assign the partial back
// (the rolled resident load-accumulate-store, byte-exact to the unrolled SSA accumulation).
// CHECK: load %{{.*}} : <!emitc.opaque<"vint16m1_t">>
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x01"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: assign %{{.*}} : !emitc.opaque<"vint16m1_t">
// After the rolled loop: promote the final i16 partials to i32 (scale-weighted vwmacc_vv)
// and run the SHARED end-of-block dual d/dmin fold (unchanged).
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
