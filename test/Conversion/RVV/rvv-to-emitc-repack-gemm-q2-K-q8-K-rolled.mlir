// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// G6-B [GAP-EMIT-UNROLL]: the SAME q2_K x q8_K 16x1-REPACKED PREFILL GEMM front-door
// region as rvv-to-emitc-repack-gemm-q2-K-q8-K.mlir, but the loop-body op carries the
// OPTIONAL emit_loop_schedule = "rolled" SCHEDULE knob. Where the default (unrolled)
// emit fully STATIC-unrolls the per-16-weight-group main term (2304 static vwmacc, ~27KB
// text), the ROLLED schedule emits that dominant inner loop as ONE runtime emitc.for with
// the per-column-per-shift i16 partials staged to a stack panel -- so the main term
// materializes an INNER (4th) contraction loop instead of unrolling. This is the emitter
// maturity lever (the compact runtime-loop emission MODE): capability-keyed by code volume
// (the knob is the explicit A/B forcing override + capability policy pin), BYTE-EXACT to the
// unrolled emit by construction (identical integer vwmacc accumulation order -- proven
// on-board: independent scalar oracle INT_mismatch=0 + direct A/B output byte-identical;
// see experiments/active/g6-b-emit-unroll/). This lit checks the ROLLED STRUCTURE only.
module {
  weft.exec.kernel @ggml_repack_gemm_q2_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q2_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q2_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q2_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit-4col", qk = 256 : i64, weight_block_stride = 1344 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 320 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "kquant_dmin_bsums_min", emit_loop_schedule = "rolled"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          %sumi:4 = weft_rvv.repack_gemm_kquant_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_kquant_core", decode_model = "q2_K", weight_quant_byte_offset = 320 : i64, activation_quant_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemm_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(
// The three outer runtime loops (row-group over nr/4, column-group over nc/16, and the
// per-strip contraction-block loop over nb) -- SAME as the unrolled emit.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The i16 partials are seeded into a stack panel (vse16) BEFORE the rolled inner loop.
// CHECK: call_opaque "__riscv_vse16_v_i16m1"
// ===== The ROLLED main-term marker: a FOURTH (inner) contraction loop -- the compact
// runtime emitc.for over the 16-weight group -- that the full-unroll emit does NOT have.
// CHECK: for %[[MM:.*]] = %{{.*}} to %{{.*}} step
// Inside it: reload the running i16 partial from the panel, weight×activation vwmacc_vx,
// store the partial back (the rolled load-accumulate-store, byte-exact to the unrolled SSA
// accumulation).
// CHECK: call_opaque "__riscv_vle16_v_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vse16_v_i16m1"
