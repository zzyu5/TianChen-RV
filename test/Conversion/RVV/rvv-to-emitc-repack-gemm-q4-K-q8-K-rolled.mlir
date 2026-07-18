// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G8 [ROLL] whole-K-nest keying: the SAME q4_K x q8_K 16x1-REPACKED PREFILL GEMM (M>>1)
// S6-TILED front-door region as rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir, but the loop-body
// op carries the OPTIONAL emit_loop_schedule = "rolled" SCHEDULE knob (the A/B override;
// the AUTO producer default already rolls this super-block K-quant main term). Where the
// unrolled emit fully STATIC-unrolls the per-16-element inner ii-loop (the 2x16 i16-chunk
// dot, the vsetvli-storm source under gcc-15.2), the ROLLED schedule emits that dominant
// inner loop as ONE runtime emitc.for with the per-column i16 partials sLoVar/sHiVar
// carried as RESIDENT SSA-register VariableOps (seeded above, load-accumulate-store
// inside; the runtime position only shifts the byte offsets -- the 16-way-interleaved
// weight nibble by ii*16, the 4-column-interleaved q8_Kx4 activation by ii*4). The 4-bit
// nibble decode stays INSIDE the ii-loop but OUTSIDE the column loop, so each nibble is
// decoded ONCE and shared across all 4 activation columns (NO re-decode, NO tile
// narrowing). The [ROLL] axis is ORTHOGONAL to the colGroupOuter loop-order axis (both
// coexist) and to the S6 stack panels (scale/min/bsums staging unchanged). CAPABILITY-
// KEYED by the stamp, BYTE-EXACT to the unrolled emit by construction (dual d/dmin +
// bsums-min fold, identical vwmacc16 accumulation order; only the ii-loop is materialized).
// Checks the ROLLED STRUCTURE only; numeric correctness rides the SHARED q4_K board oracle
// (independent scalar q4_K dequant-matmul reference, bounded-norm PASS, unchanged).
module {
  weft.exec.kernel @ggml_repack_gemm_q4_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q4_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q4_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col", qk = 256 : i64, weight_block_stride = 2304 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 256 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "kquant_dmin_bsums_min", emit_loop_schedule = "rolled"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          %sumi:4 = weft_rvv.repack_gemm_kquant_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_kquant_core", decode_model = "q4_K", weight_quant_byte_offset = 256 : i64, activation_quant_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
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
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
// The col-group-outer nest (X over nc/16 outer, Y over nr/4 middle) + the per-strip
// contraction-block loop over nb -- SAME envelope as the unrolled emit (the [ROLL] axis
// is orthogonal to the loop-order axis).
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-sub-block 6-bit scale/min unpack (SHARED across columns, OUTSIDE the rolled
// main loop): vle8 + the vand/vsrl/vsll bit-dance + vzext to i16.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// The MIN term (OUTSIDE the rolled main loop): per-column int16 bsums read + vwmacc_vx.
// CHECK: call_opaque "*(const int16_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// The per-column i16 partials sLoVar/sHiVar seeded into RESIDENT VariableOps
// (emitc.variable + assign of vmv_v_x_i16m1) BEFORE the rolled inner loop -- NOT a
// materialized stack panel (SSA-register residence).
// CHECK: emitc.variable
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// ===== The ROLLED main-term marker: a runtime for-loop over the 16-element k-chunk --
// the compact per-element decode loop the full-unroll emit lacks.
// CHECK: for %[[II:.*]] = %{{.*}} to %{{.*}} step
// Inside it: the SHARED 4-bit nibble decode (vle8 weight strip, vand 0x0F / vsrl 4,
// reinterpret to i8), then per-column reload of the running i16 partial (load), the
// per-column q8_Kx4 quant read + weight x activation vwmacc_vx, assign the partial back
// (the rolled resident load-accumulate-store, byte-exact to the unrolled SSA accumulation;
// the nibble decode is OUTSIDE the column reads so it is shared across the 4 columns).
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: load %{{.*}} : <!emitc.opaque<"vint16m1_t">>
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: assign %{{.*}} : !emitc.opaque<"vint16m1_t">
// After the rolled loop: promote the final i16 partials to i32 (scale-weighted
// vwmacc_vv), then the SHARED end-of-block dual d/dmin fold (vfmul + vfcvt + vfmacc main
// + vfnmsac MIN), then the per-column per-strip vse32 store (unchanged).
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: the dot
// accumulates LANE-WISE via vwmacc, so NO vredsum / vwredsum appears (rolled or not).
// NOWALL-NOT: redsum
