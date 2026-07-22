// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMIN
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=RETIRED

// G8 [ROLL] whole-K-nest keying: the SAME q3_K x q8_K 16x1-REPACKED PREFILL GEMM (M>>1)
// front-door region as rvv-to-emitc-repack-gemm-q3-K-q8-K.mlir, but the loop-body op
// carries the OPTIONAL main_term_form = "rolled" SCHEDULE knob (the A/B override; the
// AUTO producer default already rolls this super-block K-quant main term). Where the
// unrolled emit fully STATIC-unrolls the per-16-position inner l-loop (the vsetvli-storm
// source under gcc-15.2), the ROLLED schedule emits that dominant inner loop as ONE
// runtime emitc.for with the 4-column x per-strip x per-quadrant i16 partials carried as
// RESIDENT SSA-register VariableOps (seeded above, load-accumulate-store inside; the
// runtime position only shifts the base pointers -- weight strips [qs AND hmask] by l*16,
// the 4-column-interleaved q8_Kx4 activation by l*4). The 3-bit subtractive-hmask weight
// decode stays INSIDE the l-loop but OUTSIDE the column loop, so each weight is decoded
// ONCE and shared across all 4 activation columns (NO re-decode, NO tile narrowing -- the
// whole-K-nest, distinct from a narrow output tile). q3_K is now on the SAME code-volume
// vs I-cache [ROLL] resolver as its no-min sibling q6_K (the "rolled is q6_K-only"
// format-name hardcode is REMOVED). CAPABILITY-KEYED by the stamp, BYTE-EXACT to the
// unrolled emit by construction (single-accumulator NO-min fold, identical vwmacc16
// accumulation order; only the l-loop is materialized). Checks the ROLLED STRUCTURE only;
// numeric correctness rides the SHARED q3_K board oracle (independent scalar q3_K
// dequant-matmul reference, unchanged).
module {
  weft.exec.kernel @ggml_repack_gemm_q3_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q3_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q3_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q3_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "kquant_single_scale_no_min", main_term_form = "rolled", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          %sumi:4 = weft_rvv.repack_gemm_kquant_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_kquant_core", decode_model = "q3_K", weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
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
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K(
// The three outer runtime loops (row-group over nr/4, column-group over nc/16, and the
// per-strip contraction-block loop over nb) -- SAME as the unrolled emit.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The q3_K SIGNED int8 per-sub-block scale (SHARED across columns, OUTSIDE the rolled
// loop): vle8_v_i8 + vsext.
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf2_i16m1"
// The 4-column x per-strip x per-quadrant i16 partials are seeded into RESIDENT
// VariableOps (emitc.variable + assign of vmv_v_x_i16m1) BEFORE the rolled inner loop --
// NOT a materialized stack panel (SSA-register residence).
// CHECK: emitc.variable
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// ===== The ROLLED main-term marker: a FOURTH (inner) runtime for-loop over the 16
// sub-block positions -- the compact per-element decode loop the full-unroll emit lacks.
// CHECK: for %[[LL:.*]] = %{{.*}} to %{{.*}} step
// Inside it: the SHARED 3-bit subtractive-hmask weight decode (qs/hmask strip loads, the
// 2-bit low plane vand 0x03 reinterpreted to signed i8, then the SINGLE hmask high bit
// tested IN PLACE by vand(1<<p) + vmseq==0 and the -4 SUBTRACTIVE bias FUSED into ONE
// masked op vadd_vx_i8mf2_mu via the SHARED [QH-MASK] helper -- byte-exact to the retired
// vsll 2 | vor | vsub 4 chain), then per-column reload of the running i16 partial (load),
// weight x activation vwmacc_vx, assign the partial back (the rolled resident load-
// accumulate-store, byte-exact to the unrolled SSA accumulation; the weight decode is
// OUTSIDE the column reads so it is shared across the 4 columns -- NO re-decode).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vmseq_vx_u8mf2_b16"
// CHECK: literal "-4"
// CHECK: call_opaque "__riscv_vadd_vx_i8mf2_mu"
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: load %{{.*}} : <!emitc.opaque<"vint16m1_t">>
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: assign %{{.*}} : !emitc.opaque<"vint16m1_t">
// After the rolled loop: promote the final i16 partials to i32 (scale-weighted
// vwmacc_vv) and run the SHARED end-of-block single no-min fold (vfcvt + vfmacc), then the
// per-column per-strip vse32 store (unchanged).
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// q3_K is a SINGLE-accumulator no-min fold: NO min-term subtract (vfnmsac) and NO
// activation bsums read (int16_t) appear even under the rolled schedule.
// NOMIN-NOT: vfnmsac
// NOMIN-NOT: *(const int16_t *)

// The OLD per-lane hmask expand chain (vsll<<2 | vor | vsub 4) is fully RETIRED by the
// SHARED [QH-MASK] native-mask helper -- none survive under the rolled schedule either.
// RETIRED-NOT: __riscv_vsll_vx_u8mf2
// RETIRED-NOT: __riscv_vor_vv_u8mf2
// RETIRED-NOT: __riscv_vsub_vx_i8mf2
