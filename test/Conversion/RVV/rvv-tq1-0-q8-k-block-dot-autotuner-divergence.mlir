// The tq1_0 fused vec_dot leaf has one real VLEN-universal body -- a SINGLE
// i16m4 accumulator (vmul_vv init / vmacc chain, NO aux8 scratch) reduced by ONE
// vwredsum. It emits the SAME core at VLEN128 and VLEN256. The former
// integer_core_lmul/minimum_vlen fields were emitter-inert, so they are removed
// instead of being retained as a fake formula axis.

// The schedule formula walk is a structural no-op for this non-tunable body at
// either capability, and must not invent final schedule fields.
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv > %t.plan128.mlir
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv_zvl256b > %t.plan256.mlir
// RUN: diff %t.plan128.mlir %t.plan256.mlir
// RUN: FileCheck %s --check-prefix=NO-FAKE --implicit-check-not=integer_core_lmul --implicit-check-not=minimum_vlen < %t.plan128.mlir
//
// Then the EMISSION-LEVEL VLEN-UNIVERSAL proof: the emit is BYTE-IDENTICAL at VLEN128
// and VLEN256.
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv --weft-rvv-lower-to-emitc > %t.vlen128.mlir
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv_zvl256b --weft-rvv-lower-to-emitc > %t.vlen256.mlir
// RUN: diff %t.vlen128.mlir %t.vlen256.mlir
// RUN: FileCheck %s --check-prefix=UNIVERSAL --implicit-check-not=vwmul_vv_i16m4 --implicit-check-not=vwmul_vv_i16m2 --implicit-check-not=vwredsum_vs_i16m2_i32m1 < %t.vlen128.mlir

module {
  weft.exec.kernel @ggml_vec_dot_tq1_0_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_tq1_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          %sumi = weft_rvv.tq1_0_q8_k_ternary_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_tq1_0_q8_k_ternary_core", scale_model = "ternary-base3-single-fp16-scale-i32-domain", qk = 256 : i64, weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 0 : i64, weight_qh_byte_offset = 48 : i64, weight_d_byte_offset = 52 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
          weft_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// NO-FAKE: weft_rvv.tq1_0_q8_k_ternary_core

// ================= VLEN-UNIVERSAL FUSED LEAF (identical both VLEN) ===========
// The deployed fused leaf: a SINGLE i16m4 accumulator (vmul_vv_i16m4 INIT +
// vmacc_vv_i16m4 chain) reduced by ONE vwredsum_vs_i16m4_i32m1 -- the SAME emit at
// VLEN128 AND VLEN256 (asserted byte-identical by the `diff` above; this UNIVERSAL
// check runs against the VLEN128 emit but holds identically for the VLEN256 emit).
// The retired per-VLEN divergent dot (vwmul_vv_i16m2 / vwmul_vv_i16m4 / the i16m2
// reduce) is GONE at BOTH VLEN -- forbidden globally by --implicit-check-not.
// UNIVERSAL: emitc.func @weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_ggml_vec_dot_tq1_0_q8_K(
// UNIVERSAL: call_opaque "__riscv_vmul_vv_i16m4"
// UNIVERSAL: call_opaque "__riscv_vmacc_vv_i16m4"
// UNIVERSAL: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// UNIVERSAL: return
