// The tq1_0 FUSED vec_dot leaf VLEN-UNIVERSALITY (deployed at the P1 flip): the
// COMPILER SELECTS the ggml TQ1_0 x Q8_K integer-DOT anchor (the base-3 unpack is
// fixed), and the deployed leaf is the P1-proven owned FUSED structure -- a SINGLE
// i16m4 accumulator (vmul_vv init / vmacc chain, NO aux8 scratch) reduced by ONE
// vwredsum. That leaf is VLEN-UNIVERSAL: it emits the SAME core at VLEN128 AND
// VLEN256 (byte-identical), so ONE table entry covers both boards -- C3'
// "change VLEN, don't change the entry".
//
// The unified schedule autotuner (the SAME walk-all pass that auto-discovers every
// TunableScheduleOpInterface op, NO per-tq1_0 pass) STILL stamps a per-VLEN
// integer_core_lmul m2->m1 selection onto the CONSTRUCTED
// weft_rvv.tq1_0_q8_k_ternary_core brick (the gearbox mechanism + kernel key "tq1_0"
// are unchanged). But the FUSED leaf does NOT consume that stamp: unlike the retired
// aux8 + widening-dot form (whose dot narrowed i16m4->i16m2 WITH the anchor), the
// fused single-accumulator core is fixed m2/m4 at ANY VLEN >= 128 (the fixed vl=32
// reduce bounds it). So the stamp is now VESTIGIAL for tq1_0, and the emit does NOT
// diverge by VLEN -- the honest SUPERSEDE of the old per-VLEN gearbox divergence by a
// VLEN-universal owned leaf (byte-exact on rvv AND k1).

// First, the DECISION-LEVEL fact: the unified autotuner STILL stamps DIFFERENT
// anchors onto the SAME attr-less ternary-core brick purely by the VLEN capability
// fact (the gearbox walk is unchanged) -- even though the fused leaf no longer
// consumes it.
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv | FileCheck %s --check-prefix=STAMP-VLEN128
// RUN: weft-opt %s --weft-rvv-materialize-schedule=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=STAMP-VLEN256
//
// Then the EMISSION-LEVEL VLEN-UNIVERSAL proof: the emit is BYTE-IDENTICAL at VLEN128
// and VLEN256 (the fused single-accumulator core does NOT consume the per-VLEN
// stamp). ONE entry, both boards -- NOT the retired byte-different m2/m1 dot.
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
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_tq1_0_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_tq1_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", qk = 256 : i64, weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, fold_model = "scalar_delta_grid"} {
        ^bb0(%super_block_index: index, %sumf: f32):
          %sumi = weft_rvv.tq1_0_q8_k_ternary_core %vx, %vy, %n, %vl block %super_block_index : index {kind = "ggml_tq1_0_q8_k_ternary_core", scale_model = "ternary-base3-single-fp16-scale-i32-domain", qk = 256 : i64, weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 0 : i64, weight_qh_byte_offset = 48 : i64, weight_d_byte_offset = 52 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> i32
          weft_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}

// ============= STAMPED ANCHOR (the autotuner decision, UNCHANGED) ============
// rv64gcv (VLEN128): the autotuner still SELECTS m2 (the ONLY anchor whose e8 VLMAX
// 32 spans a 32-lane strip at VLEN128) + the SEMANTIC minimum_vlen = 128 and stamps
// the CONSTRUCTED brick. The FUSED leaf does NOT consume this stamp (VLEN-universal).
// STAMP-VLEN128: weft_rvv.tq1_0_q8_k_ternary_core
// STAMP-VLEN128-SAME: integer_core_lmul = "m2"
// STAMP-VLEN128-SAME: minimum_vlen = 128 : i64
// STAMP-VLEN128-SAME: weft_rvv.tq1_0_schedule.has_zvl128b = true
// STAMP-VLEN128-SAME: weft_rvv.tq1_0_schedule.producer = "rvv-tq1-0-autotuner"
//
// rv64gcv_zvl256b (VLEN256): the SAME brick still stamps m1 (the gearbox tie-break
// is unchanged) + minimum_vlen = 256 -- also NOT consumed by the fused leaf.
// STAMP-VLEN256: weft_rvv.tq1_0_q8_k_ternary_core
// STAMP-VLEN256-SAME: integer_core_lmul = "m1"
// STAMP-VLEN256-SAME: minimum_vlen = 256 : i64

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
