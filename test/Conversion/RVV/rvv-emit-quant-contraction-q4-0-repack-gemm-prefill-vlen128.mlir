// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMF2

// OPTION-2 GEMM finale -- REAL-PATH (真路验) auto-select + construct + emit on the
// SUPPORTED RVV1.0 regime (march=rv64gcv => Zvl128b => VLEN128). The abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction op (q4_0 / PREFILL) is
// AUTO-LOWERED by --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler
// selection picks REPACK (deriveMinimumVLEN(rv64gcv)=128 => the q4_0-prefill keep)
// and -- because m_regime == "prefill" -- the C1 bridge CONSTRUCTS the typed
// weft_rvv.typed_repack_gemm_loop_body REGION (the block-as-lane GEMM, distinct from
// the decode GEVM) carrying the block_q4_0x16 x16 weight facts 288/16/32 AND the
// block_q8_0x4 INTERLEAVED activation facts 136/4/8, half_lanes=8 => mf2,
// columnsPerPass==4, NO integer_core_lmul, PLUS the 4 emitter-INERT audit attrs. The
// bridge MATERIALIZES the two GEMM ABI values (row count nr, output row stride bs)
// the abstract op does not carry. The region's two decomposed inner GEMM bricks (the
// one-strip N-column integer CORE + the four per-column dual-fp16 scale FOLDs) are
// constructed in-region. --weft-rvv-lower-to-emitc then lowers it to the byte-exact
// repack-GEMM kernel.
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the repack GEMM region from
// a REAL quant_contraction request (not test-authored) and auto-emits the GEMM
// kernel SHAPE. The region emit is byte-exact to the hand-authored
// weft_rvv.repack_gemm_q4_0_q8_0 monolith's kernel body (EMPIRICALLY proven). NO
// perf/e2e claim -- the kernel is lit-emitted, NOT run.

module {
  weft.exec.kernel @ggml_gemm_q4_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_gemm_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_gemm_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_gemm_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The abstract op is GONE; the auto-selected repack-GEMM is lowered to the SAME
// repack GEMM kernel the hand-authored repack op produces (audit attrs inert).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_q4_0_q8_0 %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: weft_rvv.q4_0_q8_0_block_dot
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_gemm_q4_0_q8_0_kernel_ggml_gemm_q4_0_q8_0(
// The selected col_outer schedule forms the weight-group base first, then the
// activation-row-group base inside it.
// CHECK: literal "288"
// CHECK: literal "136"
// The 16-lane f32m4 accumulator set (r51g m1 flip; mf2 default was 4x8 f32m2, columnsPerPass 4).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m4"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m4"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m4"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m4"
// The disjoint repacked i8mf2 sub-load, plain sign-extension decode (NO vxor), and
// lane-wise vwmacc accumulate (NO cross-lane reduction wall).
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m2"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m4"
// The dual-fp16 scale fold and the 16-lane vector store vse32_v_f32m4.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// CHECK: return

// The board-MEASURED r51g m1 whole-LMUL form (q4_0 flipped) -- the mf2 half_lanes=8
// spellings must NOT appear, and NO cross-lane vredsum reduction wall.
// NOMF2-NOT: __riscv_vfmv_v_f_f32m2
// NOMF2-NOT: __riscv_vle8_v_i8mf2
// NOMF2-NOT: redsum
