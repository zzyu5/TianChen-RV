// [PERF-1] gate (2) -- the q4_K repack GEMM VLEN-flip {128,256} lit. The INPUT
// weft_rvv.quant_contraction request (K-quant q4_K / prefill) is BYTE-IDENTICAL
// across both RUN lines; the ONLY thing that varies is the selected -march (a
// TARGET-CAPABILITY fact) handed to --weft-rvv-lower-quant-contraction. The pass
// derives the guaranteed minimum VLEN from -march (deriveMinimumVLEN -- the SAME
// plugin-local authority every capability-gated pass uses) and, on the
// repack-SELECTED prefill request, feeds the resource-aware e16m1 strip width
// deriveRepackHalfLanes(minVLEN) = min(VLEN/16, 16) into the constructed
// weft_rvv.typed_repack_gemm_loop_body region:
//   * VLEN128 (rv64gcv):        half_lanes 8  -> numHalves 2 -> the strip vl is 8
//     (vsetvli semantics e32,m2 vl=8; two disjoint 8-lane strips per 16-block
//     group; int16 product buffers are 32-wide).
//   * VLEN256 (rv64gcv_zvl256b): half_lanes 16 -> numHalves 1 -> the strip vl is 16
//     (one 16-lane strip covering the whole 16-block-as-lane group; the SAME e32,m2
//     accumulator now fills 16 lanes; int16 product buffers are 64-wide).
// The SAME constructed region materializes TWO divergent emissions keyed on ONE
// capability fact -- the tiling/emit is CAPABILITY-KEYED on VLEN, NOT a hardcoded
// width. The block-as-lane repack is 16-way interleaved, so the 16-lane strip at
// VLEN256 reads BYTE-IDENTICAL repacked data to the two 8-lane halves at VLEN128
// (numerically identical, register-fill divergent). NO perf/e2e claim -- this is
// the gate (2) STRUCTURAL codegen-flip proof, lit-emitted, NOT run.

// --- VLEN128: the strip vl is 8 (two 8-lane halves; 32-wide int16 buffers). ---
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc \
// RUN:   | FileCheck %s --check-prefix=V128 --implicit-check-not='!emitc.array<64x!emitc.opaque<"int16_t">>'

// --- VLEN256: the strip vl is 16 (one 16-lane strip; 64-wide int16 buffers). ---
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b --weft-rvv-lower-to-emitc \
// RUN:   | FileCheck %s --check-prefix=V256 --implicit-check-not='!emitc.array<32x!emitc.opaque<"int16_t">>'

module {
  weft.exec.kernel @ggml_repack_gemm_q4_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q4_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q4_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 16 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ===================== VLEN128 -> strip vl 8 (half_lanes 8) =====================
// The constructed q4_K repack GEMM kernel; the abstract op is GONE.
// V128: emitc.func @weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
// The typed_repack_gemm_loop_body strip width materialized from the VLEN128
// capability fact: deriveRepackHalfLanes(128) = min(128/16, 16) = 8.
// V128: %[[VL8:[0-9]+]] = literal "8" : !emitc.opaque<"size_t">
// The SAME strip-vl SSA value drives the per-column f32m2 accumulator init and the
// int32m2 strip reset -- i.e. it is genuinely the vector length, not a stray byte.
// V128: call_opaque "__riscv_vfmv_v_f_f32m2"(%{{.+}}, %[[VL8]])
// V128: call_opaque "__riscv_vmv_v_x_i32m2"(%{{.+}}, %[[VL8]])
// The two-8-lane-strip tiling => 32-wide int16 product buffers (the 64-wide
// VLEN256 buffer is guarded absent by --implicit-check-not).
// V128: !emitc.array<32x!emitc.opaque<"int16_t">>

// ===================== VLEN256 -> strip vl 16 (half_lanes 16) ===================
// The SAME abstract request, SAME kernel, ONLY -march differs.
// V256: emitc.func @weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
// The strip width materialized from the VLEN256 capability fact:
// deriveRepackHalfLanes(256) = min(256/16, 16) = 16.
// V256: %[[VL16:[0-9]+]] = literal "16" : !emitc.opaque<"size_t">
// The SAME e32,m2 accumulator/reset now runs at vl 16 (one 16-lane strip).
// V256: call_opaque "__riscv_vfmv_v_f_f32m2"(%{{.+}}, %[[VL16]])
// V256: call_opaque "__riscv_vmv_v_x_i32m2"(%{{.+}}, %[[VL16]])
// The one-16-lane-strip tiling => 64-wide int16 product buffers (the 32-wide
// VLEN128 buffer is guarded absent by --implicit-check-not).
// V256: !emitc.array<64x!emitc.opaque<"int16_t">>
