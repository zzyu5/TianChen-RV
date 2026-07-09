// [PERF-1] gate (2) companion -- the q4_K repack GEVM (decode) VLEN-keyed lit. It
// records the HONEST decode-regime asymmetry that the prefill GEMM
// (rvv-q4-k-repack-gemm-march-vlen-flip-gate2.mlir) does not show: the SAME
// abstract tcrv_rvv.quant_contraction request (K-quant q4_K / DECODE) is routed by
// a VLEN-keyed selection (selectContractionAlgorithm via
// vlenOrPrefillFavorsRepack = (minVLEN==128 || Prefill)):
//   * VLEN128 (rv64gcv):        the decode regime SELECTS repack -> the constructed
//     tcrv_rvv.typed_repack_gemv_loop_body region emits the capability-derived
//     strip width deriveRepackHalfLanes(128) = 8 (vl 8; e32,m2 accumulator; two
//     8-lane halves). This is the q4_K GEVM repack path, capability-keyed.
//   * VLEN256 (rv64gcv_zvl256b): the decode regime DECLINES repack (the documented
//     decode-VLEN256 loss cell -- a memoized selection fact, NOT a hardcode). q4_K
//     has NO block-dot decline path (the block-dot identity lowering is
//     q4_0-nibble-only), so the request fails CLOSED (I7). There is therefore NO
//     vl=16 GEVM repack to check -- the VLEN-flip here is in the ROUTING, not a
//     wider strip. This is why the {128,256} vl=8/vl=16 flip is demonstrated on the
//     PREFILL GEMM (both regimes select repack), while the GEVM proves the SAME
//     selector genuinely reasons on the VLEN fact (declining at 256).
// NO perf/e2e claim -- lit-emitted / diagnostic-checked, NOT run.

// --- VLEN128 decode: the strip vl is 8 (capability-derived repack GEVM). ---
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc \
// RUN:   | FileCheck %s --check-prefix=GEVM128

// --- VLEN256 decode: the selector is VLEN-keyed to DECLINE repack; q4_K has no
// --- block-dot decline path, so the request fails closed (I7). ---
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv_zvl256b --verify-diagnostics

module {
  tcrv.exec.kernel @ggml_repack_gemv_q4_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q4_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q4_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        // expected-error@+1 {{requires a repack-affording capability}}
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 16 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// ================ VLEN128 decode -> repack GEVM, strip vl 8 ==================
// The decode regime SELECTS repack at VLEN128; the abstract op is GONE and the
// q4_K repack GEVM kernel is constructed.
// GEVM128: emitc.func @tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
// The typed_repack_gemv_loop_body strip width materialized from the VLEN128
// capability fact: deriveRepackHalfLanes(128) = 8.
// GEVM128: %[[VL8:[0-9]+]] = literal "8" : !emitc.opaque<"size_t">
// The SAME strip-vl SSA value drives the f32m2 accumulator init and the fp16 scale
// strip load -- i.e. it is genuinely the vector length.
// GEVM128: call_opaque "__riscv_vfmv_v_f_f32m2"(%{{.+}}, %[[VL8]])
// GEVM128: call_opaque "__riscv_vle16_v_f16m1"(%{{.+}}, %[[VL8]])
