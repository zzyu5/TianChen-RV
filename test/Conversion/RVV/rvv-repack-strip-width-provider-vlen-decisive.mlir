// DECISIVE EXPERIMENT (W3 pull-the-pipe, consumer C2 = repack strip width).
//
// The in-IR RVV capability provider declares a TYPED minimum_vlen = 256, while
// the strip-width pass is driven with a CONFLICTING -march (rv64gcv_zvl128b =>
// guaranteed VLEN 128). The two facts DISAGREE on purpose:
//   * If the strip width follows the PROVIDER fact (minimum_vlen 256 =>
//     half_lanes 16) the pipeline is TRULY PULLED: the consumer reads the in-IR
//     capability object, NOT a local deriveMinimumVLEN(-march) re-parse.
//   * If it followed the -march re-parse (128 => half_lanes 8) the march bypass
//     would still be alive. That is the FALSE-GREEN this task forbids.
// half_lanes 16 == GREEN (bypass dead, provider authoritative).
//
// Run 1 drives the conflicting -march straight at the consumer: it proves the
// consumer ignores its own -march option in favour of the provider fact.
//
// Run 2 threads the SAME conflicting -march through the probe-layer materializer
// FIRST. Its no-clobber policy keeps the hand-authored 256, so the whole
// production chain (materialize -> consume) honours the provider fact over
// -march. Both runs must yield half_lanes 16.

// RUN: weft-opt %s \
// RUN:   --weft-rvv-materialize-repack-strip-width=march=rv64gcv_zvl128b \
// RUN: | FileCheck %s --check-prefix=DECISIVE

// RUN: weft-opt %s \
// RUN:   --weft-rvv-materialize-probed-capability-axes=march=rv64gcv_zvl128b \
// RUN:   --weft-rvv-materialize-repack-strip-width \
// RUN: | FileCheck %s --check-prefix=DECISIVE

module {
  weft.exec.kernel @repack_gemv_q80_provider_vlen256 {
    // Provider carries the TYPED minimum_vlen fact (256), independent of -march.
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 256 : i64}
    weft.exec.variant @rvv_body attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_body, sew = 32 : i64, source_kernel = "repack_gemv_q80_provider_vlen256", status = "selected-lowering-boundary"} {
        %g = weft_rvv.repack_gemv_q8_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q8_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 544 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The strip width follows the PROVIDER minimum_vlen=256 (half_lanes 16), NOT the
// conflicting -march=zvl128b (which would give 8). Pipeline truly pulled.
// DECISIVE: weft_rvv.repack_gemv_q8_0_q8_0
// DECISIVE-SAME: half_lanes = 16 : i64
