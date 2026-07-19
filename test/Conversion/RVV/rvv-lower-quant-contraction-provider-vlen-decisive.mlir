// DECISIVE EXPERIMENT (W3 pull-the-pipe, consumer = RVVLowerQuantContraction
// lowerOne, the in-compiler contraction-algorithm selection).
//
// The in-IR RVV capability provider declares a TYPED minimum_vlen = 256, while the
// lower-quant-contraction pass is driven with a CONFLICTING -march (rv64gcv =>
// guaranteed VLEN 128). The two facts DISAGREE on purpose:
//   * At VLEN 256 the fact-driven selector commits to "block-dot"; at VLEN 128 it
//     commits to "repack". So the contraction_algorithm the pass stamps is a
//     DECISIVE observable of WHICH VLEN fact the consumer read.
//   * If it follows the PROVIDER fact (minimum_vlen 256 => "block-dot") the pipe is
//     TRULY PULLED: resolveRVVMinimumVLEN read the in-IR capability object, NOT a
//     local deriveMinimumVLEN(-march) re-parse.
//   * If it followed the -march re-parse (128 => "repack") the march bypass would
//     still be alive -- the FALSE-GREEN this task forbids.
// contraction_algorithm = "block-dot" == GREEN (bypass dead, provider authoritative).
//
// The companion strip-width decisive test proves the SAME pull for the schedule-
// layer consumer; this one proves it for the front-door quant-contraction consumer.

// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv \
// RUN: | FileCheck %s --check-prefix=DECISIVE

module attributes {"weft.exec.emit_c_identifier_scope" = "kernel"} {
  weft.exec.kernel @quant_contraction_q4_0_provider_vlen256 {
    // Provider carries the TYPED minimum_vlen fact (256), independent of -march.
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", minimum_vlen = 256 : i64}
    weft.exec.variant @rvv_body attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_body, sew = 32 : i64, source_kernel = "quant_contraction_q4_0_provider_vlen256", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The abstract quant_contraction op is CONSUMED into a typed region; the stamped
// contraction_algorithm follows the PROVIDER minimum_vlen=256 ("block-dot"), NOT
// the conflicting -march=rv64gcv (which alone would give "repack" at VLEN 128).
// Pipeline truly pulled: the consumer read the in-IR capability fact, not -march.
// DECISIVE: contraction_algorithm = "block-dot"
// DECISIVE-NOT: contraction_algorithm = "repack"
