// W3 pull-the-pipe BYTE-EXACT CHAIN (consumer C2 = repack strip width).
//
// The provider carries NO minimum_vlen. The probe-layer materializer stamps the
// typed minimum_vlen from the selected -march ONCE; the strip-width consumer then
// READS that in-IR fact (no local -march re-parse). The chain must reproduce the
// SAME half_lanes the old direct deriveMinimumVLEN(-march) produced:
//   * -march=rv64gcv         => minimum_vlen 128 => half_lanes 8
//   * -march=rv64gcv_zvl256b => minimum_vlen 256 => half_lanes 16
// This is the byte-exact witness: for NON-conflicting inputs (provider derived
// from -march) the pulled pipeline emits exactly what the march-parse did.

// RUN: weft-opt %s \
// RUN:   --weft-rvv-materialize-probed-capability-axes=march=rv64gcv \
// RUN:   --weft-rvv-materialize-repack-strip-width \
// RUN: | FileCheck %s --check-prefix=VLEN128

// RUN: weft-opt %s \
// RUN:   --weft-rvv-materialize-probed-capability-axes=march=rv64gcv_zvl256b \
// RUN:   --weft-rvv-materialize-repack-strip-width \
// RUN: | FileCheck %s --check-prefix=VLEN256

module {
  weft.exec.kernel @repack_gemv_q80_bare_provider {
    // Bare provider: identity only, NO minimum_vlen. The materializer stamps it.
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_body attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_body, sew = 32 : i64, source_kernel = "repack_gemv_q80_bare_provider", status = "selected-lowering-boundary"} {
        %g = weft_rvv.repack_gemv_q8_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q8_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 544 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// VLEN128: weft_rvv.repack_gemv_q8_0_q8_0
// VLEN128-SAME: half_lanes = 8 : i64

// VLEN256: weft_rvv.repack_gemv_q8_0_q8_0
// VLEN256-SAME: half_lanes = 16 : i64
