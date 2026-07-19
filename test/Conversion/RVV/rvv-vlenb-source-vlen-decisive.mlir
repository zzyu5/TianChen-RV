// DECISIVE EXPERIMENT (r5.1-g T1 — the PROBED board VLENB fact is the load-bearing
// VLEN source, NOT the -march guess).
//
// The in-IR `rvv.vlenb_bytes` capability op carries the PROBED hardware fact
// bytes=32 (a VLEN256 board). The probe layer (materializeRVVProviderCapabilityAxes
// via readRVVProviderVLenBBytes) now PREFERS that fact: minimum_vlen = VLENB*8 = 256,
// even though -march is driven with a CONFLICTING zvl128b (which alone parses to 128).
// The strip-width consumer then reads the stamped minimum_vlen=256 => half_lanes 16.
//   * half_lanes 16 == GREEN: the VLEN θ followed the PROBED vlenb fact, not -march.
//   * half_lanes 8 would mean -march still won (vlenb fact ignored) == FALSE-GREEN.
// This proves the real-board VLEN (vlenb) is the load-bearing input to the whole
// already-plumbed VLEN pipe (minimum_vlen -> strip_width), retiring the -march guess.

// RUN: weft-opt %s \
// RUN:   --weft-rvv-materialize-probed-capability-axes=march=rv64gcv_zvl128b \
// RUN:   --weft-rvv-materialize-repack-strip-width \
// RUN: | FileCheck %s --check-prefix=DECISIVE

// A DIFFERENT conflicting -march (zvl512b => 512) must ALSO yield half_lanes 16 --
// the θ is pinned to vlenb=32 (=>256), invariant to -march swinging 128<->512.
// RUN: weft-opt %s \
// RUN:   --weft-rvv-materialize-probed-capability-axes=march=rv64gcv_zvl512b \
// RUN:   --weft-rvv-materialize-repack-strip-width \
// RUN: | FileCheck %s --check-prefix=DECISIVE

module {
  weft.exec.kernel @repack_gemv_q80_vlenb32 {
    // The RVV capability provider (NO hand-authored minimum_vlen -- the probe stamps it).
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    // The PROBED real-board VLENB fact: 32 bytes/vreg => VLEN256. Its own id/kind.
    weft.exec.capability @rvv_vlenb {id = "rvv.vlenb_bytes", kind = "uarch", status = "available", bytes = "32"}
    weft.exec.variant @rvv_body attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_body, sew = 32 : i64, source_kernel = "repack_gemv_q80_vlenb32", status = "selected-lowering-boundary"} {
        %g = weft_rvv.repack_gemv_q8_0_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q8_0_q8_0", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 544 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The strip width follows the PROBED vlenb=32 (=>minimum_vlen 256 => half_lanes 16),
// NOT the conflicting -march (128 or 512). The board VLEN fact is load-bearing.
// DECISIVE: weft_rvv.repack_gemv_q8_0_q8_0
// DECISIVE-SAME: half_lanes = 16 : i64
