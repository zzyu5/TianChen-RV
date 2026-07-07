// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q2_K` super-block DECODE (block_q2_K -> f32 row) as a
// DISPATCH-WIRED lowering (\[L-6\] wiring != construction): the single parameterized
// typed op tcrv_rvv.dequantize_row (format="q2_K") is recognized by production
// dispatch and routed to the hand-written per-format monolith emitter that lays down
// the byte-exact AoS super-block loop reproducing ggml's reference dequantize_row_q2_K
// -- the fp16 d/dmin scales via the (float)*(const _Float16 *) seam, then the 4-bit
// packed scale/min (sc&0xF, sc>>4) and the 2-bit quant unpack ((q>>shift)&3) folded
// dl*q - ml. The q2_K decode REUSES the block-decode facts already built for the q2_K
// block-dot vec_dot.

module {
  tcrv.exec.kernel @dequant_q2_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_q2_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q2_K, sew = 32 : i64, source_kernel = "dequant_q2_K_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "q2_K"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_q2_K_kernel_dequant_q2_K(
// CHECK: div
// CHECK: for
// The two fp16 super-block scales (d, dmin) through the shared seam.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The 4-bit scale/min unpack and the 2-bit quant shift.
// CHECK: bitwise_and
// CHECK: bitwise_right_shift
