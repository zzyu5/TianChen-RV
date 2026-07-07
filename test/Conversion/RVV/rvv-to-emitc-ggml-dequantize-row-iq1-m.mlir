// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_iq1_m` block DECODE (block_iq1_m -> f32 row) as a
// DISPATCH-WIRED lowering (\[L-6\] wiring != construction): tcrv_rvv.dequantize_row
// (format="iq1_m") routes to the hand-written monolith emitter reproducing ggml's
// reference dequantize_row_iq1_m -- 2048-entry ternary grid; NO d field -- the super-block scale is the packed iq1m_scale fp16 reconstructed from the 4 scale words then read AS _Float16 (bit reinterpret); delta=+-0.125. REUSES the iq1_m block-dot vec_dot grid
// decl (byte-identical canonical grid/signs table). Numerically byte-exact to ggml
// (verified vs ggml's reference on random blocks; scalar AoS loop, no reduction).

module {
  tcrv.exec.kernel @dequant_iq1_m_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_iq1_m attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_iq1_m, sew = 32 : i64, source_kernel = "dequant_iq1_m_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "iq1_m"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_iq1_m_kernel_dequant_iq1_m(
// The grid table decl, emitted once above the super-block loop (reused from the vec_dot grid decl).
// CHECK: tcrv_iq1m_grid
// The super-block loop, then the fp16 block-scale seam inside it.
// CHECK: for %
// CHECK: call_opaque "(float)*(const _Float16 *)"
