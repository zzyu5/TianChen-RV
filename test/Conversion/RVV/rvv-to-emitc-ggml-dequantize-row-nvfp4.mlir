// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_nvfp4` block DECODE (block_nvfp4 -> f32 row) as a
// DISPATCH-WIRED lowering (\[L-6\] wiring != construction): tcrv_rvv.dequantize_row
// (format="nvfp4") routes to the hand-written monolith emitter reproducing ggml's
// reference dequantize_row_nvfp4 -- the four per-16-element UE4M3 sub-block scales
// (ggml_ue4m3_to_fp32 via ldexpf, HALF form) then the 16-entry FP4 codebook gather
// scaled per sub-block. REUSES the nvfp4 block-dot vec_dot codebook + UE4M3 scale.

module {
  tcrv.exec.kernel @dequant_nvfp4_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_nvfp4 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_nvfp4, sew = 32 : i64, source_kernel = "dequant_nvfp4_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "nvfp4"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_nvfp4_kernel_dequant_nvfp4(
// The FP4 codebook table decl (kvalues_mxfp4, reused by nvfp4), above the loop.
// CHECK: static const int8_t tcrv_dequant_nvfp4_kvalues
// The block count nb = k / 64 and the block loop.
// CHECK: div
// CHECK: for
// The UE4M3 -> fp32 HALF scale via ldexpf (NO fp16 seam).
// CHECK: call_opaque "ldexpf"
// CHECK: bitwise_and
