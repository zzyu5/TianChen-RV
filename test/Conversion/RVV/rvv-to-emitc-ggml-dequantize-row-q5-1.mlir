// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q5_1` block DECODE (block_q5_1 -> f32 row) as a
// DISPATCH-WIRED lowering ([L-6] wiring != construction): tcrv_rvv.dequantize_row
// (format="q5_1") routes to a hand-written per-format monolith emitter reproducing
// ggml's reference dequantize_row_q5_1 byte-exactly -- the fp16 scale d + fp16 min
// m via the (float)*(const _Float16 *) seam, the byte-assembled little-endian uint32
// qh 5th-bit plane, then ((qs[j]&0x0F)|xh0)*d+m -> y[j] and ((qs[j]>>4)|xh1)*d+m ->
// y[j+16]. No typed loop brick; the q5_1 qh+min decode REUSES the block-decode
// already built for the q5_1 block-dot vec_dot.

module {
  tcrv.exec.kernel @dequant_q5_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_q5_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q5_1, sew = 32 : i64, source_kernel = "dequant_q5_1_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "q5_1"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_q5_1_kernel_dequant_q5_1(
// The AoS block count + the block loop.
// CHECK: div
// CHECK: for
// The fp16 d and fp16 m scales (two seam reads).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The byte-assembled uint32 qh 5th-bit plane.
// CHECK: bitwise_left_shift
// CHECK: bitwise_or
// The nibble decode + the *d then +m affine.
// CHECK: bitwise_and
// CHECK: mul
// CHECK: add
