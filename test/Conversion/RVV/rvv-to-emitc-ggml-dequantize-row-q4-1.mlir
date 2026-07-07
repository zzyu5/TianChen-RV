// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `dequantize_row_q4_1` block DECODE (block_q4_1 -> f32 row) as a
// DISPATCH-WIRED lowering ([L-6] wiring != construction): tcrv_rvv.dequantize_row
// (format="q4_1") routes to a hand-written per-format monolith emitter reproducing
// ggml's reference dequantize_row_q4_1 byte-exactly -- the fp16 scale d + fp16 min
// m via the (float)*(const _Float16 *) seam, then the affine nibble unpack
// (qs[j]&0x0F)*d+m -> y[j] and (qs[j]>>4)*d+m -> y[j+16]. No typed loop brick; the
// q4_1 nibble+min decode REUSES the block-decode already built for the q4_1
// block-dot vec_dot.

module {
  tcrv.exec.kernel @dequant_q4_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @dequant_q4_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @dequant_q4_1, sew = 32 : i64, source_kernel = "dequant_q4_1_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.dequantize_row %x, %y, %k, %vl {format = "q4_1"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_dequant_q4_1_kernel_dequant_q4_1(
// The AoS block count nb = k / 32 and the block loop.
// CHECK: div
// CHECK: for
// The fp16 d and fp16 m scales (two seam reads).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The nibble decode + the affine *d then +m.
// CHECK: bitwise_and
// CHECK: bitwise_right_shift
// CHECK: mul
// CHECK: add
