// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml Q8_0 x Q8_0 block dot-product (tcrv_rvv.q8_0_q8_0_block_dot, the
// Family-A sibling of the Q4_0 route) lowers to the COMPLETE structured kernel
// (I5; zero raw(), every value an emitc node): the outer block loop over
// nb = n / 32, the per-block address arithmetic (both operands stride 34) + dual
// fp16 scalar scale reads, the plain i8 x i8 widening product/reduce integer core
// (NO nibble decode), and the q8_0 fp32 fold `sumf + (float)sumi * (d_x * d_y)`.
// Here the op carries the m2 anchor explicitly (the ggml-matching one-vwredsum-
// per-block reduction anchor); the inner strip loop covers the whole 32-element
// block in one strip at VLEN=128 (vsetvl_e8m2(32)) and re-strips on a smaller VLEN.

module {
  tcrv.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q8_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q8_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q8_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q8_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q8_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, integer_core_lmul = "m2"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0(
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The outer block loop.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// Per-block address arithmetic: vx + ib*34, vy + ib*34 (both block_q8_0).
// CHECK: mul %{{.*}}, %{{.*}}
// The two scalar fp16->fp32 reads.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The plain i8m2 x i8m2 widening product into i16m4 (NO vxor/vsll/vsra decode).
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m4"
// CHECK-NOT: call_opaque "__riscv_vxor_vx_i8
// CHECK-NOT: call_opaque "__riscv_vwmacc_vv_i16
// The carried-seed vwredsum into i32m1 + the scalar extract.
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The q8_0 fold: d_x * d_y FIRST, then (float)sumi * that, then sumf + that.
// CHECK: expression : !emitc.opaque<"float">
// CHECK: cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store.
// CHECK: subscript
// CHECK: assign
