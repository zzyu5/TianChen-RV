// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// tq2_0 (the 2-bit TERNARY TriLM K-quant -- one of the LAST TWO uncommon ggml dot
// kernels) -- the COMPLETE ggml ggml_vec_dot_tq2_0_q8_K as STRUCTURED emitc IR
// (I5; ZERO raw() strings). The single typed op tcrv_rvv.tq2_0_q8_k_block_dot
// REUSES q2_K's 2-bit weight unpack VERBATIM but folds a per-element `-1` ternary
// bias into the unpack (`((qs >> (l*2)) & 3) - 1` over the shifts {0,2,4,6}) and is
// genuinely SIMPLER than every K-quant sibling: NO scales[16] nibble extraction,
// NO per-sub-block scale, NO min term, NO dmin, NO bsums -- a SINGLE per-super-
// block integer accumulator + a single-fp16-scale SCALAR fp32 fold `sumf +=
// (float)sumi * d`, d = y.d * fp16(x.d) where the fp16 scale is at the END of
// block_tq2_0 (qs[64] @0, d @64, stride 66). The fp32 ordering is byte-exact vs
// _generic; pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc42-tq2_0/.

module {
  tcrv.exec.kernel @ggml_vec_dot_tq2_0_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_tq2_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_tq2_0_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_tq2_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.tq2_0_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_tq2_0_q8_k_block_dot", scale_model = "ternary-single-fp16-scale-i32-domain-scalar-fp32-fold", qk = 256 : i64, weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 0 : i64, weight_d_byte_offset = 64 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_tq2_0_q8_K_kernel_ggml_vec_dot_tq2_0_q8_K(
// The super-block count nb = n / 256, the int8_t aux8[256] scratch.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// The scalar sumf, zeroed ONCE outside the super-block loop.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// The outer super-block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The 2-bit TERNARY unpack: u8m2 load of the qs chunk, vsrl/vand over the shifts
// {0,2,4,6}, u8->i8 reinterpret, then the per-element `-1` ternary bias (vadd.vx
// of -1 in the i8 domain) BEFORE the vse8 -- this is the load-bearing ternary
// decode `((qs>>shift)&3) - 1`.
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vand_vx_u8m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CHECK: call_opaque "__riscv_vadd_vx_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// tq2_0 has NO scales / min / bsums machinery -- it must NOT misroute to a q2_K
// or q4_K-style scale/min decode. NO 4-bit nibble scale extraction (bitwise_and /
// bitwise_right_shift on a scales byte), NO 6-bit utmp/kmask bit-dance, NO bsums.
// CHECK-NOT: bitwise_and
// CHECK-NOT: bitwise_right_shift
// CHECK-NOT: bitwise_or
// CHECK-NOT: const int16_t
// The SINGLE per-super-block integer accumulator sumi, fed by the per-16-lane
// widening dot (i8m1 loads -> vwmul -> vwredsum -> vmv_x_s), summed (NO per-sub-
// block scale multiply).
// CHECK: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int">>
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The SINGLE-SCALE SCALAR fp32 fold: dy (fp32 q8_K scale) loaded once, dx via the
// fp16 read seam (ONE read, at xb+64), d = dy*dx, then `sumf += (float)sumi * d`
// as ONE emitc.expression (a cast + a mul + an add). There is exactly ONE fp16
// read (no dall/dmin pair), NO subtract (no min term), NO 8-lane vfcvt/vfmul.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: expression
// CHECK: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK-NOT: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK-NOT: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK-NOT: call_opaque "__riscv_vfmacc
// CHECK-NOT: call_opaque "__riscv_vse32_v_f32m2"
// The *s store, then return.
// CHECK: return
