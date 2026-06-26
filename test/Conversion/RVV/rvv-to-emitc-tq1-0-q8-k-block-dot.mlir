// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// tq1_0 (the 1.6875-bpw BASE-3-PACKED TERNARY TriLM K-quant -- the LAST of the 24
// ggml dot kernels, literal 100% coverage) -- the COMPLETE ggml
// ggml_vec_dot_tq1_0_q8_K as STRUCTURED emitc IR (I5; ZERO raw() strings). The
// single typed op tcrv_rvv.tq1_0_q8_k_block_dot recovers each ternary trit by a
// base-3 power-of-three multiply + the mandatory uint8 wrap (`q = (uint8_t)(byte *
// pow3[l]); xi = ((uint16_t)q * 3) >> 8; xi - 1` in {-1,0,1}, pow3 = {1,3,9,27,81})
// of the qs[48] (5 trits/byte) + qh[4] (4 trits/byte) weight arrays into an
// element-ordered aux8[256], then a single per-super-block integer accumulator + a
// single-fp16-scale SCALAR fp32 fold `sumf += (float)sum * (fp16(x.d) * y.d)` where
// the fp16 scale is at the END of block_tq1_0 (qs[48] @0, qh[4] @48, d @52, stride
// 54). The fp32 ordering is byte-exact vs _generic; pinned by the ssh-rvv artifact
// under .trellis/tasks/.../artifacts/inc43-tq1_0/.

module {
  tcrv.exec.kernel @ggml_vec_dot_tq1_0_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_tq1_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq1-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_tq1_0_q8_K, sew = 32 : i64, source_kernel = "ggml_vec_dot_tq1_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.tq1_0_q8_k_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_tq1_0_q8_k_block_dot", scale_model = "ternary-base3-single-fp16-scale-i32-domain-scalar-fp32-fold", qk = 256 : i64, weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, weight_qs_byte_offset = 0 : i64, weight_qh_byte_offset = 48 : i64, weight_d_byte_offset = 52 : i64, activation_d_byte_offset = 0 : i64, activation_quant_byte_offset = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_ggml_vec_dot_tq1_0_q8_K(
// The super-block count nb = n / 256, the int8_t aux8[256] scratch.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[AUX8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<256x!emitc.opaque<"int8_t">>
// The scalar sumf, zeroed ONCE outside the super-block loop.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// The outer super-block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The BASE-3 trit unpack (the load-bearing decode `q=(uint8_t)(byte*pow3[l]); xi=
// ((uint16_t)q*3)>>8; xi-1`): the u8m2 load of the qs chunk, then the uint8 wrap
// vmul.vx by pow3[l] (NOT widened -- the mod-256 truncation IS the decode), the
// widening vwmulu by 3, the vsrl by 8, the u16->u8 narrow, the u8->i8 reinterpret,
// then the per-element `-1` ternary bias (vadd.vx of -1) BEFORE the vse8.
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_u8m2"
// CHECK: call_opaque "__riscv_vmul_vx_u8m2"
// CHECK: call_opaque "__riscv_vwmulu_vx_u16m4"
// CHECK: call_opaque "__riscv_vsrl_vx_u16m4"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8m2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8m2_i8m2"
// CHECK: call_opaque "__riscv_vadd_vx_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// The qs TAIL + qh regions decode at e8m1 (16-lane tail, 4-lane qh).
// CHECK: call_opaque "__riscv_vmul_vx_u8m1"
// CHECK: call_opaque "__riscv_vwmulu_vx_u16m2"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8m1"
// tq1_0 is BASE-3, NOT a 2-bit field shift (tq2_0) and NOT a nibble/min K-quant. It
// must NOT misroute: NO `& 3` 2-bit field mask (the tq2_0 marker), NO 4-bit nibble
// scale extraction (bitwise_right_shift), NO bsums / min subtract.
// CHECK-NOT: call_opaque "__riscv_vand_vx_u8m2"
// CHECK-NOT: bitwise_right_shift
// CHECK-NOT: const int16_t
// The SINGLE per-super-block integer accumulator sumi, fed by the WIDE-strip
// flat-256 dot over aux8 x q8 (default anchor m2 at the VLEN-universal floor; the
// gearbox refines m2->m1 at VLEN256). The 16x narrow e8m1(16) reductions are
// replaced by 32-lane strips: vle8 i8m2 loads -> vwmul_vv_i16m4 -> vwredsum_i16m4
// -> vmv_x_s, summed (NO per-sub-block scale multiply -- tq1_0 has no scales).
// CHECK: %[[SUMI:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int">>
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m4"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The OLD narrow dot path (per-16-lane vwmul_i16m2 reduce) is gone.
// CHECK-NOT: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK-NOT: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// The SINGLE-SCALE SCALAR fp32 fold: dx via the fp16 read seam (ONE read, at xb+52,
// the END of block_tq1_0), dy (fp32 q8_K scale) loaded once, d = dx*dy, then `sumf
// += (float)sumi * d` as ONE emitc.expression (a cast + a mul + an add). There is
// exactly ONE fp16 read (no dall/dmin pair), NO subtract (no min term).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: expression
// CHECK: cast %{{.*}} : !emitc.opaque<"int"> to !emitc.opaque<"float">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK-NOT: sub %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// The *s store, then return.
// CHECK: return
