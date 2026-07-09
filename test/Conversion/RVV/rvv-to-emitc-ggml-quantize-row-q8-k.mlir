// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The COMPLETE ggml `quantize_row_q8_K` RVV-path forward-pass op (the f32 ->
// block_q8_K K-quant ACTIVATION QUANTIZER; riscv/quants.c) as STRUCTURED emitc
// IR (I5; ZERO raw() strings). The heaviest quantizer: the single typed op
// tcrv_rvv.quantize_row_q8_K lowers to a QK_K=256 super-block loop (nb = n/256)
// whose body (1) folds a min AND max over an e32m8 strip loop
// (vfmax_vv/vfmin_vv seeded -/+inf) then vfredmax/vfredmin to scalars, (2)
// computes amax via fabsf and the symmetric iscale = -127/(|max|>|min|?max:min),
// (3) on the load-bearing amax==0 case takes a STRUCTURED emitc.if/else zero path
// (float d=0, memset qs+bsums to 0 -- ggml's `continue`), else stores the FLOAT
// d = 1/iscale and runs the quantize strip loop: vfmul by iscale,
// vfcvt_x_f_v_i32m8_rm(RNE) + two vnclip_wx(RNE) f32->i32->i16->i8 narrow, the
// 256 int8 qs store, and the 16 per-16-element bsums (vslidedown-advanced
// vwredsum chunks).
//
// BYTE-EXACTNESS matches ggml's EXACT RVV method (the vfcvt/vnclip explicit RNE
// modes, the fabsf-symmetric iscale, the float d store, the vwredsum per-16
// bsums, the zero-block memset). Every emitted value is a NODE in the IR graph.
// DISPATCH-WIRED ([L-6] wiring != construction).

module {
  tcrv.exec.kernel @quantize_row_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @quantize_row_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "uint8_t *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @quantize_row_q8_K, sew = 32 : i64, source_kernel = "quantize_row_q8_K_kernel", status = "selected-lowering-boundary"} {
        %q = tcrv_rvv.quantize_row_q8_K %x, %vy, %n, %vl {kind = "ggml_quantize_row_q8_K", qk = 256 : i64, block_stride = 292 : i64, scale_byte_offset = 0 : i64, quant_byte_offset = 4 : i64, bsums_byte_offset = 260 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_quantize_row_q8_K_kernel_quantize_row_q8_K(
// FRONT-DOOR CONSTRUCTED: the abstract tcrv_rvv.quantize_row_q8_K went THROUGH the
// typed tcrv_rvv.typed_quantize_row_loop_body region (the provenance token proves it).
// CHECK: route_source_op=tcrv_rvv.typed_quantize_row_loop_body
// The super-block count nb = n / 256, the shared e32m8 vlmax, and the loop.
// CHECK: div {{.*}}, %{{.*}}
// CHECK: call_opaque "__riscv_vsetvlmax_e32m8"
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The min/max reduction: -/+inf seeds, the strip fold, then the scalar reduce.
// CHECK: literal "-__builtin_inff()"
// CHECK: literal "__builtin_inff()"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m8"
// CHECK: call_opaque "__riscv_vfmax_vv_f32m8"
// CHECK: call_opaque "__riscv_vfmin_vv_f32m8"
// CHECK: call_opaque "__riscv_vfmv_s_f_f32m1"
// CHECK: call_opaque "__riscv_vfredmax_vs_f32m8_f32m1"
// CHECK: call_opaque "__riscv_vfredmin_vs_f32m8_f32m1"
// The amax via fabsf and the symmetric selector.
// CHECK: call_opaque "fabsf"
// CHECK: cmp gt, %{{.*}}, %{{.*}}
// The load-bearing amax == 0 zero-block guard (STRUCTURED if/else, ggml's
// `continue`): float d = 0 + memset qs + memset bsums.
// CHECK: cmp eq, %{{.*}}, %{{.*}}
// CHECK: if %{{.*}} {
// CHECK: call_opaque "memset"
// CHECK: call_opaque "memset"
// CHECK: } else {
// The symmetric iscale = -127.f / den, the FLOAT d = 1/iscale store, the RNE
// rounding-mode enums.
// CHECK: literal "-127.f"
// CHECK: cast %{{.*}} : !emitc.ptr<!emitc.opaque<"uint8_t">> to !emitc.ptr<!emitc.opaque<"float">>
// CHECK: literal "__RISCV_FRM_RNE"
// CHECK: literal "__RISCV_VXRM_RNE"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The quantize strip loop: scale by iscale, f32->i32->i16->i8 RNE narrow, qs store.
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vfcvt_x_f_v_i32m8_rm"
// CHECK: call_opaque "__riscv_vnclip_wx_i16m4"
// CHECK: call_opaque "__riscv_vnclip_wx_i8m2"
// CHECK: call_opaque "__riscv_vse8_v_i8m2"
// The per-16 bsums: the first chunk, then the vslidedown-advanced inner loop.
// CHECK: call_opaque "__riscv_vget_v_i8m2_i8m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i8m1_i16m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i16m1_i16"
// CHECK: call_opaque "__riscv_vslidedown_vx_i8m2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i8m1_i16m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i16m1_i16"
// The whole kernel is structured emitc nodes; no raw C blob after the last
// intrinsic (the provenance verbatims are comment lines only).
// CHECK-NOT: emitc.verbatim {{.*}}__riscv
